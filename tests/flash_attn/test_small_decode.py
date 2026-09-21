# SPDX-License-Identifier: Apache-2.0
"""Paged decode reference checks for the tuned shapes and existing fallbacks."""

import pytest
import torch

from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("length", [1024, 32769])
@pytest.mark.parametrize("qh,kvh,dim,page,window", [
    (8, 1, 512, 64, -1),
    (8, 4, 256, 32, 1023),
    (16, 2, 512, 64, -1),
    (16, 8, 256, 32, 1023),
])
def test_small_paged_decode(batch, length, qh, kvh, dim, page, window):
    torch.manual_seed(839)
    pages = (length + page - 1) // page
    blocks = batch * pages + 3
    ids = torch.randperm(blocks)[:batch * pages].reshape(batch, pages).int()
    kcpu = torch.randn(batch, pages * page, kvh, dim).half() * .2
    vcpu = torch.randn_like(kcpu) * .2
    qcpu = torch.randn(batch, qh, dim).half() * .1
    start = max(0, length - 1 - window) if window >= 0 else 0
    expected = []
    for b in range(batch):
        qref = qcpu[b].float().reshape(kvh, qh // kvh, dim)
        kref = kcpu[b, start:length].float().permute(1, 0, 2)
        vref = vcpu[b, start:length].float().permute(1, 0, 2)
        scores = torch.bmm(qref, kref.transpose(1, 2))
        expected.append(torch.bmm(scores.softmax(-1), vref).reshape(qh, dim))
    expected = torch.stack(expected)
    # Interleaved K/V heads and unused pages exercise the stride contract.
    cache = torch.empty(blocks * 2, page, kvh, 2, dim,
                        dtype=torch.float16, device="xpu")
    key, value = cache[::2, :, :, 0, :], cache[::2, :, :, 1, :]
    device_ids = ids.flatten().long().to("xpu")
    key.index_copy_(0, device_ids,
                    kcpu.reshape(-1, page, kvh, dim).to("xpu"))
    value.index_copy_(0, device_ids,
                      vcpu.reshape(-1, page, kvh, dim).to("xpu"))
    kwargs = dict(
        q=qcpu.to("xpu"), k=key, v=value,
        cu_seqlens_q=torch.arange(batch + 1, dtype=torch.int32, device="xpu"),
        seqused_k=torch.full((batch,), length, dtype=torch.int32,
                            device="xpu"),
        block_table=ids.to("xpu"), max_seqlen_q=1, max_seqlen_k=length,
        softmax_scale=1., causal=True,
        window_size=(window, 0) if window >= 0 else (-1, -1))
    for splits in (None, 1, 8, 16):
        y = flash_attn_varlen_func(**kwargs, num_splits_kv=splits)
        torch.testing.assert_close(y.cpu().float(), expected,
                                   rtol=1e-2, atol=1e-3)
