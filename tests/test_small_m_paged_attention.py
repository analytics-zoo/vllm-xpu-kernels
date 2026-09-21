# SPDX-License-Identifier: Apache-2.0
from contextlib import nullcontext

import pytest
import torch

from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func


@pytest.mark.parametrize(
    "heads,dim", [(8, 256), (16, 256), (8, 512), (16, 512)]
)
@pytest.mark.parametrize(
    "m,length,splits,lse,single_queries,causal",
    [
        (2, 16385, None, False, False, True),
        (3, 16385, None, False, False, True),
        (4, 32768, None, False, False, True),
        (6, 32773, None, False, False, True),
        (7, 40960, None, False, False, True),
        (5, 32773, None, False, False, True),
        (8, 32768, 16, False, False, True),
        (5, 3, None, False, False, True),
        (5, 0, None, False, False, True),
        (5, 16385, None, True, False, True),
        (1, 32773, None, False, True, True),
        (5, 32773, None, False, True, True),
        (5, 32773, None, False, False, False),
    ],
)
@torch.inference_mode()
def test_small_m_causal_paged_attention(
    heads, dim, m, length, splits, lse, single_queries, causal
):
    page = 32 if dim == 256 else 64
    kv = heads // (2 if dim == 256 else 8)
    pages = max(1, (length + page - 1) // page)
    generator = torch.Generator().manual_seed(length + m + heads)
    ids = torch.randperm(pages + 3, generator=generator)[:pages]
    q = (torch.randn(m, heads, dim, generator=generator) * 0.1).half()
    k = (torch.randn(pages * page, kv, dim, generator=generator) * 0.2).half()
    v = (torch.randn(pages * page, kv, dim, generator=generator) * 0.2).half()
    expected = []
    for i in range(m):
        end = max(0, length - m + i + 1) if causal or dim == 256 else length
        start = max(0, end - 1024) if dim == 256 else 0
        if end == 0:
            expected.append(torch.zeros(heads, dim))
        else:
            kr = k[start:end].float().permute(1, 0, 2)
            vr = v[start:end].float().permute(1, 0, 2)
            scores = q[i].float().reshape(kv, heads // kv, dim) @ kr.transpose(
                1, 2
            )
            expected.append((scores.softmax(-1) @ vr).reshape(heads, dim))
    stream = torch.xpu.Stream()
    with torch.xpu.stream(stream):
        # Match the server interleaved KV layout, including the V offset.
        cache = torch.empty(
            pages + 3, page, kv, 2, dim, device="xpu", dtype=torch.float16
        )
        dk, dv = cache[:, :, :, 0], cache[:, :, :, 1]
        device_ids = ids.xpu()
        dk.index_copy_(0, device_ids, k.reshape(pages, page, kv, dim).xpu())
        dv.index_copy_(0, device_ids, v.reshape(pages, page, kv, dim).xpu())
        dq = q.xpu()
        out = torch.empty_like(dq)
        lengths = (
            [max(0, length - m + i + 1) for i in range(m)]
            if single_queries
            else [length]
        )
        offsets = list(range(m + 1)) if single_queries else [0, m]
        tables = (
            device_ids.int()[None].repeat(m, 1)
            if single_queries
            else device_ids.int()[None]
        )
        # Preserve the existing paged-LSE rejection.
        context = (
            pytest.raises(RuntimeError, match="softmax_lse output")
            if lse
            else nullcontext()
        )
        with context:
            result = flash_attn_varlen_func(
                dq,
                dk,
                dv,
                max_seqlen_q=1 if single_queries else m,
                cu_seqlens_q=torch.tensor(
                    offsets, device="xpu", dtype=torch.int32
                ),
                max_seqlen_k=max(16384, length),
                seqused_k=torch.tensor(
                    lengths, device="xpu", dtype=torch.int32
                ),
                block_table=tables,
                causal=causal,
                softmax_scale=1.0,
                window_size=(1023, 0) if dim == 256 else (-1, -1),
                num_splits_kv=splits,
                return_softmax_lse=lse,
                out=out,
            )
    stream.synchronize()
    if lse:
        return
    actual = result
    assert actual.data_ptr() == out.data_ptr()
    torch.testing.assert_close(
        actual.cpu().float(), torch.stack(expected), atol=2e-3, rtol=1e-2
    )


@pytest.mark.parametrize(
    "heads,dim", [(8, 256), (16, 256), (8, 512), (16, 512)]
)
@torch.inference_mode()
def test_small_m_attention_boundary_signal(heads, dim):
    # Future keys have deliberately larger scores and distinct values. A
    # missing causal mask must change the result by order-one amounts.
    m, length = 5, 32773
    page = 32 if dim == 256 else 64
    kv = heads // (2 if dim == 256 else 8)
    pages = (length + page - 1) // page
    k = torch.zeros(pages * page, kv, dim, dtype=torch.float16)
    v = torch.zeros_like(k)
    q = torch.zeros(m, heads, dim, dtype=torch.float16)
    q[:, :, 0] = 20
    for i in range(m):
        k[length - m + i, :, 0] = i + 1
        v[length - m + i] = (i + 1) * 0.1
    if dim == 256:
        # Included only by row0; row1's advancing left window must exclude it.
        position = length - m + 1 - 1024
        k[position, :, 0] = 10
        v[position] = -0.75
    refs = []
    for i in range(m):
        end = length - m + i + 1
        start = max(0, end - 1024) if dim == 256 else 0
        probabilities = (20 * k[start:end, 0, 0].float()).softmax(0)
        ref = (probabilities[:, None] * v[start:end, 0].float()).sum(0)
        refs.append(ref.repeat(heads, 1))
    generator = torch.Generator().manual_seed(2918)
    ids = torch.randperm(pages + 3, generator=generator)[:pages].xpu()
    cache = torch.empty(
        pages + 3, page, kv, 2, dim, dtype=torch.float16, device="xpu"
    )
    dk, dv = cache[:, :, :, 0], cache[:, :, :, 1]
    dk.index_copy_(0, ids, k.reshape(pages, page, kv, dim).xpu())
    dv.index_copy_(0, ids, v.reshape(pages, page, kv, dim).xpu())
    actual = flash_attn_varlen_func(
        q.xpu(),
        dk,
        dv,
        max_seqlen_q=m,
        cu_seqlens_q=torch.tensor([0, m], dtype=torch.int32, device="xpu"),
        max_seqlen_k=length,
        seqused_k=torch.tensor([length], dtype=torch.int32, device="xpu"),
        block_table=ids.int()[None],
        causal=True,
        softmax_scale=1.0,
        window_size=(1023, 0) if dim == 256 else (-1, -1),
    )
    torch.testing.assert_close(
        actual.cpu().float(), torch.stack(refs), atol=1e-4, rtol=1e-3
    )
