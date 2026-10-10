# SPDX-License-Identifier: Apache-2.0
"""Q4_0 bits, FP32 projection, and asynchronous ownership contracts."""

import gc
import importlib
import os

import pytest
import torch


@pytest.fixture(scope="module")
def ops():
    if not torch.xpu.is_available():
        pytest.skip("requires XPU")
    library = os.environ.get("QWEN38_SYCL_LIBRARY")
    if library:
        torch.ops.load_library(library)
    else:
        importlib.import_module("vllm_xpu_kernels.qwen38")
    return torch.ops._qwen38_C


def quantize_reference(weight):
    blocks = weight.float().reshape(weight.shape[0], -1, 128)
    pos = blocks.amax(-1)
    neg = (-blocks).amax(-1)
    signed = torch.where(pos >= neg, pos, -neg)
    d = signed / -8.0
    inv_d = torch.where(d != 0, d.reciprocal(), 0)
    q = (blocks * inv_d.unsqueeze(-1) + 8.5).trunc().clamp(0, 15)
    q = q.to(torch.int64).reshape(weight.shape[0], -1, 8)
    shifts = torch.arange(8, dtype=torch.int64) * 4
    packed = (q << shifts).sum(-1).to(torch.int32)
    return packed, d.half()


def projection_reference(x, weight, scale):
    weight = weight.cpu()
    unpacked = torch.empty((weight.shape[0], weight.shape[1] * 2))
    unpacked[:, 0::2] = (weight & 15).float() - 8
    unpacked[:, 1::2] = (weight >> 4).float() - 8
    unpacked *= scale.cpu().float().repeat_interleave(128, dim=1)
    if x.shape[0] > 1:
        unpacked = unpacked.half().float()
    return (x.cpu().float() @ unpacked.T).half()


@pytest.mark.parametrize("m", range(2, 9))
@pytest.mark.parametrize("n,k", [(4096, 2560), (2560, 1536), (2560, 2560)])
def test_small_m_xmx_matches_half_dequant_fp32_golden(ops, m, n, k):
    torch.manual_seed(102)
    x = torch.randn(m, k, dtype=torch.float16, device="xpu")
    weight = torch.randint(256, (n, k // 2), dtype=torch.uint8, device="xpu")
    scale = (torch.randn(n, k // 128) * .01).half().to("xpu")
    output = torch.empty(m, n, dtype=torch.float16, device="xpu")
    ops.int4_linear(x, weight, scale, output)
    reference = projection_reference(x, weight, scale)
    actual = output.cpu()
    torch.testing.assert_close(actual, reference, rtol=.003, atol=.003)
    near_zero = reference.abs() < .05
    assert (actual[near_zero] - reference[near_zero]).abs().max() <= .00005


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("pattern", ["random", "tie", "zero", "rounding"])
def test_quantization_matches_independent_pack_and_signed_scale(
        ops, dtype, pattern):
    torch.manual_seed(27)
    weight = torch.randn(3, 384).to(dtype)
    if pattern == "tie":
        weight[:, 0::128] = 4
        weight[:, 1::128] = -4
        weight.clamp_(-4, 4)
    elif pattern == "zero":
        weight.zero_()
    elif pattern == "rounding":
        values = (torch.arange(128) % 16 - 8).float() + 0.5
        weight = values.repeat(3, 3).to(dtype)
        weight[:, 0::128] = -8
    packed_ref, scale_ref = quantize_reference(weight)
    weight = weight.to("xpu")
    packed = torch.empty_like(packed_ref, device="xpu")
    scale = torch.empty_like(scale_ref, device="xpu")
    ops.q4_0_quantize(weight, packed, scale)
    torch.testing.assert_close(packed.cpu(), packed_ref, rtol=0, atol=0)
    torch.testing.assert_close(scale.cpu().view(torch.int16),
                               scale_ref.view(torch.int16),
                               rtol=0,
                               atol=0)


@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6, 7, 8, 17, 64])
@pytest.mark.parametrize("n,k", [(12, 2560), (24, 2560), (17, 384),
                                 (128, 2048), (256, 2560)])
def test_projection_matches_fp32_with_tail_tokens_and_rows(ops, m, n, k):
    torch.manual_seed(51)
    x = torch.randn(m, k, dtype=torch.float16, device="xpu")
    weight = torch.randint(256, (n, k // 2), dtype=torch.uint8, device="xpu")
    scale = (torch.randn(n, k // 128) * 0.04).half().to("xpu")
    output = torch.empty(m, n, dtype=torch.float16, device="xpu")
    ops.int4_linear(x, weight, scale, output)
    torch.testing.assert_close(output.cpu(),
                               projection_reference(x, weight, scale),
                               rtol=1e-3,
                               atol=2e-3)


def test_contiguous_storage_offsets_do_not_require_vector_alignment(ops):
    torch.manual_seed(5)
    m, n, k = 3, 17, 384
    x = torch.randn(m * k + 1, dtype=torch.float16, device="xpu")[1:]
    x = x.reshape(m, k)
    w = torch.randint(256, (n * (k // 2) + 1, ),
                      dtype=torch.uint8,
                      device="xpu")[1:].reshape(n, k // 2)
    s = torch.randn(n * (k // 128) + 1, dtype=torch.float16,
                    device="xpu")[1:].reshape(n, k // 128) * 0.02
    # Multiplication materializes aligned storage; make an actual offset view.
    offset_scale = torch.empty(s.numel() + 1, dtype=s.dtype, device=s.device)
    offset_scale[1:].copy_(s.flatten())
    s = offset_scale[1:].reshape_as(s)
    out = torch.empty(m * n + 1, dtype=x.dtype, device=x.device)[1:]
    out = out.reshape(m, n)
    ops.int4_linear(x, w, s, out)
    torch.testing.assert_close(out.cpu(),
                               projection_reference(x, w, s),
                               rtol=1e-3,
                               atol=2e-3)


def test_fused2_writes_both_projections_including_matrix_boundary(ops):
    torch.manual_seed(6)
    x = torch.randn(1, 384, dtype=torch.float16, device="xpu")
    pairs = []
    for n in (17, 12):
        w = torch.randint(256, (n, 192), dtype=torch.uint8, device="xpu")
        s = (torch.randn(n, 3) * 0.03).half().to("xpu")
        out = torch.empty(1, n, dtype=torch.float16, device="xpu")
        pairs.append((w, s, out))
    ops.int4_linear_fused2(x, *pairs[0], *pairs[1])
    for w, s, out in pairs:
        torch.testing.assert_close(out.cpu(),
                                   projection_reference(x, w, s),
                                   rtol=1e-3,
                                   atol=2e-3)


@pytest.mark.parametrize("offsets", [(0, 0, 0), (1, 0, 0), (2, 0, 0),
                                     (0, 1, 0), (0, 0, 1)])
@pytest.mark.parametrize("widths", [(4096, 24), (129, 5)])
def test_wide_fused2_actual_tp4_and_fallback_alignment(ops, offsets, widths):
    torch.manual_seed(109)
    k = 2560
    x_offset, w_offset, s_offset = offsets
    x = torch.randn(k + x_offset, dtype=torch.float16,
                    device="xpu")[x_offset:].view(1, k)
    pairs = []
    for n in widths:
        w = torch.randint(256, (n * (k // 2) + w_offset, ),
                          dtype=torch.uint8,
                          device="xpu")[w_offset:].view(n, k // 2)
        s = (torch.randn(n * (k // 128) + s_offset, device="xpu") *
             .02).half()[s_offset:].view(n, k // 128)
        out = torch.empty(n + 1, dtype=torch.float16,
                          device="xpu")[1:].view(1, n)
        pairs.append((w, s, out))
    ops.int4_linear_fused2(x, *pairs[0], *pairs[1])
    for w, s, out in pairs:
        torch.testing.assert_close(out.cpu(),
                                   projection_reference(x, w, s),
                                   rtol=1e-3,
                                   atol=2e-3)


def test_lazy_output_rejected_without_materializing_or_writing(ops):
    x = torch.ones(1, 128, dtype=torch.float16, device="xpu")
    w = torch.zeros(16, 64, dtype=torch.uint8, device="xpu")
    s = torch.ones(16, 1, dtype=torch.float16, device="xpu")
    output = torch.full((1, 16), 99, dtype=torch.float16, device="xpu")
    with pytest.raises(RuntimeError, match="lazy view"):
        ops.int4_linear(x, w, s, output._neg_view())
    assert torch.all(output == 99).item()


def test_fused2_validates_second_matrix_before_writing_first(ops):
    x = torch.ones(1, 128, dtype=torch.float16, device="xpu")
    w = torch.zeros(16, 64, dtype=torch.uint8, device="xpu")
    s = torch.ones(16, 1, dtype=torch.float16, device="xpu")
    out0 = torch.full((1, 16), 123, dtype=torch.float16, device="xpu")
    out1 = torch.full_like(out0, 456)
    with pytest.raises(RuntimeError, match="scale must be"):
        ops.int4_linear_fused2(x, w, s, out0, w, s[:15], out1)
    assert torch.all(out0 == 123).item()
    assert torch.all(out1 == 456).item()


def test_alias_is_rejected_without_corrupting_input(ops):
    x = torch.ones(1, 128, dtype=torch.float16, device="xpu")
    w = torch.zeros(128, 64, dtype=torch.uint8, device="xpu")
    s = torch.ones(128, 1, dtype=torch.float16, device="xpu")
    with pytest.raises(RuntimeError, match="single memory location|overlap"):
        ops.int4_linear(x, w, s, x)
    assert torch.all(x == 1).item()


def test_current_stream_and_dropped_input_lifetime(ops):
    torch.manual_seed(9)
    m, n, k = 5, 128, 2560
    x = torch.randn(m, k, dtype=torch.float16, device="xpu")
    w = torch.randint(256, (n, k // 2), dtype=torch.uint8, device="xpu")
    s = (torch.randn(n, k // 128) * 0.03).half().to("xpu")
    ref = projection_reference(x, w, s)
    out = torch.empty(m, n, dtype=torch.float16, device="xpu")
    stream = torch.xpu.Stream()
    stream.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(stream):
        # A producer on the same stream must precede the native consumer.
        x.mul_(2)
        ops.int4_linear(x, w, s, out)
    del x, w, s
    gc.collect()
    # Encourage allocator reuse on the tensors' original stream.
    junk = [
        torch.full((m, k), 77, dtype=torch.float16, device="xpu")
        for _ in range(8)
    ]
    stream.synchronize()
    torch.testing.assert_close(out.cpu(), (ref.float() * 2).half(),
                               rtol=1e-3,
                               atol=4e-3)
    del junk


def test_dlpack_physical_alias_rejected_before_submit(ops):
    x = torch.ones(1, 128, dtype=torch.float16, device="xpu")
    output = torch.from_dlpack(x)
    assert output.data_ptr() == x.data_ptr()
    assert not torch._C._overlaps(output, x)  # Independent Storage owners.
    weight = torch.zeros(128, 64, dtype=torch.uint8, device="xpu")
    scale = torch.ones(128, 1, dtype=torch.float16, device="xpu")
    with pytest.raises(RuntimeError, match="overlap"):
        ops.int4_linear(x, weight, scale, output)
    assert torch.all(x == 1).item()
