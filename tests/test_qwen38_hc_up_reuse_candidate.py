"""Isolated M2--6 HC Up reuse checks; set QWEN38_HC_CANDIDATE_DSO."""

import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


@pytest.fixture(scope="module")
def hc():
    library = os.environ.get("QWEN38_HC_CANDIDATE_DSO")
    if not library:
        pytest.skip("set QWEN38_HC_CANDIDATE_DSO to an isolated candidate SO")
    if os.environ.get("ZE_AFFINITY_MASK") not in {"4", "5", "6", "7"}:
        pytest.skip("requires one authorized physical GPU via ZE_AFFINITY_MASK")
    if not torch.xpu.is_available():
        pytest.skip("XPU unavailable")
    torch.ops.load_library(str(Path(library).resolve(strict=True)))
    return torch.ops.qwen38_hc_sycl


def golden_mix(lowrank, weight, normed):
    gate = F.linear(lowrank.float(), weight.float()).half().float()
    return (normed.float().reshape(-1, 4, 2560) *
            torch.sigmoid(gate.reshape(-1, 4, 2560))).sum(1).mul(0.25).half()


@pytest.mark.parametrize("m", [2, 3, 4, 5, 6])
def test_one_term_independent_fp16_gate_discriminator(hc, m):
    # One product removes GEMV reduction ambiguity. Distinct token values
    # catch a wrong row/stride while half-vs-FP32 gate mixing is separable.
    torch.manual_seed(19080 + m)
    lowrank = torch.zeros((m, 320), device="xpu", dtype=torch.float16)
    values = [0.9, -0.7, 1.3, 0.63, -1.1, 0.75]
    for row in range(m):
        lowrank[row, 0] = values[row]
    weight = torch.zeros((10240, 320), device="xpu", dtype=torch.float16)
    weight[:, 0] = torch.randn(10240, device="xpu", dtype=torch.float16)
    normed = torch.randn((m, 10240), device="xpu", dtype=torch.float16)
    output = torch.empty((m, 2560), device="xpu", dtype=torch.float16)
    hc.up_gate_mix(lowrank, weight, normed, output)

    product = lowrank[:, :1].float() * weight[:, 0].float()[None, :]
    rounded = (normed.float().reshape(m, 4, 2560) *
               torch.sigmoid(product.half().float().reshape(m, 4,
                                                             2560))).sum(1)
    rounded = (rounded * 0.25).half()
    unrounded = (normed.float().reshape(m, 4, 2560) *
                 torch.sigmoid(product.reshape(m, 4, 2560))).sum(1)
    unrounded = (unrounded * 0.25).half()
    rounded_mismatch = int((output != rounded).sum())
    unrounded_mismatch = int((output != unrounded).sum())
    assert rounded_mismatch <= 16 * m
    assert unrounded_mismatch >= 200 * m
    assert unrounded_mismatch >= 10 * max(rounded_mismatch, 1)


@pytest.mark.parametrize("m", [2, 3, 4, 5, 6])
def test_full_projection_strided_and_offset2_fallback(hc, m):
    torch.manual_seed(20100 + m)
    backing = torch.randn((m, 336), device="xpu", dtype=torch.float16)
    lowrank = backing[:, :320]
    weight = torch.randn((10240, 320), device="xpu",
                         dtype=torch.float16) / 20
    normed = torch.randn((m, 10240), device="xpu",
                         dtype=torch.float16) * 0.2
    expected = golden_mix(lowrank, weight, normed)
    aligned = torch.empty((m, 2560), device="xpu", dtype=torch.float16)
    hc.up_gate_mix(lowrank, weight, normed, aligned)
    torch.testing.assert_close(aligned, expected, atol=3e-3, rtol=2e-3)

    # Both odd storage offsets are legal half views but not legal u32 reads.
    odd_low_storage = torch.empty((m, 338), device="xpu",
                                  dtype=torch.float16)
    odd_low = odd_low_storage[:, 1:321]
    odd_low.copy_(lowrank)
    odd_weight_storage = torch.empty((10240 * 320 + 1, ), device="xpu",
                                     dtype=torch.float16)
    odd_weight = odd_weight_storage[1:].view(10240, 320)
    odd_weight.copy_(weight)
    unaligned = torch.empty_like(aligned)
    hc.up_gate_mix(odd_low, odd_weight, normed, unaligned)
    torch.testing.assert_close(unaligned, aligned, atol=0, rtol=0)


@pytest.mark.parametrize("m", [1, 7, 8])
def test_unmodified_fused_dispatch_and_nonfused_up(hc, m):
    torch.manual_seed(20200 + m)
    lowrank = torch.randn((m, 320), device="xpu",
                          dtype=torch.float16) * 0.2
    weight = torch.randn((10240, 320), device="xpu",
                         dtype=torch.float16) / 20
    normed = torch.randn((m, 10240), device="xpu",
                         dtype=torch.float16) * 0.2
    output = torch.empty((m, 2560), device="xpu", dtype=torch.float16)
    hc.up_gate_mix(lowrank, weight, normed, output)
    torch.testing.assert_close(output,
                               golden_mix(lowrank, weight, normed),
                               atol=3e-3,
                               rtol=2e-3)
    if m == 7:
        raw = torch.empty((m, 10240), device="xpu", dtype=torch.float16)
        hc.up(lowrank, weight, raw)
        torch.testing.assert_close(raw,
                                   F.linear(lowrank.float(),
                                            weight.float()).half(),
                                   atol=3e-3,
                                   rtol=2e-3)


def test_alias_preflight_and_dropped_input_on_current_stream(hc):
    m = 5
    torch.manual_seed(20305)
    lowrank = torch.randn((m, 320), device="xpu", dtype=torch.float16)
    weight = torch.randn((10240, 320), device="xpu",
                         dtype=torch.float16) / 20
    normed = torch.full((m, 10240), 0.25, device="xpu",
                        dtype=torch.float16)
    aliased_output = normed.flatten()[:m * 2560].view(m, 2560)
    with pytest.raises(RuntimeError):
        hc.up_gate_mix(lowrank, weight, normed, aliased_output)
    assert torch.all(normed == 0.25)

    expected = golden_mix(lowrank, weight, normed)
    stream = torch.xpu.Stream()
    with torch.xpu.stream(stream):
        temporary = lowrank * 1.0
        output = torch.empty((m, 2560), device="xpu", dtype=torch.float16)
        hc.up_gate_mix(temporary, weight, normed, output)
        del temporary
        for _ in range(8):
            torch.empty((4096, 4096), device="xpu", dtype=torch.float16)
    stream.synchronize()
    torch.testing.assert_close(output, expected, atol=3e-3, rtol=2e-3)
