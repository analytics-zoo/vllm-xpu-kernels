"""Isolated HC small-M DPAS probe; never imports a second HC native DSO."""

import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


@pytest.fixture(scope="module")
def probe():
    library = os.environ.get("QWEN38_HC_DPAS_PROBE_DSO")
    if not library:
        pytest.skip("set QWEN38_HC_DPAS_PROBE_DSO to an isolated probe SO")
    if os.environ.get("ZE_AFFINITY_MASK") != "7":
        pytest.skip("requires the authorized physical GPU7")
    if not torch.xpu.is_available():
        pytest.skip("XPU unavailable")
    torch.ops.load_library(str(Path(library).resolve(strict=True)))
    return torch.ops.qwen38_hc_dpas_probe.up_gate_mix


def golden(input_, weight, normed):
    gate = F.linear(input_.float(), weight.float()).half().float()
    return (normed.float().reshape(-1, 4, 2560) *
            torch.sigmoid(gate.reshape(-1, 4, 2560))).sum(1).mul(0.25).half()


def invoke(probe, input_, weight, normed):
    output = torch.empty((input_.shape[0], 2560),
                         dtype=torch.float16,
                         device="xpu")
    probe(input_, weight, normed, output)
    return output


def test_position_coded_vnni_fragment_and_four_branches(probe):
    # Each output column has one selected (branch,K) term. The K/column
    # pattern crosses every K16 tile and both halves of the 16-column tile.
    m = 5
    h = torch.arange(2560, device="xpu")
    branch = h.remainder(4)
    col = branch * 2560 + h
    k = (h * 37 + 11).remainder(320)
    values = ((h * 19).remainder(31).float() - 15).mul(0.075).half()
    weight = torch.zeros((10240, 320), device="xpu", dtype=torch.float16)
    weight[col, k] = values
    axis = torch.arange(320, device="xpu")
    x_base = ((axis * 23).remainder(47).float() - 23).mul(0.05).half()
    row_scale = torch.tensor([0.63, -0.75, 1.3, -0.9, 0.8],
                             device="xpu",
                             dtype=torch.float16)
    input_ = (row_scale[:, None] * x_base[None, :]).contiguous()
    normed = torch.zeros((m, 10240), device="xpu", dtype=torch.float16)
    normed[:, col] = 1

    actual = invoke(probe, input_, weight, normed)
    product = input_[:, k].float() * values.float()[None, :]
    expected = (torch.sigmoid(product.half().float()) * 0.25).half()
    error = (actual.float() - expected.float()).abs()
    assert int((error > 5e-4).sum()) == 0, (float(error.max()),
                                              int((error > 5e-4).sum()))


@pytest.mark.parametrize("m", [2, 5, 8])
def test_fp16_gate_boundary_strong_discriminator(probe, m):
    torch.manual_seed(40100 + m)
    input_ = torch.zeros((m, 320), device="xpu", dtype=torch.float16)
    selected = [0.9, -0.7, 1.3, 0.63, -1.1, 0.75, 0.8, -0.95]
    for row in range(m):
        input_[row, 0] = selected[row]
    weight = torch.zeros((10240, 320), device="xpu", dtype=torch.float16)
    weight[:, 0] = torch.randn(10240, device="xpu", dtype=torch.float16)
    normed = torch.randn((m, 10240),
                         device="xpu",
                         dtype=torch.float16)
    actual = invoke(probe, input_, weight, normed)

    product = input_[:, :1].float() * weight[:, 0].float()[None, :]
    rounded = (normed.float().reshape(m, 4, 2560) *
               torch.sigmoid(product.half().float().reshape(m, 4,
                                                             2560))).sum(1)
    rounded = (rounded * 0.25).half()
    extra_fp32 = (normed.float().reshape(m, 4, 2560) *
                  torch.sigmoid(product.reshape(m, 4, 2560))).sum(1)
    extra_fp32 = (extra_fp32 * 0.25).half()
    rounded_mismatch = int((actual != rounded).sum())
    extra_fp32_mismatch = int((actual != extra_fp32).sum())
    assert rounded_mismatch <= 16 * m
    assert extra_fp32_mismatch >= 200 * m
    assert extra_fp32_mismatch >= 10 * max(rounded_mismatch, 1)


@pytest.mark.parametrize("m,stride", [(2, 320), (5, 336), (8, 336)])
def test_full_dense_and_real_down_stride(probe, m, stride):
    torch.manual_seed(40200 + m)
    backing = torch.randn((m, stride), device="xpu", dtype=torch.float16)
    input_ = backing[:, :320]
    weight = torch.randn((10240, 320), device="xpu",
                         dtype=torch.float16) / 20
    normed = torch.randn((m, 10240), device="xpu",
                         dtype=torch.float16) * 0.2
    actual = invoke(probe, input_, weight, normed)
    torch.testing.assert_close(actual,
                               golden(input_, weight, normed),
                               atol=3e-3,
                               rtol=2e-3)


@pytest.mark.parametrize("invalid", ["m1", "weight_offset2", "alias"])
def test_preflight_error_submits_no_kernel(probe, invalid):
    from torch.profiler import ProfilerActivity, profile

    m = 1 if invalid == "m1" else 5
    input_ = torch.ones((m, 320), device="xpu", dtype=torch.float16)
    weight = torch.ones((10240, 320), device="xpu", dtype=torch.float16)
    normed = torch.ones((m, 10240), device="xpu", dtype=torch.float16)
    output = torch.full((m, 2560), 7, device="xpu", dtype=torch.float16)
    if invalid == "weight_offset2":
        backing = torch.empty((10240 * 320 + 1, ),
                              device="xpu",
                              dtype=torch.float16)
        weight = backing[1:].view(10240, 320)
    if invalid == "alias":
        output = normed.flatten()[:m * 2560].view(m, 2560)
    torch.xpu.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]
                 ) as prof, pytest.raises(RuntimeError):
        probe(input_, weight, normed, output)
    assert not [e for e in prof.events()
                if e.device_type != torch.autograd.DeviceType.CPU]
    if invalid != "alias":
        assert torch.all(output == 7)


def test_one_kernel_and_nondefault_stream_dropped_input(probe):
    from torch.profiler import ProfilerActivity, profile

    m = 5
    torch.manual_seed(40305)
    input_ = torch.randn((m, 320), device="xpu", dtype=torch.float16) * 0.2
    weight = torch.randn((10240, 320), device="xpu",
                         dtype=torch.float16) / 20
    normed = torch.randn((m, 10240), device="xpu",
                         dtype=torch.float16) * 0.2
    expected = golden(input_, weight, normed)
    output = torch.empty((m, 2560), device="xpu", dtype=torch.float16)
    torch.xpu.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]
                 ) as prof:
        probe(input_, weight, normed, output)
        torch.xpu.synchronize()
    events = [e for e in prof.events()
              if e.device_type != torch.autograd.DeviceType.CPU]
    assert len(events) == 1, [e.name for e in events]

    stream = torch.xpu.Stream()
    stream.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(stream):
        temporary = input_ * 1.0
        probe(temporary, weight, normed, output)
        del temporary
        for _ in range(8):
            torch.empty((4096, 4096), device="xpu", dtype=torch.float16)
    stream.synchronize()
    torch.testing.assert_close(output, expected, atol=3e-3, rtol=2e-3)
