# SPDX-License-Identifier: Apache-2.0
"""Standalone TP4 Qwen3.8 HC ordinary-SYCL tests (ZE_AFFINITY_MASK=5)."""

import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

LIBRARY = Path(
    os.environ.get(
        "QWEN38_HC_SYCL_LIBRARY",
        Path(__file__).resolve().parents[2] /
        "qwen38_sycl_20261004/hc_build/qwen38_hc_sycl.so",
    ))
FINAL_REFERENCE = (Path(__file__).resolve().parents[2] /
                   "qwen38_sycl_20261004/hc_final_esimd_reference_build" /
                   "qwen38_hc_final_esimd_reference.so")


@pytest.fixture(scope="module")
def hc():
    if not torch.xpu.is_available():
        pytest.skip("XPU unavailable")
    torch.ops.load_library(str(LIBRARY))
    return torch.ops.qwen38_hc_sycl


@pytest.fixture(scope="module")
def final_reference():
    if os.environ.get("QWEN38_INDEPENDENT_ONLY") == "1":
        return None
    if not FINAL_REFERENCE.is_file():
        pytest.skip("build-only final-header ESIMD oracle unavailable")
    torch.ops.load_library(str(FINAL_REFERENCE))
    return torch.ops.qwen38_hc_final_esimd_reference


@pytest.mark.parametrize("m", [1, 2])
def test_pair_default_dispatch_only_m1(hc, monkeypatch, m):
    monkeypatch.delenv("QWEN38_HC_M1_UP_GATE_PAIR", raising=False)
    low = torch.randn(m, 320, device="xpu", dtype=torch.float16) * 0.1
    weight = torch.randn(10240, 320, device="xpu", dtype=torch.float16) / 20
    normed = torch.randn(m, 10240, device="xpu", dtype=torch.float16)
    output = torch.empty(m, 2560, device="xpu", dtype=torch.float16)
    hc.up_gate_mix(low, weight, normed, output)
    torch.xpu.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.XPU]
    ) as prof:
        hc.up_gate_mix(low, weight, normed, output)
    hit = any("UpGateM1PairKernel" in event.name for event in prof.events())
    assert hit == (m == 1)


def inputs(m):
    torch.manual_seed(1908 + m)
    x = torch.randn(m, 10240, device="xpu", dtype=torch.float16) * 0.2
    block = torch.randn(m, 2560, device="xpu", dtype=torch.float16) * 0.2
    inj = torch.randn(m, 4, device="xpu", dtype=torch.float16)
    gamma = torch.randn(10240, device="xpu", dtype=torch.float16) * 0.03
    return x, block, inj, gamma


def ref_combine(x, block, inj):
    scale = 2 * torch.sigmoid(inj.float() / 4)
    return (x.float().reshape(-1, 4, 2560) +
            block.float()[:, None, :] * scale[:, :, None]).flatten(-2).half()


def ref_norm(x, gamma, eps=1e-6):
    grouped = x.float().reshape(-1, 4, 2560)
    var = grouped.square().mean(-1, keepdim=True)
    normalized = grouped * torch.rsqrt(var + eps)
    return (normalized.flatten(-2) * (1 + gamma.float())).half()


@pytest.fixture(scope="module")
def prefill_hc():
    if not torch.xpu.is_available():
        pytest.skip("XPU unavailable")
    torch.ops.load_library(str(LIBRARY))
    return torch.ops._qwen38_C


@pytest.mark.parametrize("m", [9, 17, 128, 4096])
@pytest.mark.parametrize("injection_stride", [4, 336])
@pytest.mark.parametrize("norm_local", ["256", "512"])
def test_prefill_math_and_fp16_boundary(prefill_hc, monkeypatch, m,
                                      injection_stride, norm_local):
    monkeypatch.setenv("QWEN38_HC_PREFILL_NORM_LOCAL", norm_local)
    x, block, injection, gamma = inputs(m)
    padded = torch.empty(
        (m, injection_stride), device="xpu", dtype=torch.float16)
    padded[:, :4] = injection
    strided_injection = padded[:, :4]
    combined = torch.empty_like(x)
    normed = torch.empty_like(x)
    prefill_hc.hc_prefill_combine_norm(
        x, block, strided_injection, gamma, combined, normed, 1e-6)
    expected_combined = ref_combine(x, block, injection)
    torch.testing.assert_close(
        combined, expected_combined, atol=1e-3, rtol=0)
    torch.testing.assert_close(
        normed, ref_norm(expected_combined, gamma), atol=3e-3, rtol=2e-3)


    prefill_hc.hc_prefill_combine(
        x, block, strided_injection, combined)
    torch.testing.assert_close(
        combined, expected_combined, atol=1e-3, rtol=0)
    prefill_hc.hc_prefill_grouped_norm(x, gamma, normed, 1e-6)
    torch.testing.assert_close(
        normed, ref_norm(x, gamma), atol=3e-3, rtol=2e-3)
    gate = torch.randn_like(x) * 0.3
    mixed = torch.empty_like(block)
    prefill_hc.hc_prefill_gate_mix(normed, gate, mixed)
    expected_mix = (torch.sigmoid(gate.float().reshape(m, 4, 2560)) *
                    normed.float().reshape(m, 4, 2560)).mean(dim=1).half()
    torch.testing.assert_close(
        mixed, expected_mix, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("m", range(1, 9))
def test_multi_up_pair_preserves_bits_and_m1(hc, monkeypatch, m):
    torch.manual_seed(613 + m)
    backing = torch.randn(m, 336, device="xpu", dtype=torch.float16) * 0.1
    weight = torch.randn(10240, 320, device="xpu", dtype=torch.float16) / 20
    normed = torch.randn(m, 10240, device="xpu", dtype=torch.float16) * 0.2
    expected = torch.empty(m, 2560, device="xpu", dtype=torch.float16)
    output = torch.empty_like(expected)
    monkeypatch.setenv("QWEN38_HC_MULTI_UP_GATE_PAIR", "0")
    hc.up_gate_mix(backing[:, :320], weight, normed, expected)
    monkeypatch.setenv("QWEN38_HC_MULTI_UP_GATE_PAIR", "1")
    hc.up_gate_mix(backing[:, :320], weight, normed, output)
    assert torch.equal(output, expected)


def test_prefill_preflight_preserves_outputs(prefill_hc):
    x, block, injection, gamma = inputs(9)
    output = torch.full_like(x, 7)
    with pytest.raises(RuntimeError):
        prefill_hc.hc_prefill_combine_norm(
            x, block, injection, gamma, output, output, 1e-6)
    assert torch.all(output == 7)
    for eps in (0.0, float("nan"), float("inf"), 1e-50):
        with pytest.raises(RuntimeError):
            prefill_hc.hc_prefill_grouped_norm(x, gamma, output, eps)
        assert torch.all(output == 7)
    with pytest.raises(RuntimeError):
        prefill_hc.hc_prefill_combine(x, block, injection, x)
    small_x, _, _, small_gamma = inputs(8)
    with pytest.raises(RuntimeError):
        prefill_hc.hc_prefill_grouped_norm(
            small_x, small_gamma, torch.empty_like(small_x), 1e-6)


def test_prefill_async_lifetime(prefill_hc):
    x, block, injection, gamma = inputs(17)
    combined = torch.empty_like(x)
    normed = torch.empty_like(x)
    stream = torch.xpu.Stream()
    stream.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(stream):
        temporary = x * 1.0
        prefill_hc.hc_prefill_combine_norm(
            temporary, block, injection, gamma, combined, normed, 1e-6)
        del temporary
        for _ in range(8):
            torch.empty_like(x)
    stream.synchronize()
    expected_combined = ref_combine(x, block, injection)
    torch.testing.assert_close(
        combined, expected_combined, atol=1e-3, rtol=0)
    torch.testing.assert_close(
        normed, ref_norm(expected_combined, gamma), atol=3e-3, rtol=2e-3)


def ref_down(x, weight):
    linear = F.linear(x.float(), weight.float()).half()
    scaled = (linear[:, :320].float() * 0.25).half()
    activated = F.silu(scaled.float()).half()
    if linear.shape[1] == 320:
        return activated
    return torch.cat((activated, linear[:, 320:]), dim=-1)


def ref_up_gate(low, weight, normed):
    gate = F.linear(low.float(), weight.float()).half().float()
    return (torch.sigmoid(gate).reshape(-1, 4, 2560) *
            normed.float().reshape(-1, 4, 2560)).mean(dim=1).half()


@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 8])
def test_norm_combine_and_gate(hc, m):
    x, block, inj, gamma = inputs(m)
    expected_combined = ref_combine(x, block, inj)
    expected_norm = ref_norm(expected_combined, gamma)
    combined = torch.empty_like(x)
    normed = torch.empty_like(x)
    hc.combine_norm(x, block, inj, gamma, combined, normed, 1e-6)
    torch.testing.assert_close(combined, expected_combined, atol=1e-3, rtol=0)
    torch.testing.assert_close(normed, expected_norm, atol=3e-3, rtol=2e-3)

    grouped = torch.empty_like(x)
    hc.grouped_norm(x, gamma, grouped, 1e-6)
    torch.testing.assert_close(grouped,
                               ref_norm(x, gamma),
                               atol=3e-3,
                               rtol=2e-3)
    only_combined = torch.empty_like(x)
    hc.combine(x, block, inj, only_combined)
    torch.testing.assert_close(only_combined,
                               expected_combined,
                               atol=1e-3,
                               rtol=0)

    gate = torch.randn_like(x) * 0.3
    mixed = torch.empty((m, 2560), device="xpu", dtype=torch.float16)
    hc.gate_mix(normed, gate, mixed)
    expected_mix = (torch.sigmoid(gate.float().reshape(m, 4, 2560)) *
                    normed.float().reshape(m, 4, 2560)).mean(dim=1).half()
    torch.testing.assert_close(mixed, expected_mix, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("m", range(1, 9))
@pytest.mark.parametrize("n", [320, 336])
def test_projection_and_fused_up(hc, m, n):
    torch.manual_seed(321 + m + n)
    x = torch.randn((m, 10240), device="xpu", dtype=torch.float16) * 0.2
    down_w = (torch.randn((n, 10240), device="xpu", dtype=torch.float16) / 32)
    up_w = (torch.randn((10240, 320), device="xpu", dtype=torch.float16) / 20)
    down = torch.empty((m, n), device="xpu", dtype=torch.float16)
    hc.down(x, down_w, down)
    torch.testing.assert_close(down, ref_down(x, down_w), atol=3e-3, rtol=2e-3)

    low = down[:, :320]
    up = torch.empty((m, 10240), device="xpu", dtype=torch.float16)
    hc.up(low, up_w, up)
    torch.testing.assert_close(up,
                               F.linear(low.float(), up_w.float()).half(),
                               atol=3e-3,
                               rtol=2e-3)
    normed = torch.randn_like(x) * 0.2
    mixed = torch.empty((m, 2560), device="xpu", dtype=torch.float16)
    hc.up_gate_mix(low, up_w, normed, mixed)
    torch.testing.assert_close(mixed,
                               ref_up_gate(low, up_w, normed),
                               atol=3e-3,
                               rtol=2e-3)


@pytest.mark.parametrize("m", [1, 4, 8])
def test_combined_host_chain_and_strided_injection(hc, m):
    x, block, inj, gamma = inputs(m)
    down_w = torch.randn((336, 10240), device="xpu", dtype=torch.float16) / 32
    up_w = torch.randn((10240, 320), device="xpu", dtype=torch.float16) / 20
    inject_pad = torch.empty((m, 336), device="xpu", dtype=torch.float16)
    inject_pad[:, 320:324] = inj
    inject_view = inject_pad[:, 320:324]
    combined = torch.empty_like(x)
    normed = torch.empty_like(x)
    low = torch.empty((m, 336), device="xpu", dtype=torch.float16)
    mixed = torch.empty_like(block)
    hc.combine_mix(
        x,
        block,
        inject_view,
        gamma,
        down_w,
        up_w,
        combined,
        normed,
        low,
        mixed,
        1e-6,
    )
    expected_combined = ref_combine(x, block, inj)
    expected_norm = ref_norm(expected_combined, gamma)
    expected_low = ref_down(expected_norm, down_w)
    expected_mix = ref_up_gate(expected_low[:, :320], up_w, expected_norm)
    torch.testing.assert_close(combined, expected_combined, atol=1e-3, rtol=0)
    torch.testing.assert_close(normed, expected_norm, atol=3e-3, rtol=2e-3)
    torch.testing.assert_close(low, expected_low, atol=3e-3, rtol=2e-3)
    torch.testing.assert_close(mixed, expected_mix, atol=4e-3, rtol=2e-3)


def test_preflight_and_async_stream(hc):
    x, block, inj, gamma = inputs(2)
    output = torch.full_like(x, 7)
    with pytest.raises(RuntimeError):
        hc.combine_norm(x, block, inj, gamma, output, output, 1e-6)
    assert torch.all(output == 7)
    with pytest.raises(RuntimeError):
        hc.grouped_norm(x, gamma, output, 0.0)
    assert torch.all(output == 7)

    stream = torch.xpu.Stream()
    expected = ref_combine(x, block, inj)
    with torch.xpu.stream(stream):
        temporary = x * 1.0
        combined = torch.empty_like(x)
        hc.combine(temporary, block, inj, combined)
        del temporary
        for _ in range(8):
            _ = torch.empty((4096, 4096), dtype=torch.float16, device="xpu")
    stream.synchronize()
    torch.testing.assert_close(combined, expected, atol=1e-3, rtol=0)


def test_esimd_grouped_and_combine_norm_m1(hc):
    # The installed DSO may be stale for throughput, but these two scalar
    # contracts are stable and provide an independent native oracle.
    import custom_esimd_kernels_vllm.custom_esimd_kernels  # noqa: F401

    x, block, inj, gamma = inputs(1)
    ours = torch.empty_like(x)
    theirs = torch.empty_like(x)
    hc.grouped_norm(x, gamma, ours, 1e-6)
    torch.ops.custom_esimd_kernels_vllm.hc_grouped_norm_v1(
        x, gamma, theirs, 1e-6)
    torch.testing.assert_close(ours, theirs, atol=3e-3, rtol=2e-3)
    combined = torch.empty_like(x)
    normed = torch.empty_like(x)
    expected_combined = torch.empty_like(x)
    expected_normed = torch.empty_like(x)
    hc.combine_norm(x, block, inj, gamma, combined, normed, 1e-6)
    torch.ops.custom_esimd_kernels_vllm.hc_combine_norm_v1(
        x, block, inj, gamma, expected_combined, expected_normed, 1e-6)
    torch.testing.assert_close(combined, expected_combined, atol=1e-3, rtol=0)
    torch.testing.assert_close(normed, expected_normed, atol=3e-3, rtol=2e-3)


@pytest.mark.parametrize("m", [1, 4])
def test_esimd_projection_and_unaligned_view(hc, m):
    import custom_esimd_kernels_vllm.custom_esimd_kernels  # noqa: F401

    previous = torch.ops.custom_esimd_kernels_vllm
    torch.manual_seed(900 + m)
    x = torch.randn((m, 10240), device="xpu", dtype=torch.float16) * 0.2
    down_w = torch.randn((336, 10240), device="xpu", dtype=torch.float16) / 32
    up_w = torch.randn((10240, 320), device="xpu", dtype=torch.float16) / 20
    ours_down = torch.empty((m, 336), device="xpu", dtype=torch.float16)
    theirs_down = torch.empty_like(ours_down)
    hc.down(x, down_w, ours_down)
    old_down = (previous.esimd_hc_down_fp16_out
                if m == 1 else previous.esimd_hc_down_fp16_multi_m_out_v1)
    old_down(x, down_w, theirs_down)
    torch.testing.assert_close(ours_down, theirs_down, atol=3e-3, rtol=2e-3)

    odd_x = torch.empty((x.numel() + 1, ), device="xpu",
                        dtype=torch.float16)[1:].view(m, 10240)
    odd_w = torch.empty((down_w.numel() + 1, ),
                        device="xpu",
                        dtype=torch.float16)[1:].view(336, 10240)
    odd_x.copy_(x)
    odd_w.copy_(down_w)
    odd_down = torch.empty_like(ours_down)
    hc.down(odd_x, odd_w, odd_down)
    torch.testing.assert_close(odd_down, ours_down, atol=3e-3, rtol=2e-3)

    normed = torch.randn_like(x) * 0.2
    ours = torch.empty((m, 2560), device="xpu", dtype=torch.float16)
    theirs = torch.empty_like(ours)
    hc.up_gate_mix(ours_down[:, :320], up_w, normed, ours)
    old_up = (previous.esimd_hc_up_gate_mix_m1_v1
              if m == 1 else previous.esimd_hc_up_gate_mix_multi_m_v1)
    old_up(theirs_down[:, :320], up_w, normed, theirs)
    torch.testing.assert_close(ours, theirs, atol=3e-3, rtol=2e-3)

    # Force the 2-byte-aligned fallback; neither a block-read nor an
    # erroneously rounded-up address may be used for this live view.
    padded = torch.empty((m, 338), device="xpu", dtype=torch.float16)
    padded[:, 1:321] = ours_down[:, :320]
    odd_low = padded[:, 1:321]
    odd_result = torch.empty_like(ours)
    hc.up_gate_mix(odd_low, up_w, normed, odd_result)
    torch.testing.assert_close(odd_result, ours, atol=3e-3, rtol=2e-3)


@pytest.mark.parametrize("pair", ["0", "1"])
def test_chain_preflight_before_submit_and_two_async_streams(
        hc, monkeypatch, pair):
    monkeypatch.setenv("QWEN38_HC_M1_UP_GATE_PAIR", pair)
    x, block, inj, gamma = inputs(1)
    down_w = torch.randn((336, 10240), device="xpu", dtype=torch.float16) / 32
    up_w = torch.randn((10240, 320), device="xpu", dtype=torch.float16) / 20
    combined = torch.full_like(x, 7)
    normed = torch.full_like(x, 7)
    low = torch.full((1, 336), 7, device="xpu", dtype=torch.float16)
    mixed = torch.full_like(block, 7)
    with pytest.raises(RuntimeError):
        hc.combine_mix(
            x,
            block,
            inj,
            gamma,
            down_w,
            up_w[:, 1:],
            combined,
            normed,
            low,
            mixed,
            1e-6,
        )
    assert torch.all(combined == 7)
    assert torch.all(normed == 7)
    assert torch.all(low == 7)
    assert torch.all(mixed == 7)

    torch.xpu.synchronize()
    streams = [torch.xpu.Stream(), torch.xpu.Stream()]
    outputs = []
    for stream in streams:
        with torch.xpu.stream(stream):
            source = x * 1.0
            c = torch.empty_like(x)
            n = torch.empty_like(x)
            d = torch.empty_like(low)
            o = torch.empty_like(block)
            hc.combine_mix(
                source,
                block,
                inj,
                gamma,
                down_w,
                up_w,
                c,
                n,
                d,
                o,
                1e-6,
            )
            outputs.append((c, n, d, o))
            del source
    for stream in streams:
        stream.synchronize()
    expected_combined = ref_combine(x, block, inj)
    expected_norm = ref_norm(expected_combined, gamma)
    expected_low = ref_down(expected_norm, down_w)
    expected_out = ref_up_gate(expected_low[:, :320], up_w, expected_norm)
    for c, n, d, o in outputs:
        torch.testing.assert_close(c, expected_combined, atol=1e-3, rtol=0)
        torch.testing.assert_close(n, expected_norm, atol=3e-3, rtol=2e-3)
        torch.testing.assert_close(d, expected_low, atol=3e-3, rtol=2e-3)
        torch.testing.assert_close(o, expected_out, atol=4e-3, rtol=2e-3)


@pytest.mark.parametrize("m", [1, 4, 8])
def test_final_esimd_header_projection(hc, final_reference, m):
    torch.manual_seed(917 + m)
    x = torch.randn((m, 10240), dtype=torch.float16, device="xpu") * 0.2
    down_weight = (torch.randn(
        (336, 10240), dtype=torch.float16, device="xpu") / 32)
    up_weight = (torch.randn(
        (10240, 320), dtype=torch.float16, device="xpu") / 20)
    normed = torch.randn_like(x) * 0.2
    ours_down = torch.empty((m, 336), dtype=torch.float16, device="xpu")
    theirs_down = torch.empty_like(ours_down)
    hc.down(x, down_weight, ours_down)
    final_reference.down(x, down_weight, theirs_down)
    torch.testing.assert_close(ours_down, theirs_down, atol=3e-3, rtol=2e-3)
    ours_mix = torch.empty((m, 2560), dtype=torch.float16, device="xpu")
    theirs_mix = torch.empty_like(ours_mix)
    hc.up_gate_mix(ours_down[:, :320], up_weight, normed, ours_mix)
    final_reference.up_gate_mix(theirs_down[:, :320], up_weight, normed,
                                theirs_mix)
    torch.testing.assert_close(ours_mix, theirs_mix, atol=3e-3, rtol=2e-3)


def test_final_esimd_source_combine_norm_m1(hc, final_reference):
    x, block, inj, gamma = inputs(1)
    combined = torch.empty_like(x)
    normed = torch.empty_like(x)
    final_combined = torch.empty_like(x)
    final_normed = torch.empty_like(x)
    hc.combine_norm(x, block, inj, gamma, combined, normed, 1e-6)
    final_reference.combine_norm(x, block, inj, gamma, final_combined,
                                 final_normed, 1e-6)
    torch.testing.assert_close(combined, final_combined, atol=1e-3, rtol=0)
    torch.testing.assert_close(normed, final_normed, atol=3e-3, rtol=2e-3)


def test_combine_norm_has_real_fp16_rounding(hc, final_reference):
    # Injection=0 makes the combine scale exactly 1. The rounded and
    # unrounded RMS inputs then have a substantial, reproducible difference.
    oracle_mismatch = 0
    rounded_mismatch = 0
    unrounded_mismatch = 0
    for seed in range(3):
        torch.manual_seed(seed)
        x = torch.randn((1, 10240), device="xpu", dtype=torch.float16) * 0.2
        block = torch.randn((1, 2560), device="xpu",
                            dtype=torch.float16) * 0.2
        injection = torch.zeros((1, 4), device="xpu", dtype=torch.float16)
        gamma = torch.zeros((10240, ), device="xpu", dtype=torch.float16)
        combined = torch.empty_like(x)
        normed = torch.empty_like(x)
        oracle_combined = torch.empty_like(x)
        oracle_normed = torch.empty_like(x)
        hc.combine_norm(x, block, injection, gamma, combined, normed, 1e-6)
        if final_reference is not None:
            final_reference.combine_norm(x, block, injection, gamma,
                                         oracle_combined, oracle_normed, 1e-6)
            assert torch.equal(combined, oracle_combined)
        else:
            assert torch.equal(combined, ref_combine(x, block, injection))

        unrounded = x.float().reshape(1, 4, 2560) + block.float().reshape(
            1, 1, 2560)

        def rms(values):
            return (values * torch.rsqrt(
                values.square().mean(-1, keepdim=True) + 1e-6)).reshape(
                    1, 10240).half()

        rounded = rms(unrounded.half().float())
        not_rounded = rms(unrounded)
        if final_reference is not None:
            oracle_mismatch += int((normed != oracle_normed).sum())
        rounded_mismatch += int((normed != rounded).sum())
        unrounded_mismatch += int((normed != not_rounded).sum())
    assert oracle_mismatch <= 16
    assert rounded_mismatch <= 16
    assert unrounded_mismatch >= 1000


@pytest.mark.parametrize("pair", ["0", "1"])
def test_up_gate_has_real_fp16_rounding(hc, final_reference, monkeypatch, pair):
    monkeypatch.setenv("QWEN38_HC_M1_UP_GATE_PAIR", pair)
    # A one-term dot product isolates the gate's FP32 -> FP16 -> FP32 edge.
    # An optimized-away cast changes thousands of outputs while still passing
    # the usual loose allclose check.
    oracle_mismatch = 0
    rounded_mismatch = 0
    unrounded_mismatch = 0
    for seed in range(5):
        torch.manual_seed(seed)
        low = torch.zeros((1, 320), device="xpu", dtype=torch.float16)
        low[0, 0] = torch.tensor(0.9, dtype=torch.float16)
        weight = torch.zeros((10240, 320), device="xpu", dtype=torch.float16)
        weight[:, 0] = torch.randn(10240, device="xpu", dtype=torch.float16)
        normed = torch.randn((1, 10240), device="xpu", dtype=torch.float16)
        ours = torch.empty((1, 2560), device="xpu", dtype=torch.float16)
        reference = torch.empty_like(ours)
        hc.up_gate_mix(low, weight, normed, ours)
        if final_reference is not None:
            final_reference.up_gate_mix(low, weight, normed, reference)
        linear = low[0, 0].float() * weight[:, 0].float()

        def mix(gate, normed=normed):
            return (normed.float().reshape(4, 2560) *
                    torch.sigmoid(gate.reshape(4, 2560))).sum(0).mul(
                        0.25).half()

        rounded = mix(linear.half().float())
        unrounded = mix(linear)
        if final_reference is not None:
            oracle_mismatch += int(
                (ours.flatten() != reference.flatten()).sum())
        rounded_mismatch += int((ours.flatten() != rounded).sum())
        unrounded_mismatch += int((ours.flatten() != unrounded).sum())
    assert oracle_mismatch <= 16
    assert rounded_mismatch <= 16
    assert unrounded_mismatch >= 1000


@pytest.mark.parametrize("m", [1, 2, 8])
@pytest.mark.parametrize("odd", [False, True])
def test_up_gate_pair_runtime_bitwise_ab(hc, monkeypatch, m, odd):
    torch.manual_seed(6109 + m)
    # 保留 336 的物理步长，并覆盖低秩输入与权重仅两字节对齐的回退。
    padded = torch.randn((m, 336), device="xpu", dtype=torch.float16) * 0.2
    low = padded[:, 1:321] if odd else padded[:, :320]
    weight_storage = torch.randn((10240 * 320 + int(odd), ),
                                 device="xpu",
                                 dtype=torch.float16) / 20
    weight = weight_storage[int(odd):].view(10240, 320)
    normed = torch.randn((m, 10240), device="xpu", dtype=torch.float16) * 0.2
    baseline = torch.empty((m, 2560), device="xpu", dtype=torch.float16)
    ordinary = torch.empty_like(normed)
    monkeypatch.delenv("QWEN38_HC_M1_UP_GATE_PAIR", raising=False)
    hc.up_gate_mix(low, weight, normed, baseline)
    hc.up(low, weight, ordinary)
    torch.testing.assert_close(baseline,
                               ref_up_gate(low, weight, normed),
                               atol=3e-3,
                               rtol=2e-3)
    # 同一进程、同一动态库开启后再次关闭；M>1 和普通 up 不应改路。
    for mode in ("1", "0", "true", "10", ""):
        monkeypatch.setenv("QWEN38_HC_M1_UP_GATE_PAIR", mode)
        actual = torch.empty_like(baseline)
        hc.up_gate_mix(low, weight, normed, actual)
        assert torch.equal(actual.view(torch.int16), baseline.view(torch.int16))
        actual_up = torch.empty_like(ordinary)
        hc.up(low, weight, actual_up)
        assert torch.equal(actual_up.view(torch.int16),
                           ordinary.view(torch.int16))


def test_up_gate_pair_graph_bitwise_ab(hc, monkeypatch):
    torch.manual_seed(6110)
    low = torch.randn((1, 336), device="xpu", dtype=torch.float16)[:, :320]
    weight = torch.randn((10240, 320), device="xpu", dtype=torch.float16) / 20
    normed = torch.randn((1, 10240), device="xpu", dtype=torch.float16) * 0.2
    outputs = [torch.empty((1, 2560), device="xpu", dtype=torch.float16)
               for _ in range(2)]
    graphs = []
    for mode, output in zip(("0", "1"), outputs):
        monkeypatch.setenv("QWEN38_HC_M1_UP_GATE_PAIR", mode)
        hc.up_gate_mix(low, weight, normed, output)
        torch.xpu.synchronize()
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            hc.up_gate_mix(low, weight, normed, output)
        graphs.append(graph)
    # getenv 在提交或捕获时生效；回放已有图不重新分派，A/B 需分别捕获。
    monkeypatch.setenv("QWEN38_HC_M1_UP_GATE_PAIR", "0")
    for _ in range(2):
        low.mul_(0.5)
        for graph in graphs:
            graph.replay()
        torch.xpu.synchronize()
        assert torch.equal(outputs[0].view(torch.int16),
                           outputs[1].view(torch.int16))
        eager = torch.empty_like(outputs[0])
        hc.up_gate_mix(low, weight, normed, eager)
        assert torch.equal(outputs[0].view(torch.int16),
                           eager.view(torch.int16))


@pytest.mark.parametrize("m", [1, 4, 8])
def test_pair_chain_workspace_and_full_down_contract(hc, monkeypatch, m):
    x, block, inj, gamma = inputs(m)
    down_w = torch.randn((336, 10240), device="xpu", dtype=torch.float16) / 32
    up_w = torch.randn((10240, 320), device="xpu", dtype=torch.float16) / 20
    cls = (torch.classes.qwen38_hc_sycl.HCWorkspace
           if m == 1 else torch.classes.qwen38_hc_sycl.HCMultiMWorkspaceV1)
    owner = cls()
    baseline = None
    for mode in ("0", "1", "0"):
        monkeypatch.setenv("QWEN38_HC_M1_UP_GATE_PAIR", mode)
        combined = torch.empty_like(x)
        normed = torch.empty_like(x)
        low = torch.full((m, 336), 7, device="xpu", dtype=torch.float16)
        mixed = torch.empty_like(block)
        hc.combine_mix(x, block, inj, gamma, down_w, up_w, combined, normed,
                       low, mixed, 1e-6)
        standalone = torch.empty_like(low)
        hc.down(normed, down_w, standalone)
        assert torch.equal(low.view(torch.int16), standalone.view(torch.int16))
        project_low = torch.empty_like(low)
        project_mix = torch.empty_like(mixed)
        hc.project_mix(normed, down_w, up_w, project_low, project_mix)
        assert torch.equal(project_low.view(torch.int16), low.view(torch.int16))
        assert torch.equal(project_mix.view(torch.int16),
                           mixed.view(torch.int16))

        result = owner.run(x, block, inj, gamma, down_w, up_w, 1e-6)
        assert torch.equal(result[0].view(torch.int16),
                           combined.view(torch.int16))
        assert torch.equal(result[1].view(torch.int16), mixed.view(torch.int16))
        next_inj = result[2]
        assert next_inj.stride() == (336, 1)
        # 返回的 injection 视图暴露同一 storage，不能把末尾十二行当不可观察。
        full_down = next_inj.as_strided(
            (m, 336), (336, 1), next_inj.storage_offset() - 320)
        assert torch.equal(full_down.view(torch.int16), low.view(torch.int16))
        mix_only = owner.try_mix(normed, down_w, up_w)
        assert mix_only is not None
        assert torch.equal(mix_only[0].view(torch.int16),
                           mixed.view(torch.int16))
        mix_down = mix_only[1].as_strided(
            (m, 336), (336, 1), mix_only[1].storage_offset() - 320)
        assert torch.equal(mix_down.view(torch.int16), low.view(torch.int16))

        current = (combined, normed, low, mixed)
        if baseline is None:
            baseline = tuple(t.clone() for t in current)
            torch.testing.assert_close(low,
                                       ref_down(normed, down_w),
                                       atol=3e-3,
                                       rtol=2e-3)
        else:
            for actual, expected in zip(current, baseline):
                assert torch.equal(actual.view(torch.int16),
                                   expected.view(torch.int16))


@pytest.mark.parametrize("m", range(1, 9))
def test_workspace_public_abi_live_weights_and_try_mix(hc, m):
    x, block, inj, gamma = inputs(m)
    down_w = torch.randn((336, 10240), device="xpu", dtype=torch.float16) / 32
    up_w = torch.randn((10240, 320), device="xpu", dtype=torch.float16) / 20
    cls = (torch.classes.qwen38_hc_sycl.HCWorkspace
           if m == 1 else torch.classes.qwen38_hc_sycl.HCMultiMWorkspaceV1)
    owner = cls()
    padded = torch.empty((m, 336), device="xpu", dtype=torch.float16)
    padded[:, 320:324] = inj
    injection = inj if m == 1 else padded[:, 320:324]
    result = owner.try_run(x, block, injection, gamma, down_w, up_w, 1e-6)
    assert result is not None and len(result) == 3
    combined, mixed, next_inj = result
    expected_combined = ref_combine(x, block, inj)
    expected_norm = ref_norm(expected_combined, gamma)
    expected_down = ref_down(expected_norm, down_w)
    torch.testing.assert_close(combined, expected_combined, atol=1e-3, rtol=0)
    torch.testing.assert_close(mixed,
                               ref_up_gate(expected_down[:, :320], up_w,
                                           expected_norm),
                               atol=4e-3,
                               rtol=2e-3)
    torch.testing.assert_close(next_inj,
                               expected_down[:, 320:324],
                               atol=3e-3,
                               rtol=2e-3)
    assert next_inj.stride(0) == 336

    # No captured weight pointer: changing a live Parameter-like tensor is
    # reflected at the next invocation, including the up projection.
    new_up_w = up_w * 0.5
    result2 = owner.run(x, block, injection, gamma, down_w, new_up_w, 1e-6)
    expected_mix_new = ref_up_gate(expected_down[:, :320], new_up_w,
                                   expected_norm)
    torch.testing.assert_close(result2[1],
                               expected_mix_new,
                               atol=4e-3,
                               rtol=2e-3)

    mix_only = owner.try_mix(expected_norm, down_w, new_up_w)
    assert mix_only is not None and len(mix_only) == 2
    torch.testing.assert_close(mix_only[0],
                               expected_mix_new,
                               atol=4e-3,
                               rtol=2e-3)
    torch.testing.assert_close(mix_only[1],
                               expected_down[:, 320:324],
                               atol=3e-3,
                               rtol=2e-3)


def test_workspace_optional_miss_and_pre_submit_guard(hc):
    x, block, inj, gamma = inputs(1)
    down_w = torch.randn((336, 10240), device="xpu", dtype=torch.float16) / 32
    up_w = torch.randn((10240, 320), device="xpu", dtype=torch.float16) / 20
    owner = torch.classes.qwen38_hc_sycl.HCWorkspace()
    assert owner.try_run(x, block, inj, gamma, down_w, up_w[:, 1:],
                         1e-6) is None
    assert owner.try_run(x, block, inj, gamma, down_w, up_w,
                         float("nan")) is None
    assert owner.try_run(x, block, inj, gamma, down_w, up_w,
                         float("inf")) is None
    assert owner.try_mix(x, down_w, up_w[:, 1:]) is None
    # A returned output can become a live input; owner must refresh scratch,
    # not overwrite it or bypass alias checks on a cached fast path.
    first = owner.try_run(x, block, inj, gamma, down_w, up_w, 1e-6)
    prior_combined = first[0].clone()
    second = owner.try_run(first[0], block, inj, gamma, down_w, up_w, 1e-6)
    torch.testing.assert_close(first[0], prior_combined, atol=0, rtol=0)
    assert second[0].data_ptr() != first[0].data_ptr()

    # A cached output mutated by its borrower is rejected before the first
    # launch; previous mixed output must remain untouched.
    second[1].fill_(7)
    second[0].resize_(1, 10239)
    with pytest.raises(RuntimeError, match="scratch metadata"):
        owner.try_run(x, block, inj, gamma, down_w, up_w, 1e-6)
    assert torch.all(second[1] == 7)

    low = torch.full((1, 336), 7, device="xpu", dtype=torch.float16)
    mixed = torch.full_like(block, 7)
    with pytest.raises(RuntimeError):
        hc.project_mix(x, down_w, up_w[:, 1:], low, mixed)
    assert torch.all(low == 7)
    assert torch.all(mixed == 7)


def test_workspace_async_stream_partition(hc):
    x, block, inj, gamma = inputs(4)
    down_w = torch.randn((336, 10240), device="xpu", dtype=torch.float16) / 32
    up_w = torch.randn((10240, 320), device="xpu", dtype=torch.float16) / 20
    owner = torch.classes.qwen38_hc_sycl.HCMultiMWorkspaceV1()
    torch.xpu.synchronize()
    streams = [torch.xpu.Stream(), torch.xpu.Stream()]
    results = []
    for stream in streams:
        with torch.xpu.stream(stream):
            transient = x * 1.0
            result = owner.try_run(transient, block, inj, gamma, down_w, up_w,
                                   1e-6)
            results.append(result)
            del transient
    for stream in streams:
        stream.synchronize()
    assert results[0][0].data_ptr() != results[1][0].data_ptr()
    expected = ref_combine(x, block, inj)
    for result in results:
        torch.testing.assert_close(result[0], expected, atol=1e-3, rtol=0)


@pytest.mark.parametrize("reader", ["injection", "lowrank"])
def test_padded_alias_rejected_before_submit(hc, reader):
    output = torch.full((2, 10240), 7, device="xpu", dtype=torch.float16)
    if reader == "injection":
        hidden = torch.zeros_like(output)
        block = torch.zeros((2, 2560), device="xpu", dtype=torch.float16)
        injection = output[:, :4]  # ATen overlap status is TooHard.
        with pytest.raises(RuntimeError, match="overlap"):
            hc.combine(hidden, block, injection, output)
    else:
        lowrank = output[:, :320]
        weight = torch.empty((10240, 320), device="xpu", dtype=torch.float16)
        with pytest.raises(RuntimeError, match="overlap"):
            hc.up(lowrank, weight, output)
    assert torch.all(output == 7).item()
