# SPDX-License-Identifier: Apache-2.0
"""Independent math oracle and guarded TP4 GDN projection integration tests.

The final ESIMD package is a test-only oracle; production never imports it.
GPU tests are deferred until the TP4 E2E run releases the device.
"""

import gc
import os
from pathlib import Path

import pytest
import torch

EPS = 1e-6
FINAL_ESIMD = (
    Path(__file__).resolve().parents[2]
    / "qwen38_sycl_20261004/esimd_final_reference"
    / "custom_esimd_kernels_vllm"
    / "custom_esimd_kernels.cpython-312-x86_64-linux-gnu.so"
)


def _normalize(x, z, norm, eps=EPS):
    """FP32 RMS norm and sigmoid; the projection chooses the boundary."""
    xf, zf, wf = x.float(), z.float(), norm.float()
    inv = torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + eps)
    return xf * inv * wf * torch.sigmoid(zf)


def _golden(x, z, norm, packed, scale, *, half_boundary=False):
    # This is independent of both native and ESIMD kernels. Decode nibbles
    # explicitly and use a CPU matrix product as the arithmetic oracle.
    normalized = _normalize(x.cpu(), z.cpu(), norm.cpu())
    if half_boundary:
        normalized = normalized.half().float()
    w = packed.cpu()
    q = torch.empty((w.shape[0], w.shape[1] * 2), dtype=torch.float32)
    q[:, 0::2] = (w & 15).float() - 8
    q[:, 1::2] = (w >> 4).float() - 8
    dequant = q * scale.cpu().float().repeat_interleave(128, dim=1)
    return (normalized.reshape(1, -1) @ dequant.T).half()


def _case(hv=12, n=2560, seed=419):
    gen = torch.Generator().manual_seed(seed)
    x = (torch.randn(hv, 128, generator=gen) * 0.7).half()
    z = (torch.randn(hv, 128, generator=gen) * 0.6).half()
    norm = (torch.randn(128, generator=gen) * 0.5 + 1).half()
    packed = torch.randint(256, (n, hv * 64), generator=gen).byte()
    scale = (torch.rand(n, hv, generator=gen) * 0.03).half()
    return x, z, norm, packed, scale


def _discriminator():
    """One q9 per row; select FP32-vs-half threshold crossings with margin."""
    x, z, norm, _, _ = _case(seed=733)
    normalized = _normalize(x, z, norm).reshape(-1)
    scales = torch.tensor([0.63, 0.75, 1.3], dtype=torch.half)
    exact = normalized[:, None] * scales.float()[None, :]
    rounded_norm = normalized.half().float()[:, None] * scales.float()[None, :]
    fp32_half = exact.half()
    extra_half = rounded_norm.half()
    # Distinct half outputs put a rounding threshold between the two values.
    midpoint = (fp32_half.float() + extra_half.float()) * 0.5
    margin = torch.minimum((exact - midpoint).abs(),
                           (rounded_norm - midpoint).abs())
    margin[fp32_half == extra_half] = 0
    best_margin, best_scale = margin.max(dim=1)
    positions = torch.nonzero(best_margin > 1e-5).flatten()
    assert positions.numel() >= 48, "fixture needs 48 robust FP16 crossings"
    positions = positions[:96]  # Unique head/column for each selected row.
    rows = torch.arange(positions.numel())
    packed = torch.full((2560, 768), 0x88, dtype=torch.uint8)
    scale = torch.full((2560, 12), 0.75, dtype=torch.float16)
    packed[rows, positions // 2] = torch.where(
        positions % 2 == 0, 0x89, 0x98
    ).to(torch.uint8)
    selected_scale = scales[best_scale[positions]]
    scale[rows, positions // 128] = selected_scale
    fp32_out = torch.zeros((1, 2560), dtype=torch.float16)
    half_out = torch.zeros_like(fp32_out)
    fp32_out[0, rows] = (normalized[positions] * selected_scale.float()).half()
    half_out[0, rows] = (
        normalized[positions].half().float() * selected_scale.float()
    ).half()
    assert torch.all(fp32_out[0, rows] != half_out[0, rows])
    return (x, z, norm, packed, scale), fp32_out, half_out, rows


def test_fp32_norm_boundary_discriminates_half_on_cpu():
    case, fp32, extra_half, rows = _discriminator()
    assert rows.numel() >= 48
    assert torch.all(fp32[0, rows] != extra_half[0, rows])
    assert case[3].shape == (2560, 768)


@pytest.fixture(scope="module")
def native_op():
    path = os.environ.get("QWEN38_SYCL_LIBRARY") or os.environ.get(
        "QWEN38_GDN_PROJECTION_LIBRARY"
    )
    if not path:
        pytest.skip("set QWEN38_SYCL_LIBRARY or QWEN38_GDN_PROJECTION_LIBRARY")
    torch.ops.load_library(path)
    try:
        return torch.ops._qwen38_C.gdn_norm_int4_sycl
    except AttributeError:
        return torch.ops.qwen38_gdn_projection_test.norm_int4


def _call(op, case, sigmoid=True):
    tensors = [tensor.to("xpu") for tensor in case]
    output = torch.empty(1, tensors[3].shape[0], device="xpu", dtype=torch.half)
    op(*tensors, output, tensors[0].shape[0], 128, EPS, sigmoid)
    return output.cpu()


def test_tp4_default_output_tile_hit(native_op, monkeypatch):
    monkeypatch.delenv("VLLM_XPU_QWEN38_GDN_OUTPUT_TILE", raising=False)
    case = tuple(t.to("xpu") for t in _case())
    output = torch.empty(1, 2560, dtype=torch.half, device="xpu")
    native_op(*case, output, 12, 128, EPS, True)
    torch.xpu.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.XPU]
    ) as prof:
        native_op(*case, output, 12, 128, EPS, True)
    assert any("GdnNormInt4FusedKernel<32>" in event.name
               for event in prof.events())


def test_tp4_fused_independent_golden(native_op):
    case = _case()
    actual = _call(native_op, case)
    torch.testing.assert_close(actual, _golden(*case), atol=0.025, rtol=0.002)


def test_tp4_fused_rejects_half_boundary(native_op):
    case, fp32_expected, half_expected, rows = _discriminator()
    actual = _call(native_op, case)
    good = (actual[0, rows] == fp32_expected[0, rows]).sum().item()
    bad = (actual[0, rows] == half_expected[0, rows]).sum().item()
    assert good >= int(0.9 * rows.numel())
    assert good >= bad + int(0.75 * rows.numel())
    assert torch.count_nonzero(actual[0, rows.numel():]) == 0


@pytest.mark.parametrize("tile", ["32"])
@pytest.mark.parametrize("seed", [419, 733, 20261009])
def test_tp4_output_tiles_preserve_bitwise_projection(
    native_op, monkeypatch, tile, seed
):
    # 只共享FP32 norm，不改变每个输出的half边界和pairwise归约。
    case = _case(seed=seed)
    monkeypatch.setenv("VLLM_XPU_QWEN38_GDN_OUTPUT_TILE", "16")
    expected = _call(native_op, case)
    monkeypatch.setenv("VLLM_XPU_QWEN38_GDN_OUTPUT_TILE", tile)
    actual = _call(native_op, case)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("tile", ["32"])
def test_tp4_output_tiles_keep_fp32_norm_boundary(native_op, monkeypatch, tile):
    case, fp32_expected, half_expected, rows = _discriminator()
    monkeypatch.setenv("VLLM_XPU_QWEN38_GDN_OUTPUT_TILE", "16")
    expected = _call(native_op, case)
    monkeypatch.setenv("VLLM_XPU_QWEN38_GDN_OUTPUT_TILE", tile)
    actual = _call(native_op, case)
    assert torch.equal(actual, expected)
    good = (actual[0, rows] == fp32_expected[0, rows]).sum().item()
    bad = (actual[0, rows] == half_expected[0, rows]).sum().item()
    assert good >= int(0.9 * rows.numel())
    assert good >= bad + int(0.75 * rows.numel())


def test_tp4_final_esimd_oracle_only(native_op):
    if not FINAL_ESIMD.is_file():
        pytest.skip("final ESIMD reference package is unavailable")
    torch.ops.load_library(str(FINAL_ESIMD))
    case = _case(seed=607)
    tensors = [tensor.to("xpu") for tensor in case]
    native = torch.empty(1, 2560, device="xpu", dtype=torch.half)
    esimd = torch.empty_like(native)
    native_op(*tensors, native, 12, 128, EPS, True)
    torch.ops.custom_esimd_kernels_vllm.esimd_norm_gemv_int4_sigmoid(
        *tensors[:3], tensors[3].view(torch.int32), tensors[4],
        esimd, 12, 128, EPS
    )
    expected = _golden(*case)
    torch.testing.assert_close(native.cpu(), expected, atol=0.025, rtol=0.002)
    torch.testing.assert_close(
        native.cpu(), esimd.cpu(), atol=0.025, rtol=0.002
    )


def test_strong_discriminator_matches_final_esimd(native_op):
    if not FINAL_ESIMD.is_file():
        pytest.skip("final ESIMD reference package is unavailable")
    torch.ops.load_library(str(FINAL_ESIMD))
    case, fp32, extra_half, rows = _discriminator()
    tensors = [tensor.to("xpu") for tensor in case]
    native = torch.empty(1, 2560, device="xpu", dtype=torch.half)
    esimd = torch.empty_like(native)
    native_op(*tensors, native, 12, 128, EPS, True)
    torch.ops.custom_esimd_kernels_vllm.esimd_norm_gemv_int4_sigmoid(
        *tensors[:3], tensors[3].view(torch.int32), tensors[4],
        esimd, 12, 128, EPS
    )
    native, esimd = native.cpu(), esimd.cpu()
    for result in (native, esimd):
        good = (result[0, rows] == fp32[0, rows]).sum().item()
        bad = (result[0, rows] == extra_half[0, rows]).sum().item()
        assert good >= int(0.9 * rows.numel())
        assert good >= bad + int(0.75 * rows.numel())
    assert (native[0, rows] == esimd[0, rows]).sum().item() >= int(
        0.9 * rows.numel()
    )


@pytest.mark.parametrize("hv,n,sigmoid", [
    (12, 2560, False), (12, 256, True), (6, 2560, True),
])
def test_fallback_preserves_half_boundary(native_op, hv, n, sigmoid):
    case = _case(hv=hv, n=n)
    if not sigmoid:
        x, z, norm, packed, scale = case
        normalized = _normalize(x, z, norm) * z.float()
        q = torch.empty((n, hv * 128), dtype=torch.float32)
        q[:, 0::2] = (packed & 15).float() - 8
        q[:, 1::2] = (packed >> 4).float() - 8
        expected = (normalized.half().float().reshape(1, -1) @
                    (q * scale.float().repeat_interleave(128, 1)).T).half()
    else:
        expected = _golden(*case, half_boundary=True)
    actual = _call(native_op, case, sigmoid=sigmoid)
    torch.testing.assert_close(actual, expected, atol=0.025, rtol=0.002)


@pytest.mark.parametrize("invalid", ["scale", "alias", "epsilon"])
def test_preflight_error_submits_no_kernel(native_op, invalid):
    from torch.profiler import ProfilerActivity, profile

    case = [tensor.to("xpu") for tensor in _case()]
    output = torch.full((1, 2560), 7, device="xpu", dtype=torch.half)
    if invalid == "scale":
        case[4] = case[4][:, :-1].contiguous()
    if invalid == "alias":
        output = case[4].view(-1)[:2560].view(1, 2560)
    epsilon = float("inf") if invalid == "epsilon" else EPS
    torch.xpu.synchronize()
    activities = [ProfilerActivity.CPU, ProfilerActivity.XPU]
    with profile(activities=activities) as prof, pytest.raises(RuntimeError):
        native_op(*case, output, 12, 128, epsilon, True)
    if invalid != "alias":
        assert torch.equal(output.cpu(),
                           torch.full((1, 2560), 7, dtype=torch.half))
    assert not [
        e for e in prof.events()
        if e.device_type != torch.autograd.DeviceType.CPU
    ]


@pytest.mark.parametrize("hv,n,sigmoid,kernels", [
    (12, 2560, True, 1),
    (12, 2560, False, 2),
    (12, 256, True, 2),
    (6, 2560, True, 2),
])
def test_one_kernel_only_for_tp4_sigmoid(native_op, hv, n, sigmoid, kernels):
    from torch.profiler import ProfilerActivity, profile

    case = [tensor.to("xpu") for tensor in _case(hv=hv, n=n)]
    output = torch.empty(1, n, device="xpu", dtype=torch.half)
    torch.xpu.synchronize()
    activities = [ProfilerActivity.CPU, ProfilerActivity.XPU]
    with profile(activities=activities) as prof:
        native_op(*case, output, hv, 128, EPS, sigmoid)
        torch.xpu.synchronize()
    device_events = [e for e in prof.events()
                     if e.device_type != torch.autograd.DeviceType.CPU]
    assert len(device_events) == kernels, [e.name for e in device_events]


def test_misaligned_contiguous_view_uses_fallback(native_op):
    from torch.profiler import ProfilerActivity, profile

    x, z, norm, packed, scale = _case()
    x = x.to("xpu")
    backing = torch.empty(x.numel() + 1, device="xpu", dtype=torch.half)
    offset_x = backing[1:].view_as(x)
    offset_x.copy_(x)
    tensors = [offset_x, z.to("xpu"), norm.to("xpu"),
               packed.to("xpu"), scale.to("xpu")]
    output = torch.empty(1, 2560, device="xpu", dtype=torch.half)
    torch.xpu.synchronize()
    activities = [ProfilerActivity.CPU, ProfilerActivity.XPU]
    with profile(activities=activities) as prof:
        native_op(*tensors, output, 12, 128, EPS, True)
        torch.xpu.synchronize()
    events = [e for e in prof.events()
              if e.device_type != torch.autograd.DeviceType.CPU]
    assert len(events) == 2, [e.name for e in events]
    torch.testing.assert_close(
        output.cpu(), _golden(x.cpu(), z, norm, packed, scale,
                              half_boundary=True), atol=0.025, rtol=0.002
    )


def test_nondefault_stream_dropped_inputs(native_op):
    case = [tensor.to("xpu") for tensor in _case(seed=719)]
    output = torch.empty(1, 2560, device="xpu", dtype=torch.half)
    expected = _golden(*[tensor.cpu() for tensor in case])
    stream = torch.xpu.Stream()
    stream.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(stream):
        native_op(*case, output, 12, 128, EPS, True)
    del case
    gc.collect()
    junk = [torch.empty((2560, 768), device="xpu", dtype=torch.uint8)
            for _ in range(4)]
    stream.synchronize()
    torch.testing.assert_close(output.cpu(), expected, atol=0.025, rtol=0.002)
    del junk
