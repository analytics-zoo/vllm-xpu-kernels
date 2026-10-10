# SPDX-License-Identifier: Apache-2.0
"""Build-only checks of the unified Qwen3.8 MoE DSO and Python provider.

Run this file in a fresh process. Loading the standalone MoE test DSO into the
same process would register a second owner for the Torch operator namespace.
"""

import importlib
import importlib.util
import math
import os
import sys
import types
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="module")
def provider():
    if not torch.xpu.is_available():
        pytest.skip("XPU is unavailable")
    root = Path(__file__).resolve().parents[1]
    selected = os.environ.get("QWEN38_MOE_PROVIDER_DSO")
    candidates = ([Path(selected)] if selected else list(
        (root / "build/qwen38-sycl").glob("_qwen38_C.cpython-312-*.so")))
    assert len(candidates) == 1, f"expected one unified MoE DSO: {candidates}"
    schema = "qwen38_moe_sycl::moe_forward_compact160_out_v1"
    assert not torch._C._jit_get_schemas_for_operator(schema), (
        "run the provider test in a fresh process without another MoE DSO")
    package_name = "qwen38_moe_provider_buildonly"
    package = types.ModuleType(package_name)
    package.__path__ = [str(root / "vllm_xpu_kernels")]
    sys.modules[package_name] = package
    extension_name = f"{package_name}._qwen38_C"
    spec = importlib.util.spec_from_file_location(extension_name, candidates[0])
    assert spec is not None and spec.loader is not None
    extension = importlib.util.module_from_spec(spec)
    sys.modules[extension_name] = extension
    spec.loader.exec_module(extension)
    native = importlib.import_module(f"{package_name}.qwen38_moe")
    assert extension.__file__ == str(candidates[0])
    assert not any(name.startswith("custom_esimd_kernels_vllm")
                   for name in sys.modules)
    return native


@pytest.fixture(scope="module", params=(80, 160))
def case(provider, request):
    k = request.param
    device = "xpu"
    # Signed S4: both nibbles in 0x11 represent +1. The expert-specific scale
    # makes incorrect routing observable while retaining an analytic golden.
    w13 = torch.full((512, 2 * k, 1280), 0x11, dtype=torch.uint8,
                     device=device)
    expert_scale = (0.0003 + torch.arange(512, device=device) * 0.000001).half()
    s13 = expert_scale[:, None, None, None].expand(512, 2 * k, 20, 1)
    s13 = s13.reshape(512, 2 * k, 20).contiguous()
    w2 = torch.full((512, 2560, k // 2), 0x11, dtype=torch.uint8,
                    device=device)
    s2 = torch.full((512, 2560, (k + 127) // 128), 0.003,
                    dtype=torch.float16, device=device)
    shared_up = torch.full((2 * k, 2560), 0.001,
                           dtype=torch.float16, device=device)
    shared_down = torch.full((2560, k), 0.002,
                             dtype=torch.float16, device=device)
    shared_gate = torch.full((1, 2560), 0.001,
                             dtype=torch.float16, device=device)
    weights = [w13, s13, w2, s2, shared_up, shared_down, shared_gate]
    x = torch.full((17, 2560), 0.1, dtype=torch.float16, device=device)
    x[:, 0] = 1.0 + torch.arange(17, device=device, dtype=torch.float16) / 16
    router = torch.full((512, 1280), 0x88, dtype=torch.uint8, device=device)
    router[:10, 0] = 0x89
    router_scale = torch.ones((512, 20), dtype=torch.float16, device=device)
    router_scale[:10] = (torch.arange(1, 11, device=device,
                                     dtype=torch.float16) / 4)[:, None]
    logits = torch.zeros((17, 512), dtype=torch.float16, device=device)
    logits[:, :10] = x[:, :1] * router_scale[:10, :1].T
    return dict(k=k, x=x, router=router, router_scale=router_scale,
                logits=logits, weights=weights)


def _golden(case, rows, shared_down_scale=None):
    """Independent scalar full-MoE reference for the structured test weights."""
    k = case["k"]
    x = case["x"][:rows].cpu().float()
    logits = case["logits"][:rows].cpu().float()
    expert_scale = case["weights"][1][:, 0, 0].cpu().float()
    routed_down_scale = float(case["weights"][3][0, 0, 0])
    shared_up_scale = float(case["weights"][4][0, 0])
    shared_gate_scale = float(case["weights"][6][0, 0])
    if shared_down_scale is None:
        shared_down_scale = float(case["weights"][5][0, 0])
    expected = []
    for row in range(rows):
        ids = torch.topk(logits[row], 10).indices
        scores = torch.softmax(logits[row, ids], dim=0).half().float()
        total = float(x[row].sum())
        routed = 0.0
        for rank, expert in enumerate(ids):
            projection = total * float(expert_scale[expert])
            activated = torch.tensor(
                projection * projection / (1.0 + math.exp(-projection)),
                dtype=torch.float16).item()
            routed += float(scores[rank]) * activated * k * routed_down_scale
        shared_projection = total * shared_up_scale
        shared_activation = torch.tensor(
            shared_projection * shared_projection
            / (1.0 + math.exp(-shared_projection)),
            dtype=torch.float16).item()
        shared_gate = 1.0 / (1.0 + math.exp(-total * shared_gate_scale))
        expected.append(routed + shared_gate * shared_activation * k
                        * shared_down_scale)
    return torch.tensor(expected, dtype=torch.float16)[:, None].expand(
        rows, 2560).contiguous()


def _assert_golden(actual, case, rows, shared_down_scale=None):
    expected = _golden(case, rows, shared_down_scale)
    torch.testing.assert_close(actual.cpu(), expected, atol=0.004, rtol=0.035)


@pytest.mark.parametrize("rows", (1, 4, 17))
def test_public_out_matches_full_golden(provider, case, rows):
    k = case["k"]
    output = torch.full((rows, 2560), -7, dtype=torch.float16, device="xpu")
    ops = provider.compact80_ops() if k == 80 else provider.compact160_ops()
    operation = ops[0] if rows == 1 else ops[1]
    returned = operation(case["x"][:rows], case["logits"][:rows],
                         *case["weights"], output, 10, 1, 512)
    assert returned.data_ptr() == output.data_ptr()
    _assert_golden(output, case, rows)


@pytest.mark.parametrize("rows", range(1, 9))
@pytest.mark.parametrize("direct", (False, True))
def test_workspace_torchclass_and_direct(provider, case, rows, direct):
    if rows == 1:
        factory = (provider.get_qwen38_moe_m1_direct_workspace_class()
                   if direct else provider.get_qwen38_moe_m1_workspace_class())
        args = (case["x"][:rows], case["router"], case["router_scale"],
                case["weights"], case["k"])
    else:
        multi_class = torch.classes.qwen38_moe_sycl.Qwen38MultiWorkspaceV1
        factory = (provider.get_qwen38_moe_multi_direct_workspace_class()
                   if direct else multi_class)
        args = (case["x"][:rows], case["router"], case["router_scale"],
                case["weights"], case["k"], rows in (4, 8))
    output = factory().try_run(*args)
    assert output is not None
    _assert_golden(output, case, rows)


@pytest.mark.parametrize("rows", (1, 4))
def test_retained_output_and_live_weight_rebinding(provider, case, rows):
    factory = (provider.get_qwen38_moe_m1_direct_workspace_class() if rows == 1
               else provider.get_qwen38_moe_multi_direct_workspace_class())
    workspace = factory()
    args = [case["x"][:rows], case["router"], case["router_scale"],
            case["weights"], case["k"]]
    if rows > 1:
        args.append(False)
    first = workspace.try_run(*args)
    assert first is not None
    snapshot = first.cpu().clone()
    rebound_weights = list(case["weights"])
    rebound_weights[5] = torch.full_like(rebound_weights[5], 0.004)
    args[3] = rebound_weights
    second = workspace.try_run(*args)
    assert second is not None and first.data_ptr() != second.data_ptr()
    torch.testing.assert_close(first.cpu(), snapshot, atol=0, rtol=0)
    _assert_golden(second, case, rows,
                   shared_down_scale=float(rebound_weights[5][0, 0]))
    assert not torch.equal(first.cpu(), second.cpu())


def test_stream_scoped_workspace(provider, case):
    workspace = provider.get_qwen38_moe_multi_direct_workspace_class()()
    other = dict(case)
    other["x"] = case["x"][:4].clone()
    other["x"][:, 0] *= 0.5
    other["logits"] = torch.zeros_like(case["logits"][:4])
    other["logits"][:, :10] = (
        other["x"][:, :1] * case["router_scale"][:10, :1].T)
    torch.xpu.synchronize()  # All inputs are ready before concurrent streams.
    primary = torch.xpu.Stream()
    alternate = torch.xpu.Stream()
    with torch.xpu.stream(primary):
        first = workspace.try_run(case["x"][:4], case["router"],
                                  case["router_scale"], case["weights"],
                                  case["k"], True)
    with torch.xpu.stream(alternate):
        second = workspace.try_run(other["x"], case["router"],
                                   case["router_scale"], case["weights"],
                                   case["k"], True)
    torch.xpu.current_stream().wait_stream(primary)
    torch.xpu.current_stream().wait_stream(alternate)
    assert first is not None and second is not None
    assert first.data_ptr() != second.data_ptr()
    _assert_golden(first, case, 4)
    _assert_golden(second, other, 4)


def test_invalid_preflight_submits_no_xpu_kernel(provider, case):
    workspace = provider.get_qwen38_moe_multi_direct_workspace_class()()
    bad_x = torch.empty((4, 2559), dtype=torch.float16, device="xpu")
    bad_weights = list(case["weights"])
    bad_weights[3] = bad_weights[3][:, :, :0]
    output = case["x"][:4]  # Invalid public output alias, same valid shape.
    torch.xpu.synchronize()
    with torch.profiler.profile(activities=[
            torch.profiler.ProfilerActivity.XPU], acc_events=True) as profiler:
        assert workspace.try_run(bad_x, case["router"],
                                 case["router_scale"], case["weights"],
                                 case["k"], False) is None
        assert workspace.try_run(case["x"][:4], case["router"],
                                 case["router_scale"], bad_weights,
                                 case["k"], False) is None
        operation = (provider.compact80_ops()[1] if case["k"] == 80
                     else provider.compact160_ops()[1])
        with pytest.raises(RuntimeError, match="contract"):
            operation(case["x"][:4], case["logits"][:4],
                      *case["weights"], output, 10, 1, 512)
    assert not any(event.device_type == torch.autograd.DeviceType.XPU
                   for event in profiler.events())


@pytest.mark.parametrize("k", (80, 160))
def test_prefill_provider_signed_s4_tail(provider, k):
    x = torch.empty((4, k), dtype=torch.float16, device="xpu")
    x[:, ::2] = 1
    x[:, 1::2] = 2
    packed = torch.full((512, 2560, k // 2), 0xF1,
                        dtype=torch.uint8, device="xpu")
    packed[1] = 0x1F
    scales = torch.ones((512, 2560, (k + 127) // 128),
                        dtype=torch.float16, device="xpu")
    if k == 160:
        scales[:, :, 1] = 2
    counts = torch.zeros(512, dtype=torch.int32, device="xpu")
    counts[0], counts[1] = 3, 1
    operation = (provider.compact80_ops()[2] if k == 80
                 else provider.compact160_ops()[2])
    output = operation(x, packed, scales, counts)
    expected_magnitude = 40 if k == 80 else 64 + 2 * 16
    expected = torch.tensor([-expected_magnitude] * 3 + [expected_magnitude],
                            dtype=torch.float16)[:, None].expand(4, 2560)
    torch.testing.assert_close(output.cpu(), expected, atol=0, rtol=0)


def test_public_out_rejects_independent_dlpack_alias(provider, case):
    x = case["x"][:4]
    output = torch.from_dlpack(x)
    assert output.data_ptr() == x.data_ptr()
    assert not torch._C._overlaps(x, output)
    before = x.clone()
    operation = (provider.compact80_ops()[1] if case["k"] == 80
                 else provider.compact160_ops()[1])
    with pytest.raises(RuntimeError, match="contract"):
        operation(x, case["logits"][:4], *case["weights"], output, 10, 1, 512)
    torch.testing.assert_close(x, before, atol=0, rtol=0)


@pytest.mark.parametrize("k", (80, 160))
def test_prefill_empty_and_overflowing_expert_counts(provider, k):
    packed = torch.full((512, 2560, k // 2), 0x11,
                        dtype=torch.uint8, device="xpu")
    scales = torch.ones((512, 2560, (k + 127) // 128),
                        dtype=torch.float16, device="xpu")
    counts = torch.zeros(512, dtype=torch.int32, device="xpu")
    operation = (provider.compact80_ops()[2] if k == 80
                 else provider.compact160_ops()[2])
    empty = operation(torch.empty((0, k), dtype=torch.float16, device="xpu"),
                      packed, scales, counts)
    assert empty.shape == (0, 2560)
    # The int32 sum wraps to M=4, although the real total is 2**32+4.
    counts[:3] = torch.tensor([2147483647, 2147483647, 6],
                             dtype=torch.int32, device="xpu")
    output = operation(torch.ones((4, k), dtype=torch.float16, device="xpu"),
                       packed, scales, counts)
    assert torch.isnan(output).all().item()


def test_packed_topk_finite_half_key_order_on_cpu():
    # 穷举全部有限half编码，确认键排序与float排序相同，且正负零同分。
    bits = torch.arange(65536, dtype=torch.int64)
    values = bits.to(torch.int16).view(torch.float16).float()
    mask = torch.isfinite(values)
    bits, values = bits[mask], values[mask]
    bits = torch.where((bits & 0x7fff) == 0, 0, bits)
    ordered = torch.where(bits & 0x8000 != 0, (~bits) & 0xffff,
                          bits ^ 0x8000)
    permutation = torch.argsort(ordered, stable=True)
    assert torch.equal(values[permutation], torch.sort(values).values)
    assert ordered.min() > 0  # 移除候选的key=0小于任何有限分数。
    equal_zero = ordered[values == 0]
    assert torch.equal(equal_zero, torch.full_like(equal_zero, 0x8000))


@pytest.mark.parametrize("rows", (1, 2, 8, 17))
def test_down_prefetch_bitwise(provider, case, rows, monkeypatch):
    ops = (provider.compact80_ops() if case["k"] == 80
           else provider.compact160_ops())
    operation = ops[0] if rows == 1 else ops[1]
    results = []
    for mode in ("0", "1", "0"):
        monkeypatch.setenv("QWEN38_MOE_M1_DOWN_PREFETCH", mode)
        output = torch.empty_like(case["x"][:rows])
        operation(case["x"][:rows], case["logits"][:rows],
                  *case["weights"], output, 10, 1, 512)
        results.append(output.cpu())
    assert all(torch.equal(results[0], result) for result in results[1:])


@pytest.mark.parametrize("rows", (1, 2, 8))
def test_router_wide_read_bitwise(provider, case, rows, monkeypatch):
    factory = (provider.get_qwen38_moe_m1_direct_workspace_class() if rows == 1
               else provider.get_qwen38_moe_multi_direct_workspace_class())
    workspace = factory()
    extra = () if rows == 1 else (False,)
    results = []
    for mode in ("4", "8", "4"):
        monkeypatch.setenv("QWEN38_MOE_M1_ROUTER_VEC", mode)
        output = workspace.try_run(case["x"][:rows], case["router"],
                                   case["router_scale"], case["weights"],
                                   case["k"], *extra)
        assert output is not None
        results.append(output.cpu())
    assert all(torch.equal(results[0], result) for result in results[1:])


@pytest.mark.parametrize("rows", (1, 2))
def test_m1_tuning_default_dispatch_and_legacy(
    provider, case, rows, monkeypatch
):
    factory = (provider.get_qwen38_moe_m1_direct_workspace_class() if rows == 1
               else provider.get_qwen38_moe_multi_direct_workspace_class())
    space = factory()
    extra = () if rows == 1 else (False,)
    for legacy in (False, True):
        flags = {"QWEN38_MOE_M1_ROUTER_VEC": "4",
                 "QWEN38_MOE_M1_PACKED_TOPK": "0",
                 "QWEN38_MOE_M1_DOWN_PREFETCH": "0"}
        for flag, value in flags.items():
            if legacy:
                monkeypatch.setenv(flag, value)
            else:
                monkeypatch.delenv(flag, raising=False)
        args = (case["x"][:rows], case["router"], case["router_scale"],
                case["weights"], case["k"], *extra)
        assert space.try_run(*args) is not None
        torch.xpu.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.XPU]
        ) as prof:
            assert space.try_run(*args) is not None
        names = [event.name for event in prof.events()]
        assert any("RouterBlockKernel<8>" in name for name in names) == (
            not legacy and rows == 1 and case["k"] == 160)
        assert any("TopKPackedSubgroupKernel" in name for name in names) == (
            not legacy and rows == 1)
        hit_down = any("DownBlockKernel<160, 16, 16, true>" in name
                       for name in names)
        assert hit_down == (not legacy and rows == 1 and case["k"] == 160)


@pytest.mark.parametrize("pattern",
                         ("random", "tie", "zero", "limits", "nan", "inf"))
def test_packed_topk_bitwise_and_invalid(provider, case, pattern, monkeypatch):
    generator = torch.Generator().manual_seed(20261009)
    logits = torch.randn(1, 512, generator=generator).half()
    if pattern == "tie":
        logits = (torch.arange(512) % 5 - 3).half().view(1, 512)
    elif pattern == "zero":
        logits.zero_()
        logits[:, 1::2] = -0.0
    elif pattern == "limits":
        logits = torch.linspace(-65504, 65504, 512).half().view(1, 512)
    elif pattern in ("nan", "inf"):
        logits[0, 511] = float(pattern)
    logits = logits.to("xpu")
    operation = (provider.compact80_ops() if case["k"] == 80
                 else provider.compact160_ops())[0]
    results = []
    for switch in ("0", "1", "0"):
        monkeypatch.setenv("QWEN38_MOE_M1_PACKED_TOPK", switch)
        output = torch.full_like(case["x"][:1], -7)
        operation(case["x"][:1], logits, *case["weights"], output, 10, 1, 512)
        results.append(output.cpu())
    if pattern in ("nan", "inf"):
        assert all(torch.isnan(result).all() for result in results)
    else:
        assert all(torch.equal(results[0], result) for result in results[1:])
        assert torch.isfinite(results[0]).all()


def test_packed_topk_dispatch_and_graph(provider, case, monkeypatch):
    operation = (provider.compact80_ops() if case["k"] == 80
                 else provider.compact160_ops())[0]
    x, logits = case["x"][:1].clone(), case["logits"][:1].clone()
    captures = []
    for mode in ("0", "1"):
        monkeypatch.setenv("QWEN38_MOE_M1_PACKED_TOPK", mode)
        output = torch.empty_like(x)
        operation(x, logits, *case["weights"], output, 10, 1, 512)
        torch.xpu.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.XPU]
        ) as prof:
            operation(x, logits, *case["weights"], output, 10, 1, 512)
        names = [event.name for event in prof.events()]
        hit = any("TopKPackedSubgroupKernel" in name for name in names)
        assert hit == (mode == "1")
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            operation(x, logits, *case["weights"], output, 10, 1, 512)
        captures.append((graph, output))
    for seed in (1, 2, 3):
        torch.manual_seed(seed)
        logits.copy_(torch.randn_like(logits))
        snapshots = []
        for graph, output in captures:
            graph.replay()
            snapshots.append(output.cpu())
        assert torch.equal(*snapshots)
