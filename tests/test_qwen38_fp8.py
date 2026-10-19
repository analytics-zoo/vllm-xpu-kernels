# SPDX-License-Identifier: Apache-2.0
"""Independent W8A16 math, native-width layout and dynamic graph replay."""

import importlib
import importlib.util
import os

import pytest
import torch
import torch.nn.functional as F


@pytest.fixture(scope="module")
def ops():
    if not torch.xpu.is_available():
        pytest.skip("XPU required")
    torch.set_num_threads(1)
    path = os.getenv("QWEN38_FP8_LIBRARY")
    if path:
        spec = importlib.util.spec_from_file_location("_qwen38_fp8_C", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return importlib.import_module("vllm_xpu_kernels.qwen38_fp8")


@pytest.mark.parametrize("m", [1, 2, 3, 5, 8])
@pytest.mark.parametrize(
    "n,k",
    [
        (12, 2560),
        (24, 2560),
        (160, 2560),
        (2048, 2560),
        (2560, 768),
        (2560, 80),
    ],
)
def test_linear_independent_fp32(ops, m, n, k):
    torch.manual_seed(m + n + k)
    x = torch.randn(m, k).half()
    w = (torch.randn(n, k) * 12).to(torch.float8_e4m3fn)
    scale = torch.tensor([0.0023])
    expected = (x.float() @ w.float().T * scale).half()
    actual = ops.try_qwen38_fp8_linear(
        x.to("xpu"), w.to("xpu"), scale.to("xpu")
    )
    torch.testing.assert_close(actual.cpu(), expected, atol=0.001, rtol=0.002)


def test_all_e4m3_encodings(ops):
    raw = torch.zeros(320, 2560, dtype=torch.uint8)
    raw[:256, 0] = torch.arange(256, dtype=torch.int64).to(torch.uint8)
    w = raw.view(torch.float8_e4m3fn)
    x = torch.zeros(1, 2560, dtype=torch.float16)
    x[0, 0] = 1
    actual = ops.try_qwen38_fp8_linear(
        x.to("xpu"), w.to("xpu"), torch.ones(1, device="xpu")
    ).cpu()
    expected = w[:, 0].half().view(1, -1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("m,hv", [(1, 6), (1, 12), (5, 6), (8, 12), (33, 6)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("sigmoid", [False, True])
def test_gdn_norm_keeps_weight_precision(ops, m, hv, dtype, sigmoid):
    torch.manual_seed(m + hv)
    x = torch.randn(m, hv, 128).half()
    z = torch.randn_like(x)
    weight = torch.linspace(0.7, 1.3, 128).to(dtype)
    gate = torch.sigmoid(z.float())
    if not sigmoid:
        gate *= z.float()
    expected = (
        x.float()
        * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        * weight.float()
        * gate
    ).half()
    actual = ops.try_gdn_norm(x.xpu(), z.xpu(), weight.xpu(), 1e-6, sigmoid)
    torch.testing.assert_close(actual.cpu(), expected, atol=0.001, rtol=0.001)


def test_gdn_norm_graph_replay_reads_live_weight(ops):
    x = torch.randn(1, 6, 128, device="xpu", dtype=torch.float16)
    z = torch.randn_like(x)
    weight = torch.ones(128, device="xpu", dtype=torch.float32)
    ops.try_gdn_norm(x, z, weight, 1e-6, True)
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        out = ops.try_gdn_norm(x, z, weight, 1e-6, True)
    for step in range(3):
        x.fill_(0.1 + step)
        weight.fill_(1.0003 + step * 0.1)
        expected = ops.try_gdn_norm(x, z, weight, 1e-6, True)
        graph.replay()
        torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.fixture(scope="module", params=[80, 160])
def weights(request):
    i = request.param
    g = torch.Generator().manual_seed(113 + i)

    def fp8(*shape):
        return (torch.randn(shape, generator=g) * 12).to(torch.float8_e4m3fn)

    cpu = [
        fp8(512, 2560, 2 * i),
        torch.rand(512, generator=g) * 0.001 + 0.001,
        fp8(512, i, 2560),
        torch.rand(512, generator=g) * 0.001 + 0.001,
        fp8(2 * i, 2560),
        torch.tensor([0.0015]),
        fp8(2560, i),
        torch.tensor([0.0015]),
        (torch.randn(1, 2560, generator=g) * 0.02).half(),
    ]
    router = (torch.randn(512, 2560, generator=g) * 0.02).half()
    return cpu, router, [w.to("xpu") for w in cpu], router.to("xpu")


def reference(x, rw, w):
    x = x.float()
    logits = (x @ rw.float().T).half().float()
    ids = torch.argsort(logits, descending=True, stable=True)[:, :10]
    probs = torch.softmax(logits.gather(1, ids), dim=-1)
    m, h = x.shape
    i = w[0].shape[-1] // 2
    out = torch.zeros(m, h)
    for t in range(m):
        for r in range(10):
            e = ids[t, r].item()
            gu = (x[t] @ w[0][e].float() * w[1][e]).half().float()
            inter = (F.silu(gu[:i]) * gu[i:]).half().float()
            out[t] += (
                (inter @ w[2][e].float() * (w[3][e] * probs[t, r]))
                .half()
                .float()
            )
    gu = (x @ w[4].float().T * w[5]).half().float()
    inter = (F.silu(gu[:, :i]) * gu[:, i:]).half().float()
    shared = (inter @ w[6].float().T * w[7]).half()
    gate = torch.sigmoid((x @ w[8].float().T).half().float()).half()
    return (out.half().float() + (shared * gate).float()).half()


@pytest.mark.parametrize("m", range(1, 9))
@pytest.mark.parametrize(
    "splits,fuse",
    [
        (-1, -1),
        (8, False),
        (4, False),
        (16, False),
        (4, True),
        (8, True),
        (16, True),
        (20, True),
    ],
)
def test_moe_independent_fp32(ops, weights, m, splits, fuse):
    cpu, rw, device, drw = weights
    x = torch.randn(
        m, 2560, generator=torch.Generator().manual_seed(72 + m)
    ).half()
    actual = ops.Workspace(up_splits=splits, fuse_down=fuse).try_run(
        x.to("xpu"), drw, device
    )
    torch.testing.assert_close(
        actual.cpu(), reference(x, rw, cpu), atol=3e-4, rtol=0.01
    )


def test_moe_outputs_survive_alternating_streams(ops, weights):
    cpu, rw, device, drw = weights
    workspace = ops.Workspace()
    streams = [torch.xpu.current_stream(), torch.xpu.Stream()]
    streams[1].wait_stream(streams[0])
    retained = []
    for step, m in enumerate([1, 4, 2, 8, 1, 3, 1]):
        x = torch.randn(
            m, 2560, generator=torch.Generator().manual_seed(29 + step)
        ).half()
        with torch.xpu.stream(streams[step % 2]):
            retained.append(
                (
                    workspace.try_run(x.to("xpu"), drw, device),
                    reference(x, rw, cpu),
                )
            )
    torch.xpu.synchronize()
    assert len({out.data_ptr() for out, _ in retained}) == len(retained)
    for out, expected in retained:
        torch.testing.assert_close(out.cpu(), expected, atol=3e-4, rtol=0.01)


def test_rejected_contract_has_no_submission(ops, weights):
    _, _, device, drw = weights
    workspace = ops.Workspace()
    x = torch.randn(1, 2560, device="xpu", dtype=torch.float16)
    out = workspace.try_run(x, drw, device)
    saved = out.clone()
    bad = list(device)
    bad[7] = bad[7].double()
    assert workspace.try_run(x, drw, bad) is None
    assert workspace.try_run(x[:, ::2], drw, device) is None
    assert workspace.try_run(x, drw, device[:-1]) is None
    torch.testing.assert_close(out, saved, atol=0, rtol=0)
    assert ops.try_qwen38_fp8_linear(x, device[4].T, device[5]) is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_routes_remain_in_bounds(ops, weights, bad):
    _, _, device, drw = weights
    workspace = ops.Workspace()
    x = torch.ones(1, 2560, device="xpu", dtype=torch.float16)
    out = workspace.try_run(x, torch.full_like(drw, bad), device)
    assert torch.isnan(out).all()
    torch.testing.assert_close(
        workspace.try_run(x, drw, device),
        ops.Workspace().try_run(x, drw, device),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("m", [1, 5])
@pytest.mark.parametrize("fuse", [False, True])
def test_graph_replay_uses_live_input_and_weights(ops, weights, m, fuse):
    _, _, device, drw = weights
    workspace = ops.Workspace(fuse_down=fuse)
    x = torch.randn(m, 2560, device="xpu", dtype=torch.float16)
    workspace.try_run(x, drw, device)
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        out = workspace.try_run(x, drw, device)
    original_scale = device[5].clone()
    try:
        for step in range(3):
            x.fill_(0.1 * (step + 1))
            device[5].fill_(0.0015 + step * 0.0001)
            expected = workspace.try_run(x, drw, device)
            graph.replay()
            torch.testing.assert_close(out, expected, rtol=0, atol=0)
    finally:
        device[5].copy_(original_scale)


@pytest.mark.parametrize("offset", [8, 16, 24])
def test_m1_router_partial_alignment_preserves_fallback(ops, weights, offset):
    cpu, rw, device, drw = weights

    def shifted(weight):
        storage = torch.empty(
            weight.numel() + offset, dtype=weight.dtype, device=weight.device
        )
        view = storage[offset:].view_as(weight)
        view.copy_(weight)
        assert view.data_ptr() % 64 == (offset * weight.element_size()) % 64
        return view

    routed = shifted(drw)
    active = device.copy()
    active[8] = shifted(device[8])
    torch.manual_seed(220014 + offset)
    x = (torch.randn(1, 2560) * 0.2).half()
    actual = ops.Workspace().try_run(x.to("xpu"), routed, active)
    assert actual is not None
    torch.testing.assert_close(
        actual.cpu(), reference(x, rw, cpu), atol=3e-4, rtol=0.01
    )
