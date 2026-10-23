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


@pytest.mark.parametrize(
    "n,k",
    [(14336, 5120), (5120, 7168), (8192, 5120), (5120, 3072), (7168, 5120)],
)
@pytest.mark.parametrize("m", range(1, 9))
def test_dense_fp8_independent_math_and_graph(ops, n, k, m):
    torch.manual_seed(n + k)
    x = (torch.randn(m, k) * 0.2).half().xpu()
    weight = (torch.randn(n, k) * 0.1).to(torch.float8_e4m3fn).xpu()
    scale = torch.tensor([0.37], device="xpu")
    if k == 3072 and m >= 5:
        assert ops.try_qwen4_dense_fp8_linear(x, weight, scale) is None
        return
    assert ops.try_qwen4_dense_fp8_linear(x, weight, scale) is not None
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        output = ops.try_qwen4_dense_fp8_linear(x, weight, scale)
    for factor in (1.0, -0.5, 0.25):
        x.mul_(factor)
        scale.mul_(0.5)
        graph.replay()
        expected = (
            x.cpu().float() @ weight.cpu().float().T * scale.cpu()
        ).half()
        torch.testing.assert_close(
            output.cpu(), expected, atol=0.001, rtol=0.003
        )


@pytest.mark.parametrize("m", range(1, 9))
def test_dense_fp8_pair_reads_distinct_scales_and_live_weights(ops, m):
    torch.manual_seed(5531)
    x = torch.randn(m, 5120, dtype=torch.float16, device="xpu") * 0.2
    w0 = (torch.randn(8192, 5120, device="xpu") * 0.1).to(torch.float8_e4m3fn)
    w1 = (torch.randn(48, 5120, device="xpu") * 0.1).to(torch.float8_e4m3fn)
    s0, s1 = (torch.tensor([s], device="xpu") for s in (0.031, 0.079))
    ops.try_qwen4_dense_fp8_pair(x, w0, s0, w1, s1)
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        outputs = ops.try_qwen4_dense_fp8_pair(x, w0, s0, w1, s1)
    for factor in (1.0, -0.5, 0.25):
        x.mul_(factor)
        w1.copy_((-w1.float()).to(w1.dtype))
        graph.replay()
        for output, weight, scale in zip(outputs, (w0, w1), (s0, s1)):
            expected = (
                x.cpu().float() @ weight.cpu().float().T * scale.cpu()
            ).half()
            torch.testing.assert_close(
                output.cpu(), expected, atol=0.001, rtol=0.003
            )


@pytest.mark.parametrize("invalid", ["batch", "shape", "dtype", "alignment"])
def test_dense_fp8_unsupported_keeps_fallback(ops, invalid):
    x = torch.zeros(1, 5120, device="xpu", dtype=torch.float16)
    w = torch.zeros(7168, 5120, device="xpu", dtype=torch.float8_e4m3fn)
    scale = torch.ones(1, device="xpu")
    if invalid == "batch":
        x = x.expand(9, -1).contiguous()
    elif invalid == "shape":
        w = w[:1024]
    elif invalid == "dtype":
        x = x.bfloat16()
    else:
        x = torch.zeros(5121, device="xpu", dtype=x.dtype)[1:].view(1, -1)
    assert ops.try_qwen4_dense_fp8_linear(x, w, scale) is None


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


@pytest.mark.parametrize(
    "m,hv",
    [
        (1, 6),
        (1, 12),
        (5, 6),
        (8, 12),
        (33, 6),
        (1, 24),
        (5, 24),
        (8, 24),
        (1, 48),
        (5, 48),
        (8, 48),
    ],
)
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


@pytest.mark.parametrize("column", [0, 1, 2, 3])
def test_dense_fp8_preserves_all_e4m3_encodings(ops, column):
    raw = torch.zeros(8192, 5120, dtype=torch.uint8)
    raw[:256, column] = torch.arange(256, dtype=torch.int64).to(torch.uint8)
    weight = raw.view(torch.float8_e4m3fn)
    x = torch.zeros(1, 5120, dtype=torch.float16)
    x[0, column] = 1
    actual = ops.try_qwen4_dense_fp8_linear(
        x.xpu(), weight.xpu(), torch.ones(1, device="xpu")
    )
    expected = weight[:, column].half().view(1, -1)
    torch.testing.assert_close(
        actual.cpu(), expected, atol=0, rtol=0, equal_nan=True
    )


@pytest.mark.parametrize("m", range(1, 9))
@pytest.mark.parametrize("hq,hk", [(12, 2), (24, 4)])
@pytest.mark.parametrize("mrope", ["none", "interleaved", "chunked"])
def test_dense_qkv_mrope_and_dynamic_full_graph(ops, m, hq, hk, mrope):
    torch.manual_seed(4321 + m)
    qkv = (
        torch.randn(
            m, (2 * hq + 2 * hk) * 256, device="xpu", dtype=torch.float16
        )
        * 0.2
    )
    qw, kw = (
        torch.randn(256, device="xpu", dtype=torch.float16) * 0.1
        for _ in range(2)
    )
    cache = torch.randn(4096, 64, device="xpu", dtype=torch.float16) * 0.3
    if mrope == "none":
        positions = torch.arange(m, device="xpu", dtype=torch.int64)
        sections = []
    else:
        storage = torch.arange(
            3 * (m + 9), device="xpu", dtype=torch.int64
        ).view(3, m + 9)
        positions = storage[:, :m]
        sections = [11, 11, 10]

    def run():
        output = ops.try_qwen4_dense_qkv_norm_rope(
            qkv, positions, qw, kw, cache, sections, mrope == "interleaved"
        )
        assert output is not None
        return output

    def reference():
        x = qkv.cpu()
        q, gate = x[:, : hq * 512].reshape(m, hq, 512).chunk(2, -1)
        k = x[:, hq * 512 : hq * 512 + hk * 256].reshape(m, hk, 256)
        cs = cache.cpu()[positions.cpu()]
        if mrope != "none":
            pair = torch.arange(32)
            if mrope == "interleaved":
                axis = torch.where(
                    (pair % 3 == 1) & (pair < 33),
                    1,
                    torch.where((pair % 3 == 2) & (pair < 30), 2, 0),
                )
            else:
                axis = torch.where(pair < 11, 0, torch.where(pair < 22, 1, 2))
            cos, sin = (
                cs[axis, :, pair + offset].T.float()[:, None, :]
                for offset in (0, 32)
            )
        else:
            cos, sin = (part.float()[:, None, :] for part in cs.chunk(2, -1))

        def transform(value, weight):
            value = value.float()
            normed = (
                value
                * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
                * (1 + weight.cpu().float())
            ).half()
            left, right = normed[..., :32].float(), normed[..., 32:64].float()
            return torch.cat(
                (
                    (left * cos - right * sin).half(),
                    (right * cos + left * sin).half(),
                    normed[..., 64:],
                ),
                -1,
            ).flatten(1)

        return (
            transform(q, qw),
            transform(k, kw),
            gate.float().sigmoid().half().flatten(1),
        )

    run()
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        output = run()
    for _ in range(3):
        qkv.add_(0.01)
        qw.add_(0.02)
        kw.mul_(0.9)
        positions.add_(17)
        graph.replay()
        for got, want in zip(output, reference()):
            torch.testing.assert_close(got.cpu(), want, atol=0.002, rtol=0.005)


def test_dense_qkv_unsupported_layout_misses_before_submission(ops):
    x = torch.randn(1, 7168, dtype=torch.float16, device="xpu")
    p = torch.zeros(1, dtype=torch.int64, device="xpu")
    w = torch.zeros(256, dtype=torch.float16, device="xpu")
    cache = torch.zeros(16, 64, dtype=torch.float16, device="xpu")
    assert (
        ops.try_qwen4_dense_qkv_norm_rope(x.float(), p, w, w, cache, [], False)
        is None
    )
    assert (
        ops.try_qwen4_dense_qkv_norm_rope(x, p.int(), w, w, cache, [], False)
        is None
    )
    assert (
        ops.try_qwen4_dense_qkv_norm_rope(x, p, w[:-1], w, cache, [], False)
        is None
    )


@pytest.mark.parametrize("m", range(1, 9))
def test_dense_gated_fp8_math_and_live_graph_replay(ops, m):
    torch.manual_seed(41025 + m)
    x = (torch.randn(m, 3072) * 0.2).half().xpu()
    gate = torch.sigmoid(torch.randn_like(x))
    weight = (torch.randn(5120, 3072) * 0.1).to(torch.float8_e4m3fn).xpu()
    scale = torch.tensor([0.125], device="xpu")
    stream = torch.xpu.Stream() if m % 2 == 0 else torch.xpu.current_stream()
    stream.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(stream):
        initial = ops.try_qwen4_dense_fp8_gated_linear(x, gate, weight, scale)
        assert initial is not None
        for source in (x, gate, weight, scale):
            assert not torch._C._is_alias_of(initial, source)
        stream.synchronize()
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph, stream=stream):
            output = ops.try_qwen4_dense_fp8_gated_linear(
                x, gate, weight, scale
            )
        for step in range(3):
            x.mul_(0.9).add_(0.01 * (step + 1))
            gate.mul_(0.85).add_(0.05)
            weight[::257].copy_((-weight[::257].float()).to(weight.dtype))
            scale.fill_(0.1 + 0.03 * step)
            graph.replay()
            product = (x.cpu().float() * gate.cpu().float()).half().float()
            expected = (
                F.linear(product, weight.cpu().float()) * scale.cpu()
            ).half()
            torch.testing.assert_close(
                output.cpu(), expected, atol=0.0005, rtol=0.005
            )
    stream.synchronize()
    torch.xpu.current_stream().wait_stream(stream)


@pytest.mark.parametrize("column", [0, 3])
def test_dense_gated_fp8_preserves_all_e4m3_encodings(ops, column):
    raw = torch.zeros(5120, 3072, dtype=torch.uint8)
    raw[:256, column] = torch.arange(256, dtype=torch.int64).to(torch.uint8)
    weight = raw.view(torch.float8_e4m3fn)
    x = torch.zeros(1, 3072, dtype=torch.float16)
    x[0, column] = 1.0625
    gate = torch.full_like(x, 0.9375)
    actual = ops.try_qwen4_dense_fp8_gated_linear(
        x.xpu(), gate.xpu(), weight.xpu(), torch.ones(1, device="xpu")
    )
    product = (x.float() * gate.float()).half().float()[0, column]
    expected = (weight[:, column].float() * product).half().view(1, -1)
    torch.testing.assert_close(
        actual.cpu(), expected, atol=0, rtol=0, equal_nan=True
    )


@pytest.mark.parametrize("invalid", ["dtype", "shape", "stride", "alignment"])
def test_dense_gated_fp8_rejects_incompatible_gate_before_submit(ops, invalid):
    x = torch.zeros(2, 3072, device="xpu", dtype=torch.float16)
    gate = torch.ones_like(x)
    weight = torch.zeros(5120, 3072, device="xpu", dtype=torch.float8_e4m3fn)
    scale = torch.ones(1, device="xpu")
    if invalid == "dtype":
        gate = gate.bfloat16()
    elif invalid == "shape":
        gate = gate[:, :-1]
    elif invalid == "stride":
        gate = torch.ones(3072, 2, device="xpu", dtype=x.dtype).t()
    else:
        gate = torch.ones(x.numel() + 1, device="xpu", dtype=x.dtype)[
            1:
        ].view_as(x)
    assert ops.try_qwen4_dense_fp8_gated_linear(x, gate, weight, scale) is None
