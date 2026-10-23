# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import vllm_xpu_kernels.moe_shared_fused_interface as interface
from tests.utils import seed_everything
from vllm_xpu_kernels.moe_shared_fused_interface import (
    XpuMoESharedFusedDecode, is_available, moe_shared_fused_workspace_bytes)

DEVICE = "xpu"
HIDDEN = 2048
NUM_EXPERTS = 256
TOP_K = 8

requires_moe_kernel = pytest.mark.skipif(
    not is_available(), reason="moe_shared_fused_decode op not built")


def quant_fp8_per_tensor(w: torch.Tensor):
    """Per-expert (3D) or per-tensor (2D) e4m3 quantization."""
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    if w.dim() == 3:
        scale = w.abs().amax(dim=(1, 2)) / fp8_max
        q = (w / scale[:, None, None]).to(torch.float8_e4m3fn)
        return q, scale.float(), q.float() * scale[:, None, None]
    scale = (w.abs().max() / fp8_max).reshape(1)
    q = (w / scale).to(torch.float8_e4m3fn)
    return q, scale.float(), q.float() * scale


def make_weights(inter: int, dtype=torch.float16):
    w13f = torch.randn(NUM_EXPERTS, HIDDEN, 2 * inter, device=DEVICE) / 32
    w2f = torch.randn(NUM_EXPERTS, inter, HIDDEN, device=DEVICE) / 32
    sw13f = torch.randn(HIDDEN, 2 * inter, device=DEVICE) / 32
    sw2f = torch.randn(inter, HIDDEN, device=DEVICE) / 32
    gate = (torch.randn(1, HIDDEN, device=DEVICE) / 32).to(dtype)
    return [quant_fp8_per_tensor(w) for w in (w13f, w2f, sw13f, sw2f)] + [gate]


def route(m: int):
    logits = torch.randn(m, NUM_EXPERTS, device=DEVICE)
    probs = torch.softmax(logits, dim=-1)
    topk_weights, topk_ids = torch.topk(probs, TOP_K, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights.contiguous(), topk_ids.to(torch.int32).contiguous()


def ref_moe_shared(x, weights, topk_weights, topk_ids, inter):
    """FP32 routed MoE + sigmoid-gated shared expert on dequantized weights.

    Intermediate projections and activations are rounded to the input dtype.
    """
    (_, _, w13), (_, _, w2), (_, _, sw13), (_, _, sw2), gate = weights
    xf = x.float()
    out = torch.zeros_like(xf)
    for m in range(x.shape[0]):
        for k in range(TOP_K):
            e = int(topk_ids[m, k])
            h = (xf[m] @ w13[e]).to(x.dtype).float()
            act = torch.nn.functional.silu(h[:inter]) * h[inter:]
            projection = (act.to(x.dtype).float() @ w2[e]).to(x.dtype).float()
            out[m] += topk_weights[m, k] * projection
    h = (xf @ sw13).to(x.dtype).float()
    act = torch.nn.functional.silu(h[:, :inter]) * h[:, inter:]
    projection = (act.to(x.dtype).float() @ sw2).to(x.dtype).float()
    out += torch.sigmoid(xf @ gate.float().t()) * projection
    return out


def make_op(weights, tile_m=0):
    (w13, s13, _), (w2, s2, _), (sw13, ss13, _), (sw2, ss2, _), gate = weights
    return XpuMoESharedFusedDecode(w13,
                                   s13,
                                   w2,
                                   s2,
                                   sw13,
                                   ss13,
                                   sw2,
                                   ss2,
                                   gate,
                                   TOP_K,
                                   tile_m=tile_m)


@pytest.mark.parametrize("m", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("inter", [256, 512])
@pytest.mark.parametrize("tile_m", [0, 8, 16])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@requires_moe_kernel
def test_moe_shared_fused_decode(m, inter, tile_m, dtype):
    seed_everything(7)
    weights = make_weights(inter, dtype=dtype)
    op = make_op(weights, tile_m)
    x = (torch.randn(m, HIDDEN, device=DEVICE) / 4).to(dtype)
    topk_weights, topk_ids = route(m)

    out = op.forward(torch.empty_like(x), x, topk_weights, topk_ids)
    ref = ref_moe_shared(x, weights, topk_weights, topk_ids, inter)
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@requires_moe_kernel
def test_moe_shared_fused_decode_int64_ids(dtype):
    seed_everything(7)
    weights = make_weights(256, dtype=dtype)
    op = make_op(weights)
    x = (torch.randn(4, HIDDEN, device=DEVICE) / 4).to(dtype)
    topk_weights, topk_ids = route(4)
    out32 = op.forward(torch.empty_like(x), x, topk_weights, topk_ids)
    out64 = op.forward(torch.empty_like(x), x, topk_weights,
                       topk_ids.long().contiguous())
    torch.testing.assert_close(out32, out64, rtol=0, atol=0)


@requires_moe_kernel
def test_moe_shared_fused_decode_rejects_float32():
    seed_everything(7)
    weights = make_weights(256)
    op = make_op(weights)
    x = torch.randn(2, HIDDEN, device=DEVICE, dtype=torch.float32)
    topk_weights, topk_ids = route(2)
    with pytest.raises(RuntimeError, match="fp16/bf16 activations"):
        op.forward(torch.empty_like(x), x, topk_weights, topk_ids)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@requires_moe_kernel
def test_moe_shared_fused_decode_workspace_size(dtype):
    seed_everything(7)
    inter = 256
    weights = make_weights(inter, dtype=dtype)
    (w13, s13, _), (w2, s2, _), (sw13, ss13, _), (sw2, ss2, _), gate = weights
    m = 3
    x = (torch.randn(m, HIDDEN, device=DEVICE) / 4).to(dtype)
    topk_weights, topk_ids = route(m)
    nbytes = moe_shared_fused_workspace_bytes(m, TOP_K, NUM_EXPERTS, HIDDEN,
                                              2 * inter, inter)

    def run(ws):
        torch.ops._xpu_C.moe_shared_fused_decode_interface(
            torch.empty_like(x), x, topk_ids, topk_weights, w13, s13, w2, s2,
            sw13, ss13, sw2, ss2, gate.reshape(-1), ws, 0)

    run(torch.empty(nbytes, dtype=torch.uint8, device=DEVICE))
    with pytest.raises(RuntimeError, match="workspace too small"):
        run(torch.empty(nbytes - 1, dtype=torch.uint8, device=DEVICE))


@requires_moe_kernel
def test_moe_shared_fused_decode_rejects_mixed_shared_gate_dtype():
    seed_everything(7)
    weights = make_weights(256, dtype=torch.float16)
    op = make_op(weights)
    x = torch.randn(1, HIDDEN, device=DEVICE, dtype=torch.bfloat16)
    topk_weights, topk_ids = route(1)
    with pytest.raises(RuntimeError, match="shared_gate.*input dtype"):
        op.forward(torch.empty_like(x), x, topk_weights, topk_ids)


@pytest.mark.parametrize("dtype,supported", [(torch.float16, True),
                                             (torch.bfloat16, True),
                                             (torch.float32, False),
                                             (torch.float64, False)])
def test_moe_and_router_activation_dtype_contract(monkeypatch, dtype,
                                                  supported):
    monkeypatch.setattr(interface, "is_available", lambda: True)
    monkeypatch.setattr(interface, "router_is_available", lambda: True)
    assert interface.supports(dtype, torch.float8_e4m3fn, 1, 8, 2048,
                              256) is supported
    assert interface.router_supports(dtype, 256, 8, 2048) is supported


def _small_moe_args(m=2, device=DEVICE, e=1, h=128, inter=128):
    top_k = 1
    x = torch.zeros(m, h, device=device, dtype=torch.float16)
    ids = torch.zeros(m, top_k, device=device, dtype=torch.int32)
    probs = torch.ones(m, top_k, device=device, dtype=torch.float32)
    weights = [
        torch.zeros(e, h, 2 * inter, device=device,
                    dtype=torch.float8_e4m3fn),
        torch.ones(e, device=device, dtype=torch.float32),
        torch.zeros(e, inter, h, device=device, dtype=torch.float8_e4m3fn),
        torch.ones(e, device=device, dtype=torch.float32),
        torch.zeros(h, 2 * inter, device=device,
                    dtype=torch.float8_e4m3fn),
        torch.ones(1, device=device, dtype=torch.float32),
        torch.zeros(inter, h, device=device, dtype=torch.float8_e4m3fn),
        torch.ones(1, device=device, dtype=torch.float32),
        torch.zeros(h, device=device, dtype=x.dtype),
    ]
    size = moe_shared_fused_workspace_bytes(m, top_k, e, h, 2 * inter, inter)
    ws = torch.empty(size, device=device, dtype=torch.uint8)
    return [torch.empty_like(x), x, ids, probs, *weights, ws, 8]


@pytest.mark.parametrize("invalid", ["cpu_ids", "expanded_workspace",
                                    "short_ids", "expert_mismatch",
                                    "strided_scales", "output_alias",
                                    "workspace_alias"])
@requires_moe_kernel
def test_moe_shared_fused_decode_rejects_unsafe_tensors(invalid):
    args = _small_moe_args()
    if invalid == "cpu_ids":
        args[2] = args[2].cpu()
    elif invalid == "expanded_workspace":
        args[-2] = args[-2][:1].expand_as(args[-2])
    elif invalid == "short_ids":
        args[2], args[3] = args[2][:1], args[3][:1]
    elif invalid == "expert_mismatch":
        args[6] = torch.zeros(2, 128, 128, device=DEVICE,
                              dtype=torch.float8_e4m3fn)
        args[7] = torch.ones(2, device=DEVICE)
    elif invalid == "strided_scales":
        args = _small_moe_args(e=2)
        args[5] = torch.ones(4, device=DEVICE)[::2]
    elif invalid == "output_alias":
        # Partial overlap with x, rather than the safe exact output==x.
        flat = torch.empty(args[1].numel() + 8, device=DEVICE,
                           dtype=args[1].dtype)
        args[1] = flat[:-8].reshape_as(args[1])
        args[0] = flat[8:].reshape_as(args[0])
    elif invalid == "workspace_alias":
        args[0] = args[-2][:args[0].numel() * 2].view(
            args[0].dtype).reshape_as(args[0])
    with pytest.raises(RuntimeError):
        torch.ops._xpu_C.moe_shared_fused_decode_interface(*args)


@pytest.mark.parametrize("m", [1, 8, 16])
@pytest.mark.parametrize("h,inter", [(64, 16), (128, 128)])
@requires_moe_kernel
def test_moe_shared_fused_decode_multi_tile_and_inplace(m, h, inter):
    # E=1, M=16, tile_m=8 needs TWO expert tiles. The old min(R,E)
    # map allocation had room for only one and corrupted the workspace.
    torch.manual_seed(61)
    args = _small_moe_args(m, h=h, inter=inter)
    args[1].normal_(std=0.1)
    for index in (4, 6, 8, 10):
        args[index] = (torch.randn(args[index].shape, device=DEVICE) /
                       32).to(torch.float8_e4m3fn)
    original = args[1].clone()

    def projection(w13, w2):
        h = (original.float() @ w13.float()).to(original.dtype).float()
        gate, up = h.chunk(2, -1)
        act = (torch.nn.functional.silu(gate) * up).to(original.dtype)
        return (act.float() @ w2.float()).to(original.dtype).float()

    ref = projection(args[4][0], args[6][0])
    ref += 0.5 * projection(args[8], args[10])
    torch.ops._xpu_C.moe_shared_fused_decode_interface(*args)
    torch.testing.assert_close(args[0].float(), ref, atol=2e-3, rtol=2e-3)
    args[0] = args[1]
    torch.ops._xpu_C.moe_shared_fused_decode_interface(*args)
    torch.testing.assert_close(args[0].float(), ref, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("m", [1, 8])
@requires_moe_kernel
def test_moe_shared_fused_refresh_keeps_captured_addresses(m):
    # All arithmetic references run on CPU using the live, dequantized
    # weights; the captured op must consume refreshed fixed-address copies.
    args = _small_moe_args(m)
    x = args[1]
    x.fill_(0.01)
    ids, probs = args[2:4]
    names = ("w13", "w13_scales", "w2", "w2_scales", "shared_w13",
             "shared_w13_scale", "shared_w2", "shared_w2_scale", "shared_gate")

    def sources(value):
        result = {}
        for index, name in enumerate(names, 4):
            template = args[index]
            if name in ("w13", "w2", "shared_w13", "shared_w2"):
                shape = list(template.shape)
                shape[-1], shape[-2] = shape[-2], shape[-1]
                source = torch.full(shape, value, device=DEVICE)
                result[name] = source.to(template.dtype).transpose(-1, -2)
            else:
                result[name] = torch.full_like(template, value)
        return result

    live = sources(0.125)
    op = XpuMoESharedFusedDecode(**live, top_k=1, tile_m=8)
    pointers = {name: getattr(op, name).data_ptr() for name in names}
    output = args[0]
    stream = torch.xpu.Stream()
    stream.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(stream):
        op.forward(output, x, probs, ids)
    stream.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph, stream=stream):
        op.forward(output, x, probs, ids)

    for value in (0.25, 0.0625):
        live = sources(value)
        op.refresh_weights(**live)
        graph.replay()
        xf = x.cpu().float()

        def projection(w13, s13, w2, s2, xf=xf):
            hidden = (xf @ (w13.cpu().float() * s13.cpu().float()))
            gate, up = hidden.to(x.dtype).float().chunk(2, -1)
            act = (torch.nn.functional.silu(gate) * up).to(x.dtype).float()
            down = act @ (w2.cpu().float() * s2.cpu().float())
            return down.to(x.dtype).float()

        routed = projection(live["w13"][0], live["w13_scales"][0],
                            live["w2"][0], live["w2_scales"][0])
        shared = projection(live["shared_w13"], live["shared_w13_scale"],
                            live["shared_w2"], live["shared_w2_scale"])
        gate = torch.sigmoid(xf @ live["shared_gate"].cpu().float().reshape(-1))
        reference = (routed + gate[:, None] * shared).to(x.dtype)
        torch.testing.assert_close(output.cpu(), reference, atol=2e-3,
                                   rtol=2e-3)
        assert all(getattr(op, name).data_ptr() == pointers[name]
                   for name in names)


@requires_moe_kernel
def test_moe_shared_fused_streams_have_distinct_scratch():
    args = _small_moe_args(1)
    op = XpuMoESharedFusedDecode(*args[4:13], top_k=1,
                               router_weight=torch.zeros(
                                   1, 128, device=DEVICE, dtype=torch.float16))
    buffers = []
    for stream in (torch.xpu.Stream(), torch.xpu.Stream()):
        with torch.xpu.stream(stream):
            buffers.append((op.workspace(1, args[1].device),
                            op._routing_buffers(1, args[1].device)))
    assert buffers[0][0].data_ptr() != buffers[1][0].data_ptr()
    assert all(a.data_ptr() != b.data_ptr()
               for a, b in zip(buffers[0][1], buffers[1][1]))



@requires_moe_kernel
def test_moe_shared_fused_decode_resolves_negative_view():
    args = _small_moe_args(1)
    base = torch.full_like(args[1], 0.05)
    args[1] = torch._neg_view(base)
    for index in (4, 6, 8, 10):
        args[index] = torch.full(args[index].shape, 0.125,
                                 device=DEVICE).to(torch.float8_e4m3fn)
    torch.ops._xpu_C.moe_shared_fused_decode_interface(*args)
    actual = args[0]
    args[0] = torch.empty_like(actual)
    args[1] = args[1].resolve_neg()
    torch.ops._xpu_C.moe_shared_fused_decode_interface(*args)
    torch.testing.assert_close(actual, args[0], atol=0, rtol=0)
    torch.testing.assert_close(base, torch.full_like(base, 0.05),
                               atol=0, rtol=0)
