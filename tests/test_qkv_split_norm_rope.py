# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401

DEVICE = "xpu"


def _rms(x, w, eps, weight_offset):
    orig = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x * (w.float() + weight_offset)).to(orig)


def _interleaved(x, section):
    ch = torch.arange(x.shape[-1], device=x.device)
    is_h = (ch % 3 == 1) & (ch < section[1] * 3)
    is_w = (ch % 3 == 2) & (ch < section[2] * 3)
    return torch.where(is_w, x[2], torch.where(is_h, x[1], x[0]))


def _rope(x, positions, cache, rotary_dim, section, interleaved):
    # x: [T, H, D]; NeoX rotation of the first rotary_dim channels.
    cs = cache[positions]
    cos, sin = cs.chunk(2, dim=-1)
    if positions.dim() == 2:
        if interleaved:
            cos, sin = _interleaved(cos, section), _interleaved(sin, section)
        else:
            cos = torch.cat(
                [m[i] for i, m in enumerate(cos.split(section, -1))], -1)
            sin = torch.cat(
                [m[i] for i, m in enumerate(sin.split(section, -1))], -1)
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    rot, rest = x[..., :rotary_dim].float(), x[..., rotary_dim:]
    x1, x2 = rot.chunk(2, dim=-1)
    out = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), -1)
    return torch.cat((out.to(x.dtype), rest), -1)


def _reference(qkv, positions, qw, kw, cache, hq, hk, d, rotary_dim, eps,
               weight_offset, gated, section, interleaved):
    t = qkv.shape[0]
    qsz, ksz = hq * d, hk * d
    if gated:
        q_gate, k, _ = qkv.split([2 * qsz, ksz, ksz], dim=-1)
        q, gate = torch.chunk(q_gate.view(t, hq, 2 * d), 2, dim=-1)
        gate = gate.reshape(t, qsz)
    else:
        q, k, _ = qkv.split([qsz, ksz, ksz], dim=-1)
        gate = None
    q = _rms(q.reshape(t, hq, d), qw, eps, weight_offset)
    k = _rms(k.reshape(t, hk, d), kw, eps, weight_offset)
    q = _rope(q, positions, cache, rotary_dim, section, interleaved)
    k = _rope(k, positions, cache, rotary_dim, section, interleaved)
    return q.reshape(t, qsz), k.reshape(t, ksz), gate


@pytest.mark.parametrize("num_tokens", [1, 2, 7, 64])
@pytest.mark.parametrize("hq,hk,d,rotary_dim", [(8, 1, 256, 64),
                                                (16, 2, 256, 256),
                                                (8, 2, 128, 64)])
@pytest.mark.parametrize("gated", [True, False])
@pytest.mark.parametrize("mrope", ["none", "interleaved", "chunked"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_qkv_split_norm_rope(num_tokens, hq, hk, d, rotary_dim, gated, mrope,
                             dtype):
    torch.manual_seed(0)
    width = (2 if gated else 1) * hq * d + 2 * hk * d
    qkv = torch.randn(num_tokens, width, device=DEVICE, dtype=dtype) * 3
    qw = torch.randn(d, device=DEVICE, dtype=dtype) * 0.1
    kw = torch.randn(d, device=DEVICE, dtype=dtype) * 0.1
    max_pos = 4096
    cache = torch.randn(max_pos, rotary_dim, device=DEVICE, dtype=dtype)
    half = rotary_dim // 2
    section = [half - 2 * (half // 3), half // 3, half // 3]
    if mrope == "none":
        positions = torch.randint(0, max_pos, (num_tokens, ), device=DEVICE)
    else:
        positions = torch.randint(0, max_pos, (3, num_tokens), device=DEVICE)
    eps, weight_offset = 1e-6, 1.0
    q = torch.empty(num_tokens, hq * d, device=DEVICE, dtype=dtype)
    k = torch.empty(num_tokens, hk * d, device=DEVICE, dtype=dtype)
    gate = torch.empty_like(q) if gated else None
    torch.ops._xpu_C.qkv_split_norm_rope(qkv, positions, qw, kw, cache, q, k,
                                         gate, hq, hk, d, rotary_dim, eps,
                                         weight_offset, section,
                                         mrope == "interleaved")
    rq, rk, rgate = _reference(qkv, positions, qw, kw, cache, hq, hk, d,
                               rotary_dim, eps, weight_offset, gated, section,
                               mrope == "interleaved")
    tol = 2e-2 if dtype == torch.float16 else 8e-2
    torch.testing.assert_close(q, rq, atol=tol, rtol=tol)
    torch.testing.assert_close(k, rk, atol=tol, rtol=tol)
    if gated:
        torch.testing.assert_close(gate, rgate, atol=0, rtol=0)


@pytest.mark.parametrize("invalid", ["cpu_positions", "cpu_weight",
                                    "input_alias", "output_alias",
                                    "gate_alias", "expanded_qkv",
                                    "expanded_positions",
                                    "head_overflow", "negative_section",
                                    "large_section"])
def test_qkv_split_norm_rope_rejects_unsafe_tensors(invalid):
    # Every failure must happen before the first SYCL submission.
    t, hq, hk, d, rotary = 2, 2, 1, 128, 64
    qkv = torch.zeros(t, 2 * (hq + hk) * d, device=DEVICE,
                      dtype=torch.float16)
    pos = torch.zeros(3, t, device=DEVICE, dtype=torch.int64)
    qw = torch.ones(d, device=DEVICE, dtype=qkv.dtype)
    kw = torch.ones_like(qw)
    cache = torch.ones(4, rotary, device=DEVICE, dtype=qkv.dtype)
    q = torch.empty(t, hq * d, device=DEVICE, dtype=qkv.dtype)
    k = torch.empty(t, hk * d, device=DEVICE, dtype=qkv.dtype)
    gate = torch.empty_like(q)
    section = [11, 11, 10]
    if invalid == "head_overflow":
        # Without the head capacity preflight, int64 width/output-numel
        # products wrap to the small, otherwise valid tensors above.
        hq = 2**56 + 2
    elif invalid == "negative_section":
        section = [-1, 22, 11]
    elif invalid == "large_section":
        section = [2**62, 2**62, 32]
    elif invalid == "cpu_positions":
        pos = pos.cpu()
    elif invalid == "cpu_weight":
        qw = qw.cpu()
    elif invalid == "input_alias":
        q = qkv.reshape(-1)[:q.numel()].reshape_as(q)
    elif invalid == "output_alias":
        k = q.reshape(-1)[:k.numel()].reshape_as(k)
    elif invalid == "gate_alias":
        gate = q
    elif invalid == "expanded_qkv":
        qkv = qkv[:1].expand_as(qkv)
    elif invalid == "expanded_positions":
        pos = pos[:1].expand_as(pos)
    with pytest.raises(RuntimeError, match="qkv_split_norm_rope"):
        torch.ops._xpu_C.qkv_split_norm_rope(
            qkv, pos, qw, kw, cache, q, k, gate, hq, hk, d, rotary, 1e-6,
            1.0, section, True)


def test_qkv_split_norm_rope_accepts_disjoint_output_views():
    # Sharing a Storage is safe when the actual output byte ranges differ.
    t, hq, hk, d, rotary = 2, 2, 1, 128, 64
    qkv = torch.randn(t, 2 * (hq + hk) * d, device=DEVICE,
                      dtype=torch.float16)
    pos = torch.zeros(t, device=DEVICE, dtype=torch.int64)
    qw = torch.ones(d, device=DEVICE, dtype=qkv.dtype)
    kw = torch.ones_like(qw)
    cache = torch.ones(4, rotary, device=DEVICE, dtype=qkv.dtype)
    flat = torch.empty(t * (2 * hq + hk) * d, device=DEVICE, dtype=qkv.dtype)
    q, k, gate = [v.reshape(t, -1) for v in flat.split(
        [t * hq * d, t * hk * d, t * hq * d])]
    torch.ops._xpu_C.qkv_split_norm_rope(
        qkv, pos, qw, kw, cache, q, k, gate, hq, hk, d, rotary, 1e-6, 1.0,
        [], False)
    ref = _reference(qkv, pos, qw, kw, cache, hq, hk, d, rotary, 1e-6, 1.0,
                     True, [], False)
    for actual, expected in zip((q, k, gate), ref):
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)



def test_qkv_split_norm_rope_resolves_negative_view():
    # PyTorch's Negative fallback resolves the lazy view before native
    # dispatch. Validate the value instead of expecting a host rejection.
    torch.manual_seed(187)
    t, hq, hk, d, rotary = 2, 2, 1, 128, 64
    qkv = torch.randn(t, 2 * (hq + hk) * d, device=DEVICE,
                      dtype=torch.float16)
    positions = torch.zeros(t, device=DEVICE, dtype=torch.int64)
    base = torch.full((d, ), 0.25, device=DEVICE, dtype=qkv.dtype)
    qw = torch._neg_view(base)
    kw = torch.ones_like(base)
    cache = torch.randn(4, rotary, device=DEVICE, dtype=qkv.dtype)
    q = torch.empty(t, hq * d, device=DEVICE, dtype=qkv.dtype)
    k = torch.empty(t, hk * d, device=DEVICE, dtype=qkv.dtype)
    gate = torch.empty_like(q)
    torch.ops._xpu_C.qkv_split_norm_rope(
        qkv, positions, qw, kw, cache, q, k, gate, hq, hk, d, rotary, 1e-6,
        1.0, [], False)
    reference = _reference(qkv, positions, qw.resolve_neg(), kw, cache,
                           hq, hk, d, rotary, 1e-6, 1.0, True, [], False)
    for actual, expected in zip((q, k, gate), reference):
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    assert qw.is_neg()
    torch.testing.assert_close(base, torch.full_like(base, 0.25),
                               atol=0, rtol=0)
