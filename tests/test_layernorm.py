# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm_xpu_kernels._C  # noqa: F401
from tests.ops.layernorm_op import RMSNorm
from tests.utils import opcheck


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 5120), (1, 2048), (1, 257),
                                   (1, 12, 256), (1, 8, 256), (1, 1, 256),
                                   (1, 2, 256), (1, 16, 128)])
@pytest.mark.parametrize("strided", [False, True])
@torch.inference_mode()
def test_rms_norm_fp32_weight(dtype, shape, strided):
    """Multiply FP32 weights before converting the normalized output."""
    torch.manual_seed(42)
    hidden = shape[-1]
    if strided:
        x = torch.randn((*shape[:-1], hidden * 2), device="xpu", dtype=dtype)
        x = x[..., 1:hidden + 1]
        weight = torch.randn(hidden + 1, device="xpu", dtype=torch.float32)[1:]
    else:
        x = torch.randn(shape, device="xpu", dtype=dtype)
        weight = torch.randn(hidden, device="xpu", dtype=torch.float32)
    original = x.clone()
    xf = x.float()
    ref = (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6) *
           weight).to(dtype)
    out = torch.empty(shape, device="xpu", dtype=dtype)
    torch.ops._C.rms_norm(out, x, weight, 1e-6)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=2e-3)
    torch.testing.assert_close(x, original, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("hidden", [256, 257, 2048, 5120])
@pytest.mark.parametrize("strided", [False, True])
@torch.inference_mode()
def test_fused_add_rms_norm_fp32_weight(dtype, hidden, strided):
    """Mixed-weight residual variance uses the unrounded FP32 sum."""
    torch.manual_seed(43)
    x = torch.randn((1, hidden * 2 if strided else hidden),
                    device="xpu",
                    dtype=dtype)
    if strided:
        x = x[:, 1:hidden + 1]
    residual = torch.randn((1, hidden), device="xpu", dtype=dtype)
    weight = torch.randn(hidden + int(strided),
                         device="xpu",
                         dtype=torch.float32)
    if strided:
        weight = weight[1:]
    total = x.float() + residual.float()
    ref_residual = total.to(dtype)
    ref = (total * torch.rsqrt(total.square().mean(-1, keepdim=True) + 1e-6) *
           weight).to(dtype)
    torch.ops._C.fused_add_rms_norm(x, residual, weight, 1e-6)
    torch.testing.assert_close(x, ref, atol=1e-2, rtol=2e-3)
    torch.testing.assert_close(residual, ref_residual, atol=0, rtol=0)


DTYPES = [torch.half, torch.bfloat16]
NUM_TOKENS = [1, 7, 83, 4096]  # Arbitrary values for testing
# TODO: add back  5120, 5124, 5125, 5126, 8192, 8199 after ci env issue fixed
HIDDEN_SIZES = [8, 768, 769, 770, 771, 5120, 5124, 5125, 5126, 8192,
                8199]  # Arbitrary values for testing
HEAD_DIMS = [128, 64]
NUM_Q_HEADS = [32, 40, 64]
NUM_KV_HEADS = [8, 32]
ADD_RESIDUAL = [False, True]
SEEDS = [0]
XPU_DEVICES = [
    f"xpu:{i}" for i in range(1 if torch.xpu.device_count() == 1 else 2)
]

# override pytest parameters when enable mini pytest
MINI_PYTEST_PARAMS = {
    "default": {
        "num_tokens": [7],
        "hidden_size": [8],
    },
}


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("add_residual", ADD_RESIDUAL)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", XPU_DEVICES)
@pytest.mark.parametrize("strided_input", [False, True])
@torch.inference_mode()
def test_rms_norm(
    num_tokens: int,
    hidden_size: int,
    add_residual: bool,
    dtype: torch.dtype,
    seed: int,
    device: str,
    strided_input: bool,
) -> None:
    # Note: torch.set_default_device("xpu:1") not works.
    torch.set_default_device("xpu")
    torch.xpu.set_device(device)
    layer = RMSNorm(hidden_size).to(dtype=dtype)
    layer.weight.data.normal_(mean=1.0, std=0.1)
    scale = 1 / (2 * hidden_size)
    last_dim = 2 * hidden_size if strided_input else hidden_size
    x = torch.randn(num_tokens, last_dim, dtype=dtype)
    x = x[..., :hidden_size]
    if num_tokens > 1:
        assert x.is_contiguous() != strided_input
    x *= scale
    residual = torch.randn_like(x) * scale if add_residual else None

    # NOTE(woosuk): The reference implementation should be executed first
    # because the custom kernel is in-place.
    ref_out = layer.forward_native(x, residual)
    out = layer(x, residual)
    # NOTE(woosuk): LayerNorm operators (including RMS) typically have larger
    # numerical errors than other operators because they involve reductions.
    # Therefore, we use a larger tolerance.
    if add_residual:
        torch.testing.assert_close(out[0], ref_out[0], atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(out[1], ref_out[1], atol=1e-2, rtol=1e-2)

    else:
        torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)

    if residual is not None:
        opcheck(torch.ops._C.fused_add_rms_norm,
                (x, residual, layer.weight.data, layer.variance_epsilon))
    else:
        opcheck(torch.ops._C.rms_norm,
                (out, x, layer.weight.data, layer.variance_epsilon))


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("num_q_heads", NUM_Q_HEADS)
@pytest.mark.parametrize("num_kv_heads", NUM_KV_HEADS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", XPU_DEVICES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_rms_norm_uncontigous(
    num_tokens: int,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
) -> None:
    torch.manual_seed(seed)
    torch.set_default_device("xpu")
    torch.xpu.set_device(device)

    hidden_size = (num_q_heads + 2 * num_kv_heads) * head_dim
    qkv = torch.randn(num_tokens, hidden_size, dtype=dtype)
    q_size = num_q_heads * head_dim
    kv_size = num_kv_heads * head_dim
    q, _, _ = qkv.split([q_size, kv_size, kv_size], dim=-1)
    q_by_head = q.view(*q.shape[:-1], q.shape[-1] // head_dim, head_dim)

    layer = RMSNorm(head_dim).to(dtype=dtype)
    ref_out = layer.forward_native(q_by_head)
    out = layer(q_by_head)
    torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)

    opcheck(
        torch.ops._C.rms_norm,
        (out, q_by_head, layer.weight.data, layer.variance_epsilon),
    )


@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("case", ["random", "zero", "extreme_gate"])
@pytest.mark.parametrize("rows", [16, 24])
@torch.inference_mode()
def test_gated_decode_norm_preserves_inputs(seed, case, rows):
    """GDN bsz=1: FP32 norm/gate math and a single FP16 output cast."""
    torch.manual_seed(seed)
    x = torch.randn(rows, 128, device="xpu", dtype=torch.float16)
    z = torch.randn_like(x)
    weight = torch.randn(128, device="xpu", dtype=torch.float16)
    if case == "zero":
        x.zero_()
    elif case == "extreme_gate":
        z = torch.linspace(-80, 80, 128, device="xpu",
                           dtype=torch.float16).expand(rows, -1).contiguous()
        x.mul_(100)
    originals = [t.clone() for t in (x, z, weight)]
    out = torch.full_like(x, float("nan"))
    torch.ops._C.rms_norm_gated_decode(out, x, z, weight, 1e-6)
    xf = x.float()
    expected = (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6)
                * weight.float() * torch.nn.functional.silu(z.float())).half()
    torch.testing.assert_close(out, expected, atol=2e-3, rtol=2e-3)
    for actual, original in zip((x, z, weight), originals):
        torch.testing.assert_close(actual, original, atol=0, rtol=0)
