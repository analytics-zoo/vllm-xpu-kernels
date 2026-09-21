# SPDX-License-Identifier: Apache-2.0
"""Numerical regression for the existing and extended small-M decode paths."""

import pytest
import torch

import vllm_xpu_kernels._C  # noqa: F401
import vllm_xpu_kernels._xpu_C  # noqa: F401


@pytest.mark.parametrize("m", [1, 2, 8])
@pytest.mark.parametrize("width", [256, 512, 2816, 5376])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("offset", [0, 1])
def test_rms_decode_strided(m, width, weighted, offset):
    torch.manual_seed(831)
    heads = 8 if width == 256 else 1
    storage = torch.randn(m, heads, width + 8).half().to("xpu")
    x = storage[..., offset:offset + width]
    weight = torch.randn(width).half().to("xpu") if weighted else None
    ref = x.cpu().float()
    ref = (ref * torch.rsqrt(ref.square().mean(-1, keepdim=True)
                             + 1e-6)).half()
    if weight is not None:
        ref = (ref * weight.cpu()).half()
    y = torch.ops._xpu_C.rms_norm_decode_dispatch(x, weight, 1e-6)
    torch.testing.assert_close(y.cpu(), ref, rtol=1e-3, atol=2e-3)


# Existing TP2 shapes exercise both M1 fused reduction and M2..8 Split-K.
# New shapes exercise the SLM reduction selected only for M1.
_LINEAR_CASES = [
    (m, n, k)
    for m in (1, 2, 8)
    for n, k in ((4096, 2816), (2816, 2048), (5120, 2816),
                 (2816, 4096), (2112, 2816), (2816, 1056))
] + [(1, n, k) for n, k in ((8192, 5376), (5376, 4096),
                           (10240, 5376), (5376, 8192),
                           (21504, 5376), (5376, 10752))]


@pytest.mark.parametrize("m,n,k", _LINEAR_CASES)
def test_fp8_decode_nt(m, n, k):
    torch.manual_seed(823)
    x = torch.randn(m, k).half()
    w = (torch.randn(n, k) * 16).to(torch.float8_e4m3fn)
    scale = torch.tensor([0.002], dtype=torch.float32)
    ref = x.float() @ (w.float() * scale).t()
    y = torch.ops._xpu_C.fp8_gemm_w8a16(
        x.to("xpu"), w.to("xpu").t(), scale.to("xpu"), None)
    torch.testing.assert_close(y.cpu().float(), ref, rtol=1e-2, atol=5e-3)
