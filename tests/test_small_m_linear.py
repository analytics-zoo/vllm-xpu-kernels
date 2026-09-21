# SPDX-License-Identifier: Apache-2.0
import math

import pytest
import torch

import vllm_xpu_kernels  # noqa: F401

FP16_SHAPES = [
    (512, 5632),
    (512, 10752),
    (2816, 512),
    (5376, 512),
    (2048, 1024),
    (4096, 1024),
    (8192, 1024),
    (131072, 1024),
    (1024, 2048),
    (1024, 4096),
    (1024, 8192),
    (131072, 2816),
    (131072, 5376),
    (128, 2816),
]
FP8_SHAPES = [
    (8192, 5376),
    (10240, 5376),
    (21504, 5376),
    (5376, 4096),
    (5376, 8192),
    (5376, 10752),
]


@pytest.mark.parametrize("n,k", FP16_SHAPES)
@torch.inference_mode()
def test_small_m_fp16_shape_and_stream(n, k):
    generator = torch.Generator().manual_seed(n + k)
    x = (torch.randn(8, k, generator=generator) * 0.1).half()
    w = (torch.randn(n, k, generator=generator) / math.sqrt(k)).half()
    reference = x.float() @ w.float().T
    stream = torch.xpu.Stream()
    with torch.xpu.stream(stream):
        dx, dw = x.xpu(), w.xpu()
        outputs = [
            torch.ops._xpu_C.unquantized_gemm(dx[:m], dw) for m in range(1, 9)
        ]
    stream.synchronize()
    for m, out in zip(range(1, 9), outputs):
        assert out.shape == (m, n) and out.dtype == torch.float16
        torch.testing.assert_close(
            out.cpu().float(), reference[:m], atol=2e-3, rtol=2e-3
        )
    # Reusing tensor addresses after a weight update must observe new data.
    with torch.xpu.stream(stream):
        dw.mul_(0.5)
        out = torch.ops._xpu_C.unquantized_gemm(dx[:5], dw)
    stream.synchronize()
    torch.testing.assert_close(
        out.cpu().float(), reference[:5] * 0.5, atol=2e-3, rtol=2e-3
    )


@pytest.mark.parametrize("n,k", FP8_SHAPES)
@torch.inference_mode()
def test_small_m_fp8_shape_and_stream(n, k):
    generator = torch.Generator().manual_seed(n + k)
    x = (torch.randn(8, k, generator=generator) * 0.1).half()
    w = torch.randn(n, k, generator=generator).to(torch.float8_e4m3fn)
    scale = 1 / math.sqrt(k)
    reference = (x.float() @ w.float().T) * scale
    stream = torch.xpu.Stream()
    with torch.xpu.stream(stream):
        dx, dw = x.xpu(), w.xpu().T
        ds = torch.tensor(scale, dtype=torch.float32, device="xpu")
        outputs = [
            torch.ops._xpu_C.try_fp8_gemm_w8a16_tla(dx[:m], dw, ds, None)
            for m in range(1, 9)
        ]
        # The optional dispatcher must decline unsupported rows before launch.
        unsupported = torch.ops._xpu_C.try_fp8_gemm_w8a16_tla(
            torch.zeros(9, k, dtype=torch.float16, device="xpu"), dw, ds, None
        )
    stream.synchronize()
    assert unsupported is None
    for m, out in zip(range(1, 9), outputs):
        assert out is not None and out.shape == (m, n)
        torch.testing.assert_close(
            out.cpu().float(), reference[:m], atol=2e-3, rtol=2e-3
        )
