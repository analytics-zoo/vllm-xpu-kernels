# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import torch.nn.functional as F

import vllm_xpu_kernels  # noqa: F401


@pytest.fixture
def linear_op():
    op = getattr(torch.ops._xpu_C, "unquantized_gemm", None)
    if op is None:
        pytest.skip("Optional FP16 linear extension is unavailable")
    return op


@pytest.mark.parametrize("rows", [1, 8, 9])
@pytest.mark.parametrize("bias", [False, True])
@torch.inference_mode()
def test_linear_router_and_fallback_match_reference(linear_op, rows, bias):
    generator = torch.Generator().manual_seed(18)
    x = torch.randn(rows, 2816, generator=generator).half() * 0.1
    w = torch.randn(128, 2816, generator=generator).half() * 0.1
    b = torch.randn(128, generator=generator).half() if bias else None
    reference = F.linear(x.float(), w.float(),
                         b.float() if b is not None else None).half()
    stream = torch.xpu.Stream()
    with torch.xpu.stream(stream):
        actual = linear_op(x.to("xpu"), w.to("xpu"),
                           b.to("xpu") if b is not None else None)
    stream.synchronize()
    torch.testing.assert_close(actual.cpu(), reference,
                               atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("strided", [False, True])
@torch.inference_mode()
def test_linear_fallback_accepts_offset_and_strided_input(linear_op, strided):
    # Neither input may enter the aligned, contiguous TLA path.
    storage = torch.randn(2, 5633, device="xpu", dtype=torch.float16)
    x = storage[:, 1:5633:2] if strided else storage.flatten()[1:2817].view(1, -1)
    w = torch.randn(128, 2816, device="xpu", dtype=torch.float16) * 0.01
    reference = F.linear(x.cpu().float(), w.cpu().float()).half()
    torch.testing.assert_close(linear_op(x, w).cpu(), reference,
                               atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("shape,n", [((1, 2816), 128),
                                     ((1, 2816), 131072),
                                     ((2, 3, 64), 16)])
@pytest.mark.parametrize("bias", [False, True])
def test_linear_meta_preserves_linear_shape_and_dtype(linear_op, shape, n, bias):
    x = torch.empty(shape, device="meta", dtype=torch.float16)
    w = torch.empty(n, shape[-1], device="meta", dtype=x.dtype)
    b = torch.empty(n, device="meta", dtype=x.dtype) if bias else None
    actual = linear_op(x, w, b)
    expected = F.linear(x, w, b)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype


def test_linear_meta_rejects_incompatible_dimensions(linear_op):
    x = torch.empty(1, 32, device="meta")
    w = torch.empty(16, 64, device="meta")
    with pytest.raises(RuntimeError):
        linear_op(x, w)
