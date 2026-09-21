# SPDX-License-Identifier: Apache-2.0
"""Small-batch GELU: public op, current stream, and storage offsets."""
import pytest
import torch
import torch.nn.functional as F

import vllm_xpu_kernels._C  # noqa: F401


@pytest.mark.parametrize("m,offset", [(m, offset) for m in range(1, 9)
                                      for offset in (0, 1)] + [(9, 0)])
def test_gelu_small_batch(m, offset):
    torch.manual_seed(920 + m)
    x_cpu = (torch.randn(m, 8192) * 3).half()
    x = torch.empty(m * 8192 + offset, dtype=torch.float16,
                    device="xpu")[offset:].view(m, 8192)
    out = torch.empty(m * 4096 + offset, dtype=torch.float16,
                      device="xpu")[offset:].view(m, 4096)
    # Both producer and consumer use a non-default stream.
    stream = torch.xpu.Stream()
    with torch.xpu.stream(stream):
        x.copy_(x_cpu)
        torch.ops._C.gelu_tanh_and_mul(out, x)
        result = out.clone()
    stream.synchronize()
    gate = F.gelu(x_cpu[:, :4096].float(), approximate="tanh").half()
    ref = gate * x_cpu[:, 4096:]
    torch.testing.assert_close(result.cpu(), ref, atol=2e-3, rtol=1e-3)


@pytest.mark.parametrize("dtype",
                         [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("width", [352, 1056, 4096, 10752])
def test_gelu_small_batch_fallback(dtype, width):
    torch.manual_seed(921)
    x_cpu = torch.randn(3, width * 2).to(dtype)
    x = x_cpu.to("xpu")
    out = torch.empty((3, width), dtype=dtype, device="xpu")
    torch.ops._C.gelu_tanh_and_mul(out, x)
    gate = F.gelu(x_cpu[:, :width].float(), approximate="tanh").to(dtype)
    ref = gate * x_cpu[:, width:]
    torch.testing.assert_close(out.cpu(), ref, atol=2e-3, rtol=1e-3)


def test_gelu_small_batch_special_values():
    values = torch.tensor([
        float("nan"),
        float("inf"), -float("inf"), -65504., 65504., -10., 10., 0.
    ]).half()
    gate = values.repeat(512).view(1, 4096)
    x = torch.cat((gate, torch.ones_like(gate)), dim=-1).to("xpu")
    out = torch.empty((1, 4096), device="xpu", dtype=torch.float16)
    torch.ops._C.gelu_tanh_and_mul(out, x)
    ref = F.gelu(gate.float(), approximate="tanh").half()
    torch.testing.assert_close(out.cpu(),
                               ref,
                               atol=2e-3,
                               rtol=1e-3,
                               equal_nan=True)
