# SPDX-License-Identifier: Apache-2.0
"""Small-M routed experts: independent CPU math and current-stream ordering."""
import pytest
import torch
import torch.nn.functional as F

import vllm_xpu_kernels  # noqa: F401


@pytest.fixture(scope="module")
def expert_weights():
    torch.set_num_threads(4)
    g = torch.Generator().manual_seed(92310)
    w13 = torch.randn(128, 2816, 704, generator=g).to(torch.float8_e4m3fn)
    w2 = torch.randn(128, 352, 2816, generator=g).to(torch.float8_e4m3fn)
    s13 = torch.rand(128, generator=g) * 0.015 + 0.005
    s2 = torch.rand(128, generator=g) * 0.015 + 0.005
    cpu = (w13, s13, w2, s2)
    device = tuple(t.xpu() for t in cpu)
    torch.xpu.synchronize()
    return cpu, device


def reference(x, weights, route_weights, ids):
    w13, s13, w2, s2 = weights
    partial = torch.zeros(x.shape[0], 8, 2816)
    for expert in ids.unique().tolist():
        if not 0 <= expert < 128:
            continue
        rows, slots = (ids == expert).nonzero(as_tuple=True)
        gate_up = (x[rows].float() @ w13[expert].float() * s13[expert]).half()
        gate, up = gate_up.chunk(2, dim=-1)
        activated = (F.gelu(gate.float(), approximate="tanh").half() *
                     up).half()
        partial[rows, slots] = (activated.float() @ w2[expert].float() *
                                s2[expert]).half().float()
    out = torch.zeros_like(x, dtype=torch.float32)
    for slot in range(8):
        out += partial[:, slot] * route_weights[:, slot, None]
    return out.half()


@pytest.mark.parametrize("m,pattern",
                         [(m, "random")
                          for m in range(1, 9)] + [(5, "shared"),
                                                   (5, "disjoint"),
                                                   (5, "invalid")])
def test_experts_current_stream(expert_weights, m, pattern):
    cpu, device = expert_weights
    g = torch.Generator().manual_seed(235 + m)
    x = (torch.randn(m, 2816, generator=g) * 0.5).half()
    ids = torch.rand(m, 128, generator=g).topk(8, dim=-1).indices.int()
    if pattern == "shared":
        ids[:] = ids[0].clone()
    elif pattern == "disjoint":
        ids = torch.arange(m * 8, dtype=torch.int32).reshape(m, 8)
    elif pattern == "invalid":
        ids[:, 0] = -1
        ids[:, -1] = 128
    rw = torch.softmax(torch.randn(m, 8, generator=g), dim=-1)
    expected = reference((x * 0.875).half(), cpu, rw, ids)
    stream = torch.xpu.Stream()
    with torch.xpu.stream(stream):
        x_d, rw_d, ids_d = x.xpu(), rw.xpu(), ids.xpu()
        x_d.mul_(0.875)
        out = torch.empty_like(x_d)
        for _ in range(2):
            assert torch.ops._xpu_C.fp8_moe_decode(out, x_d, *device, rw_d,
                                                   ids_d)
        # Consumer must observe the submitted output on the same stream.
        actual = out.clone()
    stream.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-2, atol=5e-3)


@pytest.mark.parametrize("case", ["m9", "alias", "strided"])
def test_unsupported_returns_false_without_writing(expert_weights, case):
    _, device = expert_weights
    m = 9 if case == "m9" else 5
    x = torch.ones(m, 2816, dtype=torch.float16, device="xpu")
    if case == "strided":
        x = torch.ones(m, 5632, dtype=torch.float16, device="xpu")[:, ::2]
    out = x if case == "alias" else torch.full(
        (m, 2816), 3., dtype=torch.float16, device="xpu")
    before = out.clone()
    ids = torch.zeros(m, 8, dtype=torch.int32, device="xpu")
    rw = torch.ones(m, 8, device="xpu") / 8
    assert not torch.ops._xpu_C.fp8_moe_decode(out, x, *device, rw, ids)
    torch.testing.assert_close(out, before, rtol=0, atol=0)
