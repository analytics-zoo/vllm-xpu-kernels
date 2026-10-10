# SPDX-License-Identifier: Apache-2.0
"""Independent TP4 M1 root-group GDN checks; GPU tests require a frozen DSO.

CPU reference never calls the native implementation. Run GPU tests only with
QWEN38_GDN_ROOT_LIBRARY set to the candidate's absolute DSO path.
"""

import math
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


H, HV, K = 4, 12, 128
DIM = (2 * H + HV) * K
SCALE = 1 / math.sqrt(K)


def make_case(seed=431, m=1, hv=HV):
    h = hv // 3
    dim = (2 * h + hv) * K
    gen = torch.Generator().manual_seed(seed)

    def half_rand(*shape, factor):
        return (torch.randn(*shape, generator=gen) * factor).half()

    return dict(
        qkvz=half_rand(m, dim + hv * K, factor=0.08),
        conv=half_rand(8, 3, dim, factor=0.03),
        weight=half_rand(dim, 4, factor=0.1),
        bias=half_rand(dim, factor=0.01),
        conv_idx=torch.arange(m, dtype=torch.int32),
        a_log=half_rand(hv, factor=0.05) + 0.3,
        dt_bias=half_rand(hv, factor=0.03),
        ba=half_rand(m, 2 * hv, factor=0.08),
        ssm=half_rand(8, hv, K, K, factor=0.002),
        ssm_idx=torch.arange(m, dtype=torch.int32),
        output=torch.full((m, hv, K), -9, dtype=torch.float16),
        z=torch.full((m, hv, K), -9, dtype=torch.float16),
    )


def clone_case(case, device=None):
    return {key: value.clone().to(device) for key, value in case.items()}


def decode_args(c):
    return tuple(c[key] for key in (
        "qkvz", "conv", "weight", "bias", "conv_idx", "a_log",
        "dt_bias", "ba", "ssm", "ssm_idx", "output", "z"
    )) + (SCALE,)


def conv_fp32(history, x, weight, bias):
    # Match C++ product order: a*w0+b*w1+c*w2+x*w3+bias, then FP32 SiLU.
    terms = (history[0], history[1], history[2], x)
    acc = terms[0].float() * weight[:, 0].float()
    for i in range(1, 4):
        acc = acc + terms[i].float() * weight[:, i].float()
    acc = acc + bias.float()
    return F.silu(acc)


def conv_wrong_half_products_probe(history, x, weight, bias):
    # Intentionally wrong probe: half-round each product/sum. This is NOT the
    # legacy two-kernel implementation, which rounds only its qkv scratch.
    terms = (history[0], history[1], history[2], x)
    acc = (terms[0] * weight[:, 0]).half()
    for i in range(1, 4):
        acc = (acc + (terms[i] * weight[:, i]).half()).half()
    acc = (acc + bias).half()
    return F.silu(acc.float()).half()


def reference_step(c, *, local_fp32=True):
    h, hv = c["ssm"].shape[1] // 3, c["ssm"].shape[1]
    dim = (2 * h + hv) * K
    out = clone_case(c)
    for t in range(c["qkvz"].shape[0]):
        ci, si = int(c["conv_idx"][t]), int(c["ssm_idx"][t])
        if not (0 <= ci < c["conv"].shape[0] and
                0 <= si < c["ssm"].shape[0]):
            out["output"][t].zero_()
            out["z"][t].zero_()
            continue
        history = c["conv"][ci].clone()
        x = c["qkvz"][t, :dim]
        qkv = conv_fp32(history, x, c["weight"], c["bias"])
        if not local_fp32:
            qkv = qkv.half().float()  # legacy two-kernel scratch boundary
        q = qkv[:h * K].reshape(h, K)
        k = qkv[h * K:2 * h * K].reshape(h, K)
        v = qkv[2 * h * K:].reshape(hv, K)
        q *= torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * SCALE
        k *= torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
        out["z"][t] = c["qkvz"][t, dim:].reshape(hv, K)
        for head in range(hv):
            kh = head // 3
            beta = torch.sigmoid(c["ba"][t, head].float())
            decay = torch.exp(
                -torch.exp(c["a_log"][head].float()) *
                F.softplus(c["ba"][t, hv + head].float() +
                           c["dt_bias"][head].float()))
            state = c["ssm"][si, head].float() * decay
            dot = (state * k[kh]).sum(-1)
            state += ((v[head] - dot) * beta)[:, None] * k[kh]
            out["output"][t, head] = (state * q[kh]).sum(-1).half()
            out["ssm"][si, head] = state.half()
        out["conv"][ci, 0:2] = history[1:3]
        out["conv"][ci, 2] = x
    return out


def make_precision_probe():
    c = make_case(719)
    c["qkvz"].zero_()
    c["conv"].zero_()
    c["weight"].zero_()
    c["bias"].zero_()
    c["ba"].zero_()
    c["ba"][0, :HV] = 8  # beta nearly one
    c["ssm"].zero_()
    # One-hot q/k produces a transparent recurrent path for v[0].
    c["bias"][0] = 2
    c["bias"][H * K] = 2
    v_feature = 2 * H * K
    gen = torch.Generator().manual_seed(818)
    histories = (torch.randn(4096, 3, generator=gen) * 3).half()
    inputs = (torch.randn(4096, generator=gen) * 3).half()
    weights = (torch.randn(4096, 4, generator=gen) * 2).half()
    biases = (torch.randn(4096, generator=gen) * 0.5).half()
    terms = (*histories.T, inputs)
    acc = terms[0].float() * weights[:, 0].float()
    wrong_half = (terms[0] * weights[:, 0]).half()
    for i in range(1, 4):
        acc = acc + terms[i].float() * weights[:, i].float()
        wrong_half = (wrong_half + (terms[i] * weights[:, i]).half()).half()
    acc = acc + biases.float()
    wrong_half = (wrong_half + biases).half()
    full = F.silu(acc)
    rounded = F.silu(wrong_half.float()).half().float()
    # Select a case that survives the FP16 SSM write. A conv-only gap can be
    # real yet disappear at that final boundary.
    gap = (full - rounded).abs()
    gap[(full.abs() > 8) | (full.abs() < 0.25)] = 0
    beta = torch.sigmoid(torch.tensor(8.0))
    distinct_state = (full * beta).half() != (full.half().float() * beta).half()
    gap[~distinct_state] = 0
    chosen = int(gap.argmax())
    assert gap[chosen] >= 0.003, "precision probe did not separate paths"
    c["conv"][0, :, v_feature] = histories[chosen]
    c["qkvz"][0, v_feature] = inputs[chosen]
    c["weight"][v_feature] = weights[chosen]
    c["bias"][v_feature] = biases[chosen]
    history = c["conv"][0]
    x = c["qkvz"][0, :DIM]
    full_v = conv_fp32(history, x, c["weight"], c["bias"])[v_feature]
    half_v = conv_wrong_half_products_probe(
        history, x, c["weight"], c["bias"])[v_feature]
    assert full_v != half_v
    assert full_v == full[chosen]
    return c, float(full_v), float(half_v)


def assert_case(actual, expected, *, precision=False):
    for name in ("conv", "z"):
        torch.testing.assert_close(actual[name].cpu(), expected[name], rtol=0,
                                   atol=0)
    for name in ("ssm", "output"):
        # Recurrence reduction order and transcendental approximations differ.
        torch.testing.assert_close(actual[name].cpu(), expected[name],
                                   rtol=0.002, atol=0.0005)
    if precision:
        # A targeted FP16 bit check, not a broad 0.006 acceptance window.
        assert actual["ssm"][0, 0, 0, 0].cpu() == expected["ssm"][0, 0, 0, 0]


def test_cpu_precision_probe_separates_fp32_and_half():
    case, full, half = make_precision_probe()
    assert abs(full - half) >= 0.003
    want = reference_step(case)
    half_boundary = reference_step(case, local_fp32=False)
    assert want["ssm"][0, 0, 0, 0] != half_boundary["ssm"][0, 0, 0, 0]
    assert torch.isfinite(want["output"]).all()
    assert torch.isfinite(want["ssm"]).all()


def test_cpu_multistep_distinct_slots_and_invalid_indices():
    case = make_case()
    for ci, si in ((1, 3), (1, 3), (-1, 3), (7, 2), (0, 99), (2, 4)):
        case["conv_idx"].fill_(ci)
        case["ssm_idx"].fill_(si)
        before = clone_case(case)
        case = reference_step(case)
        if ci < 0 or si >= 8:
            torch.testing.assert_close(case["conv"], before["conv"],
                                       rtol=0, atol=0)
            torch.testing.assert_close(case["ssm"], before["ssm"],
                                       rtol=0, atol=0)
            assert not case["output"].count_nonzero()
            assert not case["z"].count_nonzero()


@pytest.fixture(scope="module")
def root_op():
    path = os.environ.get("QWEN38_GDN_ROOT_LIBRARY")
    if not path:
        pytest.skip("await frozen candidate DSO and GPU6 authorization")
    assert Path(path).is_file(), path
    torch.ops.load_library(path)
    if torch._C._jit_get_schemas_for_operator("_qwen38_C::gdn_decode_sycl"):
        return torch.ops._qwen38_C.gdn_decode_sycl
    return torch.ops.qwen38_gdn_sycl_test.decode


def gpu_step(op, case):
    op(*decode_args(case))
    torch.xpu.synchronize()


def test_gpu_precision_probe(root_op):
    case, _, _ = make_precision_probe()
    want = reference_step(case)
    actual = clone_case(case, "xpu")
    gpu_step(root_op, actual)
    assert_case(actual, want, precision=True)


def test_gpu_multistep_m1_all_tensors(root_op):
    cpu = make_case()
    gpu = clone_case(cpu, "xpu")
    for ci, si in ((1, 3), (1, 3), (0, 2), (-1, 3), (7, 2), (0, 99), (2, 4)):
        cpu["conv_idx"].fill_(ci)
        cpu["ssm_idx"].fill_(si)
        gpu["conv_idx"].fill_(ci)
        gpu["ssm_idx"].fill_(si)
        cpu = reference_step(cpu)
        gpu_step(root_op, gpu)
        assert_case(gpu, cpu)


def _profile_call(op, args):
    with torch.profiler.profile(activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.XPU,
    ]) as prof:
        op(*args)
        torch.xpu.synchronize()
    return [e.name for e in prof.events()
            if e.device_type == torch.autograd.DeviceType.XPU]


def test_gpu_root_one_kernel_and_fallbacks(root_op):
    root = clone_case(make_case(), "xpu")
    names = _profile_call(root_op, decode_args(root))
    assert len(names) == 1, names
    assert "root" in names[0].lower(), names
    for m, hv, unaligned in ((2, HV, False), (4, HV, False),
                             (8, HV, False), (1, 6, False),
                             (1, HV, True)):
        c = clone_case(make_case(641 + m + hv, m, hv), "xpu")
        if unaligned:
            for key in ("qkvz", "ssm"):
                src = c[key]
                store = torch.empty(src.numel() + 1, dtype=src.dtype,
                                    device="xpu")
                c[key] = store[1:].view(src.shape).copy_(src)
                assert c[key].data_ptr() % 4 == 2
        expected = reference_step(clone_case(c, "cpu"), local_fp32=False)
        names = _profile_call(root_op, decode_args(c))
        assert len(names) == 2, (m, hv, unaligned, names)
        assert not any("root" in name.lower() for name in names), names
        assert_case(c, expected)


def test_gpu_spec_m2_stays_two_kernels(root_op):
    c = clone_case(make_case(773, m=2), "xpu")
    c["conv_idx"] = torch.tensor([1, 2], dtype=torch.int32, device="xpu")
    c["a_log"] = c["a_log"].float()
    tokens = torch.arange(2, dtype=torch.int32, device="xpu")
    accepted = torch.ones(1, dtype=torch.int32, device="xpu")
    if torch._C._jit_get_schemas_for_operator("_qwen38_C::gdn_spec_v2_sycl"):
        spec = torch.ops._qwen38_C.gdn_spec_v2_sycl
    else:
        spec = torch.ops.qwen38_gdn_sycl_test.spec_v2
    names = _profile_call(spec, (
        c["qkvz"], c["conv"], c["weight"], c["bias"], c["conv_idx"],
        c["a_log"], c["dt_bias"], c["ba"], c["ssm"], c["output"],
        c["z"], tokens, accepted, 1, 2, SCALE))
    assert len(names) == 2, names
    assert not any("root" in name.lower() for name in names), names


def test_gpu_alias_rejected_without_submit(root_op):
    c = clone_case(make_case(), "xpu")
    before = clone_case(c)
    args = list(decode_args(c))
    args[-2] = c["output"]  # z aliases output
    with torch.profiler.profile(activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.XPU,
    ]) as prof:
        with pytest.raises(RuntimeError):
            root_op(*args)
        torch.xpu.synchronize()
    assert not [e for e in prof.events()
                if e.device_type == torch.autograd.DeviceType.XPU]
    for key in ("conv", "ssm", "output", "z"):
        torch.testing.assert_close(c[key], before[key], rtol=0, atol=0)


def test_gpu_nondefault_stream_dropped_inputs(root_op):
    import gc

    case = make_case(957)
    want = reference_step(case)
    gpu = clone_case(case, "xpu")
    stream = torch.xpu.Stream()
    stream.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(stream):
        transient = [gpu[key].clone() for key in ("weight", "bias", "ba")]
        args = list(decode_args(gpu))
        args[2], args[3], args[7] = transient
        root_op(*args)
    del args, transient
    gc.collect()
    junk = [torch.full_like(gpu["weight"], 77) for _ in range(8)]
    stream.synchronize()
    assert_case(gpu, want)
    del junk
