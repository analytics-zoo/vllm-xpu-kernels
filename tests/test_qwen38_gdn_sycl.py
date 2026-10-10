# SPDX-License-Identifier: Apache-2.0
"""Independent mathematical reference for Qwen3.8 TP4/TP8 GDN SYCL.

The standalone DSO is built only from csrc/qwen38/gdn_sycl.cpp with
QWEN38_GDN_STANDALONE_TEST. Set QWEN38_GDN_SYCL_LIBRARY to run XPU tests.
No ESIMD implementation is used as the mathematical oracle.
"""

import gc
import math
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


def _case(h: int, hv: int, m: int, spec: bool, packed: bool = True):
    torch.manual_seed(123 + m + hv)
    dim = (2 * h + hv) * 128
    slots = max(m + 3, 6)
    return {
        "qkvz": (torch.randn(m, dim + hv * 128) * 0.08).half(),
        "conv": (
            torch.randn(slots, m + 2 if spec and packed else 3, dim) * 0.03
        ).half(),
        "weight": (torch.randn(dim, 4) * 0.1).half(),
        "bias": (torch.randn(dim) * 0.01).half(),
        "a_log": (torch.randn(hv) * 0.05 + 0.3).float()
        if spec
        else (torch.randn(hv) * 0.05 + 0.3).half(),
        "dt_bias": (torch.randn(hv) * 0.03).half(),
        "ba": (torch.randn(m, 2 * hv) * 0.08).half(),
        "ssm": (torch.randn(slots, hv, 128, 128) * 0.002).half(),
        "output": torch.empty(m, hv, 128, dtype=torch.float16),
        "z": torch.empty(m, hv, 128, dtype=torch.float16),
        "idx": torch.arange(1, m + 1, dtype=torch.int32)
        if spec
        else torch.arange(m, dtype=torch.int32),
        "token_idx": torch.arange(m - 1, -1, -1, dtype=torch.int32),
        "accepted": torch.tensor([min(m, 2)], dtype=torch.int32),
        "h": h,
        "hv": hv,
        "scale": 1 / math.sqrt(128),
    }


def _reference(case: dict, spec: bool):
    h, hv = case["h"], case["hv"]
    dim = (2 * h + hv) * 128
    m = case["qkvz"].shape[0]
    conv = case["conv"].clone()
    state = case["ssm"].clone()
    output = torch.zeros_like(case["output"])
    z = torch.zeros_like(case["z"])
    idx = case["idx"].tolist()
    order = case["token_idx"].tolist() if spec else list(range(m))
    accepted = int(case["accepted"][0])
    packed = spec and conv.shape[1] != 3
    initial_conv_idx = idx[0] if packed else idx[accepted - 1] if spec else None
    initial_ssm_idx = idx[accepted - 1] if spec else None
    history = None
    if spec and initial_conv_idx > 0 and initial_ssm_idx > 0:
        start = accepted - 1 if packed else 0
        history = conv[initial_conv_idx, start : start + 3].float().clone()
        carry = state[initial_ssm_idx].float().clone()
        if packed:
            conv[initial_conv_idx, 0:2] = history[1:3].half()
    for t, token in enumerate(order):
        save = idx[t]
        if not spec:
            initial_conv_idx = initial_ssm_idx = save
            history = conv[save].float().clone()
            carry = state[save].float().clone()
        if history is None:
            continue
        x = case["qkvz"][token, :dim].float()
        terms = torch.stack((history[0], history[1], history[2], x))
        products = terms * case["weight"].T.float()
        if spec:
            products = products.half().float()
        conv_result = case["bias"].float() + products.sum(dim=0)
        conv_result = F.silu(conv_result).half().float()
        q = conv_result[: h * 128].reshape(h, 128)
        k = conv_result[h * 128 : 2 * h * 128].reshape(h, 128)
        v = conv_result[2 * h * 128 :].reshape(hv, 128)
        if not spec or save > 0:
            z[token] = case["qkvz"][token, dim:].reshape(hv, 128)
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        q = q * case["scale"]
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        for head in range(hv):
            key_head = head // (hv // h)
            b = case["ba"][token, head].float()
            a = case["ba"][token, hv + head].float()
            decay = torch.exp(
                -torch.exp(case["a_log"][head].float())
                * F.softplus(a + case["dt_bias"][head].float())
            )
            beta = torch.sigmoid(b)
            carry[head] *= decay
            dot = (carry[head] * k[key_head]).sum(-1)
            carry[head] += ((v[head] - dot) * beta)[:, None] * k[key_head]
            if not spec or save > 0:
                output[token, head] = (carry[head] * q[key_head]).sum(-1).half()
        history = torch.stack((history[1], history[2], x))
        if spec:
            conv_slot = initial_conv_idx if packed else save
            if not packed and save > 0:
                conv[conv_slot] = history.half()
            elif packed:
                conv[conv_slot, 2 + t] = x.half()
            if save > 0:
                state[save] = carry.half()
        else:
            conv[save] = history.half()
            state[save] = carry.half()
    return output, z, conv, state


@pytest.fixture(scope="module")
def gdn_ops():
    integrated = os.environ.get("QWEN38_SYCL_LIBRARY")
    if integrated:
        torch.ops.load_library(integrated)
        native = torch.ops._qwen38_C
        return SimpleNamespace(
            decode=native.gdn_decode_sycl,
            spec_v2=native.gdn_spec_v2_sycl,
            norm_gate=native.gdn_norm_gate_sycl,
        )
    library = os.environ.get("QWEN38_GDN_SYCL_LIBRARY")
    if not library:
        pytest.skip("set QWEN38_GDN_SYCL_LIBRARY after GPU access is released")
    torch.ops.load_library(library)
    return torch.ops.qwen38_gdn_sycl_test


def _xpu(case: dict):
    return {
        k: v.xpu() if isinstance(v, torch.Tensor) else v
        for k, v in case.items()
    }


def test_cpu_reference_packed_rollback_contract():
    case = _case(4, 12, 4, spec=True, packed=True)
    before = case["conv"].clone()
    result, z, conv, state = _reference(case, spec=True)
    initial = int(case["idx"][0])
    col = int(case["accepted"][0]) - 1
    torch.testing.assert_close(
        conv[initial, :2], before[initial, col + 1 : col + 3], rtol=0, atol=0
    )
    torch.testing.assert_close(
        conv[initial, 2:6],
        case["qkvz"][:, : (2 * 4 + 12) * 128].flip(0),
        rtol=0,
        atol=0,
    )
    assert torch.isfinite(result).all()
    assert torch.isfinite(z).all()
    assert torch.isfinite(state).all()


def test_cpu_reference_null_block_is_zero():
    case = _case(4, 12, 4, spec=True, packed=True)
    case["idx"][0] = 0
    case["accepted"][0] = 1
    result, z, conv, state = _reference(case, spec=True)
    assert torch.count_nonzero(result) == 0
    assert torch.count_nonzero(z) == 0
    torch.testing.assert_close(conv, case["conv"], rtol=0, atol=0)
    torch.testing.assert_close(state, case["ssm"], rtol=0, atol=0)


@pytest.mark.parametrize("h,hv", [(4, 12), (2, 6)])
@pytest.mark.parametrize("m", range(1, 9))
def test_decode_golden(gdn_ops, h, hv, m):
    cpu = _case(h, hv, m, spec=False)
    expected = _reference(cpu, spec=False)
    gpu = _xpu(cpu)
    gdn_ops.decode(
        gpu["qkvz"],
        gpu["conv"],
        gpu["weight"],
        gpu["bias"],
        gpu["idx"],
        gpu["a_log"],
        gpu["dt_bias"],
        gpu["ba"],
        gpu["ssm"],
        gpu["idx"],
        gpu["output"],
        gpu["z"],
        gpu["scale"],
    )
    for key, want in zip(("output", "z", "conv", "ssm"), expected):
        torch.testing.assert_close(gpu[key].cpu(), want, atol=0.006, rtol=0.006)


def test_decode_unaligned_storage_offset(gdn_ops):
    cpu = _case(4, 12, 1, spec=False)
    expected = _reference(cpu, spec=False)
    gpu = _xpu(cpu)
    for key in ("qkvz", "ssm"):
        source = gpu[key]
        storage = torch.empty(
            source.numel() + 1, dtype=source.dtype, device=source.device
        )
        view = storage[1:].view(source.shape)
        view.copy_(source)
        assert view.data_ptr() % 4 == 2
        gpu[key] = view
    gdn_ops.decode(
        gpu["qkvz"],
        gpu["conv"],
        gpu["weight"],
        gpu["bias"],
        gpu["idx"],
        gpu["a_log"],
        gpu["dt_bias"],
        gpu["ba"],
        gpu["ssm"],
        gpu["idx"],
        gpu["output"],
        gpu["z"],
        gpu["scale"],
    )
    for key, want in zip(("output", "z", "conv", "ssm"), expected):
        torch.testing.assert_close(gpu[key].cpu(), want, atol=0.006, rtol=0.006)


@pytest.mark.parametrize("h,hv", [(4, 12), (2, 6)])
def test_decode_full_graph_updates_current_slot_and_inputs(gdn_ops, h, hv):
    cpu = _case(h, hv, 1, spec=False)
    gpu = _xpu(cpu)
    gpu["idx"].fill_(-1)

    def run():
        gdn_ops.decode(
            gpu["qkvz"], gpu["conv"], gpu["weight"], gpu["bias"],
            gpu["idx"], gpu["a_log"], gpu["dt_bias"], gpu["ba"],
            gpu["ssm"], gpu["idx"], gpu["output"], gpu["z"], gpu["scale"],
        )

    run()
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        run()
    # 捕获期间只用无效 slot，不污染任何真实 recurrent 历史。
    torch.testing.assert_close(gpu["conv"].cpu(), cpu["conv"], atol=0, rtol=0)
    torch.testing.assert_close(gpu["ssm"].cpu(), cpu["ssm"], atol=0, rtol=0)
    for slot in (2, 5, 2):
        cpu["idx"].fill_(slot)
        cpu["qkvz"].add_(0.01)
        gpu["idx"].copy_(cpu["idx"])
        gpu["qkvz"].copy_(cpu["qkvz"])
        expected = _reference(cpu, spec=False)
        graph.replay()
        for key, want in zip(("output", "z", "conv", "ssm"), expected):
            torch.testing.assert_close(gpu[key].cpu(), want,
                                       atol=0.006, rtol=0.006)
        cpu["conv"], cpu["ssm"] = expected[2:]


@pytest.mark.parametrize("h,hv", [(4, 12), (2, 6)])
@pytest.mark.parametrize("m", range(2, 9))
@pytest.mark.parametrize("packed", [True, False])
def test_spec_v2_golden(gdn_ops, h, hv, m, packed):
    cpu = _case(h, hv, m, spec=True, packed=packed)
    want = _reference(cpu, spec=True)
    gpu = _xpu(cpu)
    gdn_ops.spec_v2(
        gpu["qkvz"],
        gpu["conv"],
        gpu["weight"],
        gpu["bias"],
        gpu["idx"],
        gpu["a_log"],
        gpu["dt_bias"],
        gpu["ba"],
        gpu["ssm"],
        gpu["output"],
        gpu["z"],
        gpu["token_idx"],
        gpu["accepted"],
        1,
        m,
        gpu["scale"],
    )
    for key, expected in zip(("output", "z", "conv", "ssm"), want):
        torch.testing.assert_close(
            gpu[key].cpu(), expected, atol=0.008, rtol=0.008
        )


@pytest.mark.parametrize("spec", (False, True))
@pytest.mark.parametrize("separate_storage", (False, True))
def test_shared_kv_pages_disjoint_conv_and_ssm(gdn_ops, spec, separate_storage):
    cpu = _case(4, 12, 4 if spec else 2, spec=spec, packed=spec)
    expected = _reference(cpu, spec=spec)
    gpu = _xpu(cpu)
    rows = gpu["conv"].shape[0]
    conv_width = gpu["conv"][0].numel()
    ssm_width = gpu["ssm"][0].numel()
    backing = torch.full((rows, conv_width + ssm_width + 8), -19,
                         dtype=torch.float16, device="xpu")
    gpu["conv"] = backing[:, :conv_width].view_as(gpu["conv"])
    gpu["ssm"] = backing[:, conv_width:conv_width + ssm_width].view_as(
        gpu["ssm"])
    if separate_storage:
        gpu["conv"] = torch.from_dlpack(gpu["conv"])
        gpu["ssm"] = torch.from_dlpack(gpu["ssm"])
        assert not torch._C._overlaps(gpu["conv"], gpu["ssm"])
    gpu["conv"].copy_(cpu["conv"])
    gpu["ssm"].copy_(cpu["ssm"])
    common = [gpu[name] for name in (
        "qkvz", "conv", "weight", "bias", "idx", "a_log", "dt_bias", "ba",
        "ssm")]
    if spec:
        gdn_ops.spec_v2(*common, gpu["output"], gpu["z"], gpu["token_idx"],
                        gpu["accepted"], 1, 4, gpu["scale"])
    else:
        gdn_ops.decode(*common, gpu["idx"], gpu["output"], gpu["z"],
                       gpu["scale"])
    for key, want in zip(("output", "z", "conv", "ssm"), expected):
        torch.testing.assert_close(gpu[key].cpu(), want, atol=0.008, rtol=0.008)
    assert torch.all(backing[:, -8:] == -19).item()


def test_spec_null_block_outputs_zero(gdn_ops):
    case = _case(4, 12, 4, spec=True)
    case["idx"][0] = 0
    case["accepted"][0] = 1
    gpu = _xpu(case)
    gpu["output"].fill_(7)
    gpu["z"].fill_(7)
    gdn_ops.spec_v2(
        gpu["qkvz"],
        gpu["conv"],
        gpu["weight"],
        gpu["bias"],
        gpu["idx"],
        gpu["a_log"],
        gpu["dt_bias"],
        gpu["ba"],
        gpu["ssm"],
        gpu["output"],
        gpu["z"],
        gpu["token_idx"],
        gpu["accepted"],
        1,
        4,
        gpu["scale"],
    )
    assert torch.count_nonzero(gpu["output"]).item() == 0
    assert torch.count_nonzero(gpu["z"]).item() == 0


def test_spec_null_save_keeps_fp32_carry(gdn_ops):
    case = _case(4, 12, 4, spec=True)
    case["accepted"][0] = 1
    case["idx"][1] = 0
    expected = _reference(case, spec=True)
    gpu = _xpu(case)
    gdn_ops.spec_v2(
        gpu["qkvz"],
        gpu["conv"],
        gpu["weight"],
        gpu["bias"],
        gpu["idx"],
        gpu["a_log"],
        gpu["dt_bias"],
        gpu["ba"],
        gpu["ssm"],
        gpu["output"],
        gpu["z"],
        gpu["token_idx"],
        gpu["accepted"],
        1,
        4,
        gpu["scale"],
    )
    for key, want in zip(("output", "z", "conv", "ssm"), expected):
        torch.testing.assert_close(gpu[key].cpu(), want, atol=0.008, rtol=0.008)


def test_spec_two_sequences_match_independent_calls(gdn_ops):
    cpu = _case(4, 12, 4, spec=True, packed=True)
    cpu["token_idx"] = torch.arange(4, dtype=torch.int32)
    cpu["accepted"] = torch.tensor([1, 1], dtype=torch.int32)
    together = _xpu(cpu)
    separate = _xpu(cpu)
    gdn_ops.spec_v2(
        together["qkvz"],
        together["conv"],
        together["weight"],
        together["bias"],
        together["idx"],
        together["a_log"],
        together["dt_bias"],
        together["ba"],
        together["ssm"],
        together["output"],
        together["z"],
        together["token_idx"],
        together["accepted"],
        2,
        2,
        together["scale"],
    )
    for seq in range(2):
        rows = slice(seq * 2, seq * 2 + 2)
        gdn_ops.spec_v2(
            separate["qkvz"][rows],
            separate["conv"],
            separate["weight"],
            separate["bias"],
            separate["idx"][rows],
            separate["a_log"],
            separate["dt_bias"],
            separate["ba"][rows],
            separate["ssm"],
            separate["output"][rows],
            separate["z"][rows],
            torch.arange(2, dtype=torch.int32, device="xpu"),
            separate["accepted"][seq : seq + 1],
            1,
            2,
            separate["scale"],
        )
    for key in ("output", "z", "conv", "ssm"):
        torch.testing.assert_close(together[key], separate[key], rtol=0, atol=0)


def test_spec_packed_multistep_rollback(gdn_ops):
    cpu = _case(4, 12, 4, spec=True, packed=True)
    gpu = _xpu(cpu)
    for accepted in (1, 4, 2, 1, 3, 4):
        cpu["accepted"][0] = accepted
        gpu["accepted"].fill_(accepted)
        expected = _reference(cpu, spec=True)
        gdn_ops.spec_v2(
            gpu["qkvz"],
            gpu["conv"],
            gpu["weight"],
            gpu["bias"],
            gpu["idx"],
            gpu["a_log"],
            gpu["dt_bias"],
            gpu["ba"],
            gpu["ssm"],
            gpu["output"],
            gpu["z"],
            gpu["token_idx"],
            gpu["accepted"],
            1,
            4,
            gpu["scale"],
        )
        for key, want in zip(("output", "z", "conv", "ssm"), expected):
            torch.testing.assert_close(
                gpu[key].cpu(), want, atol=0.008, rtol=0.008
            )
        cpu["conv"], cpu["ssm"] = expected[2], expected[3]


def test_spec_rejects_fp16_a_log_without_state_mutation(gdn_ops):
    gpu = _xpu(_case(4, 12, 4, spec=True))
    before_conv = gpu["conv"].clone()
    before_ssm = gpu["ssm"].clone()
    with pytest.raises(RuntimeError, match="A_log"):
        gdn_ops.spec_v2(
            gpu["qkvz"],
            gpu["conv"],
            gpu["weight"],
            gpu["bias"],
            gpu["idx"],
            gpu["a_log"].half(),
            gpu["dt_bias"],
            gpu["ba"],
            gpu["ssm"],
            gpu["output"],
            gpu["z"],
            gpu["token_idx"],
            gpu["accepted"],
            1,
            4,
            gpu["scale"],
        )
    torch.testing.assert_close(gpu["conv"], before_conv, rtol=0, atol=0)
    torch.testing.assert_close(gpu["ssm"], before_ssm, rtol=0, atol=0)


def test_decode_rejects_output_alias_before_submit(gdn_ops):
    gpu = _xpu(_case(4, 12, 1, spec=False))
    before_conv = gpu["conv"].clone()
    before_ssm = gpu["ssm"].clone()
    with pytest.raises(RuntimeError):
        gdn_ops.decode(
            gpu["qkvz"],
            gpu["conv"],
            gpu["weight"],
            gpu["bias"],
            gpu["idx"],
            gpu["a_log"],
            gpu["dt_bias"],
            gpu["ba"],
            gpu["ssm"],
            gpu["idx"],
            gpu["output"],
            gpu["output"],
            gpu["scale"],
        )
    torch.testing.assert_close(gpu["conv"], before_conv, rtol=0, atol=0)
    torch.testing.assert_close(gpu["ssm"], before_ssm, rtol=0, atol=0)


@pytest.mark.parametrize("invalid", ["padded_alias", "scale_overflow",
                                     "scale_underflow"])
def test_decode_review_preflight_preserves_state(gdn_ops, invalid):
    gpu = _xpu(_case(2, 6, 2, spec=False))
    if invalid == "padded_alias":
        m, width = gpu["qkvz"].shape
        base = torch.empty(m, width + 1, dtype=torch.float16, device="xpu")
        base[:, :width].copy_(gpu["qkvz"])
        gpu["qkvz"] = base[:, :width]
        gpu["output"] = base.flatten()[:m * 6 * 128].view(m, 6, 128)
    else:
        gpu["scale"] = 1e300 if invalid == "scale_overflow" else 1e-300
    gpu["output"].fill_(7)
    before_conv, before_ssm = gpu["conv"].clone(), gpu["ssm"].clone()
    before_output = gpu["output"].clone()
    with pytest.raises(RuntimeError, match="overlap|FP32 conversion"):
        gdn_ops.decode(
            gpu["qkvz"], gpu["conv"], gpu["weight"], gpu["bias"],
            gpu["idx"], gpu["a_log"], gpu["dt_bias"], gpu["ba"],
            gpu["ssm"], gpu["idx"], gpu["output"], gpu["z"], gpu["scale"])
    torch.testing.assert_close(gpu["conv"], before_conv, rtol=0, atol=0)
    torch.testing.assert_close(gpu["ssm"], before_ssm, rtol=0, atol=0)
    torch.testing.assert_close(gpu["output"], before_output, rtol=0, atol=0)


def test_decode_nondefault_stream_keeps_dropped_weight(gdn_ops):
    cpu = _case(4, 12, 1, spec=False)
    expected = _reference(cpu, spec=False)
    gpu = _xpu(cpu)
    stream = torch.xpu.Stream()
    with torch.xpu.stream(stream):
        transient_weight = gpu["weight"].clone()
        gdn_ops.decode(
            gpu["qkvz"],
            gpu["conv"],
            transient_weight,
            gpu["bias"],
            gpu["idx"],
            gpu["a_log"],
            gpu["dt_bias"],
            gpu["ba"],
            gpu["ssm"],
            gpu["idx"],
            gpu["output"],
            gpu["z"],
            gpu["scale"],
        )
        del transient_weight
        junk = [torch.empty_like(gpu["weight"]) for _ in range(8)]
    stream.synchronize()
    del junk
    for key, want in zip(("output", "z", "conv", "ssm"), expected):
        torch.testing.assert_close(gpu[key].cpu(), want, atol=0.006, rtol=0.006)


@pytest.mark.parametrize("sigmoid", [False, True])
@pytest.mark.parametrize("m", [1, 4])
def test_gated_norm_golden(gdn_ops, sigmoid, m):
    torch.manual_seed(42 + m)
    x = torch.randn(m, 12, 128, dtype=torch.float16)
    z = torch.randn_like(x)
    weight = torch.randn(128, dtype=torch.float16)
    out = torch.empty(m, 12 * 128, dtype=torch.float16, device="xpu")
    gdn_ops.norm_gate(x.xpu(), z.xpu(), weight.xpu(), out, 1e-6, sigmoid)
    inv = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
    gate = torch.sigmoid(z.float())
    if not sigmoid:
        gate *= z.float()
    want = (x.float() * inv * weight.float() * gate).half().reshape_as(out)
    torch.testing.assert_close(out.cpu(), want, atol=0.002, rtol=0.002)


@pytest.fixture(scope="module")
def gdn_projection_ops():
    integrated = os.environ.get("QWEN38_SYCL_LIBRARY")
    if integrated:
        torch.ops.load_library(integrated)
        return SimpleNamespace(norm_int4=torch.ops._qwen38_C.gdn_norm_int4_sycl)
    library = os.environ.get("QWEN38_GDN_PROJECTION_LIBRARY")
    if not library:
        pytest.skip("set QWEN38_GDN_PROJECTION_LIBRARY for sidecar tests")
    torch.ops.load_library(library)
    return torch.ops.qwen38_gdn_projection_test


def _projection_case():
    torch.manual_seed(419)
    x = (torch.randn(12, 128, device="xpu") * 0.08).half()
    z = (torch.randn(12, 128, device="xpu") * 0.08).half()
    norm = torch.ones(128, dtype=torch.float16, device="xpu")
    weight = torch.randint(256, (256, 768), dtype=torch.uint8, device="xpu")
    scale = (torch.randn(256, 12, device="xpu") * 0.01).half()
    output = torch.empty(1, 256, dtype=torch.float16, device="xpu")
    return x, z, norm, weight, scale, output


def _projection_reference(x, z, norm, weight, scale, sigmoid):
    x, z, norm, weight, scale = [t.cpu() for t in (x, z, norm, weight, scale)]
    inv = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
    gate = torch.sigmoid(z.float())
    if not sigmoid:
        gate *= z.float()
    normalized = (x.float() * inv * norm.float() * gate).half().reshape(1, -1)
    unpacked = torch.empty(weight.shape[0], weight.shape[1] * 2)
    unpacked[:, 0::2] = (weight & 15).float() - 8
    unpacked[:, 1::2] = (weight >> 4).float() - 8
    unpacked *= scale.float().repeat_interleave(128, dim=1)
    return (normalized.float() @ unpacked.T).half()


@pytest.mark.parametrize("sigmoid", [False, True])
def test_norm_int4_sidecar_golden(gdn_projection_ops, sigmoid):
    tensors = _projection_case()
    expected = _projection_reference(*tensors[:5], sigmoid)
    gdn_projection_ops.norm_int4(*tensors, 12, 128, 1e-6, sigmoid)
    torch.testing.assert_close(
        tensors[5].cpu(), expected, atol=0.006, rtol=0.006
    )


def test_norm_int4_sidecar_rejects_before_submit(gdn_projection_ops):
    x, z, norm, weight, scale, output = _projection_case()
    output.fill_(7)
    with pytest.raises(RuntimeError, match="shapes"):
        gdn_projection_ops.norm_int4(
            x,
            z,
            norm,
            weight,
            scale[:, :-1].contiguous(),
            output,
            12,
            128,
            1e-6,
            False,
        )
    torch.testing.assert_close(
        output, torch.full_like(output, 7), atol=0, rtol=0
    )


def test_norm_int4_sidecar_nondefault_stream_drop(gdn_projection_ops):
    x, z, norm, weight, scale, output = _projection_case()
    expected = _projection_reference(
        (x.cpu() * 2).half(), z, norm, weight, scale, False
    )
    stream = torch.xpu.Stream()
    stream.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(stream):
        x.mul_(2)
        gdn_projection_ops.norm_int4(
            x, z, norm, weight, scale, output, 12, 128, 1e-6, False
        )
    del x, z, norm, weight, scale
    gc.collect()
    junk = [
        torch.full((12, 128), 77, device="xpu", dtype=torch.float16)
        for _ in range(8)
    ]
    stream.synchronize()
    torch.testing.assert_close(output.cpu(), expected, atol=0.006, rtol=0.006)
    del junk
