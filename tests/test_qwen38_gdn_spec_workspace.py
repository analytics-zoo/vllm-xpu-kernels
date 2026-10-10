# SPDX-License-Identifier: Apache-2.0
"""Spec host-transaction integration oracle, NOT independent mathematics.

Compare GDNSpecWorkspaceDirectV1 with separate existing INT4 -> spec_v2
(FP32 A_log) -> norm_gate -> INT4 calls. Mathematical correctness belongs
to the existing GDN/INT4 golden tests. No build, server, or package import
side effects: GPU execution requires an explicit QWEN38_SYCL_LIBRARY DSO.
"""

import importlib.util
import math
import os

import pytest
import torch


@pytest.fixture(scope="module")
def native():
    # Deliberately identical candidate selection to the M1 workspace tests.
    library = os.environ.get("QWEN38_SYCL_LIBRARY")
    if not library or not torch.xpu.is_available():
        pytest.skip("requires an explicit candidate DSO and XPU")
    spec = importlib.util.spec_from_file_location("_qwen38_C", library)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _page_views(page, hv, rows):
    dim = (2 * (hv // 3) + hv) * 128
    end = rows * dim
    slots = page.shape[0]
    return (
        page[:, :end].view(slots, rows, dim),
        page[:, end : end + hv * 128 * 128].view(slots, hv, 128, 128),
    )


def _case(
    hv=6,
    m=5,
    dtype=torch.float32,
    packed=True,
    paged=True,
    bias=True,
    index_layout="padded_1d",
):
    # TP4 -> HV12, TP8 -> HV6; one speculative sequence, bounded KV slots.
    generator = torch.Generator().manual_seed(3831 + hv + m)
    device = "xpu:0"
    dim = (2 * (hv // 3) + hv) * 128

    def rand(shape, kind=torch.float16, scale=0.05):
        return (torch.randn(shape, generator=generator) * scale).to(
            device=device, dtype=kind
        )

    def weight(n, k):
        return torch.randint(
            0, 256, (n, k // 2), dtype=torch.uint8, generator=generator
        ).to(device)

    def scales(n, k):
        # Packed UINT8 weights and INT4 scales stay UINT8/FP16. Only the
        # floating GDN parameters/bias accept FP16/BF16/FP32 conversion.
        return rand((n, k // 128), scale=0.002)

    w = [
        weight(dim + hv * 128, 2560),
        scales(dim + hv * 128, 2560),
        weight(2 * hv, 2560),
        scales(2 * hv, 2560),
        weight(2560, hv * 128),
        scales(2560, hv * 128),
        rand((dim, 1, 4), dtype),
        torch.linspace(0.7002, 0.9011, hv, device=device, dtype=torch.float32),
        rand((hv,), dtype),
        rand((128,), dtype) + 1,
    ]
    rows, slots = m + 2 if packed else 3, m + 3
    page = None
    if paged:
        page = torch.full(
            (slots, rows * dim + hv * 128 * 128 + 16),
            -19,
            device=device,
            dtype=torch.float16,
        )
        conv, ssm = _page_views(page, hv, rows)
        conv.copy_(rand(conv.shape))
        ssm.copy_(rand(ssm.shape, scale=0.05))
    else:
        conv = rand((slots, rows, dim))
        ssm = rand((slots, hv, 128, 128), scale=0.05)
    ids = torch.arange(1, m + 1, dtype=torch.int32, device=device)
    if index_layout == "padded_1d":
        ids = torch.cat((ids, torch.full_like(ids, -12345)))
    elif index_layout == "padded_2d":
        ids = torch.stack((ids, torch.full_like(ids, -12345)))
    elif index_layout == "exact_2d":
        ids = ids.unsqueeze(0)
    else:
        assert index_layout == "exact_1d"
    # Reversed, nonidentity mapping: every active output row is written once.
    # -1 is legal *outside* the active prefix. An active -1 leaves a core/z
    # row unwritten, so a full-output bitwise oracle is not defined there.
    tokens = torch.cat(
        (
            torch.arange(m - 1, -1, -1, dtype=torch.int32, device=device),
            torch.tensor([-1, -12345], dtype=torch.int32, device=device),
        )
    )
    return dict(
        x=rand((m, 2560)),
        w=w,
        bias=rand((dim,), dtype) if bias else None,
        conv=conv,
        ssm=ssm,
        page=page,
        indices=ids,
        tokens=tokens,
        accepted=torch.tensor([1, -12345], dtype=torch.int32, device=device),
        hv=hv,
        eps=1e-6,
    )


def _invoke(workspace, c, sigmoid=True):
    return workspace.try_run(
        c["x"],
        c["w"],
        c["bias"],
        c["conv"],
        c["ssm"],
        c["indices"],
        c["tokens"],
        c["accepted"],
        c["hv"],
        c["eps"],
        sigmoid,
    )


def _separate_stages(c, sigmoid=True):
    # Integration oracle only: do not call the workspace or fused M1 stages.
    w, hv = c["w"], c["hv"]
    m, device = c["x"].shape[0], c["x"].device
    page = c["page"].clone() if c["page"] is not None else None
    if page is None:
        conv, ssm = c["conv"].clone(), c["ssm"].clone()
    else:
        conv, ssm = _page_views(page, hv, c["conv"].shape[1])
    qkv = torch.empty((m, w[0].shape[0]), device=device, dtype=torch.float16)
    ba = torch.empty((m, 2 * hv), device=device, dtype=torch.float16)
    core = torch.empty((m, hv, 128), device=device, dtype=torch.float16)
    z = torch.empty_like(core)
    normalized = torch.empty((m, hv * 128), device=device, dtype=torch.float16)
    output = torch.empty_like(c["x"])
    bias = (
        c["bias"].detach().half().contiguous()
        if c["bias"] is not None
        else torch.zeros(conv.shape[-1], device=device, dtype=torch.float16)
    )
    ops = torch.ops._qwen38_C
    ops.int4_linear(c["x"], w[0], w[1], qkv)
    ops.int4_linear(c["x"], w[2], w[3], ba)
    ops.gdn_spec_v2_sycl(
        qkv,
        conv,
        w[6].detach().half().contiguous().view(-1, 4),
        bias,
        c["indices"].flatten()[:m],
        w[7].detach().float().contiguous(),
        w[8].detach().half().contiguous(),
        ba,
        ssm,
        core,
        z,
        c["tokens"][:m],
        c["accepted"][:1],
        1,
        m,
        1 / math.sqrt(128),
    )
    ops.gdn_norm_gate_sycl(
        core,
        z,
        w[9].detach().half().contiguous(),
        normalized,
        c["eps"],
        sigmoid,
    )
    ops.int4_linear(normalized, w[4], w[5], output)
    return output, conv, ssm, page


def _assert_bits(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype and actual.device == expected.device
    # Numerical torch.equal alone does not distinguish +0 from -0.
    assert torch.equal(
        actual.detach().resolve_neg().contiguous().view(torch.uint8),
        expected.detach().resolve_neg().contiguous().view(torch.uint8),
    )


def _check_result(c, actual, expected):
    assert actual is not None
    assert actual.shape == c["x"].shape and actual.is_contiguous()
    for got, want in zip((actual, c["conv"], c["ssm"]), expected[:3]):
        assert torch.isfinite(got).all().item()
        _assert_bits(got, want)
    if c["page"] is not None:
        _assert_bits(c["page"], expected[3])
        assert (c["page"][:, -16:] == -19).all().item()


def _assert_equivalent(workspace, c, sigmoid=True):
    with torch.no_grad():
        a_log = c["w"][7].clone()
        expected = _separate_stages(c, sigmoid)
        actual = _invoke(workspace, c, sigmoid)
    torch.xpu.synchronize()
    _check_result(c, actual, expected)
    _assert_bits(c["w"][7], a_log)
    return actual


@pytest.mark.parametrize("hv", [12, 6], ids=["tp4", "tp8"])
@pytest.mark.parametrize("m", [2, 5, 8])
@pytest.mark.parametrize("packed", [True, False], ids=["packed", "legacy"])
def test_spec_chain_all_accepted_counts_and_padded_pages(native, hv, m, packed):
    c = _case(
        hv,
        m,
        packed=packed,
        index_layout="padded_2d" if packed else "padded_1d",
    )
    workspace = native.GDNSpecWorkspaceDirectV1()
    # Reuse weights/scratch and roll the updated checkpoints forward rather
    # than allocating a new large case for every accepted count.
    for accepted in range(1, m + 1):
        c["accepted"][0] = accepted
        _assert_equivalent(workspace, c)


@pytest.mark.parametrize("m", [3, 4, 6, 7])
def test_spec_remaining_m_and_workspace_shape_reuse(native, m):
    workspace = native.GDNSpecWorkspaceDirectV1()
    for hv, current_m in ((6, 2), (12, m), (6, 2)):
        c = _case(
            hv,
            current_m,
            packed=(m % 2 == 0),
            index_layout="exact_2d" if m % 2 else "exact_1d",
        )
        c["tokens"] = c["tokens"][:current_m]
        c["accepted"] = c["accepted"][:1]
        c["accepted"][0] = current_m
        _assert_equivalent(workspace, c)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32]
)
@pytest.mark.parametrize("sigmoid", [False, True])
def test_spec_floating_parameter_conversion_and_gate(native, dtype, sigmoid):
    c = _case(
        12 if sigmoid else 6,
        3,
        dtype=dtype,
        paged=sigmoid,
        bias=(dtype != torch.float16),
    )
    c["w"][7] = c["w"][7].to(dtype)
    workspace = native.GDNSpecWorkspaceDirectV1()
    for _ in range(2):
        _assert_equivalent(workspace, c, sigmoid)


def test_spec_fp32_a_log_is_not_rounded_through_fp16(native):
    c = _case(12, 5)
    assert c["w"][7].dtype == torch.float32
    assert not torch.equal(c["w"][7], c["w"][7].half().float())
    with torch.no_grad():
        expected = _separate_stages(c)
        rounded = dict(c, w=list(c["w"]))
        rounded["w"][7] = c["w"][7].half().float()
        wrong = _separate_stages(rounded)
        actual = _invoke(native.GDNSpecWorkspaceDirectV1(), c)
    torch.xpu.synchronize()
    # Prove this fixture is sensitive to the forbidden FP16 intermediate.
    assert not torch.equal(expected[2], wrong[2])
    _check_result(c, actual, expected)


@pytest.mark.parametrize("mutation", ["replace", "inplace", "data", "bias"])
def test_spec_live_weights_refresh(native, mutation):
    c = _case()
    for i in range(6, 10):
        c["w"][i] = torch.nn.Parameter(c["w"][i])
    c["bias"] = torch.nn.Parameter(c["bias"])
    workspace = native.GDNSpecWorkspaceDirectV1()
    _assert_equivalent(workspace, c)
    with torch.no_grad():
        if mutation == "bias":
            c["bias"] = None
        else:
            # Change projections/scales as well as all converted sources.
            for i, source in enumerate(c["w"]):
                changed = (
                    source ^ 17
                    if source.dtype == torch.uint8
                    else source + (0.001 if i < 6 else 0.125)
                )
                if mutation == "replace":
                    c["w"][i] = changed
                elif mutation == "data":
                    source.data = changed
                else:
                    source.copy_(changed)
            if mutation == "data":
                c["bias"].data = c["bias"].data + 0.1
            elif mutation == "replace":
                c["bias"] = c["bias"] + 0.1
            else:
                c["bias"].add_(0.1)
    _assert_equivalent(workspace, c)
    if mutation == "bias":
        c["bias"] = torch.zeros(c["conv"].shape[-1], device=c["x"].device)
        _assert_equivalent(workspace, c)


@pytest.mark.parametrize("inference", [False, True])
def test_spec_explicit_invalidate_for_unversioned_content_reload(
    native, inference
):
    context = torch.inference_mode() if inference else torch.no_grad()
    with context:
        c, workspace = _case(), native.GDNSpecWorkspaceDirectV1()
        _assert_equivalent(workspace, c)
        for source in c["w"][6:] + [c["bias"]]:
            # Ordinary .data in-place writes bypass the version counter too.
            if inference:
                source.add_(0.125)
            else:
                source.data.add_(0.125)
        workspace.invalidate()
        workspace.invalidate()  # idempotent, including an empty cache
        _assert_equivalent(workspace, c)


def _assert_rejected(workspace, c, extra=()):
    # Snapshot every caller-visible tensor, not just the two recurrent states.
    sources = (
        [
            c[key]
            for key in (
                "x",
                "bias",
                "conv",
                "ssm",
                "page",
                "indices",
                "tokens",
                "accepted",
            )
        ]
        + c["w"]
        + list(extra)
    )
    with torch.no_grad():
        before = [(t, t.clone()) for t in sources if t is not None]
        assert _invoke(workspace, c) is None
    torch.xpu.synchronize()
    for current, original in before:
        _assert_bits(current, original)


@pytest.mark.parametrize(
    "alias",
    [
        "raw_fp32",
        "raw_bf16",
        "conv_weight",
        "bias",
        "packed_weight",
        "scale",
        "input",
        "states",
        "indices",
        "tokens",
        "accepted",
        "scratch_input",
    ],
)
def test_spec_alias_rejection_precedes_all_mutation(native, alias):
    c, workspace = _case(), native.GDNSpecWorkspaceDirectV1()
    previous = _assert_equivalent(workspace, c)
    raw = c["page"].view(-1)
    hv, m, dim = c["hv"], c["x"].shape[0], c["conv"].shape[-1]
    if alias == "raw_fp32":
        # Conversion would make a disjoint copy; the *raw* alias must fail.
        c["w"][7] = raw[: 2 * hv].view(torch.float32)
    elif alias == "raw_bf16":
        c["w"][8] = raw[:hv].view(torch.bfloat16)
    elif alias == "conv_weight":
        c["w"][6] = raw[: dim * 4].view(dim, 1, 4)
    elif alias == "bias":
        c["bias"] = raw[: 2 * dim].view(torch.float32)
    elif alias == "packed_weight":
        c["w"][2] = raw.view(torch.uint8)[: 2 * hv * 1280].view(2 * hv, 1280)
    elif alias == "scale":
        c["w"][3] = raw[: 2 * hv * 20].view(2 * hv, 20)
    elif alias == "input":
        c["x"] = raw[: m * 2560].view(m, 2560)
    elif alias == "states":
        c["ssm"] = c["page"][:, : hv * 128 * 128].view(-1, hv, 128, 128)
    elif alias in ("indices", "tokens", "accepted"):
        count = 1 if alias == "accepted" else m
        c[alias] = raw[: 2 * count].view(torch.int32)
    else:
        c["x"] = previous  # private scratch overlaps a read on this stream
    _assert_rejected(workspace, c, extra=(previous,))


@pytest.mark.parametrize(
    "invalid",
    [
        "m1",
        "m9",
        "input_width",
        "input_dtype",
        "heads",
        "weights_count",
        "output_weight",
        "conv_weight_shape",
        "a_log_shape",
        "norm_shape",
        "bias_shape",
        "raw_dtype",
        "conv_rows",
        "ssm_shape",
        "indices_short",
        "indices_2d_width",
        "indices_empty_rows",
        "indices_rank",
        "tokens_short",
        "tokens_rank",
        "accepted_empty",
        "accepted_rank",
        "indices_dtype",
        "tokens_dtype",
        "accepted_dtype",
        "indices_stride",
        "tokens_stride",
        "accepted_stride",
        "lazy_weight",
        "eps_zero",
        "eps_nan",
        "eps_underflow",
    ],
)
def test_spec_metadata_rejection_does_not_mutate(native, invalid):
    c = _case(m=1 if invalid == "m1" else 9 if invalid == "m9" else 2)
    m, device = c["x"].shape[0], c["x"].device
    if invalid == "input_width":
        c["x"] = c["x"][:, :-1].contiguous()
    elif invalid == "input_dtype":
        c["x"] = c["x"].bfloat16()
    elif invalid == "heads":
        c["hv"] = 3
    elif invalid == "weights_count":
        c["w"] = c["w"][:-1]
    elif invalid == "output_weight":
        c["w"][4] = c["w"][4][:, :-1].contiguous()
    elif invalid == "conv_weight_shape":
        c["w"][6] = c["w"][6].squeeze(1)
    elif invalid == "a_log_shape":
        c["w"][7] = c["w"][7].unsqueeze(0)
    elif invalid == "norm_shape":
        c["w"][9] = c["w"][9][:-1]
    elif invalid == "bias_shape":
        c["bias"] = c["bias"][:-1]
    elif invalid == "raw_dtype":
        c["w"][6] = c["w"][6].double()
    elif invalid == "conv_rows":
        c["conv"] = c["conv"][:, :2]  # 3 is a legal legacy layout!
    elif invalid == "ssm_shape":
        c["ssm"] = c["ssm"][:, :, :, :-1]
    elif invalid == "indices_short":
        c["indices"] = c["indices"][: m - 1]
    elif invalid == "indices_2d_width":
        c["indices"] = torch.ones((1, m + 1), device=device, dtype=torch.int32)
    elif invalid == "indices_empty_rows":
        c["indices"] = torch.empty((0, m), device=device, dtype=torch.int32)
    elif invalid == "indices_rank":
        c["indices"] = c["indices"][:m].view(1, 1, m)
    elif invalid == "tokens_short":
        c["tokens"] = c["tokens"][: m - 1]
    elif invalid == "tokens_rank":
        c["tokens"] = c["tokens"][:m].view(1, m)
    elif invalid == "accepted_empty":
        c["accepted"] = c["accepted"][:0]
    elif invalid == "accepted_rank":
        c["accepted"] = c["accepted"].view(1, -1)
    elif invalid.endswith("_dtype"):
        key = invalid.removesuffix("_dtype")
        c[key] = c[key].long()
    elif invalid.endswith("_stride"):
        key = invalid.removesuffix("_stride")
        c[key] = torch.ones((max(m, 2), 2), device=device, dtype=torch.int32)[
            :, 0
        ]
    elif invalid == "lazy_weight":
        c["w"][9] = torch._neg_view(c["w"][9])
    elif invalid.startswith("eps_"):
        c["eps"] = {
            "eps_zero": 0,
            "eps_nan": float("nan"),
            "eps_underflow": 1e-100,
        }[invalid]
    _assert_rejected(native.GDNSpecWorkspaceDirectV1(), c)


@pytest.mark.parametrize("slot", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_spec_projection_weights_and_scales_do_not_silently_convert(
    native, slot, dtype
):
    c = _case(m=2)
    c["w"][slot] = c["w"][slot].to(dtype)
    _assert_rejected(native.GDNSpecWorkspaceDirectV1(), c)


@pytest.mark.parametrize(
    "target",
    [
        "x",
        "conv",
        "ssm",
        "bias",
        "indices",
        "tokens",
        "accepted",
        "packed_weight",
        "scale",
        "raw_weight",
    ],
)
def test_spec_cpu_device_mismatch_rejected_before_mutation(native, target):
    c = _case(m=2)
    if target in ("packed_weight", "scale", "raw_weight"):
        slot = {"packed_weight": 0, "scale": 1, "raw_weight": 7}[target]
        c["w"][slot] = c["w"][slot].cpu()
    else:
        c[target] = c[target].cpu()
    _assert_rejected(native.GDNSpecWorkspaceDirectV1(), c)


def test_spec_grad_mode_rejected_before_mutation(native):
    c = _case(m=2)
    workspace = native.GDNSpecWorkspaceDirectV1()
    with torch.no_grad():
        before = (c["conv"].clone(), c["ssm"].clone())
    with torch.enable_grad():
        assert _invoke(workspace, c) is None
    torch.xpu.synchronize()
    _assert_bits(c["conv"], before[0])
    _assert_bits(c["ssm"], before[1])


def test_spec_output_metadata_is_not_private_scratch_metadata(native):
    c, workspace = _case(), native.GDNSpecWorkspaceDirectV1()
    output = _assert_equivalent(workspace, c)
    pointer = output.data_ptr()
    output.transpose_(0, 1)
    again = _assert_equivalent(workspace, c)
    assert again.data_ptr() == pointer
    assert output.shape == (2560, 5) and again.shape == (5, 2560)
    output.resize_(5 * 2560)
    assert _assert_equivalent(workspace, c).shape == (5, 2560)


def test_spec_partial_scratch_allocation_failure_can_retry(native):
    from torch.utils._python_dispatch import TorchDispatchMode

    c, workspace = _case(), native.GDNSpecWorkspaceDirectV1()
    before = (c["conv"].clone(), c["ssm"].clone())

    class FailNormalizedAllocation(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func is torch.ops.aten.empty.memory_format and tuple(
                args[0]
            ) == (5, c["hv"] * 128):
                raise RuntimeError("injected normalized allocation failure")
            return func(*args, **(kwargs or {}))

    # Fault injection at the private ABI, not a claim that the Python adapter
    # accepts modes. All scratch allocation precedes state mutation.
    with (
        torch.no_grad(),
        FailNormalizedAllocation(),
        pytest.raises(RuntimeError, match="injected normalized allocation"),
    ):
        _invoke(workspace, c)
    torch.xpu.synchronize()
    _assert_bits(c["conv"], before[0])
    _assert_bits(c["ssm"], before[1])
    _assert_equivalent(workspace, c)


def test_spec_stream_isolation_and_early_workspace_and_source_release(native):
    # Same HV/M on both streams: a shape change must not mask scratch sharing.
    cases = [_case(6, 5), _case(6, 5)]
    with torch.no_grad():
        cases[1]["x"].mul_(1.5)
        expected = [_separate_stages(c) for c in cases]
        producer = torch.xpu.current_stream()
        streams = [torch.xpu.Stream(), torch.xpu.Stream()]
        workspace = native.GDNSpecWorkspaceDirectV1()
        outputs = []
        for stream, c in zip(streams, cases):
            stream.wait_stream(producer)
            with torch.xpu.stream(stream):
                outputs.append(_invoke(workspace, c))
        assert all(out is not None for out in outputs)
        assert outputs[0].data_ptr() != outputs[1].data_ptr()
        # Keep only observable results/states, dropping every source before
        # synchronization. Kernel record_stream must retain source lifetimes.
        observed = [
            dict(
                c,
                w=[],
                x=out,
                bias=None,
                indices=None,
                tokens=None,
                accepted=None,
            )
            for c, out in zip(cases, outputs)
        ]
        workspace.invalidate()
        del workspace, cases, c
        # Bounded allocator pressure after early release, on the producer.
        pressure = [
            torch.full((5, 2560), 7, device="xpu", dtype=torch.float16)
            for _ in range(4)
        ]
    torch.xpu.synchronize()
    assert pressure[0][0, 0].item() == 7
    for c, out, want in zip(observed, outputs, expected):
        _check_result(c, out, want)
