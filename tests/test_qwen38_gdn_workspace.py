# SPDX-License-Identifier: Apache-2.0
"""Host-transaction equivalence to the separately validated SYCL stages."""

import importlib.util
import math
import os

import pytest
import torch


@pytest.fixture(scope="module")
def native():
    library = os.environ.get("QWEN38_SYCL_LIBRARY")
    if not library or not torch.xpu.is_available():
        pytest.skip("requires an explicit candidate DSO and XPU")
    spec = importlib.util.spec_from_file_location("_qwen38_C", library)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_case(hv=12, dtype=torch.float32, bias=True, paged=True):
    torch.manual_seed(3810)
    device = "xpu:0"
    h = hv // 3
    dim = (2 * h + hv) * 128

    def rand(shape, kind=torch.float16, scale=0.05):
        return (torch.randn(shape) * scale).to(device=device, dtype=kind)

    def packed(n, k):
        return torch.randint(0, 256, (n, k // 2), dtype=torch.uint8).to(device)

    def scales(n, k):
        return rand((n, k // 128), scale=0.002)

    w = [
        packed(dim + hv * 128, 2560),
        scales(dim + hv * 128, 2560),
        packed(2 * hv, 2560),
        scales(2 * hv, 2560),
        packed(2560, hv * 128),
        scales(2560, hv * 128),
        rand((dim, 1, 4), dtype),
        rand((hv,), torch.float32),
        rand((hv,), dtype),
        rand((128,), dtype) + 1,
    ]
    if paged:
        page = torch.empty(
            (4, 3 * dim + hv * 128 * 128 + 64),
            device=device,
            dtype=torch.float16,
        )
        conv = page[:, : 3 * dim].view(4, 3, dim)
        ssm = page[:, 3 * dim : 3 * dim + hv * 128 * 128].view(4, hv, 128, 128)
        conv.copy_(rand(conv.shape))
        ssm.copy_(rand(ssm.shape, scale=0.002))
    else:
        conv, ssm = rand((4, 3, dim)), rand((4, hv, 128, 128), scale=0.002)
    return dict(
        x=rand((1, 2560)),
        w=w,
        bias=rand((dim,), dtype) if bias else None,
        conv=conv,
        ssm=ssm,
        indices=torch.tensor([2], dtype=torch.int32, device=device),
        hv=hv,
    )


def invoke(workspace, c, sigmoid=True):
    return workspace.try_run(
        c["x"],
        c["w"],
        c["bias"],
        c["conv"],
        c["ssm"],
        c["indices"],
        c["hv"],
        1e-6,
        sigmoid,
    )


def separate_stages(c, sigmoid=True):
    # This is an integration oracle, not an independent mathematical oracle.
    # Existing GDN/INT4 tests cover those kernels against independent math.
    w, hv = c["w"], c["hv"]
    conv, ssm = c["conv"].clone(), c["ssm"].clone()
    qkv = torch.empty(
        (1, w[0].shape[0]), dtype=torch.float16, device=c["x"].device
    )
    ba = torch.empty((1, 2 * hv), dtype=torch.float16, device=c["x"].device)
    core = torch.empty((1, hv, 128), dtype=torch.float16, device=c["x"].device)
    z = torch.empty_like(core)
    output = torch.empty_like(c["x"])
    torch.ops._qwen38_C.int4_linear_fused2(
        c["x"], w[0], w[1], qkv, w[2], w[3], ba
    )
    bias = (
        c["bias"].half()
        if c["bias"] is not None
        else torch.zeros(
            conv.shape[-1], dtype=torch.float16, device=conv.device
        )
    )
    torch.ops._qwen38_C.gdn_decode_sycl(
        qkv,
        conv,
        w[6].half().view(-1, 4),
        bias,
        c["indices"],
        w[7].half(),
        w[8].half(),
        ba,
        ssm,
        c["indices"],
        core,
        z,
        1 / math.sqrt(128),
    )
    torch.ops._qwen38_C.gdn_norm_int4_sycl(
        core.view(hv, 128),
        z.view(hv, 128),
        w[9].half(),
        w[4],
        w[5],
        output,
        hv,
        128,
        1e-6,
        sigmoid,
    )
    return output, conv, ssm


def assert_equivalent(workspace, c, sigmoid=True):
    with torch.no_grad():
        expected, conv, ssm = separate_stages(c, sigmoid)
        actual = invoke(workspace, c, sigmoid)
    assert actual is not None
    torch.xpu.synchronize()
    assert torch.equal(actual, expected)
    assert torch.equal(c["conv"], conv)
    assert torch.equal(c["ssm"], ssm)
    return actual


@pytest.mark.parametrize("hv", [6, 12])
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32]
)
@pytest.mark.parametrize("sigmoid", [False, True])
def test_chain_matches_separate_kernels_with_shared_paged_states(
    native, hv, dtype, sigmoid
):
    c = make_case(hv, dtype, bias=(dtype != torch.float16))
    workspace = native.GDNM1WorkspaceDirectV1()
    for _ in range(3):
        assert_equivalent(workspace, c, sigmoid)


@pytest.mark.parametrize("mutation", ["inplace", "data", "replace", "bias"])
def test_chain_refreshes_live_parameter_conversion(native, mutation):
    c = make_case()
    c["w"][7] = torch.nn.Parameter(c["w"][7])  # legal in inference/no_grad
    workspace = native.GDNM1WorkspaceDirectV1()
    assert_equivalent(workspace, c)
    with torch.no_grad():
        if mutation == "inplace":
            c["w"][7].add_(0.5)
        elif mutation == "data":
            c["w"][7].data = c["w"][7].data + 0.5
        elif mutation == "replace":
            c["w"][9] = c["w"][9] + 0.25
        else:
            c["bias"] = None
    assert_equivalent(workspace, c)


def test_chain_explicit_inference_reload_invalidates_converted_sources(native):
    with torch.inference_mode():
        c = make_case()
        workspace = native.GDNM1WorkspaceDirectV1()
        assert_equivalent(workspace, c)
        c["w"][7].add_(0.4)
        workspace.invalidate()
        assert_equivalent(workspace, c)


@pytest.mark.parametrize(
    "invalid", ["m2", "output_weight", "raw_alias", "indices_alias", "lazy"]
)
def test_chain_rejection_leaves_both_states_unmodified(native, invalid):
    c = make_case()
    if invalid == "m2":
        c["x"] = c["x"].expand(2, 2560).contiguous()
    elif invalid == "output_weight":
        c["w"][4] = c["w"][4][:, :-1]
    elif invalid == "raw_alias":
        c["w"][8] = c["conv"][0, 0, : c["hv"]]
    elif invalid == "indices_alias":
        c["indices"] = c["conv"][0, 0, :2].view(torch.int32)
    else:
        c["w"][9] = torch._neg_view(c["w"][9])
    before = (c["conv"].clone(), c["ssm"].clone())
    with torch.no_grad():
        assert invoke(native.GDNM1WorkspaceDirectV1(), c) is None
    torch.xpu.synchronize()
    assert torch.equal(c["conv"], before[0])
    assert torch.equal(c["ssm"], before[1])


def test_chain_output_metadata_does_not_mutate_private_scratch(native):
    c, workspace = make_case(), native.GDNM1WorkspaceDirectV1()
    output = assert_equivalent(workspace, c)
    output.resize_(2560)
    assert assert_equivalent(workspace, c).shape == (1, 2560)


def test_chain_two_streams_and_early_workspace_release(native):
    cases = [make_case(12), make_case(6)]
    expected = []
    with torch.no_grad():
        for c in cases:
            expected.append(separate_stages(c))
        producer = torch.xpu.current_stream()
        streams = [torch.xpu.Stream(), torch.xpu.Stream()]
        workspace = native.GDNM1WorkspaceDirectV1()
        outputs = []
        for stream, c in zip(streams, cases):
            stream.wait_stream(producer)
            with torch.xpu.stream(stream):
                outputs.append(invoke(workspace, c))
        workspace.invalidate()
        del workspace
    torch.xpu.synchronize()
    for c, actual, (out, conv, ssm) in zip(cases, outputs, expected):
        assert actual is not None and torch.equal(actual, out)
        assert torch.equal(c["conv"], conv)
        assert torch.equal(c["ssm"], ssm)
