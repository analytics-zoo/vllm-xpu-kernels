# SPDX-License-Identifier: Apache-2.0
"""CPU-only contracts for persistent MoE buffers and live weight refresh."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_PATH = (Path(__file__).resolve().parents[1] / "vllm_xpu_kernels" /
         "moe_shared_fused_interface.py")
_SPEC = importlib.util.spec_from_file_location("moe_shared_cpu", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
interface = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(interface)


def _sources(value=1.0):
    e, h, inter = 2, 16, 8

    def weight(*shape):
        return torch.full(shape, value).to(torch.float8_e4m3fn)

    return dict(
        w13=weight(e, 2 * inter, h).transpose(1, 2),
        w13_scales=torch.full((e * 2, ), value)[::2],
        w2=weight(e, h, inter).transpose(1, 2),
        w2_scales=torch.full((e * 2, ), value)[::2],
        shared_w13=weight(2 * inter, h).t(),
        shared_w13_scale=torch.tensor(value),
        shared_w2=weight(h, inter).t(),
        shared_w2_scale=torch.tensor([value]),
        shared_gate=torch.full((1, h * 2), value,
                               dtype=torch.float16)[:, ::2],
        router_weight=torch.full((h, e), value, dtype=torch.float16).t(),
    )


def _op():
    return interface.XpuMoESharedFusedDecode(**_sources(), top_k=1)


def test_wrapper_cpu_cache_key_isolates_device_stream_and_m(monkeypatch):
    state = [0, 7]
    monkeypatch.setattr(torch.xpu, "current_stream", lambda device:
                        SimpleNamespace(device_index=state[0],
                                        stream_id=state[1]))
    key = interface.XpuMoESharedFusedDecode._cache_key
    a = key(1, torch.device("xpu:0"))
    assert a == key(1, torch.device("xpu:0"))
    assert a != key(8, torch.device("xpu:0"))
    state[1] = 9
    assert a != key(1, torch.device("xpu:0"))
    state[0] = 1
    state[1] = 7
    assert a != key(1, torch.device("xpu:1"))


def test_wrapper_cpu_workspace_and_routing_are_device_scoped():
    op = _op()
    cpu = op.workspace(1, torch.device("cpu"))
    meta = op.workspace(1, torch.device("meta"))
    assert cpu.device.type == "cpu" and meta.device.type == "meta"
    assert op.workspace(1, torch.device("cpu")) is cpu
    a = op._routing_buffers(1, torch.device("cpu"))
    b = op._routing_buffers(1, torch.device("meta"))
    assert a is op._routing_buffers(1, torch.device("cpu"))
    assert all(x.device.type == "cpu" for x in a)
    assert all(x.device.type == "meta" for x in b)


def test_wrapper_cpu_refresh_updates_all_weights_without_rebinding():
    op = _op()
    values = _sources(2.0)
    identities = {name: (id(getattr(op, name)), getattr(op, name).data_ptr())
                  for name in values}
    op.refresh_weights(**values, top_k=1, renormalize=True)
    for name, source in values.items():
        result = getattr(op, name)
        assert (id(result), result.data_ptr()) == identities[name]
        torch.testing.assert_close(result.float(),
                                   source.reshape_as(result).float(),
                                   rtol=0, atol=0)


def test_wrapper_cpu_refresh_observes_inplace_live_source_updates():
    values = _sources()
    op = interface.XpuMoESharedFusedDecode(**values, top_k=1)
    for source in values.values():
        source.copy_(torch.full(source.shape, 3.0).to(source.dtype))
    op.refresh_weights(**values)
    for name in values:
        result = getattr(op, name)
        torch.testing.assert_close(result.float(),
                                   torch.full(result.shape, 3.0),
                                   rtol=0, atol=0)


@pytest.mark.parametrize("invalid", ["shape", "dtype", "device", "layout",
                                    "negative", "zero_stride", "router_none",
                                    "top_k", "renormalize"])
def test_wrapper_cpu_refresh_preflights_every_source_before_copy(invalid):
    op = _op()
    values = _sources(2.0)
    kwargs = {}
    if invalid == "shape":
        values["router_weight"] = values["router_weight"][:, :-1]
    elif invalid == "dtype":
        values["router_weight"] = values["router_weight"].float()
    elif invalid == "device":
        values["router_weight"] = values["router_weight"].to("meta")
    elif invalid == "layout":
        values["router_weight"] = values["router_weight"].to_sparse()
    elif invalid == "negative":
        values["router_weight"] = torch._neg_view(values["router_weight"])
    elif invalid == "zero_stride":
        values["router_weight"] = values["router_weight"][:1].expand(2, 16)
    elif invalid == "router_none":
        values["router_weight"] = None
    elif invalid == "top_k":
        kwargs["top_k"] = 2
    else:
        kwargs["renormalize"] = False
    before = {name: getattr(op, name).clone() for name in _sources()}
    with pytest.raises(ValueError, match="refresh_weights"):
        op.refresh_weights(**values, **kwargs)
    for name, expected in before.items():
        torch.testing.assert_close(getattr(op, name).float(), expected.float(),
                                   rtol=0, atol=0)


def test_wrapper_cpu_refresh_skips_same_views(monkeypatch):
    op = _op()
    values = {name: getattr(op, name) for name in _sources()}

    def forbidden(*args, **kwargs):
        raise AssertionError("same-view refresh must not copy")

    monkeypatch.setattr(torch.Tensor, "copy_", forbidden)
    op.refresh_weights(**values)


def test_wrapper_cpu_refresh_rejects_cross_source_alias_before_copy():
    op = _op()
    values = _sources(2.0)
    values["w13_scales"] = op.w2_scales
    before = op.w13.clone()
    with pytest.raises(ValueError, match="overlap"):
        op.refresh_weights(**values)
    torch.testing.assert_close(op.w13.float(), before.float(), rtol=0, atol=0)


def test_wrapper_cpu_workspace_has_room_for_multiple_tiles_per_expert():
    # R=16,E=1,tile_m=8 has two tiles, not one. Validate the independent
    # byte carve-out with R entries in each of the three tile-map arrays.
    m, top_k, h, w13_n, inter = 16, 1, 128, 256, 128
    align = lambda size: (size + 255) // 256 * 256
    offset = align(4 * (3 * 16 + 1 + 16))
    offset = align(offset + 4 * 16)
    offset = align(offset + 4 * m)
    expected = offset + 2 * (16 * h + 32 * w13_n + 32 * inter + 16 * h + m * h)
    assert interface.moe_shared_fused_workspace_bytes(
        m, top_k, 1, h, w13_n, inter) == expected


def test_wrapper_cpu_refresh_allows_disjoint_flat_parameter_views():
    # Different dtypes and transposed KN matrices can occupy disjoint byte
    # ranges of one flat allocation. Storage-level overlap rejects this.
    flat = torch.zeros(4096, dtype=torch.uint8)
    values = _sources()
    offset = 0
    for name, template in values.items():
        offset = (offset + 15) // 16 * 16
        nbytes = template.numel() * template.element_size()
        block = flat[offset:offset + nbytes].view(template.dtype)
        if name in ("w13", "w2", "shared_w13", "shared_w2"):
            shape = list(template.shape)
            shape[-1], shape[-2] = shape[-2], shape[-1]
            source = block.reshape(shape).transpose(-1, -2)
        else:
            source = block.reshape(template.shape)
        source.copy_(template)
        values[name] = source
        offset += nbytes
    op = interface.XpuMoESharedFusedDecode(**values, top_k=1)
    assert torch._C._overlaps(values["shared_w13"], op.shared_w13_scale)
    assert not interface._byte_ranges_overlap(values["shared_w13"],
                                               op.shared_w13_scale)
    pointers = {name: getattr(op, name).data_ptr() for name in values}
    for source in values.values():
        source.copy_(torch.full(source.shape, 2.0).to(source.dtype))
    op.refresh_weights(**values)
    for name in values:
        result = getattr(op, name)
        assert result.data_ptr() == pointers[name]
        torch.testing.assert_close(result.float(),
                                   torch.full(result.shape, 2.0),
                                   rtol=0, atol=0)


@pytest.mark.parametrize("width", [16, 64, 128, 256, 384, 2048])
@pytest.mark.parametrize("routes", [1, 2, 16])
def test_wrapper_cpu_gather_grid_covers_only_valid_elements(width, routes):
    # The native contract permits H%16==0, including K/4 < local_size64.
    # Check the rounded grid against a nontrivial expert-major permutation.
    chunks = width // 4
    global_chunks = (chunks + 63) // 64 * 64
    assert global_chunks % 64 == 0
    if width % 256 == 0:
        assert global_chunks == chunks
    permutation = list(reversed(range(routes)))
    copied = []
    for route in range(routes):
        for chunk in range(global_chunks):
            column = chunk * 4
            if column >= width:
                continue
            copied.extend(permutation[route] * width + column + j
                          for j in range(4))
    assert sorted(copied) == list(range(routes * width))
    source = (_PATH.parents[1] / "csrc/xpu/grouped_gemm/xe_2/"
              "moe_shared_fused_decode_xe2.cpp").read_text()
    gather = source.split("class SpGatherLauncher", 1)[1].split(
        "// Kernels 3/5/6/8", 1)[0]
    assert "ceil_div(chunks, 64) * 64" in gather
    assert gather.index("if (c >= K) return;") < gather.index(
        "const int dst = perm_pos[r]")



def test_wrapper_cpu_dispatcher_resolves_lazy_negative_before_native():
    library = torch.library.Library("moe_negative_cpu_contract", "DEF")
    library.define("probe(Tensor x) -> Tensor")
    observed = []

    def probe(tensor):
        observed.append(tensor.is_neg())
        return tensor.clone()

    library.impl("probe", probe, "CPU")
    base = torch.tensor([1.0, 2.0])
    negative = torch._neg_view(base)
    output = torch.ops.moe_negative_cpu_contract.probe(negative)
    assert observed == [False]
    assert negative.is_neg()
    torch.testing.assert_close(output, -base, atol=0, rtol=0)
    torch.testing.assert_close(base, torch.tensor([1.0, 2.0]), atol=0, rtol=0)
