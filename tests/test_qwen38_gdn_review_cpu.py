# SPDX-License-Identifier: Apache-2.0
"""CPU reproductions and source guards, never a SYCL execution test."""

import ctypes
import importlib.util
import itertools
import math
import os
import random
import sys
import weakref
from pathlib import Path

import pytest
import torch

SOURCE = (
    Path(__file__).resolve().parents[1] / "csrc/qwen38/gdn_sycl.cpp"
).read_text()


@pytest.fixture(scope="module")
def native_host_module():
    # Loading the DSO registers ops but does not initialize XPU or submit work.
    library = os.environ.get("QWEN38_HOST_TEST_LIBRARY")
    if not library:
        pytest.skip("set QWEN38_HOST_TEST_LIBRARY to a built native DSO")
    name = "_qwen38_host_preflight_test._qwen38_C"
    spec = importlib.util.spec_from_file_location(name, library)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def native_alias_guard(native_host_module):
    return native_host_module.tensors_disjoint_host


@pytest.fixture(scope="module")
def native_binding_snapshot(native_host_module):
    return native_host_module.TensorBindingSnapshotV1


def test_native_binding_snapshot_identity_order_and_empty_list(
    native_binding_snapshot,
):
    first, second = torch.ones(4), torch.zeros(4)
    snapshot = native_binding_snapshot([first, second])
    assert snapshot.matches([first, second])
    assert not snapshot.matches([second, first])
    assert not snapshot.matches([first])
    assert not snapshot.matches([first, second, first])
    assert not snapshot.matches([first, second.view_as(second)])
    assert native_binding_snapshot([]).matches([])


@pytest.mark.parametrize("shape", [(), (0,), (3, 0), (2, 3)])
def test_native_binding_snapshot_cpu_metadata_only(
    native_binding_snapshot, shape
):
    was_initialized = torch.xpu.is_initialized()
    source = torch.empty(shape, dtype=torch.float16)
    snapshot = native_binding_snapshot([source])
    assert snapshot.matches([source])
    assert not snapshot.matches([source.clone()])
    assert torch.xpu.is_initialized() is was_initialized


@pytest.mark.parametrize(
    "mutation", ["data", "dtype", "resize", "set", "inplace"]
)
def test_native_binding_snapshot_detects_live_mutation(
    native_binding_snapshot, mutation
):
    source = torch.nn.Parameter(torch.ones(2, 4), requires_grad=False)
    snapshot = native_binding_snapshot([source])
    original_identity = source._cdata
    original_version = source._version
    if mutation == "data":
        source.data = torch.zeros_like(source)
    elif mutation == "dtype":
        source.data = source.data.to(torch.float64)
    elif mutation == "resize":
        source.resize_(4, 2)
    elif mutation == "set":
        source.set_(torch.zeros_like(source))
    else:
        source.add_(1)
    assert source._cdata == original_identity
    if mutation in ("data", "dtype"):
        assert source._version == original_version
    assert not snapshot.matches([source])
    assert native_binding_snapshot([source]).matches([source])


def test_native_binding_snapshot_version_survives_metadata_restore(
    native_binding_snapshot,
):
    source = torch.ones(2, 4)
    snapshot = native_binding_snapshot([source])
    pointer = source.data_ptr()
    source.resize_(4, 2)
    source.resize_(2, 4)
    assert source.data_ptr() == pointer
    assert not snapshot.matches([source])


def test_native_binding_snapshot_inference_metadata_and_reload_contract(
    native_binding_snapshot,
):
    with torch.inference_mode():
        source = torch.ones(2, 4)
        snapshot = native_binding_snapshot([source])
        assert snapshot.matches(
            [source]
        )  # No version counter access exception.
        source.add_(1)
        assert snapshot.matches(
            [source]
        )  # Content reload must invalidate cache.
        source.data = torch.zeros_like(source)
        assert not snapshot.matches([source])
        snapshot = native_binding_snapshot([source])
        source.resize_(4, 2)
        assert not snapshot.matches([source])


def test_native_binding_snapshot_strides_pointer_and_lazy_views(
    native_binding_snapshot,
):
    page = torch.ones(8, 32)
    source = page[:, :8]
    snapshot = native_binding_snapshot([source])
    assert snapshot.matches([source])
    pointer, version = source.data_ptr(), source._version
    source.data = page.as_strided((8, 8), (16, 1))
    assert source.data_ptr() == pointer and source._version == version
    assert not snapshot.matches([source])
    transposed = page.T
    assert native_binding_snapshot([transposed]).matches([transposed])
    negated = torch._neg_view(page)
    assert not native_binding_snapshot([negated]).matches([negated])
    conjugate = torch.ones(4, dtype=torch.complex64).conj()
    assert not native_binding_snapshot([conjugate]).matches([conjugate])


def test_native_binding_snapshot_retains_source_storage(
    native_binding_snapshot,
):
    class Backing(bytearray):
        pass

    backing = Backing(64)
    backing_ref = weakref.ref(backing)
    source = torch.frombuffer(backing, dtype=torch.float32)
    snapshot = native_binding_snapshot([source])
    del source, backing
    assert backing_ref() is not None  # The snapshot's Tensor owns the buffer.
    del snapshot
    assert backing_ref() is None


@pytest.mark.parametrize("layout", ["meta", "sparse"])
def test_native_binding_snapshot_unsupported_layout_fails_closed(
    native_binding_snapshot, layout
):
    source = (
        torch.empty(4, device="meta")
        if layout == "meta"
        else torch.ones(4).to_sparse()
    )
    snapshot = native_binding_snapshot([source])
    assert not snapshot.matches([source])
    valid = native_binding_snapshot([torch.ones(4)])
    assert not valid.matches([source])


def test_native_binding_snapshot_invalid_abi_is_not_hidden(
    native_binding_snapshot,
):
    source = torch.ones(4)
    snapshot = native_binding_snapshot([source])
    with pytest.raises(TypeError):
        native_binding_snapshot([None])
    with pytest.raises(TypeError):
        snapshot.matches([None])


def _tensor_bytes(tensor):
    if tensor.numel() == 0:
        return set()
    return {
        tensor.data_ptr()
        + tensor.element_size()
        * sum(index * stride for index, stride in zip(indices, tensor.stride()))
        + byte
        for indices in itertools.product(
            *(range(size) for size in tensor.shape)
        )
        for byte in range(tensor.element_size())
    }


def test_native_host_alias_guard_matches_independent_byte_sets(
    native_alias_guard,
):
    storage = torch.empty(4096, dtype=torch.uint8)
    rng = random.Random(2031)
    for _ in range(400):
        views = []
        for _ in range(4):
            pitch = rng.randint(1, 32)
            views.append(
                storage.as_strided(
                    (rng.randint(0, 9), rng.randint(1, pitch)),
                    (pitch, 1),
                    rng.randint(0, 256),
                )
            )
        exact = [_tensor_bytes(view) for view in views]
        expected = not any(
            exact[i] & exact[j] for i in range(2) for j in range(i + 1, 4)
        )
        assert native_alias_guard(views[:2], [*views[2:], None]) is expected


def test_native_host_alias_guard_paged_state_and_live_rebinding(
    native_alias_guard,
):
    page = torch.zeros(16, 1024, dtype=torch.float16)
    conv, ssm = page[:, :128], page[:, 128:]
    output = torch.empty(1, 32, dtype=torch.float16)
    assert native_alias_guard([conv, ssm, output], [None])
    assert not native_alias_guard([conv, ssm], [page[4, 127:129]])
    output.set_(page[5, :32])
    assert not native_alias_guard([conv, ssm, output], [])


def test_native_host_alias_guard_distinct_storage_same_address(
    native_alias_guard,
):
    backing = bytearray(128)
    first = torch.frombuffer(backing, dtype=torch.uint8)
    second = torch.frombuffer(backing, dtype=torch.float16)
    assert first.untyped_storage()._cdata != second.untyped_storage()._cdata
    assert native_alias_guard([first[:32]], [second[16:]])
    assert not native_alias_guard([first[:32]], [second[15:]])
    assert native_alias_guard([], [first, first])


def test_native_host_alias_guard_mixed_dtype_pitched_storage(
    native_alias_guard,
):
    backing = bytearray(4096)
    byte_view = torch.frombuffer(backing, dtype=torch.uint8)
    half_view = torch.frombuffer(backing, dtype=torch.float16)
    writer = byte_view.as_strided((9, 8), (64, 1), 8)
    for offset in (0, 3, 4, 7, 8, 12, 31, 32, 35):
        reader = half_view.as_strided((8, 4), (32, 1), offset)
        expected = not bool(_tensor_bytes(writer) & _tensor_bytes(reader))
        assert native_alias_guard([writer], [reader]) is expected


@pytest.mark.parametrize(
    "layout", ["transpose", "inner_stride", "broadcast", "meta"]
)
def test_native_host_alias_guard_rejects_unhandled_layout(
    native_alias_guard, layout
):
    tensor = torch.empty(8, 16)
    invalid = {
        "transpose": lambda: tensor.T,
        "inner_stride": lambda: tensor[:, ::2],
        "broadcast": lambda: tensor[:1].expand(8, 16),
        "meta": lambda: torch.empty(4, device="meta"),
    }[layout]()
    assert not native_alias_guard([invalid], [])


def test_cpu_pitched_overlap_algorithm_matches_explicit_bytes():
    def optimized(a, b):
        ab, aw, ap, ar = a
        bb, bw, bp, br = b
        if (
            not ar
            or not br
            or ab + (ar - 1) * ap + aw <= bb
            or bb + (br - 1) * bp + bw <= ab
        ):
            return False
        if ap == bp:
            if ab > bb:
                ab, aw, ap, ar, bb, bw, bp, br = bb, bw, bp, br, ab, aw, ap, ar
            row, offset = divmod(bb - ab, ap)
            return (row < ar and offset < aw) or (
                row + 1 < ar and bw > ap - offset
            )
        i = j = 0
        while i < ar and j < br:
            x, y = ab + i * ap, bb + j * bp
            if x <= y and y - x >= aw:
                i += (y - x - aw) // ap + 1
            elif y <= x and x - y >= bw:
                j += (x - y - bw) // bp + 1
            else:
                return True
        return False

    rng = random.Random(7021)
    for _ in range(10000):
        ap, bp = rng.randint(1, 32), rng.randint(1, 32)
        if rng.random() < 0.5:
            bp = ap
        a = (rng.randint(0, 200), rng.randint(1, ap), ap, rng.randint(0, 9))
        b = (rng.randint(0, 200), rng.randint(1, bp), bp, rng.randint(0, 9))
        exact = lambda t: {
            t[0] + row * t[2] + byte
            for row in range(t[3])
            for byte in range(t[1])
        }
        assert optimized(a, b) == bool(exact(a) & exact(b)), (a, b)


@pytest.mark.parametrize(
    "padded,separate_storage", [(True, False), (True, True), (False, True)]
)
def test_cpu_physical_alias_can_escape_torch_overlap_assert(
    padded, separate_storage
):
    # Inspect the existing CPU Torch implementation, without compiling code.
    library = ctypes.CDLL(
        str(Path(torch.__file__).parent / "lib/libtorch_cpu.so")
    )
    overlap = getattr(
        library, "_ZN2at18get_overlap_statusEPKN3c1010TensorImplES3_", None
    )
    if overlap is None:
        pytest.skip("Torch does not export the TensorImpl overlap probe")
    overlap.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    overlap.restype = ctypes.c_int
    buffer = bytearray(64)
    base = torch.frombuffer(buffer, dtype=torch.float16).view(2, 16)
    other = (
        torch.frombuffer(buffer, dtype=torch.float16).view(2, 16)
        if separate_storage
        else base
    )
    reader = base[:, :8] if padded else base
    aliased = other.flatten()[:16].view(2, 8)
    assert reader.data_ptr() == aliased.data_ptr()
    assert torch._C._overlaps(reader, aliased) is not separate_storage
    # Full=0, Partial=1, No=2, TooHard=3 in ATen/MemoryOverlap.h.
    assert overlap(reader._cdata, aliased._cdata) == (3 if padded else 2)
    assert "physical_overlap(a, b)" in SOURCE
    assert "GDN alias check requires dense inner rows" in SOURCE
    assert "left.pitch == right.pitch" in SOURCE
    checks = SOURCE.split("void check_no_cross_alias(", 1)[1]
    checks = checks.split("\n}\n", 1)[0]
    assert "check_no_overlap(*a, *b);" in checks
    # Dense separate Storage owners need a pointer check even for status No.
    helper = SOURCE.split("void check_no_overlap(", 1)[1]
    helper = helper.split("\n}\n", 1)[0]
    assert "physical_overlap(a, b)" in helper
    rows = SOURCE.split("MemoryRows memory_rows(", 1)[1].split("\n}\n", 1)[0]
    assert "t.const_data_ptr()" in rows


@pytest.mark.parametrize(
    "name,first_submit",
    [
        ("gdn_decode_sycl", "launch_decode_root("),
        ("gdn_spec_v2_sycl", "launch_conv<true>("),
        ("gdn_norm_gate_sycl", "queue.parallel_for<"),
        ("gdn_spec_conv_probe_sycl", "launch_conv<true>("),
    ],
)
def test_cpu_gdn_records_ownership_before_first_submit(name, first_submit):
    body = SOURCE.split(f"void {name}(", 1)[1].split("\n}\n", 1)[0]
    assert body.index("record_stream(") < body.index(first_submit)
    if name in ("gdn_decode_sycl", "gdn_spec_v2_sycl"):
        allocation = body.index("auto qkv = at::empty(")
        assert allocation < body.index("record_stream({&qkv}, stream);")
        assert body.index("record_stream({&qkv}, stream);") < body.index(
            "launch_conv<"
        )


@pytest.mark.parametrize("scale", [1e300, 1e-300])
def test_cpu_gdn_scale_must_remain_positive_finite_fp32(scale):
    assert math.isfinite(scale) and scale > 0
    converted = torch.tensor(scale, dtype=torch.float64).float().item()
    assert not (math.isfinite(converted) and converted > 0)
    assert "std::isfinite(scale_fp32) && scale_fp32 > 0.0f" in SOURCE


def test_cpu_gdn_accepted_subtraction_cannot_overflow_int32():
    accepted = torch.tensor([-(2**31)], dtype=torch.int32)
    assert int((accepted - 1)[0]) == 2**31 - 1  # Previous int32 expression.
    assert int((accepted.long() - 1)[0]) == -(2**31) - 1
    lines = [
        line
        for line in SOURCE.splitlines()
        if "accepted[" in line and "- 1" in line
    ]
    assert len(lines) == 4
    assert all("int64_t(accepted[" in line for line in lines)


def test_cpu_gdn_projection_checks_stream_before_fused_or_two_stage_path():
    source = (
        Path(__file__).resolve().parents[1]
        / "csrc/qwen38/gdn_sycl_projection.cpp"
    ).read_text()
    body = source.split("void gdn_norm_int4_sycl(", 1)[1]
    # External SYCL queues can be unordered. The two-stage path otherwise
    # allows projection to read normalized before norm_gate has written it.
    assert body.index("queue.is_in_order()") < body.index("if (hv == kHeads")
