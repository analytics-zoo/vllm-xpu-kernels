# SPDX-License-Identifier: Apache-2.0
"""CPU reproducers and source contracts; do not load a DSO or query XPU."""

import ast
import ctypes
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def test_independent_storage_overlap_cpu():
    buffer = bytearray(128)
    first = torch.frombuffer(buffer, dtype=torch.int64, count=8)
    other = torch.frombuffer(buffer, dtype=torch.int64, count=8, offset=32)
    assert not torch._C._overlaps(first, other)
    other.fill_(7)
    assert first.tolist() == [0] * 4 + [7] * 4
    source = (ROOT / "csrc/qwen38/moe_sycl_workspace.cpp").read_text()
    body = source.split("bool output_overlaps(", 1)[1].split(
        "std::optional<at::Tensor>", 1)[0]
    assert "byte_range(output)" in body
    assert "byte_range(input)" in body
    assert "output_range.begin < input_range.end" in body


def test_prefill_count_overflow_cpu():
    counts = [(1 << 31) - 1, (1 << 31) - 1, 514] + [0] * 509
    assert sum(counts) != 512
    # The old int32 sentinel passes, and the last expert starts at -2.
    assert ctypes.c_int32(sum(counts)).value == 512
    assert ctypes.c_int32(sum(counts[:2])).value == -2
    source = (ROOT / "csrc/qwen38/moe_sycl_prefill.cpp").read_text()
    prefix = source.split("void launch_prefix(", 1)[1].split(
        "template <int K>", 1)[0]
    assert "const int64_t count" in prefix
    assert "sycl::plus<int64_t>()" in prefix
    assert "row_start + count != a.m" in prefix
    assert "row_start <= a.m" in prefix


def test_empty_prefill_needs_no_pointers_cpu():
    assert torch.empty((0, 80), dtype=torch.float16).data_ptr() == 0
    source = (ROOT / "csrc/qwen38/moe_sycl_prefill.cpp").read_text()
    body = source.split("bool try_prefill_down(", 1)[1]
    assert body.index("if (a.m == 0) return true;") < body.index("!a.x")


@pytest.mark.parametrize("compact", (80, 160))
def test_signed_byte_view_and_submit_error_propagation_cpu(compact):
    # Execute the real Python functions without importing the native package.
    tree = ast.parse((ROOT / "vllm_xpu_kernels/qwen38_moe.py").read_text())
    name = f"moe_compact{compact}_down_grouped_gemm"
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ("_signed_down_weight", name)]
    seen = []

    def operation(x, weight, scale, counts):
        seen.append(weight)
        raise RuntimeError("submitted stage failed")

    namespace = {"torch": torch, "_ops": SimpleNamespace(**{name: operation})}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "provider", "exec"),
         namespace)
    weight = torch.arange(256, dtype=torch.int16).to(torch.uint8)
    with pytest.raises(RuntimeError, match="submitted stage failed"):
        namespace[name](None, weight, None, None)
    assert len(seen) == 1
    assert seen[0].dtype == torch.int8
    assert seen[0].data_ptr() == weight.data_ptr()
    assert torch.equal(seen[0].view(torch.uint8), weight)
