# SPDX-License-Identifier: Apache-2.0
"""CPU source-contract checks; these do not execute or compile SYCL."""

import ctypes
from pathlib import Path

import pytest
import torch


SOURCE = (Path(__file__).resolve().parents[1] /
          "csrc/qwen38/hc_sycl.cpp").read_text()


@pytest.mark.parametrize("name", [
    "grouped_norm", "gate_mix", "combine", "combine_norm", "down", "up",
    "up_gate_mix", "combine_mix_impl", "project_mix"
])
def test_cpu_hc_records_ownership_before_first_submit(name):
    # If a later submit throws, stack unwinding must not release memory still
    # used by the first kernel on a different stream.
    signature = f"void {name}("
    body = SOURCE.split(signature, 1)[1].split("\n}\n", 1)[0]
    assert body.index("record(") < body.index("launch_")


def test_cpu_hc_strided_alias_checks_do_not_ignore_too_hard():
    assert "MemOverlapStatus::TooHard" in SOURCE
    assert "storage().nbytes()" in SOURCE
    # Both allowed strided readers must reach the strengthened check.
    assert "check_no_overlap(output, injection);" in SOURCE
    assert "check_no_overlap(output, input);" in SOURCE
    transaction = SOURCE.split("void check_distinct_outputs(", 1)[1]
    transaction = transaction.split("\n}\n", 1)[0]
    assert "check_no_overlap(*out, *in);" in transaction


@pytest.mark.parametrize("width", [4, 320])
def test_cpu_hc_large_row_stride_is_not_narrowed_to_int(width):
    # Metadata-only: a valid two-row layout above INT_MAX, no huge allocation.
    tensor = torch.empty_strided((2, width), (2**31, 1), device="meta")
    assert tensor.stride(0) >= width and tensor.stride(1) == 1
    assert ctypes.c_int(tensor.stride(0)).value < 0
    if width == 4:
        manual = (Path(__file__).resolve().parents[1] /
                  "csrc/qwen38/hc_norm_m1_manual.h").read_text()
        assert SOURCE.count("int64_t injection_stride,") == 2
        assert manual.count("int64_t injection_stride,") == 2
    else:
        assert "int64_t input_stride," in SOURCE


@pytest.mark.parametrize("name", ["combine_mix_impl", "project_mix"])
def test_cpu_hc_transaction_rejects_unordered_stream_before_submit(name):
    body = SOURCE.split(f"void {name}(", 1)[1].split("\n}\n", 1)[0]
    assert body.index("queue.is_in_order()") < body.index("launch_")


def test_cpu_hc_binding_and_isolated_probe_cover_physical_alias():
    root = Path(__file__).resolve().parents[1] / "csrc/qwen38"
    binding = (root / "hc_bindings.cpp").read_text()
    assert "output.const_data_ptr()" in binding
    assert "input.const_data_ptr()" in binding
    probe = (root / "hc_up_dpas_probe.cpp").read_text()
    assert "check_no_overlap(output, input);" in probe
    assert "MemOverlapStatus::TooHard" in probe
    body = probe.split("void up_gate_mix(", 1)[1]
    assert body.index("record(input, stream);") < body.index("launch<")
