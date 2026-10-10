# SPDX-License-Identifier: Apache-2.0
"""CPU source checks for physical overlap with independent Storage owners."""

from pathlib import Path

import pytest


@pytest.mark.parametrize("filename", ["int4.cpp", "gdn_sycl_projection.cpp"])
def test_cpu_int4_checks_physical_bytes_not_only_storage_identity(filename):
    source = (Path(__file__).resolve().parents[1] /
              "csrc/qwen38" / filename).read_text()
    helper = source.split("void check_no_overlap(", 1)[1]
    helper = helper.split("\n}\n", 1)[0]
    assert "a.const_data_ptr()" in helper
    assert "b.const_data_ptr()" in helper
    assert "a.numel() * a.element_size()" in helper
    remainder = source.split("\n}\n", 1)[1]
    assert "check_no_overlap(output," in remainder
