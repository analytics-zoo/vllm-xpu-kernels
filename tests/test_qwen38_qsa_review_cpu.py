# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for direct-entry preflight and exception ownership ordering."""

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1] / "csrc" / "qwen38"


def body(source: str, name: str) -> str:
    text = (ROOT / source).read_text()
    start = text.index("{", text.index(f" {name}("))
    depth = 0
    for index in range(start, len(text)):
        depth += (text[index] == "{") - (text[index] == "}")
        if depth == 0:
            return text[start:index + 1]
    raise AssertionError("unclosed function")


@pytest.mark.parametrize("source,name,anchor", [
    ("qsa_sycl.cpp", "group_compress_v2", "raw_keys"),
    ("qsa_sycl.cpp", "token_split_attention_v3", "q"),
    ("qsa_sycl.cpp", "select_paged_tokens_v2", "q"),
    ("qsa_sycl_aux.cpp", "qkv_postprocess_v1", "qkv"),
    ("qsa_sycl_aux.cpp", "indexer_norm_rope_v2", "input"),
])
def test_direct_anchor_is_checked_before_recording(source, name, anchor):
    text = body(source, name)
    assert text.index(f"check_xpu({anchor}, {anchor},") < text.index(
        "record_tensors(")


@pytest.mark.parametrize("source,name,submit", [
    ("qsa_sycl.cpp", "group_compress_v2", "launch_compress<"),
    ("qsa_sycl.cpp", "token_split_attention_v3", "launch_phase0<"),
    ("qsa_sycl.cpp", "select_paged_tokens_v2", "launch_selection_phase0<"),
    ("qsa_sycl_aux.cpp", "qkv_postprocess_v1", "launch_qkv<"),
    ("qsa_sycl_aux.cpp", "indexer_norm_rope_v2", "launch_indexer<"),
    ("qsa_sycl_aux.cpp", "indexer_projection_int4_v1", "launch_projection_int4("),
    ("qsa_sycl_owner.cpp", "store_cache_rows_v3", "launch_store_typed("),
    ("qsa_sycl_owner.cpp", "try_store_m1_transaction_v1", "launch_store_typed("),
    ("qsa_sycl_owner.cpp", "try_store_m1_transaction_fused_v1",
     "launch_store_m1_transaction_fused("),
])
def test_owners_recorded_before_first_submit(source, name, submit):
    text = body(source, name)
    assert text.count("record_tensors(") == 1
    assert text.index("record_tensors(") < text.index(submit)
