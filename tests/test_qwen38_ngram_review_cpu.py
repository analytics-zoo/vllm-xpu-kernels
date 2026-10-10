# SPDX-License-Identifier: Apache-2.0
"""CPU alias reproducer plus pre-submit source guards; no native execution."""

from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize("lookup", (False, True))
def test_independent_storage_alias_cpu(lookup):
    buffer = bytearray(5120)
    ids = torch.frombuffer(buffer, dtype=torch.int64, count=16)
    output = torch.frombuffer(
        buffer, dtype=(torch.float16 if lookup else torch.int64)
    )
    assert not torch._C._overlaps(ids, output)
    assert ids.data_ptr() == output.data_ptr()
    source = (
        Path(__file__).resolve().parents[1] / "csrc/qwen38/ngram.cpp"
    ).read_text()
    body = source.split(
        "at::Tensor ngram_host_lookup_chunked("
        if lookup
        else "static void ngram_decode_ids_impl(",
        1,
    )[1]
    assert "check_no_alias(output," in body
    assert body.index("check_no_alias(output,") < body.index("parallel_for")
    assert "output_begin - input_begin < input.nbytes()" in source
    assert "input_begin - output_begin < output.nbytes()" in source
    if lookup:
        assert "check_no_alias(output, weight)" in body
