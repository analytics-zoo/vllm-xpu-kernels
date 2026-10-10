# SPDX-License-Identifier: Apache-2.0
"""CPU state regressions and source-order guards; no GPU/build required."""

import re
from pathlib import Path

import pytest
import torch

from tests.test_qwen38_ple_sycl import _conv_golden

SOURCE = (Path(__file__).resolve().parents[1] /
          "csrc/qwen38/ple_sycl.cpp").read_text()


def test_lazy_state_rejected_cpu():
    state = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    lazy = torch._neg_view(state)
    assert lazy.is_neg() and lazy.is_contiguous()
    assert lazy.data_ptr() == state.data_ptr()
    assert not torch.equal(lazy, state)
    body = SOURCE.split("void check_state(", 1)[1].split("inline float", 1)[0]
    assert 'check_xpu(state, "conv_state")' in body


def test_previous_acceptance_can_exceed_new_query_length_cpu():
    # Previous query has four tokens. Accept all four, then schedule one.
    state = torch.tensor([[[10., 11., -1., -1., -1.]]])
    weight = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float16)
    previous = torch.tensor([[1.], [2.], [3.], [4.]], dtype=torch.float16)
    _, cached = _conv_golden(previous, state, weight, [0], None, [0, 4],
                             [1], 1, True, -1, "spec", 3)
    current = torch.tensor([[5.]], dtype=torch.float16)
    output, updated = _conv_golden(current, cached, weight, [0], None,
                                   [0, 1], [4], 1, True, -1, "spec", 3)
    expected = torch.nn.functional.silu(torch.nn.functional.conv1d(
        torch.tensor([[[3., 4., 5.]]]), weight.float().unsqueeze(1))).half()
    torch.testing.assert_close(output, expected.reshape(1, 1))
    assert updated[0, 0, :2].tolist() == [4., 5.]
    # Padding/null requests do not read accepted values or mutate a cache.
    _, unchanged = _conv_golden(current, cached, weight, [-1], None,
                                [0, 1], [0], 1, True, -1, "spec", 3)
    assert torch.equal(unchanged, cached)
    assert "counts[r] <= query_length" not in SOURCE
    assert "slot_values[r] != null_id && query_length > 0" in SOURCE


@pytest.mark.parametrize("name", (
    "ngram_ids", "embedding_gather", "grouped_norm", "score_gate",
    "gated_value", "gated_value_grouped_norm", "residual_add",
    "short_conv_impl",
))
def test_allocator_recording_precedes_submit_cpu(name):
    match = re.search(r"at::Tensor\s+" + name + r"\(", SOURCE)
    body = SOURCE[match.end():].split("\nat::Tensor", 1)[0]
    boundary = "auto run =" if name == "short_conv_impl" else "queue.parallel_for"
    records = [match.start() for match in re.finditer(r"record_stream\(", body)]
    assert records and max(records) < body.index(boundary)
