"""Independent math/state checks for the isolated ordinary-SYCL PLE build.

Build the test-only DSO from ``ple_sycl.cpp`` with
``QWEN38_PLE_SYCL_STANDALONE_TEST`` and set
``QWEN38_PLE_SYCL_TEST_LIBRARY`` before running this file on an assigned GPU.
No ESIMD module is needed for the mathematical checks.
"""

import math
import os
from itertools import product
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def ops():
    integrated = os.environ.get("QWEN38_SYCL_LIBRARY")
    if integrated:
        if not torch.xpu.is_available():
            pytest.skip("XPU unavailable")
        torch.ops.load_library(integrated)
        return torch.ops._qwen38_C
    lib = os.environ.get("QWEN38_PLE_SYCL_TEST_LIBRARY")
    if not lib or not Path(lib).is_file():
        pytest.skip("Set QWEN38_PLE_SYCL_TEST_LIBRARY to the built test DSO")
    if not torch.xpu.is_available():
        pytest.skip("XPU unavailable")
    torch.ops.load_library(lib)
    return torch.ops._qwen38_ple_sycl_test


@pytest.fixture(scope="module")
def esimd_ops():
    lib = os.environ.get("QWEN38_PLE_ESIMD_LIBRARY")
    if not lib or not Path(lib).is_file():
        pytest.skip("Set QWEN38_PLE_ESIMD_LIBRARY to the final ESIMD DSO")
    torch.ops.load_library(lib)
    return torch.ops.custom_esimd_kernels_vllm


def _xpu(data, dtype=None):
    return torch.as_tensor(data, dtype=dtype).to("xpu")


def _assert_half(actual, expected, atol=2e-3, rtol=1e-3):
    torch.testing.assert_close(actual.cpu(),
                               expected.cpu(),
                               atol=atol,
                               rtol=rtol)


def _ngram_golden(ids, starts, context, multipliers, vocab, offsets, eos,
                  heads_per_ngram):
    heads = len(vocab)
    width = len(context[0])
    result = [[0] * heads for _ in ids]
    for request in range(len(starts) - 1):
        sequence = context[request] + ids[starts[request]:starts[request + 1]]
        for local, token in enumerate(ids[starts[request]:starts[request +
                                                                 1]]):
            position = width + local
            preceding_eos = max(
                (i for i in range(position + 1) if sequence[i] == eos),
                default=-1,
            )
            for head in range(heads):
                ngram = head // heads_per_ngram + 2
                mixed = (token * multipliers[0]) & ((1 << 64) - 1)
                for shift in range(1, ngram):
                    predecessor = (sequence[position - shift] if position -
                                   shift > preceding_eos else eos)
                    mixed ^= ((predecessor * multipliers[shift]) &
                              ((1 << 64) - 1))
                signed = mixed if mixed < (1 << 63) else mixed - (1 << 64)
                result[starts[request] + local][head] = (signed % vocab[head] +
                                                         offsets[head])
    return result


def _ngram_finite_window(ids, starts, context, multipliers, vocab, offsets,
                         eos, heads_per_ngram):
    """Python transcription of the finite-predecessor SYCL logic."""
    width = len(context[0])
    heads = width * heads_per_ngram
    result = [[0] * heads for _ in ids]
    mask = (1 << 64) - 1
    for request in range(len(starts) - 1):
        for token in range(starts[request], starts[request + 1]):
            position = width + token - starts[request]
            current = ids[token]
            for head in range(heads):
                ngram = head // heads_per_ngram + 2
                mixed = (current * multipliers[0]) & mask
                after_eos = current == eos
                for shift in range(1, ngram):
                    prior = eos
                    predecessor = position - shift
                    if not after_eos and predecessor >= 0:
                        if predecessor < width:
                            prior = context[request][predecessor]
                        else:
                            prior = ids[starts[request] + predecessor - width]
                        after_eos = prior == eos
                    mixed ^= (prior * multipliers[shift]) & mask
                signed = mixed if mixed < (1 << 63) else mixed - (1 << 64)
                result[token][head] = signed % vocab[head] + offsets[head]
    return result


def test_ngram_finite_window_matches_whole_prefix_cpu():
    for width in range(1, 5):
        multipliers = [((1 << 63) + 8191 * (i + 1)) for i in range(width + 1)]
        for heads_per_ngram in (1, 3):
            heads = width * heads_per_ngram
            vocab = [17 + 2 * i for i in range(heads)]
            offsets = [-100 + 31 * i for i in range(heads)]
            for prefix in product((0, 1), repeat=width):
                context = [list(prefix), list(reversed(prefix))]
                for length in range(5):
                    for ids in product((0, 1), repeat=length):
                        ids = list(ids)
                        for split in range(length + 1):
                            starts = [0, split, length]
                            expected = _ngram_golden(
                                ids, starts, context, multipliers, vocab,
                                offsets, 0, heads_per_ngram)
                            actual = _ngram_finite_window(
                                ids, starts, context, multipliers, vocab,
                                offsets, 0, heads_per_ngram)
                            assert actual == expected
    # Signed token IDs and an EOS exactly on the context/query boundary.
    ids = [-7, 0, -2, 5]
    starts = [0, 0, 4]
    context = [[-4, 0, -1], [-2, 3, 0]]
    multipliers = [(1 << 63) + 3, 77, 99, 123]
    vocab = [19, 23, 29]
    offsets = [-19, 1, 27]
    assert _ngram_finite_window(ids, starts, context, multipliers, vocab,
                                offsets, 0, 1) == _ngram_golden(
                                    ids, starts, context, multipliers, vocab,
                                    offsets, 0, 1)


def test_independent_storage_address_overlap_cpu():
    lib = os.environ.get("QWEN38_PLE_SYCL_TEST_LIBRARY")
    if not lib or not Path(lib).is_file():
        pytest.skip("Set QWEN38_PLE_SYCL_TEST_LIBRARY")
    torch.ops.load_library(lib)
    storage = np.arange(20, dtype=np.int64)
    first = torch.from_numpy(storage[:8])
    overlapping = torch.from_numpy(storage[4:12])
    adjacent = torch.from_numpy(storage[8:16])
    assert not torch._C._is_alias_of(first, overlapping)
    operation = torch.ops._qwen38_ple_sycl_test.ple_test_no_address_alias
    with pytest.raises(RuntimeError, match="address range"):
        operation(first, overlapping)
    assert operation(first, adjacent)


def test_spec_rollback_window_cpu():
    """After accepting `a` candidates, the next history starts after `a`."""
    for length in range(1, 6):
        for num_spec in range(0, 6):
            capacity = length + num_spec
            prior = list(range(100, 100 + length))
            for query_len in range(num_spec + 2):
                candidates = list(range(200, 200 + query_len))
                cached = prior + [-1] * num_spec
                if not query_len:
                    # Empty requests have no update, regardless of padding.
                    assert cached == prior + [-1] * num_spec
                    continue
                extended = prior + candidates
                update_len = length + query_len - 1
                for position in range(update_len):
                    cached[position] = extended[position + 1]
                for accepted in range(1, query_len + 1):
                    rollback = accepted - 1
                    actual = cached[rollback:rollback + length]
                    expected = extended[accepted:accepted + length]
                    assert actual == expected
                assert len(cached) == capacity


def test_ngram_eos_and_embedding(ops):
    ids = [4, 0, 5, 6, 8, 9, 0, 10]
    starts = [0, 4, 8]
    context = [[11, 12, 0], [7, 8, 9]]
    multipliers = [8191, 65537, 131071, 524287]
    vocab = [19, 19, 23, 23, 29, 29]
    offsets = [0, 19, 38, 61, 84, 113]
    out = torch.empty((len(ids), len(vocab)), dtype=torch.int64, device="xpu")
    ops.ple_ngram_ids(_xpu(ids, torch.int64), _xpu(starts, torch.int64),
                      _xpu(context, torch.int64),
                      _xpu(multipliers, torch.int64), _xpu(vocab, torch.int64),
                      _xpu(offsets, torch.int64), out, 0, 2)
    expected = torch.tensor(
        _ngram_golden(ids, starts, context, multipliers, vocab, offsets, 0, 2))
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)

    torch.manual_seed(17)
    weight = torch.randn((20, 5), dtype=torch.float16, device="xpu")
    gathered = torch.empty((len(ids), len(vocab) * 5),
                           dtype=torch.float16,
                           device="xpu")
    ops.ple_embedding_gather(out, weight, _xpu([13], torch.int64),
                             _xpu([17], torch.int64), gathered)
    golden = torch.zeros_like(gathered.cpu())
    weight_cpu = weight.cpu()
    for token in range(out.size(0)):
        for head in range(out.size(1)):
            idx = int(expected[token, head]) - 13
            if 0 <= idx < 17:
                golden[token, head * 5:(head + 1) * 5] = weight_cpu[idx]
    torch.testing.assert_close(gathered.cpu(), golden, atol=0, rtol=0)


@pytest.mark.parametrize("rows,width,group_size", [
    (1, 10240, 2560),
    (4, 10240, 2560),
    (8, 10240, 2560),
    (2, 84, 21),
    (1, 1536, 384),
])
def test_grouped_norm(ops, rows, width, group_size):
    torch.manual_seed(rows + width)
    x = torch.randn((rows, width), dtype=torch.float16, device="xpu")
    w = torch.randn((width, ), dtype=torch.float16, device="xpu") * 0.1
    y = torch.empty_like(x)
    ops.ple_grouped_norm(x, w, y, 1e-6, group_size)
    xf = x.cpu().float().reshape(rows, -1, group_size)
    wf = w.cpu().float().reshape(-1, group_size)
    expected = (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6) *
                (1 + wf)).reshape(rows, width).half()
    _assert_half(y, expected)


@pytest.mark.parametrize("rows,heads,hidden", [(1, 4, 2560), (6, 4, 2560),
                                               (2, 3, 32)])
def test_gate_and_residual(ops, rows, heads, hidden):
    torch.manual_seed(rows + hidden)
    key = torch.randn((rows, heads * hidden),
                      dtype=torch.float16,
                      device="xpu")
    query = torch.randn_like(key)
    gate = torch.empty((rows, heads), dtype=torch.float16, device="xpu")
    ops.ple_score_gate(key, query, gate, hidden)
    dot = (key.cpu().float().reshape(rows, heads, hidden) *
           query.cpu().float().reshape(rows, heads, hidden)).sum(-1)
    dot /= math.sqrt(hidden)
    signed_root = torch.sign(dot) * torch.sqrt(dot.abs().clamp_min(1e-6))
    signed_root = torch.where(dot == 0, 0, signed_root)
    expected_gate = torch.sigmoid(signed_root).half()
    _assert_half(gate, expected_gate)

    value = torch.randn((rows, hidden), dtype=torch.float16, device="xpu")
    raw = torch.empty((rows, heads * hidden),
                      dtype=torch.float16,
                      device="xpu")
    ops.ple_gated_value(gate, value, raw, heads)
    expected_raw = (gate.cpu().float().unsqueeze(-1) *
                    value.cpu().float().unsqueeze(1)).reshape(rows, -1).half()
    _assert_half(raw, expected_raw, atol=0, rtol=0)
    residual = torch.randn_like(raw)
    output = torch.empty_like(raw)
    ops.ple_residual_add(raw, residual, output)
    expected = (raw.cpu().float() + residual.cpu().float()).half()
    _assert_half(output, expected, atol=0, rtol=0)


def test_fused_gated_value_norm(ops):
    torch.manual_seed(31)
    gate = torch.rand((1, 4), dtype=torch.float16, device="xpu")
    value = torch.randn((1, 2560), dtype=torch.float16, device="xpu")
    weight = torch.randn((10240, ), dtype=torch.float16, device="xpu") * 0.1
    raw = torch.empty((1, 10240), dtype=torch.float16, device="xpu")
    normalized = torch.empty_like(raw)
    ops.ple_gated_value_grouped_norm(gate, value, weight, raw, normalized,
                                     1e-6)
    expected_raw = (gate.cpu().float().unsqueeze(-1) *
                    value.cpu().float().unsqueeze(1)).half()
    torch.testing.assert_close(raw.cpu().reshape(1, 4, 2560),
                               expected_raw,
                               atol=0,
                               rtol=0)
    rounded = expected_raw.float()
    expected_norm = (
        rounded * torch.rsqrt(rounded.square().mean(-1, keepdim=True) + 1e-6) *
        (1 + weight.cpu().float().reshape(4, 2560))).half()
    _assert_half(normalized.reshape(4, 2560), expected_norm.reshape(4, 2560))


def _state_view(slots, width, capacity, dim_first, dtype):
    if dim_first:
        base = torch.randn((slots, width, capacity + 3), dtype=dtype)
        return base[:, :, 3:]
    base = torch.randn((slots, capacity + 3, width), dtype=dtype)
    return base[:, 3:, :]


def _conv_golden(x, state, weight, slots, initial, starts, accepted, dilation,
                 dim_first, null_id, mode, num_spec):
    x = x.cpu().clone()
    state = state.cpu().clone()
    weight = weight.cpu().clone()
    output = torch.empty_like(x)
    width = x.shape[1]
    kernel_size = weight.shape[1]
    length = (kernel_size - 1) * dilation
    requests = len(slots)
    for request in range(requests):
        begin = request if mode == "decode" else starts[request]
        end = begin + 1 if mode == "decode" else starts[request + 1]
        slot = slots[request]
        if slot == null_id or slot < 0 or slot >= state.shape[0]:
            output[begin:end] = 0
            continue
        if begin == end:
            continue
        for channel in range(width):

            def get(position, slot=slot, channel=channel):
                return float(state[slot, channel,
                                   position] if dim_first else state[slot,
                                                                     position,
                                                                     channel])

            def put(position, value, slot=slot, channel=channel):
                if dim_first:
                    state[slot, channel, position] = value
                else:
                    state[slot, position, channel] = value

            rollback = accepted[request] - 1 if mode == "spec" else 0
            history = [
                get(rollback +
                    p) if mode == "spec" or initial[request] else 0.0
                for p in range(length)
            ]
            extended = history.copy()
            for token in range(begin, end):
                sample = float(x[token, channel])
                current = history + [sample]
                summed = sum(
                    float(weight[channel, k]) * current[k * dilation]
                    for k in range(kernel_size))
                output[token, channel] = (summed / (1 + math.exp(-summed)))
                extended.append(sample)
                history = current[1:]
            if mode == "spec":
                keep = min(max(length + end - begin - 1, 0), length + num_spec)
                for position in range(keep):
                    put(position, extended[position + 1])
            else:
                for position, sample in enumerate(history):
                    put(position, sample)
    return output, state


@pytest.mark.parametrize("trusted", (False, True))
@pytest.mark.parametrize("mode", ("decode", "prefill", "spec"))
def test_lazy_state_rejected_without_write(ops, trusted, mode):
    x = torch.ones((2, 3), dtype=torch.float16, device="xpu")
    weight = torch.ones((3, 2), dtype=torch.float16, device="xpu")
    backing = torch.full((3, 3, 6), 7, dtype=torch.float16, device="xpu")
    state = backing._neg_view()
    output = torch.full_like(x, 9)
    slots = _xpu([1, 2], torch.int32)
    starts = _xpu([0, 1, 2], torch.int32)
    operation = getattr(ops, "ple_short_conv_" + mode +
                        ("_trusted" if trusted else ""))
    null_id = 0 if trusted else -1
    with pytest.raises(RuntimeError, match="lazy"):
        if mode == "decode":
            operation(x, state, weight, slots, _xpu([True, True]),
                      output, 1, True, null_id)
        elif mode == "prefill":
            operation(x, starts, state, weight, slots, _xpu([True, True]),
                      output, 1, True, null_id)
        else:
            operation(x, starts, state, weight, slots, _xpu([1, 1], torch.int32),
                      output, 3, 1, True, null_id)
    assert torch.all(backing == 7).item()
    assert torch.all(output == 9).item()


@pytest.mark.parametrize("trusted", (False, True))
def test_spec_previous_acceptance_exceeds_current_length(ops, trusted):
    torch.manual_seed(731)
    x = torch.randn((2, 3), dtype=torch.float16, device="xpu")
    weight = torch.randn((3, 2), dtype=torch.float16, device="xpu") * 0.1
    state = torch.randn((3, 3, 6), dtype=torch.float16, device="xpu")
    null_id = 0 if trusted else -1
    slots, starts, accepted = [1, null_id], [0, 1, 2], [4, 0]
    expected, final_state = _conv_golden(
        x, state, weight, slots, None, starts, accepted, 1, True, null_id,
        "spec", 3)
    output = torch.empty_like(x)
    operation = getattr(ops, "ple_short_conv_spec" +
                        ("_trusted" if trusted else ""))
    operation(x, _xpu(starts, torch.int32), state, weight,
              _xpu(slots, torch.int32), _xpu(accepted, torch.int32), output,
              3, 1, True, null_id)
    _assert_half(output, expected)
    _assert_half(state, final_state)


@pytest.mark.parametrize("mode,dim_first,state_dtype", [
    ("decode", True, torch.float16),
    ("decode", False, torch.float32),
    ("prefill", True, torch.float16),
    ("prefill", False, torch.float32),
    ("spec", True, torch.float16),
    ("spec", False, torch.float32),
])
def test_short_conv_trusted_and_untrusted(ops, mode, dim_first, state_dtype):
    torch.manual_seed(91)
    starts = [0, 3, 4, 7] if mode != "decode" else None
    accepted = [2, 1, 3] if mode == "spec" else None
    rows = 3 if mode == "decode" else 7
    num_spec = 4
    dilation = 2
    width = 13
    state_len = 6
    capacity = state_len + (num_spec if mode == "spec" else 0)
    x = torch.randn((rows, width), dtype=torch.float16, device="xpu")
    w = torch.randn((width, 4), dtype=torch.float16, device="xpu") * 0.1
    state_cpu = _state_view(4, width, capacity, dim_first, state_dtype)
    initial = [True, False, True]
    for suffix in ("", "_trusted"):
        null_id = 0 if suffix else -1
        slots = [1, null_id, 2]
        golden, final_state = _conv_golden(
            x,
            state_cpu,
            w,
            slots,
            initial,
            starts,
            accepted,
            dilation,
            dim_first,
            null_id,
            mode,
            num_spec,
        )
        if dim_first:
            backing = torch.full((4, width, capacity + 3),
                                 -19,
                                 dtype=state_dtype,
                                 device="xpu")
            state = backing[:, :, 3:]
        else:
            backing = torch.full((4, capacity + 3, width),
                                 -19,
                                 dtype=state_dtype,
                                 device="xpu")
            state = backing[:, 3:, :]
        state.copy_(state_cpu.to("xpu"))
        output = torch.empty_like(x)
        indices = _xpu(slots, torch.int32)
        if mode == "decode":
            getattr(ops, "ple_short_conv_decode" + suffix)(
                x,
                state,
                w,
                indices,
                _xpu(initial, torch.bool),
                output,
                dilation,
                dim_first,
                null_id,
            )
        elif mode == "prefill":
            getattr(ops, "ple_short_conv_prefill" + suffix)(
                x,
                _xpu(starts, torch.int32),
                state,
                w,
                indices,
                _xpu(initial, torch.bool),
                output,
                dilation,
                dim_first,
                null_id,
            )
        else:
            getattr(ops, "ple_short_conv_spec" + suffix)(
                x,
                _xpu(starts, torch.int32),
                state,
                w,
                indices,
                _xpu(accepted, torch.int32),
                output,
                num_spec,
                dilation,
                dim_first,
                null_id,
            )
        _assert_half(output, golden, atol=3e-3, rtol=2e-3)
        torch.testing.assert_close(state.cpu(),
                                   final_state,
                                   atol=2e-3,
                                   rtol=1e-3)
        padding = backing[:, :, :3] if dim_first else backing[:, :3, :]
        torch.testing.assert_close(padding.cpu(),
                                   torch.full_like(padding.cpu(), -19),
                                   atol=0,
                                   rtol=0)


def test_preflight_rejects_without_state_or_output_write(ops):
    x = torch.ones((2, 3), dtype=torch.float16, device="xpu")
    weights = torch.ones((3, 2), dtype=torch.float16, device="xpu")
    state = torch.full((3, 3, 3), 7, dtype=torch.float16, device="xpu")
    output = torch.full_like(x, 9)
    slots = _xpu([1, 1], torch.int32)
    with pytest.raises(RuntimeError, match="duplicate state index"):
        ops.ple_short_conv_decode(
            x,
            state,
            weights,
            slots,
            _xpu([True, True], torch.bool),
            output,
            1,
            True,
            -1,
        )
    torch.testing.assert_close(state.cpu(), torch.full_like(state.cpu(), 7))
    torch.testing.assert_close(output.cpu(), torch.full_like(output.cpu(), 9))

    # Acceptance belongs to the previous query and may exceed the new length.
    expected, expected_state = _conv_golden(
        x, state, weights, [1, 2], None, [0, 1, 2], [2, 1],
        1, True, -1, "spec", 2)
    ops.ple_short_conv_spec(
        x, _xpu([0, 1, 2], torch.int32), state, weights,
        _xpu([1, 2], torch.int32), _xpu([2, 1], torch.int32),
        output, 2, 1, True, -1)
    _assert_half(output, expected)
    torch.testing.assert_close(state.cpu(), expected_state, atol=0, rtol=0)
    state.fill_(7)
    output.fill_(9)

    # A failing spec metadata check must also precede the first state update.
    with pytest.raises(RuntimeError, match="num_accepted_tokens"):
        ops.ple_short_conv_spec(
            x,
            _xpu([0, 1, 2], torch.int32),
            state,
            weights,
            _xpu([1, 2], torch.int32),
            _xpu([4, 1], torch.int32),
            output,
            2,
            1,
            True,
            -1,
        )
    torch.testing.assert_close(state.cpu(), torch.full_like(state.cpu(), 7))
    torch.testing.assert_close(output.cpu(), torch.full_like(output.cpu(), 9))


@pytest.mark.parametrize("offset_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("accepted_dtype", [torch.int32, torch.int64])
def test_spec_mixed_metadata_dtypes(ops, offset_dtype, accepted_dtype):
    torch.manual_seed(19)
    x = torch.randn((5, 7), dtype=torch.float16, device="xpu")
    weight = torch.randn((7, 3), dtype=torch.float16, device="xpu")
    cache = torch.randn((3, 7, 8), dtype=torch.float32, device="xpu")
    starts = [0, 3, 5]
    slots = [1, 2]
    accepted = [2, 1]
    expected, expected_cache = _conv_golden(
        x,
        cache,
        weight,
        slots,
        [True, True],
        starts,
        accepted,
        2,
        True,
        -1,
        "spec",
        4,
    )
    output = torch.empty_like(x)
    ops.ple_short_conv_spec(
        x,
        _xpu(starts, offset_dtype),
        cache,
        weight,
        _xpu(slots, torch.int64),
        _xpu(accepted, accepted_dtype),
        output,
        4,
        2,
        True,
        -1,
    )
    _assert_half(output, expected, atol=3e-3, rtol=2e-3)
    torch.testing.assert_close(cache.cpu(), expected_cache, atol=0, rtol=0)


def test_final_esimd_differential(ops, esimd_ops):
    torch.manual_seed(27)
    x = torch.randn((4, 10240), dtype=torch.float16, device="xpu")
    w = torch.randn((10240, ), dtype=torch.float16, device="xpu") * 0.1
    ours = torch.empty_like(x)
    reference = torch.empty_like(x)
    ops.ple_grouped_norm(x, w, ours, 1e-6, 2560)
    esimd_ops.ple_grouped_norm(x, w, reference, 1e-6, 2560)
    _assert_half(ours, reference)

    gate = torch.rand((1, 4), dtype=torch.float16, device="xpu")
    value = torch.randn((1, 2560), dtype=torch.float16, device="xpu")
    ours_raw = torch.empty((1, 10240), dtype=torch.float16, device="xpu")
    ours_norm = torch.empty_like(ours_raw)
    ref_raw = torch.empty_like(ours_raw)
    ref_norm = torch.empty_like(ours_raw)
    ops.ple_gated_value_grouped_norm(gate, value, w, ours_raw, ours_norm, 1e-6)
    esimd_ops.ple_gated_value_grouped_norm(gate, value, w, ref_raw, ref_norm,
                                           1e-6)
    _assert_half(ours_raw, ref_raw, atol=0, rtol=0)
    _assert_half(ours_norm, ref_norm)

    spec_input = torch.randn((3, 11), dtype=torch.float16, device="xpu")
    conv_weight = torch.randn((11, 4), dtype=torch.float16, device="xpu")
    starts = _xpu([0, 3], torch.int32)
    indices = _xpu([1], torch.int32)
    accepted = _xpu([2], torch.int32)
    cache = torch.randn((3, 11, 10), dtype=torch.float16, device="xpu")
    ours_cache = cache.clone()
    ref_cache = cache.clone()
    ours_out = torch.empty_like(spec_input)
    ref_out = torch.empty_like(spec_input)
    ops.ple_short_conv_spec_trusted(spec_input, starts, ours_cache,
                                    conv_weight, indices, accepted, ours_out,
                                    4, 2, True, 0)
    esimd_ops.ple_short_conv_spec_trusted(spec_input, starts, ref_cache,
                                          conv_weight, indices, accepted,
                                          ref_out, 4, 2, True, 0)
    _assert_half(ours_out, ref_out, atol=3e-3, rtol=2e-3)
    _assert_half(ours_cache, ref_cache, atol=0, rtol=0)


def test_final_esimd_schema_contract(ops, esimd_ops):
    del ops, esimd_ops
    names = (
        "ple_ngram_ids", "ple_embedding_gather", "ple_grouped_norm",
        "ple_score_gate", "ple_gated_value", "ple_gated_value_grouped_norm",
        "ple_residual_add", "ple_short_conv_decode",
        "ple_short_conv_decode_trusted", "ple_short_conv_prefill",
        "ple_short_conv_prefill_trusted", "ple_short_conv_spec",
        "ple_short_conv_spec_trusted",
    )
    for name in names:
        sycl = torch._C._jit_get_schemas_for_operator(
            f"_qwen38_ple_sycl_test::{name}")
        final = torch._C._jit_get_schemas_for_operator(
            f"custom_esimd_kernels_vllm::{name}")
        assert len(sycl) == len(final) == 1
        normalized = str(sycl[0]).replace(
            "_qwen38_ple_sycl_test::", "custom_esimd_kernels_vllm::")
        assert normalized == str(final[0])


def test_nondefault_stream_and_dropped_inputs(ops):
    torch.manual_seed(55)
    x_cpu = torch.randn((4, 32), dtype=torch.float16)
    w_cpu = torch.randn((32, ), dtype=torch.float16) * 0.1
    xf = x_cpu.float().reshape(4, 2, 16)
    expected = (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6) *
                (1 + w_cpu.float().reshape(2, 16))).reshape(4, 32).half()
    stream = torch.xpu.Stream()
    with torch.xpu.stream(stream):
        x = x_cpu.to("xpu")
        w = w_cpu.to("xpu")
        output = torch.empty_like(x)
        ops.ple_grouped_norm(x, w, output, 1e-6, 16)
        del x, w
        pressure = torch.empty((4_000_000, ),
                               dtype=torch.float16,
                               device="xpu")
        pressure.fill_(1)
    torch.xpu.current_stream().wait_stream(stream)
    _assert_half(output, expected)


def test_spec_cross_stream_handoff_and_lifetime(ops):
    """The scheduler orders producers; native ops own inputs in flight."""
    torch.manual_seed(72)
    x_cpu = torch.randn((2, 17), dtype=torch.float16)
    weights_cpu = torch.randn((17, 3), dtype=torch.float16) * 0.1
    state_cpu = torch.randn((3, 17, 5), dtype=torch.float16)
    golden, golden_state = _conv_golden(
        x_cpu, state_cpu, weights_cpu, [1], [True], [0, 2], [1], 1,
        True, 0, "spec", 3)
    producer = torch.xpu.Stream()
    consumer = torch.xpu.Stream()
    with torch.xpu.stream(producer):
        x = x_cpu.to("xpu")
        weights = weights_cpu.to("xpu")
        state = state_cpu.to("xpu")
        starts = _xpu([0, 2], torch.int32)
        indices = _xpu([1], torch.int32)
        accepted = _xpu([1], torch.int32)
    with torch.xpu.stream(consumer):
        consumer.wait_stream(producer)
        output = torch.empty_like(x)
        ops.ple_short_conv_spec_trusted(x, starts, state, weights, indices,
                                        accepted, output, 3, 1, True, 0)
        del x, weights, starts, indices, accepted
        torch.empty((4_000_000, ), dtype=torch.float16,
                    device="xpu").fill_(1)
    torch.xpu.current_stream().wait_stream(consumer)
    _assert_half(output, golden, atol=3e-3, rtol=2e-3)
    torch.testing.assert_close(state.cpu(), golden_state, atol=0, rtol=0)
