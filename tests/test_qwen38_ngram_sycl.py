# SPDX-License-Identifier: Apache-2.0
"""Independent signed-overflow hashing and raw-bit host-USM lookup tests."""

import gc
import importlib
import os
import random

import pytest
import torch

VOCAB = [
    20000003,
    20000023,
    20000033,
    20000047,
    20000059,
    20000063,
    20000069,
    20000077,
    20000081,
    20000093,
    20000107,
    20000147,
    20000153,
    20000159,
    20000161,
    20000171,
]
MULT = [23703573157769, 20109073645365, 8052911324071]


@pytest.fixture(scope="module")
def ops():
    if not torch.xpu.is_available():
        pytest.skip("requires XPU")
    library = os.environ.get("QWEN38_SYCL_LIBRARY")
    if library:
        torch.ops.load_library(library)
    else:
        importlib.import_module("vllm_xpu_kernels.qwen38")
    return torch.ops._qwen38_C


def hash_reference(inputs, history, multipliers):
    mask = (1 << 64) - 1
    result = []
    for current, (previous_2, previous) in zip(inputs, history):
        bigram = ((current * multipliers[0]) & mask) ^ (
            (previous * multipliers[1]) & mask
        )
        trigram = bigram ^ ((previous_2 * multipliers[2]) & mask)
        offset = 0
        row = []
        for head, vocab in enumerate(VOCAB):
            mixed = bigram if head < 8 else trigram
            signed = mixed if mixed < (1 << 63) else mixed - (1 << 64)
            row.append(signed % vocab + offset)
            offset += vocab
        result.append(row)
    return torch.tensor(result, dtype=torch.int64)


@pytest.mark.parametrize("m", [1, 2, 8, 33])
@pytest.mark.parametrize("pattern", ["random", "minmax", "zero"])
def test_hash_matches_python_signed_wraparound(ops, m, pattern):
    rng = random.Random(743)
    values = [rng.randrange(-(1 << 63), 1 << 63) for _ in range(m * 3)]
    mult = MULT
    if pattern == "minmax":
        values = [-(1 << 63), (1 << 63) - 1, -1] * m
        mult = [1, 0, 0]  # Exercise abs(INT64_MIN), not signed-overflow UB.
    elif pattern == "zero":
        mult = [0, 0, 0]
    inputs = values[:m]
    history = [values[m + row * 2 : m + row * 2 + 2] for row in range(m)]
    x = torch.tensor(inputs, device="xpu")
    ctx = torch.tensor(history, device="xpu")
    multipliers = torch.tensor(mult, device="xpu")
    out = torch.empty(m, 16, dtype=torch.int64, device="xpu")
    ops.ngram_decode_ids(x, ctx, multipliers, out)
    torch.testing.assert_close(
        out.cpu(), hash_reference(inputs, history, mult), rtol=0, atol=0
    )


@pytest.mark.parametrize("eos", [2, -(1 << 63), (1 << 63) - 1])
def test_eos_hash_matches_independent_segment_reference(ops, eos):
    # 全部 EOS 组合，另含有符号溢出边界；参考不调用被测实现。
    rows = [
        (
            eos if bits & 1 else 271,
            eos if bits & 2 else 11316,
            eos if bits & 4 else 13,
        )
        for bits in range(8)
    ]
    rows += [(-1, -(1 << 63), (1 << 63) - 1)]
    inputs = [row[0] for row in rows]
    history = [list(row[1:]) for row in rows]
    effective = []
    for current, (previous_2, previous) in zip(inputs, history):
        if current == eos:
            effective.append([eos, eos])
        elif previous == eos:
            effective.append([eos, previous])
        else:
            effective.append([previous_2, previous])
    x = torch.tensor(inputs, device="xpu")
    ctx = torch.tensor(history, device="xpu")
    multipliers = torch.tensor(MULT, device="xpu")
    out = torch.empty(len(inputs), 16, dtype=torch.int64, device="xpu")
    ops.ngram_decode_ids_eos(x, ctx, multipliers, out, eos)
    torch.testing.assert_close(
        out.cpu(), hash_reference(inputs, effective, MULT), rtol=0, atol=0
    )


def test_eos_hash_graph_replay_uses_current_device_inputs(ops):
    x = torch.tensor([271], device="xpu")
    ctx = torch.tensor([[11316, 13]], device="xpu")
    mult = torch.tensor(MULT, device="xpu")
    out = torch.empty(1, 16, dtype=torch.int64, device="xpu")
    ops.ngram_decode_ids_eos(x, ctx, mult, out, 2)
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        ops.ngram_decode_ids_eos(x, ctx, mult, out, 2)
    for current, history, effective in [
        (271, [11316, 13], [11316, 13]),
        (271, [11316, 2], [2, 2]),
        (2, [11316, 13], [2, 2]),
        (271, [2, 13], [2, 13]),
    ]:
        x.copy_(torch.tensor([current], device="xpu"))
        ctx.copy_(torch.tensor([history], device="xpu"))
        graph.replay()
        torch.testing.assert_close(
            out.cpu(),
            hash_reference([current], [effective], MULT),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("with_eos", [False, True])
def test_hash_rejects_alias_before_submission(ops, with_eos):
    storage = torch.full((16,), 77, dtype=torch.int64, device="xpu")
    ctx = torch.zeros(1, 2, dtype=torch.int64, device="xpu")
    mult = torch.tensor(MULT, device="xpu")
    with pytest.raises(RuntimeError, match="overlap|single memory location"):
        if with_eos:
            ops.ngram_decode_ids_eos(
                storage[:1], ctx, mult, storage.view(1, 16), 2
            )
        else:
            ops.ngram_decode_ids(storage[:1], ctx, mult, storage.view(1, 16))
    assert torch.all(storage == 77).item()


def test_hash_rejects_independent_dlpack_storage_alias(ops):
    storage = torch.full((16,), 77, dtype=torch.int64, device="xpu")
    inputs = torch.from_dlpack(storage[:1])
    assert inputs.data_ptr() == storage.data_ptr()
    assert not torch._C._overlaps(inputs, storage)
    ctx = torch.zeros((1, 2), dtype=torch.int64, device="xpu")
    mult = torch.tensor(MULT, device="xpu")
    with pytest.raises(RuntimeError, match="overlap|single memory location"):
        ops.ngram_decode_ids(inputs, ctx, mult, storage.view(1, 16))
    assert torch.all(storage == 77).item()


def pinned_bits(rows, dtype):
    weight = torch.empty(rows, 160, dtype=dtype, pin_memory=True)
    # Include signed zero, subnormal values, infinities and NaN payloads;
    # lookup must not perform a floating-point conversion on these bits.
    values = torch.randint(-32768, 32768, (rows, 160), dtype=torch.int16)
    weight.view(torch.int16).copy_(values)
    return weight


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("chunks", [1, 2, 3, 8])
@pytest.mark.parametrize("m", [0, 1, 8])
def test_host_lookup_preserves_bits_and_chunk_boundaries(ops, dtype, chunks, m):
    weights = [
        pinned_bits(9 if i < chunks - 1 else 7, dtype) for i in range(chunks)
    ]
    rows = sum(w.shape[0] for w in weights)
    vocab_start = 1003
    ids_cpu = torch.tensor(
        [
            vocab_start - 1,
            vocab_start,
            vocab_start + 6,
            vocab_start + rows - 1,
            vocab_start + rows,
            vocab_start + 8,
            vocab_start + 9,
            -1,
        ]
        * (m * 2),
        dtype=torch.int64,
    ).reshape(m, 16)
    ids = ids_cpu.to("xpu")
    out = torch.empty(m, 2560, dtype=dtype, device="xpu")
    if chunks == 1:
        returned = ops.ngram_host_lookup(
            weights[0], ids, out, vocab_start, vocab_start + rows
        )
    else:
        returned = ops.ngram_host_lookup_chunked(
            weights, ids, out, vocab_start, vocab_start + rows
        )
    assert returned.data_ptr() == out.data_ptr()
    table = torch.cat([w.view(torch.int16) for w in weights])
    expected = torch.zeros(m * 16, 160, dtype=torch.int16)
    for index, global_id in enumerate(ids_cpu.flatten().tolist()):
        if vocab_start <= global_id < vocab_start + rows:
            expected[index].copy_(table[global_id - vocab_start])
    torch.testing.assert_close(
        out.cpu().view(torch.int16).reshape(-1, 160), expected, rtol=0, atol=0
    )


def test_unpinned_weight_rejected_before_output_write(ops):
    weight = torch.ones(10, 160, dtype=torch.float16)
    ids = torch.zeros(1, 16, dtype=torch.int64, device="xpu")
    out = torch.full((1, 2560), 123, dtype=torch.float16, device="xpu")
    with pytest.raises(RuntimeError, match="pinned host"):
        ops.ngram_host_lookup(weight, ids, out, 0, 10)
    assert torch.all(out == 123).item()


def test_later_bad_chunk_rejected_before_any_submission(ops):
    weights = [pinned_bits(9, torch.float16), pinned_bits(10, torch.float16)]
    ids = torch.zeros(1, 16, dtype=torch.int64, device="xpu")
    out = torch.full((1, 2560), 123, dtype=torch.float16, device="xpu")
    with pytest.raises(RuntimeError, match="equal full sizes"):
        ops.ngram_host_lookup_chunked(weights, ids, out, 0, 19)
    assert torch.all(out == 123).item()


def test_cross_stream_retains_device_and_pinned_allocations(ops):
    stream = torch.xpu.Stream()
    weights = [pinned_bits(9, torch.float16), pinned_bits(7, torch.float16)]
    expected = (
        weights[1][2].view(torch.int16).clone().repeat(16).reshape(1, 2560)
    )
    ids = torch.full((1, 16), 11, dtype=torch.int64, device="xpu")
    stream.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(stream):
        ids = ids + 0
        out = torch.empty(1, 2560, dtype=torch.float16, device="xpu")
        ops.ngram_host_lookup_chunked(weights, ids, out, 0, 16)
    del weights, ids
    gc.collect()
    pressure = [pinned_bits(9, torch.float16) for _ in range(32)]
    torch.xpu.current_stream().wait_stream(stream)
    torch.testing.assert_close(
        out.cpu().view(torch.int16), expected, rtol=0, atol=0
    )
    del pressure
