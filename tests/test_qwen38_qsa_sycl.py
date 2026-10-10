# SPDX-License-Identifier: Apache-2.0
"""Independent QSA math, final-ESIMD, and stream-lifetime checks."""

import gc
import importlib.util
import os
import sys
import types
from functools import lru_cache
from pathlib import Path

import pytest
import torch


def _final_reference():
    # Independent math must run without a DSO built for a different Torch ABI.
    if os.environ.get("QWEN38_INDEPENDENT_ONLY") == "1":
        return None
    return pytest.importorskip("custom_esimd_kernels_vllm.qsa_ops")


@lru_cache(maxsize=1)
def _integrated_native():
    path = os.environ["QWEN38_SYCL_LIBRARY"]
    spec = importlib.util.spec_from_file_location("_qwen38_C", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def native():
    if not torch.xpu.is_available():
        pytest.skip("XPU is unavailable")
    if os.environ.get("QWEN38_SYCL_LIBRARY"):
        return _integrated_native()
    path = os.environ.get("QWEN38_QSA_SYCL_LIBRARY")
    if not path:
        pytest.skip("set QWEN38_QSA_SYCL_LIBRARY to the isolated build")
    spec = importlib.util.spec_from_file_location("qwen38_qsa_sycl_dev", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def owner_native():
    if not torch.xpu.is_available():
        pytest.skip("XPU is unavailable")
    if os.environ.get("QWEN38_SYCL_LIBRARY"):
        return _integrated_native()
    path = os.environ.get("QWEN38_QSA_SYCL_OWNER_LIBRARY")
    if not path:
        pytest.skip("set QWEN38_QSA_SYCL_OWNER_LIBRARY")
    spec = importlib.util.spec_from_file_location(
        "qwen38_qsa_sycl_owner_dev", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_provider(monkeypatch, extension):
    package = types.ModuleType("vllm_xpu_kernels")
    package.__path__ = [str(Path(__file__).resolve().parents[1] /
                            "vllm_xpu_kernels")]
    package._qwen38_C = extension
    monkeypatch.setitem(sys.modules, "vllm_xpu_kernels", package)
    monkeypatch.setitem(sys.modules, "vllm_xpu_kernels._qwen38_C", extension)
    provider_path = Path(package.__path__[0]) / "qwen38_qsa.py"
    spec = importlib.util.spec_from_file_location(
        "vllm_xpu_kernels.qwen38_qsa", provider_path)
    provider = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, provider)
    spec.loader.exec_module(provider)
    return provider


def compression_case(rows, dtype, ring_size=8):
    torch.manual_seed(113)
    raw = torch.randn(rows, 1, 128, dtype=dtype)
    raw_pos = torch.arange(rows * 3, dtype=torch.int64).reshape(rows, 1, 3)
    ring = torch.randn(2, ring_size, 1, 128, dtype=dtype)
    ring_pos = torch.arange(2 * ring_size * 3, dtype=torch.int64).reshape(
        2, ring_size, 1, 3)
    table = torch.tensor([[1]], dtype=torch.int32)
    requests = torch.zeros(rows, dtype=torch.int32)
    starts = torch.tensor([0, rows], dtype=torch.int32)
    logical = torch.arange(19, 19 + rows, dtype=torch.int64)
    slots = torch.arange(rows, dtype=torch.int64)
    tensors = (raw, raw_pos, ring, ring_pos, table, requests, starts, logical,
               slots)
    return tuple(t.to("xpu") for t in tensors)


def compression_golden(case, capacity):
    raw, raw_pos, ring, ring_pos, table, reqs, starts, logical, slots = (
        x.cpu() for x in case)
    rows = raw.size(0)
    expected = torch.zeros(rows, 1, 128, dtype=raw.dtype)
    first = torch.zeros(rows, 3, dtype=torch.int64)
    for row in range(rows):
        req = int(reqs[row])
        if req < 0 or req >= starts.numel() - 1:
            continue
        start, end = int(starts[req]), int(starts[req + 1])
        pos = int(logical[row])
        if not (start <= row < end and pos >= 3 and 0 <= slots[row] <
                capacity):
            continue
        chunk_start = pos - (row - start)
        block = int(table[req, 0])
        values = []
        for source_pos in range(pos - 3, pos + 1):
            if source_pos >= chunk_start:
                source_row = start + source_pos - chunk_start
                values.append(raw[source_row, 0].float())
            elif 0 <= block < ring.size(0):
                values.append(ring[block, source_pos % ring.size(1), 0].float())
            else:
                values.append(torch.zeros(128))
        expected[row, 0] = (sum(values) * 0.25).to(raw.dtype)
        if pos - 3 >= chunk_start:
            source_row = start + (pos - 3) - chunk_start
            first[row] = raw_pos[source_row, 0]
        elif 0 <= block < ring.size(0):
            first[row] = ring_pos[block, (pos - 3) % ring.size(1), 0]
    return expected, first


@pytest.mark.parametrize("rows", range(1, 9))
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_group_compression_golden_and_final_esimd(native, rows, dtype):
    reference = _final_reference()

    case = compression_case(rows, dtype)
    pooled = torch.empty_like(case[0])
    first = torch.empty(rows, 3, dtype=torch.int64, device="xpu")
    returned = native.group_compress_v2(*case, pooled, first, 4, 100, True)
    assert returned.data_ptr() == pooled.data_ptr()
    expected, expected_first = compression_golden(case, 100)
    torch.testing.assert_close(pooled.cpu(), expected, atol=0, rtol=0)
    torch.testing.assert_close(first.cpu(), expected_first, atol=0, rtol=0)
    if reference is None:
        return
    reference_pooled = torch.empty_like(pooled)
    reference_first = torch.empty_like(first)
    reference.qsa_group_compress_v2(*case, reference_pooled,
                                    reference_first, 4, 100, True)
    torch.testing.assert_close(pooled.cpu(), reference_pooled.cpu(),
                               atol=0, rtol=0)
    torch.testing.assert_close(first.cpu(), reference_first.cpu(),
                               atol=0, rtol=0)


def test_group_compression_rejects_unproven_history_before_write(native):
    case = compression_case(2, torch.float16)
    pooled = torch.full_like(case[0], 7)
    first = torch.full((2, 3), 11, dtype=torch.int64, device="xpu")
    with pytest.raises(RuntimeError, match="historical ring proof"):
        native.group_compress_v2(*case, pooled, first, 4, 100, False)
    assert bool((pooled == 7).all())
    assert bool((first == 11).all())


def test_group_compression_rejects_alias_before_write(native):
    case = compression_case(1, torch.float16)
    first = torch.empty((1, 3), dtype=torch.int64, device="xpu")
    with pytest.raises(RuntimeError, match="aliases"):
        native.group_compress_v2(*case, case[0], first, 4, 100, True)


def test_group_compression_packed_key_position_views(native):
    case = list(compression_case(2, torch.float16))
    # Real ring storage interleaves 256 key bytes and 24 position bytes.
    backing = torch.empty((2, 8, 280), dtype=torch.uint8, device="xpu")
    keys = backing[..., :256].view(torch.float16).unsqueeze(2)
    positions = backing[..., 256:].view(torch.int64).unsqueeze(2)
    keys.copy_(case[2])
    positions.copy_(case[3])
    case[2], case[3] = keys, positions
    pooled = torch.empty_like(case[0])
    first = torch.empty((2, 3), dtype=torch.int64, device="xpu")
    native.group_compress_v2(*case, pooled, first, 4, 100, True)
    expected, expected_first = compression_golden(case, 100)
    torch.testing.assert_close(pooled.cpu(), expected, atol=0, rtol=0)
    torch.testing.assert_close(first.cpu(), expected_first, atol=0, rtol=0)


def test_group_compression_non_power_of_two_ring(native):
    case = compression_case(4, torch.float16, ring_size=6)
    pooled = torch.empty_like(case[0])
    first = torch.empty((4, 3), dtype=torch.int64, device="xpu")
    native.group_compress_v2(*case, pooled, first, 4, 100, True)
    expected, expected_first = compression_golden(case, 100)
    torch.testing.assert_close(pooled.cpu(), expected, atol=0, rtol=0)
    torch.testing.assert_close(first.cpu(), expected_first, atol=0, rtol=0)


def test_group_compression_mixed_requests(native):
    reference = _final_reference()

    case = list(compression_case(4, torch.float16))
    case[4] = torch.tensor([[0], [1]], dtype=torch.int32, device="xpu")
    case[5] = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device="xpu")
    case[6] = torch.tensor([0, 2, 4], dtype=torch.int32, device="xpu")
    case[7] = torch.tensor([19, 20, 31, 32], dtype=torch.int64,
                           device="xpu")
    pooled = torch.empty_like(case[0])
    first = torch.empty((4, 3), dtype=torch.int64, device="xpu")
    native.group_compress_v2(*case, pooled, first, 4, 100, True)
    expected, expected_first = compression_golden(case, 100)
    torch.testing.assert_close(pooled.cpu(), expected, atol=0, rtol=0)
    torch.testing.assert_close(first.cpu(), expected_first, atol=0, rtol=0)
    if reference is None:
        return
    reference_pooled = torch.empty_like(pooled)
    reference_first = torch.empty_like(first)
    reference.qsa_group_compress_v2(*case, reference_pooled,
                                    reference_first, 4, 100, True)
    torch.testing.assert_close(pooled.cpu(), reference_pooled.cpu(),
                               atol=0, rtol=0)
    torch.testing.assert_close(first.cpu(), reference_first.cpu(),
                               atol=0, rtol=0)


def test_group_compression_prefill_4096(native):
    case = compression_case(4096, torch.float16)
    pooled = torch.empty_like(case[0])
    first = torch.empty((4096, 3), dtype=torch.int64, device="xpu")
    native.group_compress_v2(*case, pooled, first, 4, 100, True)
    expected, expected_first = compression_golden(case, 100)
    torch.testing.assert_close(pooled.cpu(), expected, atol=0, rtol=0)
    torch.testing.assert_close(first.cpu(), expected_first, atol=0, rtol=0)


def test_group_compression_nondefault_stream_drop_inputs(native):
    stream = torch.xpu.Stream()
    with torch.xpu.stream(stream):
        case = compression_case(4, torch.float16)
        expected, expected_first = compression_golden(case, 100)
        pooled = torch.empty_like(case[0])
        first = torch.empty((4, 3), dtype=torch.int64, device="xpu")
        native.group_compress_v2(*case, pooled, first, 4, 100, True)
        del case
        gc.collect()
        temporary = [torch.empty((1024, 1024), dtype=torch.float16,
                                 device="xpu") for _ in range(8)]
        del temporary
    stream.synchronize()
    torch.testing.assert_close(pooled.cpu(), expected, atol=0, rtol=0)
    torch.testing.assert_close(first.cpu(), expected_first, atol=0, rtol=0)


def attention_case(rows, heads, length, page_size):
    torch.manual_seed(223 + rows + length)
    q = torch.randn(rows, heads, 256, dtype=torch.float16)
    pages = (length + page_size - 1) // page_size
    packed = torch.randn(pages, 1, page_size, 512, dtype=torch.float16)
    table = torch.arange(pages, dtype=torch.int32).reshape(1, pages)
    requests = torch.zeros(rows, dtype=torch.int32)
    indices = torch.full((rows, 2051), -1, dtype=torch.int32)
    for row in range(rows):
        # Short/holed and fully selected long paths use the same fixed ABI.
        count = 67 if length <= 1024 else 2051
        sample = torch.randperm(length)[:count].to(torch.int32)
        indices[row, :count] = sample
    tensors = (q, packed, indices, table, requests)
    return tuple(t.to("xpu") for t in tensors)


def attention_golden(case):
    q, packed, indices, table, requests = (x.cpu() for x in case)
    page_size = packed.size(2)
    result = torch.zeros_like(q)
    for row in range(q.size(0)):
        logical = indices[row]
        logical = logical[logical >= 0].long()
        physical = table[requests[row], logical // page_size].long()
        kv = packed[physical, 0, logical % page_size]
        scores = q[row].float() @ kv[:, :256].float().T
        probability = torch.softmax(scores * 0.0625, dim=-1)
        result[row] = (probability @ kv[:, 256:].float()).half()
    return result


@pytest.mark.parametrize("rows,heads,length,page_size", [
    (1, 6, 1024, 256), (2, 6, 1024, 256), (8, 6, 1024, 256),
    (1, 6, 131072, 512), (4, 6, 131072, 512),
    (8, 6, 131072, 512), (1, 3, 1024, 256),
    (4, 3, 1024, 256),
])
def test_attention_golden_and_final_esimd(native, rows, heads, length,
                                           page_size):
    reference = _final_reference()

    case = attention_case(rows, heads, length, page_size)
    output = torch.empty_like(case[0])
    partials = torch.empty(rows, heads, 43, 258, dtype=torch.float32,
                           device="xpu")
    returned = native.token_split_attention_v3(*case, page_size, output,
                                               partials)
    assert returned.data_ptr() == output.data_ptr()
    expected = attention_golden(case)
    torch.testing.assert_close(output.cpu(), expected, atol=0.004,
                               rtol=0.004)
    if reference is None:
        return
    ref_out = torch.empty_like(output)
    ref_partial = torch.empty_like(partials)
    reference_op = (reference.sparse_attention_token_split_candidate_q6_v1
                    if heads == 6 else
                    reference.sparse_attention_token_split_candidate_v3)
    reference_op(*case, page_size, ref_out, ref_partial)
    torch.testing.assert_close(output.cpu(), ref_out.cpu(), atol=0.004,
                               rtol=0.004)


def test_attention_full_graph_replays_overallocated_page_table(native):
    # 页表容量由最长上下文决定，不能要求它小于 profiling 的物理 cache。
    q = torch.zeros((1, 6, 256), dtype=torch.float16, device="xpu")
    packed = torch.zeros((1, 1, 256, 512), dtype=torch.float16, device="xpu")
    indices = torch.full((1, 2051), -1, dtype=torch.int32, device="xpu")
    table = torch.tensor([[0, 2, -1, 0]], dtype=torch.int32, device="xpu")
    requests = torch.zeros(1, dtype=torch.int32, device="xpu")
    output = torch.empty_like(q)
    partials = torch.empty((1, 6, 43, 258), dtype=torch.float32, device="xpu")

    def run():
        native.token_split_attention_v3(q, packed, indices, table, requests,
                                         256, output, partials)

    run()
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        run()
    for value in (1, 3):
        packed[..., 256:].fill_(value)
        indices.fill_(-1)
        indices[0, :3].copy_(torch.tensor([0, 256, 512], dtype=torch.int32,
                                         device="xpu"))
        graph.replay()
        torch.testing.assert_close(output.cpu(),
                                   torch.full_like(output.cpu(), value),
                                   atol=0, rtol=0)
    indices.fill_(-1)
    graph.replay()
    assert torch.equal(output.cpu(), torch.zeros_like(output.cpu()))


@pytest.mark.parametrize("rows", [129, 256])
def test_attention_prefill_cpp_chunk_owner(native, rows):
    case = attention_case(rows, 6, 1024, 256)
    output = torch.empty_like(case[0])
    partials = torch.empty(min(rows, 128), 6, 43, 258,
                           dtype=torch.float32, device="xpu")
    native.token_split_attention_v3(*case, 256, output, partials)
    expected = attention_golden(case)
    torch.testing.assert_close(output.cpu(), expected, atol=0.004,
                               rtol=0.004)


@pytest.mark.parametrize("tokens", ["64", "128", "256", "invalid"])
@pytest.mark.parametrize("rows", [9, 129])
@pytest.mark.parametrize("dpas", ["0", "1"])
def test_attention_prefill_partition_override(native, monkeypatch, tokens, rows,
                                               dpas):
    monkeypatch.setenv("QWEN38_QSA_PREFILL_TOKENS_PER_PARTIAL", tokens)
    monkeypatch.setenv("QWEN38_QSA_PREFILL_QK_DPAS", dpas)
    case = attention_case(rows, 6, 4096, 512)
    output = torch.empty_like(case[0])
    partials = torch.empty(min(rows, 128), 6, 43, 258,
                           dtype=torch.float32, device="xpu")
    native.token_split_attention_v3(*case, 512, output, partials)
    torch.testing.assert_close(output.cpu(), attention_golden(case),
                               atol=0.004, rtol=0.004)


def test_attention_prefill_4096_cpp_chunk_owner(native):
    case = attention_case(4096, 6, 128, 256)
    output = torch.empty_like(case[0])
    partials = torch.empty(128, 6, 43, 258, dtype=torch.float32,
                           device="xpu")
    native.token_split_attention_v3(*case, 256, output, partials)
    assert bool(torch.isfinite(output).all())
    samples = [0, 127, 128, 2048, 4095]
    small_case = (case[0][samples], case[1], case[2][samples],
                  case[3], case[4][samples])
    expected = attention_golden(small_case)
    torch.testing.assert_close(output[samples].cpu(), expected,
                               atol=0.004, rtol=0.004)


def test_attention_prefill_4096_all_rows_constant_value_golden(native):
    # An identical V vector at every selected token has an exact analytic
    # attention result independent of Q, K, softmax order, and chunk boundary.
    case = list(attention_case(4096, 6, 128, 256))
    values = (torch.arange(256, device="xpu", dtype=torch.float32) / 256).half()
    case[1][:, :, :, 256:] = values
    output = torch.empty_like(case[0])
    partials = torch.empty(128, 6, 43, 258, dtype=torch.float32,
                           device="xpu")
    native.token_split_attention_v3(*case, 256, output, partials)
    torch.testing.assert_close(
        output.cpu(), values.cpu().expand_as(output.cpu()),
        atol=0.004, rtol=0.004)


def test_attention_rejects_workspace_alias_before_write(native):
    case = attention_case(1, 6, 1024, 256)
    output = torch.full_like(case[0], 7)
    bad_partials = torch.empty((1, 6, 43, 258), dtype=torch.float32,
                               device="xpu")
    with pytest.raises(RuntimeError, match="aliases"):
        native.token_split_attention_v3(case[0], case[1], case[2],
                                        case[3], case[4], 256, case[0],
                                        bad_partials)
    assert bool((output == 7).all())


def test_attention_nondefault_stream_drop_inputs(native):
    stream = torch.xpu.Stream()
    with torch.xpu.stream(stream):
        case = attention_case(2, 6, 1024, 256)
        expected = attention_golden(case)
        output = torch.empty_like(case[0])
        partials = torch.empty(2, 6, 43, 258, dtype=torch.float32,
                               device="xpu")
        native.token_split_attention_v3(*case, 256, output, partials)
        del case
        gc.collect()
        temporary = [torch.empty((1024, 1024), dtype=torch.float16,
                                 device="xpu") for _ in range(8)]
        del temporary
    stream.synchronize()
    torch.testing.assert_close(output.cpu(), expected, atol=0.004,
                               rtol=0.004)


def test_attention_unaligned_packed_cache(native):
    case = list(attention_case(1, 6, 1024, 256))
    packed = case[1]
    storage = torch.empty(packed.numel() + 1, dtype=torch.float16,
                          device="xpu")
    unaligned = storage[1:].view_as(packed)
    unaligned.copy_(packed)
    case[1] = unaligned
    output = torch.empty_like(case[0])
    partials = torch.empty((1, 6, 43, 258), dtype=torch.float32,
                           device="xpu")
    native.token_split_attention_v3(*case, 256, output, partials)
    torch.testing.assert_close(output.cpu(), attention_golden(case),
                               atol=0.004, rtol=0.004)


def test_attention_mixed_request_block_tables(native):
    reference = _final_reference()

    case = list(attention_case(4, 6, 1024, 256))
    second = torch.randn_like(case[1])
    case[1] = torch.cat((case[1], second), dim=0)
    case[3] = torch.arange(8, dtype=torch.int32,
                           device="xpu").reshape(2, 4)
    case[4] = torch.tensor([0, 1, 0, 1], dtype=torch.int32, device="xpu")
    output = torch.empty_like(case[0])
    partials = torch.empty((4, 6, 43, 258), dtype=torch.float32,
                           device="xpu")
    native.token_split_attention_v3(*case, 256, output, partials)
    expected = attention_golden(case)
    torch.testing.assert_close(output.cpu(), expected, atol=0.004,
                               rtol=0.004)
    if reference is None:
        return
    ref_out = torch.empty_like(output)
    reference.sparse_attention_token_split_candidate_q6_v1(
        *case, 256, ref_out, torch.empty_like(partials))
    torch.testing.assert_close(output.cpu(), ref_out.cpu(), atol=0.004,
                               rtol=0.004)


def selection_case(rows, length, page_size, *, scratch_rows=32):
    torch.manual_seed(307 + rows + length)
    pages = (length + 4 * page_size - 1) // (4 * page_size)
    q = torch.randn(rows, 4, 128, dtype=torch.float16, device="xpu")
    cache = torch.randn(pages, page_size, 1, 128, dtype=torch.float16,
                        device="xpu")
    table = torch.arange(pages, device="xpu", dtype=torch.int32).reshape(
        1, pages)
    reqs = torch.zeros(rows, dtype=torch.int32, device="xpu")
    positions = torch.full((rows, ), length - 1, dtype=torch.int64,
                           device="xpu")
    seq_lengths = torch.tensor([length], dtype=torch.int32, device="xpu")
    out = torch.empty(rows, 2051, dtype=torch.int32, device="xpu")
    score_a = torch.empty(min(rows, scratch_rows), 32, 512,
                          dtype=torch.float32,
                          device="xpu")
    index_a = torch.empty(min(rows, scratch_rows), 32, 512,
                          dtype=torch.int32,
                          device="xpu")
    score_b = torch.empty_like(score_a)
    index_b = torch.empty_like(index_a)
    return (q, cache, table, reqs, positions, seq_lengths, page_size,
            length, out, score_a, index_a, score_b, index_b)


def selection_golden(case):
    q, cache, table, reqs, positions, seq_lengths, page_size, _, *_ = case
    q = q.cpu().float()
    cache = cache.cpu().float()
    table = table.cpu()
    reqs = reqs.cpu()
    positions = positions.cpu()
    seq_lengths = seq_lengths.cpu()
    out = torch.full((q.size(0), 2051), -1, dtype=torch.int32)
    for row in range(q.size(0)):
        request = int(reqs[row])
        if not 0 <= request < seq_lengths.numel():
            continue
        visible = max(0, int(positions[row]) + 1)
        sequence_length = max(0, int(seq_lengths[request]))
        blocks = min(visible // 4, sequence_length // 4)
        pages = table[request, torch.arange(blocks) // page_size]
        rows = torch.arange(blocks) % page_size
        keys = cache[pages.long(), rows.long(), 0]
        dots = q[row] @ keys.T
        scores = dots.relu().sum(0) * (128 ** -0.5)
        selected = torch.argsort(scores, descending=True,
                                  stable=True)[:min(blocks, 512)]
        tokens = (selected[:, None] * 4 + torch.arange(4)).flatten()
        out[row, :tokens.numel()] = tokens.int()
        tail = visible % 4
        if tail:
            tail_tokens = torch.arange(visible - tail, visible,
                                       dtype=torch.int32)
            for offset, token in enumerate(tail_tokens):
                if int(token) < sequence_length:
                    out[row, tokens.numel() + offset] = token
    return out


@pytest.mark.parametrize("rows", [1, 2, 33])
@pytest.mark.parametrize("tile", [64, 128])
def test_m1_attention_merge_tile_preserves_bits_and_batch_tail(native,
                                                              monkeypatch,
                                                              rows, tile):
    case = attention_case(rows, 6, 1024, 512)
    partials = torch.empty(rows, 6, 43, 258, dtype=torch.float32, device="xpu")
    baseline, actual = torch.empty_like(case[0]), torch.empty_like(case[0])
    monkeypatch.setenv("VLLM_XPU_QWEN38_QSA_M1_MERGE_TILE", "256")
    native.token_split_attention_v3(*case, 512, baseline, partials)
    monkeypatch.setenv("VLLM_XPU_QWEN38_QSA_M1_MERGE_TILE", str(tile))
    native.token_split_attention_v3(*case, 512, actual, partials)
    torch.testing.assert_close(actual, baseline, atol=0, rtol=0)




def assert_random_selection_membership_and_tail(actual, expected):
    """随机近 tie 不强求不同 FP32 归约树的 block 内部次序。"""
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual[:, 2048:], expected[:, 2048:],
                               atol=0, rtol=0)
    actual_tokens = actual[:, :2048].reshape(-1, 512, 4)
    expected_tokens = expected[:, :2048].reshape(-1, 512, 4)
    offsets = torch.arange(4, dtype=torch.int32).reshape(1, 1, 4)
    torch.testing.assert_close(actual_tokens,
                               actual_tokens[:, :, :1] + offsets,
                               atol=0, rtol=0)
    torch.testing.assert_close(
        actual_tokens[:, :, 0].sort(dim=1).values,
        expected_tokens[:, :, 0].sort(dim=1).values,
        atol=0, rtol=0)


@pytest.mark.parametrize("rows,length,page_size", [
    (1, 1024, 128), (2, 1024, 128), (8, 1024, 128),
    (1, 131072, 128), (4, 131072, 128), (8, 131072, 128),
    (1, 131071, 64), (1, 256000, 128),
])
def test_selection_golden_and_final_esimd(native, rows, length, page_size):
    reference = _final_reference()

    case = selection_case(rows, length, page_size)
    native.select_paged_tokens_v2(*case)
    expected = selection_golden(case)
    torch.testing.assert_close(case[8].cpu(), expected, atol=0, rtol=0)
    if reference is None:
        return
    ref_out = torch.empty_like(case[8])
    reference_op = (reference.qsa_select_paged_tokens_local_v1
                    if length < 4096 else
                    reference.qsa_select_paged_tokens_parallel_v1)
    reference_op(*case[:6], 2048, 4, page_size, ref_out)
    torch.testing.assert_close(case[8].cpu(), ref_out.cpu(), atol=0, rtol=0)


@pytest.mark.parametrize("rows,length", [(33, 4096), (4096, 16)])
def test_selection_prefill_cpp_chunk_owner(native, rows, length):
    case = selection_case(rows, length, 128)
    native.select_paged_tokens_v2(*case)
    result = case[8].cpu()
    if rows == 33:
        torch.testing.assert_close(result, selection_golden(case),
                                   atol=0, rtol=0)
    else:
        expected = torch.arange(16, dtype=torch.int32).expand(rows, 16)
        torch.testing.assert_close(result[:, :16].sort().values,
                                   expected, atol=0, rtol=0)
        assert bool((result[:, 16:] == -1).all())


@pytest.mark.parametrize("rows,length,page_size", [
    (128, 4096, 128), (129, 131071, 64), (4096, 4096, 128),
])
def test_selection_wide_legacy_and_sampled_golden(native, rows, length,
                                                  page_size):
    assert native.qsa_sycl_selection_wide_scratch_abi_version == 1
    wide = selection_case(rows, length, page_size, scratch_rows=128)
    old = list(wide)
    old[8] = torch.empty_like(wide[8])
    old[9] = torch.empty(32, 32, 512, device="xpu", dtype=torch.float32)
    old[10] = torch.empty(32, 32, 512, device="xpu", dtype=torch.int32)
    old[11] = torch.empty_like(old[9])
    old[12] = torch.empty_like(old[10])
    native.select_paged_tokens_v2(*wide)
    native.select_paged_tokens_v2(*old)
    assert torch.equal(wide[8].cpu(), old[8].cpu())
    selected_rows = torch.tensor(sorted({0, 31, 32, 63, 64, 127,
                                         rows - 1}), device="xpu")
    sample = list(wide)
    for slot in (0, 3, 4, 8):
        sample[slot] = wide[slot].index_select(0, selected_rows)
    assert_random_selection_membership_and_tail(
        sample[8].cpu(), selection_golden(sample))


def test_provider_selection_wide_and_old_dso_probe(native, monkeypatch):
    assert native.qsa_sycl_selection_wide_scratch_abi_version == 1
    provider = load_provider(monkeypatch, native)
    case = selection_case(128, 4096, 128, scratch_rows=128)
    provider.qsa_select_paged_tokens_v2(*case[:6], 2048, 4, 128,
                                        case[8], max_seq_len=4096)
    assert provider._scratch.selection(case[0], 4096)[0].size(0) == 128
    assert_random_selection_membership_and_tail(
        case[8].cpu(), selection_golden(case))
    old_dso = types.SimpleNamespace(
        select_paged_tokens_v2=native.select_paged_tokens_v2)
    provider_old = load_provider(monkeypatch, old_dso)
    assert provider_old.qsa_sycl_selection_wide_scratch_abi_version == 0
    case_old = selection_case(128, 4096, 128)
    provider_old.qsa_select_paged_tokens_v2(*case_old[:6], 2048, 4, 128,
                                            case_old[8], max_seq_len=4096)
    assert provider_old._scratch.selection(case_old[0], 4096)[0].size(0) == 32


@pytest.mark.parametrize("rows,length,page_size", [
    (128, 4096, 128), (129, 131071, 64), (4096, 4096, 128),
])
def test_selection_wide_exact_binary_rank_tie_and_tail(native, rows, length,
                                                       page_size):
    # 每第七个 block 分数精确为 1，其余精确为 0；高低两组内部均为真
    # tie，应按较小 block id 排序。此参考不调用 Torch matmul/topk。
    case = list(selection_case(rows, length, page_size, scratch_rows=128))
    query, cache = case[:2]
    query.zero_()
    query[:, 0, 0] = 1
    cache.zero_()
    physical_blocks = cache.size(0) * page_size
    ids = torch.arange(physical_blocks, device="xpu")
    cache[:, :, 0, 0] = ((ids % 7) == 0).reshape(
        cache.size(0), page_size).half()
    old = list(case)
    old[8] = torch.empty_like(case[8])
    old[9] = torch.empty(32, 32, 512, device="xpu", dtype=torch.float32)
    old[10] = torch.empty(32, 32, 512, device="xpu", dtype=torch.int32)
    old[11] = torch.empty_like(old[9])
    old[12] = torch.empty_like(old[10])
    native.select_paged_tokens_v2(*case)
    native.select_paged_tokens_v2(*old)
    blocks = length // 4
    higher = [index for index in range(blocks) if index % 7 == 0]
    lower = [index for index in range(blocks) if index % 7 != 0]
    selected = (higher + lower)[:512]
    expected = torch.full((rows, 2051), -1, dtype=torch.int32)
    tokens = (torch.tensor(selected)[:, None] * 4 +
              torch.arange(4)).flatten().int()
    expected[:, :tokens.numel()] = tokens
    for offset in range(length % 4):
        expected[:, tokens.numel() + offset] = length - length % 4 + offset
    torch.testing.assert_close(case[8].cpu(), expected, atol=0, rtol=0)
    torch.testing.assert_close(old[8].cpu(), expected, atol=0, rtol=0)


def test_provider_fused_selection_wide(native, monkeypatch):
    provider = load_provider(monkeypatch, native)
    case = selection_case(128, 4096, 128, scratch_rows=128)
    projected = case[0]
    weight = torch.zeros(128, device="xpu", dtype=torch.float16)
    positions = torch.zeros(128, device="xpu", dtype=torch.int32)
    rope = torch.zeros(1, 64, device="xpu", dtype=torch.float16)
    rope[:, :32] = 1
    output = torch.empty_like(case[8])
    provider.qsa_q_norm_rope_select_v1(
        projected, weight, positions, rope, *case[1:6],
        projected.new_empty(0), output, False, True,
        max_seq_len=4096, compressed_page_size=128)
    normalized = provider._scratch.normalized_q(projected)
    direct = list(case)
    direct[0] = normalized
    direct[8] = torch.empty_like(output)
    native.select_paged_tokens_v2(*direct)
    assert torch.equal(output.cpu(), direct[8].cpu())
    assert provider._scratch.selection(projected, 4096)[0].size(0) == 128


def test_selection_wide_rejected_before_any_submit(native):
    wide = list(selection_case(128, 4096, 128, scratch_rows=128))
    wrong_shape = list(wide)
    wrong_shape[9] = torch.empty(64, 32, 512, dtype=torch.float32,
                                 device="xpu")
    forbidden_short = selection_case(128, 1024, 128, scratch_rows=128)
    forbidden_small = selection_case(127, 4096, 128, scratch_rows=128)
    weight = torch.zeros(128, device="xpu", dtype=torch.float16)
    positions = torch.zeros(128, device="xpu", dtype=torch.int32)
    rope = torch.zeros(1, 64, device="xpu", dtype=torch.float16)
    q_output = torch.empty_like(wide[0])
    fused = (wide[0], weight, positions, rope, *wide[1:6],
             q_output, wide[8], *wrong_shape[9:13], 128, 4096,
             False, False, True)
    torch.xpu.synchronize()
    with torch.profiler.profile(activities=[
            torch.profiler.ProfilerActivity.XPU]) as profiler:
        for case in (wrong_shape, forbidden_short, forbidden_small):
            with pytest.raises(RuntimeError, match="scratch rows"):
                native.select_paged_tokens_v2(*case)
        with pytest.raises(RuntimeError, match="scratch rows"):
            native.qsa_sycl_q_norm_rope_select_v1(*fused)
    assert not [event for event in profiler.events()
                if event.device_type == torch.autograd.DeviceType.XPU]


@pytest.mark.parametrize("q_heads,rows,mrope", [
    (3, 1, False), (3, 8, True), (6, 1, False), (6, 65, True),
])
def test_aux_qkv_postprocess_golden(native, q_heads, rows, mrope):
    torch.manual_seed(932 + q_heads + rows)
    width = (2 * q_heads + 2) * 256
    packed = torch.randn(rows, width, dtype=torch.float16, device="xpu")
    q = torch.empty(rows, q_heads * 256, dtype=torch.float16, device="xpu")
    gate = torch.empty_like(q)
    k = torch.empty(rows, 256, dtype=torch.float16, device="xpu")
    v = torch.empty_like(k)
    weight = torch.zeros(256, dtype=torch.float16, device="xpu")
    cache = torch.zeros(2, 64, dtype=torch.float16, device="xpu")
    cache[:, :32] = 1
    pos = torch.zeros((3, rows) if mrope else (rows,),
                      dtype=torch.int32, device="xpu")
    native.qsa_sycl_qkv_postprocess_v1(
        packed, q, gate, k, v, weight, weight, pos, cache,
        q_heads, 1, True, mrope, True)
    packed_cpu = packed.cpu().float().reshape(rows, 2 * q_heads + 2, 256)
    q_expected = []
    gate_expected = []
    for head in range(q_heads):
        segment = packed_cpu[:, 2 * head]
        q_expected.append((segment / torch.sqrt(
            segment.square().mean(-1, keepdim=True) + 1e-6)).half())
        gate_expected.append(packed_cpu[:, 2 * head + 1].sigmoid().half())
    k_segment = packed_cpu[:, 2 * q_heads]
    k_expected = (k_segment / torch.sqrt(
        k_segment.square().mean(-1, keepdim=True) + 1e-6)).half()
    torch.testing.assert_close(q.cpu().reshape(rows, q_heads, 256),
                               torch.stack(q_expected, 1), atol=0.004,
                               rtol=0.004)
    torch.testing.assert_close(gate.cpu().reshape(rows, q_heads, 256),
                               torch.stack(gate_expected, 1), atol=0.002,
                               rtol=0.002)
    torch.testing.assert_close(k.cpu(), k_expected, atol=0.004, rtol=0.004)
    torch.testing.assert_close(v.cpu(), packed_cpu[:, -1].half(),
                               atol=0, rtol=0)


def test_aux_qkv_rejects_missing_position_proof_before_submit(native):
    rows = 1
    packed = torch.zeros(rows, 8 * 256, dtype=torch.float16, device="xpu")
    q = torch.full((rows, 3 * 256), 7, dtype=torch.float16, device="xpu")
    gate = torch.full_like(q, 7)
    k = torch.full((rows, 256), 7, dtype=torch.float16, device="xpu")
    v = torch.full_like(k, 7)
    weight = torch.zeros(256, dtype=torch.float16, device="xpu")
    pos = torch.zeros(rows, dtype=torch.int32, device="xpu")
    cache = torch.zeros(1, 64, dtype=torch.float16, device="xpu")
    with pytest.raises(RuntimeError, match="position-bound proof"):
        native.qsa_sycl_qkv_postprocess_v1(
            packed, q, gate, k, v, weight, weight, pos, cache,
            3, 1, True, False, False)
    for tensor in (q, gate, k, v):
        assert bool((tensor == 7).all())


@pytest.mark.parametrize("rows,mrope", [(1, False), (8, True), (65, True),
                                      (4096, False), (4096, True)])
def test_aux_indexer_norm_rope_final_esimd(native, rows, mrope):
    reference = _final_reference()

    torch.manual_seed(377 + rows)
    x = torch.randn(rows, 4, 128, dtype=torch.float16, device="xpu")
    weight = torch.randn(128, dtype=torch.float16, device="xpu") * 0.1
    angles = torch.arange(32).float()[None, :] * 0.03
    angles = angles.expand(16, -1)
    cache = torch.cat((angles.cos(), angles.sin()), dim=1).half().to("xpu")
    positions = (torch.arange(rows, dtype=torch.int32) % 16).to("xpu")
    if mrope:
        positions = positions.expand(3, rows).contiguous()
    output = torch.empty_like(x)
    native.qsa_sycl_indexer_norm_rope_v2(
        x, output, weight, positions, cache, mrope, True, True)
    if rows == 1:
        values = x.cpu().float()
        expected = ((values * torch.rsqrt(
            values.square().mean(-1, keepdim=True) + 1e-6)) *
            (1.0 + weight.cpu().float()))
        cosine = cache[0, :32].cpu().float()
        sine = cache[0, 32:].cpu().float()
        first, second = expected[..., :32].clone(), expected[..., 32:64].clone()
        expected[..., :32] = first * cosine - second * sine
        expected[..., 32:64] = first * sine + second * cosine
        expected = expected.half()
        torch.testing.assert_close(output.cpu(), expected,
                                   atol=0.003, rtol=0.003)
    else:
        values = x.cpu().float()
        normalized = ((values * torch.rsqrt(
            values.square().mean(-1, keepdim=True) + 1e-6)) *
            (1.0 + weight.cpu().float())).half()
        # All positions/axes in this fixture share identical sin/cos values.
        cosine = cache[0, :32].cpu().float()
        sine = cache[0, 32:].cpu().float()
        first = normalized[..., :32].float()
        second = normalized[..., 32:64].float()
        expected = torch.cat((
            (first * cosine).half().float() - (second * sine).half().float(),
            (second * cosine).half().float() + (first * sine).half().float(),
            normalized[..., 64:].float()), dim=-1).half()
        torch.testing.assert_close(output.cpu(), expected,
                                   atol=0.002, rtol=0.002)
    if 1 < rows <= 8 and reference is not None:
        expected = torch.empty_like(x)
        reference.qsa_indexer_norm_rope_v2(
            x, expected, weight, positions, cache, mrope, True, False)
        torch.testing.assert_close(output.cpu(), expected.cpu(),
                                   atol=0.002, rtol=0.002)


def test_aux_indexer_fp16_rounding_boundaries_fresh_golden(native):
    # Isolate the two FP16 boundaries: norm output and each RoPE product.
    # The first nine outputs are exact subnormal integer units, not tolerance.
    values = torch.tensor(
        [0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, -3.5, 1023.5],
        dtype=torch.float32)
    x = torch.ones(2, 4, 128, dtype=torch.float16, device="xpu")
    weight_cpu = torch.full((128,), -1, dtype=torch.float16)
    weight_cpu[:9] = (values - 1).half()
    weight = weight_cpu.to("xpu")
    cache_cpu = torch.zeros(1, 64, dtype=torch.float16)
    cache_cpu[0, :9] = 2.0 ** -24
    cache_cpu[0, 9] = 1 + 2.0 ** -10
    cache_cpu[0, 10] = 1 + 3 * 2.0 ** -10
    cache = cache_cpu.to("xpu")
    positions = torch.zeros(2, dtype=torch.int32, device="xpu")
    output = torch.empty_like(x)
    native.qsa_sycl_indexer_norm_rope_v2(
        x, output, weight, positions, cache, False, True, True)
    actual = output.cpu()
    units = torch.tensor([0, 2, 2, 4, 0, -2, -2, -4, 1024],
                         dtype=torch.float32)
    torch.testing.assert_close(actual[0, 0, :9].float() / (2.0 ** -24),
                               units, atol=0, rtol=0)
    # Independent materialized CPU FP16 stages for the whole 128-wide row.
    normed = ((x.cpu().float() * torch.rsqrt(
        x.cpu().float().square().mean(-1, keepdim=True) + 1e-6)) *
        (1.0 + weight_cpu.float())).half()
    cosine, sine = cache_cpu[0, :32], cache_cpu[0, 32:]
    first, second = normed[..., :32], normed[..., 32:64]
    first_cos = (first.float() * cosine.float()).half()
    second_sin = (second.float() * sine.float()).half()
    second_cos = (second.float() * cosine.float()).half()
    first_sin = (first.float() * sine.float()).half()
    expected = torch.cat((
        (first_cos.float() - second_sin.float()).half(),
        (second_cos.float() + first_sin.float()).half(),
        normed[..., 64:]), dim=-1)
    assert torch.equal(actual, expected)

    weight.fill_(-1)
    weight[:3] = 0.5
    cache.zero_()
    cache[0, :3] = torch.tensor([43648, 43680, -43680],
                                dtype=torch.float16, device="xpu")
    native.qsa_sycl_indexer_norm_rope_v2(
        x, output, weight, positions, cache, False, True, True)
    overflow = output.cpu()[..., :3]
    assert bool(torch.isfinite(overflow[..., 0]).all())
    assert bool(torch.isposinf(overflow[..., 1]).all())
    assert bool(torch.isneginf(overflow[..., 2]).all())


@pytest.mark.parametrize("rows", range(1, 9))
def test_aux_tp4_q4_indexer_projection_golden(native, rows):
    x = torch.zeros(rows, 2560, device="xpu", dtype=torch.float16)
    for row in range(rows):
        x[row, 1::2] = row + 1
    packed = torch.full((640, 1280), 0x98, device="xpu",
                        dtype=torch.uint8)
    scales = torch.full((640, 20), 0.1, device="xpu",
                        dtype=torch.float16)
    output = torch.empty(rows, 640, device="xpu", dtype=torch.float16)
    native.qsa_sycl_indexer_projection_int4_v1(x, packed, scales, output)
    expected = (1280 * float(scales[0, 0].item()) *
                torch.arange(1, rows + 1)).unsqueeze(1).expand(rows, 640)
    torch.testing.assert_close(output.cpu().float(), expected,
                               atol=0.5, rtol=0.001)


def test_aux_norm_rope_select_cpp_chunk_and_preflight(native):
    case = selection_case(33, 4096, 128)
    projected = case[0]
    q_output = torch.empty_like(projected)
    weight = torch.zeros(128, device="xpu", dtype=torch.float16)
    positions = torch.zeros(33, device="xpu", dtype=torch.int32)
    cache = torch.zeros(1, 64, device="xpu", dtype=torch.float16)
    cache[:, :32] = 1
    out = torch.full_like(case[8], -77)
    args = (projected, weight, positions, cache, *case[1:6],
            q_output, out, *case[9:13], 128, 4096, False, False)
    torch.xpu.synchronize()
    with torch.profiler.profile(activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.XPU,
    ]) as profiler, pytest.raises(RuntimeError, match="position-bound proof"):
        native.qsa_sycl_q_norm_rope_select_v1(*args, False)
    assert not any(event.device_type == torch.autograd.DeviceType.XPU
                   for event in profiler.events())
    assert bool((out == -77).all())
    native.qsa_sycl_q_norm_rope_select_v1(*args, True)
    reference_q = torch.empty_like(projected)
    reference_out = torch.empty_like(out)
    native.qsa_sycl_indexer_norm_rope_v2(
        projected, reference_q, weight, positions, cache,
        False, False, True)
    native.select_paged_tokens_v2(
        reference_q, *case[1:8], reference_out, *case[9:13])
    torch.testing.assert_close(q_output.cpu(), reference_q.cpu(),
                               atol=0, rtol=0)
    torch.testing.assert_close(out.cpu(), reference_out.cpu(),
                               atol=0, rtol=0)


def test_provider_actual_legacy_signatures_and_host_bound(native, monkeypatch):
    provider = load_provider(monkeypatch, native)
    assert provider.qsa_sycl_abi_version == 1
    assert not hasattr(provider, "kv_tile")
    assert provider.qsa_sycl_row_store_v3 == int(callable(
        getattr(native, "qsa_sycl_store_cache_rows_v3", None)))
    assert provider.qsa_store_m1_transaction_abi_version == int(callable(
        getattr(native, "qsa_sycl_try_store_m1_transaction_v1", None)))

    case = selection_case(33, 4096, 128)
    with pytest.raises(TypeError, match="max_seq_len"):
        provider.qsa_select_paged_tokens_v2(*case[:6], 2048, 4, 128,
                                            case[8])
    provider.qsa_select_paged_tokens_v2(*case[:6], 2048, 4, 128,
                                        case[8], max_seq_len=4096)
    torch.testing.assert_close(case[8].cpu(), selection_golden(case),
                               atol=0, rtol=0)

    q, packed, logical, block_table, requests = attention_case(
        129, 6, 1024, 256)
    k_cache = packed[:, 0, :, :256].unsqueeze(2)
    v_cache = packed[:, 0, :, 256:].unsqueeze(2)
    out = torch.empty_like(q)
    provider.sparse_paged_attention_q6_v1(
        q, k_cache, v_cache, logical, block_table, requests, 256, out)
    torch.testing.assert_close(
        out.cpu(), attention_golden((q, packed, logical, block_table,
                                     requests)), atol=0.004, rtol=0.004)
    out_candidate = torch.empty_like(q)
    partials = torch.empty(128, 6, 43, 258, device="xpu",
                           dtype=torch.float32)
    provider.sparse_attention_token_split_candidate_q6_v1(
        q, packed, logical, block_table, requests, 256,
        out_candidate, partials)
    torch.testing.assert_close(out_candidate.cpu(), out.cpu(),
                               atol=0, rtol=0)


def test_provider_legacy_indexer_and_fused_selection(native, monkeypatch):
    provider = load_provider(monkeypatch, native)
    compression = compression_case(1, torch.float16)
    pooled = torch.empty_like(compression[0])
    first = torch.empty((1, 3), dtype=torch.int64, device="xpu")
    returned = provider.qsa_group_compress_v2(
        *compression, pooled, first, 4, 100, True)
    assert returned.data_ptr() == pooled.data_ptr()
    expected_pooled, expected_first = compression_golden(compression, 100)
    assert torch.equal(pooled.cpu(), expected_pooled)
    assert torch.equal(first.cpu(), expected_first)

    case = selection_case(1, 1024, 128)
    projected = case[0]
    weight = torch.zeros(128, device="xpu", dtype=torch.float16)
    positions = torch.zeros(1, device="xpu", dtype=torch.int32)
    rope = torch.zeros(1, 64, device="xpu", dtype=torch.float16)
    rope[:, :32] = 1
    expected_v1 = torch.empty_like(projected)
    actual_v1 = torch.empty_like(projected)
    native.qsa_sycl_indexer_norm_rope_v2(
        projected, expected_v1, weight, positions, rope,
        False, False, True)
    provider.qsa_indexer_norm_rope_v1(
        projected, actual_v1, weight, positions, rope, False, True)
    assert torch.equal(actual_v1.cpu(), expected_v1.cpu())

    expected_v2 = torch.empty_like(projected)
    actual_v2 = torch.empty_like(projected)
    native.qsa_sycl_indexer_norm_rope_v2(
        projected, expected_v2, weight, positions, rope,
        False, True, False)
    provider.qsa_indexer_norm_rope_v2(
        projected, actual_v2, weight, positions, rope,
        False, True, False)
    assert torch.equal(actual_v2.cpu(), expected_v2.cpu())
    actual_v2.fill_(17)
    with pytest.raises(ValueError, match="NeoX"):
        provider.qsa_indexer_norm_rope_v2(
            projected, actual_v2, weight, positions, rope,
            False, False, False)
    assert bool((actual_v2 == 17).all())

    output = torch.empty_like(case[8])
    with pytest.raises(TypeError, match="max_seq_len"):
        provider.qsa_q_norm_rope_select_v1(
            projected, weight, positions, rope, *case[1:6],
            projected.new_empty(0), output, False, True)
    provider.qsa_q_norm_rope_select_v1(
        projected, weight, positions, rope, *case[1:6],
        projected.new_empty(0), output, False, True,
        max_seq_len=1024, compressed_page_size=128)
    expected_output = torch.empty_like(output)
    native.select_paged_tokens_v2(
        expected_v1, *case[1:8], expected_output, *case[9:13])
    assert torch.equal(output.cpu(), expected_output.cpu())


def test_provider_m1_fused_probe_and_old_dso(monkeypatch):
    calls = []
    result = {"fused": True}

    def legacy(stores):
        calls.append(("legacy", stores))
        return True

    def fused(stores):
        calls.append(("fused", stores))
        return result["fused"]

    extension = types.SimpleNamespace(
        qsa_sycl_try_store_m1_transaction_v1=legacy,
        qsa_sycl_try_store_m1_transaction_fused_v1=fused)
    provider = load_provider(monkeypatch, extension)
    stores = ((object(), object(), object()), )
    assert provider.qsa_store_m1_transaction_abi_version == 1
    assert provider.try_store_m1_transaction_v1(stores, True)
    assert calls.pop() == ("fused", stores)
    assert provider.try_store_m1_transaction_v1(stores, False)
    assert calls.pop() == ("legacy", stores)
    # Unsupported fused calls must go to vLLM's full fallback, not replay an
    # already-submitted transaction through the legacy C++ entry point.
    result["fused"] = False
    assert provider.try_store_m1_transaction_v1(stores, True) is False
    assert calls.pop() == ("fused", stores)
    del extension.qsa_sycl_try_store_m1_transaction_fused_v1
    assert provider.try_store_m1_transaction_v1(stores, True) is False
    assert calls.pop() == ("fused", stores)  # import-time capability cache

    old_provider = load_provider(monkeypatch, types.SimpleNamespace(
        qsa_sycl_try_store_m1_transaction_v1=legacy))
    assert old_provider.qsa_store_m1_transaction_abi_version == 1
    assert old_provider.try_store_m1_transaction_v1(stores, True)
    assert calls.pop() == ("legacy", stores)
    with pytest.raises(TypeError, match="enable_fused"):
        old_provider.try_store_m1_transaction_v1(stores, 1)


def test_provider_owner_rowstore_transaction(owner_native, monkeypatch):
    provider = load_provider(monkeypatch, owner_native)
    assert provider.qsa_sycl_row_store_v3 == 1
    assert provider.qsa_sycl_row_store_parallel_v1 == 1
    assert provider.qsa_store_m1_transaction_abi_version == 1
    assert callable(getattr(
        owner_native, "qsa_sycl_try_store_m1_transaction_fused_v1", None))
    assert not hasattr(provider, "qsa_store_cache_rows_v4")

    cache = torch.zeros(2, 4, 1, 128, dtype=torch.float16, device="xpu")
    slots = torch.tensor([1, 1, 4, -1], dtype=torch.int64, device="xpu")
    rows = torch.arange(1, 5, dtype=torch.float16,
                        device="xpu")[:, None].expand(4, 128).contiguous()
    provider.qsa_store_cache_rows_v3(cache, slots, rows)
    assert bool((cache[0, 1, 0] == 2).all())
    assert bool((cache[1, 0, 0] == 3).all())
    assert bool((cache[0, 0, 0] == 0).all())

    long_cache = torch.zeros(2, 4, 1, 3, dtype=torch.int64, device="xpu")
    unique = torch.tensor([0, 5], dtype=torch.int64, device="xpu")
    long_rows = torch.tensor([[1, 2, 3], [4, 5, 6]],
                             dtype=torch.int64, device="xpu")
    provider.qsa_store_cache_rows_r_aware_v1(
        long_cache, unique, long_rows, True)
    assert torch.equal(long_cache[0, 0, 0].cpu(), long_rows[0].cpu())
    assert torch.equal(long_cache[1, 1, 0].cpu(), long_rows[1].cpu())

    for fused in (True, False):
        target = torch.zeros_like(cache)
        target_long = torch.zeros_like(long_cache)
        stores = ((target, slots[:1], rows[:1]),
                  (target_long, unique[:1], long_rows[:1]))
        assert provider.try_store_m1_transaction_v1(stores, fused)
        assert bool((target[0, 1, 0] == 1).all())
        assert torch.equal(target_long[0, 0, 0].cpu(), long_rows[0].cpu())

    untouched = torch.zeros_like(cache)
    unsupported = ((untouched, slots[:1], rows[:1]),
                   (long_cache, unique, long_rows))
    torch.xpu.synchronize()
    with torch.profiler.profile(activities=[
            torch.profiler.ProfilerActivity.XPU]) as profiler:
        assert provider.try_store_m1_transaction_v1(unsupported, True) is False
    assert not any(event.device_type == torch.autograd.DeviceType.XPU
                   for event in profiler.events())
    assert bool((untouched == 0).all())
    invalid_shape = torch.zeros(1, 127, dtype=torch.float16, device="xpu")
    invalid = ((untouched, slots[:1], rows[:1]),
               (cache, slots[:1], invalid_shape))
    torch.xpu.synchronize()
    with torch.profiler.profile(activities=[
            torch.profiler.ProfilerActivity.XPU]) as profiler:
        assert provider.try_store_m1_transaction_v1(invalid, True) is False
    assert not any(event.device_type == torch.autograd.DeviceType.XPU
                   for event in profiler.events())
    assert bool((untouched == 0).all())

    lazy_backing = torch.zeros_like(cache)
    lazy_cache = lazy_backing._neg_view()
    assert lazy_cache.is_neg()
    lazy_last = ((untouched, slots[:1], rows[:1]),
                 (lazy_cache, slots[:1], rows[:1]))
    for fused in (True, False):
        torch.xpu.synchronize()
        with torch.profiler.profile(activities=[
                torch.profiler.ProfilerActivity.XPU]) as profiler:
            assert provider.try_store_m1_transaction_v1(
                lazy_last, fused) is False
        assert not any(event.device_type == torch.autograd.DeviceType.XPU
                       for event in profiler.events())
        assert bool((untouched == 0).all())
        assert bool((lazy_backing == 0).all())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("count", [2, 3])
@pytest.mark.parametrize("slot", [-1, 3, 8])
@pytest.mark.parametrize("page_size", [7, 8])
def test_owner_m1_fused_raw_bits_position_stride_and_invalid_slots(
        owner_native, dtype, count, slot, page_size):
    source_bits = torch.tensor([0x7e01, 0x8000, 0x7c00, 0xfc00],
                               dtype=torch.uint16).repeat(32)
    source = source_bits.view(dtype).reshape(1, 128).to("xpu")
    raw = torch.zeros((1, page_size, 1, 130), dtype=dtype,
                      device="xpu")[:, :, :, :128]
    compressed = torch.zeros((1, page_size, 1, 130), dtype=dtype,
                             device="xpu")[:, :, :, :128]
    position = torch.zeros((1, page_size, 1, 5), dtype=torch.int64,
                           device="xpu")[:, :, :, :3]
    position_rows = torch.tensor([[11, -4, 22, -4, 33, -4]],
                                 dtype=torch.int64, device="xpu")[:, ::2]
    assert position_rows.stride(-1) == 2
    slots = torch.tensor([slot], dtype=torch.int64, device="xpu")
    stores = ((raw, slots, source),
              (position, slots, position_rows))
    if count == 3:
        stores = ((compressed, slots, source), ) + stores
    assert owner_native.qsa_sycl_try_store_m1_transaction_fused_v1(stores)
    torch.xpu.synchronize()
    if 0 <= slot < page_size:
        assert torch.equal(raw[0, slot, 0].cpu().view(torch.uint16),
                           source[0].cpu().view(torch.uint16))
        assert torch.equal(position[0, slot, 0].cpu(),
                           position_rows[0].cpu())
        if count == 3:
            assert torch.equal(
                compressed[0, slot, 0].cpu().view(torch.uint16),
                source[0].cpu().view(torch.uint16))
    else:
        assert torch.count_nonzero(raw) == 0
        assert torch.count_nonzero(position) == 0
        assert torch.count_nonzero(compressed) == 0


@pytest.mark.parametrize("count", [2, 3])
def test_owner_m1_fused_cross_store_order_and_single_submit(
        owner_native, count):
    first = torch.zeros((1, 8, 1, 128), dtype=torch.float16, device="xpu")
    second = torch.zeros_like(first)
    source = torch.arange(128, dtype=torch.float16,
                          device="xpu").reshape(1, 128)
    slot = torch.tensor([3], dtype=torch.int64, device="xpu")
    stores = ((first, slot, source), (second, slot, first[0, 3, 0][None, :]))
    if count == 3:
        compressed = torch.zeros_like(first)
        stores = ((compressed, slot, source), ) + stores
    torch.xpu.synchronize()
    with torch.profiler.profile(activities=[
            torch.profiler.ProfilerActivity.XPU]) as profiler:
        assert owner_native.qsa_sycl_try_store_m1_transaction_fused_v1(stores)
        torch.xpu.synchronize()
    launches = [event.name for event in profiler.events()
                if event.device_type == torch.autograd.DeviceType.XPU]
    assert sum("M1StoreTransactionFusedKernel" in name
               for name in launches) == 1, launches
    assert not any("RowStoreKernel" in name for name in launches), launches
    assert torch.equal(second[0, 3, 0].cpu(), source[0].cpu())


def test_owner_m1_fused_cross_store_slot_dependency(owner_native):
    positions = torch.full((1, 8, 1, 3), -1, dtype=torch.int64, device="xpu")
    keys = torch.zeros((1, 8, 1, 128), dtype=torch.float16, device="xpu")
    first_slot = torch.tensor([2], dtype=torch.int64, device="xpu")
    position_row = torch.tensor([[5, 17, 29]], dtype=torch.int64, device="xpu")
    next_slot = positions[0, 2, 0, 0].view(1)
    assert next_slot.is_contiguous()
    source = torch.arange(128, dtype=torch.float16,
                          device="xpu").reshape(1, 128)
    stores = ((positions, first_slot, position_row),
              (keys, next_slot, source))
    assert owner_native.qsa_sycl_try_store_m1_transaction_fused_v1(stores)
    torch.xpu.synchronize()
    assert torch.equal(positions[0, 2, 0].cpu(), position_row[0].cpu())
    assert torch.equal(keys[0, 5, 0].cpu(), source[0].cpu())


def test_owner_m1_fused_invalid_first_slot_keeps_later_store(owner_native):
    first = torch.zeros((1, 8, 1, 128), dtype=torch.float16, device="xpu")
    second = torch.zeros_like(first)
    skipped = torch.tensor([-1], dtype=torch.int64, device="xpu")
    valid = torch.tensor([3], dtype=torch.int64, device="xpu")
    source = torch.ones((1, 128), dtype=torch.float16, device="xpu")
    stores = ((first, skipped, source), (second, valid, source))
    assert owner_native.qsa_sycl_try_store_m1_transaction_fused_v1(stores)
    torch.xpu.synchronize()
    assert torch.count_nonzero(first) == 0
    assert torch.equal(second[0, 3, 0].cpu(), source[0].cpu())


@pytest.mark.parametrize("count", [2, 3])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_owner_m1_fused_nondefault_stream_all_store_lifetimes(
        owner_native, count, dtype):
    producer = torch.xpu.current_stream()

    def submit_and_release(stream, marker):
        # All caches, slots and rows are allocated on the producer stream.
        # None of them escape this helper; only separate, stream-ordered
        # snapshots remain after its return.
        raw = torch.zeros((1, 8, 1, 128), dtype=dtype, device="xpu")
        raw_slot = torch.tensor([1], dtype=torch.int64, device="xpu")
        raw_row = torch.full((1, 128), marker, dtype=dtype, device="xpu")
        position = torch.zeros((1, 8, 1, 3), dtype=torch.int64, device="xpu")
        position_slot = torch.tensor([2], dtype=torch.int64, device="xpu")
        position_row = torch.tensor([[marker + 11, marker + 22,
                                      marker + 33]], dtype=torch.int64,
                                    device="xpu")
        stores = ((raw, raw_slot, raw_row),
                  (position, position_slot, position_row))
        if count == 3:
            compressed = torch.zeros_like(raw)
            compressed_slot = torch.tensor([3], dtype=torch.int64,
                                           device="xpu")
            compressed_row = torch.full((1, 128), marker + 7,
                                        dtype=dtype, device="xpu")
            stores = ((compressed, compressed_slot, compressed_row), ) + stores
        stream.wait_stream(producer)
        with torch.xpu.stream(stream):
            assert owner_native.qsa_sycl_try_store_m1_transaction_fused_v1(
                stores)
            snapshots = tuple(cache.clone() for cache, _, _ in stores)
        return snapshots

    pending = []
    for marker in (1, 2):
        stream = torch.xpu.Stream()
        pending.append((stream, marker, submit_and_release(stream, marker)))
    # Encourage reuse of every released cache, slot and row size on the
    # original stream while both non-default streams may still be in flight.
    junk = []
    for _ in range(8):
        junk.extend((
            torch.full((1, 8, 1, 128), -9, dtype=dtype, device="xpu"),
            torch.full((1, 8, 1, 3), -9, dtype=torch.int64, device="xpu"),
            torch.full((1, 128), -9, dtype=dtype, device="xpu"),
            torch.full((1, 3), -9, dtype=torch.int64, device="xpu"),
            torch.full((1,), -9, dtype=torch.int64, device="xpu"),
        ))
    for stream, marker, snapshots in pending:
        stream.synchronize()
        if count == 3:
            compressed, raw, position = snapshots
            assert torch.equal(
                compressed[0, 3, 0].cpu(),
                torch.full((128,), marker + 7, dtype=dtype))
            assert torch.count_nonzero(compressed[0, 0, 0]) == 0
        else:
            raw, position = snapshots
        assert torch.equal(raw[0, 1, 0].cpu(),
                           torch.full((128,), marker, dtype=dtype))
        assert torch.count_nonzero(raw[0, 0, 0]) == 0
        assert torch.equal(position[0, 2, 0].cpu(),
                           torch.tensor([marker + 11, marker + 22,
                                         marker + 33], dtype=torch.int64))
        assert torch.count_nonzero(position[0, 0, 0]) == 0
    del junk


def test_selection_rejects_alias_before_write(native):
    case = list(selection_case(1, 1024, 128))
    shared = torch.zeros((1, 2051), dtype=torch.int32, device="xpu")
    case[2] = shared
    case[8] = shared
    with pytest.raises(RuntimeError, match="aliases"):
        native.select_paged_tokens_v2(*case)
    assert bool((shared == 0).all())


@pytest.mark.parametrize("entry", ["compression", "selection", "attention",
                                   "indexer", "qkv"])
def test_lazy_anchor_rejected_before_any_output_write(native, entry):
    # Direct Pybind bypasses the Torch dispatcher negative-view fallback.
    # Shape/dtype/contiguity alone cannot prove data_ptr() is the logical input.
    if entry == "compression":
        case = list(compression_case(1, torch.float16))
        case[0] = case[0]._neg_view()
        pooled = torch.full_like(case[0], 7)
        first = torch.full((1, 3), 11, dtype=torch.int64, device="xpu")
        outputs = (pooled, first)
        invoke = lambda: native.group_compress_v2(
            *case, pooled, first, 4, 100, True)
    elif entry == "selection":
        case = list(selection_case(1, 1024, 128))
        case[0] = case[0]._neg_view()
        outputs = tuple(case[8:])
        for output in outputs:
            output.fill_(7)
        invoke = lambda: native.select_paged_tokens_v2(*case)
    elif entry == "attention":
        case = list(attention_case(1, 6, 1024, 256))
        case[0] = case[0]._neg_view()
        output = torch.full_like(case[0], 7)
        partials = torch.full((1, 6, 43, 258), 7, dtype=torch.float32,
                              device="xpu")
        outputs = (output, partials)
        invoke = lambda: native.token_split_attention_v3(
            *case, 256, output, partials)
    elif entry == "indexer":
        input = torch.ones(1, 4, 128, dtype=torch.float16,
                           device="xpu")._neg_view()
        output = torch.full_like(input, 7)
        weight = torch.zeros(128, dtype=torch.float16, device="xpu")
        positions = torch.zeros(1, dtype=torch.int32, device="xpu")
        cache = torch.ones(1, 64, dtype=torch.float16, device="xpu")
        outputs = (output,)
        invoke = lambda: native.qsa_sycl_indexer_norm_rope_v2(
            input, output, weight, positions, cache, False, False, True)
    else:
        input = torch.ones(1, 8 * 256, dtype=torch.float16,
                           device="xpu")._neg_view()
        q = torch.full((1, 3 * 256), 7, dtype=torch.float16, device="xpu")
        gate = torch.full_like(q, 7)
        k = torch.full((1, 256), 7, dtype=torch.float16, device="xpu")
        v = torch.full_like(k, 7)
        weight = torch.zeros(256, dtype=torch.float16, device="xpu")
        positions = torch.zeros(1, dtype=torch.int32, device="xpu")
        cache = torch.ones(1, 64, dtype=torch.float16, device="xpu")
        outputs = (q, gate, k, v)
        invoke = lambda: native.qsa_sycl_qkv_postprocess_v1(
            input, q, gate, k, v, weight, weight, positions, cache,
            3, 1, True, False, True)
    snapshots = tuple(output.clone() for output in outputs)
    with pytest.raises(RuntimeError, match="lazy view"):
        invoke()
    torch.xpu.synchronize()
    for output, snapshot in zip(outputs, snapshots):
        assert torch.equal(output.cpu(), snapshot.cpu())


@pytest.mark.parametrize("position,sequence_length", [(-2, 1024),
                                                      (1023, -1)])
def test_selection_invalid_device_metadata_is_empty(native, position,
                                                    sequence_length):
    case = list(selection_case(1, 1024, 128))
    case[4].fill_(position)
    case[5].fill_(sequence_length)
    native.select_paged_tokens_v2(*case)
    assert bool((case[8] == -1).all())


def test_selection_mixed_request_lengths_and_tables(native):
    reference = _final_reference()

    case = list(selection_case(4, 1024, 128))
    case[1] = torch.cat((case[1], torch.randn_like(case[1])), dim=0)
    case[2] = torch.arange(4, dtype=torch.int32,
                           device="xpu").reshape(2, 2)
    case[3] = torch.tensor([0, 1, 0, 1], dtype=torch.int32, device="xpu")
    case[4] = torch.tensor([1023, 511, 1022, 510],
                           dtype=torch.int64, device="xpu")
    case[5] = torch.tensor([1024, 512], dtype=torch.int32, device="xpu")
    native.select_paged_tokens_v2(*case)
    expected = selection_golden(case)
    torch.testing.assert_close(case[8].cpu(), expected, atol=0, rtol=0)
    if reference is None:
        return
    ref_out = torch.empty_like(case[8])
    reference.qsa_select_paged_tokens_local_v1(
        *case[:6], 2048, 4, 128, ref_out)
    torch.testing.assert_close(case[8].cpu(), ref_out.cpu(), atol=0, rtol=0)


def test_selection_long_mixed_request_tail_position(native):
    reference = _final_reference()

    case = list(selection_case(2, 131072, 128))
    physical_pages = case[1].size(0)
    case[1] = torch.cat((case[1], torch.randn_like(case[1])), dim=0)
    case[2] = torch.arange(physical_pages * 2, dtype=torch.int32,
                           device="xpu").reshape(2, physical_pages)
    case[3] = torch.tensor([0, 1], dtype=torch.int32, device="xpu")
    case[4] = torch.tensor([131070, 1022], dtype=torch.int64,
                           device="xpu")
    case[5] = torch.tensor([131072, 1024], dtype=torch.int32,
                           device="xpu")
    native.select_paged_tokens_v2(*case)
    expected = selection_golden(case)
    torch.testing.assert_close(case[8].cpu(), expected, atol=0, rtol=0)
    if reference is None:
        return
    ref_out = torch.empty_like(case[8])
    reference.qsa_select_paged_tokens_parallel_v1(
        *case[:6], 2048, 4, 128, ref_out)
    torch.testing.assert_close(case[8].cpu(), ref_out.cpu(), atol=0, rtol=0)


def test_selection_nondefault_stream(native):
    stream = torch.xpu.Stream()
    with torch.xpu.stream(stream):
        case = selection_case(1, 131072, 128)
        expected = selection_golden(case)
        native.select_paged_tokens_v2(*case)
        output = case[8]
        del case
        gc.collect()
    stream.synchronize()
    torch.testing.assert_close(output.cpu(), expected, atol=0, rtol=0)


def test_two_nondefault_streams_keep_private_workspaces(native):
    reference = _final_reference()

    streams = [torch.xpu.Stream(), torch.xpu.Stream()]
    results = []
    for stream in streams:
        with torch.xpu.stream(stream):
            attention = attention_case(2, 6, 1024, 256)
            selection = selection_case(2, 131071, 128)
            if reference is None:
                # Exact binary scores make CPU/native ordering comparable;
                # random FP32 scores can swap ranks within one ulp.
                query, cache = selection[:2]
                query.zero_()
                query[:, 0, 0] = 1
                cache.zero_()
                ids = torch.arange(cache.size(0) * 128, device="xpu")
                cache[:, :, 0, 0] = ((ids % 7) == 0).reshape(
                    cache.size(0), 128).half()
            expected_attention = attention_golden(attention)
            expected_selection = (selection_golden(selection)
                                  if reference is None else None)
            attended = torch.empty_like(attention[0])
            partials = torch.empty((2, 6, 43, 258), dtype=torch.float32,
                                   device="xpu")
            native.token_split_attention_v3(*attention, 256, attended,
                                            partials)
            native.select_paged_tokens_v2(*selection)
            if reference is None:
                ref_selected = expected_selection
            else:
                ref_selected = torch.empty_like(selection[8])
                reference.qsa_select_paged_tokens_parallel_v1(
                    *selection[:6], 2048, 4, 128, ref_selected)
            selected = selection[8]
            results.append((attended, selected, expected_attention,
                            ref_selected))
            del attention, selection, partials
            gc.collect()
    for stream in streams:
        stream.synchronize()
    for attended, selected, expected_attention, ref_selected in results:
        torch.testing.assert_close(attended.cpu(), expected_attention,
                                   atol=0.004, rtol=0.004)
        # Keep exact order/tie/tail checks, not a relaxed float tolerance.
        torch.testing.assert_close(selected.cpu(), ref_selected.cpu(),
                                   atol=0, rtol=0)


def test_rejected_calls_submit_no_xpu_kernel(native):
    compression = compression_case(2, torch.float16)
    pooled = torch.empty_like(compression[0])
    first = torch.empty((2, 3), dtype=torch.int64, device="xpu")
    attention = attention_case(1, 6, 1024, 256)
    attention_partials = torch.empty((1, 6, 43, 258), dtype=torch.float32,
                                     device="xpu")
    selection = list(selection_case(1, 1024, 128))
    selection[11] = selection[9]
    torch.xpu.synchronize()
    with torch.profiler.profile(activities=[
            torch.profiler.ProfilerActivity.XPU]) as profiler:
        with pytest.raises(RuntimeError):
            native.group_compress_v2(*compression, pooled, first,
                                     4, 100, False)
        with pytest.raises(RuntimeError):
            native.token_split_attention_v3(
                *attention, 256, attention[0], attention_partials)
        with pytest.raises(RuntimeError):
            native.select_paged_tokens_v2(*selection)
    submitted = [event for event in profiler.events()
                 if event.device_type == torch.autograd.DeviceType.XPU]
    assert submitted == []
