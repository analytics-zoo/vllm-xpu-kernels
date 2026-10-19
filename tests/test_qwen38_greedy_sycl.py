# SPDX-License-Identifier: Apache-2.0
"""Independent full-vocabulary argmax and transport/state contracts."""

import importlib
import importlib.util
import os

import pytest
import torch


@pytest.fixture(scope="module")
def ops():
    if not torch.xpu.is_available():
        pytest.skip("XPU required")
    torch.set_num_threads(1)
    path = os.getenv("QWEN38_GREEDY_LIBRARY")
    if not path:
        return importlib.import_module("vllm_xpu_kernels._qwen38_C")
    spec = importlib.util.spec_from_file_location("_qwen38_C", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("n", [1, 32, 511, 31040, 62080, 65535])
@pytest.mark.parametrize(
    "kind", ["random", "ties", "nan", "inf", "minus_inf", "zero"]
)
@pytest.mark.parametrize("pad", [False, True])
@pytest.mark.parametrize("tp_size", [4, 8])
def test_full_vocab_independent(ops, n, kind, pad, tp_size):
    torch.manual_seed(180916 + n)
    values = torch.randn(tp_size, n).half()
    if kind == "ties":
        values.fill_(1)
    elif kind == "nan":
        values[2, 0] = float("nan")
        values[tp_size - 1, 0] = float("nan")
    elif kind == "inf":
        values[2, 0] = float("inf")
        values[tp_size - 1, 0] = float("inf")
    elif kind == "minus_inf":
        values.fill_(-float("inf"))
    elif kind == "zero":
        values.zero_()
        values[0, 0] = -0.0
    padding = min(n - 1, 5) if pad else 0
    real = n - padding
    if padding:
        values[:, -padding:] = float("inf")
    expected = values[:, :real].float().reshape(-1).argmax().view(1)
    device = values.to("xpu")
    packets = [
        ops.try_local_argmax_packet(device[r : r + 1], r * real, padding)
        for r in range(tp_size)
    ]
    for packet in packets:
        assert packet.dtype == torch.float32
        assert packet.shape == (1, 64)
        torch.testing.assert_close(
            packet[:, 2:], torch.zeros_like(packet[:, 2:])
        )
    actual = ops.try_global_argmax_token(torch.cat(packets))
    assert actual.dtype == torch.int64
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    compact = ops.try_global_argmax_token_int32(torch.cat(packets))
    assert compact.dtype == torch.int32
    torch.testing.assert_close(compact.cpu(), expected.int(), rtol=0, atol=0)


@pytest.mark.parametrize("tp_size", [4, 8])
def test_global_preserves_float32_and_large_indices(ops, tp_size):
    packets = torch.zeros(tp_size, 64)
    packets[:, 1] = torch.arange(tp_size) + (2**24 - tp_size)
    packets[:, 0] = 1
    packets[tp_size - 1, 0] = torch.nextafter(
        torch.tensor(1.0), torch.tensor(2.0)
    )
    actual = ops.try_global_argmax_token(packets.to("xpu"))
    assert actual.item() == 2**24 - 1
    packets[2, 0] = float("nan")
    packets[tp_size - 1, 0] = float("nan")
    actual = ops.try_global_argmax_token(packets.to("xpu"))
    assert actual.item() == 2**24 - tp_size + 2


@pytest.mark.parametrize("tp_size", [4, 8])
def test_dynamic_graph_and_independent_eager_outputs(ops, tp_size):
    n = 31040
    values = torch.zeros(tp_size, n, device="xpu", dtype=torch.float16)
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        packets = torch.cat([
            ops.try_local_argmax_packet(values[r : r + 1], r * n, 0)
            for r in range(tp_size)
        ])
        captured = ops.try_global_argmax_token(packets)
    retained = []
    for rank, index in [(0, 0), (tp_size // 2, 999), (tp_size - 1, n - 1)]:
        values.zero_()
        values[rank, index] = 100
        graph.replay()
        expected = rank * n + index
        assert captured.item() == expected
        eager_packets = torch.cat([
            ops.try_local_argmax_packet(values[r : r + 1], r * n, 0)
            for r in range(tp_size)
        ])
        eager = ops.try_global_argmax_token(eager_packets)
        retained.append((eager, expected))
    for output, expected in retained:
        assert output.item() == expected
    assert len({o.data_ptr() for o, _ in retained}) == len(retained)


@pytest.mark.parametrize("tp_size", [4, 8])
def test_eligibility_and_all_padding(ops, tp_size):
    good = torch.ones(1, 32, device="xpu", dtype=torch.float16)
    for bad in [
        good.cpu(),
        good.float(),
        good.expand(2, -1),
        good[:, ::2],
        torch.empty(1, 0, device="xpu", dtype=torch.float16),
        torch.ones_like(good, requires_grad=True),
        torch._neg_view(good),
    ]:
        assert ops.try_local_argmax_packet(bad, 0, 0) is None
    for start, padding in [(-1, 0), (2**24, 0), (0, -1), (0, 33)]:
        assert ops.try_local_argmax_packet(good, start, padding) is None
    for ranks in (1, 2, 3, 5, 9):
        invalid = torch.ones(ranks, 64, device="xpu")
        assert ops.try_global_argmax_token(invalid) is None
        assert ops.try_global_argmax_token_int32(invalid) is None
    packet = ops.try_local_argmax_packet(good, 700, 32).cpu()
    assert packet[0, 0] == -float("inf")
    assert packet[0, 1] == 700
    assert ops.try_global_argmax_token(torch.ones(tp_size, 64)) is None
    invalid = torch.ones(tp_size, 63, device="xpu")
    assert ops.try_global_argmax_token(invalid) is None


@pytest.mark.parametrize("tp_size", [4, 8])
def test_nondefault_stream_retains_inputs(ops, tp_size):
    stream = torch.xpu.Stream()
    with torch.xpu.stream(stream):
        values = torch.zeros(1, 31040, device="xpu", dtype=torch.float16)
        values[0, -1] = 10
        packet = ops.try_local_argmax_packet(values, 0, 0)
        del values
        packets = packet.expand(tp_size, -1).contiguous()
        token = ops.try_global_argmax_token(packets)
        del packet, packets
    stream.synchronize()
    assert token.item() == 31039


@pytest.mark.parametrize("tp_size", [4, 8])
def test_int32_global_dynamic_graph(ops, tp_size):
    packets = torch.zeros(tp_size, 64, device="xpu")
    packets[:, 1] = 2**24 - tp_size + torch.arange(tp_size, device="xpu")
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        token = ops.try_global_argmax_token_int32(packets)
    for rank in (0, tp_size // 2, tp_size - 1):
        packets[:, 0] = 0
        packets[rank, 0] = 1
        graph.replay()
        assert token.dtype == torch.int32
        assert token.item() == 2**24 - tp_size + rank
