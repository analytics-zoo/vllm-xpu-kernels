# SPDX-License-Identifier: Apache-2.0
"""Standalone compact MoE DOWN contract checks; no vLLM server required."""

import random
import os
from functools import lru_cache

import pytest


@lru_cache(maxsize=1)
def _prefill_op():
    torch = pytest.importorskip("torch")
    integrated = os.environ.get("QWEN38_SYCL_LIBRARY")
    if integrated:
        torch.ops.load_library(integrated)
        def operation(x, weight, scales, counts):
            compact = x.shape[1]
            return getattr(torch.ops.qwen38_moe_sycl,
                           f"moe_compact{compact}_down_grouped_gemm")(
                               x, weight, scales, counts)
        return operation
    library = os.environ.get("MOE_SYCL_TEST_LIB")
    if library:
        torch.ops.load_library(library)
        return torch.ops.moe_sycl_test.prefill_down
    try:
        return torch.ops._qwen38_C.moe_compact_down_grouped_gemm
    except AttributeError:
        pytest.skip("sidecar Torch ABI has not been integrated")


@pytest.mark.parametrize("m", [1, 2, 8, 9, 31, 257])
def test_prefill_tile_upper_bound(m: int) -> None:
    rng = random.Random(m)
    counts = [0] * 512
    for _ in range(m):
        counts[rng.randrange(512)] += 1
    row_prefix = [0]
    tile_prefix = [0]
    for count in counts:
        row_prefix.append(row_prefix[-1] + count)
        tile_prefix.append(tile_prefix[-1] + (count + 7) // 8)
    assert row_prefix[-1] == m
    assert tile_prefix[-1] <= (m + 7) // 8 + 511
    for expert, count in enumerate(counts):
        for tile in range(tile_prefix[expert], tile_prefix[expert + 1]):
            first = row_prefix[expert] + (tile - tile_prefix[expert]) * 8
            last = min(row_prefix[expert + 1], first + 8)
            assert first < last <= row_prefix[expert] + count


@pytest.mark.parametrize("k", [80, 160])
@pytest.mark.parametrize("nibble", [1, 15])
def test_prefill_down_xpu(k: int, nibble: int) -> None:
    torch = pytest.importorskip("torch")
    if not torch.xpu.is_available():
        pytest.skip("XPU not available")
    op = _prefill_op()

    # Deliberately cross an expert boundary inside the first 8-row tile and
    # exercise the K=160 final group of 32, without adding weight padding.
    x = torch.ones((4, k), dtype=torch.float16, device="xpu")
    byte = nibble | (nibble << 4)
    packed = torch.full((512, 2560, k // 2), byte, dtype=torch.uint8,
                        device="xpu").view(torch.int8)
    scale = torch.ones((512, 2560, (k + 127) // 128),
                       dtype=torch.float16, device="xpu")
    if k == 160:
        scale[:, :, 1] = 2
    counts = torch.zeros(512, dtype=torch.int32, device="xpu")
    counts[0] = 3
    counts[1] = 1
    result = op(x, packed, scale, counts)
    q = nibble if nibble < 8 else nibble - 16
    expected = q * (80 if k == 80 else 128 + 2 * 32)
    torch.testing.assert_close(
        result, torch.full_like(result, expected), atol=0.1, rtol=0)


@pytest.mark.parametrize("k", [80, 160])
def test_prefill_down_independent_full_output(k: int) -> None:
    torch = pytest.importorskip("torch")
    if not torch.xpu.is_available():
        pytest.skip("XPU not available")
    op = _prefill_op()
    torch.manual_seed(20261004 + k)
    counts_cpu = torch.zeros(512, dtype=torch.int32)
    counts_cpu[0], counts_cpu[13], counts_cpu[511] = 3, 1, 5
    x_cpu = torch.randn((9, k), dtype=torch.float16) * 0.2
    packed_cpu = torch.randint(0, 256, (512, 2560, k // 2), dtype=torch.uint8)
    scale_cpu = torch.randn((512, 2560, (k + 127) // 128),
                            dtype=torch.float16) * 0.02
    result = op(x_cpu.xpu(), packed_cpu.view(torch.int8).xpu(),
                scale_cpu.xpu(), counts_cpu.xpu()).cpu()

    golden = torch.empty_like(result)
    first = 0
    for expert in (0, 13, 511):
        rows = int(counts_cpu[expert])
        packed = packed_cpu[expert].to(torch.int16)
        low = packed & 15
        high = (packed >> 4) & 15
        nib = torch.stack((low, high), dim=-1).reshape(2560, k)
        signed = torch.where(nib < 8, nib, nib - 16).float()
        expanded_scale = scale_cpu[expert].float().repeat_interleave(128, dim=1)[:, :k]
        # Match the FP16 dequantized operand, then independently accumulate
        # in FP32 and round the final result to FP16.
        dequant = (signed * expanded_scale).half().float()
        golden[first:first + rows] = (x_cpu[first:first + rows].float()
                                      @ dequant.T).half()
        first += rows
    assert first == 9
    torch.testing.assert_close(result, golden, atol=0.05, rtol=0.01)


@pytest.mark.parametrize("k", [80, 160])
def test_prefill_down_many_rows_full_output(k: int) -> None:
    torch = pytest.importorskip("torch")
    if not torch.xpu.is_available():
        pytest.skip("XPU not available")
    op = _prefill_op()
    torch.manual_seed(20261005 + k)
    rows = 512  # Exercise the by-expert schedule, not the short-prefill path.
    x = (torch.randn((rows, k), dtype=torch.float16, device="xpu") * 0.2)
    packed = torch.full((512, 2560, k // 2), 0x11,
                        dtype=torch.uint8, device="xpu")
    packed[0] = torch.randint(0, 256, (2560, k // 2),
                              dtype=torch.uint8, device="xpu")
    scales = torch.full((512, 2560, (k + 127) // 128), 0.01,
                        dtype=torch.float16, device="xpu")
    counts = torch.zeros(512, dtype=torch.int32, device="xpu")
    counts[0] = rows
    result = op(x, packed.view(torch.int8), scales, counts)

    bytes_ = packed[0].to(torch.int16)
    nibble = torch.stack((bytes_ & 15, bytes_ >> 4), dim=-1)
    nibble = nibble.reshape(2560, k)
    signed = torch.where(nibble < 8, nibble, nibble - 16).float()
    expanded_scale = scales[0].float().repeat_interleave(128, dim=1)[:, :k]
    dequant = (signed * expanded_scale).half().float()
    golden = (x.float() @ dequant.T).half()
    torch.testing.assert_close(result, golden, atol=0.05, rtol=0.01)


@pytest.mark.parametrize("k", [80, 160])
def test_prefill_down_all_experts_active(k: int) -> None:
    torch = pytest.importorskip("torch")
    if not torch.xpu.is_available():
        pytest.skip("XPU not available")
    op = _prefill_op()
    x = torch.ones((512, k), dtype=torch.float16, device="xpu")
    packed = torch.full((512, 2560, k // 2), 0x11,
                        dtype=torch.int8, device="xpu")
    per_expert = (0.005 + torch.arange(512, device="xpu") * 0.00001).half()
    scale = per_expert[:, None, None].expand(
        512, 2560, (k + 127) // 128).contiguous()
    counts = torch.ones(512, dtype=torch.int32, device="xpu")
    result = op(x, packed, scale, counts)
    expected = (per_expert * k)[:, None].expand_as(result)
    torch.testing.assert_close(result, expected, atol=0.03, rtol=0.01)


@pytest.mark.parametrize("rows", [4, 512])
@pytest.mark.parametrize("invalid", ["negative", "wrong_sum"])
def test_prefill_invalid_gpu_counts_mark_all_output(rows: int,
                                                     invalid: str) -> None:
    torch = pytest.importorskip("torch")
    if not torch.xpu.is_available():
        pytest.skip("XPU not available")
    op = _prefill_op()
    k = 80
    x = torch.ones((rows, k), dtype=torch.float16, device="xpu")
    packed = torch.full((512, 2560, k // 2), 0x11,
                        dtype=torch.int8, device="xpu")
    scale = torch.ones((512, 2560, 1), dtype=torch.float16,
                       device="xpu")
    counts = torch.zeros(512, dtype=torch.int32, device="xpu")
    if invalid == "negative":
        counts[0] = -1
        counts[1] = rows + 1
    else:
        counts[0] = rows - 1
    result = op(x, packed, scale, counts)
    assert bool(torch.isnan(result).all())
