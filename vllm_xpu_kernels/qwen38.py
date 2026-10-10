# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8 native SYCL operations; no custom ESIMD package is imported."""

import torch

from . import _qwen38_C  # noqa: F401

def has_esimd_gemm_int4_small_n_v1() -> bool:
    return True


def __getattr__(name: str):
    if name in {
        "esimd_qkv_split_norm_rope",
        "esimd_qkv_split_norm_rope_mrope_v1",
    }:
        from . import qwen38_qsa

        return getattr(qwen38_qsa, name)
    raise AttributeError(f"unsupported Qwen3.8 SYCL provider operation: {name}")


def q4_0_quantize(
    input: torch.Tensor, qweight: torch.Tensor, scale: torch.Tensor
) -> None:
    torch.ops._qwen38_C.q4_0_quantize(input, qweight, scale)


def int4_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    torch.ops._qwen38_C.int4_linear(input, weight, scale, output)
    # 旧 indexer 会校验返回值必须就是调用方输出，不能暴露底层 void ABI。
    return output


def int4_linear_fused2(
    input: torch.Tensor,
    weight0: torch.Tensor,
    scale0: torch.Tensor,
    output0: torch.Tensor,
    weight1: torch.Tensor,
    scale1: torch.Tensor,
    output1: torch.Tensor,
) -> torch.Tensor:
    torch.ops._qwen38_C.int4_linear_fused2(
        input, weight0, scale0, output0, weight1, scale1, output1
    )
    return output0


# 保留旧调用方返回值契约；底层 Torch schemas 仍为原地写入的 void。
esimd_gemv_int4 = int4_linear
esimd_gemm_int4_pgrp = int4_linear
esimd_gemm_int4_small_n_v1 = int4_linear
esimd_gemv_int4_fused2 = int4_linear_fused2


def ngram_decode_ids(
    input_ids: torch.Tensor,
    context: torch.Tensor,
    multipliers: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    torch.ops._qwen38_C.ngram_decode_ids(
        input_ids, context, multipliers, output
    )
    return output


def ngram_host_lookup(
    weight: torch.Tensor,
    ids: torch.Tensor,
    output: torch.Tensor,
    vocab_start: int,
    vocab_end: int,
) -> torch.Tensor:
    return torch.ops._qwen38_C.ngram_host_lookup(
        weight, ids, output, vocab_start, vocab_end
    )


def ngram_host_lookup_chunked(
    weights: list[torch.Tensor],
    ids: torch.Tensor,
    output: torch.Tensor,
    vocab_start: int,
    vocab_end: int,
) -> torch.Tensor:
    return torch.ops._qwen38_C.ngram_host_lookup_chunked(
        weights, ids, output, vocab_start, vocab_end
    )
