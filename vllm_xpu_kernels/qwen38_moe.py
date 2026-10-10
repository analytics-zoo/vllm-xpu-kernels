# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8 compact MoE SYCL provider; no ESIMD package dependency."""

from functools import lru_cache

import torch

from . import _qwen38_C

_ops = torch.ops.qwen38_moe_sycl

moe_forward_m1_cutlass_nmajor_int4_fp16_shared_compact80_out_v1 = (
    _ops.moe_forward_m1_cutlass_nmajor_int4_fp16_shared_compact80_out_v1
)
moe_forward_multi_m_cutlass_nmajor_int4_fp16_shared_compact80_out_v1 = (
    _ops.moe_forward_multi_m_cutlass_nmajor_int4_fp16_shared_compact80_out_v1
)
moe_forward_multi_m_cutlass_nmajor_int4_fp16_shared_compact80_grouped_out_v1 = (
    _ops.moe_forward_multi_m_cutlass_nmajor_int4_fp16_shared_compact80_grouped_out_v1
)
moe_forward_m1_cutlass_nmajor_int4_fp16_shared_compact80_router_out_v1 = (
    _ops.moe_forward_m1_cutlass_nmajor_int4_fp16_shared_compact80_router_out_v1
)
moe_forward_compact160_out_v1 = _ops.moe_forward_compact160_out_v1
moe_forward_compact160_router_out_v1 = _ops.moe_forward_compact160_router_out_v1
qwen38_moe_compact80_weight_contract_v1 = (
    _ops.qwen38_moe_compact80_weight_contract_v1
)
qwen38_moe_compact160_weight_contract_v1 = (
    _ops.qwen38_moe_compact160_weight_contract_v1
)
qwen38_moe_output_overlaps_v1 = _ops.qwen38_moe_output_overlaps_v1


def _signed_down_weight(weight: torch.Tensor) -> torch.Tensor:
    # Reinterpret live signed-S4 bytes without copying or requantizing.
    return weight.view(torch.int8) if weight.dtype == torch.uint8 else weight


def moe_compact80_down_grouped_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    rows_per_expert: torch.Tensor,
) -> torch.Tensor:
    return _ops.moe_compact80_down_grouped_gemm(
        x, _signed_down_weight(weight), scale, rows_per_expert
    )


def moe_compact160_down_grouped_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    rows_per_expert: torch.Tensor,
) -> torch.Tensor:
    return _ops.moe_compact160_down_grouped_gemm(
        x, _signed_down_weight(weight), scale, rows_per_expert
    )


@lru_cache(maxsize=1)
def compact80_ops():
    """Legacy tuple order used by Compact80FusedMoe and model.py."""
    return (
        moe_forward_m1_cutlass_nmajor_int4_fp16_shared_compact80_out_v1,
        moe_forward_multi_m_cutlass_nmajor_int4_fp16_shared_compact80_out_v1,
        moe_compact80_down_grouped_gemm,
        qwen38_moe_compact80_weight_contract_v1,
    )


@lru_cache(maxsize=1)
def compact160_ops():
    """Legacy tuple order; the same out op handles M=1 and M>1."""
    return (
        moe_forward_compact160_out_v1,
        moe_forward_compact160_out_v1,
        moe_compact160_down_grouped_gemm,
        qwen38_moe_compact160_weight_contract_v1,
        moe_forward_compact160_router_out_v1,
    )


def get_qwen38_moe_m1_workspace_class():
    return torch.classes.qwen38_moe_sycl.Qwen38M1WorkspaceV1


def get_qwen38_moe_m1_direct_workspace_class():
    return _qwen38_C.Qwen38M1WorkspaceDirectV1


def get_qwen38_moe_multi_direct_workspace_class():
    return _qwen38_C.Qwen38MultiWorkspaceDirectV1


def get_qwen38_moe_compact80_grouped_op():
    return (
        moe_forward_multi_m_cutlass_nmajor_int4_fp16_shared_compact80_grouped_out_v1
    )


def get_qwen38_moe_m1_router_hostchain_op():
    return (
        moe_forward_m1_cutlass_nmajor_int4_fp16_shared_compact80_router_out_v1
    )
