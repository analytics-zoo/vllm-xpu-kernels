# SPDX-License-Identifier: Apache-2.0
"""Native Qwen3.8 GDN provider for the vLLM sequential ESIMD-call ABI.

Only the supported TP4/TP8 sequential core and v2 speculative layout are
exported. An older native DSO without a GDN schema must leave the matching
function absent so the vLLM adapter can choose its normal fallback before any
state mutation. This module never imports the ESIMD package.
"""

import os
from functools import lru_cache

import torch

_build_only_library = os.environ.get("QWEN38_SYCL_LIBRARY")
if _build_only_library:
    torch.ops.load_library(_build_only_library)
else:
    from . import _qwen38_C  # noqa: F401 - registers the native Torch ops


@lru_cache(maxsize=4)
def _native_op_registered(name: str) -> bool:
    # The DSO is loaded before these probes and its exports below are fixed
    # at import time. Avoid a schema/dispatch lookup per layer on MTP verify.
    # Only availability is cached: geometry, tensor/state and stream guards
    # remain in the adapter and native entry points.
    qualified = f"_qwen38_C::{name}"
    return bool(torch._C._jit_get_schemas_for_operator(qualified)) and bool(
        torch._C._dispatch_has_kernel_for_dispatch_key(qualified, "XPU")
    )


def supports_gdn_geometry(h: int, hv: int, k: int, v: int) -> bool:
    """Return the only geometries implemented by the native GDN kernels."""
    return (h, hv, k, v) in ((4, 12, 128, 128), (2, 6, 128, 128))


def has_esimd_gdn_conv_fused_seq(h: int, hv: int, k: int, v: int) -> bool:
    return _native_op_registered("gdn_decode_sycl") and supports_gdn_geometry(
        h, hv, k, v
    )


def has_esimd_gdn_conv_fused_seq_spec_v2() -> bool:
    """Legacy adapter capability probe; geometry is checked by its guard."""
    return _native_op_registered("gdn_spec_v2_sycl")


if _native_op_registered("gdn_decode_sycl"):

    def esimd_gdn_conv_fused_seq(
        qkvz: torch.Tensor,
        conv_state: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_bias: torch.Tensor,
        conv_state_indices: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ba: torch.Tensor,
        ssm_state: torch.Tensor,
        ssm_state_indices: torch.Tensor,
        output: torch.Tensor,
        z_out: torch.Tensor,
        n: int,
        h: int,
        hv: int,
        k: int,
        v: int,
        scale: float,
    ) -> torch.Tensor:
        if not supports_gdn_geometry(h, hv, k, v):
            raise ValueError(
                "native sequential GDN supports only TP4/TP8 Qwen3.8"
            )
        if n != qkvz.shape[0]:
            raise ValueError("N must match the projected token count")
        torch.ops._qwen38_C.gdn_decode_sycl(
            qkvz,
            conv_state,
            conv_weight,
            conv_bias,
            conv_state_indices,
            a_log,
            dt_bias,
            ba,
            ssm_state,
            ssm_state_indices,
            output,
            z_out,
            scale,
        )
        return output


if _native_op_registered("gdn_spec_v2_sycl"):

    def esimd_gdn_conv_fused_seq_spec_v2(
        qkvz: torch.Tensor,
        conv_state: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_bias: torch.Tensor,
        spec_state_indices: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ba: torch.Tensor,
        ssm_state: torch.Tensor,
        output: torch.Tensor,
        z_out: torch.Tensor,
        token_indx: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        num_spec_decodes: int,
        num_spec_tokens: int,
        h: int,
        hv: int,
        k: int,
        v: int,
        scale: float,
    ) -> torch.Tensor:
        if not (
            supports_gdn_geometry(h, hv, k, v) and 2 <= num_spec_tokens <= 8
        ):
            raise ValueError("native spec-v2 GDN requires TP4/TP8 and M=2..8")
        if num_spec_decodes * num_spec_tokens != qkvz.shape[0]:
            raise ValueError("spec-v2 sequence and token counts do not match")
        torch.ops._qwen38_C.gdn_spec_v2_sycl(
            qkvz,
            conv_state,
            conv_weight,
            conv_bias,
            spec_state_indices,
            a_log,
            dt_bias,
            ba,
            ssm_state,
            output,
            z_out,
            token_indx,
            num_accepted_tokens,
            num_spec_decodes,
            num_spec_tokens,
            scale,
        )
        return output


if _native_op_registered("gdn_norm_gate_sycl"):

    def esimd_rms_norm_gated(
        x: torch.Tensor,
        z: torch.Tensor,
        weight: torch.Tensor,
        output: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        # The adapter passes [M*HV, 128]. Views do not allocate a per-token
        # scratch buffer; the native helper writes into caller-owned output.
        rows = x.shape[0]
        torch.ops._qwen38_C.gdn_norm_gate_sycl(
            x.view(1, rows, 128),
            z.view(1, rows, 128),
            weight,
            output.view(1, rows * 128),
            eps,
            False,
        )
        return output


def has_gdn_norm_int4_sycl() -> bool:
    return _native_op_registered("gdn_norm_int4_sycl")


def _packed_int4_weight(weight: torch.Tensor) -> torch.Tensor:
    if weight.dtype == torch.uint8:
        return weight
    if weight.dtype == torch.int32:
        return weight.view(torch.uint8)
    raise TypeError("GDN INT4 output weight must be uint8 or int32 packed")


if has_gdn_norm_int4_sycl():

    def esimd_norm_gemv_int4_pert(
        x: torch.Tensor,
        z: torch.Tensor,
        norm_weight: torch.Tensor,
        gemv_weight: torch.Tensor,
        gemv_scale: torch.Tensor,
        output: torch.Tensor,
        hv: int,
        v: int,
        eps: float,
    ) -> torch.Tensor:
        # The current adapter caches an int32 view. A native-only plan may
        # cache uint8 directly and avoid this CPU-only dtype view.
        packed = _packed_int4_weight(gemv_weight)
        torch.ops._qwen38_C.gdn_norm_int4_sycl(
            x,
            z,
            norm_weight,
            packed,
            gemv_scale,
            output,
            hv,
            v,
            eps,
            False,
        )
        return output

    def esimd_norm_gemv_int4_sigmoid(
        x: torch.Tensor,
        z: torch.Tensor,
        norm_weight: torch.Tensor,
        gemv_weight: torch.Tensor,
        gemv_scale: torch.Tensor,
        output: torch.Tensor,
        hv: int,
        v: int,
        eps: float,
    ) -> torch.Tensor:
        packed = _packed_int4_weight(gemv_weight)
        torch.ops._qwen38_C.gdn_norm_int4_sycl(
            x,
            z,
            norm_weight,
            packed,
            gemv_scale,
            output,
            hv,
            v,
            eps,
            True,
        )
        return output
