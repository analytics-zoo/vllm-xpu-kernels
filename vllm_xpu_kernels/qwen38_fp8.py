# SPDX-License-Identifier: Apache-2.0
"""Native-width Qwen3.8 W8A16 SYCL-TLA kernels."""

# Missing optional exports in an older DSO are eligibility misses.
from . import _qwen38_fp8_C as _native
from ._qwen38_fp8_C import (Workspace, try_gdn_norm,  # noqa: F401
                            try_qwen38_fp8_linear)

try_qwen4_dense_fp8_linear = getattr(
    _native, "try_qwen4_dense_fp8_linear", None)
try_qwen4_dense_fp8_pair = getattr(_native, "try_qwen4_dense_fp8_pair", None)

try_qwen4_dense_qkv_norm_rope = getattr(
    _native, "try_qwen4_dense_qkv_norm_rope", None)

try_qwen4_dense_fp8_gated_linear = getattr(
    _native, "try_qwen4_dense_fp8_gated_linear", None)
