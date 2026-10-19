# SPDX-License-Identifier: Apache-2.0
"""Native-width Qwen3.8 W8A16 SYCL-TLA kernels."""

from ._qwen38_fp8_C import (Workspace, try_gdn_norm,  # noqa: F401
                            try_qwen38_fp8_linear)
