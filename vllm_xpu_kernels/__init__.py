# SPDX-License-Identifier: Apache-2.0

from importlib.util import find_spec

from .flash_attn_interface import flash_attn_varlen_func  # noqa: F401

# Optional module may be absent in older/precompiled packages. A present but
# broken extension must propagate its ABI/load error.
LINEAR_TLA_KERNELS_AVAILABLE = find_spec(f"{__name__}._linear_C") is not None
if LINEAR_TLA_KERNELS_AVAILABLE:
    from . import _linear_C  # noqa: F401

# Optional independent RMSNorm module. Surface ABI/load errors if present.
RMS_KERNELS_AVAILABLE = find_spec(f"{__name__}._rms_C") is not None
if RMS_KERNELS_AVAILABLE:
    from . import _rms_C  # noqa: F401

# Missing optional module permits other paths; existing ABI/load errors propagate.
FP16_LINEAR_KERNELS_AVAILABLE = find_spec(f"{__name__}._fp16_C") is not None
if FP16_LINEAR_KERNELS_AVAILABLE:
    from . import _fp16_C  # noqa: F401
