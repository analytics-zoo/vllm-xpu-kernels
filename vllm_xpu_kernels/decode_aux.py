# SPDX-License-Identifier: Apache-2.0
"""Optional small-decode helpers; unsupported shapes use the original ops."""
import torch

_PREPARE = getattr(torch.ops._xpu_C, "moe_prepare_small_m", None)
_GATHER = getattr(torch.ops._xpu_C, "moe_gather_small_m", None)
_GELU = getattr(torch.ops._xpu_C, "gelu_tanh_and_mul_small_m", None)
_TRY_ROPE = getattr(torch.ops._xpu_C, "try_rotary_embedding_small_m", None)
_ROPE = getattr(torch.ops._xpu_C, "rotary_embedding_small_m", None)


def gelu_tanh_and_mul(out, x):
    if (_GELU is not None and x.device.type == "xpu"
            and x.dtype == torch.float16 and x.ndim == 2
            and 1 <= x.shape[0] <= 64 and x.shape[1] in (704, 2112)
            and x.is_contiguous() and x.storage_offset() % 4 == 0):
        _GELU(out, x)
        return True
    return False


def rotary_embedding(positions, query, key, head, cache, neox):
    if (_TRY_ROPE is not None and neox and key is not None
            and query.device.type == "xpu"):
        return _TRY_ROPE(positions, query, key, head, cache)
    if (_ROPE is not None and neox and key is not None
            and query.device.type == "xpu" and query.dtype == torch.float16
            and key.dtype == torch.float16 and cache.dtype == torch.float16
            and positions.dtype == torch.int64 and positions.ndim == 1
            and 1 <= positions.numel() <= 8 and query.ndim == key.ndim == 2
            and head in (256, 512) and query.stride(1) == key.stride(1) == 1
            and positions.is_contiguous() and cache.is_contiguous()
            and cache.ndim == 2 and 0 < cache.shape[1] <= head
            and cache.shape[1] % 64 == 0):
        _ROPE(positions, query, key, head, cache)
        return True
    return False
