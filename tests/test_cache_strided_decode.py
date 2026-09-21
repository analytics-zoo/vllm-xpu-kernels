# SPDX-License-Identifier: Apache-2.0
"""Check strided decode cache writes and untouched padding."""

import pytest
import torch

import vllm_xpu_kernels._C  # noqa: F401


@pytest.mark.parametrize("num_heads", [1, 2, 8])
@pytest.mark.parametrize("page_size", [64, 256])
@pytest.mark.parametrize(
    "num_tokens,dtype,offset",
    [(1, torch.float16, 0), (8, torch.float16, 1),
     (9, torch.float16, 0), (4, torch.bfloat16, 1)],
)
def test_small_decode_strided_cache(num_heads, page_size, num_tokens, dtype,
                                    offset):
    torch.manual_seed(823)
    head_size = 512
    token_stride = num_heads * head_size + 32
    key_storage = torch.randn(num_tokens * token_stride + 1, dtype=dtype)
    value_storage = torch.randn_like(key_storage)
    shape = (num_tokens, num_heads, head_size)
    strides = (token_stride, head_size, 1)
    key = key_storage.as_strided(shape, strides, offset)
    value = value_storage.as_strided(shape, strides, offset)

    # Include gaps between tokens and pages, as well as interleaved K/V heads.
    page_stride = num_heads * 2 * head_size + 32
    cache_strides = (page_size * page_stride + 32, page_stride,
                     2 * head_size, 1)
    cache_shape = (3, page_size, num_heads, head_size)
    storage_size = (2 * cache_strides[0] + (page_size - 1) * page_stride
                    + num_heads * 2 * head_size)
    expected = torch.full((storage_size,), -9, dtype=dtype)
    storage = expected.to("xpu")
    slots = torch.arange(num_tokens, dtype=torch.int64) * 3 + page_size - 1
    if num_tokens > 1:
        slots[-1] = -1
    expected_k = expected.as_strided(cache_shape, cache_strides, 0)
    expected_v = expected.as_strided(cache_shape, cache_strides, head_size)
    for token, slot in enumerate(slots.tolist()):
        if slot >= 0:
            expected_k[slot // page_size, slot % page_size] = key[token]
            expected_v[slot // page_size, slot % page_size] = value[token]

    key_xpu = key_storage.to("xpu").as_strided(shape, strides, offset)
    value_xpu = value_storage.to("xpu").as_strided(shape, strides, offset)
    scale = torch.ones((), device="xpu", dtype=torch.float32)
    torch.ops._C_cache_ops.reshape_and_cache_flash(
        key_xpu, value_xpu,
        storage.as_strided(cache_shape, cache_strides, 0),
        storage.as_strided(cache_shape, cache_strides, head_size),
        slots.to("xpu"), "auto", scale, scale)
    # Check the entire allocation, including every untouched slot and gap.
    torch.testing.assert_close(storage.cpu(), expected, rtol=0, atol=0)
