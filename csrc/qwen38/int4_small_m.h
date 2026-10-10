// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>

namespace vllm::qwen38 {

// One asynchronous submission on the caller's queue. Returns false without
// submitting when the layout/shape is unsupported; caller owns tensor and
// stream lifetime validation. The packed Q4_0 matrix is [N,K/2], low nibble
// first, with one signed FP16 scale per 128 consecutive K elements. The
// packed weight pointer must be 4-byte aligned for the word-load fast path.
// The input pointer must be 64-byte aligned for native Xe 2D A block reads.
// Only BMG G31, M=2..8, N>=256 divisible by 16, and positive K divisible
// by 128 are accepted; all other cases return false before queue submission.
bool launch_int4_small_m(
    sycl::queue& queue,
    const sycl::half* x,
    const uint8_t* w,
    const sycl::half* s,
    sycl::half* out,
    int m,
    int n,
    int k);

}  // namespace vllm::qwen38
