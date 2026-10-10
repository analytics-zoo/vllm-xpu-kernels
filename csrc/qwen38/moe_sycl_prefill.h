// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>

namespace vllm::qwen38::moe_sycl {

// The caller retains every allocation through completion of the current
// in-order queue. counts are row counts, not cumulative offsets; their sum
// must equal m. Weight rows contain exactly k/2 bytes (no K padding).
struct PrefillDown {
  const sycl::half* x;         // [m,k]
  const std::uint8_t* w2;      // [512,2560,k/2], signed S4
  const sycl::half* s2;        // [512,2560,ceil(k/128)]
  const std::int32_t* counts;  // [512]
  sycl::half* output;          // [m,2560]
  std::int32_t* row_prefix;    // [513], workspace
  std::int32_t* tile_prefix;   // [513], workspace
  int m;
  int k;  // 80 or 160
};

// False means unsupported before any submission. Device-side counts must
// already satisfy the contract; no host readback or synchronization occurs.
bool try_prefill_down(sycl::queue& queue, const PrefillDown& args);

}  // namespace vllm::qwen38::moe_sycl
