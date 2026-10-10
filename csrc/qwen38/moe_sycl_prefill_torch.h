// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <ATen/ATen.h>

namespace vllm::qwen38::moe_sycl {

// Independent Torch ABI for the Compact80FusedMoe direct/prefill DOWN stage.
// The main integration registers this under its chosen extension namespace.
at::Tensor compact_down_grouped_gemm(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& scale,
    const at::Tensor& rows_per_expert);

}  // namespace vllm::qwen38::moe_sycl
