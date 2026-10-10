// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/types.h>

namespace vllm::qwen38 {

// M=1 GDN gated norm followed by the main-line INT4 output projection.
// The two submissions share the current XPU stream. Scratch is per call and
// owned/recorded by the native allocator, never allocated from Python.
void gdn_norm_int4_sycl(
    const torch::Tensor& x,
    const torch::Tensor& z,
    const torch::Tensor& norm_weight,
    const torch::Tensor& int4_weight,
    const torch::Tensor& int4_scale,
    torch::Tensor& output,
    int64_t hv,
    int64_t v,
    double eps,
    bool sigmoid_gate);

}  // namespace vllm::qwen38
