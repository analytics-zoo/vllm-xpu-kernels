// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/types.h>

namespace vllm::qwen38 {

void q4_0_quantize(
    const torch::Tensor& input, torch::Tensor& qweight, torch::Tensor& scale);

void int4_linear(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& scale,
    torch::Tensor& output);

void int4_linear_fused2(
    const torch::Tensor& input,
    const torch::Tensor& weight0,
    const torch::Tensor& scale0,
    torch::Tensor& output0,
    const torch::Tensor& weight1,
    const torch::Tensor& scale1,
    torch::Tensor& output1);

}  // namespace vllm::qwen38
