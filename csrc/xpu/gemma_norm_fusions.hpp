#pragma once

#include <torch/torch.h>

#include <tuple>

namespace vllm::gemma_norm {

// The returned pair is (normalized_output, residual_output). Inputs are not
// mutated. FP16 storage rounding matches the separate upstream operations.
std::tuple<at::Tensor, at::Tensor> scaled_add_rms_norm(
    const at::Tensor& input,
    const at::Tensor& residual,
    const at::Tensor& weight,
    double epsilon,
    double scalar);

// Device-scalar variant. A single FP16 scalar stays on the current XPU stream;
// there is no host readback. The add and multiply each round to FP16.
std::tuple<at::Tensor, at::Tensor> scaled_add_rms_norm_tensor(
    const at::Tensor& input,
    const at::Tensor& residual,
    const at::Tensor& weight,
    double epsilon,
    const at::Tensor& scalar);

std::tuple<at::Tensor, at::Tensor> norm_add_norm(
    const at::Tensor& input,
    const at::Tensor& residual,
    const at::Tensor& weight1,
    const at::Tensor& weight2,
    double epsilon1,
    double epsilon2);

}  // namespace vllm::gemma_norm
