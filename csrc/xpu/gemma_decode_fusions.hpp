#pragma once

#include <torch/torch.h>

#include <tuple>

namespace vllm::gemma_decode {

std::tuple<at::Tensor, at::Tensor, at::Tensor> qkv_norm_rope(
    const at::Tensor& qkv,
    const at::Tensor& q_weight,
    const at::Tensor& k_weight,
    const at::Tensor& positions,
    const at::Tensor& cos_sin_cache,
    double q_epsilon,
    double k_epsilon,
    double v_epsilon);

std::tuple<at::Tensor, at::Tensor> norm_router_norm(
    const at::Tensor& input,
    const at::Tensor& router_scale,
    const at::Tensor& root_size,
    const at::Tensor& projection,
    const at::Tensor& moe_weight,
    double router_epsilon,
    double moe_epsilon);

}  // namespace vllm::gemma_decode
