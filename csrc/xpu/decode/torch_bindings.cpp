#include "core/registration.h"
#include "xpu/gemma_decode_fusions.hpp"
#include "xpu/gemma_norm_fusions.hpp"

#include <torch/library.h>

namespace {

std::tuple<at::Tensor, at::Tensor> scaled_add_meta(
    const at::Tensor& input,
    const at::Tensor& residual,
    const at::Tensor& weight,
    double epsilon,
    double scalar) {
  return {at::empty_like(input), at::empty_like(input)};
}

std::tuple<at::Tensor, at::Tensor> norm_add_meta(
    const at::Tensor& input,
    const at::Tensor& residual,
    const at::Tensor& weight1,
    const at::Tensor& weight2,
    double epsilon1,
    double epsilon2) {
  return {at::empty_like(input), at::empty_like(input)};
}

std::tuple<at::Tensor, at::Tensor> scaled_add_tensor_meta(
    const at::Tensor& input,
    const at::Tensor& residual,
    const at::Tensor& weight,
    double epsilon,
    const at::Tensor& scalar) {
  return {at::empty_like(input), at::empty_like(input)};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> qkv_meta(
    const at::Tensor& qkv,
    const at::Tensor& q_weight,
    const at::Tensor& k_weight,
    const at::Tensor& positions,
    const at::Tensor& cos_sin_cache,
    double q_epsilon,
    double k_epsilon,
    double v_epsilon) {
  // This interface's supported contract is Q8/KV4/head_dim256.
  return {
      at::empty_symint({qkv.sym_size(0), 2048}, qkv.options()),
      at::empty_symint({qkv.sym_size(0), 1024}, qkv.options()),
      at::empty_symint({qkv.sym_size(0), 1024}, qkv.options())};
}

std::tuple<at::Tensor, at::Tensor> norm_router_meta(
    const at::Tensor& input,
    const at::Tensor& router_scale,
    const at::Tensor& root_size,
    const at::Tensor& projection,
    const at::Tensor& moe_weight,
    double router_epsilon,
    double moe_epsilon) {
  return {
      at::empty_symint(
          {input.sym_size(0), projection.sym_size(0)},
          input.options().dtype(at::kFloat)),
      at::empty_like(input)};
}

}  // namespace

// Add new operations without replacing any existing _xpu_C operator or ABI.
TORCH_LIBRARY_FRAGMENT(_xpu_C, m) {
  m.def(
      "gemma_scaled_add_rms_norm(Tensor input, Tensor residual, Tensor weight, "
      "float epsilon, float scalar) -> (Tensor, Tensor)");
  m.def(
      "gemma_scaled_add_rms_norm_tensor(Tensor input, Tensor residual, Tensor "
      "weight, float epsilon, Tensor scalar) -> (Tensor, Tensor)");
  m.def(
      "gemma_norm_add_norm(Tensor input, Tensor residual, Tensor weight1, "
      "Tensor weight2, float epsilon1, float epsilon2) -> (Tensor, Tensor)");
  m.def(
      "gemma_qkv_norm_rope(Tensor qkv, Tensor q_weight, Tensor k_weight, "
      "Tensor positions, Tensor cos_sin_cache, float q_epsilon, "
      "float k_epsilon, float v_epsilon) -> (Tensor, Tensor, Tensor)");
  m.def(
      "gemma_norm_router_norm(Tensor input, Tensor router_scale, Tensor "
      "root_size, Tensor projection, Tensor moe_weight, float router_epsilon, "
      "float moe_epsilon) -> (Tensor, Tensor)");
  m.impl(
      "gemma_scaled_add_rms_norm",
      torch::kXPU,
      &vllm::gemma_norm::scaled_add_rms_norm);
  m.impl(
      "gemma_scaled_add_rms_norm_tensor",
      torch::kXPU,
      &vllm::gemma_norm::scaled_add_rms_norm_tensor);
  m.impl(
      "gemma_norm_add_norm", torch::kXPU, &vllm::gemma_norm::norm_add_norm);
  m.impl(
      "gemma_qkv_norm_rope", torch::kXPU, &vllm::gemma_decode::qkv_norm_rope);
  m.impl(
      "gemma_norm_router_norm",
      torch::kXPU,
      &vllm::gemma_decode::norm_router_norm);
  m.impl("gemma_scaled_add_rms_norm", torch::kMeta, &scaled_add_meta);
  m.impl("gemma_scaled_add_rms_norm_tensor", torch::kMeta, &scaled_add_tensor_meta);
  m.impl("gemma_norm_add_norm", torch::kMeta, &norm_add_meta);
  m.impl("gemma_qkv_norm_rope", torch::kMeta, &qkv_meta);
  m.impl("gemma_norm_router_norm", torch::kMeta, &norm_router_meta);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
