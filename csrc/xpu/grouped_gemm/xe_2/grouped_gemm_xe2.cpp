#include <torch/all.h>
#include "grouped_gemm_xe2.h"
#include "grouped_gemm_xe2_interface.hpp"

torch::Tensor cutlass_grouped_gemm_xe2(
    torch::Tensor ptr_A,
    torch::Tensor ptr_B,
    const c10::optional<at::Tensor>& ptr_scales,
    const c10::optional<at::Tensor>& ptr_bias,
    torch::Tensor ptr_D,
    torch::Tensor rows_per_expert,
    int64_t N,
    int64_t K,
    int64_t num_experts,
    bool is_B_int4,
    bool is_B_mxfp4,
    bool is_B_fp8_block) {
  return MoE::cutlass_grouped_gemm_xe2_impl(
      ptr_A,
      ptr_B,
      ptr_scales,
      ptr_bias,
      ptr_D,
      rows_per_expert,
      N,
      K,
      num_experts,
      is_B_int4,
      is_B_mxfp4,
      is_B_fp8_block);
}

torch::Tensor fp8_block_gemm_xe2(
    const torch::Tensor& ptr_A,
    const torch::Tensor& ptr_B,
    const torch::Tensor& ptr_scales) {
  return MoE::fp8_block_gemm_xe2_impl(ptr_A, ptr_B, ptr_scales);
}

torch::Tensor fp8_block_dequant_xe2(
    const torch::Tensor& ptr_B,
    const torch::Tensor& ptr_scales) {
  return MoE::fp8_block_dequant_xe2_impl(ptr_B, ptr_scales);
}