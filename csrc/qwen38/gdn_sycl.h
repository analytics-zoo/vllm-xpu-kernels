// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/types.h>
#include <optional>
#include <vector>

namespace vllm::qwen38 {

// Host-only, side-effect-free preflight for dense/paged tensor views. It
// checks physical addresses even when distinct Storage objects share memory.
bool tensors_disjoint_host(
    const std::vector<torch::Tensor>& writes,
    const std::vector<std::optional<torch::Tensor>>& reads);

// Sequential [q|k|v|z] layout. TP4 is H=4, HV=12, K=V=128;
// TP8 (H=2, HV=6) uses the same mapping. All outputs are caller-owned.
void gdn_decode_sycl(
    const torch::Tensor& qkvz,
    torch::Tensor& conv_state,
    const torch::Tensor& conv_weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& conv_indices,
    const torch::Tensor& a_log,
    const torch::Tensor& dt_bias,
    const torch::Tensor& ba,
    torch::Tensor& ssm_state,
    const torch::Tensor& ssm_indices,
    torch::Tensor& output,
    torch::Tensor& z,
    double scale);

// MTP v2: [slot, M+2, dim] packed conv state (or legacy [slot, 3, dim]);
// A_log is FP32 and state carry remains FP32 across all M draft tokens.
// A_log FP16 / interleaved qkvz are deliberately not accepted here.
void gdn_spec_v2_sycl(
    const torch::Tensor& qkvz,
    torch::Tensor& conv_state,
    const torch::Tensor& conv_weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& spec_indices,
    const torch::Tensor& a_log,
    const torch::Tensor& dt_bias,
    const torch::Tensor& ba,
    torch::Tensor& ssm_state,
    torch::Tensor& output,
    torch::Tensor& z,
    const torch::Tensor& token_indices,
    const torch::Tensor& accepted,
    int64_t sequences,
    int64_t tokens_per_sequence,
    double scale);

// Head-wise RMSNorm followed by either SiLU(z) or sigmoid(z). `normalized`
// is [M,HV*128] and may be passed directly to the main-line INT4 GEMV/GEMM.
// This is a preprocessing helper, not a renamed ESIMD operator.
void gdn_norm_gate_sycl(
    const torch::Tensor& x,
    const torch::Tensor& z,
    const torch::Tensor& weight,
    torch::Tensor& normalized,
    double eps,
    bool sigmoid_gate);

}  // namespace vllm::qwen38
