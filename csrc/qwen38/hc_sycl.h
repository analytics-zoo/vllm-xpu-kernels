// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/types.h>

namespace vllm::qwen38::hc {

// Fixed Qwen3.8 TP4 shapes: H=2560, HC=4, low rank=320, M=1..8.
// Every output is caller-owned. All functions validate before enqueueing on
// the current XPU stream; none allocates device scratch or synchronizes.
void grouped_norm(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    torch::Tensor& output,
    double eps);
void gate_mix(
    const torch::Tensor& input,
    const torch::Tensor& gate,
    torch::Tensor& output);
void combine(
    const torch::Tensor& hidden,
    const torch::Tensor& block,
    const torch::Tensor& injection,
    torch::Tensor& output);
void combine_norm(
    const torch::Tensor& hidden,
    const torch::Tensor& block,
    const torch::Tensor& injection,
    const torch::Tensor& weight,
    torch::Tensor& combined,
    torch::Tensor& normed,
    double eps);

// 独立 prefill 元素算子：M=9..4096，FP16 HC4/H2560，不调用投影 GEMV。
// 输出均由 caller 持有；保留 decode 算子的舍入、别名与 stream 契约。
void prefill_grouped_norm(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    torch::Tensor& output,
    double eps);
void prefill_gate_mix(
    const torch::Tensor& input,
    const torch::Tensor& gate,
    torch::Tensor& output);
void prefill_combine(
    const torch::Tensor& hidden,
    const torch::Tensor& block,
    const torch::Tensor& injection,
    torch::Tensor& output);
void prefill_combine_norm(
    const torch::Tensor& hidden,
    const torch::Tensor& block,
    const torch::Tensor& injection,
    const torch::Tensor& weight,
    torch::Tensor& combined,
    torch::Tensor& normed,
    double eps);

void down(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    torch::Tensor& output);
void up(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    torch::Tensor& output);
void up_gate_mix(
    const torch::Tensor& lowrank,
    const torch::Tensor& weight,
    const torch::Tensor& normed,
    torch::Tensor& output);
void combine_mix(
    const torch::Tensor& hidden,
    const torch::Tensor& block,
    const torch::Tensor& injection,
    const torch::Tensor& norm_weight,
    const torch::Tensor& down_weight,
    const torch::Tensor& up_weight,
    torch::Tensor& combined,
    torch::Tensor& normed,
    torch::Tensor& down_output,
    torch::Tensor& mixed,
    double eps);

// Workspace-only: caller has checked every live input's metadata and eps.
// This still validates all owned outputs and all aliases before any submit.
void combine_mix_prechecked_inputs(
    const torch::Tensor& hidden,
    const torch::Tensor& block,
    const torch::Tensor& injection,
    const torch::Tensor& norm_weight,
    const torch::Tensor& down_weight,
    const torch::Tensor& up_weight,
    torch::Tensor& combined,
    torch::Tensor& normed,
    torch::Tensor& down_output,
    torch::Tensor& mixed,
    double eps);

// Validates both projection stages and all aliases before the down submit.
void project_mix(
    const torch::Tensor& normed,
    const torch::Tensor& down_weight,
    const torch::Tensor& up_weight,
    torch::Tensor& down_output,
    torch::Tensor& mixed);

}  // namespace vllm::qwen38::hc
