// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <ATen/ATen.h>

namespace vllm::qwen38 {

void ngram_decode_ids(
    const at::Tensor& input_ids,
    const at::Tensor& context,
    const at::Tensor& multipliers,
    at::Tensor& output);

void ngram_decode_ids_eos(
    const at::Tensor& input_ids,
    const at::Tensor& context,
    const at::Tensor& multipliers,
    at::Tensor& output,
    int64_t eos_token_id);

at::Tensor ngram_host_lookup(
    const at::Tensor& weight,
    const at::Tensor& ids,
    at::Tensor output,
    int64_t vocab_start,
    int64_t vocab_end);

at::Tensor ngram_host_lookup_chunked(
    at::TensorList weights,
    const at::Tensor& ids,
    at::Tensor output,
    int64_t vocab_start,
    int64_t vocab_end);

}  // namespace vllm::qwen38
