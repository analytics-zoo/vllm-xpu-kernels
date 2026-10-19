// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <torch/types.h>
#include <optional>

namespace vllm::qwen38 {
// Private TP8 M1 transport ABI. Packets are FP32 [1,64]; global input is eight
// such packets in rank order. Indices must be exact integers below 2^24.
// Eligibility misses return nullopt before submission. Python checks tensor
// subclasses and Torch modes. Every result owns independent output storage.
std::optional<at::Tensor> try_local_argmax_packet(
    const at::Tensor& logits, int64_t vocab_start, int64_t num_padding);
std::optional<at::Tensor> try_global_argmax_token(const at::Tensor& packets);
std::optional<at::Tensor>
try_global_argmax_token_int32(const at::Tensor& packets);
}  // namespace vllm::qwen38
