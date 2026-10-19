// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/types.h>
#include <vector>

namespace vllm::qwen38 {
// Eligibility misses return false before any submission or output mutation.
// The Python caller must check tensor types and Torch modes before direct use.
bool try_copy_metadata_out(
    const std::vector<at::Tensor>& sources,
    const std::vector<at::Tensor>& outputs);
}  // namespace vllm::qwen38
