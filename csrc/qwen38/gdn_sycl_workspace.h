// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <c10/core/Stream.h>
#include <torch/types.h>

#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

#include "qwen38/tensor_binding.h"

namespace vllm::qwen38 {

// One host transaction, not one device kernel. Reuses the existing three
// projection/core/projection implementations without changing their math.
class GdnSyclM1Workspace final {
 public:
  // weights: qkv W/S, ba W/S, out W/S, conv W, A_log, dt_bias, norm W.
  // Python guards exact tensor types/modes and layer/metadata semantics.
  std::optional<at::Tensor> try_run(
      at::Tensor input,
      const std::vector<at::Tensor>& weights,
      const std::optional<at::Tensor>& conv_bias,
      at::Tensor conv_state,
      at::Tensor ssm_state,
      at::Tensor indices,
      int64_t local_v_heads,
      double eps,
      bool sigmoid_gate);

  // Required by the model's explicit inference-weight reload hook.
  void invalidate();

 private:
  friend class GdnSyclSpecWorkspace;
  std::optional<at::Tensor> try_run_impl(
      at::Tensor input,
      const std::vector<at::Tensor>& weights,
      const std::optional<at::Tensor>& conv_bias,
      at::Tensor conv_state,
      at::Tensor ssm_state,
      at::Tensor indices,
      const std::optional<at::Tensor>& token_indices,
      const std::optional<at::Tensor>& accepted,
      int64_t local_v_heads,
      double eps,
      bool sigmoid_gate);
  struct Scratch {
    int64_t hv = 0, m = 0;
    at::Tensor qkvz, ba, core, z, core_rows, z_rows, normalized, output;
    at::Tensor conv_weight, conv_bias, a_log, dt_bias, norm_weight;
    std::unique_ptr<TensorBindingSnapshotV1> binding;
  };
  std::mutex mutex_;
  std::unordered_map<c10::Stream, Scratch> by_stream_;
};

// MTP verify: one speculative request, M=2..8. The state update, FP32 A_log
// and separate batched INT4 stages are identical to the existing spec path.
class GdnSyclSpecWorkspace final {
 public:
  std::optional<at::Tensor> try_run(
      at::Tensor input,
      const std::vector<at::Tensor>& weights,
      const std::optional<at::Tensor>& conv_bias,
      at::Tensor conv_state,
      at::Tensor ssm_state,
      at::Tensor indices,
      at::Tensor token_indices,
      at::Tensor accepted,
      int64_t local_v_heads,
      double eps,
      bool sigmoid_gate);
  void invalidate() { workspace_.invalidate(); }

 private:
  GdnSyclM1Workspace workspace_;
};

}  // namespace vllm::qwen38
