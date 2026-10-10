// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <c10/core/Stream.h>
#include <torch/custom_class.h>
#include <torch/types.h>

#include <array>
#include <mutex>
#include <optional>
#include <tuple>
#include <unordered_map>

namespace vllm::qwen38::hc {

using HcRunResult = std::tuple<at::Tensor, at::Tensor, at::Tensor>;
using HcMixResult = std::tuple<at::Tensor, at::Tensor>;

struct HcScratch {
  at::Tensor combined;
  at::Tensor normed;
  at::Tensor down;
  at::Tensor mixed;
};

template <bool Multi>
class HcSyclWorkspace final : public torch::CustomClassHolder {
 public:
  std::optional<HcRunResult> try_run(
      at::Tensor hidden,
      at::Tensor block,
      at::Tensor injection,
      at::Tensor norm_weight,
      at::Tensor down_weight,
      at::Tensor up_weight,
      double eps);

  HcRunResult
  run(at::Tensor hidden,
      at::Tensor block,
      at::Tensor injection,
      at::Tensor norm_weight,
      at::Tensor down_weight,
      at::Tensor up_weight,
      double eps);

  // Optional extension: project an already normalized HC state. This is not
  // part of the original ESIMD workspace ABI and is never used by try_run.
  std::optional<HcMixResult>
  try_mix(at::Tensor normed, at::Tensor down_weight, at::Tensor up_weight);

 private:
  bool eligible_run(
      const at::Tensor& hidden,
      const at::Tensor& block,
      const at::Tensor& injection,
      const at::Tensor& norm_weight,
      const at::Tensor& down_weight,
      const at::Tensor& up_weight,
      double eps) const;

  HcRunResult run_impl(
      at::Tensor hidden,
      at::Tensor block,
      at::Tensor injection,
      at::Tensor norm_weight,
      at::Tensor down_weight,
      at::Tensor up_weight,
      double eps);

  std::mutex mutex_;
  std::unordered_map<c10::Stream, std::array<HcScratch, 9>> scratch_by_stream_;
};

using HcSyclM1Workspace = HcSyclWorkspace<false>;
using HcSyclMultiMWorkspace = HcSyclWorkspace<true>;

}  // namespace vllm::qwen38::hc
