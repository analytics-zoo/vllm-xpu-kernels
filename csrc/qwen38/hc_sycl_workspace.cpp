// SPDX-License-Identifier: Apache-2.0
#include "qwen38/hc_sycl_workspace.h"

#include "qwen38/hc_sycl.h"

#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <utility>

namespace vllm::qwen38::hc {
namespace {

bool compatible(
    const at::Tensor& tensor,
    const at::Device& device,
    at::IntArrayRef sizes,
    bool contiguous = true) {
  return tensor.defined() && tensor.device() == device &&
         tensor.layout() == at::kStrided && tensor.scalar_type() == at::kHalf &&
         tensor.sizes() == sizes && (!contiguous || tensor.is_contiguous()) &&
         !tensor.is_neg() && !tensor.is_conj() &&
         tensor.storage_offset() % 2 == 0 &&
         reinterpret_cast<std::uintptr_t>(tensor.const_data_ptr()) % 4 == 0;
}

HcScratch make_scratch(const at::Tensor& hidden) {
  const auto options = hidden.options().requires_grad(false);
  const int64_t rows = hidden.size(0);
  return {
      at::empty({rows, 10240}, options),
      at::empty({rows, 10240}, options),
      at::empty({rows, 336}, options),
      at::empty({rows, 2560}, options)};
}

bool scratch_aliases(
    const HcScratch& scratch, const std::array<at::Tensor, 6>& inputs) {
  const std::array<const at::Tensor*, 4> outputs = {
      &scratch.combined, &scratch.normed, &scratch.down, &scratch.mixed};
  for (size_t i = 0; i < outputs.size(); ++i) {
    for (const auto& input : inputs) {
      if (outputs[i]->is_alias_of(input)) return true;
    }
    for (size_t j = i + 1; j < outputs.size(); ++j) {
      if (outputs[i]->is_alias_of(*outputs[j])) return true;
    }
  }
  return false;
}

void check_cached_scratch(const HcScratch& scratch, const at::Tensor& hidden) {
  const std::array<const at::Tensor*, 4> outputs = {
      &scratch.combined, &scratch.normed, &scratch.down, &scratch.mixed};
  constexpr std::array<int64_t, 4> widths = {10240, 10240, 336, 2560};
  for (size_t i = 0; i < outputs.size(); ++i) {
    TORCH_CHECK(
        compatible(*outputs[i], hidden.device(), {hidden.size(0), widths[i]}),
        "HC SYCL workspace scratch metadata changed");
  }
}

void check_current_stream(const at::Tensor& hidden) {
  const auto stream = c10::xpu::getCurrentXPUStream(hidden.device().index());
  TORCH_CHECK(
      stream.queue().has_property<sycl::property::queue::in_order>(),
      "HC SYCL workspace requires an in-order current stream");
}

}  // namespace

template <bool Multi>
bool HcSyclWorkspace<Multi>::eligible_run(
    const at::Tensor& hidden,
    const at::Tensor& block,
    const at::Tensor& injection,
    const at::Tensor& norm_weight,
    const at::Tensor& down_weight,
    const at::Tensor& up_weight,
    double eps) const {
  if (!hidden.defined() || !hidden.device().is_xpu() || hidden.dim() != 2)
    return false;
  const int64_t rows = hidden.size(0);
  if constexpr (Multi) {
    if (rows < 2 || rows > 8) return false;
  } else if (rows != 1) {
    return false;
  }
  const auto device = hidden.device();
  const float eps_fp32 = static_cast<float>(eps);
  return compatible(hidden, device, {rows, 10240}) &&
         compatible(block, device, {rows, 2560}) &&
         compatible(injection, device, {rows, 4}, !Multi) &&
         injection.stride(1) == 1 && injection.stride(0) >= 4 &&
         compatible(norm_weight, device, {10240}) &&
         compatible(down_weight, device, {336, 10240}) &&
         compatible(up_weight, device, {10240, 320}) && std::isfinite(eps) &&
         eps > 0.0 && std::isfinite(eps_fp32) && eps_fp32 > 0.0f;
}

template <bool Multi>
std::optional<HcRunResult> HcSyclWorkspace<Multi>::try_run(
    at::Tensor hidden,
    at::Tensor block,
    at::Tensor injection,
    at::Tensor norm_weight,
    at::Tensor down_weight,
    at::Tensor up_weight,
    double eps) {
  // Unsupported live TensorImpl metadata is an availability miss before
  // allocation/submit. A failure in run_impl is never replayed in fallback.
  if (!eligible_run(
          hidden, block, injection, norm_weight, down_weight, up_weight, eps))
    return std::nullopt;
  return run_impl(
      std::move(hidden),
      std::move(block),
      std::move(injection),
      std::move(norm_weight),
      std::move(down_weight),
      std::move(up_weight),
      eps);
}

template <bool Multi>
HcRunResult HcSyclWorkspace<Multi>::run(
    at::Tensor hidden,
    at::Tensor block,
    at::Tensor injection,
    at::Tensor norm_weight,
    at::Tensor down_weight,
    at::Tensor up_weight,
    double eps) {
  TORCH_CHECK(
      eligible_run(
          hidden, block, injection, norm_weight, down_weight, up_weight, eps),
      "HC SYCL workspace inputs do not meet the final public ABI");
  return run_impl(
      std::move(hidden),
      std::move(block),
      std::move(injection),
      std::move(norm_weight),
      std::move(down_weight),
      std::move(up_weight),
      eps);
}

template <bool Multi>
HcRunResult HcSyclWorkspace<Multi>::run_impl(
    at::Tensor hidden,
    at::Tensor block,
    at::Tensor injection,
    at::Tensor norm_weight,
    at::Tensor down_weight,
    at::Tensor up_weight,
    double eps) {
  std::lock_guard<std::mutex> lock(mutex_);
  check_current_stream(hidden);
  const c10::Stream stream =
      c10::xpu::getCurrentXPUStream(hidden.device().index()).unwrap();
  auto& scratch = scratch_by_stream_[stream][hidden.size(0)];
  const std::array<at::Tensor, 6> inputs = {
      hidden, block, injection, norm_weight, down_weight, up_weight};
  if (!scratch.combined.defined() || scratch_aliases(scratch, inputs))
    scratch = make_scratch(hidden);
  check_cached_scratch(scratch, hidden);
  // The native entry validates every stage, output alias and live weight
  // before submitting the first kernel. No Python-side per-op preflight.
  combine_mix_prechecked_inputs(
      hidden,
      block,
      injection,
      norm_weight,
      down_weight,
      up_weight,
      scratch.combined,
      scratch.normed,
      scratch.down,
      scratch.mixed,
      eps);
  const auto next_injection = scratch.down.narrow(1, 320, 4);
  return {scratch.combined, scratch.mixed, next_injection};
}

template <bool Multi>
std::optional<HcMixResult> HcSyclWorkspace<Multi>::try_mix(
    at::Tensor normed, at::Tensor down_weight, at::Tensor up_weight) {
  if (!normed.defined() || !normed.device().is_xpu() || normed.dim() != 2)
    return std::nullopt;
  const int64_t rows = normed.size(0);
  if constexpr (Multi) {
    if (rows < 2 || rows > 8) return std::nullopt;
  } else if (rows != 1) {
    return std::nullopt;
  }
  const auto device = normed.device();
  if (!compatible(normed, device, {rows, 10240}) ||
      !compatible(down_weight, device, {336, 10240}) ||
      !compatible(up_weight, device, {10240, 320}))
    return std::nullopt;
  std::lock_guard<std::mutex> lock(mutex_);
  check_current_stream(normed);
  const c10::Stream stream =
      c10::xpu::getCurrentXPUStream(normed.device().index()).unwrap();
  auto& scratch = scratch_by_stream_[stream][rows];
  if (!scratch.combined.defined() ||
      scratch_aliases(
          scratch,
          {normed, down_weight, up_weight, normed, down_weight, up_weight}))
    scratch = make_scratch(normed);
  check_cached_scratch(scratch, normed);
  project_mix(normed, down_weight, up_weight, scratch.down, scratch.mixed);
  const auto next_injection = scratch.down.narrow(1, 320, 4);
  return HcMixResult{scratch.mixed, next_injection};
}

template class HcSyclWorkspace<false>;
template class HcSyclWorkspace<true>;

}  // namespace vllm::qwen38::hc

static auto register_hc_sycl_m1_workspace =
    torch::class_<vllm::qwen38::hc::HcSyclM1Workspace>(
        "qwen38_hc_sycl", "HCWorkspace")
        .def(torch::init<>())
        .def("try_run", &vllm::qwen38::hc::HcSyclM1Workspace::try_run)
        .def("run", &vllm::qwen38::hc::HcSyclM1Workspace::run)
        .def("try_mix", &vllm::qwen38::hc::HcSyclM1Workspace::try_mix);

static auto register_hc_sycl_multi_workspace =
    torch::class_<vllm::qwen38::hc::HcSyclMultiMWorkspace>(
        "qwen38_hc_sycl", "HCMultiMWorkspaceV1")
        .def(torch::init<>())
        .def("try_run", &vllm::qwen38::hc::HcSyclMultiMWorkspace::try_run)
        .def("run", &vllm::qwen38::hc::HcSyclMultiMWorkspace::run)
        .def("try_mix", &vllm::qwen38::hc::HcSyclMultiMWorkspace::try_mix);
