#include "moe_ops.h"
#include "core/registration.h"

#include <ATen/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>

#include <cstdint>
#include <limits>

namespace {

constexpr int kExperts = 128;
constexpr int kTopK = 8;

template <typename Input>
struct Gemma4SmallMTopK {
  const Input* logits;
  const void* expert_scales;
  bool scale_is_half;
  float* weights;
  int32_t* ids;
  sycl::local_accessor<int64_t, 1> ordered_keys;
  sycl::local_accessor<float, 1> selected_exp;

  [[sycl::reqd_sub_group_size(16)]] void
  operator()(sycl::nd_item<1> item) const {
    const int expert = item.get_local_linear_id();
    const int token = item.get_group_linear_id();
    const float value = static_cast<float>(logits[token * kExperts + expert]);

    // The model's Triton router sorts this same signed 64-bit key. It orders
    // logits descending and resolves exact ties by ascending expert ID;
    // the float-bit ordering also preserves its +0/-0 behavior.
    const uint32_t bits = sycl::bit_cast<uint32_t>(value);
    const uint32_t key =
        (bits & 0x80000000u) ? (bits ^ 0x80000000u) : (bits ^ 0xffffffffu);
    const int64_t ordered = sycl::bit_cast<int64_t>(
        (static_cast<uint64_t>(key) << 32) | static_cast<uint64_t>(expert));
    ordered_keys[expert] = ordered;
    // Match Triton's maximum: NaN logits do not poison finite lanes.
    // Preserve the original value for sorting and the selected exponential.
    const float maximum = sycl::reduce_over_group(
        item.get_group(),
        sycl::isnan(value) ? -std::numeric_limits<float>::infinity() : value,
        sycl::maximum<float>());
    item.barrier(sycl::access::fence_space::local_space);

    int rank = 0;
    for (int other = 0; other < kExperts; ++other) {
      rank += ordered_keys[other] < ordered;
    }
    if (rank < kTopK) {
      selected_exp[rank] = sycl::exp2((value - maximum) * 1.4426950408889634f);
    }
    item.barrier(sycl::access::fence_space::local_space);

    if (rank < kTopK) {
      float sum = 0.0f;
#pragma unroll
      for (int selected = 0; selected < kTopK; ++selected) {
        sum += selected_exp[selected];
      }
      sum = sum > 0.0f ? sum : 1.0f;
      const float scale =
          !expert_scales
              ? 1.0f
              : (scale_is_half
                     ? static_cast<float>(static_cast<const sycl::half*>(
                           expert_scales)[expert])
                     : static_cast<const float*>(expert_scales)[expert]);
      weights[token * kTopK + rank] = selected_exp[rank] * (1.0f / sum) * scale;
      ids[token * kTopK + rank] = expert;
    }
  }
};

template <typename Input>
void launch_topk(
    const torch::Tensor& logits,
    const std::optional<torch::Tensor>& expert_scales,
    torch::Tensor& weights,
    torch::Tensor& ids) {
  auto& queue = c10::xpu::getCurrentXPUStream(logits.get_device()).queue();
  const size_t tokens = logits.size(0);
  queue.submit([&](sycl::handler& handler) {
    sycl::local_accessor<int64_t, 1> keys(sycl::range<1>(kExperts), handler);
    sycl::local_accessor<float, 1> selected_exp(sycl::range<1>(kTopK), handler);
    handler.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(tokens * kExperts), sycl::range<1>(kExperts)),
        Gemma4SmallMTopK<Input>{
            reinterpret_cast<const Input*>(logits.data_ptr()),
            expert_scales ? expert_scales->data_ptr() : nullptr,
            expert_scales && expert_scales->scalar_type() == at::kHalf,
            weights.data_ptr<float>(),
            ids.data_ptr<int32_t>(),
            keys,
            selected_exp});
  });
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> gemma4_small_m_topk(
    const torch::Tensor& logits,
    const std::optional<torch::Tensor>& per_expert_scale,
    int64_t topk) {
  TORCH_CHECK(
      logits.is_xpu() && logits.dim() == 2 && logits.is_contiguous(),
      "gemma4_small_m_topk requires contiguous 2D XPU logits");
  TORCH_CHECK(
      logits.size(0) >= 1 && logits.size(0) <= 8 &&
          logits.size(1) == kExperts && topk == kTopK,
      "gemma4_small_m_topk supports M=1..8, E=128 and topk=8");
  TORCH_CHECK(
      logits.scalar_type() == at::kHalf || logits.scalar_type() == at::kFloat,
      "gemma4_small_m_topk requires FP16 or FP32 logits");
  if (per_expert_scale) {
    TORCH_CHECK(
        per_expert_scale->device() == logits.device() &&
            (per_expert_scale->scalar_type() == at::kFloat ||
             per_expert_scale->scalar_type() == at::kHalf) &&
            per_expert_scale->dim() == 1 &&
            per_expert_scale->numel() == kExperts &&
            per_expert_scale->is_contiguous(),
        "per_expert_scale must be contiguous XPU FP16/FP32 [128] on the logits "
        "device");
  }
  const at::DeviceGuard guard(logits.device());
  auto weights =
      at::empty({logits.size(0), topk}, logits.options().dtype(at::kFloat));
  auto ids =
      at::empty({logits.size(0), topk}, logits.options().dtype(at::kInt));
  if (logits.scalar_type() == at::kHalf) {
    launch_topk<sycl::half>(logits, per_expert_scale, weights, ids);
  } else {
    launch_topk<float>(logits, per_expert_scale, weights, ids);
  }
  return {weights, ids};
}

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, Meta, m) {
  m.impl(
      "gemma4_small_m_topk",
      [](const torch::Tensor& logits,
         const std::optional<torch::Tensor>& per_expert_scale,
         int64_t topk) {
        return std::make_tuple(
            at::empty_symint(
                {logits.sym_size(0), c10::SymInt(topk)},
                logits.options().dtype(at::kFloat)),
            at::empty_symint(
                {logits.sym_size(0), c10::SymInt(topk)},
                logits.options().dtype(at::kInt)));
      });
}
