// SPDX-License-Identifier: Apache-2.0
#include "qwen38/moe_sycl_prefill_torch.h"

#include "qwen38/moe_sycl_prefill.h"

#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUStream.h>

#include <cstdint>
#include <limits>

namespace vllm::qwen38::moe_sycl {
namespace {

void validate(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& scale,
    const at::Tensor& counts) {
  TORCH_CHECK(
      x.is_xpu() && x.is_contiguous() && x.dim() == 2 &&
          x.scalar_type() == at::kHalf,
      "compact DOWN requires contiguous FP16 XPU activations");
  const auto device = x.device();
  for (const auto* t : {&x, &weight, &scale, &counts}) {
    TORCH_CHECK(
        t->device() == device && t->is_contiguous() && !t->is_neg() &&
            !t->is_conj(),
        "compact DOWN requires same-device contiguous non-lazy inputs");
  }
  const int64_t k = x.size(1);
  TORCH_CHECK(k == 80 || k == 160, "compact DOWN K must be 80 or 160");
  TORCH_CHECK(
      x.size(0) <= std::numeric_limits<int>::max() - 8192,
      "compact DOWN has too many rows");
  TORCH_CHECK(
      weight.scalar_type() == at::kChar && weight.dim() == 3 &&
          weight.size(0) == 512 && weight.size(1) == 2560 &&
          weight.size(2) == k / 2,
      "compact DOWN signed-S4 weight shape must be [512,2560,K/2]");
  TORCH_CHECK(
      scale.scalar_type() == at::kHalf && scale.dim() == 3 &&
          scale.size(0) == 512 && scale.size(1) == 2560 &&
          scale.size(2) == (k + 127) / 128,
      "compact DOWN scale shape must be [512,2560,ceil(K/128)]");
  TORCH_CHECK(
      counts.scalar_type() == at::kInt && counts.dim() == 1 &&
          counts.numel() == 512,
      "compact DOWN rows_per_expert must be [512] int32 row counts");
  TORCH_CHECK(
      (reinterpret_cast<uintptr_t>(weight.const_data_ptr()) & 3u) == 0,
      "compact DOWN weight must be dword aligned");
  TORCH_CHECK(
      (reinterpret_cast<uintptr_t>(x.const_data_ptr()) & 63u) == 0,
      "compact DOWN 2D activation load needs 64-byte alignment");
}

void record(
    c10::xpu::XPUStream stream,
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& scale,
    const at::Tensor& counts,
    const at::Tensor& output,
    const at::Tensor& prefixes) {
  for (const auto* t : {&x, &weight, &scale, &counts, &output, &prefixes}) {
    c10::xpu::XPUCachingAllocator::recordStream(
        t->storage().data_ptr(), stream);
  }
}

}  // namespace

at::Tensor compact_down_grouped_gemm(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& scale,
    const at::Tensor& rows_per_expert) {
  validate(x, weight, scale, rows_per_expert);
  const c10::OptionalDeviceGuard guard(x.device());
  auto stream = c10::xpu::getCurrentXPUStream(x.device().index());
  auto& queue = stream.queue();
  TORCH_CHECK(queue.is_in_order(), "compact DOWN needs an in-order XPU queue");
  auto output = at::empty({x.size(0), 2560}, x.options());
  auto prefixes = at::empty({2, 513}, rows_per_expert.options());
  const PrefillDown args{
      reinterpret_cast<const sycl::half*>(x.const_data_ptr()),
      reinterpret_cast<const uint8_t*>(weight.const_data_ptr()),
      reinterpret_cast<const sycl::half*>(scale.const_data_ptr()),
      rows_per_expert.data_ptr<int32_t>(),
      reinterpret_cast<sycl::half*>(output.data_ptr()),
      prefixes.data_ptr<int32_t>(),
      prefixes.data_ptr<int32_t>() + 513,
      static_cast<int>(x.size(0)),
      static_cast<int>(x.size(1))};
  // Record before either submit: if the second submit throws, the first
  // kernel can still be running while this stack unwinds.
  record(stream, x, weight, scale, rows_per_expert, output, prefixes);
  TORCH_CHECK(
      try_prefill_down(queue, args),
      "compact DOWN raw SYCL contract was rejected");
  return output;
}

}  // namespace vllm::qwen38::moe_sycl
