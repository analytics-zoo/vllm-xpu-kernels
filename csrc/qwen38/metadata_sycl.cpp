// SPDX-License-Identifier: Apache-2.0
#include "qwen38/metadata_sycl.h"

#include <ATen/MemoryOverlap.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>

#include <array>
#include <cstdint>

namespace vllm::qwen38 {
namespace {
constexpr int kMaxFields = 32;
constexpr int kWorkGroup = 128;
constexpr int64_t kMaxFieldBytes = 1024 * 1024;
struct Copy {
  const char* source;
  char* output;
  int64_t bytes;
};
struct Plan {
  std::array<Copy, kMaxFields> fields;
};
class MetadataCopyKernel;
}  // namespace

bool try_copy_metadata_out(
    const std::vector<at::Tensor>& sources,
    const std::vector<at::Tensor>& outputs) {
  if (sources.empty() || sources.size() > kMaxFields ||
      sources.size() != outputs.size() || !sources.front().defined() ||
      !sources.front().is_xpu()) {
    return false;
  }
  const auto device = sources.front().device();
  Plan plan{};
  for (size_t i = 0; i < sources.size(); ++i) {
    const auto& source = sources[i];
    const auto& output = outputs[i];
    if (!source.defined() || !output.defined() ||
        source.layout() != at::kStrided || output.layout() != at::kStrided ||
        source.is_nested() || output.is_nested() || source.requires_grad() ||
        output.requires_grad() || source.device() != device ||
        output.device() != device ||
        source.scalar_type() != output.scalar_type() ||
        source.sizes() != output.sizes() || !source.is_contiguous() ||
        !output.is_contiguous() || source.is_neg() || source.is_conj() ||
        output.is_neg() || output.is_conj() ||
        source.numel() > kMaxFieldBytes / source.element_size()) {
      return false;
    }
    plan.fields[i] = {
        static_cast<const char*>(source.const_data_ptr()),
        static_cast<char*>(output.data_ptr()),
        source.numel() * static_cast<int64_t>(source.element_size())};
  }
  for (size_t i = 0; i < outputs.size(); ++i) {
    for (const auto& source : sources) {
      if (at::get_overlap_status(outputs[i], source) !=
          at::MemOverlapStatus::No) {
        return false;
      }
    }
    for (size_t j = 0; j < i; ++j) {
      if (at::get_overlap_status(outputs[i], outputs[j]) !=
          at::MemOverlapStatus::No) {
        return false;
      }
    }
  }
  const c10::DeviceGuard guard(device);
  const auto stream = c10::xpu::getCurrentXPUStream(device.index());
  auto& queue = stream.queue();
  if (!queue.is_in_order()) return false;
  for (const auto& tensor : sources) {
    c10::xpu::XPUCachingAllocator::recordStream(
        tensor.storage().data_ptr(), stream);
  }
  for (const auto& tensor : outputs) {
    c10::xpu::XPUCachingAllocator::recordStream(
        tensor.storage().data_ptr(), stream);
  }
  const size_t groups = sources.size();
  queue.parallel_for<MetadataCopyKernel>(
      sycl::nd_range<1>(groups * kWorkGroup, kWorkGroup),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
        const Copy field = plan.fields[item.get_group_linear_id()];
        const int lane = item.get_local_linear_id();
        if (reinterpret_cast<uintptr_t>(field.source) % 16 == 0 &&
            reinterpret_cast<uintptr_t>(field.output) % 16 == 0) {
          const int64_t full = field.bytes / 16;
          for (int64_t i = lane; i < full; i += kWorkGroup) {
            *reinterpret_cast<sycl::vec<uint32_t, 4>*>(field.output + i * 16) =
                *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(
                    field.source + i * 16);
          }
          for (int64_t i = full * 16 + lane; i < field.bytes; i += kWorkGroup) {
            field.output[i] = field.source[i];
          }
        } else {
          for (int64_t i = lane; i < field.bytes; i += kWorkGroup) {
            field.output[i] = field.source[i];
          }
        }
      });
  return true;
}
}  // namespace vllm::qwen38
