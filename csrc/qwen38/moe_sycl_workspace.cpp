// SPDX-License-Identifier: Apache-2.0
#include "qwen38/moe_sycl_workspace.h"

#include "qwen38/moe_sycl.h"

#include <ATen/MemoryOverlap.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUStream.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <utility>

namespace vllm::qwen38::moe_sycl {
namespace {

constexpr int kHidden = 2560;
constexpr int kExperts = 512;
constexpr int kTopK = 10;

struct ByteRange {
  uintptr_t begin, end;
};

// Independently wrapped tensors may share an allocation without sharing a
// StorageImpl. Use a conservative strided address span as well as ATen's check.
ByteRange byte_range(const at::Tensor& t) {
  TORCH_CHECK(
      t.layout() == at::kStrided, "MoE alias check requires strided tensors");
  const auto begin = reinterpret_cast<uintptr_t>(t.const_data_ptr());
  if (t.numel() == 0) return {begin, begin};
  constexpr auto max = std::numeric_limits<uintptr_t>::max();
  uintptr_t last = 0;
  for (int64_t d = 0; d < t.dim(); ++d) {
    TORCH_CHECK(
        t.stride(d) >= 0, "MoE alias check requires nonnegative strides");
    const auto extent = static_cast<uintptr_t>(t.size(d) - 1);
    const auto stride = static_cast<uintptr_t>(t.stride(d));
    TORCH_CHECK(
        stride == 0 || extent <= max / stride, "MoE address range overflows");
    const auto contribution = extent * stride;
    TORCH_CHECK(last <= max - contribution, "MoE address range overflows");
    last += contribution;
  }
  const auto item_bytes = static_cast<uintptr_t>(t.element_size());
  TORCH_CHECK(
      item_bytes > 0 && last < max / item_bytes, "MoE byte range overflows");
  const auto bytes = (last + 1) * item_bytes;
  TORCH_CHECK(begin <= max - bytes, "MoE byte range overflows");
  return {begin, begin + bytes};
}

bool compatible(
    const at::Tensor& t,
    c10::Device device,
    at::ScalarType dtype,
    at::IntArrayRef shape,
    uintptr_t alignment = 16) {
  return t.defined() && t.device() == device && t.layout() == at::kStrided &&
         t.scalar_type() == dtype && t.sizes() == shape && t.is_contiguous() &&
         !t.is_neg() && !t.is_conj() &&
         (reinterpret_cast<uintptr_t>(t.const_data_ptr()) & (alignment - 1)) ==
             0;
}

bool weight_dtype(at::ScalarType dtype) {
  // Quantized routed weights are signed S4 physical bytes. A byte view is
  // equally valid: the device decodes the low/high two's-complement nibbles.
  return dtype == at::kByte || dtype == at::kChar;
}

bool is_router_meta(
    const at::Tensor& x,
    const at::Tensor& router_or_logits,
    const at::Tensor& router_scale,
    int rows) {
  const auto device = x.device();
  if (router_scale.defined()) {
    return compatible(router_or_logits, device, at::kByte, {512, 1280}) &&
           compatible(router_scale, device, at::kHalf, {512, 20});
  }
  return compatible(router_or_logits, device, at::kHalf, {rows, 512});
}

bool overlaps_any(
    const at::Tensor& output,
    const at::Tensor& x,
    const at::Tensor& router_or_logits,
    const at::Tensor& router_scale,
    const std::vector<at::Tensor>& weights) {
  if (output_overlaps(output, {x, router_or_logits, router_scale})) return true;
  return output_overlaps(output, weights);
}

bool reusable_output(
    const at::Tensor& output,
    const at::Tensor& x,
    const at::Tensor& router_or_logits,
    const at::Tensor& router_scale,
    const std::vector<at::Tensor>& weights,
    int rows) {
  // A distinct TensorImpl may still be a view of this slot's storage.
  // storage() returns const Storage&, so this check does not create a
  // temporary owner that would change the count being tested.
  return output.defined() && output.has_storage() && output.use_count() == 1 &&
         output.storage().use_count() == 1 &&
         compatible(output, x.device(), at::kHalf, {rows, kHidden}) &&
         !overlaps_any(output, x, router_or_logits, router_scale, weights);
}

void record_tensor(const at::Tensor& t, c10::xpu::XPUStream stream) {
  if (t.defined())
    c10::xpu::XPUCachingAllocator::recordStream(t.storage().data_ptr(), stream);
}

void record_inputs(
    c10::xpu::XPUStream stream,
    const at::Tensor& x,
    const at::Tensor& router_or_logits,
    const at::Tensor& router_scale,
    const std::vector<at::Tensor>& weights) {
  record_tensor(x, stream);
  record_tensor(router_or_logits, stream);
  record_tensor(router_scale, stream);
  for (const auto& weight : weights)
    record_tensor(weight, stream);
}

void check_queue(const sycl::queue& queue, const at::Tensor& x) {
  TORCH_CHECK(
      queue.is_in_order() &&
          queue.get_context() == c10::xpu::get_device_context() &&
          queue.get_device() == c10::xpu::get_raw_device(x.device().index()),
      "Qwen3.8 SYCL MoE needs the current in-order XPU queue/device/context");
}

}  // namespace

int64_t compact_weight_contract(
    at::TensorList weights, c10::Device device, int intermediate) {
  TORCH_CHECK(weights.size() == 7, "compact MoE expects seven weights");
  TORCH_CHECK(
      intermediate == 80 || intermediate == 160,
      "compact MoE requires physical I=80 or 160");
  const std::array<std::array<int64_t, 3>, 7> shapes{{
      {kExperts, 2 * intermediate, kHidden / 2},
      {kExperts, 2 * intermediate, kHidden / 128},
      {kExperts, kHidden, intermediate / 2},
      {kExperts, kHidden, (intermediate + 127) / 128},
      {2 * intermediate, kHidden, 0},
      {kHidden, intermediate, 0},
      {1, kHidden, 0},
  }};
  for (size_t i = 0; i < weights.size(); ++i) {
    const auto& t = weights[i];
    if (!t.defined() || t.device() != device) return 1;
    if (i == 0 || i == 2) {
      if (!weight_dtype(t.scalar_type())) return 2;
    } else if (t.scalar_type() != at::kHalf) {
      return 2;
    }
    const int dims = i < 4 ? 3 : 2;
    if (t.dim() != dims) return 3;
    for (int d = 0; d < dims; ++d)
      if (t.size(d) != shapes[i][d]) return 3;
    if (t.layout() != at::kStrided || !t.is_contiguous() || t.is_neg() ||
        t.is_conj() ||
        (reinterpret_cast<uintptr_t>(t.const_data_ptr()) & 15u) != 0)
      return 4;
  }
  return 0;
}

bool output_overlaps(const at::Tensor& output, at::TensorList inputs) {
  if (!output.defined() || output.numel() == 0) return false;
  const auto output_range = byte_range(output);
  for (const auto& input : inputs) {
    if (!input.defined() || input.device() != output.device() ||
        input.numel() == 0)
      continue;
    if (at::get_overlap_status(output, input) != at::MemOverlapStatus::No)
      return true;
    const auto input_range = byte_range(input);
    if (output_range.begin < input_range.end &&
        input_range.begin < output_range.end)
      return true;
  }
  return false;
}

std::optional<at::Tensor> MoeSyclWorkspaceCore::try_run(
    const at::Tensor& x,
    const at::Tensor& router,
    const at::Tensor& router_scale,
    std::vector<at::Tensor> weights,
    int64_t intermediate,
    bool grouped) {
  (void)grouped;  // Device-side grouped UP choice is compile-time in Gibbs.
  if (!x.defined() || !x.is_xpu() || x.dim() != 2 || x.size(0) < 1 ||
      x.size(0) > 8 || (intermediate != 80 && intermediate != 160) ||
      weights.size() != 7)
    return std::nullopt;
  const int rows = int(x.size(0));
  const int width = int(intermediate);
  const auto device = x.device();
  if (!compatible(x, device, at::kHalf, {rows, kHidden}) ||
      !is_router_meta(x, router, router_scale, rows) ||
      !router_scale.defined() ||
      compact_weight_contract(weights, device, width) != 0)
    return std::nullopt;

  const c10::OptionalDeviceGuard guard(device);
  const auto stream = c10::xpu::getCurrentXPUStream(device.index());
  auto& queue = stream.queue();
  check_queue(queue, x);
  std::lock_guard<std::mutex> lock(mutex_);
  auto& scratch = buffers_[stream.unwrap()][width == 160][rows];
  const auto half = x.options().requires_grad(false);
  const auto fresh = [&] {
    scratch.logits = at::empty({rows, kExperts}, half);
    scratch.ids = at::empty({rows, kTopK}, half.dtype(at::kInt));
    scratch.weights = at::empty({rows, kTopK}, half);
    scratch.routed = at::empty({rows * kTopK, width}, half);
    scratch.shared = at::empty({rows, width}, half);
    scratch.gates = at::empty({rows}, half.dtype(at::kFloat));
  };
  const auto scratch_valid = [&] {
    return compatible(scratch.logits, device, at::kHalf, {rows, kExperts}) &&
           compatible(scratch.ids, device, at::kInt, {rows, kTopK}) &&
           compatible(scratch.weights, device, at::kHalf, {rows, kTopK}) &&
           compatible(
               scratch.routed, device, at::kHalf, {rows * kTopK, width}) &&
           compatible(scratch.shared, device, at::kHalf, {rows, width}) &&
           compatible(scratch.gates, device, at::kFloat, {rows});
  };
  if (!scratch_valid() ||
      overlaps_any(scratch.logits, x, router, router_scale, weights) ||
      overlaps_any(scratch.ids, x, router, router_scale, weights) ||
      overlaps_any(scratch.weights, x, router, router_scale, weights) ||
      overlaps_any(scratch.routed, x, router, router_scale, weights) ||
      overlaps_any(scratch.shared, x, router, router_scale, weights) ||
      overlaps_any(scratch.gates, x, router, router_scale, weights))
    fresh();
  at::Tensor* selected = nullptr;
  for (auto& slot : scratch.outputs) {
    if (reusable_output(slot, x, router, router_scale, weights, rows)) {
      selected = &slot;
      break;
    }
  }
  if (selected == nullptr) {
    for (auto& slot : scratch.outputs) {
      // A slot with no outside TensorImpl/storage owner may be repaired if
      // its metadata was changed by a previous caller. Never evict a view.
      const bool replaceable =
          !slot.defined() ||
          (slot.use_count() == 1 &&
           (!slot.has_storage() || slot.storage().use_count() == 1) &&
           (!slot.has_storage() ||
            !compatible(slot, device, at::kHalf, {rows, kHidden})));
      if (replaceable) {
        slot = at::empty({rows, kHidden}, half);
        selected = &slot;
        break;
      }
    }
  }
  // Both bounded slots are still visible to callers (or their storage views),
  // or overlap live inputs. The overflow output is never installed in a slot.
  at::Tensor overflow;
  if (selected == nullptr) {
    overflow = at::empty({rows, kHidden}, half);
    selected = &overflow;
  }
  // Never expose the cached TensorImpl to Python. Torch can preserve its
  // Python wrapper while a C++ cache owns it, leaving use_count()==2 even
  // after the caller releases the result. A detached metadata copy owns the
  // same StorageImpl but no reference to the private cache's TensorImpl.
  // Thus a live result/view keeps storage.use_count()>1, while releasing it
  // permits reuse without probing Python reference counts or the GIL.
  auto* selected_impl = selected->unsafeGetTensorImpl();
  at::Tensor output{selected_impl->shallow_copy_and_detach(
      selected_impl->version_counter(), /*allow_tensor_metadata_change=*/true)};

  // Keep every allocation alive even if a later stage throws after enqueue.
  record_inputs(stream, x, router, router_scale, weights);
  for (const auto* t :
       {&scratch.logits,
        &scratch.ids,
        &scratch.weights,
        &scratch.routed,
        &scratch.shared,
        &scratch.gates,
        &output})
    record_tensor(*t, stream);
  const Inputs input{
      reinterpret_cast<const sycl::half*>(x.const_data_ptr()),
      nullptr,
      reinterpret_cast<const uint8_t*>(router.const_data_ptr()),
      reinterpret_cast<const sycl::half*>(router_scale.const_data_ptr()),
      reinterpret_cast<const uint8_t*>(weights[0].const_data_ptr()),
      reinterpret_cast<const sycl::half*>(weights[1].const_data_ptr()),
      reinterpret_cast<const uint8_t*>(weights[2].const_data_ptr()),
      reinterpret_cast<const sycl::half*>(weights[3].const_data_ptr()),
      reinterpret_cast<const sycl::half*>(weights[4].const_data_ptr()),
      reinterpret_cast<const sycl::half*>(weights[5].const_data_ptr()),
      reinterpret_cast<const sycl::half*>(weights[6].const_data_ptr()),
      reinterpret_cast<sycl::half*>(output.data_ptr()),
      rows,
      width};
  const Workspace workspace{
      reinterpret_cast<sycl::half*>(scratch.logits.data_ptr()),
      scratch.ids.data_ptr<int32_t>(),
      reinterpret_cast<sycl::half*>(scratch.weights.data_ptr()),
      reinterpret_cast<sycl::half*>(scratch.routed.data_ptr()),
      reinterpret_cast<sycl::half*>(scratch.shared.data_ptr()),
      scratch.gates.data_ptr<float>()};
  if (!try_forward(queue, input, workspace)) return std::nullopt;
  return output;
}

at::Tensor MoeSyclWorkspaceCore::run_out(
    const at::Tensor& x,
    const at::Tensor& logits_or_router,
    const at::Tensor& router_scale,
    std::vector<at::Tensor> weights,
    const at::Tensor& output,
    int64_t intermediate,
    bool grouped) {
  (void)grouped;
  TORCH_CHECK(
      x.defined() && x.is_xpu() && x.dim() == 2 && x.size(0) >= 1 &&
          x.size(0) <= std::numeric_limits<int>::max() / kHidden &&
          (intermediate == 80 || intermediate == 160) && weights.size() == 7,
      "unsupported Qwen3.8 compact MoE geometry");
  const int rows = int(x.size(0));
  const int width = int(intermediate);
  const auto device = x.device();
  TORCH_CHECK(
      compatible(x, device, at::kHalf, {rows, kHidden}) &&
          is_router_meta(x, logits_or_router, router_scale, rows) &&
          compatible(output, device, at::kHalf, {rows, kHidden}) &&
          compact_weight_contract(weights, device, width) == 0 &&
          !overlaps_any(output, x, logits_or_router, router_scale, weights),
      "Qwen3.8 compact MoE input/output contract failed");

  const c10::OptionalDeviceGuard guard(device);
  const auto stream = c10::xpu::getCurrentXPUStream(device.index());
  auto& queue = stream.queue();
  check_queue(queue, x);
  const int capacity = std::min(rows, 8);
  std::lock_guard<std::mutex> lock(mutex_);
  auto& scratch = buffers_[stream.unwrap()][width == 160][capacity];
  const auto half = x.options().requires_grad(false);
  const auto fresh = [&] {
    scratch.logits = at::empty({capacity, kExperts}, half);
    scratch.ids = at::empty({capacity, kTopK}, half.dtype(at::kInt));
    scratch.weights = at::empty({capacity, kTopK}, half);
    scratch.routed = at::empty({capacity * kTopK, width}, half);
    scratch.shared = at::empty({capacity, width}, half);
    scratch.gates = at::empty({capacity}, half.dtype(at::kFloat));
  };
  if (!compatible(scratch.logits, device, at::kHalf, {capacity, kExperts}) ||
      !compatible(scratch.ids, device, at::kInt, {capacity, kTopK}) ||
      !compatible(scratch.weights, device, at::kHalf, {capacity, kTopK}) ||
      !compatible(
          scratch.routed, device, at::kHalf, {capacity * kTopK, width}) ||
      !compatible(scratch.shared, device, at::kHalf, {capacity, width}) ||
      !compatible(scratch.gates, device, at::kFloat, {capacity}) ||
      overlaps_any(
          scratch.logits, x, logits_or_router, router_scale, weights) ||
      overlaps_any(scratch.ids, x, logits_or_router, router_scale, weights) ||
      overlaps_any(
          scratch.weights, x, logits_or_router, router_scale, weights) ||
      overlaps_any(
          scratch.routed, x, logits_or_router, router_scale, weights) ||
      overlaps_any(
          scratch.shared, x, logits_or_router, router_scale, weights) ||
      overlaps_any(scratch.gates, x, logits_or_router, router_scale, weights) ||
      output_overlaps(
          output,
          {scratch.logits,
           scratch.ids,
           scratch.weights,
           scratch.routed,
           scratch.shared,
           scratch.gates}))
    fresh();
  TORCH_CHECK(
      !output_overlaps(
          output,
          {scratch.logits,
           scratch.ids,
           scratch.weights,
           scratch.routed,
           scratch.shared,
           scratch.gates}),
      "Qwen3.8 compact MoE output aliases scratch");

  // All chunk contracts and allocations are checked before the first launch.
  record_inputs(stream, x, logits_or_router, router_scale, weights);
  record_tensor(output, stream);
  for (const auto* t :
       {&scratch.logits,
        &scratch.ids,
        &scratch.weights,
        &scratch.routed,
        &scratch.shared,
        &scratch.gates})
    record_tensor(*t, stream);
  const auto* x_ptr = reinterpret_cast<const sycl::half*>(x.const_data_ptr());
  const auto* logits_ptr = router_scale.defined()
                               ? nullptr
                               : reinterpret_cast<const sycl::half*>(
                                     logits_or_router.const_data_ptr());
  const auto* router_ptr =
      router_scale.defined()
          ? reinterpret_cast<const uint8_t*>(logits_or_router.const_data_ptr())
          : nullptr;
  const auto* router_scale_ptr =
      router_scale.defined()
          ? reinterpret_cast<const sycl::half*>(router_scale.const_data_ptr())
          : nullptr;
  auto* out_ptr = reinterpret_cast<sycl::half*>(output.data_ptr());
  const Workspace workspace{
      reinterpret_cast<sycl::half*>(scratch.logits.data_ptr()),
      scratch.ids.data_ptr<int32_t>(),
      reinterpret_cast<sycl::half*>(scratch.weights.data_ptr()),
      reinterpret_cast<sycl::half*>(scratch.routed.data_ptr()),
      reinterpret_cast<sycl::half*>(scratch.shared.data_ptr()),
      scratch.gates.data_ptr<float>()};
  for (int offset = 0; offset < rows; offset += 8) {
    const int count = std::min(8, rows - offset);
    const Inputs input{
        x_ptr + int64_t(offset) * kHidden,
        logits_ptr ? logits_ptr + int64_t(offset) * kExperts : nullptr,
        router_ptr,
        router_scale_ptr,
        reinterpret_cast<const uint8_t*>(weights[0].const_data_ptr()),
        reinterpret_cast<const sycl::half*>(weights[1].const_data_ptr()),
        reinterpret_cast<const uint8_t*>(weights[2].const_data_ptr()),
        reinterpret_cast<const sycl::half*>(weights[3].const_data_ptr()),
        reinterpret_cast<const sycl::half*>(weights[4].const_data_ptr()),
        reinterpret_cast<const sycl::half*>(weights[5].const_data_ptr()),
        reinterpret_cast<const sycl::half*>(weights[6].const_data_ptr()),
        out_ptr + int64_t(offset) * kHidden,
        count,
        width};
    TORCH_CHECK(
        try_forward(queue, input, workspace),
        "prevalidated Qwen3.8 compact MoE chunk was rejected");
  }
  return output;
}

std::optional<at::Tensor> Qwen38M1Workspace::try_run(
    at::Tensor x,
    at::Tensor router,
    at::Tensor router_scale,
    std::vector<at::Tensor> weights,
    int64_t intermediate) {
  if (!x.defined() || x.dim() != 2 || x.size(0) != 1) return std::nullopt;
  return core_.try_run(
      x, router, router_scale, std::move(weights), intermediate, false);
}

std::optional<at::Tensor> Qwen38MultiWorkspace::try_run(
    at::Tensor x,
    at::Tensor router,
    at::Tensor router_scale,
    std::vector<at::Tensor> weights,
    int64_t intermediate,
    bool grouped) {
  if (!x.defined() || x.dim() != 2 || x.size(0) < 2 || x.size(0) > 8)
    return std::nullopt;
  return core_.try_run(
      x, router, router_scale, std::move(weights), intermediate, grouped);
}

}  // namespace vllm::qwen38::moe_sycl
