// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <c10/core/TensorImpl.h>
#include <c10/util/Exception.h>
#include <torch/types.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace vllm::qwen38 {

// Private host-only ABI: the Python caller filters None and checks exact
// builtin/vLLM tensor types, lazy views and Torch modes before either call.
// No device allocation, submission or initialization; no tensor mutation.
// Inference content updates require the caller's existing reload invalidation.
// Concurrent tensor metadata mutation is outside this contract.
class TensorBindingSnapshotV1 final {
 public:
  explicit TensorBindingSnapshotV1(const std::vector<torch::Tensor>& tensors)
      : sources_(tensors) {
    entries_.reserve(sources_.size());
    for (const auto& tensor : sources_) {
      if (!supported(tensor)) return;
      const void* pointer;
      std::optional<uint32_t> version;
      if (!read_pointer(tensor, pointer) || !read_version(tensor, version))
        return;
      entries_.push_back(
          {tensor.unsafeGetTensorImpl(),
           tensor.device(),
           tensor.scalar_type(),
           tensor.sizes().vec(),
           tensor.strides().vec(),
           pointer,
           version});
    }
    valid_ = true;
  }

  bool matches(const std::vector<torch::Tensor>& tensors) const {
    if (!valid_ || tensors.size() != entries_.size()) return false;
    for (size_t i = 0; i < tensors.size(); ++i) {
      const auto& tensor = tensors[i];
      const auto& saved = entries_[i];
      if (!supported(tensor) ||
          tensor.unsafeGetTensorImpl() != saved.identity ||
          tensor.device() != saved.device ||
          tensor.scalar_type() != saved.dtype ||
          tensor.sizes() != at::IntArrayRef(saved.sizes) ||
          tensor.strides() != at::IntArrayRef(saved.strides))
        return false;
      const void* pointer;
      std::optional<uint32_t> version;
      if (!read_pointer(tensor, pointer) || pointer != saved.pointer ||
          !read_version(tensor, version) || version != saved.version)
        return false;
    }
    return true;
  }

 private:
  struct Entry {
    const c10::TensorImpl* identity;
    c10::Device device;
    at::ScalarType dtype;
    std::vector<int64_t> sizes, strides;
    const void* pointer;
    std::optional<uint32_t> version;
  };

  static bool supported(const torch::Tensor& tensor) {
    return tensor.defined() && tensor.layout() == at::kStrided &&
           !tensor.is_meta() && !tensor.is_nested() && tensor.has_storage() &&
           !tensor.is_conj() && !tensor.is_neg() &&
           !tensor.unsafeGetTensorImpl()->has_symbolic_sizes_strides();
  }

  static bool read_pointer(const torch::Tensor& tensor, const void*& pointer) {
    // Only pointer-access errors are converted to a miss. Do not hide OOM,
    // Python argument/ABI errors, or errors in other metadata inspection.
    try {
      pointer = tensor.const_data_ptr();
    } catch (const c10::OutOfMemoryError&) {
      throw;
    } catch (const c10::Error&) {
      return false;
    }
    return true;
  }

  static bool
  read_version(const torch::Tensor& tensor, std::optional<uint32_t>& version) {
    version.reset();
    if (tensor.is_inference()) return true;
    const auto& counter = tensor.unsafeGetTensorImpl()->version_counter();
    if (!counter.enabled()) return false;
    version = counter.current_version();
    return true;
  }

  // Keep the original TensorImpls alive, including unsupported sources. This
  // prevents an old identity address being recycled after caller rebinding.
  std::vector<torch::Tensor> sources_;
  std::vector<Entry> entries_;
  bool valid_ = false;
};

}  // namespace vllm::qwen38
