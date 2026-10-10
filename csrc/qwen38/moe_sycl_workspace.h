// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <ATen/ATen.h>
#include <c10/core/Stream.h>
#include <torch/custom_class.h>

#include <array>
#include <cstdint>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

namespace pybind11 {
class module_;
}

namespace vllm::qwen38::moe_sycl {

// Metadata-only status: 0=valid, 1=device, 2=dtype, 3=shape,
// 4=layout/alignment. Matches the legacy compact80/160 preflight ABI.
int64_t compact_weight_contract(
    at::TensorList weights, c10::Device device, int intermediate);
bool output_overlaps(const at::Tensor& output, at::TensorList inputs);

// Owns scratch per stream, physical intermediate and M. Scratch is reused only
// on the same in-order queue; two output slots may be reused only when neither
// the TensorImpl nor its storage is shared with a caller or a view. Returned
// tensors have independent metadata and share only the private slot's storage;
// Python wrapper preservation cannot pin the private TensorImpl's refcount.
// All metadata/alias/queue checks and allocations precede the first submit.
class MoeSyclWorkspaceCore {
 public:
  std::optional<at::Tensor> try_run(
      const at::Tensor& x,
      const at::Tensor& router,
      const at::Tensor& router_scale,
      std::vector<at::Tensor> weights,
      int64_t intermediate,
      bool grouped);

  at::Tensor run_out(
      const at::Tensor& x,
      const at::Tensor& logits_or_router,
      const at::Tensor& router_scale,
      std::vector<at::Tensor> weights,
      const at::Tensor& output,
      int64_t intermediate,
      bool grouped);

 private:
  struct Scratch {
    at::Tensor logits, ids, weights, routed, shared, gates;
    std::array<at::Tensor, 2> outputs;
  };
  using Slots = std::array<std::array<Scratch, 9>, 2>;
  std::mutex mutex_;
  std::unordered_map<c10::Stream, Slots> buffers_;
};

class Qwen38M1Workspace final : public torch::CustomClassHolder {
 public:
  std::optional<at::Tensor> try_run(
      at::Tensor x,
      at::Tensor router,
      at::Tensor router_scale,
      std::vector<at::Tensor> weights,
      int64_t intermediate);

 private:
  MoeSyclWorkspaceCore core_;
};

class Qwen38MultiWorkspace final : public torch::CustomClassHolder {
 public:
  std::optional<at::Tensor> try_run(
      at::Tensor x,
      at::Tensor router,
      at::Tensor router_scale,
      std::vector<at::Tensor> weights,
      int64_t intermediate,
      bool grouped);

 private:
  MoeSyclWorkspaceCore core_;
};

// Called by the main _qwen38_C PYBIND11_MODULE after all native sources link.
void bind_moe_sycl_workspace(pybind11::module_& module);

}  // namespace vllm::qwen38::moe_sycl
