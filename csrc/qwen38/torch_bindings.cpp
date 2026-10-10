// SPDX-License-Identifier: Apache-2.0
#include <torch/library.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>

#include <memory>
#include <vector>

#include "core/registration.h"
#include "qwen38/gdn_sycl.h"
#include "qwen38/gdn_sycl_workspace.h"
#include "qwen38/hc_sycl_workspace.h"
#include "qwen38/ngram.h"
#include "qwen38/moe_sycl_workspace.h"
#include "qwen38/ops.h"
#include "qwen38/qsa_sycl.h"
#include "qwen38/qsa_sycl_aux.h"
#include "qwen38/qsa_sycl_owner.h"
#include "qwen38/tensor_binding.h"

namespace {
bool direct_tensor_flags_host(const std::vector<at::Tensor>& tensors) {
  for (const auto& tensor : tensors) {
    if (!tensor.defined() || tensor.is_conj() || tensor.is_neg()) {
      return false;
    }
  }
  return true;
}
}  // namespace

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  m.def(
      "q4_0_quantize(Tensor input, Tensor(a!) qweight, Tensor(b!) scale) -> "
      "()");
  m.impl("q4_0_quantize", torch::kXPU, &vllm::qwen38::q4_0_quantize);
  m.def(
      "int4_linear(Tensor input, Tensor weight, Tensor scale, "
      "Tensor(a!) output) -> ()");
  m.impl("int4_linear", torch::kXPU, &vllm::qwen38::int4_linear);
  m.def(
      "int4_linear_fused2(Tensor input, Tensor weight0, Tensor scale0, "
      "Tensor(a!) output0, Tensor weight1, Tensor scale1, "
      "Tensor(b!) output1) -> ()");
  m.impl("int4_linear_fused2", torch::kXPU, &vllm::qwen38::int4_linear_fused2);
  m.def(
      "ngram_decode_ids(Tensor input_ids, Tensor context, Tensor multipliers, "
      "Tensor(a!) output) -> ()");
  m.impl("ngram_decode_ids", torch::kXPU, &vllm::qwen38::ngram_decode_ids);
  m.def(
      "ngram_decode_ids_eos(Tensor input_ids, Tensor context, Tensor "
      "multipliers, "
      "Tensor(a!) output, int eos_token_id) -> ()");
  m.impl(
      "ngram_decode_ids_eos", torch::kXPU, &vllm::qwen38::ngram_decode_ids_eos);
  m.def(
      "ngram_host_lookup(Tensor weight, Tensor ids, Tensor(a!) output, "
      "int vocab_start, int vocab_end) -> Tensor(a!)");
  m.impl("ngram_host_lookup", torch::kXPU, &vllm::qwen38::ngram_host_lookup);
  m.def(
      "ngram_host_lookup_chunked(Tensor[] weights, Tensor ids, "
      "Tensor(a!) output, int vocab_start, int vocab_end) -> Tensor(a!)");
  m.impl(
      "ngram_host_lookup_chunked",
      torch::kXPU,
      &vllm::qwen38::ngram_host_lookup_chunked);
}

// Keep lazy views visible to the native preflight instead of materializing
// output arguments in a dispatcher fallback.
TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, Negative, m) {
  m.impl("q4_0_quantize", torch::CppFunction::makeFallthrough());
  m.impl("int4_linear", torch::CppFunction::makeFallthrough());
  m.impl("int4_linear_fused2", torch::CppFunction::makeFallthrough());
  m.impl("ngram_decode_ids", torch::CppFunction::makeFallthrough());
  m.impl("ngram_decode_ids_eos", torch::CppFunction::makeFallthrough());
  m.impl("ngram_host_lookup", torch::CppFunction::makeFallthrough());
  m.impl("ngram_host_lookup_chunked", torch::CppFunction::makeFallthrough());
}
TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, Conjugate, m) {
  m.impl("q4_0_quantize", torch::CppFunction::makeFallthrough());
  m.impl("int4_linear", torch::CppFunction::makeFallthrough());
  m.impl("int4_linear_fused2", torch::CppFunction::makeFallthrough());
  m.impl("ngram_decode_ids", torch::CppFunction::makeFallthrough());
  m.impl("ngram_decode_ids_eos", torch::CppFunction::makeFallthrough());
  m.impl("ngram_host_lookup", torch::CppFunction::makeFallthrough());
  m.impl("ngram_host_lookup_chunked", torch::CppFunction::makeFallthrough());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  namespace py = pybind11;
  module.def("direct_tensor_flags_host", &direct_tensor_flags_host);
  module.def("tensors_disjoint_host", &vllm::qwen38::tensors_disjoint_host);
  py::class_<vllm::qwen38::GdnSyclM1Workspace>(module, "GDNM1WorkspaceDirectV1")
      .def(py::init<>())
      .def("try_run", &vllm::qwen38::GdnSyclM1Workspace::try_run)
      .def("invalidate", &vllm::qwen38::GdnSyclM1Workspace::invalidate);
  py::class_<vllm::qwen38::GdnSyclSpecWorkspace>(
      module, "GDNSpecWorkspaceDirectV1")
      .def(py::init<>())
      .def("try_run", &vllm::qwen38::GdnSyclSpecWorkspace::try_run)
      .def("invalidate", &vllm::qwen38::GdnSyclSpecWorkspace::invalidate);
  py::class_<vllm::qwen38::TensorBindingSnapshotV1>(
      module, "TensorBindingSnapshotV1")
      .def(py::init<const std::vector<torch::Tensor>&>())
      .def("matches", &vllm::qwen38::TensorBindingSnapshotV1::matches);
  using vllm::qwen38::hc::HcSyclM1Workspace;
  using vllm::qwen38::hc::HcSyclMultiMWorkspace;
  py::class_<HcSyclM1Workspace, std::shared_ptr<HcSyclM1Workspace>>(
      module, "HCWorkspaceDirectV1")
      .def(py::init<>())
      .def("try_run", &HcSyclM1Workspace::try_run)
      .def("run", &HcSyclM1Workspace::run);
  py::class_<HcSyclMultiMWorkspace, std::shared_ptr<HcSyclMultiMWorkspace>>(
      module, "HCMultiMWorkspaceDirectV1")
      .def(py::init<>())
      .def("try_run", &HcSyclMultiMWorkspace::try_run)
      .def("run", &HcSyclMultiMWorkspace::run);
  vllm::qwen38::moe_sycl::bind_moe_sycl_workspace(module);
  module.attr("qsa_sycl_selection_wide_scratch_abi_version") = 1;
  module.def("group_compress_v2", &vllm::qwen38::qsa_sycl::group_compress_v2);
  module.def(
      "select_paged_tokens_v2",
      &vllm::qwen38::qsa_sycl::select_paged_tokens_v2);
  module.def(
      "token_split_attention_v3",
      &vllm::qwen38::qsa_sycl::token_split_attention_v3);
  vllm::qwen38::qsa_sycl::bind_qsa_sycl_aux(module);
  vllm::qwen38::qsa_sycl::bind_qsa_sycl_owner(module);
}
