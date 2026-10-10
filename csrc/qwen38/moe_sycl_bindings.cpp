// SPDX-License-Identifier: Apache-2.0
#include "qwen38/moe_sycl_workspace.h"

#include "qwen38/moe_sycl_prefill_torch.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/library.h>

#include <vector>

namespace vllm::qwen38::moe_sycl {
namespace {

using Tensor = at::Tensor;

Tensor forward_out(
    Tensor x,
    Tensor logits_or_router,
    Tensor router_scale,
    Tensor w13,
    Tensor s13,
    Tensor w2,
    Tensor s2,
    Tensor shared_up,
    Tensor shared_down,
    Tensor shared_gate,
    Tensor output,
    int64_t top_k,
    int64_t num_shared,
    int64_t num_experts,
    int intermediate,
    bool grouped) {
  TORCH_CHECK(
      top_k == 10 && num_shared == 1 && num_experts == 512,
      "Qwen3.8 compact MoE requires top-k=10, one shared expert, E=512");
  // Scratch is stream/M-specific and kept by this thread-local owner. No
  // physical weight conversion or copy: signed S4 is read as raw bytes.
  static thread_local MoeSyclWorkspaceCore owner;
  return owner.run_out(
      x,
      logits_or_router,
      router_scale,
      {w13, s13, w2, s2, shared_up, shared_down, shared_gate},
      output,
      intermediate,
      grouped);
}

Tensor compact80_m1_out(
    Tensor x,
    Tensor logits,
    Tensor w13,
    Tensor s13,
    Tensor w2,
    Tensor s2,
    Tensor shared_up,
    Tensor shared_down,
    Tensor shared_gate,
    Tensor output,
    int64_t top_k,
    int64_t num_shared,
    int64_t num_experts) {
  TORCH_CHECK(
      x.dim() == 2 && x.size(0) == 1, "compact80 M1 public op requires M=1");
  return forward_out(
      x,
      logits,
      {},
      w13,
      s13,
      w2,
      s2,
      shared_up,
      shared_down,
      shared_gate,
      output,
      top_k,
      num_shared,
      num_experts,
      80,
      false);
}

Tensor compact80_multi_out(
    Tensor x,
    Tensor logits,
    Tensor w13,
    Tensor s13,
    Tensor w2,
    Tensor s2,
    Tensor shared_up,
    Tensor shared_down,
    Tensor shared_gate,
    Tensor output,
    int64_t top_k,
    int64_t num_shared,
    int64_t num_experts) {
  TORCH_CHECK(
      x.dim() == 2 && x.size(0) >= 2,
      "compact80 multi public op requires M>=2");
  return forward_out(
      x,
      logits,
      {},
      w13,
      s13,
      w2,
      s2,
      shared_up,
      shared_down,
      shared_gate,
      output,
      top_k,
      num_shared,
      num_experts,
      80,
      false);
}

Tensor compact80_multi_grouped_out(
    Tensor x,
    Tensor logits,
    Tensor w13,
    Tensor s13,
    Tensor w2,
    Tensor s2,
    Tensor shared_up,
    Tensor shared_down,
    Tensor shared_gate,
    Tensor output,
    int64_t top_k,
    int64_t num_shared,
    int64_t num_experts) {
  TORCH_CHECK(
      x.dim() == 2 && x.size(0) >= 2,
      "compact80 grouped public op requires M>=2");
  return forward_out(
      x,
      logits,
      {},
      w13,
      s13,
      w2,
      s2,
      shared_up,
      shared_down,
      shared_gate,
      output,
      top_k,
      num_shared,
      num_experts,
      80,
      true);
}

Tensor compact80_router_out(
    Tensor x,
    Tensor router,
    Tensor router_scale,
    Tensor w13,
    Tensor s13,
    Tensor w2,
    Tensor s2,
    Tensor shared_up,
    Tensor shared_down,
    Tensor shared_gate,
    Tensor output,
    int64_t top_k,
    int64_t num_shared,
    int64_t num_experts) {
  return forward_out(
      x,
      router,
      router_scale,
      w13,
      s13,
      w2,
      s2,
      shared_up,
      shared_down,
      shared_gate,
      output,
      top_k,
      num_shared,
      num_experts,
      80,
      false);
}

Tensor compact160_out(
    Tensor x,
    Tensor logits,
    Tensor w13,
    Tensor s13,
    Tensor w2,
    Tensor s2,
    Tensor shared_up,
    Tensor shared_down,
    Tensor shared_gate,
    Tensor output,
    int64_t top_k,
    int64_t num_shared,
    int64_t num_experts) {
  return forward_out(
      x,
      logits,
      {},
      w13,
      s13,
      w2,
      s2,
      shared_up,
      shared_down,
      shared_gate,
      output,
      top_k,
      num_shared,
      num_experts,
      160,
      false);
}

Tensor compact160_router_out(
    Tensor x,
    Tensor router,
    Tensor router_scale,
    Tensor w13,
    Tensor s13,
    Tensor w2,
    Tensor s2,
    Tensor shared_up,
    Tensor shared_down,
    Tensor shared_gate,
    Tensor output,
    int64_t top_k,
    int64_t num_shared,
    int64_t num_experts) {
  return forward_out(
      x,
      router,
      router_scale,
      w13,
      s13,
      w2,
      s2,
      shared_up,
      shared_down,
      shared_gate,
      output,
      top_k,
      num_shared,
      num_experts,
      160,
      false);
}

int64_t compact80_contract(at::TensorList weights, c10::Device device) {
  return compact_weight_contract(weights, device, 80);
}

int64_t compact160_contract(at::TensorList weights, c10::Device device) {
  return compact_weight_contract(weights, device, 160);
}

// Register both Torch classes for the established M1 lookup and direct
// pybind classes for the lower-overhead guarded model path.
auto m1_class =
    torch::class_<Qwen38M1Workspace>("qwen38_moe_sycl", "Qwen38M1WorkspaceV1")
        .def(torch::init<>())
        .def("try_run", &Qwen38M1Workspace::try_run);
auto multi_class = torch::class_<Qwen38MultiWorkspace>(
                       "qwen38_moe_sycl", "Qwen38MultiWorkspaceV1")
                       .def(torch::init<>())
                       .def("try_run", &Qwen38MultiWorkspace::try_run);

}  // namespace

void bind_moe_sycl_workspace(pybind11::module_& module) {
  namespace py = pybind11;
  py::class_<Qwen38M1Workspace>(module, "Qwen38M1WorkspaceDirectV1")
      .def(py::init<>())
      .def("try_run", &Qwen38M1Workspace::try_run);
  py::class_<Qwen38MultiWorkspace>(module, "Qwen38MultiWorkspaceDirectV1")
      .def(py::init<>())
      .def("try_run", &Qwen38MultiWorkspace::try_run);
}

}  // namespace vllm::qwen38::moe_sycl

TORCH_LIBRARY(qwen38_moe_sycl, m) {
  using namespace vllm::qwen38::moe_sycl;
  m.def(
      "moe_forward_m1_cutlass_nmajor_int4_fp16_shared_compact80_out_v1("
      "Tensor x, Tensor logits, Tensor w13_qweight_s4, Tensor w13_scales, "
      "Tensor w2_qweight_s4, Tensor w2_scales, Tensor shared_gate_up_weight, "
      "Tensor shared_down_weight, Tensor shared_expert_gate_weight, "
      "Tensor(a!) output, int top_k, int num_shared_experts, "
      "int n_routed_experts) -> "
      "Tensor(a!)");
  m.impl(
      "moe_forward_m1_cutlass_nmajor_int4_fp16_shared_compact80_out_v1",
      at::kXPU,
      &compact80_m1_out);
  m.def(
      "moe_forward_multi_m_cutlass_nmajor_int4_fp16_shared_compact80_out_v1("
      "Tensor x, Tensor logits, Tensor w13_qweight_s4, Tensor w13_scales, "
      "Tensor w2_qweight_s4, Tensor w2_scales, Tensor shared_gate_up_weight, "
      "Tensor shared_down_weight, Tensor shared_expert_gate_weight, "
      "Tensor(a!) output, int top_k, int num_shared_experts, "
      "int n_routed_experts) -> "
      "Tensor(a!)");
  m.impl(
      "moe_forward_multi_m_cutlass_nmajor_int4_fp16_shared_compact80_out_v1",
      at::kXPU,
      &compact80_multi_out);
  m.def(
      "moe_forward_multi_m_cutlass_nmajor_int4_fp16_shared_compact80_grouped_"
      "out_v1("
      "Tensor x, Tensor logits, Tensor w13_qweight_s4, Tensor w13_scales, "
      "Tensor w2_qweight_s4, Tensor w2_scales, Tensor shared_gate_up_weight, "
      "Tensor shared_down_weight, Tensor shared_expert_gate_weight, "
      "Tensor(a!) output, int top_k, int num_shared_experts, "
      "int n_routed_experts) -> "
      "Tensor(a!)");
  m.impl(
      "moe_forward_multi_m_cutlass_nmajor_int4_fp16_shared_compact80_grouped_"
      "out_v1",
      at::kXPU,
      &compact80_multi_grouped_out);
  m.def(
      "moe_forward_m1_cutlass_nmajor_int4_fp16_shared_compact80_router_out_v1("
      "Tensor x, Tensor router_weight, Tensor router_scale, "
      "Tensor w13_qweight_s4, Tensor w13_scales, Tensor w2_qweight_s4, "
      "Tensor w2_scales, Tensor shared_gate_up_weight, "
      "Tensor shared_down_weight, Tensor shared_expert_gate_weight, "
      "Tensor(a!) output, int top_k, int num_shared_experts, "
      "int n_routed_experts) -> Tensor(a!)");
  m.impl(
      "moe_forward_m1_cutlass_nmajor_int4_fp16_shared_compact80_router_out_v1",
      at::kXPU,
      &compact80_router_out);
  m.def(
      "moe_forward_compact160_out_v1(Tensor x, Tensor logits, "
      "Tensor w13_qweight_s4, Tensor w13_scales, Tensor w2_qweight_s4, "
      "Tensor w2_scales, Tensor shared_gate_up_weight, "
      "Tensor shared_down_weight, Tensor shared_expert_gate_weight, "
      "Tensor(a!) output, int top_k, int num_shared_experts, "
      "int n_routed_experts) -> Tensor(a!)");
  m.impl("moe_forward_compact160_out_v1", at::kXPU, &compact160_out);
  m.def(
      "moe_forward_compact160_router_out_v1(Tensor x, Tensor router_weight, "
      "Tensor router_scale, Tensor w13_qweight_s4, Tensor w13_scales, "
      "Tensor w2_qweight_s4, Tensor w2_scales, Tensor shared_gate_up_weight, "
      "Tensor shared_down_weight, Tensor shared_expert_gate_weight, "
      "Tensor(a!) output, int top_k, int num_shared_experts, "
      "int n_routed_experts) -> "
      "Tensor(a!)");
  m.impl(
      "moe_forward_compact160_router_out_v1", at::kXPU, &compact160_router_out);
  m.def(
      "moe_compact80_down_grouped_gemm(Tensor x, Tensor weight, Tensor scale, "
      "Tensor rows_per_expert) -> Tensor");
  m.impl(
      "moe_compact80_down_grouped_gemm", at::kXPU, &compact_down_grouped_gemm);
  m.def(
      "moe_compact160_down_grouped_gemm(Tensor x, Tensor weight, Tensor scale, "
      "Tensor rows_per_expert) -> Tensor");
  m.impl(
      "moe_compact160_down_grouped_gemm", at::kXPU, &compact_down_grouped_gemm);
  m.def(
      "qwen38_moe_compact80_weight_contract_v1(Tensor[] weights, Device "
      "device) -> int",
      &compact80_contract);
  m.def(
      "qwen38_moe_compact160_weight_contract_v1(Tensor[] weights, Device "
      "device) -> int",
      &compact160_contract);
  m.def(
      "qwen38_moe_output_overlaps_v1(Tensor output, Tensor[] inputs) -> bool",
      &output_overlaps);
}

TORCH_LIBRARY_IMPL(qwen38_moe_sycl, Negative, m) {
  for (const char* name :
       {"moe_forward_m1_cutlass_nmajor_int4_fp16_shared_compact80_out_v1",
        "moe_forward_multi_m_cutlass_nmajor_int4_fp16_shared_compact80_out_v1",
        "moe_forward_multi_m_cutlass_nmajor_int4_fp16_shared_compact80_grouped_"
        "out_v1",
        "moe_forward_m1_cutlass_nmajor_int4_fp16_shared_compact80_router_out_"
        "v1",
        "moe_forward_compact160_out_v1",
        "moe_forward_compact160_router_out_v1"})
    m.impl(name, torch::CppFunction::makeFallthrough());
}
TORCH_LIBRARY_IMPL(qwen38_moe_sycl, Conjugate, m) {
  for (const char* name :
       {"moe_forward_m1_cutlass_nmajor_int4_fp16_shared_compact80_out_v1",
        "moe_forward_multi_m_cutlass_nmajor_int4_fp16_shared_compact80_out_v1",
        "moe_forward_multi_m_cutlass_nmajor_int4_fp16_shared_compact80_grouped_"
        "out_v1",
        "moe_forward_m1_cutlass_nmajor_int4_fp16_shared_compact80_router_out_"
        "v1",
        "moe_forward_compact160_out_v1",
        "moe_forward_compact160_router_out_v1"})
    m.impl(name, torch::CppFunction::makeFallthrough());
}
