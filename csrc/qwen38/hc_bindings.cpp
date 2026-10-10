// SPDX-License-Identifier: Apache-2.0
#include <ATen/MemoryOverlap.h>
#include <torch/library.h>

#include <cstdint>

#include "core/registration.h"
#include "qwen38/hc_sycl.h"

#define QWEN38_LIBRARY_FRAGMENT(NAME, MODULE) \
  TORCH_LIBRARY_FRAGMENT(NAME, MODULE)

namespace {

bool hc_outputs_alias_inputs(at::TensorList outputs, at::TensorList inputs) {
  for (const auto& output : outputs) {
    for (const auto& input : inputs) {
      // Retain the conservative storage-alias rule, and also reject overlap
      // between separate Storage owners wrapping the same physical memory.
      if (output.is_alias_of(input) ||
          at::get_overlap_status(output, input) != at::MemOverlapStatus::No) {
        return true;
      }
      const auto out_start =
          reinterpret_cast<uintptr_t>(output.const_data_ptr());
      const auto in_start = reinterpret_cast<uintptr_t>(input.const_data_ptr());
      if (out_start <= in_start
              ? in_start - out_start < output.numel() * output.element_size()
              : out_start - in_start < input.numel() * input.element_size()) {
        return true;
      }
    }
  }
  return false;
}

}  // namespace

QWEN38_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  namespace hc = vllm::qwen38::hc;
  m.def(
      "hc_grouped_norm_v1(Tensor input, Tensor weight, Tensor(a!) output, "
      "float eps) -> ()");
  m.impl("hc_grouped_norm_v1", torch::kXPU, &hc::grouped_norm);
  for (const auto* name :
       {"hc_gate_mix_v1", "hc_gate_mix_m4_v1", "hc_gate_mix_multi_m_v1"}) {
    const auto schema = std::string(name) +
                        "(Tensor input, Tensor gate, Tensor(a!) output) -> ()";
    m.def(schema.c_str());
    m.impl(name, torch::kXPU, &hc::gate_mix);
  }
  m.def(
      "hc_combine_v1(Tensor hidden, Tensor block, Tensor injection, "
      "Tensor(a!) output) -> ()");
  m.impl("hc_combine_v1", torch::kXPU, &hc::combine);
  for (const auto* name :
       {"hc_combine_norm_v1",
        "hc_combine_norm_m4_v1",
        "hc_combine_norm_multi_m_v1",
        "hc_combine_norm_multi_m_strided_v1"}) {
    const auto schema =
        std::string(name) +
        "(Tensor hidden, Tensor block, Tensor injection, Tensor weight, "
        "Tensor(a!) combined, Tensor(b!) normed, float eps) -> ()";
    m.def(schema.c_str());
    m.impl(name, torch::kXPU, &hc::combine_norm);
  }
  m.def("hc_down(Tensor input, Tensor weight, Tensor(a!) output) -> ()");
  m.impl("hc_down", torch::kXPU, &hc::down);
  m.def("hc_up(Tensor input, Tensor weight, Tensor(a!) output) -> ()");
  m.impl("hc_up", torch::kXPU, &hc::up);
  m.def(
      "hc_up_gate_mix(Tensor lowrank, Tensor weight, Tensor normed, "
      "Tensor(a!) output) -> ()");
  m.impl("hc_up_gate_mix", torch::kXPU, &hc::up_gate_mix);
  m.def(
      "hc_prefill_grouped_norm(Tensor input, Tensor weight, "
      "Tensor(a!) output, float eps) -> ()");
  m.impl("hc_prefill_grouped_norm", torch::kXPU, &hc::prefill_grouped_norm);
  m.def(
      "hc_prefill_combine(Tensor hidden, Tensor block, Tensor injection, "
      "Tensor(a!) output) -> ()");
  m.impl("hc_prefill_combine", torch::kXPU, &hc::prefill_combine);
  m.def(
      "hc_prefill_combine_norm(Tensor hidden, Tensor block, Tensor injection, "
      "Tensor weight, Tensor(a!) combined, Tensor(b!) normed, float eps) -> "
      "()");
  m.impl("hc_prefill_combine_norm", torch::kXPU, &hc::prefill_combine_norm);
  m.def(
      "hc_prefill_gate_mix(Tensor input, Tensor gate, Tensor(a!) output) -> "
      "()");
  m.impl("hc_prefill_gate_mix", torch::kXPU, &hc::prefill_gate_mix);
  m.def(
      "hc_outputs_alias_inputs_v1(Tensor[] outputs, Tensor[] inputs) -> bool",
      &hc_outputs_alias_inputs);
}

// Do not let dispatcher fallbacks materialize lazy output views before the
// native whole-transaction preflight can reject them.
TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, Negative, m) {
  for (const auto* name :
       {"hc_grouped_norm_v1",
        "hc_gate_mix_v1",
        "hc_gate_mix_m4_v1",
        "hc_gate_mix_multi_m_v1",
        "hc_combine_v1",
        "hc_combine_norm_v1",
        "hc_combine_norm_m4_v1",
        "hc_combine_norm_multi_m_v1",
        "hc_combine_norm_multi_m_strided_v1",
        "hc_down",
        "hc_up",
        "hc_up_gate_mix",
        "hc_outputs_alias_inputs_v1",
        "hc_prefill_grouped_norm",
        "hc_prefill_combine",
        "hc_prefill_combine_norm",
        "hc_prefill_gate_mix"}) {
    m.impl(name, torch::CppFunction::makeFallthrough());
  }
}
TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, Conjugate, m) {
  for (const auto* name :
       {"hc_grouped_norm_v1",
        "hc_gate_mix_v1",
        "hc_gate_mix_m4_v1",
        "hc_gate_mix_multi_m_v1",
        "hc_combine_v1",
        "hc_combine_norm_v1",
        "hc_combine_norm_m4_v1",
        "hc_combine_norm_multi_m_v1",
        "hc_combine_norm_multi_m_strided_v1",
        "hc_down",
        "hc_up",
        "hc_up_gate_mix",
        "hc_outputs_alias_inputs_v1",
        "hc_prefill_grouped_norm",
        "hc_prefill_combine",
        "hc_prefill_combine_norm",
        "hc_prefill_gate_mix"}) {
    m.impl(name, torch::CppFunction::makeFallthrough());
  }
}
