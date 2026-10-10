// SPDX-License-Identifier: Apache-2.0
#include <torch/library.h>
#include <torch/types.h>

#include "core/registration.h"
#include "qwen38/gdn_sycl.h"
#include "qwen38/gdn_sycl_projection.h"

#define QWEN38_LIBRARY_FRAGMENT(NAME, MODULE) TORCH_LIBRARY_FRAGMENT(NAME, MODULE)

namespace {
constexpr const char* kGdnNames[] = {
    "gdn_decode_sycl", "gdn_spec_v2_sycl", "gdn_norm_gate_sycl",
    "gdn_norm_int4_sycl"};
}

QWEN38_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  m.def("gdn_decode_sycl(Tensor qkvz, Tensor(a!) conv_state, Tensor conv_weight, "
        "Tensor conv_bias, Tensor conv_indices, Tensor a_log, Tensor dt_bias, "
        "Tensor ba, Tensor(b!) ssm_state, Tensor ssm_indices, "
        "Tensor(c!) output, Tensor(d!) z, float scale) -> ()");
  m.impl("gdn_decode_sycl", torch::kXPU, &vllm::qwen38::gdn_decode_sycl);
  m.def("gdn_spec_v2_sycl(Tensor qkvz, Tensor(a!) conv_state, Tensor conv_weight, "
        "Tensor conv_bias, Tensor spec_indices, Tensor a_log, Tensor dt_bias, "
        "Tensor ba, Tensor(b!) ssm_state, Tensor(c!) output, Tensor(d!) z, "
        "Tensor token_indices, Tensor accepted, int sequences, "
        "int tokens_per_sequence, float scale) -> ()");
  m.impl("gdn_spec_v2_sycl", torch::kXPU, &vllm::qwen38::gdn_spec_v2_sycl);
  m.def("gdn_norm_gate_sycl(Tensor x, Tensor z, Tensor weight, "
        "Tensor(a!) normalized, float eps, bool sigmoid_gate) -> ()");
  m.impl("gdn_norm_gate_sycl", torch::kXPU, &vllm::qwen38::gdn_norm_gate_sycl);
  m.def("gdn_norm_int4_sycl(Tensor x, Tensor z, Tensor norm_weight, "
        "Tensor weight, Tensor scale, Tensor(a!) output, int hv, int v, "
        "float eps, bool sigmoid_gate) -> ()");
  m.impl("gdn_norm_int4_sycl", torch::kXPU, &vllm::qwen38::gdn_norm_int4_sycl);
}

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, Negative, m) {
  for (const auto* name : kGdnNames)
    m.impl(name, torch::CppFunction::makeFallthrough());
}
TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, Conjugate, m) {
  for (const auto* name : kGdnNames)
    m.impl(name, torch::CppFunction::makeFallthrough());
}
