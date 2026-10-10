// SPDX-License-Identifier: Apache-2.0
#include <torch/library.h>
#include <torch/types.h>

#include <utility>

#include "core/registration.h"
#include "qwen38/ple_sycl.h"

#define QWEN38_LIBRARY_FRAGMENT(NAME, MODULE) TORCH_LIBRARY_FRAGMENT(NAME, MODULE)

namespace {

// The portable functions return their caller-owned output, but the existing
// model dispatcher ABI intentionally returns void. Preserve that ABI without
// adding a Python wrapper or an extra allocation to each call.
template <auto Function>
struct VoidResult;

template <typename Result, typename... Args, Result (*Function)(Args...)>
struct VoidResult<Function> {
  static void call(Args... args) {
    (void)Function(std::forward<Args>(args)...);
  }
};

constexpr const char* kPleNames[] = {
    "ple_ngram_ids", "ple_embedding_gather", "ple_grouped_norm",
    "ple_score_gate", "ple_gated_value", "ple_gated_value_grouped_norm",
    "ple_residual_add", "ple_short_conv_decode", "ple_short_conv_prefill",
    "ple_short_conv_spec", "ple_short_conv_decode_trusted",
    "ple_short_conv_prefill_trusted", "ple_short_conv_spec_trusted"};

}  // namespace

QWEN38_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  namespace ple = vllm::qwen38::ple_sycl;
  m.def("ple_ngram_ids(Tensor input_ids, Tensor query_start_loc, "
        "Tensor ngram_context, Tensor layer_multipliers, "
        "Tensor ngram_heads_vocab_sizes, Tensor ngram_heads_offsets, "
        "Tensor(a!) output, int eos_token_id, int heads_per_ngram) -> ()");
  m.impl("ple_ngram_ids", torch::kXPU, &VoidResult<&ple::ngram_ids>::call);
  m.def("ple_embedding_gather(Tensor ngram_ids, Tensor local_weight, "
        "Tensor local_vocab_start, Tensor local_num_rows, "
        "Tensor(a!) local_partial) -> ()");
  m.impl("ple_embedding_gather", torch::kXPU,
         &VoidResult<&ple::embedding_gather>::call);
  m.def("ple_grouped_norm(Tensor input, Tensor weight, Tensor(a!) output, "
        "float eps, int group_size) -> ()");
  m.impl("ple_grouped_norm", torch::kXPU, &VoidResult<&ple::grouped_norm>::call);
  m.def("ple_score_gate(Tensor key, Tensor query, Tensor(a!) output, "
        "int hidden_size) -> ()");
  m.impl("ple_score_gate", torch::kXPU, &VoidResult<&ple::score_gate>::call);
  m.def("ple_gated_value(Tensor gate, Tensor value, Tensor(a!) output, "
        "int hc_count) -> ()");
  m.impl("ple_gated_value", torch::kXPU, &VoidResult<&ple::gated_value>::call);
  m.def("ple_gated_value_grouped_norm(Tensor gate, Tensor value, Tensor weight, "
        "Tensor(a!) raw_output, Tensor(b!) normalized_output, float eps) -> ()");
  m.impl("ple_gated_value_grouped_norm", torch::kXPU,
         &VoidResult<&ple::gated_value_grouped_norm>::call);
  m.def("ple_residual_add(Tensor gated_value_flat, Tensor conv_output, "
        "Tensor(a!) output) -> ()");
  m.impl("ple_residual_add", torch::kXPU, &VoidResult<&ple::residual_add>::call);
  for (const char* name : {"ple_short_conv_decode",
                           "ple_short_conv_decode_trusted"}) {
    const auto schema = std::string(name) +
        "(Tensor input, Tensor(a!) conv_state, Tensor conv_weights, "
        "Tensor state_indices, Tensor has_initial_state, Tensor(b!) output, "
        "int dilation, bool state_dim_first, int null_block_id) -> ()";
    m.def(schema.c_str());
  }
  m.impl("ple_short_conv_decode", torch::kXPU,
         &VoidResult<&ple::short_conv_decode>::call);
  m.impl("ple_short_conv_decode_trusted", torch::kXPU,
         &VoidResult<&ple::short_conv_decode_trusted>::call);
  for (const char* name : {"ple_short_conv_prefill",
                           "ple_short_conv_prefill_trusted"}) {
    const auto schema = std::string(name) +
        "(Tensor input, Tensor query_start_loc, Tensor(a!) conv_state, "
        "Tensor conv_weights, Tensor state_indices, Tensor has_initial_state, "
        "Tensor(b!) output, int dilation, bool state_dim_first, "
        "int null_block_id) -> ()";
    m.def(schema.c_str());
  }
  m.impl("ple_short_conv_prefill", torch::kXPU,
         &VoidResult<&ple::short_conv_prefill>::call);
  m.impl("ple_short_conv_prefill_trusted", torch::kXPU,
         &VoidResult<&ple::short_conv_prefill_trusted>::call);
  for (const char* name : {"ple_short_conv_spec",
                           "ple_short_conv_spec_trusted"}) {
    const auto schema = std::string(name) +
        "(Tensor input, Tensor query_start_loc, Tensor(a!) conv_state, "
        "Tensor conv_weights, Tensor state_indices, Tensor num_accepted_tokens, "
        "Tensor(b!) output, int num_spec_tokens, int dilation, "
        "bool state_dim_first, int null_block_id) -> ()";
    m.def(schema.c_str());
  }
  m.impl("ple_short_conv_spec", torch::kXPU,
         &VoidResult<&ple::short_conv_spec>::call);
  m.impl("ple_short_conv_spec_trusted", torch::kXPU,
         &VoidResult<&ple::short_conv_spec_trusted>::call);
}

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, Negative, m) {
  for (const auto* name : kPleNames)
    m.impl(name, torch::CppFunction::makeFallthrough());
}
TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, Conjugate, m) {
  for (const auto* name : kPleNames)
    m.impl(name, torch::CppFunction::makeFallthrough());
}
