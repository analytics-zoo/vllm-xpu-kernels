// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <ATen/ATen.h>
#include <pybind11/pybind11.h>

namespace vllm::qwen38::qsa_sycl {

// Qwen3.8 QSA Q/G/K/V order is [Q0,G0,...,K0,V0] when gated.
// All outputs are caller-owned, disjoint FP16 tensors. The producer must prove
// every device position is within cos_sin_cache before calling this API.
at::Tensor qkv_postprocess_v1(
    const at::Tensor& qkv, at::Tensor q_out, at::Tensor gate_out,
    at::Tensor k_out, at::Tensor v_out, const at::Tensor& norm_wq,
    const at::Tensor& norm_wk, const at::Tensor& positions,
    const at::Tensor& cos_sin_cache, int64_t q_heads, int64_t kv_heads,
    bool attn_output_gate, bool mrope, bool positions_bounds_proven);

// A single portable-SYCL entry covers ESIMD indexer v1 (FP32 RoPE) and v2
// (eager FP16 normalization/product boundaries). Input may have a padded row
// stride; output is contiguous caller-owned storage.
at::Tensor indexer_norm_rope_v2(
    const at::Tensor& input, at::Tensor output, const at::Tensor& weight,
    const at::Tensor& positions, const at::Tensor& cos_sin_cache,
    bool mrope, bool eager_fp16, bool positions_bounds_proven);

// Symmetric GGML q4_0: low nibble is even K, high nibble odd K, zero=8,
// one FP16 scale per 128 K values. Supports the QSA indexer TP4 [2560,640]
// geometry and M=1..8. Caller splits output[:, :512] and output[:, 512:].
at::Tensor indexer_projection_int4_v1(
    const at::Tensor& input, const at::Tensor& packed_weight,
    const at::Tensor& group_scales, at::Tensor output);

// Non-fused composition. The caller supplies both norm/RoPE output and all
// selection workspaces. The implementation preflights both stages before the
// first submit; an in-order current stream carries the data dependency.
at::Tensor q_norm_rope_select_v1(
    const at::Tensor& projected_q, const at::Tensor& norm_weight,
    const at::Tensor& positions, const at::Tensor& cos_sin_cache,
    const at::Tensor& compressed_key_cache, const at::Tensor& page_table,
    const at::Tensor& token_to_req, const at::Tensor& query_positions,
    const at::Tensor& sequence_lengths, at::Tensor q_output, at::Tensor out,
    at::Tensor scores_a, at::Tensor indices_a, at::Tensor scores_b,
    at::Tensor indices_b, int64_t compressed_page_size, int64_t max_seq_len,
    bool mrope, bool eager_fp16, bool positions_bounds_proven);

// Called by the production extension's existing PYBIND11_MODULE body.
void bind_qsa_sycl_aux(pybind11::module_& module);

}  // namespace vllm::qwen38::qsa_sycl
