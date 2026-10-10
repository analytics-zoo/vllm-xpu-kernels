// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <ATen/ATen.h>
#include <algorithm>
#include <cstdint>

namespace vllm::qwen38::qsa_sycl {

// ABI: legacy callers always retain [min(M,32),32,512]. Wide scratch is
// opt-in only for long-context prefill; both native entry points preflight it.
inline bool valid_selection_scratch_rows(int64_t m, int64_t max_seq_len,
                                         int64_t scratch_rows) {
  return scratch_rows == std::min<int64_t>(m, 32) ||
      (m >= 128 && max_seq_len >= 4096 && scratch_rows == 128);
}

// All output and scratch storage belongs to the caller. No operation allocates,
// synchronizes the host, or changes persistent cache contents.
at::Tensor group_compress_v2(
    const at::Tensor& raw_keys, const at::Tensor& raw_positions,
    const at::Tensor& compressor_state_cache,
    const at::Tensor& rope_position_cache,
    const at::Tensor& compressor_state_block_table,
    const at::Tensor& token_to_req, const at::Tensor& query_start_loc,
    const at::Tensor& logical_positions, const at::Tensor& compressed_slots,
    at::Tensor pooled, at::Tensor first_positions, int64_t compress_ratio,
    int64_t compressed_capacity, bool historical_ring_proven);

// packed_kv: [pages,1,page_size,512], with K immediately followed by V.
// partials: [min(M,128),H,43,258] fp32, H in {3,6}. Large M is
// processed in C++ chunks after whole-call validation, reusing this scratch.
at::Tensor token_split_attention_v3(
    const at::Tensor& q, const at::Tensor& packed_kv,
    const at::Tensor& logical_indices, const at::Tensor& block_table,
    const at::Tensor& token_to_req, int64_t page_size, at::Tensor output,
    at::Tensor partials);

// Preprocessed indexer query. 32-way selection uses caller-owned ping-pong
// workspaces [min(M,32),32,512], or [128,32,512] for M>=128 and
// max_seq_len>=4096 (FP32 scores and int32 indices). The validated scratch
// first dimension is also the C++ chunk size; no workspace is allocated here.
at::Tensor select_paged_tokens_v2(
    const at::Tensor& q, const at::Tensor& compressed_key_cache,
    const at::Tensor& page_table, const at::Tensor& token_to_req,
    const at::Tensor& query_positions, const at::Tensor& sequence_lengths,
    int64_t compressed_page_size, int64_t max_seq_len, at::Tensor out,
    at::Tensor scores_a, at::Tensor indices_a, at::Tensor scores_b,
    at::Tensor indices_b);

}  // namespace vllm::qwen38::qsa_sycl
