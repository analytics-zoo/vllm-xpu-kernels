// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <ATen/ATen.h>
#include <pybind11/pybind11.h>

#include <tuple>
#include <vector>

namespace vllm::qwen38::qsa_sycl {

// Caller-owned raw/compressed cache row store. A false unique_slots_proven
// keeps duplicate physical slots deterministic in input row order.
at::Tensor store_cache_rows_v3(at::Tensor cache,
                               const at::Tensor& slot_mapping,
                               const at::Tensor& rows,
                               bool unique_slots_proven);

// Mirrors the M=1 non-receipt transaction boundary: false means no kernel
// submission, true means all stores were submitted in order. Exceptions after
// the first submit propagate and must never trigger a replaying fallback.
bool try_store_m1_transaction_v1(
    const std::vector<std::tuple<at::Tensor, at::Tensor, at::Tensor>>& stores);

// Device-side helpers for the Python-owned QSAHistoricalRingLease ledger.
// They do not create/verify lease generations or request incarnations.
at::Tensor compressed_first_positions_v1(
    const at::Tensor& logical_positions, const at::Tensor& slots,
    at::Tensor out, int64_t ratio);

at::Tensor check_history_v1(
    const at::Tensor& tags, const at::Tensor& token_to_req,
    const at::Tensor& query_start_loc, const at::Tensor& logical_positions,
    const at::Tensor& block_table, const at::Tensor& compressed_slots,
    at::Tensor result, int64_t ring_size, int64_t num_blocks,
    int64_t compress_ratio);

at::Tensor acknowledge_history_v1(
    at::Tensor tags, const at::Tensor& slot_mapping,
    const at::Tensor& logical_positions, int64_t capacity,
    bool unique_slots_proven);

at::Tensor reset_history_blocks_v1(
    at::Tensor tags, const at::Tensor& changed_blocks,
    int64_t ring_size, int64_t num_blocks);

// Bind only these verified row-store/transaction entries in the central
// production module. History helpers remain on the existing Triton path.
void bind_qsa_sycl_owner(pybind11::module_& module);

}  // namespace vllm::qwen38::qsa_sycl
