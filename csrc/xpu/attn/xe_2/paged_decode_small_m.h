// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <sycl/sycl.hpp>
#include <torch/all.h>

// Return false before submitting work if the small decode policy is unsuitable.
bool try_paged_decode_small_m_xe2(
    sycl::queue& queue,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    at::Tensor& out,
    at::Tensor& temporary,
    at::Tensor& exp_sums,
    at::Tensor& max_logits,
    const at::Tensor& block_table,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& seqlens_k,
    int max_seqlen_q,
    int max_seqlen_k,
    int num_kv_splits,
    double softmax_scale,
    const bool* skip_rows);
