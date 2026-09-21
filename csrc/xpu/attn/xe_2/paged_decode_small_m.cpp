// SPDX-License-Identifier: Apache-2.0
#include "paged_decode_small_m.h"
#include "paged_decode.hpp"

#if __has_include("paged_decode_enabled_policies_gen.hpp")
  #include "paged_decode_enabled_policies_gen.hpp"
#endif

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
    const bool* skip_rows) {
#if __has_include("paged_decode_enabled_policies_gen.hpp")
  // Respect selective builds just like the general decode dispatcher.
  if constexpr (!is_decode_policy_tuple_enabled<
                    decode_policy_q8_h512_p64,
                    false,
                    false,
                    false>::value) {
    return false;
  }
#endif
  // FP16 GQA8 single-query sequences, including small-M verification
  // expanded by the public wrapper. Keep unsupported split plans unchanged.
  if (query.scalar_type() != at::kHalf || key.scalar_type() != at::kHalf ||
      value.scalar_type() != at::kHalf || query.dim() != 3 ||
      query.size(0) < 1 || query.size(0) > 8 ||
      (query.size(1) != 8 && query.size(1) != 16) || query.size(2) != 512 ||
      key.dim() != 4 || key.size(1) != 64 || key.size(2) * 8 != query.size(1) ||
      key.size(3) != 512 || value.sizes() != key.sizes() || max_seqlen_q != 1 ||
      max_seqlen_k < 16384 || max_seqlen_k > 40960 ||
      (num_kv_splits != 8 && num_kv_splits != 16 && num_kv_splits != 32) ||
      query.stride(-1) != 1 || key.stride(-1) != 1 || value.stride(-1) != 1 ||
      out.scalar_type() != at::kHalf || out.sizes() != query.sizes() ||
      !out.is_contiguous() || temporary.scalar_type() != at::kHalf ||
      !temporary.is_contiguous() ||
      temporary.numel() !=
          query.size(0) * query.size(1) * num_kv_splits * 512 ||
      exp_sums.scalar_type() != at::kFloat ||
      max_logits.scalar_type() != at::kFloat || !exp_sums.is_contiguous() ||
      !max_logits.is_contiguous() ||
      exp_sums.numel() != query.size(0) * query.size(1) * num_kv_splits ||
      max_logits.numel() != query.size(0) * query.size(1) * num_kv_splits ||
      block_table.scalar_type() != at::kInt || !block_table.is_contiguous() ||
      block_table.dim() != 2 || block_table.size(0) != query.size(0) ||
      cu_seqlens_q.scalar_type() != at::kInt || !cu_seqlens_q.is_contiguous() ||
      cu_seqlens_q.numel() != query.size(0) + 1 ||
      seqlens_k.scalar_type() != at::kInt || !seqlens_k.is_contiguous() ||
      seqlens_k.numel() != query.size(0)) {
    return false;
  }
  check_paged_kv_cache_strides(key, value);

  paged_decode_args_t args{};
  args.query = query.data_ptr();
  args.key = key.data_ptr();
  args.value = value.data_ptr();
  args.out = out.data_ptr();
  args.tem_out = temporary.data_ptr();
  args.exp_sums = exp_sums.data_ptr();
  args.max_logits = max_logits.data_ptr();
  args.block_table = block_table.data_ptr();
  args.cu_seqlens_q = cu_seqlens_q.data_ptr();
  args.cu_seqlens_k = seqlens_k.data_ptr();
  args.max_queries = 1;
  args.max_keys = max_seqlen_k;
  args.total_seqlen_q = query.size(0);
  args.total_seqlen_k =
      static_cast<int>(get_paged_kv_cache_effective_total_seqlen(key));
  args.sm_scale = static_cast<float>(softmax_scale);
  args.batch_size = query.size(0);
  args.num_heads_q = query.size(1);
  args.num_heads_k = key.size(2);
  args.head_size = 512;
  args.v_head_size = 512;
  args.max_blocks_per_seq = block_table.size(1);
  args.block_size = 64;
  args.window_size_left = max_seqlen_k;
  args.window_size_right = max_seqlen_k;
  args.is_varlen = true;
  args.is_paged = true;
  args.is_prefill = const_cast<bool*>(skip_rows);
  args.num_kv_splits = num_kv_splits;
  args.k_stride_page = key.stride(0);
  args.k_stride_seq = key.stride(1);
  args.k_stride_heads = key.stride(2);
  args.v_stride_page = value.stride(0);
  args.v_stride_seq = value.stride(1);
  args.v_stride_heads = value.stride(2);
  args.q_stride_seq = query.stride(0);
  args.q_stride_heads = query.stride(1);
  args.page_stride_elements =
      static_cast<int>(get_paged_kv_cache_page_stride_elements(key));

  // Tile the V/output dimension in two work-groups. The 256-column tile
  // reduces accumulator and shared-memory requirements; the existing TLA
  // mainloop, softmax, and split reduction retain their public contracts.
  using Policy = decode_policy_q8_h512_p64;
  PagedDecodeConfig<
      Policy::ShapeQK,
      Policy::ShapePV,
      cute::Shape<cute::_8, cute::_256>,
      Policy::SubgroupLayoutQK,
      void,
      1,
      false,
      false,
      false,
      cutlass::half_t,
      cutlass::half_t,
      cutlass::half_t,
      cutlass::half_t>::kernel_dispatch(queue, args);
  return true;
}
