// SPDX-License-Identifier: Apache-2.0
#include "qwen38/qsa_sycl.h"
#ifdef QWEN38_QSA_STANDALONE
  #include "qwen38/qsa_sycl_aux.h"
#endif

#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/ext/oneapi/experimental/group_load_store.hpp>
#include <sycl/ext/oneapi/experimental/device_architecture.hpp>
#include <sycl/sycl.hpp>
#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <type_traits>

#include <cute/tensor.hpp>
#include <cute/arch/mma_xe.hpp>
#ifdef printf
  #undef printf
#endif

namespace vllm::qwen38::qsa_sycl {
namespace {

using bf16 = sycl::ext::oneapi::bfloat16;
using half = sycl::half;
constexpr int kHeadDim = 256;
constexpr int kSlots = 2051;
constexpr int kPartialStride = kHeadDim + 2;
constexpr int kStoragePartials = 43;
constexpr int kSg = 16;

template <typename T, bool PowerOfTwo>
class CompressKernel;
template <int PageSize, bool BlockRead, int Subgroup, int Prefetch>
class AttentionPhase0;
template <int PageSize, bool BlockRead, int Subgroup, int Prefetch>
class AttentionPhase0MultiKv;
template <int PageSize>
class AttentionPrefillQkDpasPhase0;
template <int Columns, int Prefetch>
class AttentionPhase1;

void check_xpu(
    const at::Tensor& t,
    const at::Tensor& anchor,
    at::ScalarType dtype,
    const char* name) {
  TORCH_CHECK(t.device() == anchor.device(), name, " has the wrong device");
  TORCH_CHECK(t.scalar_type() == dtype, name, " has the wrong dtype");
  TORCH_CHECK(!t.is_neg() && !t.is_conj(), name, " must not be a lazy view");
}

void record_tensors(
    std::initializer_list<const at::Tensor*> tensors,
    c10::xpu::XPUStream stream) {
  for (const auto* tensor : tensors) {
    c10::xpu::XPUCachingAllocator::recordStream(
        tensor->storage().data_ptr(), stream);
  }
}

// This packed allocation is deliberately shared by key and position views.
// Each row contains the 128 half/BF16 key values followed by 3 int64 axes.
bool packed_key_position_views_disjoint(
    const at::Tensor& left, const at::Tensor& right) {
  const at::Tensor* key = nullptr;
  const at::Tensor* positions = nullptr;
  if ((left.scalar_type() == at::kHalf ||
       left.scalar_type() == at::kBFloat16) &&
      right.scalar_type() == at::kLong) {
    key = &left;
    positions = &right;
  } else if (
      (right.scalar_type() == at::kHalf ||
       right.scalar_type() == at::kBFloat16) &&
      left.scalar_type() == at::kLong) {
    key = &right;
    positions = &left;
  } else {
    return false;
  }
  return key->dim() == 4 && positions->dim() == 4 &&
         key->size(0) == positions->size(0) &&
         key->size(1) == positions->size(1) && key->size(0) > 0 &&
         key->size(1) >= 4 && key->size(2) == 1 && positions->size(2) == 1 &&
         key->size(3) == 128 && positions->size(3) == 3 &&
         key->stride(3) == 1 && positions->stride(3) == 1 &&
         key->stride(0) > 0 && key->stride(0) == positions->stride(0) * 4 &&
         key->stride(1) == positions->stride(1) * 4 && key->stride(1) >= 140 &&
         reinterpret_cast<uintptr_t>(positions->data_ptr()) ==
             reinterpret_cast<uintptr_t>(key->data_ptr()) + 256;
}

// Conservative storage-range test; a false negative would make async output
// writes race an input read. An unknown/overflowing stride is treated as an
// overlap. Exact packed key/position views are the one intentional exception.
bool may_overlap(const at::Tensor& left, const at::Tensor& right) {
  if (packed_key_position_views_disjoint(left, right)) {
    return false;
  }
  if (left.numel() == 0 || right.numel() == 0) {
    return false;
  }
  auto range =
      [](const at::Tensor& t, uintptr_t& first, uintptr_t& last) -> bool {
    constexpr auto limit = std::numeric_limits<uintptr_t>::max();
    uintptr_t extent = 0;
    for (int i = 0; i < t.dim(); ++i) {
      if (t.stride(i) < 0 || t.size(i) < 0) {
        return false;
      }
      const auto stride = static_cast<uintptr_t>(t.stride(i));
      const auto count =
          static_cast<uintptr_t>(std::max<int64_t>(t.size(i) - 1, 0));
      if (stride && count > (limit - extent) / stride) {
        return false;
      }
      extent += count * stride;
    }
    const auto item = static_cast<uintptr_t>(t.element_size());
    if (extent >= limit / item) {
      return false;
    }
    const auto bytes = (extent + 1) * item;
    first = reinterpret_cast<uintptr_t>(t.data_ptr());
    if (first > limit - bytes) {
      return false;
    }
    last = first + bytes;
    return true;
  };
  uintptr_t left_first = 0, left_last = 0, right_first = 0, right_last = 0;
  return !range(left, left_first, left_last) ||
         !range(right, right_first, right_last) ||
         (left_first < right_last && right_first < left_last);
}

template <typename Scalar, bool PowerOfTwo>
void launch_compress(
    sycl::queue& queue,
    const at::Tensor& raw_keys,
    const at::Tensor& raw_positions,
    const at::Tensor& ring_keys,
    const at::Tensor& ring_positions,
    const at::Tensor& block_table,
    const at::Tensor& token_to_req,
    const at::Tensor& query_start_loc,
    const at::Tensor& logical_positions,
    const at::Tensor& compressed_slots,
    at::Tensor pooled,
    at::Tensor first_positions,
    int64_t capacity) {
  const auto* raw = static_cast<const Scalar*>(raw_keys.data_ptr());
  const auto* raw_pos = raw_positions.data_ptr<int64_t>();
  const auto* ring = static_cast<const Scalar*>(ring_keys.data_ptr());
  const auto* ring_pos = ring_positions.data_ptr<int64_t>();
  const auto* table = block_table.data_ptr<int32_t>();
  const auto* reqs = token_to_req.data_ptr<int32_t>();
  const auto* starts = query_start_loc.data_ptr<int32_t>();
  const auto* logical = logical_positions.data_ptr<int64_t>();
  const auto* slots = compressed_slots.data_ptr<int64_t>();
  auto* result = static_cast<Scalar*>(pooled.data_ptr());
  auto* first = first_positions.data_ptr<int64_t>();
  const int64_t rows = raw_keys.size(0);
  const int64_t requests = query_start_loc.size(0) - 1;
  const int64_t pages = ring_keys.size(0);
  const int64_t ring_size = ring_keys.size(1);
  const int64_t ring_mask = ring_size - 1;
  const int64_t raw_row_stride = raw_keys.stride(0);
  const int64_t raw_pos_row_stride = raw_positions.stride(0);
  const int64_t ring_page_stride = ring_keys.stride(0);
  const int64_t ring_row_stride = ring_keys.stride(1);
  const int64_t pos_page_stride = ring_positions.stride(0);
  const int64_t pos_row_stride = ring_positions.stride(1);
  const int64_t table_row_stride = block_table.stride(0);
  const int64_t pooled_row_stride = pooled.stride(0);
  const int64_t first_row_stride = first_positions.stride(0);
  queue.parallel_for<CompressKernel<Scalar, PowerOfTwo>>(
      sycl::nd_range<1>(sycl::range<1>(rows * 128), sycl::range<1>(128)),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
        const int64_t row = item.get_group_linear_id();
        const int64_t column = item.get_local_linear_id();
        const int request = reqs[row];
        const bool valid_request = request >= 0 && request < requests;
        const int32_t start = valid_request ? starts[request] : -1;
        const int32_t end = valid_request ? starts[request + 1] : -1;
        const int64_t position = logical[row];
        const int64_t chunk_start = position - (row - start);
        const int block =
            valid_request ? table[request * table_row_stride] : -1;
        const bool history_valid = block >= 0 && block < pages;
        const bool valid = valid_request && start >= 0 && end >= start &&
                           end <= rows && row >= start && row < end &&
                           position >= 3 && slots[row] >= 0 &&
                           slots[row] < capacity;
        float sum = 0.0f;
        if (valid) {
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const int64_t source_position = position - 3 + i;
            if (source_position >= chunk_start) {
              const int64_t raw_row = start + source_position - chunk_start;
              if (raw_row >= start && raw_row < end && raw_row < rows) {
                sum +=
                    static_cast<float>(raw[raw_row * raw_row_stride + column]);
              }
            } else if (history_valid) {
              const int64_t ring_slot = PowerOfTwo
                                            ? (source_position & ring_mask)
                                            : (source_position % ring_size);
              sum += static_cast<float>(
                  ring
                      [block * ring_page_stride + ring_slot * ring_row_stride +
                       column]);
            }
          }
        }
        result[row * pooled_row_stride + column] =
            static_cast<Scalar>(sum * 0.25f);
        if (column < 3) {
          const int64_t first_position = position - 3;
          int64_t value = 0;
          if (valid && first_position >= chunk_start) {
            const int64_t raw_row = start + first_position - chunk_start;
            if (raw_row >= start && raw_row < end && raw_row < rows) {
              value = raw_pos[raw_row * raw_pos_row_stride + column];
            }
          } else if (valid && history_valid) {
            const int64_t ring_slot = PowerOfTwo ? (first_position & ring_mask)
                                                 : (first_position % ring_size);
            value = ring_pos
                [block * pos_page_stride + ring_slot * pos_row_stride + column];
          }
          first[row * first_row_stride + column] = value;
        }
      });
}

template <int PageSize>
sycl::event launch_prefill_qk_dpas_phase0(
    sycl::queue& queue,
    const half* q,
    const half* packed,
    const int32_t* indices,
    const int32_t* table,
    const int32_t* reqs,
    float* partials,
    int rows,
    int table_rows,
    int table_stride,
    int physical_pages,
    int tokens_per_partial,
    int partial_count,
    sycl::event previous,
    bool has_previous) {
  using Mma = cute::XE_DPAS_TT<4, float, cute::half_t>;
  static_assert(Mma::K == 16 && kHeadDim % Mma::K == 0);
  constexpr int total_heads = 6;
  constexpr int heads = 3;
  constexpr int head_groups = total_heads / heads;
  constexpr int subgroups = 4;
  constexpr int values_per_lane = kHeadDim / kSg;
  constexpr float log2_e = 1.4426950408889634f;
  const int64_t tasks =
      static_cast<int64_t>(rows) * partial_count * head_groups;
  const int64_t groups = (tasks + subgroups - 1) / subgroups;
  return queue.submit([&](sycl::handler& cgh) {
    if (has_previous) cgh.depends_on(previous);
    cgh.parallel_for<AttentionPrefillQkDpasPhase0<PageSize>>(
        sycl::nd_range<1>(
            sycl::range<1>(groups * subgroups * kSg),
            sycl::range<1>(subgroups * kSg)),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(kSg)]] {
          const auto sg = item.get_sub_group();
          const int lane = int(sg.get_local_linear_id());
          const int64_t task = int64_t(item.get_group_linear_id()) * subgroups +
                               int(item.get_local_linear_id()) / kSg;
          // 每个子组独占一组头的一个 partial；尾部子组独立退出，无组间
          // barrier。
          if (task >= tasks) return;
          const int head_base = int(task % head_groups) * heads;
          const int64_t row_partial = task / head_groups;
          const int row = int(row_partial / partial_count);
          const int partial = int(row_partial % partial_count);
          const int request = reqs[row];
          const bool valid_request = request >= 0 && request < table_rows;
          const int32_t* row_indices = indices + int64_t(row) * kSlots;
          const half* row_q =
              q + (int64_t(row) * total_heads + head_base) * kHeadDim;
          float maximum[heads];
          float denominator[heads] = {};
          float output[heads][values_per_lane] = {};
#pragma unroll
          for (int h = 0; h < heads; ++h)
            maximum[h] = -1.0e30f;
          namespace sx = sycl::ext::oneapi::experimental;
          constexpr auto props = sx::properties{
              sx::data_placement_striped,
              sx::contiguous_memory,
              sx::full_group};

          for (int tile = 0; tile < tokens_per_partial; tile += kSg) {
            const int slot = partial * tokens_per_partial + tile + lane;
            const int logical =
                tile + lane < tokens_per_partial && slot < kSlots
                    ? row_indices[slot]
                    : -1;
            const int logical_page = logical >= 0 ? logical / PageSize : -1;
            const bool table_ok = valid_request && logical_page >= 0 &&
                                  logical_page < table_stride;
            const int physical =
                table_ok ? table[int64_t(request) * table_stride + logical_page]
                         : -1;
            const int valid = physical >= 0 && physical < physical_pages;
            // 空tile不贡献任何softmax质量，也不需执行QK或重缩放已有PV。
            if (!sycl::any_of_group(sg, valid != 0)) continue;
            const int64_t base =
                valid
                    ? (int64_t(physical) * PageSize + logical % PageSize) * 512
                    : 0;
            const int token0 = lane / 2;
            const int token1 = token0 + 8;
            const int valid0 = sycl::select_from_group(sg, valid, token0);
            const int valid1 = sycl::select_from_group(sg, valid, token1);
            const int64_t base0 = sycl::select_from_group(sg, base, token0);
            const int64_t base1 = sycl::select_from_group(sg, base, token1);
            const int parity = lane & 1;
            typename Mma::DVector scores = {};
            for (int kb = 0; kb < kHeadDim; kb += Mma::K) {
              typename Mma::AVector a_frag;
              typename Mma::BVector b_frag;
#pragma unroll
              for (int h = 0; h < 4; ++h) {
                const half value =
                    h < heads ? row_q[h * kHeadDim + kb + lane] : half(0);
                a_frag[h] = sycl::bit_cast<uint16_t>(value);
              }
          // 沿用 HC probe 已验证的 B fragment：两列交错，lane 奇偶选 K。
#pragma unroll
              for (int j = 0; j < 8; ++j) {
                const int k = kb + 2 * j + parity;
                const half key0 = valid0 ? packed[base0 + k] : half(0);
                const half key1 = valid1 ? packed[base1 + k] : half(0);
                b_frag[2 * j] = sycl::bit_cast<uint16_t>(key0);
                b_frag[2 * j + 1] = sycl::bit_cast<uint16_t>(key1);
              }
              Mma::fma(scores, a_frag, b_frag, scores);
            }

            float probability[heads];
#pragma unroll
            for (int h = 0; h < heads; ++h) {
              const float score = valid ? scores[h] * 0.0625f : -1.0e30f;
              const float tile_max =
                  sycl::reduce_over_group(sg, score, sycl::maximum<float>());
              const float next_max = sycl::fmax(maximum[h], tile_max);
              const float rescale =
                  sycl::native::exp2((maximum[h] - next_max) * log2_e);
              // 全空 tile 的质量必须为零，不能把哨兵之间的 exp(0) 算入分母。
              probability[h] =
                  valid ? sycl::native::exp2((score - next_max) * log2_e)
                        : 0.0f;
              const float tile_sum = sycl::reduce_over_group(
                  sg, probability[h], sycl::plus<float>());
              denominator[h] = denominator[h] * rescale + tile_sum;
#pragma unroll
              for (int d = 0; d < values_per_lane; ++d)
                output[h][d] *= rescale;
              maximum[h] = next_max;
            }

            for (int token = 0; token < kSg; ++token) {
              const int token_valid = sycl::select_from_group(sg, valid, token);
              if (!token_valid) continue;
              const int64_t value_base =
                  sycl::select_from_group(sg, base, token) + kHeadDim;
              // 广播后的地址在子组内一致，V 位值仅加载一次共用本组的三或六头。
              sycl::vec<uint32_t, values_per_lane / 2> value_words;
              sx::group_load(
                  sg,
                  sycl::address_space_cast<
                      sycl::access::address_space::global_space,
                      sycl::access::decorated::yes>(
                      reinterpret_cast<const uint32_t*>(packed + value_base)),
                  value_words,
                  props);
              float token_probability[heads];
#pragma unroll
              for (int h = 0; h < heads; ++h)
                token_probability[h] =
                    sycl::select_from_group(sg, probability[h], token);
#pragma unroll
              for (int d = 0; d < values_per_lane; ++d) {
                const uint32_t bits = value_words[d / 2];
                const float value = static_cast<float>(sycl::bit_cast<half>(
                    uint16_t((d & 1) ? bits >> 16 : bits)));
#pragma unroll
                for (int h = 0; h < heads; ++h)
                  output[h][d] =
                      sycl::fma(token_probability[h], value, output[h][d]);
              }
            }
          }

#pragma unroll
          for (int h = 0; h < heads; ++h) {
            const int64_t offset =
                ((int64_t(row) * total_heads + head_base + h) *
                     kStoragePartials +
                 partial) *
                kPartialStride;
            if (lane == 0) {
              partials[offset] = maximum[h];
              partials[offset + 1] = denominator[h];
            }
#pragma unroll
            for (int d = 0; d < values_per_lane; ++d) {
              const int column = 2 * (lane + kSg * (d / 2)) + (d & 1);
              partials[offset + 2 + column] = output[h][d];
            }
          }
        });
  });
}

template <
    int PageSize,
    bool BlockRead,
    int Subgroup,
    int Prefetch = 1,
    int KvHeads = 1>
sycl::event launch_phase0(
    sycl::queue& queue,
    const half* q,
    const half* packed,
    const int32_t* indices,
    const int32_t* table,
    const int32_t* reqs,
    float* partials,
    int rows,
    int heads,
    int table_rows,
    int table_stride,
    int physical_pages,
    int tokens_per_partial,
    int partial_count,
    sycl::event previous,
    bool has_previous) {
  static_assert(KvHeads == 1 || KvHeads == 2);
  using Kernel = std::conditional_t<
      KvHeads == 1,
      AttentionPhase0<PageSize, BlockRead, Subgroup, Prefetch>,
      AttentionPhase0MultiKv<PageSize, BlockRead, Subgroup, Prefetch>>;
  constexpr int workers = 4;
  constexpr int values_per_lane = kHeadDim / Subgroup;
  const int64_t groups = static_cast<int64_t>(rows) * heads * partial_count;
  return queue.submit([&](sycl::handler& cgh) {
    if (has_previous) cgh.depends_on(previous);
    sycl::local_accessor<float, 1> scratch(
        sycl::range<1>(workers * (kPartialStride)), cgh);
    cgh.parallel_for<Kernel>(
        sycl::nd_range<1>(
            sycl::range<1>(groups * Subgroup * workers),
            sycl::range<1>(Subgroup * workers)),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(Subgroup)]] {
          const int worker = item.get_local_linear_id() / Subgroup;
          const int lane = item.get_local_linear_id() % Subgroup;
          const int group = item.get_group_linear_id();
          const int partial = group % partial_count;
          const int head = (group / partial_count) % heads;
          const int row = group / (partial_count * heads);
          // Q12/KV2 has six query heads per KV group. Keep KV1 address
          // arithmetic compiled exactly as the original Q3/Q6 kernel.
          const int kv_head = KvHeads == 1 ? 0 : head / 6;
          const int request = reqs[row];
          const bool valid_request = request >= 0 && request < table_rows;
          const int32_t* row_indices = indices + row * kSlots;
          const half* row_q = q + (row * heads + head) * kHeadDim;
          float query[values_per_lane];
          namespace sx = sycl::ext::oneapi::experimental;
          constexpr auto props = sx::properties{
              sx::data_placement_striped,
              sx::contiguous_memory,
              sx::full_group};
          if constexpr (BlockRead) {
            sycl::vec<uint32_t, values_per_lane / 2> query_words;
            sx::group_load(
                item.get_sub_group(),
                sycl::address_space_cast<
                    sycl::access::address_space::global_space,
                    sycl::access::decorated::yes>(
                    reinterpret_cast<const uint32_t*>(row_q)),
                query_words,
                props);
#pragma unroll
            for (int d = 0; d < values_per_lane / 2; ++d) {
              query[d * 2] = static_cast<float>(
                  sycl::bit_cast<half>(uint16_t(query_words[d])));
              query[d * 2 + 1] = static_cast<float>(
                  sycl::bit_cast<half>(uint16_t(query_words[d] >> 16)));
            }
          } else {
#pragma unroll
            for (int d = 0; d < values_per_lane; ++d) {
              query[d] = static_cast<float>(row_q[lane * values_per_lane + d]);
            }
          }
          float output[values_per_lane] = {};
          float maximum = -1.0e30f;
          float denominator = 0.0f;
          const int per_worker = tokens_per_partial / workers;
          // Prefetch changes only read scheduling; every head keeps the same
          // token, FP32 dot/reduce, online-softmax and partial-merge order.
          if constexpr (Prefetch == 1) {
            for (int t = 0; t < per_worker; ++t) {
              const int slot =
                  partial * tokens_per_partial + worker * per_worker + t;
              const int logical = slot < kSlots ? row_indices[slot] : -1;
              const int logical_page = logical >= 0 ? logical / PageSize : -1;
              const bool table_ok = valid_request && logical_page >= 0 &&
                                    logical_page < table_stride;
              const int physical =
                  table_ok ? table[request * table_stride + logical_page] : -1;
              const bool valid = physical >= 0 && physical < physical_pages;
              if (!valid) {
                continue;
              }
              const int64_t base =
                  ((static_cast<int64_t>(physical) * KvHeads + kv_head) *
                       PageSize +
                   (logical % PageSize)) *
                  512;
              const half* kv = packed + base;
              float dot = 0.0f;
              sycl::vec<uint32_t, values_per_lane / 2> key_words;
              sycl::vec<uint32_t, values_per_lane / 2> value_words;
              if constexpr (BlockRead) {
                sx::group_load(
                    item.get_sub_group(),
                    sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(
                        reinterpret_cast<const uint32_t*>(kv)),
                    key_words,
                    props);
                sx::group_load(
                    item.get_sub_group(),
                    sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(
                        reinterpret_cast<const uint32_t*>(kv + 256)),
                    value_words,
                    props);
              }
#pragma unroll
              for (int d = 0; d < values_per_lane; ++d) {
                float key;
                if constexpr (BlockRead) {
                  const uint32_t bits = key_words[d / 2];
                  key = static_cast<float>(sycl::bit_cast<half>(
                      uint16_t((d & 1) ? bits >> 16 : bits)));
                } else {
                  key = static_cast<float>(kv[lane * values_per_lane + d]);
                }
                dot += query[d] * key;
              }
              dot = sycl::reduce_over_group(
                  item.get_sub_group(), dot, sycl::plus<float>());
              const float score = dot * 0.0625f;
              const float next_max = sycl::fmax(maximum, score);
              constexpr float log2_e = 1.4426950408889634f;
              const float rescale =
                  sycl::native::exp2((maximum - next_max) * log2_e);
              const float probability =
                  sycl::native::exp2((score - next_max) * log2_e);
              denominator = denominator * rescale + probability;
#pragma unroll
              for (int d = 0; d < values_per_lane; ++d) {
                float value;
                if constexpr (BlockRead) {
                  const uint32_t bits = value_words[d / 2];
                  value = static_cast<float>(sycl::bit_cast<half>(
                      uint16_t((d & 1) ? bits >> 16 : bits)));
                } else {
                  value =
                      static_cast<float>(kv[256 + lane * values_per_lane + d]);
                }
                output[d] = output[d] * rescale + probability * value;
              }
              maximum = next_max;
            }
          } else {
            static_assert(Prefetch == 2 && BlockRead && Subgroup == 32);
            // Cache each worker's index/page vector once; only metadata access
            // changes. A holed list keeps all valid tokens in original order.
            const auto sg = item.get_sub_group();
            const int first_slot =
                partial * tokens_per_partial + worker * per_worker;
            const int cached_logical =
                lane < per_worker && first_slot + lane < kSlots
                    ? row_indices[first_slot + lane]
                    : -1;
            const int cached_page =
                cached_logical >= 0 ? cached_logical / PageSize : -1;
            const bool cached_table_ok =
                valid_request && cached_page >= 0 && cached_page < table_stride;
            const int cached_physical =
                cached_table_ok ? table[request * table_stride + cached_page]
                                : -1;
            const bool cached_valid =
                cached_physical >= 0 && cached_physical < physical_pages;
            const int last_live = sycl::reduce_over_group(
                sg, cached_valid ? lane : -1, sycl::maximum<int>());

            static_assert(BlockRead);
            for (int first = 0; first <= last_live; first += Prefetch) {
              sycl::vec<uint32_t, values_per_lane / 2> key_batch[Prefetch],
                  value_batch[Prefetch];
              int64_t bases[Prefetch] = {};
              bool valid_batch[Prefetch] = {};
#pragma unroll
              for (int b = 0; b < Prefetch; ++b) {
                const int t = first + b;
                if (t >= per_worker) continue;
                const int logical =
                    sycl::select_from_group(sg, cached_logical, t);
                const int physical =
                    sycl::select_from_group(sg, cached_physical, t);
                const bool valid = physical >= 0 && physical < physical_pages;
                valid_batch[b] = valid;
                if (!valid) {
                  continue;
                }
                const int64_t base =
                    ((static_cast<int64_t>(physical) * KvHeads + kv_head) *
                         PageSize +
                     (logical % PageSize)) *
                    512;
                bases[b] = base;
                const half* kv = packed + base;
                if constexpr (BlockRead) {
                  sx::group_load(
                      item.get_sub_group(),
                      sycl::address_space_cast<
                          sycl::access::address_space::global_space,
                          sycl::access::decorated::yes>(
                          reinterpret_cast<const uint32_t*>(kv)),
                      key_batch[b],
                      props);
                  sx::group_load(
                      item.get_sub_group(),
                      sycl::address_space_cast<
                          sycl::access::address_space::global_space,
                          sycl::access::decorated::yes>(
                          reinterpret_cast<const uint32_t*>(kv + 256)),
                      value_batch[b],
                      props);
                }
              }
#pragma unroll
              for (int b = 0; b < Prefetch; ++b) {
                if (!valid_batch[b]) continue;
                const half* kv = packed + bases[b];
                const auto key_words = key_batch[b];
                const auto value_words = value_batch[b];
                float dot = 0.0f;
#pragma unroll
                for (int d = 0; d < values_per_lane; ++d) {
                  float key;
                  if constexpr (BlockRead) {
                    const uint32_t bits = key_words[d / 2];
                    key = static_cast<float>(sycl::bit_cast<half>(
                        uint16_t((d & 1) ? bits >> 16 : bits)));
                  } else {
                    key = static_cast<float>(kv[lane * values_per_lane + d]);
                  }
                  dot += query[d] * key;
                }
                dot = sycl::reduce_over_group(
                    item.get_sub_group(), dot, sycl::plus<float>());
                const float score = dot * 0.0625f;
                const float next_max = sycl::fmax(maximum, score);
                constexpr float log2_e = 1.4426950408889634f;
                const float rescale =
                    sycl::native::exp2((maximum - next_max) * log2_e);
                const float probability =
                    sycl::native::exp2((score - next_max) * log2_e);
                denominator = denominator * rescale + probability;
#pragma unroll
                for (int d = 0; d < values_per_lane; ++d) {
                  float value;
                  if constexpr (BlockRead) {
                    const uint32_t bits = value_words[d / 2];
                    value = static_cast<float>(sycl::bit_cast<half>(
                        uint16_t((d & 1) ? bits >> 16 : bits)));
                  } else {
                    value = static_cast<float>(
                        kv[256 + lane * values_per_lane + d]);
                  }
                  output[d] = output[d] * rescale + probability * value;
                }
                maximum = next_max;
              }
            }
          }
          const int local_offset = worker * kPartialStride;
          if (lane == 0) {
            scratch[local_offset] = maximum;
            scratch[local_offset + 1] = denominator;
          }
#pragma unroll
          for (int d = 0; d < values_per_lane; ++d) {
            const int column = BlockRead
                                   ? 2 * (lane + Subgroup * (d / 2)) + (d & 1)
                                   : lane * values_per_lane + d;
            scratch[local_offset + 2 + column] = output[d];
          }
          item.barrier(sycl::access::fence_space::local_space);
          if (worker != 0) {
            return;
          }
          float all_max = -1.0e30f;
#pragma unroll
          for (int w = 0; w < workers; ++w) {
            if (scratch[w * kPartialStride + 1] > 0.0f) {
              all_max = sycl::fmax(all_max, scratch[w * kPartialStride]);
            }
          }
          float all_sum = 0.0f;
          float merged[values_per_lane] = {};
#pragma unroll
          for (int w = 0; w < workers; ++w) {
            const int src = w * kPartialStride;
            if (scratch[src + 1] == 0.0f) {
              continue;
            }
            const float factor = sycl::native::exp2(
                (scratch[src] - all_max) * 1.4426950408889634f);
            all_sum += factor * scratch[src + 1];
#pragma unroll
            for (int d = 0; d < values_per_lane; ++d) {
              merged[d] +=
                  factor * scratch[src + 2 + lane * values_per_lane + d];
            }
          }
          const int64_t offset =
              ((static_cast<int64_t>(row) * heads + head) * kStoragePartials +
               partial) *
              kPartialStride;
          if (lane == 0) {
            partials[offset] = all_max;
            partials[offset + 1] = all_sum;
          }
#pragma unroll
          for (int d = 0; d < values_per_lane; ++d) {
            partials[offset + 2 + lane * values_per_lane + d] = merged[d];
          }
        });
  });
}

template <int Columns, int Prefetch = 1>
sycl::event launch_phase1_columns(
    sycl::queue& queue,
    sycl::event phase0,
    const float* partials,
    half* output,
    int rows,
    int heads,
    int partial_count) {
  constexpr int merge_wg = 64;
  constexpr int tiles = kHeadDim / Columns;
  constexpr int values = Columns / merge_wg;
  const int64_t groups = static_cast<int64_t>(rows) * heads * tiles;
  return queue.submit([&](sycl::handler& cgh) {
    cgh.depends_on(phase0);
    sycl::local_accessor<float, 1> weights(
        sycl::range<1>(kStoragePartials), cgh);
    cgh.parallel_for<AttentionPhase1<Columns, Prefetch>>(
        sycl::nd_range<1>(
            sycl::range<1>(groups * merge_wg), sycl::range<1>(merge_wg)),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(kSg)]] {
          const int lane = item.get_local_linear_id();
          const int group = item.get_group_linear_id();
          const int head = group / tiles;
          const int tile = group % tiles;
          const float* head_partials = partials + static_cast<int64_t>(head) *
                                                      kStoragePartials *
                                                      kPartialStride;
          const bool live = lane < partial_count &&
                            head_partials[lane * kPartialStride + 1] > 0.0f;
          const float local_max =
              live ? head_partials[lane * kPartialStride] : -1.0e30f;
          const float maximum = sycl::reduce_over_group(
              item.get_group(), local_max, sycl::maximum<float>());
          const float factor = live ? sycl::exp(local_max - maximum) : 0.0f;
          const float local_sum =
              live ? factor * head_partials[lane * kPartialStride + 1] : 0.0f;
          const float denominator = sycl::reduce_over_group(
              item.get_group(), local_sum, sycl::plus<float>());
          if (lane < partial_count) {
            weights[lane] = factor;
          }
          item.barrier(sycl::access::fence_space::local_space);
          float accumulation[values] = {};
          if constexpr (Prefetch == 1) {
            for (int p = 0; p < partial_count; ++p) {
              const float* part = head_partials + p * kPartialStride;
#pragma unroll
              for (int d = 0; d < values; ++d) {
                accumulation[d] +=
                    weights[p] * part[2 + tile * Columns + lane * values + d];
              }
            }
          } else {
            for (int first = 0; first < partial_count; first += Prefetch) {
              float factors[Prefetch] = {};
              float loaded[Prefetch][values] = {};
#pragma unroll
              for (int b = 0; b < Prefetch; ++b) {
                const int p = first + b;
                if (p >= partial_count) continue;
                factors[b] = weights[p];
                const float* part = head_partials + p * kPartialStride;
#pragma unroll
                for (int d = 0; d < values; ++d)
                  loaded[b][d] = part[2 + tile * Columns + lane * values + d];
              }
#pragma unroll
              for (int b = 0; b < Prefetch; ++b) {
                if (first + b >= partial_count) continue;
#pragma unroll
                for (int d = 0; d < values; ++d)
                  accumulation[d] += factors[b] * loaded[b][d];
              }
            }
          }
          const float inv = denominator > 0.0f ? 1.0f / denominator : 0.0f;
#pragma unroll
          for (int d = 0; d < values; ++d) {
            output[head * kHeadDim + tile * Columns + lane * values + d] =
                static_cast<half>(accumulation[d] * inv);
          }
        });
  });
}

sycl::event launch_phase1(
    sycl::queue& queue,
    sycl::event phase0,
    const float* partials,
    half* output,
    int rows,
    int heads,
    int partial_count,
    bool m1_scope) {
  // Column ownership changes only; the reduction and per-column FP32 order
  // stay identical. A one-row tail of a larger batch is not the M1 path.
  const char* tile = std::getenv("VLLM_XPU_QWEN38_QSA_M1_MERGE_TILE");
  if (m1_scope && !tile)
    return launch_phase1_columns<64, 4>(
        queue, phase0, partials, output, rows, heads, partial_count);
  if (m1_scope && tile && std::strcmp(tile, "64") == 0)
    return launch_phase1_columns<64>(
        queue, phase0, partials, output, rows, heads, partial_count);
  if (m1_scope && (!tile || std::strcmp(tile, "128") == 0))
    return launch_phase1_columns<128>(
        queue, phase0, partials, output, rows, heads, partial_count);
  return launch_phase1_columns<256>(
      queue, phase0, partials, output, rows, heads, partial_count);
}

}  // namespace

at::Tensor group_compress_v2(
    const at::Tensor& raw_keys,
    const at::Tensor& raw_positions,
    const at::Tensor& compressor_state_cache,
    const at::Tensor& rope_position_cache,
    const at::Tensor& compressor_state_block_table,
    const at::Tensor& token_to_req,
    const at::Tensor& query_start_loc,
    const at::Tensor& logical_positions,
    const at::Tensor& compressed_slots,
    at::Tensor pooled,
    at::Tensor first_positions,
    int64_t compress_ratio,
    int64_t compressed_capacity,
    bool historical_ring_proven) {
  TORCH_CHECK(raw_keys.is_xpu(), "raw_keys must be XPU");
  TORCH_CHECK(
      historical_ring_proven,
      "group compression requires historical ring proof");
  TORCH_CHECK(
      compress_ratio == 4 && compressed_capacity > 0,
      "group compression requires ratio=4 and positive capacity");
  TORCH_CHECK(
      raw_keys.scalar_type() == at::kHalf ||
          raw_keys.scalar_type() == at::kBFloat16,
      "raw_keys must be FP16 or BF16");
  check_xpu(raw_keys, raw_keys, raw_keys.scalar_type(), "raw_keys");
  check_xpu(raw_positions, raw_keys, at::kLong, "raw_positions");
  check_xpu(
      compressor_state_cache,
      raw_keys,
      raw_keys.scalar_type(),
      "compressor_state_cache");
  check_xpu(rope_position_cache, raw_keys, at::kLong, "rope_position_cache");
  check_xpu(
      compressor_state_block_table,
      raw_keys,
      at::kInt,
      "compressor_state_block_table");
  check_xpu(token_to_req, raw_keys, at::kInt, "token_to_req");
  check_xpu(query_start_loc, raw_keys, at::kInt, "query_start_loc");
  check_xpu(logical_positions, raw_keys, at::kLong, "logical_positions");
  check_xpu(compressed_slots, raw_keys, at::kLong, "compressed_slots");
  check_xpu(pooled, raw_keys, raw_keys.scalar_type(), "pooled");
  check_xpu(first_positions, raw_keys, at::kLong, "first_positions");
  TORCH_CHECK(
      raw_keys.dim() == 3 && raw_keys.size(1) == 1 && raw_keys.size(2) == 128 &&
          raw_keys.stride(2) == 1,
      "raw_keys must be [M,1,128] with inner stride 1");
  const int64_t rows = raw_keys.size(0);
  TORCH_CHECK(
      raw_positions.sizes() == at::IntArrayRef({rows, 1, 3}) &&
          raw_positions.is_contiguous() && raw_positions.stride(2) == 1,
      "raw_positions must be contiguous [M,1,3]");
  TORCH_CHECK(
      compressor_state_cache.dim() == 4 && compressor_state_cache.size(0) > 0 &&
          compressor_state_cache.size(1) >= 4 &&
          compressor_state_cache.size(2) == 1 &&
          compressor_state_cache.size(3) == 128 &&
          compressor_state_cache.stride(3) == 1,
      "compressor_state_cache must be [pages,ring>=4,1,128]");
  TORCH_CHECK(
      rope_position_cache.sizes() == at::IntArrayRef(
                                         {compressor_state_cache.size(0),
                                          compressor_state_cache.size(1),
                                          1,
                                          3}) &&
          rope_position_cache.stride(3) == 1,
      "rope_position_cache must match [pages,ring,1,3]");
  TORCH_CHECK(
      compressor_state_block_table.dim() == 2 &&
          compressor_state_block_table.size(0) > 0 &&
          compressor_state_block_table.size(1) > 0 &&
          compressor_state_block_table.is_contiguous(),
      "compressor_state_block_table must be contiguous 2D");
  TORCH_CHECK(
      query_start_loc.dim() == 1 && query_start_loc.numel() >= 2 &&
          query_start_loc.is_contiguous() &&
          compressor_state_block_table.size(0) >= query_start_loc.numel() - 1,
      "query_start_loc must be contiguous [requests+1]");
  TORCH_CHECK(
      token_to_req.sizes() == at::IntArrayRef({rows}) &&
          token_to_req.is_contiguous() &&
          logical_positions.sizes() == at::IntArrayRef({rows}) &&
          logical_positions.is_contiguous() &&
          compressed_slots.sizes() == at::IntArrayRef({rows}) &&
          compressed_slots.is_contiguous(),
      "row metadata must be contiguous [M]");
  TORCH_CHECK(
      pooled.sizes() == raw_keys.sizes() && pooled.is_contiguous() &&
          first_positions.sizes() == at::IntArrayRef({rows, 3}) &&
          first_positions.is_contiguous(),
      "caller outputs must be contiguous [M,1,128] and [M,3]");
  // The cache sidecar may share a packed storage allocation, but its byte
  // intervals may not overlap the key rows. All output aliases are rejected.
  TORCH_CHECK(
      !may_overlap(compressor_state_cache, rope_position_cache),
      "key and position cache views overlap");
  const std::array<const at::Tensor*, 9> inputs = {
      &raw_keys,
      &raw_positions,
      &compressor_state_cache,
      &rope_position_cache,
      &compressor_state_block_table,
      &token_to_req,
      &query_start_loc,
      &logical_positions,
      &compressed_slots};
  for (const auto* input : inputs) {
    TORCH_CHECK(
        !may_overlap(pooled, *input) && !may_overlap(first_positions, *input),
        "group compression output aliases an input");
  }
  TORCH_CHECK(
      !may_overlap(pooled, first_positions),
      "group compression outputs overlap");
  TORCH_CHECK(
      !may_overlap(raw_keys, compressor_state_cache) &&
          !may_overlap(raw_positions, compressor_state_cache),
      "raw input aliases compressor state");
  if (rows == 0) {
    return pooled;
  }
  TORCH_CHECK(
      rows <= std::numeric_limits<int32_t>::max() / 128,
      "too many compression rows");
  c10::OptionalDeviceGuard guard(raw_keys.device());
  auto stream = c10::xpu::getCurrentXPUStream(raw_keys.get_device());
  auto& queue = stream.queue();
  record_tensors(
      {&raw_keys,
       &raw_positions,
       &compressor_state_cache,
       &rope_position_cache,
       &compressor_state_block_table,
       &token_to_req,
       &query_start_loc,
       &logical_positions,
       &compressed_slots,
       &pooled,
       &first_positions},
      stream);
  const auto ring_size = compressor_state_cache.size(1);
  const bool power_of_two = (ring_size & (ring_size - 1)) == 0;
  if (raw_keys.scalar_type() == at::kHalf) {
    if (power_of_two) {
      launch_compress<half, true>(
          queue,
          raw_keys,
          raw_positions,
          compressor_state_cache,
          rope_position_cache,
          compressor_state_block_table,
          token_to_req,
          query_start_loc,
          logical_positions,
          compressed_slots,
          pooled,
          first_positions,
          compressed_capacity);
    } else {
      launch_compress<half, false>(
          queue,
          raw_keys,
          raw_positions,
          compressor_state_cache,
          rope_position_cache,
          compressor_state_block_table,
          token_to_req,
          query_start_loc,
          logical_positions,
          compressed_slots,
          pooled,
          first_positions,
          compressed_capacity);
    }
  } else if (power_of_two) {
    launch_compress<bf16, true>(
        queue,
        raw_keys,
        raw_positions,
        compressor_state_cache,
        rope_position_cache,
        compressor_state_block_table,
        token_to_req,
        query_start_loc,
        logical_positions,
        compressed_slots,
        pooled,
        first_positions,
        compressed_capacity);
  } else {
    launch_compress<bf16, false>(
        queue,
        raw_keys,
        raw_positions,
        compressor_state_cache,
        rope_position_cache,
        compressor_state_block_table,
        token_to_req,
        query_start_loc,
        logical_positions,
        compressed_slots,
        pooled,
        first_positions,
        compressed_capacity);
  }
  return pooled;
}

at::Tensor token_split_attention_v3(
    const at::Tensor& q,
    const at::Tensor& packed_kv,
    const at::Tensor& logical_indices,
    const at::Tensor& block_table,
    const at::Tensor& token_to_req,
    int64_t page_size,
    at::Tensor output,
    at::Tensor partials) {
  TORCH_CHECK(q.is_xpu(), "q must be XPU");
  check_xpu(q, q, at::kHalf, "q");
  check_xpu(packed_kv, q, at::kHalf, "packed_kv");
  check_xpu(logical_indices, q, at::kInt, "logical_indices");
  check_xpu(block_table, q, at::kInt, "block_table");
  check_xpu(token_to_req, q, at::kInt, "token_to_req");
  check_xpu(output, q, at::kHalf, "output");
  check_xpu(partials, q, at::kFloat, "partials");
  TORCH_CHECK(
      q.scalar_type() == at::kHalf && q.dim() == 3 && q.size(0) >= 1 &&
          q.size(0) <= 4096 &&
          (q.size(1) == 3 || q.size(1) == 6 || q.size(1) == 12) &&
          q.size(2) == kHeadDim && q.is_contiguous(),
      "q must be contiguous FP16 [1..4096,3|6|12,256]");
  TORCH_CHECK(
      page_size == 256 || page_size == 512, "page_size must be 256 or 512");
  TORCH_CHECK(
      packed_kv.dim() == 4 && packed_kv.size(0) > 0 &&
          packed_kv.size(1) == (q.size(1) == 12 ? 2 : 1) &&
          packed_kv.size(2) == page_size && packed_kv.size(3) == 512 &&
          packed_kv.stride(0) == packed_kv.size(1) * page_size * 512 &&
          (packed_kv.size(1) == 1 || packed_kv.stride(1) == page_size * 512) &&
          packed_kv.stride(2) == 512 && packed_kv.stride(3) == 1,
      "packed_kv must expose [pages,KV,page_size,512], KV=2 for Q12");
  TORCH_CHECK(
      logical_indices.sizes() == at::IntArrayRef({q.size(0), kSlots}) &&
          logical_indices.is_contiguous(),
      "logical_indices must be contiguous [M,2051]");
  // 页表是最长上下文的预分配容量；设备端另行校验实际物理页号。
  TORCH_CHECK(
      block_table.dim() == 2 && block_table.size(0) >= 1 &&
          block_table.size(1) >= 1 &&
          block_table.size(0) <=
              std::numeric_limits<int32_t>::max() / block_table.size(1) &&
          block_table.is_contiguous(),
      "block_table must be contiguous [requests,pages]");
  TORCH_CHECK(
      token_to_req.sizes() == at::IntArrayRef({q.size(0)}) &&
          token_to_req.is_contiguous(),
      "token_to_req must be contiguous [M]");
  TORCH_CHECK(
      output.sizes() == q.sizes() && output.is_contiguous(),
      "output must be caller-owned contiguous [M,H,256]");
  TORCH_CHECK(
      partials.sizes() == at::IntArrayRef(
                              {std::min<int64_t>(q.size(0), 128),
                               q.size(1),
                               kStoragePartials,
                               kPartialStride}) &&
          partials.is_contiguous(),
      "partials must be caller-owned contiguous [min(M,128),H,43,258]");
  for (const auto* in :
       {&q, &packed_kv, &logical_indices, &block_table, &token_to_req}) {
    TORCH_CHECK(
        !may_overlap(output, *in) && !may_overlap(partials, *in),
        "attention output/workspace aliases an input");
  }
  TORCH_CHECK(
      !may_overlap(output, partials), "attention output and workspace overlap");
  c10::OptionalDeviceGuard guard(q.device());
  auto stream = c10::xpu::getCurrentXPUStream(q.get_device());
  auto& queue = stream.queue();
  TORCH_CHECK(
      queue.is_in_order(),
      "attention scratch reuse requires an in-order current stream");
  // A later phase/chunk can throw after earlier work was submitted. Record
  // every allocation first so exception unwinding cannot recycle live USM.
  record_tensors(
      {&q,
       &packed_kv,
       &logical_indices,
       &block_table,
       &token_to_req,
       &output,
       &partials},
      stream);
  const int total_rows = static_cast<int>(q.size(0));
  const int heads = static_cast<int>(q.size(1));
  auto* partial_ptr = partials.data_ptr<float>();
  const auto* q_ptr = static_cast<const half*>(q.data_ptr());
  const auto* packed_ptr = static_cast<const half*>(packed_kv.data_ptr());
  const auto* indices_ptr = logical_indices.data_ptr<int32_t>();
  const auto* table_ptr = block_table.data_ptr<int32_t>();
  const auto* req_ptr = token_to_req.data_ptr<int32_t>();
  auto* output_ptr = static_cast<half*>(output.data_ptr());
  const bool aligned = ((reinterpret_cast<uintptr_t>(q_ptr) |
                         reinterpret_cast<uintptr_t>(packed_ptr)) &
                        3) == 0;
  bool prefill_qk_dpas = false;
  if (total_rows >= 128 && heads == 6 && aligned) {
    const char* enabled = std::getenv("QWEN38_QSA_PREFILL_QK_DPAS");
    namespace sx = sycl::ext::oneapi::experimental;
    prefill_qk_dpas =
        (!enabled || std::strcmp(enabled, "1") == 0) &&
        queue.get_device().get_info<sx::info::device::architecture>() ==
            sx::architecture::intel_gpu_bmg_g31;
  }
  int tokens_per_partial = total_rows == 1 ? 64 : 48;
  if (total_rows >= 128) {
    // 三头DPAS使用48slot；旧prefill后端用256slot。小M沿用原选择。
    tokens_per_partial = prefill_qk_dpas ? 48 : 256;
    const char* tokens = std::getenv("QWEN38_QSA_PREFILL_TOKENS_PER_PARTIAL");
    if (tokens && std::strcmp(tokens, "48") == 0)
      tokens_per_partial = 48;
    else if (tokens && std::strcmp(tokens, "64") == 0)
      tokens_per_partial = 64;
    else if (tokens && std::strcmp(tokens, "128") == 0)
      tokens_per_partial = 128;
    else if (tokens && std::strcmp(tokens, "256") == 0)
      tokens_per_partial = 256;
  }
  const int partial_count =
      (kSlots + tokens_per_partial - 1) / tokens_per_partial;
  auto launch = [&](auto page_tag,
                    auto block_tag,
                    int rows,
                    int offset,
                    sycl::event previous,
                    bool has_previous) {
    const auto* row_q = q_ptr + static_cast<int64_t>(offset) * heads * kHeadDim;
    const auto* row_indices =
        indices_ptr + static_cast<int64_t>(offset) * kSlots;
    const auto* row_reqs = req_ptr + offset;
    if constexpr (decltype(block_tag)::value) {
      if (prefill_qk_dpas) {
        return launch_prefill_qk_dpas_phase0<decltype(page_tag)::value>(
            queue,
            row_q,
            packed_ptr,
            row_indices,
            table_ptr,
            row_reqs,
            partial_ptr,
            rows,
            block_table.size(0),
            block_table.size(1),
            packed_kv.size(0),
            tokens_per_partial,
            partial_count,
            previous,
            has_previous);
      }
    }
    auto launch_generic = [&](auto kv_tag) {
      if (total_rows == 1) {
        return launch_phase0<
            decltype(page_tag)::value,
            decltype(block_tag)::value,
            32,
            decltype(block_tag)::value ? 2 : 1,
            decltype(kv_tag)::value>(
            queue,
            row_q,
            packed_ptr,
            row_indices,
            table_ptr,
            row_reqs,
            partial_ptr,
            rows,
            heads,
            block_table.size(0),
            block_table.size(1),
            packed_kv.size(0),
            tokens_per_partial,
            partial_count,
            previous,
            has_previous);
      }
      return launch_phase0<
          decltype(page_tag)::value,
          decltype(block_tag)::value,
          16,
          1,
          decltype(kv_tag)::value>(
          queue,
          row_q,
          packed_ptr,
          row_indices,
          table_ptr,
          row_reqs,
          partial_ptr,
          rows,
          heads,
          block_table.size(0),
          block_table.size(1),
          packed_kv.size(0),
          tokens_per_partial,
          partial_count,
          previous,
          has_previous);
    };
    if (heads == 12) {
      return launch_generic(std::integral_constant<int, 2>{});
    }
    return launch_generic(std::integral_constant<int, 1>{});
  };
  sycl::event previous;
  bool has_previous = false;
  for (int offset = 0; offset < total_rows; offset += 128) {
    const int rows = std::min(128, total_rows - offset);
    sycl::event phase0;
    if (page_size == 256) {
      phase0 = aligned ? launch(
                             std::integral_constant<int, 256>{},
                             std::true_type{},
                             rows,
                             offset,
                             previous,
                             has_previous)
                       : launch(
                             std::integral_constant<int, 256>{},
                             std::false_type{},
                             rows,
                             offset,
                             previous,
                             has_previous);
    } else {
      phase0 = aligned ? launch(
                             std::integral_constant<int, 512>{},
                             std::true_type{},
                             rows,
                             offset,
                             previous,
                             has_previous)
                       : launch(
                             std::integral_constant<int, 512>{},
                             std::false_type{},
                             rows,
                             offset,
                             previous,
                             has_previous);
    }
    previous = launch_phase1(
        queue,
        phase0,
        partial_ptr,
        output_ptr + static_cast<int64_t>(offset) * heads * kHeadDim,
        rows,
        heads,
        partial_count,
        q.size(0) == 1);
    has_previous = true;
  }
  return output;
}

namespace {

constexpr int kIndexHeads = 4;
constexpr int kIndexDim = 128;
constexpr int kTopBlocks = 512;
constexpr int kSortSlots = 1024;
constexpr int kSelectionWg = 1024;
constexpr int kMergeWg = 512;
constexpr int kSelectionSg = 32;
constexpr int kSelectionPartitions = 32;
constexpr int kWorkspaceRowStride = kSelectionPartitions * kTopBlocks;

template <int Page>
class SelectionSplitScore;
template <int Page>
sycl::event launch_selection_split_score(
    sycl::queue& queue,
    const half* q,
    const half* cache,
    const int32_t* table,
    const int32_t* reqs,
    const int64_t* positions,
    const int32_t* lengths,
    float* sa,
    int32_t* ia,
    float* sb,
    int32_t* ib,
    int requests,
    int table_stride,
    int physical_pages,
    int64_t page_stride,
    int64_t row_stride,
    int captured_bpp,
    sycl::event previous,
    bool has_previous) {
  constexpr int Groups = 256, Subgroups = 8;
  return queue.submit([&](sycl::handler& cgh) {
    if (has_previous) cgh.depends_on(previous);
    cgh.parallel_for<SelectionSplitScore<Page>>(
        sycl::nd_range<1>(Groups * Subgroups * 32, Subgroups * 32),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
          auto sg = item.get_sub_group();
          const int lane = sg.get_local_linear_id();
          const int task = item.get_group_linear_id() * Subgroups +
                           int(sg.get_group_linear_id());
          const int req = reqs[0];
          const bool request_ok = req >= 0 && req < requests;
          const int64_t seq =
              request_ok ? sycl::max(int64_t(lengths[req]), int64_t(0)) : 0;
          const int64_t visible =
              sycl::min(sycl::max(positions[0] + 1, int64_t(0)), seq) / 4;
          const int64_t limit = sycl::min(visible, int64_t(32) * captured_bpp);
          if (task >= limit) return;
          const int per_group = int((limit + 31) / 32);
          const int bpp = sycl::max(512, ((per_group + 511) / 512) * 512);
          float query[4][4];
          {
#pragma unroll
            for (int h = 0; h < 4; ++h) {
#pragma unroll
              for (int frag = 0; frag < 4; ++frag)
                query[h][frag] = float(q[h * 128 + lane + frag * 32]);
            }
          }
          for (int logical = task; logical < limit;
               logical += Groups * Subgroups) {
            const int lp = logical / Page;
            int physical = -1;
            if (request_ok && lp < table_stride)
              physical = table[req * table_stride + lp];
            const bool valid = physical >= 0 && physical < physical_pages;
            float score = -std::numeric_limits<float>::infinity();
            if (valid) {
              const int64_t offset = int64_t(physical) * page_stride +
                                     (logical % Page) * row_stride;
              float dot[4] = {};
#pragma unroll
              for (int frag = 0; frag < 4; ++frag) {
                const int dim = lane + frag * 32;
                const float key = float(cache[offset + dim]);
#pragma unroll
                for (int h = 0; h < 4; ++h)
                  dot[h] += query[h][frag] * key;
              }
              score = 0;
#pragma unroll
              for (int h = 0; h < 4; ++h) {
                const float value =
                    sycl::reduce_over_group(sg, dot[h], sycl::plus<float>());
                score += sycl::fmax(value, 0.0f);
              }
              score *= 0.08838834764831845f;
            }
            if (lane == 0) {
              const int partition = logical / bpp, within = logical % bpp,
                        chunk = within / 512,
                        at = partition * 512 + within % 512;
              if (chunk == 0)
                sa[at] = score;
              else if (chunk == 1)
                ia[at] = sycl::bit_cast<int32_t>(score);
              else if (chunk == 2)
                sb[at] = score;
              else
                ib[at] = sycl::bit_cast<int32_t>(score);
            }
          }
        });
  });
}

// Valid QSA scores are nonnegative after fmax; zero is reserved for a
// missing index. The low 32 bits retain the lower-logical-index tie rule.
inline uint64_t selection_score_key(float score, int32_t index) {
  return index < 0 ? uint64_t(0)
                   : (uint64_t(sycl::bit_cast<uint32_t>(score)) << 32) |
                         (uint64_t(UINT32_MAX) - uint32_t(index));
}
inline float selection_key_score(uint64_t key) {
  return key ? sycl::bit_cast<float>(uint32_t(key >> 32))
             : -std::numeric_limits<float>::infinity();
}
inline int32_t selection_key_index(uint64_t key) {
  return key ? int32_t(UINT32_MAX - uint32_t(key)) : -1;
}
// Hierarchical merge with all dependent levels inside one workgroup.
template <int FanIn, bool Final>
class SelectionFold;
template <int FanIn, bool Final>
sycl::event launch_selection_fold(
    sycl::queue& queue,
    sycl::event previous,
    const float* srcs,
    const int32_t* srci,
    float* dsts,
    int32_t* dsti,
    const int32_t* reqs,
    const int64_t* positions,
    const int32_t* lengths,
    int32_t* out,
    int requests,
    int groups,
    int captured_bpp) {
  return queue.submit([&](sycl::handler& cgh) {
    cgh.depends_on(previous);
    sycl::local_accessor<uint64_t, 1> a(FanIn * 512, cgh), b(FanIn * 256, cgh);
    cgh.parallel_for<SelectionFold<FanIn, Final>>(
        sycl::nd_range<1>(groups * 1024, 1024),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
          const int group = it.get_group_linear_id(),
                    lane = it.get_local_linear_id();
          const int request = reqs[0],
                    seq = request >= 0 && request < requests
                              ? sycl::max(lengths[request], 0)
                              : 0;
          const int64_t visible = sycl::max(positions[0] + 1, int64_t(0));
          const int64_t blocks = sycl::min(visible, int64_t(seq)) / 4;
          const int per_group =
              int((sycl::min(blocks, int64_t(32) * captured_bpp) + 31) / 32);
          const int bpp = sycl::max(512, ((per_group + 511) / 512) * 512);
          const int covered = 32 / groups;
          const bool empty = int64_t(group) * covered * bpp >= blocks;
          const bool only_left =
              int64_t(group * covered + covered / FanIn) * bpp >= blocks;
          uint64_t key = 0;
          if (empty || only_left) {
            if (lane < 512 && !empty)
              key = selection_score_key(
                  srcs[group * FanIn * 512 + lane],
                  srci[group * FanIn * 512 + lane]);
          } else {
            for (int i = lane; i < FanIn * 512; i += 1024)
              a[i] = selection_score_key(
                  srcs[group * FanIn * 512 + i], srci[group * FanIn * 512 + i]);
            it.barrier(sycl::access::fence_space::local_space);
            bool source_a = true;
            for (int lists = FanIn; lists > 1; lists /= 2) {
              for (int task = lane; task < (lists / 2) * 512; task += 1024) {
                const int pair = task / 512, k = task % 512 + 1,
                          base = pair * 1024;
                int lo = 0, hi = k;
                auto get = [&](int x) { return source_a ? a[x] : b[x]; };
                while (lo < hi) {
                  int i = (lo + hi + 1) / 2, j = k - i;
                  if (get(base + i - 1) >= get(base + 512 + j))
                    lo = i;
                  else
                    hi = i - 1;
                }
                const int i = lo, j = k - i;
                uint64_t winner =
                    i == 0 ? get(base + 512 + j - 1)
                    : j == 0
                        ? get(base + i - 1)
                        : sycl::min(get(base + i - 1), get(base + 512 + j - 1));
                if (source_a)
                  b[task] = winner;
                else
                  a[task] = winner;
              }
              it.barrier(sycl::access::fence_space::local_space);
              source_a = !source_a;
            }
            if (lane < 512) key = source_a ? a[lane] : b[lane];
          }
          if (lane >= 512) return;
          const int block = selection_key_index(key);
          if constexpr (Final) {
            const int selected = int(sycl::min(blocks, int64_t(512)));
            const int64_t tail_start = visible / 4 * 4,
                          tail = visible - tail_start;
#pragma unroll
            for (int offset = 0; offset < 4; ++offset) {
              int token = block * 4 + offset;
              out[lane * 4 + offset] =
                  lane < selected && block >= 0 && token < seq ? token
                  : lane == selected && offset < tail &&
                          tail_start + offset < seq
                      ? int32_t(tail_start + offset)
                      : -1;
            }
            if (lane < 3)
              out[2048 + lane] =
                  selected == 512 && lane < tail && tail_start + lane < seq
                      ? int32_t(tail_start + lane)
                      : -1;
          } else {
            dsts[group * 512 + lane] = selection_key_score(key);
            dsti[group * 512 + lane] = block;
          }
        });
  });
}
// Fixed networks keep shuffle distances compile-time constants; score and id
// remain separate 32-bit registers. Two SLM banks avoid a read/write race.
template <int Count>
inline void hybrid_float_top(
    sycl::nd_item<1> item,
    sycl::local_accessor<float, 1> scores,
    sycl::local_accessor<int32_t, 1> ids,
    int offset) {
  const int lane = item.get_local_linear_id();
  auto sg = item.get_sub_group();
  float score = lane < Count ? scores[offset + lane]
                             : -std::numeric_limits<float>::infinity();
  int32_t id = lane < Count ? ids[offset + lane] : -1;
  int bank = 0;
#pragma unroll
  for (int size = 2; size <= Count; size <<= 1) {
#pragma unroll
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      float other;
      int32_t oi;
      if (stride >= 32) {
        if (lane < Count) {
          scores[bank * 1024 + lane] = score;
          ids[bank * 1024 + lane] = id;
        }
        item.barrier(sycl::access::fence_space::local_space);
        other = lane < Count ? scores[bank * 1024 + (lane ^ stride)] : score;
        oi = lane < Count ? ids[bank * 1024 + (lane ^ stride)] : id;
        bank ^= 1;
      } else {
        other = sycl::permute_group_by_xor(sg, score, stride);
        oi = sycl::permute_group_by_xor(sg, id, stride);
      }
      const bool low = ((lane & size) == 0) == ((lane & stride) == 0);
      const bool greater = score > other || (score == other && id < oi);
      if (greater == low) {
        score = other;
        id = oi;
      }
    }
  }
  if (lane < Count) {
    scores[bank * 1024 + lane] = score;
    ids[bank * 1024 + lane] = id;
  }
  item.barrier(sycl::access::fence_space::local_space);
  float winner = lane < 512 ? scores[bank * 1024 + Count - 1 - lane]
                            : -std::numeric_limits<float>::infinity();
  int32_t wi = lane < 512 ? ids[bank * 1024 + Count - 1 - lane] : -1;
  item.barrier(sycl::access::fence_space::local_space);
  if (lane < 512) {
    scores[lane] = winner;
    ids[lane] = wi;
  }
  item.barrier(sycl::access::fence_space::local_space);
}

template <int PageSize, bool Short, bool Bounded, bool Split>
class SelectionPhase0;
template <bool Bounded>
class SelectionMerge;

template <int PageSize, bool Short, bool Bounded, bool Split = false>
sycl::event launch_selection_phase0(
    sycl::queue& queue,
    const half* q,
    const half* cache,
    const int32_t* table,
    const int32_t* reqs,
    const int64_t* positions,
    const int32_t* seq_lengths,
    int32_t* out,
    float* scores_a,
    int32_t* indices_a,
    int rows,
    int requests,
    int table_stride,
    int physical_pages,
    int64_t cache_page_stride,
    int64_t cache_row_stride,
    int partitions,
    int captured_blocks_per_partition,
    sycl::event previous,
    bool has_previous,
    const float* raw_sb = nullptr,
    const int32_t* raw_ib = nullptr) {
  return queue.submit([&](sycl::handler& cgh) {
    if (has_previous) cgh.depends_on(previous);
    sycl::local_accessor<half, 1> query_shared(
        sycl::range<1>(kIndexHeads * kIndexDim), cgh);
    sycl::local_accessor<float, 1> scores(
        sycl::range<1>(Split ? 2 * kSortSlots : kSortSlots), cgh);
    sycl::local_accessor<int32_t, 1> indices(
        sycl::range<1>(Split ? 2 * kSortSlots : kSortSlots), cgh);
    cgh.parallel_for<SelectionPhase0<PageSize, Short, Bounded, Split>>(
        sycl::nd_range<1>(
            sycl::range<1>(rows * partitions * kSelectionWg),
            sycl::range<1>(kSelectionWg)),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(kSelectionSg)]] {
          const int group = item.get_group_linear_id();
          const int row = group / partitions;
          const int partition = group % partitions;
          const int lane = item.get_local_linear_id();
          const int subgroup = lane / kSelectionSg;
          const int dim_lane = lane % kSelectionSg;
          const int request = reqs[row];
          const bool valid_req = request >= 0 && request < requests;
          const int32_t seq =
              valid_req ? sycl::max(seq_lengths[request], int32_t(0)) : 0;
          const int64_t visible_tokens =
              sycl::max(positions[row] + 1, int64_t(0));
          const int64_t visible_blocks =
              sycl::min(visible_tokens, static_cast<int64_t>(seq)) / 4;
          int blocks_per_partition = captured_blocks_per_partition;
          if constexpr (Bounded && !Short) {
            if (partitions == 32) {
              const int64_t scanned = sycl::min(
                  visible_blocks,
                  int64_t(partitions) * captured_blocks_per_partition);
              const int per_group =
                  int((scanned + partitions - 1) / partitions);
              blocks_per_partition = sycl::max(
                  kTopBlocks,
                  ((per_group + kTopBlocks - 1) / kTopBlocks) * kTopBlocks);
            }
          }
          // All lanes agree on row/partition. An entirely invisible partition
          // contributes the same sentinels as the full score/sort path.
          if (partitions > 1 &&
              static_cast<int64_t>(partition) * blocks_per_partition >=
                  visible_blocks) {
            if (lane < kTopBlocks) {
              const int64_t offset =
                  static_cast<int64_t>(row) * kWorkspaceRowStride +
                  partition * kTopBlocks + lane;
              scores_a[offset] = -std::numeric_limits<float>::infinity();
              indices_a[offset] = -1;
            }
            return;
          }
          if (lane < kIndexHeads * kIndexDim) {
            query_shared[lane] = q[row * kIndexHeads * kIndexDim + lane];
          }
          for (int s = lane; s < kTopBlocks; s += kSelectionWg) {
            scores[s] = -std::numeric_limits<float>::infinity();
            indices[s] = -1;
          }
          item.barrier(sycl::access::fence_space::local_space);
          const int active_blocks =
              Bounded ? int(sycl::min(
                            int64_t(blocks_per_partition),
                            sycl::max(
                                visible_blocks -
                                    int64_t(partition) * blocks_per_partition,
                                int64_t(0))))
                      : blocks_per_partition;
          for (int base = 0; base < active_blocks; base += kTopBlocks) {
            const bool first_chunk = base == 0;
            int first_sort_size = kTopBlocks;
            if constexpr (Short) {
              const int64_t available = sycl::max(
                  visible_blocks -
                      static_cast<int64_t>(partition) * blocks_per_partition,
                  int64_t(0));
              first_sort_size = 2;
              while (first_sort_size <
                     sycl::min(
                         available,
                         int64_t(sycl::min(blocks_per_partition, kTopBlocks))))
                first_sort_size <<= 1;
            }
            int score_count = kTopBlocks;
            if constexpr (Short) {
              const int64_t available = sycl::max(
                  visible_blocks -
                      static_cast<int64_t>(partition) * blocks_per_partition -
                      base,
                  int64_t(0));
              score_count = int(sycl::min(
                  available,
                  int64_t(sycl::min(blocks_per_partition - base, kTopBlocks))));
              if (lane >= score_count && lane < kTopBlocks) {
                scores[kTopBlocks + lane] =
                    -std::numeric_limits<float>::infinity();
                indices[kTopBlocks + lane] = -1;
              }
            }
            if constexpr (Split) {
              // Each consumer owns one scratch row. Chunk 0/1 are read
              // before this row is overwritten by its final sorted leaf.
              for (int b = lane; b < score_count; b += kSelectionWg) {
                const int64_t logical =
                    int64_t(partition) * blocks_per_partition + base + b;
                float score = -std::numeric_limits<float>::infinity();
                if (logical < visible_blocks) {
                  const int chunk = (base + b) / 512,
                            from = partition * 512 + b;
                  if (chunk == 0)
                    score = scores_a[from];
                  else if (chunk == 1)
                    score = sycl::bit_cast<float>(indices_a[from]);
                  else if (chunk == 2)
                    score = raw_sb[from];
                  else
                    score = sycl::bit_cast<float>(raw_ib[from]);
                }
                scores[512 + b] = score;
                indices[512 + b] = score >= 0 ? int32_t(logical) : -1;
              }
            } else {
              for (int b = subgroup; b < score_count;
                   b += kSelectionWg / kSelectionSg) {
                const int64_t logical =
                    static_cast<int64_t>(partition) * blocks_per_partition +
                    base + b;
                const int64_t logical_page = logical / PageSize;
                bool valid =
                    valid_req && logical < visible_blocks &&
                    (!Short || logical < static_cast<int64_t>(partition + 1) *
                                             blocks_per_partition) &&
                    logical_page < table_stride;
                int physical = -1;
                if (valid) {
                  physical = table[request * table_stride + logical_page];
                  valid = physical >= 0 && physical < physical_pages;
                }
                const int64_t cache_offset =
                    valid ? static_cast<int64_t>(physical) * cache_page_stride +
                                (logical % PageSize) * cache_row_stride
                          : 0;
                float dot[4] = {};
#pragma unroll
                for (int frag = 0; frag < 4; ++frag) {
                  const int dim = dim_lane + frag * kSelectionSg;
                  const float key =
                      valid ? static_cast<float>(cache[cache_offset + dim])
                            : 0.0f;
#pragma unroll
                  for (int h = 0; h < kIndexHeads; ++h) {
                    dot[h] +=
                        static_cast<float>(query_shared[h * kIndexDim + dim]) *
                        key;
                  }
                }
                float score = 0.0f;
#pragma unroll
                for (int h = 0; h < kIndexHeads; ++h) {
                  const float value = sycl::reduce_over_group(
                      item.get_sub_group(), dot[h], sycl::plus<float>());
                  score += sycl::fmax(value, 0.0f);
                }
                if (dim_lane == 0) {
                  scores[kTopBlocks + b] =
                      valid ? score * 0.08838834764831845f
                            : -std::numeric_limits<float>::infinity();
                  indices[kTopBlocks + b] =
                      valid ? static_cast<int32_t>(logical) : -1;
                }
              }
            }
            item.barrier(sycl::access::fence_space::local_space);
            // Sort ascending score, descending index. Reverse the high half
            // for final descending score and deterministic low-index ties.
            const int sort_offset = first_chunk ? kTopBlocks : 0;
            const int sort_size = first_chunk ? first_sort_size : kSortSlots;
            if constexpr (Split) {
              if (first_chunk)
                hybrid_float_top<512>(item, scores, indices, 512);
              else
                hybrid_float_top<1024>(item, scores, indices, 0);
            } else {
              for (int size = 2; size <= sort_size; size <<= 1) {
                for (int stride = size >> 1; stride > 0; stride >>= 1) {
                  for (int logical_left = lane; logical_left < sort_size;
                       logical_left += kSelectionWg) {
                    const int left = logical_left + sort_offset;
                    const int right = (logical_left ^ stride) + sort_offset;
                    if (right <= left) {
                      continue;
                    }
                    const float a = scores[left], bscore = scores[right];
                    const int32_t ai = indices[left], bi = indices[right];
                    const bool ascending = (logical_left & size) == 0;
                    const bool swap =
                        ascending ? (a > bscore || (a == bscore && ai < bi))
                                  : (a < bscore || (a == bscore && ai > bi));
                    if (swap) {
                      scores[left] = bscore;
                      scores[right] = a;
                      indices[left] = bi;
                      indices[right] = ai;
                    }
                  }
                  item.barrier(sycl::access::fence_space::local_space);
                }
              }
              float winning_score = -std::numeric_limits<float>::infinity();
              int32_t winning_index = -1;
              if (lane < kTopBlocks && lane < sort_size) {
                const int from = sort_offset + sort_size - 1 - lane;
                winning_score = scores[from];
                winning_index = indices[from];
              }
              item.barrier(sycl::access::fence_space::local_space);
              if (lane < kTopBlocks) {
                scores[lane] = winning_score;
                indices[lane] = winning_index;
              }
              item.barrier(sycl::access::fence_space::local_space);
            }
          }
          if constexpr (Split)
            item.barrier(sycl::access::fence_space::global_and_local);
          if (lane >= kTopBlocks) {
            return;
          }
          if (partitions == 1) {
            const int block = indices[lane];
            const int selected_blocks =
                sycl::min(visible_blocks, static_cast<int64_t>(kTopBlocks));
            const int64_t tail_start = visible_tokens / 4 * 4;
            const int64_t tail_count = visible_tokens - tail_start;
#pragma unroll
            for (int offset = 0; offset < 4; ++offset) {
              const int token = block * 4 + offset;
              out[row * kSlots + lane * 4 + offset] =
                  lane < selected_blocks && block >= 0 && token < seq
                      ? token
                      : (lane == selected_blocks && offset < tail_count &&
                                 tail_start + offset < seq
                             ? static_cast<int32_t>(tail_start + offset)
                             : -1);
            }
            if (lane < 3) {
              out[row * kSlots + 2048 + lane] =
                  selected_blocks == kTopBlocks && lane < tail_count &&
                          tail_start + lane < seq
                      ? static_cast<int32_t>(tail_start + lane)
                      : -1;
            }
          } else {
            const int64_t offset =
                static_cast<int64_t>(row) * kWorkspaceRowStride +
                partition * kTopBlocks + lane;
            scores_a[offset] = scores[lane];
            indices_a[offset] = indices[lane];
          }
        });
  });
}

template <bool Bounded>
sycl::event launch_selection_merge(
    sycl::queue& queue,
    sycl::event previous,
    const float* src_scores,
    const int32_t* src_indices,
    float* dst_scores,
    int32_t* dst_indices,
    const int32_t* reqs,
    const int64_t* positions,
    const int32_t* seq_lengths,
    int32_t* out,
    int rows,
    int requests,
    int pairs,
    int captured_blocks_per_partition,
    int source_partitions) {
  return queue.submit([&](sycl::handler& cgh) {
    cgh.depends_on(previous);
    sycl::local_accessor<float, 1> scores(sycl::range<1>(kSortSlots), cgh);
    sycl::local_accessor<int32_t, 1> indices(sycl::range<1>(kSortSlots), cgh);
    cgh.parallel_for<SelectionMerge<Bounded>>(
        sycl::nd_range<1>(
            sycl::range<1>(rows * pairs * kMergeWg), sycl::range<1>(kMergeWg)),
        [=](sycl::nd_item<1> item) {
          const int row = item.get_group_linear_id() / pairs;
          const int pair = item.get_group_linear_id() % pairs;
          const int lane = item.get_local_linear_id();
          const int base = row * kWorkspaceRowStride + pair * kSortSlots;
          scores[lane] = src_scores[base + lane];
          indices[lane] = src_indices[base + lane];
          bool right_empty = false;
          if constexpr (Bounded) {
            const int req = reqs[row];
            const int64_t seq =
                req >= 0 && req < requests
                    ? sycl::max(int64_t(seq_lengths[req]), int64_t(0))
                    : 0;
            const int64_t visible = sycl::max(positions[row] + 1, int64_t(0));
            const int64_t blocks = sycl::min(visible, seq) / 4;
            int blocks_per_partition = captured_blocks_per_partition;
            const int partitions = 2 * pairs * source_partitions;
            if (partitions == 32) {
              const int64_t scanned = sycl::min(
                  blocks, int64_t(partitions) * captured_blocks_per_partition);
              const int per_group =
                  int((scanned + partitions - 1) / partitions);
              blocks_per_partition = sycl::max(
                  kTopBlocks,
                  ((per_group + kTopBlocks - 1) / kTopBlocks) * kTopBlocks);
            }
            right_empty = int64_t(pair * 2 + 1) * source_partitions *
                              blocks_per_partition >=
                          blocks;
          }
          if (!right_empty) {
            scores[kSortSlots - 1 - lane] =
                src_scores[base + kTopBlocks + lane];
            indices[kSortSlots - 1 - lane] =
                src_indices[base + kTopBlocks + lane];
            item.barrier(sycl::access::fence_space::local_space);
            for (int stride = kTopBlocks; stride > 0; stride >>= 1) {
              const int left = (lane / stride) * (2 * stride) + lane % stride;
              const int right = left + stride;
              const float a = scores[left], b = scores[right];
              const int32_t ai = indices[left], bi = indices[right];
              if (a < b || (a == b && ai > bi)) {
                scores[left] = b;
                scores[right] = a;
                indices[left] = bi;
                indices[right] = ai;
              }
              item.barrier(sycl::access::fence_space::local_space);
            }
          }
          if (pairs == 1) {
            const int req = reqs[row];
            const int seq = req >= 0 && req < requests
                                ? sycl::max(seq_lengths[req], int32_t(0))
                                : 0;
            const int64_t visible = sycl::max(positions[row] + 1, int64_t(0));
            const int64_t visible_blocks =
                sycl::min(visible / 4, static_cast<int64_t>(seq / 4));
            const int selected_blocks = static_cast<int>(
                sycl::min(visible_blocks, static_cast<int64_t>(kTopBlocks)));
            const int64_t tail_start = visible / 4 * 4;
            const int64_t tail_count = visible - tail_start;
            const int block = indices[lane];
#pragma unroll
            for (int offset = 0; offset < 4; ++offset) {
              const int token = block * 4 + offset;
              out[row * kSlots + lane * 4 + offset] =
                  lane < selected_blocks && block >= 0 && token < seq
                      ? token
                      : (lane == selected_blocks && offset < tail_count &&
                                 tail_start + offset < seq
                             ? static_cast<int32_t>(tail_start + offset)
                             : -1);
            }
            if (lane < 3) {
              out[row * kSlots + 2048 + lane] =
                  selected_blocks == kTopBlocks && lane < tail_count &&
                          tail_start + lane < seq
                      ? static_cast<int32_t>(tail_start + lane)
                      : -1;
            }
          } else {
            const int offset =
                row * kWorkspaceRowStride + pair * kTopBlocks + lane;
            dst_scores[offset] = scores[lane];
            dst_indices[offset] = indices[lane];
          }
        });
  });
}

}  // namespace

at::Tensor select_paged_tokens_v2(
    const at::Tensor& q,
    const at::Tensor& compressed_key_cache,
    const at::Tensor& page_table,
    const at::Tensor& token_to_req,
    const at::Tensor& query_positions,
    const at::Tensor& sequence_lengths,
    int64_t compressed_page_size,
    int64_t max_seq_len,
    at::Tensor out,
    at::Tensor scores_a,
    at::Tensor indices_a,
    at::Tensor scores_b,
    at::Tensor indices_b) {
  TORCH_CHECK(q.is_xpu(), "selection query must be on XPU");
  check_xpu(q, q, at::kHalf, "selection query");
  check_xpu(compressed_key_cache, q, at::kHalf, "compressed_key_cache");
  check_xpu(page_table, q, at::kInt, "page_table");
  check_xpu(token_to_req, q, at::kInt, "token_to_req");
  check_xpu(query_positions, q, at::kLong, "query_positions");
  check_xpu(sequence_lengths, q, at::kInt, "sequence_lengths");
  check_xpu(out, q, at::kInt, "out");
  check_xpu(scores_a, q, at::kFloat, "scores_a");
  check_xpu(indices_a, q, at::kInt, "indices_a");
  check_xpu(scores_b, q, at::kFloat, "scores_b");
  check_xpu(indices_b, q, at::kInt, "indices_b");
  TORCH_CHECK(
      q.scalar_type() == at::kHalf && q.dim() == 3 && q.size(0) >= 1 &&
          q.size(0) <= 4096 && q.size(1) == kIndexHeads &&
          q.size(2) == kIndexDim && q.is_contiguous(),
      "q must be contiguous FP16 [1..4096,4,128]");
  TORCH_CHECK(
      compressed_page_size == 64 || compressed_page_size == 128,
      "compressed_page_size must be 64 or 128");
  TORCH_CHECK(
      compressed_key_cache.dim() == 4 && compressed_key_cache.size(0) > 0 &&
          compressed_key_cache.size(1) == compressed_page_size &&
          compressed_key_cache.size(2) == 1 &&
          compressed_key_cache.size(3) == kIndexDim &&
          compressed_key_cache.stride(3) == 1 &&
          compressed_key_cache.stride(1) >= kIndexDim &&
          compressed_key_cache.stride(0) > 0,
      "compressed cache shape/strides are invalid");
  TORCH_CHECK(
      page_table.dim() == 2 && page_table.size(0) > 0 &&
          page_table.size(1) > 0 && page_table.is_contiguous() &&
          token_to_req.sizes() == at::IntArrayRef({q.size(0)}) &&
          token_to_req.is_contiguous() &&
          query_positions.sizes() == at::IntArrayRef({q.size(0)}) &&
          query_positions.is_contiguous() &&
          sequence_lengths.sizes() == at::IntArrayRef({page_table.size(0)}) &&
          sequence_lengths.is_contiguous(),
      "selection metadata shape/strides are invalid");
  TORCH_CHECK(
      max_seq_len > 0 && max_seq_len <= 256000 &&
          max_seq_len <= page_table.size(1) * compressed_page_size * 4,
      "max_seq_len exceeds selection table coverage");
  TORCH_CHECK(
      out.sizes() == at::IntArrayRef({q.size(0), kSlots}) &&
          out.is_contiguous(),
      "out must be caller-owned [M,2051]");
  const int64_t scratch_rows = scores_a.dim() == 3 ? scores_a.size(0) : -1;
  TORCH_CHECK(
      valid_selection_scratch_rows(q.size(0), max_seq_len, scratch_rows),
      "selection scratch rows must be min(M,32), or 128 only for "
      "M>=128 and max_seq_len>=4096");
  const std::array<int64_t, 3> workspace_dimensions = {
      scratch_rows, kSelectionPartitions, kTopBlocks};
  const at::IntArrayRef workspace_shape(workspace_dimensions);
  TORCH_CHECK(
      scores_a.sizes() == workspace_shape && scores_a.is_contiguous() &&
          indices_a.sizes() == workspace_shape && indices_a.is_contiguous() &&
          scores_b.sizes() == workspace_shape && scores_b.is_contiguous() &&
          indices_b.sizes() == workspace_shape && indices_b.is_contiguous(),
      "caller workspaces must be matching [scratch_rows,32,512] "
      "contiguous tensors");
  const std::array<const at::Tensor*, 6> inputs = {
      &q,
      &compressed_key_cache,
      &page_table,
      &token_to_req,
      &query_positions,
      &sequence_lengths};
  const std::array<const at::Tensor*, 5> outputs = {
      &out, &scores_a, &indices_a, &scores_b, &indices_b};
  for (const auto* output : outputs) {
    for (const auto* input : inputs) {
      TORCH_CHECK(
          !may_overlap(*output, *input),
          "selection output/workspace aliases an input");
    }
  }
  for (size_t a = 0; a < outputs.size(); ++a) {
    for (size_t b = a + 1; b < outputs.size(); ++b) {
      TORCH_CHECK(
          !may_overlap(*outputs[a], *outputs[b]),
          "selection output/workspaces overlap");
    }
  }
  c10::OptionalDeviceGuard guard(q.device());
  auto stream = c10::xpu::getCurrentXPUStream(q.get_device());
  auto& queue = stream.queue();
  TORCH_CHECK(
      queue.is_in_order(),
      "selection scratch reuse requires an in-order current stream");
  record_tensors(
      {&q,
       &compressed_key_cache,
       &page_table,
       &token_to_req,
       &query_positions,
       &sequence_lengths,
       &out,
       &scores_a,
       &indices_a,
       &scores_b,
       &indices_b},
      stream);
  // Small captures use four groups. For larger M1 captures the fixed 32
  // groups partition the live visible range on device, keeping the same
  // scratch ABI and graph topology. Every per-block FP32 score and the total
  // score/index ordering are unchanged. M>1 keeps its original partitioning.
  const bool short_selection = q.size(0) == 1 && max_seq_len <= 2048;
  const bool bounded = q.size(0) == 1;
  const int partitions =
      short_selection ? 4 : (max_seq_len >= 4096 ? kSelectionPartitions : 1);
  const int max_blocks = static_cast<int>((max_seq_len + 3) / 4);
  const int per_partition = (max_blocks + partitions - 1) / partitions;
  int blocks_per_partition =
      ((per_partition + kTopBlocks - 1) / kTopBlocks) * kTopBlocks;
  if (short_selection) {
    blocks_per_partition = 1;
    while (blocks_per_partition < per_partition)
      blocks_per_partition <<= 1;
  }
  const bool split_selection =
      bounded && partitions == 32 && blocks_per_partition <= 2048;
  const auto* query_ptr = static_cast<const half*>(q.data_ptr());
  const auto* cache_ptr =
      static_cast<const half*>(compressed_key_cache.data_ptr());
  const auto* table_ptr = page_table.data_ptr<int32_t>();
  const auto* req_ptr = token_to_req.data_ptr<int32_t>();
  const auto* pos_ptr = query_positions.data_ptr<int64_t>();
  const auto* seq_ptr = sequence_lengths.data_ptr<int32_t>();
  auto* out_ptr = out.data_ptr<int32_t>();
  auto* sa = scores_a.data_ptr<float>();
  auto* ia = indices_a.data_ptr<int32_t>();
  auto* sb = scores_b.data_ptr<float>();
  auto* ib = indices_b.data_ptr<int32_t>();
  sycl::event previous;
  bool has_previous = false;
  const int chunk_rows = static_cast<int>(scratch_rows);
  for (int offset = 0; offset < q.size(0); offset += chunk_rows) {
    const int rows = std::min<int64_t>(chunk_rows, q.size(0) - offset);
    const auto* chunk_q =
        query_ptr + static_cast<int64_t>(offset) * kIndexHeads * kIndexDim;
    const auto* chunk_reqs = req_ptr + offset;
    const auto* chunk_positions = pos_ptr + offset;
    auto* chunk_out = out_ptr + static_cast<int64_t>(offset) * kSlots;
    auto submit_phase0 = [&]<bool Short, bool Bounded>() {
      // Four existing [32,512] arrays hold up to 65536 raw scores.
      // M>1, unbounded selection and larger captures retain the old path.
      if constexpr (Bounded && !Short) {
        if (split_selection) {
          auto split = [&]<int Page>() {
            auto scored = launch_selection_split_score<Page>(
                queue,
                chunk_q,
                cache_ptr,
                table_ptr,
                chunk_reqs,
                chunk_positions,
                seq_ptr,
                sa,
                ia,
                sb,
                ib,
                page_table.size(0),
                page_table.size(1),
                compressed_key_cache.size(0),
                compressed_key_cache.stride(0),
                compressed_key_cache.stride(1),
                blocks_per_partition,
                previous,
                has_previous);
            return launch_selection_phase0<Page, false, true, true>(
                queue,
                chunk_q,
                cache_ptr,
                table_ptr,
                chunk_reqs,
                chunk_positions,
                seq_ptr,
                chunk_out,
                sa,
                ia,
                rows,
                page_table.size(0),
                page_table.size(1),
                compressed_key_cache.size(0),
                compressed_key_cache.stride(0),
                compressed_key_cache.stride(1),
                partitions,
                blocks_per_partition,
                scored,
                true,
                sb,
                ib);
          };
          return compressed_page_size == 64 ? split.template operator()<64>()
                                            : split.template operator()<128>();
        }
      }
      return compressed_page_size == 64
                 ? launch_selection_phase0<64, Short, Bounded>(
                       queue,
                       chunk_q,
                       cache_ptr,
                       table_ptr,
                       chunk_reqs,
                       chunk_positions,
                       seq_ptr,
                       chunk_out,
                       sa,
                       ia,
                       rows,
                       page_table.size(0),
                       page_table.size(1),
                       compressed_key_cache.size(0),
                       compressed_key_cache.stride(0),
                       compressed_key_cache.stride(1),
                       partitions,
                       blocks_per_partition,
                       previous,
                       has_previous)
                 : launch_selection_phase0<128, Short, Bounded>(
                       queue,
                       chunk_q,
                       cache_ptr,
                       table_ptr,
                       chunk_reqs,
                       chunk_positions,
                       seq_ptr,
                       chunk_out,
                       sa,
                       ia,
                       rows,
                       page_table.size(0),
                       page_table.size(1),
                       compressed_key_cache.size(0),
                       compressed_key_cache.stride(0),
                       compressed_key_cache.stride(1),
                       partitions,
                       blocks_per_partition,
                       previous,
                       has_previous);
    };
    auto phase = short_selection
                     ? submit_phase0.template operator()<true, true>()
                 : (bounded && q.size(0) == 1)
                     ? submit_phase0.template operator()<false, true>()
                     : submit_phase0.template operator()<false, false>();
    if (split_selection) {
      auto first = launch_selection_fold<4, false>(
          queue,
          phase,
          sa,
          ia,
          sb,
          ib,
          chunk_reqs,
          chunk_positions,
          seq_ptr,
          chunk_out,
          page_table.size(0),
          8,
          blocks_per_partition);
      phase = launch_selection_fold<8, true>(
          queue,
          first,
          sb,
          ib,
          sa,
          ia,
          chunk_reqs,
          chunk_positions,
          seq_ptr,
          chunk_out,
          page_table.size(0),
          1,
          blocks_per_partition);
    } else if (partitions > 1) {
      bool source_a = true;
      for (int pairs = partitions / 2; pairs >= 1; pairs /= 2) {
        auto submit_merge = bounded && q.size(0) == 1
                                ? launch_selection_merge<true>
                                : launch_selection_merge<false>;
        phase = source_a ? submit_merge(
                               queue,
                               phase,
                               sa,
                               ia,
                               sb,
                               ib,
                               chunk_reqs,
                               chunk_positions,
                               seq_ptr,
                               chunk_out,
                               rows,
                               page_table.size(0),
                               pairs,
                               blocks_per_partition,
                               partitions / (2 * pairs))
                         : submit_merge(
                               queue,
                               phase,
                               sb,
                               ib,
                               sa,
                               ia,
                               chunk_reqs,
                               chunk_positions,
                               seq_ptr,
                               chunk_out,
                               rows,
                               page_table.size(0),
                               pairs,
                               blocks_per_partition,
                               partitions / (2 * pairs));
        source_a = !source_a;
      }
    }
    previous = phase;
    has_previous = true;
  }
  return out;
}

}  // namespace vllm::qwen38::qsa_sycl

#ifdef QWEN38_QSA_STANDALONE
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.attr("qsa_sycl_abi_version") = 1;
  module.attr("qsa_sycl_selection_wide_scratch_abi_version") = 1;
  module.def("group_compress_v2", &vllm::qwen38::qsa_sycl::group_compress_v2);
  module.def(
      "token_split_attention_v3",
      &vllm::qwen38::qsa_sycl::token_split_attention_v3);
  module.def(
      "select_paged_tokens_v2",
      &vllm::qwen38::qsa_sycl::select_paged_tokens_v2);
  vllm::qwen38::qsa_sycl::bind_qsa_sycl_aux(module);
}
#endif  // QWEN38_QSA_STANDALONE
