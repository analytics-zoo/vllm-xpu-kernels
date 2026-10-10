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
template <int PageSize, bool BlockRead, int Subgroup>
class AttentionPhase0;
template <int PageSize>
class AttentionPrefillQkDpasPhase0;
template <int Columns>
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

template <int PageSize, bool BlockRead, int Subgroup>
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
  constexpr int workers = 4;
  constexpr int values_per_lane = kHeadDim / Subgroup;
  const int64_t groups = static_cast<int64_t>(rows) * heads * partial_count;
  return queue.submit([&](sycl::handler& cgh) {
    if (has_previous) cgh.depends_on(previous);
    sycl::local_accessor<float, 1> scratch(
        sycl::range<1>(workers * (kPartialStride)), cgh);
    cgh.parallel_for<AttentionPhase0<PageSize, BlockRead, Subgroup>>(
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
            const int64_t base = (static_cast<int64_t>(physical) * PageSize +
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

template <int Columns>
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
    cgh.parallel_for<AttentionPhase1<Columns>>(
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
          for (int p = 0; p < partial_count; ++p) {
            const float* part = head_partials + p * kPartialStride;
#pragma unroll
            for (int d = 0; d < values; ++d) {
              accumulation[d] +=
                  weights[p] * part[2 + tile * Columns + lane * values + d];
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
          q.size(0) <= 4096 && (q.size(1) == 3 || q.size(1) == 6) &&
          q.size(2) == kHeadDim && q.is_contiguous(),
      "q must be contiguous FP16 [1..4096,3|6,256]");
  TORCH_CHECK(
      page_size == 256 || page_size == 512, "page_size must be 256 or 512");
  TORCH_CHECK(
      packed_kv.dim() == 4 && packed_kv.size(0) > 0 && packed_kv.size(1) == 1 &&
          packed_kv.size(2) == page_size && packed_kv.size(3) == 512 &&
          packed_kv.stride(0) == page_size * 512 &&
          packed_kv.stride(2) == 512 && packed_kv.stride(3) == 1,
      "packed_kv must expose [pages,1,page_size,512]");
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
    if (total_rows == 1) {
      return launch_phase0<
          decltype(page_tag)::value,
          decltype(block_tag)::value,
          32>(
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
        16>(
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

template <int PageSize>
class SelectionPhase0;
class SelectionMerge;

template <int PageSize>
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
    int blocks_per_partition,
    sycl::event previous,
    bool has_previous) {
  return queue.submit([&](sycl::handler& cgh) {
    if (has_previous) cgh.depends_on(previous);
    sycl::local_accessor<half, 1> query_shared(
        sycl::range<1>(kIndexHeads * kIndexDim), cgh);
    sycl::local_accessor<float, 1> scores(sycl::range<1>(kSortSlots), cgh);
    sycl::local_accessor<int32_t, 1> indices(sycl::range<1>(kSortSlots), cgh);
    cgh.parallel_for<SelectionPhase0<PageSize>>(
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
          for (int base = 0; base < blocks_per_partition; base += kTopBlocks) {
            for (int b = subgroup; b < kTopBlocks;
                 b += kSelectionWg / kSelectionSg) {
              const int64_t logical =
                  static_cast<int64_t>(partition) * blocks_per_partition +
                  base + b;
              const int64_t logical_page = logical / PageSize;
              bool valid = valid_req && logical < visible_blocks &&
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
            item.barrier(sycl::access::fence_space::local_space);
            // Sort ascending score, descending index. Reverse the high half
            // for final descending score and deterministic low-index ties.
            const bool first_chunk = base == 0;
            const int sort_offset = first_chunk ? kTopBlocks : 0;
            const int sort_size = first_chunk ? kTopBlocks : kSortSlots;
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
            if (lane < kTopBlocks) {
              const int from = kSortSlots - 1 - lane;
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
    int pairs) {
  return queue.submit([&](sycl::handler& cgh) {
    cgh.depends_on(previous);
    sycl::local_accessor<float, 1> scores(sycl::range<1>(kSortSlots), cgh);
    sycl::local_accessor<int32_t, 1> indices(sycl::range<1>(kSortSlots), cgh);
    cgh.parallel_for<SelectionMerge>(
        sycl::nd_range<1>(
            sycl::range<1>(rows * pairs * kMergeWg), sycl::range<1>(kMergeWg)),
        [=](sycl::nd_item<1> item) {
          const int row = item.get_group_linear_id() / pairs;
          const int pair = item.get_group_linear_id() % pairs;
          const int lane = item.get_local_linear_id();
          const int base = row * kWorkspaceRowStride + pair * kSortSlots;
          scores[lane] = src_scores[base + lane];
          indices[lane] = src_indices[base + lane];
          scores[kSortSlots - 1 - lane] = src_scores[base + kTopBlocks + lane];
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
  const int partitions = max_seq_len >= 4096 ? kSelectionPartitions : 1;
  const int max_blocks = static_cast<int>((max_seq_len + 3) / 4);
  const int per_partition = (max_blocks + partitions - 1) / partitions;
  const int blocks_per_partition =
      ((per_partition + kTopBlocks - 1) / kTopBlocks) * kTopBlocks;
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
    auto phase = compressed_page_size == 64
                     ? launch_selection_phase0<64>(
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
                     : launch_selection_phase0<128>(
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
    if (partitions > 1) {
      bool source_a = true;
      for (int pairs = partitions / 2; pairs >= 1; pairs /= 2) {
        phase = source_a ? launch_selection_merge(
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
                               pairs)
                         : launch_selection_merge(
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
                               pairs);
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
