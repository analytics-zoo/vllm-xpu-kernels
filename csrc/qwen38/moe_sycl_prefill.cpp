// SPDX-License-Identifier: Apache-2.0
#include "qwen38/moe_sycl_prefill.h"

#include <cute/tensor.hpp>
#include <cute/arch/mma_xe.hpp>
#include <cute/arch/copy_xe_legacy_U16.hpp>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace vllm::qwen38::moe_sycl {
namespace {

constexpr int kExperts = 512;
constexpr int kN = 2560;
constexpr int kTileM = 8;
constexpr int kTileN = 16;
constexpr int kNGroups = kN / kTileN;

class MoePrefillPrefix;
template <int K>
class MoePrefillDown;
template <int K>
class MoePrefillDownByActiveExpert;

void launch_prefix(sycl::queue& queue, const PrefillDown& a, bool active_mode) {
  queue.submit([&](sycl::handler& cgh) {
    const auto* counts = a.counts;
    auto* rows = a.row_prefix;
    auto* tiles = a.tile_prefix;
    cgh.parallel_for<MoePrefillPrefix>(
        sycl::nd_range<1>(kExperts, kExperts), [=](sycl::nd_item<1> item) {
          const int e = int(item.get_local_linear_id());
          const int64_t raw_count = counts[e];
          const int64_t count = sycl::max(raw_count, int64_t(0));
          const int64_t tile_count = (count + kTileM - 1) / kTileM;
          const auto group = item.get_group();
          const int any_negative = sycl::reduce_over_group(
              group, raw_count < 0 ? 1 : 0, sycl::maximum<int>());
          // A malformed int32 sum can wrap to m and pass the old sentinel,
          // leaving negative row offsets. Scan wide and narrow only in bounds.
          const int64_t row_start = sycl::exclusive_scan_over_group(
              group, count, sycl::plus<int64_t>());
          rows[e] = row_start <= a.m ? int(row_start) : -1;
          if (active_mode) {
            const int active_rank = sycl::exclusive_scan_over_group(
                group, count > 0 ? 1 : 0, sycl::plus<int>());
            if (count > 0) tiles[active_rank] = e;
            if (e == kExperts - 1)
              tiles[kExperts] = active_rank + (count > 0 ? 1 : 0);
          } else {
            const int64_t tile_start = sycl::exclusive_scan_over_group(
                group, tile_count, sycl::plus<int64_t>());
            const int64_t bound = int64_t(a.m) + kExperts;
            tiles[e] = tile_start <= bound ? int(tile_start) : -1;
            if (e == kExperts - 1)
              tiles[kExperts] = tile_start + tile_count <= bound
                                    ? int(tile_start + tile_count)
                                    : -1;
          }
          if (e == kExperts - 1) {
            rows[kExperts] = any_negative || row_start + count != a.m
                                 ? -1
                                 : a.m;
          }
        });
  });
}

template <int K>
inline void compute_tile(
    const PrefillDown& a,
    int lane,
    int expert,
    int row0,
    int row1,
    int n_base) {
  using Mma = cute::XE_DPAS_TT<kTileM, float, cute::half_t>;
  constexpr int groups = (K + 127) / 128;
  const int col0 = n_base + lane / 2;
  const auto* w0 = a.w2 + (int64_t(expert) * kN + col0) * (K / 2);
  const auto* w1 = w0 + int64_t(8) * (K / 2);
  const auto* s0 = a.s2 + (int64_t(expert) * kN + col0) * groups;
  const auto* s1 = s0 + int64_t(8) * groups;
  typename Mma::DVector accum = {};
  for (int group = 0; group < groups; ++group) {
    const sycl::half scale0 = s0[group];
    const sycl::half scale1 = s1[group];
    for (int kb = group * 128; kb < sycl::min((group + 1) * 128, K); kb += 16) {
      typename Mma::AVector a_frag;
      typename Mma::BVector b_frag;
      cute::intel::coord_t coord_a = {kb, row0};
      cute::XE_2D_U16x8x16_LD_N::copy(
          a.x,
          K * 2,
          a.m,
          K * 2,
          coord_a,
          reinterpret_cast<uint16_t*>(&a_frag));
      const auto* p0 = reinterpret_cast<const uint32_t*>(w0 + kb / 2);
      const auto* p1 = reinterpret_cast<const uint32_t*>(w1 + kb / 2);
      const uint32_t raw00 = p0[0], raw01 = p0[1];
      const uint32_t raw10 = p1[0], raw11 = p1[1];
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const int byte = i / 2;
        const bool second = (i & 1) != 0;
        const uint32_t raw =
            second ? (byte < 4 ? raw10 : raw11) : (byte < 4 ? raw00 : raw01);
        const int nib = int((raw >> (8 * (byte & 3) + 4 * (lane & 1))) & 15u);
        const int signed4 = (nib ^ 8) - 8;
        b_frag[i] = sycl::bit_cast<uint16_t>(
            sycl::half(float(signed4) * float(second ? scale1 : scale0)));
      }
      Mma::fma(accum, a_frag, b_frag, accum);
    }
  }
#pragma unroll
  for (int r = 0; r < kTileM; ++r) {
    if (row0 + r < row1)
      a.output[int64_t(row0 + r) * kN + n_base + lane] = sycl::half(accum[r]);
  }
}

inline void mark_invalid_counts(const PrefillDown& a, sycl::nd_item<1> item) {
  const auto stride = int64_t(item.get_global_range(0));
  for (int64_t index = item.get_global_linear_id(); index < int64_t(a.m) * kN;
       index += stride)
    a.output[index] = sycl::half(std::numeric_limits<float>::quiet_NaN());
}

template <int K>
void launch_dpas(sycl::queue& queue, const PrefillDown& a) {
  // Sum_e ceil(count[e]/8) <= ceil(m/8) + E - 1.
  const int64_t upper_tiles =
      (int64_t(a.m) + kTileM - 1) / kTileM + kExperts - 1;
  queue.submit([&](sycl::handler& cgh) {
    const auto* rows = a.row_prefix;
    const auto* tiles = a.tile_prefix;
    cgh.parallel_for<MoePrefillDown<K>>(
        sycl::nd_range<1>(upper_tiles * kNGroups * kTileN, kTileN),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(kTileN)]] {
          if (rows[kExperts] != a.m) {
            mark_invalid_counts(a, item);
            return;
          }
          const int lane = int(item.get_local_linear_id());
          const int64_t work = item.get_group_linear_id();
          const int tile = int(work / kNGroups);
          if (tile >= tiles[kExperts]) return;
          const int n_base = int(work % kNGroups) * kTileN;
          int lo = 0, hi = kExperts;
          while (lo + 1 < hi) {
            const int mid = (lo + hi) / 2;
            if (tiles[mid] <= tile)
              lo = mid;
            else
              hi = mid;
          }
          const int expert = lo;
          const int row0 = rows[expert] + (tile - tiles[expert]) * kTileM;
          const int row1 = sycl::min(rows[expert + 1], row0 + kTileM);
          if (row0 >= row1) return;
          compute_tile<K>(a, lane, expert, row0, row1, n_base);
        });
  });
}

template <int K>
void launch_dpas_by_active_expert(sycl::queue& queue, const PrefillDown& a) {
  constexpr int kWorkerExperts = 128;
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<MoePrefillDownByActiveExpert<K>>(
        sycl::nd_range<1>(int64_t(kWorkerExperts) * kNGroups * kTileN, kTileN),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(kTileN)]] {
          if (a.row_prefix[kExperts] != a.m) {
            mark_invalid_counts(a, item);
            return;
          }
          const int lane = int(item.get_local_linear_id());
          const int work = int(item.get_group_linear_id());
          const int worker = work / kNGroups;
          const int n_base = (work % kNGroups) * kTileN;
          const int active_count = a.tile_prefix[kExperts];
          for (int index = worker; index < active_count;
               index += kWorkerExperts) {
            const int expert = a.tile_prefix[index];
            const int end = a.row_prefix[expert + 1];
            for (int row = a.row_prefix[expert]; row < end; row += kTileM)
              compute_tile<K>(
                  a, lane, expert, row, sycl::min(row + kTileM, end), n_base);
          }
        });
  });
}

}  // namespace

bool try_prefill_down(sycl::queue& queue, const PrefillDown& a) {
  if (!queue.is_in_order() || a.m < 0 ||
      a.m > std::numeric_limits<int>::max() - 8192 ||
      (a.k != 80 && a.k != 160)) {
    return false;
  }
  // Empty tensors legitimately have null pointers; no storage is accessed.
  if (a.m == 0) return true;
  if (!a.x || !a.w2 || !a.s2 || !a.counts || !a.output || !a.row_prefix ||
      !a.tile_prefix ||
      (reinterpret_cast<uintptr_t>(a.x) & 63u) != 0 ||
      (reinterpret_cast<uintptr_t>(a.w2) & 3u) != 0) {
    return false;
  }
  const bool long_prefill = a.m >= 512;
  launch_prefix(queue, a, long_prefill);
  // Long prefill reuses tile_prefix as a compact active-expert list, avoiding
  // workgroups for empty experts without a new workspace allocation.
  if (a.k == 80) {
    if (long_prefill)
      launch_dpas_by_active_expert<80>(queue, a);
    else
      launch_dpas<80>(queue, a);
  } else {
    if (long_prefill)
      launch_dpas_by_active_expert<160>(queue, a);
    else
      launch_dpas<160>(queue, a);
  }
  return true;
}

}  // namespace vllm::qwen38::moe_sycl
