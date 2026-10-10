// SPDX-License-Identifier: Apache-2.0
#include "qwen38/int4_small_m.h"

#include <cute/tensor.hpp>
#include <cute/arch/mma_xe.hpp>
#include <cute/arch/copy_xe_legacy_U16.hpp>
#include <sycl/ext/oneapi/experimental/device_architecture.hpp>

#include <cstdint>
#include <limits>
#include <optional>

namespace vllm::qwen38 {
namespace {

constexpr int kTileM = 8;
constexpr int kTileN = 16;
constexpr int kTileK = 16;
constexpr int kQGroup = 128;
#ifndef QWEN38_SMALL_M_SPLITS_LOW
  #define QWEN38_SMALL_M_SPLITS_LOW 8
#endif
#ifndef QWEN38_SMALL_M_SPLITS_HIGH
  #define QWEN38_SMALL_M_SPLITS_HIGH 4
#endif
#ifndef QWEN38_SMALL_M_NTILES
  #define QWEN38_SMALL_M_NTILES 1
#endif
constexpr int kSplitsLow = QWEN38_SMALL_M_SPLITS_LOW;
constexpr int kSplitsHigh = QWEN38_SMALL_M_SPLITS_HIGH;
constexpr int kNTiles = QWEN38_SMALL_M_NTILES;
static_assert(
    kSplitsLow >= 1 && kSplitsHigh >= 1 && kNTiles >= 1 &&
    kSplitsLow * kNTiles * kTileN <= 512 &&
    kSplitsHigh * kNTiles * kTileN <= 512);

template <int Rows, int Splits>
class Int4SmallMXmx;

template <int Rows, int Splits>
void launch_kernel(
    sycl::queue& queue,
    const sycl::half* x,
    const uint8_t* w,
    const sycl::half* s,
    sycl::half* out,
    int m,
    int n,
    int k) {
  using Mma = cute::XE_DPAS_TT<Rows, float, cute::half_t>;
  // Splits subgroups share each N tile and split its quantization groups.
  // One local-memory reduction writes the final output in this same launch.
  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> partial(
        kNTiles * Splits * kTileN * Rows, cgh);
    cgh.parallel_for<Int4SmallMXmx<Rows, Splits>>(
        sycl::nd_range<1>(int64_t(n) * Splits, kTileN * Splits * kNTiles),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(kTileN)]] {
          const int lane = int(item.get_local_linear_id()) % kTileN;
          const int subgroup = int(item.get_local_linear_id()) / kTileN;
          const int split = subgroup % Splits;
          const int ntile = subgroup / Splits;
          const int n_base =
              (int(item.get_group_linear_id()) * kNTiles + ntile) * kTileN;
          typename Mma::DVector c = {};
          // For Xe DPAS TT's VNNI B layout, a lane owns one nibble parity and
          // two columns (lane/2 and lane/2+8) for a 16-wide output tile.
          const int first_col = n_base + lane / 2;
          const uint8_t* w0 = w + int64_t(first_col) * (k / 2);
          const uint8_t* w1 = w0 + int64_t(8) * (k / 2);
          const sycl::half* s0 = s + int64_t(first_col) * (k / kQGroup);
          const sycl::half* s1 = s0 + int64_t(8) * (k / kQGroup);
          for (int group = split; group < k / kQGroup; group += Splits) {
            const sycl::half d0 = s0[group];
            const sycl::half d1 = s1[group];
#pragma unroll
            for (int tile = 0; tile < kQGroup / kTileK; ++tile) {
              const int k_base = group * kQGroup + tile * kTileK;
              typename Mma::AVector a_frag;
              typename Mma::BVector b_frag;
              if constexpr (Rows == 8) {
                cute::intel::coord_t coord_a = {k_base, 0};
                cute::XE_2D_U16x8x16_LD_N::copy(
                    x,
                    k * 2,
                    m,
                    k * 2,
                    coord_a,
                    reinterpret_cast<uint16_t*>(&a_frag));
              } else if constexpr (Rows == 4) {
                cute::intel::coord_t coord_a = {k_base, 0};
                cute::XE_2D_U16x4x16_LD_N::copy(
                    x,
                    k * 2,
                    m,
                    k * 2,
                    coord_a,
                    reinterpret_cast<uint16_t*>(&a_frag));
              } else if constexpr (Rows == 2) {
                cute::intel::coord_t coord_a = {k_base, 0};
                cute::XE_2D_U16x2x16_LD_N::copy(
                    x,
                    k * 2,
                    m,
                    k * 2,
                    coord_a,
                    reinterpret_cast<uint16_t*>(&a_frag));
              } else {
#pragma unroll
                for (int i = 0; i < Rows; ++i) {
                  const sycl::half value =
                      i < m ? x[int64_t(i) * k + k_base + lane] : sycl::half(0);
                  a_frag[i] = sycl::bit_cast<uint16_t>(value);
                }
              }
              // Eight bytes from each row supply the B VNNI fragment. Lane
              // parity chooses low/high nibble; i alternates the two rows.
              const uint32_t* packed0 =
                  reinterpret_cast<const uint32_t*>(w0 + k_base / 2);
              const uint32_t* packed1 =
                  reinterpret_cast<const uint32_t*>(w1 + k_base / 2);
              const uint32_t raw00 = packed0[0], raw01 = packed0[1];
              const uint32_t raw10 = packed1[0], raw11 = packed1[1];
#pragma unroll
              for (int i = 0; i < 16; ++i) {
                const int byte = i / 2;
                const bool second = (i & 1) != 0;
                const uint32_t packed = second ? (byte < 4 ? raw10 : raw11)
                                               : (byte < 4 ? raw00 : raw01);
                const int quant =
                    (int((packed >> (8 * (byte & 3) + 4 * (lane & 1))) & 15) -
                     8);
                b_frag[i] = sycl::bit_cast<uint16_t>(
                    sycl::half(float(quant) * float(second ? d1 : d0)));
              }
              Mma::fma(c, a_frag, b_frag, c);
            }
          }
          const int base = ((ntile * Splits + split) * kTileN + lane) * Rows;
#pragma unroll
          for (int row = 0; row < Rows; ++row)
            partial[base + row] = c[row];
          item.barrier(sycl::access::fence_space::local_space);
          if (split == 0) {
#pragma unroll
            for (int row = 0; row < Rows; ++row) {
              if (row < m) {
                float sum = 0.0f;
#pragma unroll
                for (int part = 0; part < Splits; ++part) {
                  sum += partial
                      [((ntile * Splits + part) * kTileN + lane) * Rows + row];
                }
                out[int64_t(row) * n + n_base + lane] = sycl::half(sum);
              }
            }
          }
        });
  });
}

}  // namespace

bool launch_int4_small_m(
    sycl::queue& queue,
    const sycl::half* x,
    const uint8_t* w,
    const sycl::half* s,
    sycl::half* out,
    int m,
    int n,
    int k) {
  if (m < 2 || m > kTileM || n < 256 || n % (kTileN * kNTiles) != 0 || k <= 0 ||
      k > std::numeric_limits<int>::max() / 2 || k % kQGroup != 0 || !x || !w ||
      !s || !out || (reinterpret_cast<uintptr_t>(w) & 3u) != 0 ||
      (reinterpret_cast<uintptr_t>(x) & 63u) != 0) {
    return false;
  }
  namespace sx = sycl::ext::oneapi::experimental;
  // Only immutable device capability is cached; no tensor or request state.
  static thread_local std::optional<sycl::device> cached_device;
  static thread_local bool cached_supported = false;
  const sycl::device device = queue.get_device();
  if (!cached_device || *cached_device != device) {
    cached_supported = device.get_info<sx::info::device::architecture>() ==
                       sx::architecture::intel_gpu_bmg_g31;
    cached_device = device;
  }
  if (!cached_supported) {
    return false;
  }
  if (m <= 2) {
    launch_kernel<2, kSplitsLow>(queue, x, w, s, out, m, n, k);
  } else if (m <= 4) {
    launch_kernel<4, kSplitsHigh>(queue, x, w, s, out, m, n, k);
  } else {
    launch_kernel<8, kSplitsHigh>(queue, x, w, s, out, m, n, k);
  }
  return true;
}

}  // namespace vllm::qwen38
