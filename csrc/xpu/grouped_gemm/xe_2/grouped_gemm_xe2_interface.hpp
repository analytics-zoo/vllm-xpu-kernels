/***************************************************************************************************
 * Copyright (c) 2025 Intel Corporation. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief CUTLASS Intel BMG MoE API example based on sycl-tla Group GEMM

*/
#include <torch/all.h>
#include "csrc/utils.h"

#include <cute/tensor.hpp>
#include <limits>
#include <random>

#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>
#include <sycl/sycl.hpp>

#include <cute/tensor.hpp>

#include "cutlass/kernel_hardware_info.h"
#include "cutlass/platform/platform.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"

#include "gemm_xe2_policy.hpp"
#include "grouped_gemm_xe2.hpp"

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

namespace MoE {
using namespace cute;

// type tag to define a unique sycl kernel name
template <typename, typename, typename, typename, char, char, class>
class GemmCuteName;

template <typename, class>
class DenseFp8BlockGemmCuteName;

class Fp8BlockDequantXe2Name;

void Fp8BlockDequantLauncher(
    sycl::queue& stream,
    const float_e4m3_t* weights,
    const float* scales,
    half_t* output,
    int64_t gemm_n,
    int64_t gemm_k) {
  namespace esimd = sycl::ext::intel::esimd;
  namespace xmem = sycl::ext::intel::experimental::esimd;
  constexpr int block_size = 128;
  const int64_t blocks_k = gemm_k / block_size;
  const int64_t row_segments = gemm_n * blocks_k;

  stream.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<Fp8BlockDequantXe2Name>(
        sycl::range<1>(row_segments),
        [=](sycl::id<1> item) SYCL_ESIMD_KERNEL {
          const int64_t segment = item[0];
          const int64_t row = segment / blocks_k;
          const int64_t block_k = segment % blocks_k;
          const int64_t block_id = (row / block_size) * blocks_k + block_k;
          const float scale = scales[block_id];
          const int64_t index = row * gemm_k + block_k * block_size;
          auto bytes = xmem::lsc_block_load<
              uint8_t,
              block_size,
              xmem::lsc_data_size::default_size,
              xmem::cache_hint::streaming,
              xmem::cache_hint::cached>(
              reinterpret_cast<const uint8_t*>(weights) + index);

          esimd::simd<uint16_t, block_size> fp8 = 0;
          fp8.template bit_cast_view<uint8_t>()
              .template select<block_size, 2>(0) = bytes;
          esimd::simd<uint16_t, block_size> bits = (fp8 & 0x7f) << 7;
          bits += 0x2000;
          bits |= (fp8 & 0x80) << 8;

          const auto sign = (fp8 & 0x80) << 8;
          const auto mantissa = fp8 & 0x07;
          const auto subnormal = (fp8 & 0x78) == 0;
          bits.merge(sign, subnormal && mantissa == 0);
          bits.merge(sign | 0x1800, subnormal && mantissa == 1);
          bits.merge(sign | 0x1c00, subnormal && mantissa == 2);
          bits.merge(sign | 0x1e00, subnormal && mantissa == 3);
          bits.merge(sign | 0x2000, subnormal && mantissa == 4);
          bits.merge(sign | 0x2100, subnormal && mantissa == 5);
          bits.merge(sign | 0x2200, subnormal && mantissa == 6);
          bits.merge(sign | 0x2300, subnormal && mantissa == 7);
          bits.merge(sign | 0x7e00, (fp8 & 0x7f) == 0x7f);

          esimd::simd<sycl::half, block_size> values =
              bits.template bit_cast_view<sycl::half>();
          esimd::simd<float, block_size> scaled = values;
          scaled *= scale;
          xmem::lsc_block_store<
              sycl::half,
              block_size,
              xmem::lsc_data_size::default_size,
              xmem::cache_hint::streaming,
              xmem::cache_hint::write_back>(
              reinterpret_cast<sycl::half*>(output) + index,
              esimd::simd<sycl::half, block_size>(scaled));
        });
  });
}

template <typename ElementA, class policy>
void DenseFp8BlockGEMMLauncher(
    sycl::queue& stream,
    const ElementA* activations,
    const float_e4m3_t* weights,
    const float* scales,
    ElementA* outputs,
    int gemm_m,
    int gemm_n,
    int gemm_k) {
  auto op = XE_DPAS_TT<8, float, ElementA>{};
  using WGTile = typename policy::WGTile;
  using SGLayout = typename policy::SGLayout;
  using MMA = typename TiledMMAHelper<
      MMA_Atom<decltype(op)>,
      Layout<WGTile>,
      SGLayout>::TiledMMA;
  auto mma = MMA{};

  const int threads_per_workgroup = size(mma);
  const int tiles_m = ceil_div(gemm_m, int(size<0>(WGTile{})));
  const int tiles_n = ceil_div(gemm_n, int(size<1>(WGTile{})));
  sycl::range<3> local(1, 1, threads_per_workgroup);
  sycl::range<3> global(1, tiles_m * tiles_n, 1);

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;
  syclex::properties kernel_props{
      syclex::sub_group_size<16>, intelex::grf_size<256>};

  using GmemTiledCopyA = typename policy::GmemTiledCopyA;
  using GmemTiledCopyB = typename policy::GmemTiledCopyB;
  using GmemTiledCopyD = typename policy::GmemTiledCopyD;

  stream.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<DenseFp8BlockGemmCuteName<ElementA, policy>>(
        sycl::nd_range<3>{global * local, local}, kernel_props, [=](auto item) {
          const int tile_id = item.get_group_linear_id();
          const int tile_m = tile_id / tiles_n;
          const int tile_n = tile_id % tiles_n;
          auto A = make_moe_tensor<ElementA, 'R'>(
              const_cast<ElementA*>(activations), gemm_m, gemm_k);
          auto B = make_moe_tensor<float_e4m3_t, 'R'>(
              const_cast<float_e4m3_t*>(weights), gemm_n, gemm_k);
          auto D = make_moe_tensor<ElementA, 'R'>(
              outputs, gemm_m, gemm_n);
          auto tile_coord = make_coord(tile_m, tile_n, _, 0);
          xe_gemm_4bits<
              GmemTiledCopyA,
              GmemTiledCopyB,
              GmemTiledCopyD,
              128>(A, B, scales, static_cast<ElementA*>(nullptr), D,
                   tile_coord, mma);
        });
  });
}

template <
    char layoutA,
    char layoutB,
    class policy,
    typename ElementA,
    typename ElementB,
    typename ElementS,
    typename ElementBI,
    typename ElementD>
void MoEGEMMLauncher(
    sycl::queue& stream,
    const ElementA* activations,
    const ElementB* weights,
    const ElementS* scales,
    const ElementBI* bias,
    ElementD* outputs,
    const int gemm_n,
    const int gemm_k,
    const int* rows_per_expert,
    const int num_experts,
    const int group_size,
    int32_t* atomic_buffer) {
  using ElementA_non_CV = cutlass::platform::remove_cv_t<ElementA>;
  auto op = XE_DPAS_TT<8, float, ElementA_non_CV>{};

  using WGTile = typename policy::WGTile;
  using SGLayout = typename policy::SGLayout;
  using MMA = typename TiledMMAHelper<
      MMA_Atom<decltype(op)>,
      Layout<WGTile>,
      SGLayout>::TiledMMA;
  auto mma = MMA{};

  int sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  auto MaxThreadsPerWorkgroup = size(mma);

  static constexpr int MaxThreadsPerSM = 512;

  TORCH_CHECK(
      MaxThreadsPerSM % MaxThreadsPerWorkgroup == 0,
      "MaxThreadsPerSM must be divisible by MaxThreadsPerWorkgroup")

  sycl::range<3> local(1, 1, MaxThreadsPerWorkgroup);
  sycl::range<3> global(
      1, sm_count * MaxThreadsPerSM / MaxThreadsPerWorkgroup, 1);

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;

  syclex::properties kernel_props{
      syclex::sub_group_size<16>, intelex::grf_size<256>};

  using GmemTiledCopyA = typename policy::GmemTiledCopyA;
  using GmemTiledCopyB = typename policy::GmemTiledCopyB;
  using GmemTiledCopyD = typename policy::GmemTiledCopyD;

  stream.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<int32_t, 1> local_mem(sycl::range<1>(1), cgh);
    cgh.parallel_for<GemmCuteName<
        ElementA,
        ElementB,
        ElementS,
        ElementD,
        layoutA,
        layoutB,
        policy>>(
        sycl::nd_range<3>{global * local, local}, kernel_props, [=](auto) {
          MoE::MoEGEMM<
              GmemTiledCopyA,
              GmemTiledCopyB,
              GmemTiledCopyD,
              layoutA,
              layoutB,
              'R'>(
              activations,
              weights,
              scales,
              bias,
              outputs,
              mma,
              rows_per_expert,
              num_experts,
              group_size,
              gemm_n,
              gemm_k,
              atomic_buffer,
              local_mem);
        });
  });
}

at::Tensor cutlass_grouped_gemm_xe2_impl(
    at::Tensor& ptr_A,
    at::Tensor& ptr_B,
    const c10::optional<at::Tensor>& ptr_scales,
    const c10::optional<at::Tensor>& ptr_bias,
    at::Tensor& ptr_D,
    at::Tensor& rows_per_expert,
    int64_t N,
    int64_t K,
    int64_t num_experts,
    bool is_B_int4,
    bool is_B_mxfp4,
    bool is_B_fp8_block) {
  auto& dpcpp_queue =
      at::xpu::getCurrentXPUStream(ptr_A.device().index()).queue();
  auto A_dtype = ptr_A.dtype();
  auto B_dtype = ptr_B.dtype();
  bool is_weight_fp8 =
      ((B_dtype == at::kFloat8_e4m3fn) || (B_dtype == at::kFloat8_e5m2));

  TORCH_CHECK(N % 32 == 0, "N must be divisible by 32");

  TORCH_CHECK(ptr_A.dim() == 2, "ptr_A must be 2D [Total_M, K]");
  TORCH_CHECK(ptr_B.dim() == 3, "ptr_B must be 3D [num_experts, K, N]");
  TORCH_CHECK(ptr_D.dim() == 2, "ptr_D must be 2D [Total_M, N]");
  if (ptr_bias.has_value()) {
    TORCH_CHECK(ptr_bias->dim() == 2, "ptr_bias must be 2D [num_experts, N]");
  }

  TORCH_CHECK(ptr_A.is_contiguous(), "ptr_A must be contiguous");
  TORCH_CHECK(ptr_B.is_contiguous(), "ptr_B must be contiguous");
  TORCH_CHECK(ptr_D.is_contiguous(), "ptr_D must be contiguous");
  if (ptr_bias.has_value()) {
    TORCH_CHECK(ptr_bias->is_contiguous(), "ptr_bias must be contiguous");
  }

  int A_total_M = ptr_A.size(0);
  int A_K = ptr_A.size(1);

  int B_E = ptr_B.size(0);
  int B_K = ptr_B.size(1);
  int B_N = ptr_B.size(2);
  if (is_B_int4 || is_B_mxfp4) {
    B_K = ptr_B.size(2) * 2;
    B_N = ptr_B.size(1);
  } else if (is_B_fp8_block) {
    B_K = ptr_B.size(2);
    B_N = ptr_B.size(1);
  }

  int D_total_M = ptr_D.size(0);
  int D_N = ptr_D.size(1);
  int group_size = -1;
  int A_avg_M = A_total_M / num_experts;

  TORCH_CHECK(B_E == num_experts, "ptr_B.size(0) must match num_experts");
  TORCH_CHECK(A_total_M == D_total_M, "ptr_A.size(0) must match ptr_D.size(0)");
  TORCH_CHECK(A_K == B_K && B_K == K, "ptr_A.size(1) must match ptr_B.size(1)");
  TORCH_CHECK(B_N == D_N && D_N == N, "ptr_B.size(2) must match ptr_D.size(1)");
  if (ptr_bias.has_value()) {
    TORCH_CHECK(
        ptr_bias->size(0) == num_experts,
        "ptr_bias.size(0) must match num_experts");
    TORCH_CHECK(ptr_bias->size(1) == N, "ptr_bias.size(1) must match N");
  }

  at::Tensor atomic_buffer =
      at::empty({static_cast<long>(1)}, ptr_A.options().dtype(at::kInt));

#define MoEGEMMLauncherCallER(                                                 \
    LayoutA, LayoutB, Policy, ElementA, ElementB, ElementS)                    \
  MoEGEMMLauncher<LayoutA, LayoutB, Policy>(                                   \
      dpcpp_queue,                                                             \
      reinterpret_cast<ElementA*>(ptr_A.data_ptr()),                           \
      reinterpret_cast<ElementB*>(ptr_B.data_ptr()),                           \
      ptr_scales.has_value()                                                   \
          ? reinterpret_cast<ElementS*>(ptr_scales->data_ptr())                \
          : static_cast<ElementS*>(nullptr),                                   \
      ptr_bias.has_value() ? reinterpret_cast<ElementA*>(ptr_bias->data_ptr()) \
                           : static_cast<ElementA*>(nullptr),                  \
      reinterpret_cast<ElementA*>(ptr_D.data_ptr()),                           \
      N,                                                                       \
      K,                                                                       \
      reinterpret_cast<int*>(rows_per_expert.data_ptr()),                      \
      num_experts,                                                             \
      group_size,                                                              \
      static_cast<int*>(atomic_buffer.data_ptr()));

  if (is_B_int4 || is_B_mxfp4) {
    TORCH_CHECK(ptr_scales.has_value(), "w8a16 grouped gemm must have scales");
    TORCH_CHECK(ptr_scales->is_contiguous(), "ptr_scales must be contiguous");
    TORCH_CHECK(
        ptr_scales->dim() == 3,
        "ptr_scales of int4 must be 3D [num_experts, group_num, N]");
    TORCH_CHECK(
        ptr_scales->size(0) == num_experts,
        "ptr_scales.size(0) of int4 must match num_experts");
    TORCH_CHECK(
        K % ptr_scales->size(2) == 0,
        "ptr_scales.size(2) of int4 must be divisible by K");
    TORCH_CHECK(
        ptr_scales->size(1) == N, "ptr_scales.size(1) of int4 must match N");
    int group_num = ptr_scales->size(2);
    group_size = K / group_num;

    TORCH_CHECK(
        group_size == 32 || group_size == 64 || group_size == 128 ||
            group_size == 256,
        "group_size must be 32, 64, 128 or 256");

#define W4A16LauncherCallER(policy)                                         \
  if (is_B_int4) {                                                          \
    if (A_dtype == at::kBFloat16) {                                         \
      using scalar_t = bfloat16_t;                                          \
      MoEGEMMLauncherCallER('R', 'C', policy, scalar_t, uint8_t, scalar_t); \
    } else if (A_dtype == at::kHalf) {                                      \
      using scalar_t = half_t;                                              \
      MoEGEMMLauncherCallER('R', 'C', policy, scalar_t, uint8_t, scalar_t); \
    }                                                                       \
  } else if (is_B_mxfp4) {                                                  \
    if (A_dtype == at::kBFloat16) {                                         \
      using scalar_t = bfloat16_t;                                          \
      MoEGEMMLauncherCallER('R', 'C', policy, scalar_t, uint8_t, uint8_t);  \
    } else if (A_dtype == at::kHalf) {                                      \
      using scalar_t = half_t;                                              \
      MoEGEMMLauncherCallER('R', 'C', policy, scalar_t, uint8_t, uint8_t);  \
    }                                                                       \
  }

    if (A_avg_M <= 4) {
      using policy = w4a16_policy_m_8;
      W4A16LauncherCallER(policy);
    } else if (A_avg_M <= 8) {
      using policy = w4a16_policy_m_16;
      W4A16LauncherCallER(policy);
    } else if (A_avg_M <= 128) {
      using policy = w4a16_policy_m_32;
      W4A16LauncherCallER(policy);
    } else {
      using policy = w4a16_policy;
      W4A16LauncherCallER(policy);
    }
#undef W4A16LauncherCallER
  } else if (is_weight_fp8) {
    TORCH_CHECK(ptr_scales.has_value(), "w8a16 grouped gemm must have scales");
    TORCH_CHECK(ptr_scales->is_contiguous(), "ptr_scales must be contiguous");
    TORCH_CHECK(ptr_scales->dtype() == at::kFloat, "ptr_scales must be float");

#define W8A16LauncherCallER(policy)                                          \
  if (B_dtype == at::kFloat8_e4m3fn && A_dtype == at::kHalf) {               \
    using scalar_t = half_t;                                                 \
    if (is_B_fp8_block) {                                                    \
      MoEGEMMLauncherCallER('R', 'C', policy, scalar_t, float_e4m3_t, float); \
    } else {                                                                 \
      MoEGEMMLauncherCallER('R', 'R', policy, scalar_t, float_e4m3_t, float); \
    }                                                                        \
  } else if (B_dtype == at::kFloat8_e5m2 && A_dtype == at::kHalf) {          \
    using scalar_t = half_t;                                                 \
    MoEGEMMLauncherCallER('R', 'R', policy, scalar_t, float_e5m2_t, float);  \
  } else if (B_dtype == at::kFloat8_e4m3fn && A_dtype == at::kBFloat16) {    \
    using scalar_t = bfloat16_t;                                             \
    if (is_B_fp8_block) {                                                    \
      MoEGEMMLauncherCallER('R', 'C', policy, scalar_t, float_e4m3_t, float); \
    } else {                                                                 \
      MoEGEMMLauncherCallER('R', 'R', policy, scalar_t, float_e4m3_t, float); \
    }                                                                        \
  } else if (B_dtype == at::kFloat8_e5m2 && A_dtype == at::kBFloat16) {      \
    using scalar_t = bfloat16_t;                                             \
    MoEGEMMLauncherCallER('R', 'R', policy, scalar_t, float_e5m2_t, float);  \
  }

    if (is_B_fp8_block) {
      TORCH_CHECK(
          B_dtype == at::kFloat8_e4m3fn,
          "block-scaled grouped GEMM only supports float8_e4m3fn weights");
      TORCH_CHECK(
          N % 128 == 0 && K % 128 == 0,
          "block-scaled grouped GEMM requires N and K divisible by 128");
      TORCH_CHECK(
          ptr_scales->dim() == 3,
          "block scales must be 3D [num_experts, N / 128, K / 128]");
      TORCH_CHECK(
          ptr_scales->size(0) == num_experts &&
              ptr_scales->size(1) == N / 128 &&
              ptr_scales->size(2) == K / 128,
          "block scales must have shape [num_experts, N / 128, K / 128]");
      group_size = 128;
    } else {
      TORCH_CHECK(
          ptr_scales->dim() == 1,
          "ptr_scales of fp8 must be 1D [num_experts]");
      TORCH_CHECK(
          ptr_scales->size(0) == num_experts,
          "ptr_scales.size(0) of fp8 must match num_experts");
    }

    if (A_avg_M <= 8) {
      using policy = w8a16_policy_m_16;
      W8A16LauncherCallER(policy);
    } else if (A_avg_M <= 32) {
      using policy = w8a16_policy_m_32;
      W8A16LauncherCallER(policy);
    } else {
      using policy = w8a16_policy;
      W8A16LauncherCallER(policy);
    }
#undef W8A16LauncherCallER
  } else {
    TORCH_CHECK(
        !ptr_scales.has_value(), "w16a16 grouped gemm must not have scales");

#define W16A16LauncherCallER(policy)                                       \
  if (A_dtype == at::kBFloat16) {                                          \
    using scalar_t = bfloat16_t;                                           \
    MoEGEMMLauncherCallER('R', 'R', policy, scalar_t, scalar_t, scalar_t); \
  } else if (A_dtype == at::kHalf) {                                       \
    using scalar_t = half_t;                                               \
    MoEGEMMLauncherCallER('R', 'R', policy, scalar_t, scalar_t, scalar_t); \
  }

    if (A_avg_M <= 8) {
      using policy = w16a16_policy_m_16;
      W16A16LauncherCallER(policy);
    } else if (A_avg_M <= 16) {
      using policy = w16a16_policy_m_32;
      W16A16LauncherCallER(policy);
    } else {
      if (B_N <= 64) {
        using policy = w16a16_policy_n_64;
        W16A16LauncherCallER(policy);
      } else if (B_N <= 512) {
        using policy = w16a16_policy_n_128;
        W16A16LauncherCallER(policy);
      } else {
        using policy = w16a16_policy;
        W16A16LauncherCallER(policy);
      }
    }
#undef W16A16LauncherCallER
  }
#undef MoEGEMMLauncherCallER
  return ptr_D;
}

at::Tensor fp8_block_gemm_xe2_impl(
    const at::Tensor& ptr_A,
    const at::Tensor& ptr_B,
    const at::Tensor& ptr_scales) {
  TORCH_CHECK(
      ptr_A.scalar_type() == at::kHalf ||
          ptr_A.scalar_type() == at::kBFloat16,
      "activation must be float16 or bfloat16");
  TORCH_CHECK(
      ptr_B.scalar_type() == at::kFloat8_e4m3fn,
      "weight must be float8_e4m3fn");
  TORCH_CHECK(
      ptr_scales.scalar_type() == at::kFloat,
      "weight scales must be float32");
  TORCH_CHECK(ptr_A.dim() == 2, "activation must be 2D [M, K]");
  TORCH_CHECK(ptr_B.dim() == 2, "weight must be 2D [N, K]");
  TORCH_CHECK(
      ptr_scales.dim() == 2,
      "weight scales must be 2D [N / 128, K / 128]");
  TORCH_CHECK(
      ptr_A.device() == ptr_B.device() &&
          ptr_A.device() == ptr_scales.device(),
      "activation, weight, and scales must be on the same device");
  TORCH_CHECK(
      ptr_A.is_contiguous() && ptr_B.is_contiguous() &&
          ptr_scales.is_contiguous(),
      "activation, weight, and scales must be contiguous");

  const int64_t M = ptr_A.size(0);
  const int64_t K = ptr_A.size(1);
  const int64_t N = ptr_B.size(0);
  TORCH_CHECK(ptr_B.size(1) == K, "activation and weight K dimensions differ");
  TORCH_CHECK(
      N % 256 == 0 && K % 128 == 0,
      "N must be divisible by 256 and K must be divisible by 128");
  constexpr int64_t max_int = std::numeric_limits<int>::max();
  TORCH_CHECK(
      M <= max_int && N <= max_int && K <= max_int &&
          M * K <= max_int && N * K <= max_int && M * N <= max_int,
      "dense FP8 block GEMM requires dimensions and tensor element counts "
      "to fit in 32-bit indexing");
  TORCH_CHECK(
      ptr_scales.size(0) == N / 128 &&
          ptr_scales.size(1) == K / 128,
      "weight scales must have shape [N / 128, K / 128]");

  auto output = at::empty({M, N}, ptr_A.options());
  auto& queue = at::xpu::getCurrentXPUStream(ptr_A.device().index()).queue();
  using policy = dense_w8a16_policy;
  if (ptr_A.scalar_type() == at::kHalf) {
    DenseFp8BlockGEMMLauncher<half_t, policy>(
        queue,
        reinterpret_cast<const half_t*>(ptr_A.data_ptr()),
        reinterpret_cast<const float_e4m3_t*>(ptr_B.data_ptr()),
        reinterpret_cast<const float*>(ptr_scales.data_ptr()),
        reinterpret_cast<half_t*>(output.data_ptr()),
        M,
        N,
        K);
  } else {
    DenseFp8BlockGEMMLauncher<bfloat16_t, policy>(
        queue,
        reinterpret_cast<const bfloat16_t*>(ptr_A.data_ptr()),
        reinterpret_cast<const float_e4m3_t*>(ptr_B.data_ptr()),
        reinterpret_cast<const float*>(ptr_scales.data_ptr()),
        reinterpret_cast<bfloat16_t*>(output.data_ptr()),
        M,
        N,
        K);
  }
  return output;
}

at::Tensor fp8_block_dequant_xe2_impl(
    const at::Tensor& ptr_B,
    const at::Tensor& ptr_scales) {
  TORCH_CHECK(
      ptr_B.scalar_type() == at::kFloat8_e4m3fn,
      "weight must be float8_e4m3fn");
  TORCH_CHECK(
      ptr_scales.scalar_type() == at::kFloat,
      "weight scales must be float32");
  TORCH_CHECK(ptr_B.dim() == 2, "weight must be 2D [N, K]");
  TORCH_CHECK(
      ptr_scales.dim() == 2,
      "weight scales must be 2D [N / 128, K / 128]");
  TORCH_CHECK(
      ptr_B.device() == ptr_scales.device(),
      "weight and scales must be on the same device");
  TORCH_CHECK(
      ptr_B.is_contiguous() && ptr_scales.is_contiguous(),
      "weight and scales must be contiguous");

  const int64_t N = ptr_B.size(0);
  const int64_t K = ptr_B.size(1);
  TORCH_CHECK(
      N % 128 == 0 && K % 128 == 0,
      "N and K must be divisible by 128");
  TORCH_CHECK(
      ptr_scales.size(0) == N / 128 &&
          ptr_scales.size(1) == K / 128,
      "weight scales must have shape [N / 128, K / 128]");

  auto output = at::empty({N, K}, ptr_B.options().dtype(at::kHalf));
  auto& queue = at::xpu::getCurrentXPUStream(ptr_B.device().index()).queue();
  Fp8BlockDequantLauncher(
      queue,
      reinterpret_cast<const float_e4m3_t*>(ptr_B.data_ptr()),
      reinterpret_cast<const float*>(ptr_scales.data_ptr()),
      reinterpret_cast<half_t*>(output.data_ptr()),
      N,
      K);
  return output;
}
}  // namespace MoE