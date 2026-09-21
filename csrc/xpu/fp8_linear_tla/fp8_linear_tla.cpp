// SPDX-License-Identifier: Apache-2.0
#include "core/registration.h"
#include "nt_split_mainloop.hpp"
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/xpu/XPUFunctions.h>

#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <cutlass/numeric_types.h>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <torch/library.h>
#include <torch/torch.h>

#include <cstdint>
#include <optional>

namespace vllm::linear_tla {

constexpr int kTileN = 64;
constexpr int kTileK = 32;
using ElementA = cutlass::half_t;
using ElementB = cutlass::float_e4m3_t;
using WGTile = Shape<_8, _64, _32>;
using SGLayout = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
using MMA = typename TiledMMAHelper<
    MMA_Atom<XE_DPAS_TT<8, float, ElementA>>,
    Layout<WGTile>,
    SGLayout>::TiledMMA;

template <int Splits>
class NtSplitKGemm;
template <int Splits>
class NtSplitKReduce;

template <int kSplits>
at::Tensor fp8_gemm_impl(
    const at::Tensor& input,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& scale,
    const std::optional<at::Tensor>& bias) {
  TORCH_CHECK(
      input.is_xpu() && input.scalar_type() == at::kHalf && input.dim() == 2 &&
          input.is_contiguous(),
      "linear_tla_unsupported: contiguous FP16 XPU A[M,K] required");
  const int64_t m = input.size(0);
  const int64_t k = input.size(1);
  TORCH_CHECK(
      weight.device() == input.device() && weight.dim() == 2 &&
          weight.scalar_type() == at::ScalarType::Float8_e4m3fn &&
          weight.size(0) == k && weight.stride(0) == 1 && weight.stride(1) == k,
      "linear_tla_unsupported: FP8 B[K,N] stride(1,K) required");
  const int64_t n = weight.size(1);
  TORCH_CHECK(
      m >= 1 && m <= 8 && k >= kSplits * kTileK && k <= 65536 &&
          k % kTileK == 0 && n >= kTileN && n <= 65536 && n % kTileN == 0,
      "linear_tla_unsupported: M1..8, K%32=0/K>=256, N%64=0 required");
  TORCH_CHECK(
      !bias.has_value() && scale.has_value() &&
          scale->device() == input.device() &&
          scale->scalar_type() == at::kFloat && scale->numel() == 1 &&
          scale->is_contiguous(),
      "linear_tla_unsupported: scalar FP32 scale and no bias required");
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(input.data_ptr()) % 64 == 0 &&
          reinterpret_cast<uintptr_t>(weight.data_ptr()) % 64 == 0,
      "linear_tla_unsupported: A/B base must satisfy 64-byte 2D-I/O alignment");
  const c10::DeviceGuard guard(input.device());
  auto output = at::empty({m, n}, input.options());
  auto partial = at::empty({kSplits, m, n}, input.options().dtype(at::kFloat));
  const auto* a = reinterpret_cast<const ElementA*>(input.data_ptr<at::Half>());
  const auto* b = reinterpret_cast<const ElementB*>(weight.data_ptr());
  auto* p = partial.data_ptr<float>();
  auto* out = reinterpret_cast<sycl::half*>(output.data_ptr<at::Half>());
  const auto* s = scale->data_ptr<float>();
  auto& queue = c10::xpu::getCurrentXPUStream(input.get_device()).queue();
  constexpr int group_size = size(MMA{});
  static_assert(group_size == 64);
  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;
  const syclex::properties props{
      syclex::sub_group_size<16>, intelex::grf_size<256>};
  queue.parallel_for<NtSplitKGemm<kSplits>>(
      sycl::nd_range<3>(
          sycl::range<3>(kSplits, n / kTileN, group_size),
          sycl::range<3>(1, 1, group_size)),
      props,
      [=](sycl::nd_item<3> item) {
        const int split = item.get_group(0);
        const int tile_n = item.get_group(1);
        const int tiles_k = k / kTileK;
        const int k_begin = tiles_k * split / kSplits;
        const int k_end = tiles_k * (split + 1) / kSplits;
        // Keep the real full-tensor pitch. Partition K by tile coordinates,
        // rather than rebasing B to an incorrectly aligned/repacked pointer.
        auto A = make_tensor(
            make_gmem_ptr(a),
            make_layout(make_shape(int(m), int(k)), make_stride(int(k), _1{})));
        auto B = make_tensor(
            make_gmem_ptr(b),
            make_layout(make_shape(int(n), int(k)), make_stride(int(k), _1{})));
        auto C = make_tensor(
            make_gmem_ptr(p + int64_t(split) * m * n),
            make_layout(make_shape(int(m), int(n)), make_stride(int(n), _1{})));
        nt_split_mainloop<void, void, void>(
            A, B, C, make_coord(0, tile_n, _, 0), MMA{}, k_begin, k_end);
      });
  const int64_t total = m * n;
  queue.parallel_for<NtSplitKReduce<kSplits>>(
      sycl::nd_range<1>(((total + 255) / 256) * 256, 256),
      [=](sycl::nd_item<1> item) {
        const int64_t index = item.get_global_linear_id();
        if (index < total) {
          float sum = 0.0f;
#pragma unroll
          for (int split = 0; split < kSplits; ++split) {
            sum += p[int64_t(split) * total + index];
          }
          out[index] = sycl::half(sum * s[0]);
        }
      });
  return output;
}

template <int N, int Splits>
class NtFusedSplitKGemm;
template <int kTileNFused, int kSplits>
at::Tensor fp8_gemm_fused_impl(
    const at::Tensor& input,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& scale,
    const std::optional<at::Tensor>& bias) {
  constexpr int kTileN = kTileNFused;
  using Tile = Shape<_8, Int<kTileN>, _32>;
  using LayoutSG =
      Layout<Shape<_1, Int<kTileN / 16>, _1>, Stride<Int<kTileN / 16>, _1, _0>>;
  using MMA = typename TiledMMAHelper<
      MMA_Atom<XE_DPAS_TT<8, float, ElementA>>,
      Layout<Tile>,
      LayoutSG>::TiledMMA;
  TORCH_CHECK(
      input.is_xpu() && input.scalar_type() == at::kHalf && input.dim() == 2 &&
          input.is_contiguous(),
      "linear_tla_unsupported: contiguous FP16 XPU A[M,K] required");
  const int64_t m = input.size(0);
  const int64_t k = input.size(1);
  TORCH_CHECK(
      weight.device() == input.device() && weight.dim() == 2 &&
          weight.scalar_type() == at::ScalarType::Float8_e4m3fn &&
          weight.size(0) == k && weight.stride(0) == 1 && weight.stride(1) == k,
      "linear_tla_unsupported: FP8 B[K,N] stride(1,K) required");
  const int64_t n = weight.size(1);
  TORCH_CHECK(
      m >= 1 && m <= 8 && k >= kSplits * kTileK && k <= 65536 &&
          k % kTileK == 0 && n >= kTileN && n <= 65536 && n % kTileN == 0,
      "linear_tla_unsupported: M1..8, K%32=0/K>=256, N%64=0 required");
  TORCH_CHECK(
      !bias.has_value() && scale.has_value() &&
          scale->device() == input.device() &&
          scale->scalar_type() == at::kFloat && scale->numel() == 1 &&
          scale->is_contiguous(),
      "linear_tla_unsupported: scalar FP32 scale and no bias required");
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(input.data_ptr()) % 64 == 0 &&
          reinterpret_cast<uintptr_t>(weight.data_ptr()) % 64 == 0,
      "linear_tla_unsupported: A/B base must satisfy 64-byte 2D-I/O alignment");
  const c10::DeviceGuard guard(input.device());
  auto output = at::empty({m, n}, input.options());
  auto partial = at::empty({kSplits, m, n}, input.options().dtype(at::kFloat));
  const auto* a = reinterpret_cast<const ElementA*>(input.data_ptr<at::Half>());
  const auto* b = reinterpret_cast<const ElementB*>(weight.data_ptr());
  auto* p = partial.data_ptr<float>();
  auto* out = reinterpret_cast<sycl::half*>(output.data_ptr<at::Half>());
  const auto* s = scale->data_ptr<float>();
  auto& queue = c10::xpu::getCurrentXPUStream(input.get_device()).queue();
  constexpr int group_size = size(MMA{});
  static_assert(group_size == kTileN);
  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;
  const syclex::properties props{
      syclex::sub_group_size<16>, intelex::grf_size<256>};
  queue.parallel_for<NtFusedSplitKGemm<kTileNFused, kSplits>>(
      sycl::nd_range<3>(
          sycl::range<3>(1, n / kTileN, kSplits * group_size),
          sycl::range<3>(1, 1, kSplits * group_size)),
      props,
      [=](sycl::nd_item<3> item) {
        const int split = item.get_local_linear_id() / group_size;
        const int tile_n = item.get_group(1);
        const int tiles_k = k / kTileK;
        const int chunk = (tiles_k + kSplits - 1) / kSplits;
        const int k_begin = chunk * split;
        const int k_end = k_begin + chunk;
        auto A = make_tensor(
            make_gmem_ptr(a),
            make_layout(make_shape(int(m), int(k)), make_stride(int(k), _1{})));
        auto B = make_tensor(
            make_gmem_ptr(b),
            make_layout(make_shape(int(n), int(k)), make_stride(int(k), _1{})));
        auto C = make_tensor(
            make_gmem_ptr(p + int64_t(split) * m * n),
            make_layout(make_shape(int(m), int(n)), make_stride(int(n), _1{})));
        nt_split_mainloop<void, void, void>(
            A, B, C, make_coord(0, tile_n, _, 0), MMA{}, k_begin, k_end);
        // Every split for this N tile belongs to this workgroup; no global
        // inter-workgroup barrier or counter is required.
        item.barrier(sycl::access::fence_space::global_and_local);
        for (int index = item.get_local_linear_id(); index < m * kTileN;
             index += kSplits * group_size) {
          const int row = index / kTileN,
                    col = tile_n * kTileN + index % kTileN;
          float sum = 0.f;
#pragma unroll
          for (int part = 0; part < kSplits; ++part)
            sum += p[(int64_t(part) * m + row) * n + col];
          out[row * n + col] = sycl::half(sum * s[0]);
        }
      });
  return output;
}

// Keep the original fused launcher above intact: SLM requires accessor-based
// submission, while the existing shapes retain their no-accessor fast path.
template <int N, int Splits>
class NtSlmSplitKGemm;
template <int kTileNFused, int kSplits>
at::Tensor fp8_gemm_slm_impl(
    const at::Tensor& input,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& scale,
    const std::optional<at::Tensor>& bias) {
  constexpr bool UseSlm = true;
  constexpr int kTileN = kTileNFused;
  using Tile = Shape<_8, Int<kTileN>, _32>;
  using LayoutSG =
      Layout<Shape<_1, Int<kTileN / 16>, _1>, Stride<Int<kTileN / 16>, _1, _0>>;
  using MMA = typename TiledMMAHelper<
      MMA_Atom<XE_DPAS_TT<8, float, ElementA>>,
      Layout<Tile>,
      LayoutSG>::TiledMMA;
  TORCH_CHECK(
      input.is_xpu() && input.scalar_type() == at::kHalf && input.dim() == 2 &&
          input.is_contiguous(),
      "linear_tla_unsupported: contiguous FP16 XPU A[M,K] required");
  const int64_t m = input.size(0);
  const int64_t k = input.size(1);
  TORCH_CHECK(
      weight.device() == input.device() && weight.dim() == 2 &&
          weight.scalar_type() == at::ScalarType::Float8_e4m3fn &&
          weight.size(0) == k && weight.stride(0) == 1 && weight.stride(1) == k,
      "linear_tla_unsupported: FP8 B[K,N] stride(1,K) required");
  const int64_t n = weight.size(1);
  TORCH_CHECK(
      m >= 1 && m <= 8 && k >= kSplits * kTileK && k <= 65536 &&
          k % kTileK == 0 && n >= kTileN && n <= 65536 && n % kTileN == 0,
      "linear_tla_unsupported: M1..8, K%32=0/K>=256, N%64=0 required");
  TORCH_CHECK(
      !bias.has_value() && scale.has_value() &&
          scale->device() == input.device() &&
          scale->scalar_type() == at::kFloat && scale->numel() == 1 &&
          scale->is_contiguous(),
      "linear_tla_unsupported: scalar FP32 scale and no bias required");
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(input.data_ptr()) % 64 == 0 &&
          reinterpret_cast<uintptr_t>(weight.data_ptr()) % 64 == 0,
      "linear_tla_unsupported: A/B base must satisfy 64-byte 2D-I/O alignment");
  const c10::DeviceGuard guard(input.device());
  auto output = at::empty({m, n}, input.options());
  at::Tensor partial;
  if constexpr (!UseSlm)
    partial = at::empty({kSplits, m, n}, input.options().dtype(at::kFloat));
  const auto* a = reinterpret_cast<const ElementA*>(input.data_ptr<at::Half>());
  const auto* b = reinterpret_cast<const ElementB*>(weight.data_ptr());
  float* global_partial = UseSlm ? nullptr : partial.data_ptr<float>();
  auto* out = reinterpret_cast<sycl::half*>(output.data_ptr<at::Half>());
  const auto* s = scale->data_ptr<float>();
  auto& queue = c10::xpu::getCurrentXPUStream(input.get_device()).queue();
  constexpr int group_size = size(MMA{});
  static_assert(group_size == kTileN);
  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;
  const syclex::properties props{
      syclex::sub_group_size<16>, intelex::grf_size<256>};
  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> scratch;
    if constexpr (UseSlm)
      scratch = sycl::local_accessor<float, 1>(kSplits * m * kTileN, cgh);
    cgh.parallel_for<NtSlmSplitKGemm<kTileNFused, kSplits>>(
        sycl::nd_range<3>(
            sycl::range<3>(1, n / kTileN, kSplits * group_size),
            sycl::range<3>(1, 1, kSplits * group_size)),
        props,
        [=](sycl::nd_item<3> item) {
          const int split = item.get_local_linear_id() / group_size;
          const int tile_n = item.get_group(1);
          const int tiles_k = k / kTileK;
          const int chunk = (tiles_k + kSplits - 1) / kSplits;
          const int k_begin = chunk * split;
          const int k_end = k_begin + chunk;
          auto A = make_tensor(
              make_gmem_ptr(a),
              make_layout(
                  make_shape(int(m), int(k)), make_stride(int(k), _1{})));
          auto B = make_tensor(
              make_gmem_ptr(b),
              make_layout(
                  make_shape(int(n), int(k)), make_stride(int(k), _1{})));
          float* p = global_partial;
          if constexpr (UseSlm)
            p = scratch.get_multi_ptr<sycl::access::decorated::no>().get();
          // The SLM epilogue only needs C's shape to map accumulator
          // coordinates.
          float* c_pointer = UseSlm ? reinterpret_cast<float*>(out)
                                    : p + int64_t(split) * m * n;
          auto C = make_tensor(
              make_gmem_ptr(c_pointer),
              make_layout(
                  make_shape(int(m), int(n)), make_stride(int(n), _1{})));
          nt_split_mainloop<void, void, void, UseSlm>(
              A,
              B,
              C,
              make_coord(0, tile_n, _, 0),
              MMA{},
              k_begin,
              k_end,
              p,
              int(m),
              kTileN,
              split);
          // Every split for this N tile belongs to this workgroup; no global
          // inter-workgroup barrier or counter is required.
          item.barrier(
              UseSlm ? sycl::access::fence_space::local_space
                     : sycl::access::fence_space::global_and_local);
          for (int index = item.get_local_linear_id(); index < m * kTileN;
               index += kSplits * group_size) {
            const int row = index / kTileN,
                      col = tile_n * kTileN + index % kTileN;
            float sum = 0.f;
#pragma unroll
            for (int part = 0; part < kSplits; ++part)
              sum += UseSlm ? p[(part * m + row) * kTileN + col % kTileN]
                            : p[(int64_t(part) * m + row) * n + col];
            out[row * n + col] = sycl::half(sum * s[0]);
          }
        });
  });
  return output;
}

at::Tensor fp8_gemm_w8a16(
    const at::Tensor& input,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& scale,
    const std::optional<at::Tensor>& bias) {
  TORCH_CHECK(
      input.dim() == 2 && weight.dim() == 2,
      "linear_tla_unsupported: 2D tensors required");
  const auto n = weight.size(1);
  const auto k = input.size(1);
  // Gemma31B TP2 M1..8: reduce Split-K partials within the same workgroup.
  // All selection has already been checked by supported().
  if (k == 5376 && (n == 8192 || n == 10240 || n == 21504))
    return fp8_gemm_slm_impl<32, 8>(input, weight, scale, bias);
  if (n == 5376 && (k == 4096 || k == 8192))
    return fp8_gemm_slm_impl<32, 2>(input, weight, scale, bias);
  if (n == 5376 && k == 10752)
    return fp8_gemm_slm_impl<64, 2>(input, weight, scale, bias);
  // Keep the established Gemma26B M1 and M2..8 policies.
  if (input.size(0) == 1) {
    if (n == 5120 && k == 2816)
      return fp8_gemm_fused_impl<32, 2>(input, weight, scale, bias);
    if (n == 2816 && k == 4096)
      return fp8_gemm_fused_impl<16, 4>(input, weight, scale, bias);
    if ((n == 4096 && k == 2816) || (n == 2816 && k == 2048) ||
        (n == 2112 && k == 2816) || (n == 2816 && k == 1056))
      return fp8_gemm_fused_impl<64, 4>(input, weight, scale, bias);
  }
  // Final measured policy: the two QKV projections use Split16.
  // All four other Gemma shapes retain Split8; all use Ktile32.
  const bool large = (n == 4096 && k == 2816) || (n == 5120 && k == 2816);
  if (large) {
    return fp8_gemm_impl<16>(input, weight, scale, bias);
  }
  return fp8_gemm_impl<8>(input, weight, scale, bias);
}

// Selection is completed before any candidate allocation or device submission.
// The only cached object below is an immutable dispatcher schema handle.
bool supported(
    const at::Tensor& a,
    const at::Tensor& b,
    const std::optional<at::Tensor>& scale,
    const std::optional<at::Tensor>& bias) {
  if (!a.is_xpu() || a.scalar_type() != at::kHalf || a.dim() != 2 ||
      !a.is_contiguous() || a.size(0) < 1 || a.size(0) > 8 ||
      b.device() != a.device() || b.dim() != 2 ||
      b.scalar_type() != at::ScalarType::Float8_e4m3fn ||
      b.size(0) != a.size(1) || b.stride(0) != 1 || b.stride(1) != a.size(1) ||
      bias.has_value() || !scale.has_value() || scale->device() != a.device() ||
      scale->scalar_type() != at::kFloat || scale->numel() != 1 ||
      !scale->is_contiguous()) {
    return false;
  }
  const auto k = a.size(1);
  const auto n = b.size(1);
  // Both Gemma shape families support M1..8; unsupported tensor contracts
  // retain the original API before any device submission.
  const bool gemma31_small_m =
      ((k == 5376 && (n == 8192 || n == 10240 || n == 21504)) ||
       (n == 5376 && (k == 4096 || k == 8192 || k == 10752)));
  if (!((n == 4096 && k == 2816) || (n == 2816 && k == 2048) ||
        (n == 5120 && k == 2816) || (n == 2816 && k == 4096) ||
        (n == 2112 && k == 2816) || (n == 2816 && k == 1056) ||
        gemma31_small_m)) {
    return false;
  }
  if (reinterpret_cast<uintptr_t>(a.data_ptr()) % 64 != 0 ||
      reinterpret_cast<uintptr_t>(b.data_ptr()) % 64 != 0) {
    return false;
  }
  // Device architecture is immutable for a process. Cache only this metadata,
  // never the active weights, pointers, layouts, scales or request state.
  static thread_local int last_device = -1;
  static thread_local bool last_supported = false;
  const int device = a.get_device();
  if (last_device != device) {
    namespace sx = sycl::ext::oneapi::experimental;
    last_supported = c10::xpu::get_raw_device(device)
                         .get_info<sx::info::device::architecture>() ==
                     sx::architecture::intel_gpu_bmg_g31;
    last_device = device;
  }
  return last_supported;
}

std::optional<at::Tensor> try_fp8_gemm(
    const at::Tensor& input,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& scale,
    const std::optional<at::Tensor>& bias) {
  if (!supported(input, weight, scale, bias)) return std::nullopt;
  return fp8_gemm_w8a16(input, weight, scale, bias);
}

at::Tensor fp8_gemm_dispatch(
    const at::Tensor& input,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& scale,
    const std::optional<at::Tensor>& bias) {
  if (supported(input, weight, scale, bias)) {
    return fp8_gemm_w8a16(input, weight, scale, bias);
  }
  using Signature = at::Tensor(
      const at::Tensor&,
      const at::Tensor&,
      const std::optional<at::Tensor>&,
      const std::optional<at::Tensor>&);
  static auto fallback = c10::Dispatcher::singleton()
                             .findSchemaOrThrow("_xpu_C::fp8_gemm_w8a16", "")
                             .typed<Signature>();
  return fallback.call(input, weight, scale, bias);
}

at::Tensor fp8_gemm_meta(
    const at::Tensor& input,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& scale,
    const std::optional<at::Tensor>& bias) {
  TORCH_CHECK(
      input.dim() == 2 || input.dim() == 3,
      "FP8 matmul only supports 2D or 3D inputs");
  TORCH_CHECK(
      weight.dim() == 2 || weight.dim() == 3,
      "FP8 matmul only supports 2D or 3D weights");
  TORCH_CHECK(
      weight.dim() != 3 || input.dim() == 3,
      "Batched weights require batched inputs");
  auto shape = input.sym_sizes().vec();
  auto strides = input.sym_strides().vec();
  const auto k = shape.back();
  const auto n = weight.sym_size(-1);
  shape.back() = n;
  for (int i = 0; i < strides.size() - 1; ++i) {
    strides[i] = strides[i] / k * n;
  }
  return at::empty_strided_symint(shape, strides, input.options());
}

}  // namespace vllm::linear_tla

TORCH_LIBRARY_FRAGMENT(_xpu_C, m) {
  // No fallback here: the original API owns fallback selection, avoiding
  // recursion.
  m.def(
      "try_fp8_gemm_w8a16_tla(Tensor A, Tensor B, Tensor? B_scale_, Tensor? "
      "bias_=None) -> Tensor?");
  m.impl(
      "try_fp8_gemm_w8a16_tla", torch::kXPU, &vllm::linear_tla::try_fp8_gemm);
  m.def(
      "fp8_gemm_w8a16_tla(Tensor A, Tensor B, Tensor? B_scale_, Tensor? "
      "bias_=None) -> Tensor");
  m.impl(
      "fp8_gemm_w8a16_tla", torch::kXPU, &vllm::linear_tla::fp8_gemm_dispatch);
  m.impl("fp8_gemm_w8a16_tla", torch::kMeta, &vllm::linear_tla::fp8_gemm_meta);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
