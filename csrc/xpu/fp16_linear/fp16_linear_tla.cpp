#include "core/registration.h"
#include "nt_split_mainloop.hpp"
#include "../fp8_linear_tla/nt_split_mainloop.hpp"

#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <c10/xpu/XPUFunctions.h>
#include <ATen/ops/linear.h>
#include <cutlass/numeric_types.h>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <torch/library.h>
#include <torch/torch.h>

#include <cstdint>
#include <optional>

namespace vllm::fp16_linear_tla {

constexpr int kHidden = 2816;
constexpr int kRouterN = 128;
constexpr int kTileK = 32;

bool small_m_shape(int64_t n, int64_t k) {
  return (n == 512 && (k == 5632 || k == 10752)) ||
         (k == 512 && (n == 2816 || n == 5376)) ||
         (k == 1024 && (n == 2048 || n == 4096 || n == 8192 || n == 131072)) ||
         (n == 1024 && (k == 2048 || k == 4096 || k == 8192)) ||
         (n == 131072 && (k == 2816 || k == 5376)) || (n == 128 && k == 2816);
}

constexpr int kRouterSplits = 16;
using Element = cutlass::half_t;

template <int TileN, int Subgroups>
using NtMma = typename TiledMMAHelper<
    MMA_Atom<XE_DPAS_TT<8, float, Element>>,
    Layout<Shape<_8, Int<TileN>, _32>>,
    Layout<Shape<_1, Int<Subgroups>, _1>, Stride<Int<Subgroups>, _1, _0>>>::
    TiledMMA;

class Fp16DirectNt;
class Fp16RouterNtSplitK;
class Fp16RouterReduce;

template <int TileN, int Splits>
class Fp16SmallMSlm;

template <int TileN, int Splits>
at::Tensor fp16_slm_gemm(const at::Tensor& input, const at::Tensor& weight) {
  using Mma = NtMma<TileN, TileN / 16>;
  constexpr int group_size = size(Mma{});
  const int m = input.size(0), k = input.size(1), n = weight.size(0);
  const c10::DeviceGuard guard(input.device());
  auto output = at::empty({m, n}, input.options());
  const auto* a = reinterpret_cast<const Element*>(input.data_ptr<at::Half>());
  const auto* b = reinterpret_cast<const Element*>(weight.data_ptr<at::Half>());
  auto* out = reinterpret_cast<Element*>(output.data_ptr<at::Half>());
  auto& queue = c10::xpu::getCurrentXPUStream(input.get_device()).queue();
  namespace sx = sycl::ext::oneapi::experimental;
  namespace ix = sycl::ext::intel::experimental;
  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> partial(Splits * m * TileN, cgh);
    cgh.parallel_for<Fp16SmallMSlm<TileN, Splits>>(
        sycl::nd_range<3>(
            sycl::range<3>(1, n / TileN, group_size * Splits),
            sycl::range<3>(1, 1, group_size * Splits)),
        sx::properties{sx::sub_group_size<16>, ix::grf_size<256>},
        [=](sycl::nd_item<3> item) {
          const int split = item.get_local_linear_id() / group_size;
          const int tile = item.get_group(1);
          // Equal iteration counts are required by the workgroup barrier.
          // The 2D loads zero-fill tiles beyond the true K surface.
          const int chunk = (k / kTileK + Splits - 1) / Splits;
          auto A = make_tensor(
              make_gmem_ptr(a),
              make_layout(make_shape(m, k), make_stride(k, _1{})));
          auto B = make_tensor(
              make_gmem_ptr(b),
              make_layout(make_shape(n, k), make_stride(k, _1{})));
          auto C = make_tensor(
              make_gmem_ptr(out),
              make_layout(make_shape(m, n), make_stride(n, _1{})));
          auto* p =
              partial.template get_multi_ptr<sycl::access::decorated::no>()
                  .get();
          vllm::linear_tla::nt_split_mainloop<void, void, void, true>(
              A,
              B,
              C,
              make_coord(0, tile, _, 0),
              Mma{},
              split * chunk,
              (split + 1) * chunk,
              p,
              m,
              TileN,
              split);
          item.barrier(sycl::access::fence_space::local_space);
          for (int i = item.get_local_linear_id(); i < m * TileN;
               i += group_size * Splits) {
            float sum = 0.f;
#pragma unroll
            for (int s = 0; s < Splits; ++s)
              sum += p[s * m * TileN + i];
            out[(i / TileN) * n + tile * TileN + i % TileN] = Element(sum);
          }
        });
  });
  return output;
}

at::Tensor
fp16_nt_gemm_impl(const at::Tensor& input, const at::Tensor& weight) {
  TORCH_CHECK(
      input.is_xpu() && input.scalar_type() == at::kHalf && input.dim() == 2 &&
          input.is_contiguous(),
      "fp16_nt_gemm: expected contiguous FP16 XPU input [M,K]");
  const int64_t m = input.size(0);
  const int64_t k = input.size(1);
  TORCH_CHECK(
      m >= 1 && m <= 8 && k % kTileK == 0,
      "fp16_nt_gemm: expected M1..8 and K%32=0");
  TORCH_CHECK(
      weight.device() == input.device() && weight.scalar_type() == at::kHalf &&
          weight.dim() == 2 && weight.is_contiguous() && weight.size(1) == k &&
          small_m_shape(weight.size(0), k),
      "fp16_nt_gemm: unsupported FP16 weight [N,K] or device");
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(input.data_ptr()) % 64 == 0 &&
          reinterpret_cast<uintptr_t>(weight.data_ptr()) % 64 == 0,
      "fp16_nt_gemm: 2D block-I/O requires 64-byte A/W alignment");
  const int n = weight.size(0);
  if (n == 512 || n == 1024) return fp16_slm_gemm<16, 8>(input, weight);
  const c10::DeviceGuard guard(input.device());
  auto output = at::empty({m, n}, input.options());
  const auto* a = reinterpret_cast<const Element*>(input.data_ptr<at::Half>());
  const auto* b = reinterpret_cast<const Element*>(weight.data_ptr<at::Half>());
  auto* out = reinterpret_cast<Element*>(output.data_ptr<at::Half>());
  auto& queue = c10::xpu::getCurrentXPUStream(input.get_device()).queue();
  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;
  const syclex::properties props{
      syclex::sub_group_size<16>, intelex::grf_size<256>};

  if (n != kRouterN) {
    using Mma = NtMma<64, 4>;
    constexpr int group_size = size(Mma{});
    static_assert(group_size == 64);
    // Direct NT tiles accumulate all K in FP32 and write FP16 once.
    // Narrow projections use the SLM Split-K path above.
    queue.parallel_for<Fp16DirectNt>(
        sycl::nd_range<3>(
            sycl::range<3>(1, n / 64, group_size),
            sycl::range<3>(1, 1, group_size)),
        props,
        [=](sycl::nd_item<3> item) {
          auto A = make_tensor(
              make_gmem_ptr(a),
              make_layout(
                  make_shape(int(m), int(k)), make_stride(int(k), _1{})));
          auto B = make_tensor(
              make_gmem_ptr(b),
              make_layout(make_shape(n, int(k)), make_stride(int(k), _1{})));
          auto C = make_tensor(
              make_gmem_ptr(out),
              make_layout(make_shape(int(m), n), make_stride(n, _1{})));
          nt_split_mainloop<void, void, void>(
              A,
              B,
              C,
              make_coord(0, int(item.get_group(1)), _, 0),
              Mma{},
              0,
              k / kTileK);
        });
    return output;
  }

  using Mma = NtMma<16, 1>;
  constexpr int group_size = size(Mma{});
  static_assert(group_size == 16);
  auto partial =
      at::empty({kRouterSplits, m, n}, input.options().dtype(at::kFloat));
  auto* p = partial.data_ptr<float>();
  // Narrow N gives only eight N tiles. Split K into 16 groups so that 128
  // work-groups execute 5/6 K tiles each; all 88 K tiles are covered exactly.
  queue.parallel_for<Fp16RouterNtSplitK>(
      sycl::nd_range<3>(
          sycl::range<3>(kRouterSplits, n / 16, group_size),
          sycl::range<3>(1, 1, group_size)),
      props,
      [=](sycl::nd_item<3> item) {
        const int split = item.get_group(0);
        constexpr int tiles_k = kHidden / kTileK;
        const int k_begin = tiles_k * split / kRouterSplits;
        const int k_end = tiles_k * (split + 1) / kRouterSplits;
        auto A = make_tensor(
            make_gmem_ptr(a),
            make_layout(
                make_shape(int(m), kHidden), make_stride(kHidden, _1{})));
        auto B = make_tensor(
            make_gmem_ptr(b),
            make_layout(make_shape(n, kHidden), make_stride(kHidden, _1{})));
        auto C = make_tensor(
            make_gmem_ptr(p + int64_t(split) * m * n),
            make_layout(make_shape(int(m), n), make_stride(n, _1{})));
        nt_split_mainloop<void, void, void>(
            A,
            B,
            C,
            make_coord(0, int(item.get_group(1)), _, 0),
            Mma{},
            k_begin,
            k_end);
      });
  const int total = int(m) * n;
  queue.parallel_for<Fp16RouterReduce>(
      sycl::nd_range<1>(((total + 255) / 256) * 256, 256),
      [=](sycl::nd_item<1> item) {
        const int index = item.get_global_linear_id();
        if (index < total) {
          float sum = 0.0f;
#pragma unroll
          for (int split = 0; split < kRouterSplits; ++split) {
            sum += p[int64_t(split) * total + index];
          }
          // Preserve GateLinear's FP16 GEMM storage rounding. Its later
          // FP32 cast remains outside this operator and is not folded away.
          out[index] = Element(sum);
        }
      });
  return output;
}

// All selection happens before candidate allocations or device submissions.
bool fp16_tla_supported(const at::Tensor& input, const at::Tensor& weight) {
  if (!input.is_xpu() || input.scalar_type() != at::kHalf || input.dim() != 2 ||
      !input.is_contiguous() || input.size(0) < 1 || input.size(0) > 8 ||
      weight.device() != input.device() || weight.scalar_type() != at::kHalf ||
      weight.dim() != 2 || !weight.is_contiguous() ||
      weight.size(1) != input.size(1) ||
      !small_m_shape(weight.size(0), input.size(1))) {
    return false;
  }
  if (reinterpret_cast<uintptr_t>(input.data_ptr()) % 64 != 0 ||
      reinterpret_cast<uintptr_t>(weight.data_ptr()) % 64 != 0) {
    return false;
  }
  // Cache only immutable architecture metadata, never tensor/request state.
  static thread_local int last_device = -1;
  static thread_local bool last_supported = false;
  const int device = input.get_device();
  if (last_device != device) {
    namespace sx = sycl::ext::oneapi::experimental;
    last_supported = c10::xpu::get_raw_device(device)
                         .get_info<sx::info::device::architecture>() ==
                     sx::architecture::intel_gpu_bmg_g31;
    last_device = device;
  }
  return last_supported;
}

at::Tensor linear_fallback(
    const at::Tensor& input,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias) {
  // Upstream oneDNN can miscompute FP16 GEMM with an unaligned activation
  // address. Keep ordinary fallbacks unchanged, but realign only tensors whose
  // addresses do not meet the 2D-I/O alignment. These copies are confined to
  // the unsupported path; aligned target shapes returned above without them.
  if (input.is_xpu() && weight.device() == input.device()) {
    const c10::DeviceGuard guard(input.device());
    auto aligned_input = input;
    auto aligned_weight = weight;
    if (input.numel() != 0 &&
        reinterpret_cast<uintptr_t>(input.data_ptr()) % 64 != 0) {
      aligned_input = input.clone(at::MemoryFormat::Contiguous);
    }
    if (weight.numel() != 0 &&
        reinterpret_cast<uintptr_t>(weight.data_ptr()) % 64 != 0) {
      aligned_weight = weight.clone(at::MemoryFormat::Contiguous);
    }
    return at::linear(aligned_input, aligned_weight, bias);
  }
  return at::linear(input, weight, bias);
}

at::Tensor fp16_nt_gemm(const at::Tensor& input, const at::Tensor& weight) {
  if (fp16_tla_supported(input, weight)) {
    return fp16_nt_gemm_impl(input, weight);
  }
  return linear_fallback(input, weight, std::nullopt);
}

at::Tensor unquantized_gemm(
    const at::Tensor& input,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias) {
  // Keep selection and validation inside the existing XPU operator.
  if (!bias.has_value() && fp16_tla_supported(input, weight)) {
    return fp16_nt_gemm_impl(input, weight);
  }
  return linear_fallback(input, weight, bias);
}

at::Tensor unquantized_gemm_meta(
    const at::Tensor& input,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias) {
  return at::linear(input, weight, bias);
}

at::Tensor
fp16_nt_gemm_meta(const at::Tensor& input, const at::Tensor& weight) {
  // Meta/FakeTensor execution never inspects a device or data pointer. The
  // native fallback has this same public linear contract for every shape.
  return at::linear(input, weight);
}

}  // namespace vllm::fp16_linear_tla

TORCH_LIBRARY_FRAGMENT(_xpu_C, m) {
  m.def(
      "unquantized_gemm(Tensor input, Tensor weight, Tensor? bias=None) -> "
      "Tensor");
  m.impl(
      "unquantized_gemm",
      torch::kXPU,
      &vllm::fp16_linear_tla::unquantized_gemm);
  m.impl(
      "unquantized_gemm",
      torch::kMeta,
      &vllm::fp16_linear_tla::unquantized_gemm_meta);
  m.def("fp16_nt_gemm(Tensor input, Tensor weight) -> Tensor");
  m.impl("fp16_nt_gemm", torch::kXPU, &vllm::fp16_linear_tla::fp16_nt_gemm);
  m.impl(
      "fp16_nt_gemm", torch::kMeta, &vllm::fp16_linear_tla::fp16_nt_gemm_meta);
}

REGISTER_EXTENSION(_fp16_C)
