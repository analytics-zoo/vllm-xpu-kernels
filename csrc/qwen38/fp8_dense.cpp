// SPDX-License-Identifier: Apache-2.0
// Dense Qwen4 TP2 decode: ordinary SYCL with the shared SYCL-TLA GEMV body.
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/ext/oneapi/experimental/device_architecture.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <torch/extension.h>
#include <climits>

#include "../xpu/gemv/xe_2/fp8_gemv.hpp"

namespace {
namespace sx = sycl::ext::oneapi::experimental;
namespace gemv = vllm::fp8_gemv;
using half = sycl::half;

bool compatible(
    const at::Tensor& x, const at::Tensor& w, const at::Tensor& scale) {
  return x.is_xpu() && x.scalar_type() == at::kHalf && x.dim() == 2 &&
         x.size(0) >= 1 && x.size(0) <= 8 && x.is_contiguous() && !x.is_neg() &&
         !x.is_conj() && w.device() == x.device() && w.dim() == 2 &&
         w.scalar_type() == at::ScalarType::Float8_e4m3fn &&
         w.size(1) == x.size(1) && w.is_contiguous() && !w.is_neg() &&
         !w.is_conj() && scale.device() == x.device() &&
         scale.scalar_type() == at::kFloat && scale.numel() == 1 &&
         scale.is_contiguous() && !scale.is_neg() && !scale.is_conj() &&
         reinterpret_cast<uintptr_t>(x.const_data_ptr()) % 64 == 0 &&
         reinterpret_cast<uintptr_t>(w.const_data_ptr()) % 64 == 0;
}

int splits(int64_t m, int64_t n, int64_t k) {
  if (k == 5120 && (n == 14336 || n == 8192 || n == 7168)) return 31;
  if (n == 5120 && k == 7168) return m == 1 ? 62 : 31;
  if (n == 5120 && k == 3072 && m <= 4) return 20;
  return 0;
}

bool is_b70(const c10::xpu::XPUStream& stream) {
  static thread_local int last_device = -1;
  static thread_local bool supported = false;
  if (last_device != stream.device_index()) {
    supported = stream.queue()
                    .get_device()
                    .get_info<sx::info::device::architecture>() ==
                sx::architecture::intel_gpu_bmg_g31;
    last_device = stream.device_index();
  }
  return supported;
}

void record(
    std::initializer_list<const at::Tensor*> tensors,
    c10::xpu::XPUStream stream) {
  for (const auto* tensor : tensors)
    c10::xpu::XPUCachingAllocator::recordStream(
        tensor->storage().data_ptr(), stream);
}

gemv::Seg<half>
segment(const at::Tensor& w, const at::Tensor& scale, at::Tensor& output) {
  return {
      reinterpret_cast<const uint8_t*>(w.const_data_ptr()),
      scale.const_data_ptr<float>(),
      reinterpret_cast<half*>(output.mutable_data_ptr<at::Half>()),
      int(w.size(0)),
      int(w.size(0)),
      0};
}

template <int MP, int KS, bool Pair>
struct DenseFp8Kernel {
  const half* input;
  int m, k;
  gemv::Seg<half> first;
  gemv::Seg<half> second;

  [[sycl::reqd_sub_group_size(gemv::kSg)]] void
  operator()(sycl::nd_item<1> item) const {
#ifdef __SYCL_DEVICE_ONLY__
    auto* partial = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[KS * MP * gemv::kSg]>(item.get_group());
    const int group = item.get_group(0);
    const auto& seg = Pair && group >= second.wg_begin ? second : first;
    const int row = (group - seg.wg_begin) * gemv::kRowBlock;
    const int rows = MP == 1 ? 1 : m;
    const gemv::GlobalA<MP, half> a{input, k, rows, k * 2};
    const gemv::u32x8 unused[1][gemv::kChunkSteps] = {};
    gemv::gemv_body<MP, KS, 1, gemv::GlobalA<MP, half>, 1, 0, half, true>(
        item, seg, row, a, partial, rows, k, unused, false);
#endif
  }
};

template <int MP, int KS, bool Pair>
void launch(
    c10::xpu::XPUStream stream,
    const at::Tensor& x,
    gemv::Seg<half> first,
    gemv::Seg<half> second) {
  const int first_groups = (first.N + gemv::kRowBlock - 1) / gemv::kRowBlock;
  second.wg_begin = first_groups;
  const int groups =
      first_groups +
      (Pair ? (second.N + gemv::kRowBlock - 1) / gemv::kRowBlock : 0);
  sx::nd_launch(
      stream.queue(),
      sycl::nd_range<1>(size_t(groups * KS * gemv::kSg), KS * gemv::kSg),
      DenseFp8Kernel<MP, KS, Pair>{
          reinterpret_cast<const half*>(x.const_data_ptr<at::Half>()),
          int(x.size(0)),
          int(x.size(1)),
          first,
          second});
}

std::optional<at::Tensor> try_linear(
    const at::Tensor& x, const at::Tensor& weight, const at::Tensor& scale) {
  if (!compatible(x, weight, scale)) return std::nullopt;
  const int ks = splits(x.size(0), weight.size(0), weight.size(1));
  if (!ks) return std::nullopt;
  const c10::DeviceGuard guard(x.device());
  const auto stream = c10::xpu::getCurrentXPUStream(x.get_device());
  if (!is_b70(stream)) return std::nullopt;
  auto output = at::empty({x.size(0), weight.size(0)}, x.options());
  record({&x, &weight, &scale, &output}, stream);
  const auto seg = segment(weight, scale, output);
  auto dispatch = [&]<int MP>() {
    if (ks == 31)
      launch<MP, 31, false>(stream, x, seg, seg);
    else if (ks == 62)
      launch<MP, 62, false>(stream, x, seg, seg);
    else
      launch<MP, 20, false>(stream, x, seg, seg);
  };
  if (x.size(0) == 1)
    dispatch.template operator()<1>();
  else if (x.size(0) == 2)
    dispatch.template operator()<2>();
  else if (x.size(0) <= 4)
    dispatch.template operator()<4>();
  else
    dispatch.template operator()<8>();
  return output;
}

std::optional<std::tuple<at::Tensor, at::Tensor>> try_pair(
    const at::Tensor& x,
    const at::Tensor& w0,
    const at::Tensor& s0,
    const at::Tensor& w1,
    const at::Tensor& s1) {
  if (!compatible(x, w0, s0) || !compatible(x, w1, s1) || w0.size(0) != 8192 ||
      w1.size(0) != 48 || x.size(1) != 5120)
    return std::nullopt;
  const c10::DeviceGuard guard(x.device());
  const auto stream = c10::xpu::getCurrentXPUStream(x.get_device());
  if (!is_b70(stream)) return std::nullopt;
  auto output0 = at::empty({x.size(0), 8192}, x.options());
  auto output1 = at::empty({x.size(0), 48}, x.options());
  record({&x, &w0, &s0, &w1, &s1, &output0, &output1}, stream);
  // Schedule the tiny a/b projection first so it cannot form a kernel tail.
  auto dispatch = [&]<int MP>() {
    launch<MP, 31, true>(
        stream, x, segment(w1, s1, output1), segment(w0, s0, output0));
  };
  if (x.size(0) == 1)
    dispatch.template operator()<1>();
  else if (x.size(0) == 2)
    dispatch.template operator()<2>();
  else if (x.size(0) <= 4)
    dispatch.template operator()<4>();
  else
    dispatch.template operator()<8>();
  return std::make_tuple(output0, output1);
}
template <int MP>
struct DenseGatedA {
  const half* input;
  const half* gate;
  int m;

  inline void
  operator()(const sycl::sub_group&, int kc, int, gemv::AOps<MP>& a) const {
#ifdef __SYCL_DEVICE_ONLY__
    constexpr int k = 3072;
    uint32_t product[gemv::kChunkSteps][MP];
  #pragma unroll
    for (int step = 0; step < gemv::kChunkSteps; ++step) {
      uint32_t x[MP], g[MP];
      const cute::intel::coord_t coord = {(kc + step * gemv::kStepK) / 2, 0};
      cute::detail::XeSubgroup2DBlockLoad<4, 16, MP, 1>{}(
          input, k * 2, m, k * 2, coord, &x[0]);
      cute::detail::XeSubgroup2DBlockLoad<4, 16, MP, 1>{}(
          gate, k * 2, m, k * 2, coord, &g[0]);
  #pragma unroll
      for (int row = 0; row < MP; ++row) {
        const auto xv = sycl::bit_cast<sycl::vec<half, 2>>(x[row]);
        const auto gv = sycl::bit_cast<sycl::vec<half, 2>>(g[row]);
        // DPAS consumes the actual FP16 product bits, preserving the original
        // elementwise multiply's rounding without volatile private storage.
        const sycl::vec<half, 2> value = xv * gv;
        product[step][row] = value.template as<sycl::vec<uint32_t, 1>>()[0];
      }
    }
    gemv::a_from_dwords<MP>(product, a);
#endif
  }
};

template <int MP>
struct DenseGatedFp8Kernel {
  DenseGatedA<MP> a;
  gemv::Seg<half> segment;

  [[sycl::reqd_sub_group_size(gemv::kSg)]] void
  operator()(sycl::nd_item<1> item) const {
#ifdef __SYCL_DEVICE_ONLY__
    constexpr int ks = 31;
    auto* partial = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[ks * MP * gemv::kSg]>(item.get_group());
    const gemv::u32x8 unused[1][gemv::kChunkSteps] = {};
    gemv::gemv_body<MP, ks, 1, DenseGatedA<MP>, 1, 0, half, true>(
        item,
        segment,
        item.get_group(0) * gemv::kRowBlock,
        a,
        partial,
        a.m,
        3072,
        unused,
        false);
#endif
  }
};

std::optional<at::Tensor> try_gated_linear(
    const at::Tensor& x,
    const at::Tensor& gate,
    const at::Tensor& weight,
    const at::Tensor& scale) {
  if (!compatible(x, weight, scale) || x.size(1) != 3072 ||
      weight.size(0) != 5120 || gate.device() != x.device() ||
      gate.scalar_type() != at::kHalf || gate.sizes() != x.sizes() ||
      !gate.is_contiguous() || gate.is_neg() || gate.is_conj() ||
      reinterpret_cast<uintptr_t>(gate.const_data_ptr()) % 64 != 0)
    return std::nullopt;
  const c10::DeviceGuard guard(x.device());
  const auto stream = c10::xpu::getCurrentXPUStream(x.get_device());
  if (!is_b70(stream) || !stream.queue().is_in_order()) return std::nullopt;
  auto output = at::empty({x.size(0), 5120}, x.options());
  record({&x, &gate, &weight, &scale, &output}, stream);
  auto launch = [&]<int MP>() {
    constexpr int ks = 31;
    sx::nd_launch(
        stream.queue(),
        sycl::nd_range<1>(
            size_t((5120 / gemv::kRowBlock) * ks * gemv::kSg), ks * gemv::kSg),
        DenseGatedFp8Kernel<MP>{
            {reinterpret_cast<const half*>(x.const_data_ptr()),
             reinterpret_cast<const half*>(gate.const_data_ptr()),
             int(x.size(0))},
            segment(weight, scale, output)});
  };
  if (x.size(0) == 1)
    launch.template operator()<1>();
  else if (x.size(0) == 2)
    launch.template operator()<2>();
  else if (x.size(0) <= 4)
    launch.template operator()<4>();
  else
    launch.template operator()<8>();
  return output;
}

inline float rounded(float v) {
  volatile uint16_t bits = sycl::bit_cast<uint16_t>(half(v));
  return float(sycl::bit_cast<half>(uint16_t(bits)));
}
template <int L, int SG>
struct DenseQkvKernel {
  const half *x, *qw, *kw, *cache;
  const int64_t* pos;
  half *q, *k, *gate;
  int hq, hk, cache_rows;
  int64_t pos_stride;
  bool mrope, interleaved;
  int s0, s1, s2;
  [[sycl::reqd_sub_group_size(SG)]] void operator()(sycl::nd_item<1> it) const {
#ifdef __SYCL_DEVICE_ONLY__
  #pragma clang fp contract(off) reassociate(off)
    const int group = it.get_group(0), t = group / (hq + hk),
              head = group % (hq + hk), local = it.get_local_id(0);
    const bool isq = head < hq;
    const int h = isq ? head : head - hq;
    const half* src = x + int64_t(t) * (2 * hq + 2 * hk) * 256 +
                      (isq ? h * 512 : 2 * hq * 256 + h * 256);
    const half* w = isq ? qw : kw;
    half* out = (isq ? q : k) + int64_t(t) * (isq ? hq : hk) * 256 + h * 256;
    constexpr int E = 256 / L;
    float v[E], sum = 0.f;
  #pragma unroll
    for (int j = 0; j < E; j++) {
      v[j] = float(src[local + j * L]);
      sum += v[j] * v[j];
    }
    float inv = sycl::rsqrt(
        sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>()) /
            256.f +
        1e-6f);

  #pragma unroll
    for (int j = 0; j < E; j++) {
      int d = local + j * L;
      float z = rounded((v[j] * inv) * (1.f + float(w[d])));
      if (isq)
        gate[int64_t(t) * hq * 256 + h * 256 + d] =
            half(1.f / (1.f + sycl::exp(-float(src[256 + d]))));
      if (d < 64) {
        int pair = d % 32, partner = d < 32 ? d + 32 : d - 32;
        int axis = 0;
        if (mrope) {
          if (interleaved) {
            if (pair % 3 == 1 && pair < s1 * 3)
              axis = 1;
            else if (pair % 3 == 2 && pair < s2 * 3)
              axis = 2;
          } else
            axis = pair < s0 ? 0 : pair < s0 + s1 ? 1 : 2;
        }
        const int64_t position = pos[int64_t(axis) * pos_stride + t];
        if (position < 0 || position >= cache_rows) {
          out[d] = half(0.f);
          continue;
        }
        float other =
            rounded((float(src[partner]) * inv) * (1.f + float(w[partner])));
        float c = float(cache[position * 64 + pair]),
              s = float(cache[position * 64 + 32 + pair]);
        z = d < 32 ? z * c - other * s : z * c + other * s;
      }
      out[d] = half(z);
    }
#endif
  }
};

std::optional<std::tuple<at::Tensor, at::Tensor, at::Tensor>> try_qkv(
    const at::Tensor& input,
    const at::Tensor& positions,
    const at::Tensor& qw,
    const at::Tensor& kw,
    const at::Tensor& cache,
    const std::vector<int64_t>& sections,
    bool interleaved) {
  if (!(input.is_xpu() && input.scalar_type() == at::kHalf &&
        input.dim() == 2 && input.size(0) >= 1 && input.size(0) <= 8 &&
        input.is_contiguous() &&
        (input.size(1) == 7168 || input.size(1) == 14336)))
    return std::nullopt;
  const int m = input.size(0);
  const bool mrope = positions.dim() == 2;
  if (!(positions.device() == input.device() &&
        positions.scalar_type() == at::kLong &&
        ((positions.dim() == 1 && positions.size(0) == m &&
          positions.stride(0) == 1) ||
         (mrope && positions.size(0) == 3 && positions.size(1) == m &&
          positions.stride(1) == 1 && positions.stride(0) >= m))))
    return std::nullopt;
  for (const auto* t : {&input, &positions, &qw, &kw, &cache})
    if (t->layout() != at::kStrided || t->is_neg() || t->is_conj())
      return std::nullopt;
  for (const auto* t : {&qw, &kw, &cache})
    if (t->device() != input.device() || t->scalar_type() != at::kHalf ||
        !t->is_contiguous())
      return std::nullopt;
  if (!(qw.dim() == 1 && qw.numel() == 256 && kw.dim() == 1 &&
        kw.numel() == 256 && cache.dim() == 2 && cache.size(1) == 64 &&
        cache.size(0) > 0 && cache.size(0) <= INT_MAX))
    return std::nullopt;
  if (mrope &&
      !(sections.size() == 3 && sections[0] >= 0 && sections[0] <= 32 &&
        sections[1] >= 0 && sections[1] <= 32 && sections[2] >= 0 &&
        sections[2] <= 32 && sections[0] + sections[1] + sections[2] == 32))
    return std::nullopt;
  const c10::DeviceGuard guard(input.device());
  const auto stream = c10::xpu::getCurrentXPUStream(input.get_device());
  if (!is_b70(stream)) return std::nullopt;
  const int hq = input.size(1) == 7168 ? 12 : 24, hk = hq / 6;
  auto q = at::empty({m, hq * 256}, input.options());
  auto k = at::empty({m, hk * 256}, input.options());
  auto gate = at::empty_like(q);
  record({&input, &positions, &qw, &kw, &cache, &q, &k, &gate}, stream);
  sx::nd_launch(
      stream.queue(),
      sycl::nd_range<1>(size_t(m * (hq + hk) * 256), 256),
      DenseQkvKernel<256, 16>{
          reinterpret_cast<const half*>(input.const_data_ptr()),
          reinterpret_cast<const half*>(qw.const_data_ptr()),
          reinterpret_cast<const half*>(kw.const_data_ptr()),
          reinterpret_cast<const half*>(cache.const_data_ptr()),
          positions.const_data_ptr<int64_t>(),
          reinterpret_cast<half*>(q.mutable_data_ptr()),
          reinterpret_cast<half*>(k.mutable_data_ptr()),
          reinterpret_cast<half*>(gate.mutable_data_ptr()),
          hq,
          hk,
          int(cache.size(0)),
          mrope ? positions.stride(0) : 0,
          mrope,
          interleaved,
          mrope ? int(sections[0]) : 0,
          mrope ? int(sections[1]) : 0,
          mrope ? int(sections[2]) : 0});
  return std::make_tuple(q, k, gate);
}
}  // namespace

void init_qwen4_dense_fp8(pybind11::module_& module) {
  module.def("try_qwen4_dense_fp8_linear", &try_linear);
  module.def("try_qwen4_dense_fp8_pair", &try_pair);
  module.def("try_qwen4_dense_fp8_gated_linear", &try_gated_linear);
  module.def("try_qwen4_dense_qkv_norm_rope", &try_qkv);
}
