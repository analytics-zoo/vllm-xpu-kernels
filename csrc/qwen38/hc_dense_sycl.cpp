// SPDX-License-Identifier: Apache-2.0
#include <ATen/MemoryOverlap.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/ext/oneapi/experimental/group_load_store.hpp>
#include <sycl/sycl.hpp>
#include <torch/library.h>

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <optional>

#include "core/registration.h"
#include "../xpu/gemv/xe_2/fp8_gemv.hpp"

namespace {
using half = sycl::half;
namespace sx = sycl::ext::oneapi::experimental;
namespace gemv = vllm::fp8_gemv;
constexpr int kHidden = 5120;
constexpr int kStreams = 4;
constexpr int kWidth = kHidden * kStreams;
constexpr int kLocal = 512;
constexpr int kSg = 16;

void check_tensor(
    const at::Tensor& value,
    const at::Tensor& hidden,
    at::IntArrayRef shape,
    bool contiguous = true) {
  TORCH_CHECK(
      value.device() == hidden.device() && value.scalar_type() == at::kHalf &&
          value.layout() == at::kStrided && value.sizes() == shape &&
          (!contiguous || value.is_contiguous()) && !value.is_neg() &&
          !value.is_conj(),
      "Qwen4 HC expects same-device, dense FP16 tensors with the stated shape");
}

void check_hidden(const at::Tensor& hidden) {
  TORCH_CHECK(
      hidden.is_xpu() && hidden.dim() == 2 && hidden.size(0) >= 1 &&
          hidden.size(0) <= 8,
      "Qwen4 HC supports XPU [M,20480], M=1..8");
  check_tensor(hidden, hidden, {hidden.size(0), kWidth});
}

void check_disjoint(
    std::initializer_list<const at::Tensor*> outputs,
    std::initializer_list<const at::Tensor*> inputs) {
  for (const auto* output : outputs) {
    for (const auto* value : inputs) {
      // Include the storage span for the padded injection view. Separate
      // Storage owners can wrap overlapping physical memory as well.
      const auto a = reinterpret_cast<uintptr_t>(output->const_data_ptr());
      const auto b = reinterpret_cast<uintptr_t>(value->storage().data());
      TORCH_CHECK(
          !output->is_alias_of(*value) &&
              (a <= b ? b - a >= output->numel() * output->element_size()
                      : a - b >= value->storage().nbytes()),
          "Qwen4 HC outputs must not overlap any input");
    }
    for (const auto* other : outputs) {
      if (output == other) continue;
      const auto a = reinterpret_cast<uintptr_t>(output->const_data_ptr());
      const auto b = reinterpret_cast<uintptr_t>(other->const_data_ptr());
      TORCH_CHECK(
          !output->is_alias_of(*other) &&
              (a <= b ? b - a >= output->numel() * output->element_size()
                      : a - b >= other->numel() * other->element_size()),
          "Qwen4 HC outputs must not overlap each other");
    }
  }
}

void record(
    std::initializer_list<const at::Tensor*> tensors,
    c10::xpu::XPUStream stream) {
  for (const auto* value : tensors)
    c10::xpu::XPUCachingAllocator::recordStream(
        value->storage().data_ptr(), stream);
}

const half* data(const at::Tensor& value) {
  return reinterpret_cast<const half*>(value.const_data_ptr<at::Half>());
}
half* data(at::Tensor& value) {
  return reinterpret_cast<half*>(value.mutable_data_ptr<at::Half>());
}

template <bool Combine>
class DenseNormKernel;

template <bool Combine>
void launch_norm(
    sycl::queue& queue,
    const half* hidden,
    const half* block,
    const half* injection,
    int64_t injection_stride,
    const half* weight,
    half* combined,
    half* normed,
    int rows,
    float eps) {
  queue.submit([&](sycl::handler& handler) {
    sycl::local_accessor<float, 1> partial(kLocal / kSg, handler);
    handler.parallel_for<DenseNormKernel<Combine>>(
        sycl::nd_range<1>(size_t(rows * kStreams * kLocal), kLocal),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(kSg)]] {
          const int group = item.get_group_linear_id();
          const int row = group / kStreams;
          const int branch = group % kStreams;
          const int local = item.get_local_linear_id();
          const auto sg = item.get_sub_group();
          const int lane = sg.get_local_linear_id();
          const int offset = row * kWidth + branch * kHidden;
          float scale = 0.0f;
          if constexpr (Combine)
            scale =
                2.0f /
                (1.0f + sycl::exp(
                            -float(injection[row * injection_stride + branch]) *
                            0.25f));
          half values[kHidden / kLocal];
          float sum = 0.0f;
#pragma unroll
          for (int i = 0; i < kHidden / kLocal; ++i) {
            const int col = local + i * kLocal;
            half value = hidden[offset + col];
            if constexpr (Combine) {
              value = half(
                  float(value) + float(block[row * kHidden + col]) * scale);
              // The residual is materialized in FP16 before computing RMSNorm.
              volatile uint16_t bits = sycl::bit_cast<uint16_t>(value);
              value = sycl::bit_cast<half>(uint16_t(bits));
              combined[offset + col] = value;
            }
            values[i] = value;
            sum += float(value) * float(value);
          }
          const float sg_sum =
              sycl::reduce_over_group(sg, sum, sycl::plus<float>());
          if (lane == 0) partial[local / kSg] = sg_sum;
          item.barrier(sycl::access::fence_space::local_space);
          const float total = sycl::reduce_over_group(
              sg, partial[lane] + partial[lane + kSg], sycl::plus<float>());
          const float inv = sycl::rsqrt(total / float(kHidden) + eps);
#pragma unroll
          for (int i = 0; i < kHidden / kLocal; ++i) {
            const int col = local + i * kLocal;
            normed[offset + col] = half(
                (float(values[i]) * inv) *
                (1.0f + float(weight[branch * kHidden + col])));
          }
        });
  });
}

void check_norm(
    const at::Tensor& hidden,
    const at::Tensor& weight,
    const at::Tensor& output,
    double eps) {
  check_hidden(hidden);
  check_tensor(weight, hidden, {kWidth});
  check_tensor(output, hidden, {hidden.size(0), kWidth});
  TORCH_CHECK(
      std::isfinite(eps) && eps > 0 && std::isfinite(float(eps)) &&
          float(eps) > 0,
      "Qwen4 HC eps must be positive finite FP32");
}

void grouped_norm(
    const at::Tensor& hidden,
    const at::Tensor& weight,
    at::Tensor& output,
    double eps) {
  check_norm(hidden, weight, output, eps);
  check_disjoint({&output}, {&hidden, &weight});
  const c10::DeviceGuard guard(hidden.device());
  const auto stream = c10::xpu::getCurrentXPUStream(hidden.get_device());
  record({&hidden, &weight, &output}, stream);
  launch_norm<false>(
      stream.queue(),
      data(hidden),
      nullptr,
      nullptr,
      0,
      data(weight),
      nullptr,
      data(output),
      hidden.size(0),
      eps);
}

void combine_norm(
    const at::Tensor& hidden,
    const at::Tensor& block,
    const at::Tensor& injection,
    const at::Tensor& weight,
    at::Tensor& combined,
    at::Tensor& normed,
    double eps) {
  check_norm(hidden, weight, normed, eps);
  check_tensor(combined, hidden, {hidden.size(0), kWidth});
  check_tensor(block, hidden, {hidden.size(0), kHidden});
  check_tensor(injection, hidden, {hidden.size(0), kStreams}, false);
  TORCH_CHECK(
      injection.stride(1) == 1 && injection.stride(0) >= kStreams,
      "Qwen4 HC injection must have nonoverlapping rows and unit inner stride");
  check_disjoint({&combined, &normed}, {&hidden, &block, &injection, &weight});
  const c10::DeviceGuard guard(hidden.device());
  const auto stream = c10::xpu::getCurrentXPUStream(hidden.get_device());
  record({&hidden, &block, &injection, &weight, &combined, &normed}, stream);
  launch_norm<true>(
      stream.queue(),
      data(hidden),
      data(block),
      data(injection),
      injection.stride(0),
      data(weight),
      data(combined),
      data(normed),
      hidden.size(0),
      eps);
}

constexpr int kDenseRank = 320;
constexpr int kDownSplits = 40;
constexpr auto kDownLoad = sx::properties{
    sx::data_placement_striped,
    sx::contiguous_memory,
    sx::full_group,
    sx::alignment<4>};

sycl::vec<half, 16> load_down(sycl::sub_group sg, const half* ptr) {
  sycl::vec<uint32_t, 8> packed;
  auto global =
      sycl::address_space_cast<
          sycl::access::address_space::global_space,
          sycl::access::decorated::yes>(reinterpret_cast<const uint32_t*>(ptr))
          .get_decorated();
  sx::group_load(sg, global, packed, kDownLoad);
  return sycl::bit_cast<sycl::vec<half, 16>>(packed);
}

struct DenseDownSiluKernel {
  const half* input;
  const half* weight;
  half* output;

  [[sycl::reqd_sub_group_size(kSg)]] void
  operator()(sycl::nd_item<1> item) const {
#ifdef __SYCL_DEVICE_ONLY__
    auto* partial = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[kDownSplits]>(item.get_group());
    const auto sg = item.get_sub_group();
    const int row = item.get_group(0);
    const int split = item.get_local_id(0) / kSg;
    static_assert(kWidth == 2 * kDownSplits * 256);
    const int col = split * 256;
    const auto w0 = load_down(sg, weight + int64_t(row) * kWidth + col);
    const auto w1 =
        load_down(sg, weight + int64_t(row) * kWidth + col + kWidth / 2);
    const auto x0 = load_down(sg, input + col);
    const auto x1 = load_down(sg, input + col + kWidth / 2);
    float acc[16];
  #pragma unroll
    for (int i = 0; i < 16; ++i) {
      acc[i] = sycl::fma(float(x0[i]), float(w0[i]), 0.0f);
      acc[i] = sycl::fma(float(x1[i]), float(w1[i]), acc[i]);
    }
    float sum = 0;
  #pragma unroll
    for (int i = 0; i < 16; ++i)
      sum += acc[i];
    sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>());
    if (sg.get_local_id()[0] == 0) partial[split] = sum;
    item.barrier(sycl::access::fence_space::local_space);
    if (item.get_local_id(0) == 0) {
      sum = 0;
  #pragma unroll
      for (int i = 0; i < kDownSplits; ++i)
        sum += partial[i];
      half value = half(sum);
      if (row < kDenseRank) {
        // Match linear -> FP16 -> divide by four -> FP16 -> SiLU -> FP16.
        volatile uint16_t bits = sycl::bit_cast<uint16_t>(value);
        const half scaled =
            half(float(sycl::bit_cast<half>(uint16_t(bits))) * 0.25f);
        volatile uint16_t scaled_bits = sycl::bit_cast<uint16_t>(scaled);
        const float z = float(sycl::bit_cast<half>(uint16_t(scaled_bits)));
        value = half(z / (1.0f + sycl::exp(-z)));
      }
      output[row] = value;
    }
#endif
  }
};

void down_silu(
    const at::Tensor& input, const at::Tensor& weight, at::Tensor& output) {
  check_hidden(input);
  TORCH_CHECK(input.size(0) == 1, "Qwen4 HC down supports one decode token");
  TORCH_CHECK(
      weight.dim() == 2 &&
          (weight.size(0) == kDenseRank || weight.size(0) == kDenseRank + 16),
      "Qwen4 HC down expects 320 or 336 weight rows");
  check_tensor(weight, input, {weight.size(0), kWidth});
  // The merged projection consumes 320 low-rank and four injection rows;
  // twelve checkpoint alignment rows need no computation or output storage.
  const int rows = weight.size(0) == kDenseRank ? kDenseRank : kDenseRank + 4;
  check_tensor(output, input, {1, rows});
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(input.const_data_ptr()) % 4 == 0 &&
          reinterpret_cast<uintptr_t>(weight.const_data_ptr()) % 4 == 0,
      "Qwen4 HC down inputs must be four-byte aligned");
  check_disjoint({&output}, {&input, &weight});
  const c10::DeviceGuard guard(input.device());
  const auto stream = c10::xpu::getCurrentXPUStream(input.get_device());
  record({&input, &weight, &output}, stream);
  sx::nd_launch(
      stream.queue(),
      sycl::nd_range<1>(size_t(rows * kDownSplits * kSg), kDownSplits * kSg),
      DenseDownSiluKernel{data(input), data(weight), data(output)});
}

template <int MP, int KS>
struct DenseDownDpasKernel {
  const half *x, *w;
  half* out;
  int m, n, k, rows;
  bool silu;
  [[sycl::reqd_sub_group_size(kSg)]] void
  operator()(sycl::nd_item<1> it) const {
#ifdef __SYCL_DEVICE_ONLY__
    auto* partial = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[KS * MP * kSg]>(it.get_group());
    auto sg = it.get_sub_group();
    int lane = sg.get_local_linear_id(), ks = sg.get_group_linear_id();
    int nb = it.get_group(0) * kSg;
    using D = gemv::Dpas<MP, half>;
    typename D::C c;
  #pragma unroll
    for (int i = 0; i < MP; i++)
      gemv::vset<MP>(c, i, 0.f);
    for (int kb = ks * 16; kb < k; kb += KS * 16) {
      cute::intel::int8 b{};
      cute::intel::coord_t co = {kb / 2, nb};
      cute::detail::XeSubgroup2DBlockLoadTranspose<4, 8, 16, 1>{}(
          w, k * 2, rows, k * 2, co, &b);
      typename D::A a;
      cute::intel::coord_t ac = {kb, 0};
      cute::detail::XeSubgroup2DBlockLoad<2, 16, MP, 1>{}(
          x, k * 2, m, k * 2, ac, &a);
      D::mma(c, a, b);
    }
  #pragma unroll
    for (int i = 0; i < MP; i++)
      partial[(ks * MP + i) * kSg + lane] = gemv::vget<MP>(c, i);
    sycl::group_barrier(it.get_group());
    if (nb + lane < rows)
      for (int i = ks; i < m; i += KS) {
        float sum = 0.f;
  #pragma unroll
        for (int s = 0; s < KS; s++)
          sum += partial[(s * MP + i) * kSg + lane];
        half value = half(sum);
        if (silu && nb + lane < 320) {
          volatile uint16_t bits = sycl::bit_cast<uint16_t>(value);
          const half scaled =
              half(float(sycl::bit_cast<half>(uint16_t(bits))) * .25f);
          volatile uint16_t sbits = sycl::bit_cast<uint16_t>(scaled);
          float z = float(sycl::bit_cast<half>(uint16_t(sbits)));
          value = half(z / (1.f + sycl::exp(-z)));
        }
        out[int64_t(i) * rows + nb + lane] = value;
      }
#endif
  }
};
template <int MP, int KS>
struct DenseDownVectorKernel {
  const half *x, *w;
  half* out;
  int m, rows;
  bool silu;
  [[sycl::reqd_sub_group_size(kSg)]] void
  operator()(sycl::nd_item<1> it) const {
#ifdef __SYCL_DEVICE_ONLY__
    auto* partial =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[KS * MP]>(
            it.get_group());
    auto sg = it.get_sub_group();
    int split = sg.get_group_linear_id(), row = it.get_group(0),
        lane = sg.get_local_linear_id();
    constexpr int K = 20480;
    float acc[MP] = {};
    for (int kb = split * 256; kb < K; kb += KS * 256) {
      auto load = [&](const half* p) {
        sycl::vec<uint32_t, 8> u;
        auto g = sycl::address_space_cast<
                     sycl::access::address_space::global_space,
                     sycl::access::decorated::yes>(
                     reinterpret_cast<const uint32_t*>(p))
                     .get_decorated();
        sycl::ext::oneapi::experimental::group_load(
            sg,
            g,
            u,
            sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::data_placement_striped,
                sycl::ext::oneapi::experimental::contiguous_memory,
                sycl::ext::oneapi::experimental::full_group,
                sycl::ext::oneapi::experimental::alignment<4>});
        return sycl::bit_cast<sycl::vec<half, 16>>(u);
      };
      auto wv = load(w + int64_t(row) * K + kb);
  #pragma unroll
      for (int i = 0; i < MP; i++)
        if (i < m) {
          auto xv = load(x + int64_t(i) * K + kb);
  #pragma unroll
          for (int j = 0; j < 16; j++)
            acc[i] = sycl::fma(float(wv[j]), float(xv[j]), acc[i]);
        }
    }
  #pragma unroll
    for (int i = 0; i < MP; i++) {
      float sum = sycl::reduce_over_group(sg, acc[i], sycl::plus<float>());
      if (lane == 0) partial[split * MP + i] = sum;
    }
    sycl::group_barrier(it.get_group());
    if (it.get_local_id(0) == 0)
      for (int i = 0; i < m; i++) {
        float sum = 0.f;
  #pragma unroll
        for (int s = 0; s < KS; s++)
          sum += partial[s * MP + i];
        half value = half(sum);
        if (silu && row < 320) {
          volatile uint16_t bits = sycl::bit_cast<uint16_t>(value);
          const half scaled =
              half(float(sycl::bit_cast<half>(uint16_t(bits))) * .25f);
          volatile uint16_t sbits = sycl::bit_cast<uint16_t>(scaled);
          float z = float(sycl::bit_cast<half>(uint16_t(sbits)));
          value = half(z / (1.f + sycl::exp(-z)));
        }
        out[int64_t(i) * rows + row] = value;
      }
#endif
  }
};

void down_silu_smallm(
    const at::Tensor& input, const at::Tensor& weight, at::Tensor& output) {
  check_hidden(input);
  TORCH_CHECK(input.size(0) >= 2, "Qwen4 HC small-M down expects M=2..8");
  TORCH_CHECK(
      weight.dim() == 2 &&
          (weight.size(0) == kDenseRank || weight.size(0) == kDenseRank + 16),
      "Qwen4 HC down expects 320 or 336 weight rows");
  check_tensor(weight, input, {weight.size(0), kWidth});
  const int m = input.size(0);
  const int rows = weight.size(0) == kDenseRank ? kDenseRank : kDenseRank + 4;
  check_tensor(output, input, {m, rows});
  const uintptr_t alignment = m == 2 ? 4 : 64;
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(input.const_data_ptr()) % alignment == 0 &&
          reinterpret_cast<uintptr_t>(weight.const_data_ptr()) % alignment == 0,
      "Qwen4 HC down inputs do not meet block-load alignment");
  check_disjoint({&output}, {&input, &weight});
  const c10::DeviceGuard guard(input.device());
  const auto stream = c10::xpu::getCurrentXPUStream(input.get_device());
  record({&input, &weight, &output}, stream);
  if (m == 2) {
    sx::nd_launch(
        stream.queue(),
        sycl::nd_range<1>(size_t(rows * 20 * kSg), 20 * kSg),
        DenseDownVectorKernel<2, 20>{
            data(input), data(weight), data(output), m, rows, true});
  } else {
    auto launch = [&]<int MP>() {
      sx::nd_launch(
          stream.queue(),
          sycl::nd_range<1>(size_t(((rows + 15) / 16) * 31 * kSg), 31 * kSg),
          DenseDownDpasKernel<MP, 31>{
              data(input),
              data(weight),
              data(output),
              m,
              int(weight.size(0)),
              kWidth,
              rows,
              true});
    };
    if (m <= 4)
      launch.template operator()<4>();
    else
      launch.template operator()<8>();
  }
}

namespace dense_tp1 {
constexpr int kRank = 640;
inline sycl::vec<half, 16> load(sycl::sub_group sg, const half* ptr) {
  sycl::vec<uint32_t, 8> packed;
  auto global =
      sycl::address_space_cast<
          sycl::access::address_space::global_space,
          sycl::access::decorated::yes>(reinterpret_cast<const uint32_t*>(ptr))
          .get_decorated();
  sx::group_load(
      sg,
      global,
      packed,
      sx::properties{
          sx::data_placement_striped,
          sx::contiguous_memory,
          sx::full_group,
          sx::alignment<4>});
  return sycl::bit_cast<sycl::vec<half, 16>>(packed);
}

inline void store(float sum, int token, int row, half* low, half* injection) {
  half value = half(sum);
  if (row < kRank) {
    volatile uint16_t bits = sycl::bit_cast<uint16_t>(value);
    const half scaled =
        half(float(sycl::bit_cast<half>(uint16_t(bits))) * 0.25f);
    volatile uint16_t scaled_bits = sycl::bit_cast<uint16_t>(scaled);
    const float z = float(sycl::bit_cast<half>(uint16_t(scaled_bits)));
    low[token * kRank + row] = half(z / (1.0f + sycl::exp(-z)));
  } else {
    injection[token * 4 + row - kRank] = value;
  }
}

template <int MP, int KS>
struct DownDpas {
  const half* x;
  const half* w;
  half* low;
  half* injection;
  int m;
  int rows;
  [[sycl::reqd_sub_group_size(kSg)]] void
  operator()(sycl::nd_item<1> it) const {
#ifdef __SYCL_DEVICE_ONLY__
    auto* partial = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[KS * MP * kSg]>(it.get_group());
    const auto sg = it.get_sub_group();
    const int lane = sg.get_local_linear_id();
    const int split = sg.get_group_linear_id();
    const int nb = it.get_group(0) * kSg;
    using D = gemv::Dpas<MP, half>;
    typename D::C acc;
  #pragma unroll
    for (int token = 0; token < MP; ++token)
      gemv::vset<MP>(acc, token, 0.0f);
    for (int kb = split * 16; kb < kWidth; kb += KS * 16) {
      cute::intel::int8 b{};
      const cute::intel::coord_t wc = {kb / 2, nb};
      cute::detail::XeSubgroup2DBlockLoadTranspose<4, 8, 16, 1>{}(
          w, kWidth * 2, rows, kWidth * 2, wc, &b);
      typename D::A a;
      const cute::intel::coord_t xc = {kb, 0};
      cute::detail::XeSubgroup2DBlockLoad<2, 16, MP, 1>{}(
          x, kWidth * 2, m, kWidth * 2, xc, &a);
      D::mma(acc, a, b);
    }
  #pragma unroll
    for (int token = 0; token < MP; ++token)
      partial[(split * MP + token) * kSg + lane] = gemv::vget<MP>(acc, token);
    sycl::group_barrier(it.get_group());
    if (nb + lane < rows)
      for (int token = split; token < m; token += KS) {
        float sum = 0.0f;
  #pragma unroll
        for (int s = 0; s < KS; ++s)
          sum += partial[(s * MP + token) * kSg + lane];
        store(sum, token, nb + lane, low, injection);
      }
#endif
  }
};

template <int MP, int KS>
struct DownVector {
  const half* x;
  const half* w;
  half* low;
  half* injection;
  int m;
  [[sycl::reqd_sub_group_size(kSg)]] void
  operator()(sycl::nd_item<1> it) const {
#ifdef __SYCL_DEVICE_ONLY__
    auto* partial =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[KS * MP]>(
            it.get_group());
    const auto sg = it.get_sub_group();
    const int split = sg.get_group_linear_id();
    const int row = it.get_group(0);
    const int lane = sg.get_local_linear_id();
    float acc[MP] = {};
    for (int kb = split * 256; kb < kWidth; kb += KS * 256) {
      const auto weights = load(sg, w + int64_t(row) * kWidth + kb);
  #pragma unroll
      for (int token = 0; token < MP; ++token)
        if (token < m) {
          const auto values = load(sg, x + int64_t(token) * kWidth + kb);
  #pragma unroll
          for (int i = 0; i < 16; ++i)
            acc[token] =
                sycl::fma(float(weights[i]), float(values[i]), acc[token]);
        }
    }
  #pragma unroll
    for (int token = 0; token < MP; ++token) {
      const float sum =
          sycl::reduce_over_group(sg, acc[token], sycl::plus<float>());
      if (lane == 0) partial[split * MP + token] = sum;
    }
    sycl::group_barrier(it.get_group());
    if (it.get_local_id(0) == 0)
      for (int token = 0; token < m; ++token) {
        float sum = 0.0f;
  #pragma unroll
        for (int s = 0; s < KS; ++s)
          sum += partial[s * MP + token];
        store(sum, token, row, low, injection);
      }
#endif
  }
};

struct DownM1 {
  const half* x;
  const half* w;
  half* low;
  half* injection;
  [[sycl::reqd_sub_group_size(kSg)]] void
  operator()(sycl::nd_item<1> it) const {
#ifdef __SYCL_DEVICE_ONLY__
    constexpr int KS = 40;
    auto* partial =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[KS]>(
            it.get_group());
    const auto sg = it.get_sub_group();
    const int row = it.get_group(0);
    const int split = it.get_local_id(0) / kSg;
    const int col = split * 256;
    const auto w0 = load(sg, w + int64_t(row) * kWidth + col);
    const auto w1 = load(sg, w + int64_t(row) * kWidth + col + kWidth / 2);
    const auto x0 = load(sg, x + col);
    const auto x1 = load(sg, x + col + kWidth / 2);
    float acc[16];
  #pragma unroll
    for (int i = 0; i < 16; ++i) {
      acc[i] = sycl::fma(float(x0[i]), float(w0[i]), 0.0f);
      acc[i] = sycl::fma(float(x1[i]), float(w1[i]), acc[i]);
    }
    float sum = 0.0f;
  #pragma unroll
    for (int i = 0; i < 16; ++i)
      sum += acc[i];
    sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>());
    if (sg.get_local_linear_id() == 0) partial[split] = sum;
    it.barrier(sycl::access::fence_space::local_space);
    if (it.get_local_id(0) == 0) {
      sum = 0.0f;
  #pragma unroll
      for (int s = 0; s < KS; ++s)
        sum += partial[s];
      store(sum, 0, row, low, injection);
    }
#endif
  }
};

inline float round_gate(float value) {
  // GEMM先舍入至FP16，再使用FP32 sigmoid；禁止折叠舍入边界。
  const half rounded = half(value);
  volatile uint16_t bits = sycl::bit_cast<uint16_t>(rounded);
  return float(sycl::bit_cast<half>(uint16_t(bits)));
}

template <int MP, int KS>
struct UpGateKernel {
  const half* input;
  const half* weight;
  const half* normed;
  half* output;
  int m;

  [[sycl::reqd_sub_group_size(kSg)]] void
  operator()(sycl::nd_item<1> item) const {
#ifdef __SYCL_DEVICE_ONLY__
    using D = gemv::Dpas<MP, half>;
    auto* partial = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[kStreams * KS * MP * kSg]>(item.get_group());
    const auto sg = item.get_sub_group();
    const int lane = sg.get_local_linear_id();
    const int subgroup = sg.get_group_linear_id();
    const int branch = subgroup / KS;
    const int split = subgroup % KS;
    const int feature = item.get_group(0) * kSg;
    typename D::C sums;
  #pragma unroll
    for (int row = 0; row < MP; ++row)
      gemv::vset<MP>(sums, row, 0.0f);
    for (int kb = split * 16; kb < kRank; kb += KS * 16) {
      typename D::A a;
      const cute::intel::coord_t a_coord = {kb, 0};
      cute::detail::XeSubgroup2DBlockLoad<2, 16, MP, 1>{}(
          input, kRank * 2, m, kRank * 2, a_coord, &a);
      cute::intel::int8 b{};
      const cute::intel::coord_t b_coord = {kb / 2, branch * kHidden + feature};
      cute::detail::XeSubgroup2DBlockLoadTranspose<4, 8, 16, 1>{}(
          weight, kRank * 2, kWidth, kRank * 2, b_coord, &b);
      D::mma(sums, a, b);
    }
  #pragma unroll
    for (int row = 0; row < MP; ++row)
      partial[((branch * KS + split) * MP + row) * kSg + lane] =
          gemv::vget<MP>(sums, row);
    sycl::group_barrier(item.get_group());
    for (int row = subgroup; row < m; row += kStreams * KS) {
      float mixed = 0.0f;
  #pragma unroll
      for (int stream = 0; stream < kStreams; ++stream) {
        float sum = 0.0f;
  #pragma unroll
        for (int s = 0; s < KS; ++s)
          sum += partial[((stream * KS + s) * MP + row) * kSg + lane];
        const float gate = round_gate(sum);
        const int offset = row * kWidth + stream * kHidden + feature + lane;
        const float sigmoid = 1.0f / (1.0f + sycl::exp(-gate));
        mixed += float(normed[offset]) * sigmoid;
      }
      output[row * kHidden + feature + lane] = half(mixed * 0.25f);
    }
#endif
  }
};

void down_silu(
    const at::Tensor& input,
    const at::Tensor& weight,
    at::Tensor& low,
    std::optional<at::Tensor> injection) {
  check_hidden(input);
  const int m = input.size(0);
  TORCH_CHECK(
      weight.dim() == 2 &&
          (weight.size(0) == kRank || weight.size(0) == kRank + 16),
      "Qwen4 TP1 HC down expects 640 or 656 weight rows");
  check_tensor(weight, input, {weight.size(0), kWidth});
  check_tensor(low, input, {m, kRank});
  const bool merged = weight.size(0) != kRank;
  TORCH_CHECK(
      injection.has_value() == merged,
      "Qwen4 TP1 HC injection output mismatch");
  if (merged) {
    check_tensor(*injection, input, {m, 4});
    check_disjoint({&low, &*injection}, {&input, &weight});
  } else {
    check_disjoint({&low}, {&input, &weight});
  }
  const int alignment = m >= 5 ? 64 : 4;
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(input.const_data_ptr()) % alignment == 0 &&
          reinterpret_cast<uintptr_t>(weight.const_data_ptr()) % alignment == 0,
      "Qwen4 TP1 HC down inputs violate block-load alignment");
  const c10::DeviceGuard guard(input.device());
  const auto stream = c10::xpu::getCurrentXPUStream(input.get_device());
  TORCH_CHECK(
      stream.queue().is_in_order(),
      "Qwen4 TP1 HC down requires in-order stream");
  record({&input, &weight, &low}, stream);
  if (merged) record({&*injection}, stream);
  const auto* x = data(input);
  const auto* w = data(weight);
  auto* output = data(low);
  auto* inj = merged ? data(*injection) : nullptr;
  // The physical checkpoint has 12 alignment rows after injection. They are
  // excluded from the load surface and output; low stays contiguous [M,640].
  const int rows = kRank + (merged ? 4 : 0);
  if (m == 1) {
    sx::nd_launch(
        stream.queue(),
        sycl::nd_range<1>(size_t(rows * 40 * kSg), 40 * kSg),
        DownM1{x, w, output, inj});
  } else if (m == 2) {
    sx::nd_launch(
        stream.queue(),
        sycl::nd_range<1>(size_t(rows * 40 * kSg), 40 * kSg),
        DownVector<2, 40>{x, w, output, inj, m});
  } else if (m <= 4) {
    sx::nd_launch(
        stream.queue(),
        sycl::nd_range<1>(size_t(rows * 20 * kSg), 20 * kSg),
        DownVector<4, 20>{x, w, output, inj, m});
  } else {
    sx::nd_launch(
        stream.queue(),
        sycl::nd_range<1>(size_t(((rows + 15) / 16) * 20 * kSg), 20 * kSg),
        DownDpas<8, 20>{x, w, output, inj, m, rows});
  }
}

void up_gate_mix(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& normed,
    at::Tensor& output) {
  check_hidden(normed);
  const int m = normed.size(0);
  check_tensor(input, normed, {m, kRank});
  check_tensor(weight, normed, {kWidth, kRank});
  check_tensor(output, normed, {m, kHidden});
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(input.const_data_ptr()) % 64 == 0 &&
          reinterpret_cast<uintptr_t>(weight.const_data_ptr()) % 64 == 0,
      "Qwen4 TP1 HC up requires 64-byte aligned block-load inputs");
  check_disjoint({&output}, {&input, &weight, &normed});
  const c10::DeviceGuard guard(input.device());
  const auto stream = c10::xpu::getCurrentXPUStream(input.get_device());
  TORCH_CHECK(
      stream.queue().is_in_order(), "Qwen4 TP1 HC up requires in-order stream");
  record({&input, &weight, &normed, &output}, stream);
  auto launch = [&]<int MP, int KS>() {
    constexpr int local = kStreams * KS * kSg;
    sx::nd_launch(
        stream.queue(),
        sycl::nd_range<1>(size_t(kHidden / kSg) * local, local),
        UpGateKernel<MP, KS>{
            data(input), data(weight), data(normed), data(output), m});
  };
  if (m == 1)
    launch.template operator()<1, 10>();
  else if (m == 2)
    launch.template operator()<2, 10>();
  else if (m <= 4)
    launch.template operator()<4, 5>();
  else
    launch.template operator()<8, 5>();
}

}  // namespace dense_tp1

class DenseGateKernel;
void gate_mix(
    const at::Tensor& input, const at::Tensor& gate, at::Tensor& output) {
  check_hidden(input);
  check_tensor(gate, input, {input.size(0), kWidth});
  check_tensor(output, input, {input.size(0), kHidden});
  check_disjoint({&output}, {&input, &gate});
  const c10::DeviceGuard guard_device(input.device());
  const auto stream = c10::xpu::getCurrentXPUStream(input.get_device());
  record({&input, &gate, &output}, stream);
  const auto* x = data(input);
  const auto* gates = data(gate);
  auto* out = data(output);
  stream.queue().parallel_for<DenseGateKernel>(
      sycl::nd_range<1>(size_t(input.size(0) * kHidden), 256),
      [=](sycl::nd_item<1> item) {
        const int index = item.get_global_linear_id();
        const int row = index / kHidden;
        const int col = index % kHidden;
        float sum = 0;
#pragma unroll
        for (int branch = 0; branch < kStreams; ++branch) {
          const int offset = row * kWidth + branch * kHidden + col;
          const float sigmoid =
              1.0f / (1.0f + sycl::exp(-float(gates[offset])));
          sum += float(x[offset]) * sigmoid;
        }
        out[index] = half(sum * 0.25f);
      });
}
}  // namespace

#define QWEN4_LIBRARY_FRAGMENT(NAME, MODULE) \
  TORCH_LIBRARY_FRAGMENT(NAME, MODULE)
QWEN4_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  m.def(
      "hc_dense_grouped_norm(Tensor hidden, Tensor weight, Tensor(a!) output, "
      "float eps) -> ()");
  m.impl("hc_dense_grouped_norm", at::kXPU, &grouped_norm);
  m.def(
      "hc_dense_combine_norm(Tensor hidden, Tensor block, Tensor injection, "
      "Tensor weight, Tensor(a!) combined, Tensor(b!) normed, float eps) -> "
      "()");
  m.impl("hc_dense_combine_norm", at::kXPU, &combine_norm);
  m.def(
      "hc_dense_gate_mix(Tensor input, Tensor gate, Tensor(a!) output) -> ()");
  m.impl("hc_dense_gate_mix", at::kXPU, &gate_mix);
  m.def(
      "hc_dense_down_silu(Tensor input, Tensor weight, Tensor(a!) output) -> "
      "()");
  m.impl("hc_dense_down_silu", at::kXPU, &down_silu);
  m.def(
      "hc_dense_down_silu_smallm(Tensor input, Tensor weight, Tensor(a!) "
      "output) -> ()");
  m.impl("hc_dense_down_silu_smallm", at::kXPU, &down_silu_smallm);
  m.def(
      "hc_dense_down_silu_tp1(Tensor input, Tensor weight, Tensor(a!) low, "
      "Tensor(b!)? injection) -> ()");
  m.impl("hc_dense_down_silu_tp1", at::kXPU, &dense_tp1::down_silu);
  m.def(
      "hc_dense_up_gate_mix_tp1(Tensor input, Tensor weight, Tensor normed, "
      "Tensor(a!) output) -> ()");
  m.impl("hc_dense_up_gate_mix_tp1", at::kXPU, &dense_tp1::up_gate_mix);
}
TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, Negative, m) {
  for (const auto* name :
       {"hc_dense_grouped_norm",
        "hc_dense_combine_norm",
        "hc_dense_gate_mix",
        "hc_dense_down_silu",
        "hc_dense_down_silu_smallm",
        "hc_dense_down_silu_tp1",
        "hc_dense_up_gate_mix_tp1"})
    m.impl(name, torch::CppFunction::makeFallthrough());
}
TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, Conjugate, m) {
  for (const auto* name :
       {"hc_dense_grouped_norm",
        "hc_dense_combine_norm",
        "hc_dense_gate_mix",
        "hc_dense_down_silu",
        "hc_dense_down_silu_smallm",
        "hc_dense_down_silu_tp1",
        "hc_dense_up_gate_mix_tp1"})
    m.impl(name, torch::CppFunction::makeFallthrough());
}
