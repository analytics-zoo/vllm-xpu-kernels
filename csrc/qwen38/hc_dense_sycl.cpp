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

#include "core/registration.h"

namespace {
using half = sycl::half;
namespace sx = sycl::ext::oneapi::experimental;
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
}
TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, Negative, m) {
  for (const auto* name :
       {"hc_dense_grouped_norm",
        "hc_dense_combine_norm",
        "hc_dense_gate_mix",
        "hc_dense_down_silu"})
    m.impl(name, torch::CppFunction::makeFallthrough());
}
TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, Conjugate, m) {
  for (const auto* name :
       {"hc_dense_grouped_norm",
        "hc_dense_combine_norm",
        "hc_dense_gate_mix",
        "hc_dense_down_silu"})
    m.impl(name, torch::CppFunction::makeFallthrough());
}
