// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>
#include <torch/library.h>

namespace vllm::small_m_rms {
namespace {

struct Rows {
  int64_t stride[3];
  int64_t size[3];
};

struct alignas(16) Half8 {
  sycl::half values[8];
};

template <int Width, bool Weighted, bool Aligned>
struct SmallMRms {
  const sycl::half* input;
  const sycl::half* weight;
  sycl::half* output;
  Rows rows;
  float epsilon;

  void operator()
      [[sycl::reqd_sub_group_size(32)]] (sycl::nd_item<3> item) const {
    constexpr int Threads = Width / 8;
    const int lane = item.get_local_id(2);
    const int64_t row =
        (item.get_group(0) * rows.size[1] + item.get_group(1)) * rows.size[2] +
        item.get_group(2);
    const int64_t input_offset = item.get_group(0) * rows.stride[0] +
                                 item.get_group(1) * rows.stride[1] +
                                 item.get_group(2) * rows.stride[2];
    Half8 values;
    if constexpr (Aligned) {
      values = reinterpret_cast<const Half8*>(input + input_offset)[lane];
    } else {
#pragma unroll
      for (int i = 0; i < 8; ++i)
        values.values[i] = input[input_offset + lane * 8 + i];
    }
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float value = float(values.values[i]);
      sum += value * value;
    }
    if constexpr (Threads == 32) {
      sum = sycl::reduce_over_group(
          item.get_sub_group(), sum, sycl::plus<float>());
    } else {
      sum = sycl::reduce_over_group(item.get_group(), sum, sycl::plus<float>());
    }
    const float inverse_rms = sycl::rsqrt(sum / Width + epsilon);
    Half8 weights;
    if constexpr (Weighted && Aligned) {
      weights = reinterpret_cast<const Half8*>(weight)[lane];
    }
    Half8 result;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int column = lane * 8 + i;
      // Match upstream weighted RMS: round normalization before weighting.
      sycl::half value = sycl::half(float(values.values[i]) * inverse_rms);
      if constexpr (Weighted) {
        const float w =
            Aligned ? float(weights.values[i]) : float(weight[column]);
        value = sycl::half(float(value) * w);
      }
      result.values[i] = value;
    }
    if constexpr (Aligned) {
      reinterpret_cast<Half8*>(output + row * Width)[lane] = result;
    } else {
#pragma unroll
      for (int i = 0; i < 8; ++i)
        output[row * Width + lane * 8 + i] = result.values[i];
    }
  }
};

void check(
    const at::Tensor& input,
    const std::optional<at::Tensor>& weight,
    double epsilon) {
  TORCH_CHECK(
      input.is_xpu() && input.scalar_type() == at::kHalf,
      "rms_norm_small_m expects FP16 XPU input");
  TORCH_CHECK(
      input.dim() >= 2 && input.dim() <= 4 && input.size(0) >= 1 &&
          input.size(0) <= 8 && input.stride(-1) == 1,
      "Expected rank 2..4, M=1..8 and unit final stride");
  const auto width = input.size(-1);
  TORCH_CHECK(
      width == 256 || width == 512 || width == 2816,
      "Expected hidden size 256, 512 or 2816");
  TORCH_CHECK(
      input.numel() / width >= 1 && input.numel() / width <= 128,
      "Expected 1..128 normalized rows");
  TORCH_CHECK(epsilon >= 0, "Expected nonnegative epsilon");
  if (weight.has_value()) {
    TORCH_CHECK(
        weight->device() == input.device() &&
            weight->scalar_type() == input.scalar_type() &&
            weight->dim() == 1 && weight->numel() == width &&
            weight->is_contiguous(),
        "Expected contiguous FP16 weight [hidden] on input device");
  }
}

template <int Width, bool Weighted, bool Aligned>
void launch(
    at::Tensor& output,
    const at::Tensor& input,
    const std::optional<at::Tensor>& weight,
    float epsilon,
    Rows rows) {
  constexpr int Threads = Width / 8;
  auto& queue = c10::xpu::getCurrentXPUStream(input.get_device()).queue();
  queue.parallel_for(
      sycl::nd_range<3>(
          {size_t(rows.size[0]),
           size_t(rows.size[1]),
           size_t(rows.size[2]) * Threads},
          {1, 1, Threads}),
      SmallMRms<Width, Weighted, Aligned>{
          reinterpret_cast<const sycl::half*>(input.data_ptr<at::Half>()),
          Weighted ? reinterpret_cast<const sycl::half*>(
                         weight->data_ptr<at::Half>())
                   : nullptr,
          reinterpret_cast<sycl::half*>(output.data_ptr<at::Half>()),
          rows,
          epsilon});
}

void dispatch(
    at::Tensor& output,
    const at::Tensor& input,
    const std::optional<at::Tensor>& weight,
    double epsilon) {
  Rows rows{{0, 0, 0}, {1, 1, 1}};
  for (int i = 0; i < input.dim() - 1; ++i) {
    const int j = 4 - input.dim() + i;
    rows.size[j] = input.size(i);
    rows.stride[j] = input.stride(i);
  }
  bool aligned = uintptr_t(input.data_ptr()) % 16 == 0 &&
                 uintptr_t(output.data_ptr()) % 16 == 0 &&
                 (!weight || uintptr_t(weight->data_ptr()) % 16 == 0);
  for (int i = 0; i < 3; ++i)
    aligned &= rows.stride[i] % 8 == 0;
#define LAUNCH_CASE(W, WEIGHT)                                            \
  if (aligned)                                                            \
    launch<W, WEIGHT, true>(output, input, weight, float(epsilon), rows); \
  else                                                                    \
    launch<W, WEIGHT, false>(output, input, weight, float(epsilon), rows)
#define LAUNCH_WIDTH(W)       \
  case W:                     \
    if (weight.has_value()) { \
      LAUNCH_CASE(W, true);   \
    } else {                  \
      LAUNCH_CASE(W, false);  \
    }                         \
    break
  switch (input.size(-1)) {
    LAUNCH_WIDTH(256);
    LAUNCH_WIDTH(512);
    LAUNCH_WIDTH(2816);
  }
#undef LAUNCH_WIDTH
#undef LAUNCH_CASE
}

}  // namespace

at::Tensor rms_norm_small_m(
    const at::Tensor& input,
    const std::optional<at::Tensor>& weight,
    double epsilon) {
  check(input, weight, epsilon);
  const c10::DeviceGuard guard(input.device());
  auto output = at::empty(input.sizes(), input.options());
  dispatch(output, input, weight, epsilon);
  return output;
}

void rms_norm_small_m_out(
    at::Tensor output,
    const at::Tensor& input,
    const std::optional<at::Tensor>& weight,
    double epsilon) {
  check(input, weight, epsilon);
  TORCH_CHECK(
      output.device() == input.device() &&
          output.scalar_type() == input.scalar_type() &&
          output.sizes() == input.sizes() && output.is_contiguous(),
      "Expected contiguous output matching input shape/dtype/device");
  at::assert_no_overlap(output, input);
  if (weight.has_value()) at::assert_no_overlap(output, *weight);
  const c10::DeviceGuard guard(input.device());
  dispatch(output, input, weight, epsilon);
}

}  // namespace vllm::small_m_rms

TORCH_LIBRARY_FRAGMENT(_xpu_C, m) {
  m.def(
      "rms_norm_small_m(Tensor input, Tensor? weight, float epsilon) -> "
      "Tensor");
  m.def(
      "rms_norm_small_m.out(Tensor(a!) output, Tensor input, Tensor? weight, "
      "float epsilon) -> ()");
  m.impl(
      "rms_norm_small_m",
      c10::DispatchKey::XPU,
      &vllm::small_m_rms::rms_norm_small_m);
  m.impl(
      "rms_norm_small_m.out",
      c10::DispatchKey::XPU,
      &vllm::small_m_rms::rms_norm_small_m_out);
  m.impl(
      "rms_norm_small_m",
      c10::DispatchKey::Meta,
      [](const at::Tensor& input,
         const std::optional<at::Tensor>& weight,
         double epsilon) {
        return at::empty_symint(input.sym_sizes(), input.options());
      });
  m.impl(
      "rms_norm_small_m.out",
      c10::DispatchKey::Meta,
      [](at::Tensor output,
         const at::Tensor& input,
         const std::optional<at::Tensor>& weight,
         double epsilon) {});
}
