#include "gemma_norm_fusions.hpp"

#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>

namespace vllm::gemma_norm {
namespace {

constexpr int kGroupSize = 256;

void check_inputs(
    const at::Tensor& input,
    const at::Tensor& residual,
    const at::Tensor& weight) {
  TORCH_CHECK(input.is_xpu() && input.scalar_type() == at::kHalf,
              "Expected FP16 XPU input");
  TORCH_CHECK(input.dim() == 2 && input.size(0) >= 1 && input.size(0) <= 8 &&
                  input.size(1) > 0 && input.is_contiguous(),
              "Expected contiguous [M, K], 1 <= M <= 8 and K > 0");
  TORCH_CHECK(residual.device() == input.device() &&
                  residual.scalar_type() == at::kHalf &&
                  residual.sizes() == input.sizes() && residual.is_contiguous(),
              "Residual must match the input shape, dtype and device");
  TORCH_CHECK(weight.device() == input.device() &&
                  weight.scalar_type() == at::kHalf && weight.dim() == 1 &&
                  weight.numel() == input.size(1) && weight.is_contiguous(),
              "Expected contiguous FP16 weight [K] on the input device");
}

template <bool DeviceScalar>
struct ScaledAddRmsNormKernel {
  const sycl::half* input;
  const sycl::half* residual;
  const sycl::half* weight;
  sycl::half* output;
  sycl::half* residual_output;
  int width;
  float epsilon;
  float scalar;
  const sycl::half* scalar_tensor;

  void operator()(sycl::nd_item<1> item) const {
    const int lane = item.get_local_linear_id();
    const int64_t row = item.get_group_linear_id() * int64_t(width);
    float scale = scalar;
    if constexpr (DeviceScalar) {
      scale = float(scalar_tensor[0]);
    }
    float squared_sum = 0.0f;
    for (int k = lane; k < width; k += kGroupSize) {
      const sycl::half added = sycl::half(float(input[row + k]) +
                                        float(residual[row + k]));
      const sycl::half scaled = sycl::half(float(added) * scale);
      residual_output[row + k] = scaled;
      squared_sum += float(scaled) * float(scaled);
    }
    squared_sum = sycl::reduce_over_group(
        item.get_group(), squared_sum, sycl::plus<float>());
    const float inverse_rms = sycl::rsqrt(squared_sum / width + epsilon);
    for (int k = lane; k < width; k += kGroupSize) {
      const sycl::half normalized =
          sycl::half(float(residual_output[row + k]) * inverse_rms);
      output[row + k] =
          sycl::half(float(normalized) * float(weight[k]));
    }
  }
};

struct NormAddNormKernel {
  const sycl::half* input;
  const sycl::half* residual;
  const sycl::half* weight1;
  const sycl::half* weight2;
  sycl::half* output;
  sycl::half* residual_output;
  int width;
  float epsilon1;
  float epsilon2;

  void operator()(sycl::nd_item<1> item) const {
    const int lane = item.get_local_linear_id();
    const int64_t row = item.get_group_linear_id() * int64_t(width);
    float squared_sum = 0.0f;
    for (int k = lane; k < width; k += kGroupSize) {
      const float value = float(input[row + k]);
      squared_sum += value * value;
    }
    squared_sum = sycl::reduce_over_group(
        item.get_group(), squared_sum, sycl::plus<float>());
    const float inverse_rms1 = sycl::rsqrt(squared_sum / width + epsilon1);
    float squared_sum2 = 0.0f;
    for (int k = lane; k < width; k += kGroupSize) {
      const sycl::half unit_normalized =
          sycl::half(float(input[row + k]) * inverse_rms1);
      const sycl::half normalized =
          sycl::half(float(unit_normalized) * float(weight1[k]));
      const sycl::half added =
          sycl::half(float(normalized) + float(residual[row + k]));
      residual_output[row + k] = added;
      squared_sum2 += float(added) * float(added);
    }
    squared_sum2 = sycl::reduce_over_group(
        item.get_group(), squared_sum2, sycl::plus<float>());
    const float inverse_rms2 = sycl::rsqrt(squared_sum2 / width + epsilon2);
    for (int k = lane; k < width; k += kGroupSize) {
      const sycl::half normalized =
          sycl::half(float(residual_output[row + k]) * inverse_rms2);
      output[row + k] =
          sycl::half(float(normalized) * float(weight2[k]));
    }
  }
};

const sycl::half* ptr(const at::Tensor& tensor) {
  return reinterpret_cast<const sycl::half*>(tensor.data_ptr<at::Half>());
}

sycl::half* mutable_ptr(at::Tensor& tensor) {
  return reinterpret_cast<sycl::half*>(tensor.data_ptr<at::Half>());
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> scaled_add_rms_norm(
    const at::Tensor& input,
    const at::Tensor& residual,
    const at::Tensor& weight,
    double epsilon,
    double scalar) {
  check_inputs(input, residual, weight);
  TORCH_CHECK(epsilon >= 0.0, "Expected nonnegative epsilon");
  const c10::DeviceGuard guard(input.device());
  auto output = at::empty_like(input);
  auto residual_output = at::empty_like(input);
  auto& queue = c10::xpu::getCurrentXPUStream(input.get_device()).queue();
  queue.parallel_for(
      sycl::nd_range<1>(input.size(0) * kGroupSize, kGroupSize),
      ScaledAddRmsNormKernel<false>{
          ptr(input), ptr(residual), ptr(weight), mutable_ptr(output),
          mutable_ptr(residual_output), int(input.size(1)), float(epsilon),
          float(scalar), nullptr});
  return {output, residual_output};
}

std::tuple<at::Tensor, at::Tensor> scaled_add_rms_norm_tensor(
    const at::Tensor& input,
    const at::Tensor& residual,
    const at::Tensor& weight,
    double epsilon,
    const at::Tensor& scalar) {
  check_inputs(input, residual, weight);
  TORCH_CHECK(epsilon >= 0.0, "Expected nonnegative epsilon");
  TORCH_CHECK(scalar.device() == input.device() &&
                  scalar.scalar_type() == at::kHalf && scalar.numel() == 1 &&
                  scalar.is_contiguous(),
              "Expected one contiguous FP16 scalar on the input XPU device");
  const c10::DeviceGuard guard(input.device());
  auto output = at::empty_like(input);
  auto residual_output = at::empty_like(input);
  auto& queue = c10::xpu::getCurrentXPUStream(input.get_device()).queue();
  queue.parallel_for(
      sycl::nd_range<1>(input.size(0) * kGroupSize, kGroupSize),
      ScaledAddRmsNormKernel<true>{
          ptr(input), ptr(residual), ptr(weight), mutable_ptr(output),
          mutable_ptr(residual_output), int(input.size(1)), float(epsilon),
          0.0f, ptr(scalar)});
  return {output, residual_output};
}

std::tuple<at::Tensor, at::Tensor> norm_add_norm(
    const at::Tensor& input,
    const at::Tensor& residual,
    const at::Tensor& weight1,
    const at::Tensor& weight2,
    double epsilon1,
    double epsilon2) {
  check_inputs(input, residual, weight1);
  check_inputs(input, residual, weight2);
  TORCH_CHECK(epsilon1 >= 0.0 && epsilon2 >= 0.0,
              "Expected nonnegative epsilons");
  const c10::DeviceGuard guard(input.device());
  auto output = at::empty_like(input);
  auto residual_output = at::empty_like(input);
  auto& queue = c10::xpu::getCurrentXPUStream(input.get_device()).queue();
  queue.parallel_for(
      sycl::nd_range<1>(input.size(0) * kGroupSize, kGroupSize),
      NormAddNormKernel{ptr(input), ptr(residual), ptr(weight1), ptr(weight2),
                       mutable_ptr(output), mutable_ptr(residual_output),
                       int(input.size(1)), float(epsilon1), float(epsilon2)});
  return {output, residual_output};
}

}  // namespace vllm::gemma_norm
