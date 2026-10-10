// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

namespace vllm::qwen38::hc::manual_norm {
using half = sycl::half;
constexpr int kStreams = 4;
constexpr int kHidden = 2560;
constexpr int kSubgroup = 16;
constexpr int kLocal = 512;

struct RoundedFeedback {
  uint16_t bits;
  float value;
};

inline RoundedFeedback round_feedback(float value) {
  const uint32_t bits = sycl::bit_cast<uint32_t>(value);
  const uint32_t magnitude = bits & 0x7fffffffu;
  if (magnitude >= 0x38800000u && magnitude < 0x477ff000u) {
    // Preserve the actual FP16 RNE boundary in integer bits. Do not replace
    // this feedback with float(half(value)): IGC can promote that to value.
    const uint32_t rounded = magnitude + 0xfffu + ((magnitude >> 13) & 1u);
    const uint16_t half_bits = uint16_t(
        ((bits >> 16) & 0x8000u) | ((rounded - 0x38000000u) >> 13));
    const uint32_t feedback =
        (bits & 0x80000000u) | (rounded & 0xffffe000u);
    return {half_bits, sycl::bit_cast<float>(feedback)};
  }
  // Preserve the previous conversion for small values, overflow and NaNs;
  // in particular, do not invent a new subnormal or NaN-payload contract.
  const half rounded = half(value);
  volatile uint16_t boundary = sycl::bit_cast<uint16_t>(rounded);
  const uint16_t stored = uint16_t(boundary);
  return {stored, float(sycl::bit_cast<half>(stored))};
}

class NormM1ManualKernel;

// The existing HC entry point performs all shape/alias/stream preflights.
// Its M=1 combine+norm dispatch is the only caller of this helper.
inline void launch(
    sycl::queue& queue,
    const half* hidden,
    const half* block,
    const half* injection,
    int64_t injection_stride,
    const half* weight,
    half* combined,
    half* normed,
    float eps) {
  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> partial(kLocal / kSubgroup + 1, cgh);
    cgh.parallel_for<NormM1ManualKernel>(
        sycl::nd_range<1>(size_t(kStreams * kLocal), kLocal),
        [=](sycl::nd_item<1> item)
            [[sycl::reqd_sub_group_size(kSubgroup)]] {
          const int branch = item.get_group_linear_id();
          const int lane = item.get_local_linear_id();
          const auto sg = item.get_sub_group();
          const int subgroup = lane / kSubgroup;
          const int subgroup_lane = sg.get_local_linear_id();
          const int offset = branch * kHidden;
          (void)injection_stride;  // The only row is zero.
          const float injection_value = float(injection[branch]) * 0.25f;
          const float injection_scale =
              2.0f * (1.0f / (1.0f + sycl::exp(-injection_value)));
          float feedback[kHidden / kLocal];
          float sum_sq = 0.0f;
#pragma unroll
          for (int i = 0; i < kHidden / kLocal; ++i) {
            const int col = lane + i * kLocal;
            const float h = float(hidden[offset + col]);
            const float b = float(block[col]);
            const auto rounded = round_feedback(h + b * injection_scale);
            combined[offset + col] = sycl::bit_cast<half>(rounded.bits);
            feedback[i] = rounded.value;
            sum_sq += rounded.value * rounded.value;
          }
          const float sg_sum =
              sycl::reduce_over_group(sg, sum_sq, sycl::plus<float>());
          if (subgroup_lane == 0) partial[subgroup] = sg_sum;
          item.barrier(sycl::access::fence_space::local_space);
          const float pair =
              partial[subgroup_lane] + partial[subgroup_lane + kSubgroup];
          const float total =
              sycl::reduce_over_group(sg, pair, sycl::plus<float>());
          const float inv = sycl::rsqrt(total / float(kHidden) + eps);
#pragma unroll
          for (int i = 0; i < kHidden / kLocal; ++i) {
            const int col = lane + i * kLocal;
            const float w = float(weight[offset + col]);
            normed[offset + col] = half(feedback[i] * inv * (1.0f + w));
          }
        });
  });
}

class NormSmallMManualKernel;

// Deliberately leave the validated M1 launch/body above unchanged. All small-M
// rows share one kernel and the same round_feedback/RNE boundary. The caller
// retains live shape/stride/alias/stream/lifetime preflights. Unsupported M
// returns false before submission; execution errors must propagate.
inline bool launch_small_m(
    sycl::queue& queue,
    const half* hidden,
    const half* block,
    const half* injection,
    int64_t injection_stride,
    const half* weight,
    half* combined,
    half* normed,
    int m,
    float eps) {
  if (m < 2 || m > 8) return false;
  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> partial(kLocal / kSubgroup + 1, cgh);
    cgh.parallel_for<NormSmallMManualKernel>(
        sycl::nd_range<1>(size_t(m * kStreams * kLocal), kLocal),
        [=](sycl::nd_item<1> item)
            [[sycl::reqd_sub_group_size(kSubgroup)]] {
          const int group = item.get_group_linear_id();
          const int row = group / kStreams;
          const int branch = group % kStreams;
          const int lane = item.get_local_linear_id();
          const auto sg = item.get_sub_group();
          const int subgroup = lane / kSubgroup;
          const int subgroup_lane = sg.get_local_linear_id();
          const int offset = group * kHidden;
          const float injection_value =
              float(injection[row * injection_stride + branch]) * 0.25f;
          const float injection_scale =
              2.0f * (1.0f / (1.0f + sycl::exp(-injection_value)));
          float feedback[kHidden / kLocal];
          float sum_sq = 0.0f;
#pragma unroll
          for (int i = 0; i < kHidden / kLocal; ++i) {
            const int col = lane + i * kLocal;
            const float h = float(hidden[offset + col]);
            const float b = float(block[row * kHidden + col]);
            const auto rounded = round_feedback(h + b * injection_scale);
            combined[offset + col] = sycl::bit_cast<half>(rounded.bits);
            feedback[i] = rounded.value;
            sum_sq += rounded.value * rounded.value;
          }
          const float sg_sum =
              sycl::reduce_over_group(sg, sum_sq, sycl::plus<float>());
          if (subgroup_lane == 0) partial[subgroup] = sg_sum;
          item.barrier(sycl::access::fence_space::local_space);
          const float pair =
              partial[subgroup_lane] + partial[subgroup_lane + kSubgroup];
          const float total =
              sycl::reduce_over_group(sg, pair, sycl::plus<float>());
          const float inv = sycl::rsqrt(total / float(kHidden) + eps);
#pragma unroll
          for (int i = 0; i < kHidden / kLocal; ++i) {
            const int col = lane + i * kLocal;
            const float w = float(weight[branch * kHidden + col]);
            normed[offset + col] = half(feedback[i] * inv * (1.0f + w));
          }
        });
  });
  return true;
}
}  // namespace vllm::qwen38::hc::manual_norm
