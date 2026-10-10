// SPDX-License-Identifier: Apache-2.0
#include "qwen38/hc_sycl.h"
#include "qwen38/hc_norm_m1_manual.h"

#include <ATen/MemoryOverlap.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/ext/intel/math/imf_fp_conversions.hpp>
#include <sycl/ext/oneapi/experimental/group_load_store.hpp>
#include <sycl/sycl.hpp>
#include <torch/library.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <type_traits>

namespace vllm::qwen38::hc {
namespace {

using half = sycl::half;
namespace sx = sycl::ext::oneapi::experimental;
constexpr int kStreams = 4;
constexpr int kHidden = 2560;
constexpr int kWidth = kStreams * kHidden;
constexpr int kRank = 320;
constexpr int kDownMerged = 336;
constexpr int kSg = 16;
constexpr int kLocal = 256;
constexpr int kNormLocal = 512;
constexpr int kPrefillMaxM = 4096;
constexpr auto kLoadProperties = sx::properties{
    sx::data_placement_striped,
    sx::contiguous_memory,
    sx::full_group,
    sx::alignment<4>};

void check_no_overlap(const torch::Tensor& a, const torch::Tensor& b) {
  at::assert_no_overlap(a, b);
  // ATen allows TooHard for padded views and No for separate Storage owners
  // of the same physical bytes. Dense tensors use their actual byte ranges.
  const bool padded =
      at::get_overlap_status(a, b) == at::MemOverlapStatus::TooHard;
  const auto a_start = reinterpret_cast<uintptr_t>(
      padded ? a.storage().data() : a.const_data_ptr());
  const auto b_start = reinterpret_cast<uintptr_t>(
      padded ? b.storage().data() : b.const_data_ptr());
  const auto a_bytes =
      padded ? a.storage().nbytes() : a.numel() * a.element_size();
  const auto b_bytes =
      padded ? b.storage().nbytes() : b.numel() * b.element_size();
  TORCH_CHECK(
      a_start <= b_start ? b_start - a_start >= a_bytes
                         : a_start - b_start >= b_bytes,
      "HC tensors have overlapping storage or unprovable overlap");
}

void check_input(
    const torch::Tensor& input, int width, const char* name, int max_m = 8) {
  TORCH_CHECK(input.is_xpu(), name, " must be on XPU");
  TORCH_CHECK(input.scalar_type() == at::kHalf, name, " must be FP16");
  TORCH_CHECK(
      input.dim() == 2 && input.size(0) >= 1 && input.size(0) <= max_m &&
          input.size(1) == width && input.is_contiguous() && !input.is_neg() &&
          !input.is_conj(),
      name,
      " has an invalid Qwen3.8 HC shape/layout");
}

void check_same(
    const torch::Tensor& t, const torch::Tensor& first, const char* name) {
  TORCH_CHECK(t.device() == first.device(), name, " must be on input device");
  TORCH_CHECK(t.scalar_type() == at::kHalf, name, " must be FP16");
  TORCH_CHECK(!t.is_neg() && !t.is_conj(), name, " must not be a lazy view");
}

void check_matrix(
    const torch::Tensor& t,
    const torch::Tensor& first,
    int rows,
    int cols,
    const char* name) {
  check_same(t, first, name);
  TORCH_CHECK(
      t.dim() == 2 && t.size(0) == rows && t.size(1) == cols &&
          t.is_contiguous(),
      name,
      " has an invalid shape/layout");
}

void check_norm(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& output,
    double eps,
    int max_m = 8) {
  check_input(input, kWidth, "input", max_m);
  check_same(weight, input, "norm weight");
  TORCH_CHECK(
      weight.dim() == 1 && weight.numel() == kWidth && weight.is_contiguous(),
      "norm weight must be [10240]");
  check_matrix(output, input, input.size(0), kWidth, "norm output");
  TORCH_CHECK(
      std::isfinite(eps) && eps > 0.0 &&
          std::isfinite(static_cast<float>(eps)) &&
          static_cast<float>(eps) > 0.0f,
      "eps must be positive finite FP32");
  check_no_overlap(output, input);
  check_no_overlap(output, weight);
}

void check_gate(
    const torch::Tensor& input,
    const torch::Tensor& gate,
    const torch::Tensor& output,
    int max_m = 8) {
  check_input(input, kWidth, "input", max_m);
  check_matrix(gate, input, input.size(0), kWidth, "gate");
  check_matrix(output, input, input.size(0), kHidden, "mixed output");
  check_no_overlap(output, input);
  check_no_overlap(output, gate);
}

void check_combine(
    const torch::Tensor& hidden,
    const torch::Tensor& block,
    const torch::Tensor& injection,
    const torch::Tensor& output,
    int max_m = 8) {
  check_input(hidden, kWidth, "hidden", max_m);
  check_matrix(block, hidden, hidden.size(0), kHidden, "block");
  check_same(injection, hidden, "injection");
  TORCH_CHECK(
      injection.dim() == 2 && injection.size(0) == hidden.size(0) &&
          injection.size(1) == kStreams && injection.stride(1) == 1 &&
          injection.stride(0) >= kStreams,
      "injection must be nonoverlapping [M,4] with unit column stride");
  check_matrix(output, hidden, hidden.size(0), kWidth, "combined output");
  check_no_overlap(output, hidden);
  check_no_overlap(output, block);
  check_no_overlap(output, injection);
}

void check_down(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& output) {
  check_input(input, kWidth, "down input");
  check_same(weight, input, "down weight");
  TORCH_CHECK(
      weight.dim() == 2 && weight.is_contiguous() &&
          (weight.size(0) == kRank || weight.size(0) == kDownMerged) &&
          weight.size(1) == kWidth,
      "down weight must be [320|336,10240]");
  check_matrix(output, input, input.size(0), weight.size(0), "down output");
  check_no_overlap(output, input);
  check_no_overlap(output, weight);
}

void check_up_input(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& output,
    int out_width) {
  TORCH_CHECK(
      input.is_xpu() && input.scalar_type() == at::kHalf && input.dim() == 2 &&
          input.size(0) >= 1 && input.size(0) <= 8 && input.size(1) == kRank &&
          input.stride(1) == 1 && input.stride(0) >= kRank && !input.is_neg() &&
          !input.is_conj(),
      "up input must be nonoverlapping FP16 XPU [M,320]");
  check_matrix(weight, input, kWidth, kRank, "up weight");
  check_matrix(output, input, input.size(0), out_width, "up output");
  check_no_overlap(output, input);
  check_no_overlap(output, weight);
}

void check_up_gate(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& normed,
    const torch::Tensor& output) {
  check_up_input(input, weight, output, kHidden);
  check_matrix(normed, input, input.size(0), kWidth, "normed");
  check_no_overlap(output, normed);
}

void check_distinct_outputs(
    std::initializer_list<const torch::Tensor*> outputs,
    std::initializer_list<const torch::Tensor*> inputs) {
  for (const auto* out : outputs) {
    for (const auto* in : inputs)
      check_no_overlap(*out, *in);
    for (const auto* other : outputs) {
      if (out != other) check_no_overlap(*out, *other);
    }
  }
}

void record(
    std::initializer_list<const torch::Tensor*> tensors,
    c10::xpu::XPUStream stream) {
  for (const auto* t : tensors)
    c10::xpu::XPUCachingAllocator::recordStream(
        t->storage().data_ptr(), stream);
}

const half* data(const torch::Tensor& t) {
  return reinterpret_cast<const half*>(t.data_ptr<at::Half>());
}

half* mutable_data(torch::Tensor& t) {
  return reinterpret_cast<half*>(t.data_ptr<at::Half>());
}

template <int Count, bool PackedCast = false>
inline sycl::vec<half, Count>
load_vector(sycl::sub_group sg, const half* ptr, bool aligned) {
  static_assert(Count % 2 == 0);
  sycl::vec<half, Count> result;
  if (aligned) {
    // The native subgroup block-read handles 32-bit elements. Requesting
    // half vectors directly gave invalid values on the B70/2025.3 stack.
    sycl::vec<uint32_t, Count / 2> packed;
    auto global_ptr = sycl::address_space_cast<
                          sycl::access::address_space::global_space,
                          sycl::access::decorated::yes>(
                          reinterpret_cast<const uint32_t*>(ptr))
                          .get_decorated();
    sx::group_load(sg, global_ptr, packed, kLoadProperties);
    if constexpr (PackedCast) {
      result = sycl::bit_cast<sycl::vec<half, Count>>(packed);
    } else {
#pragma unroll
      for (int i = 0; i < Count / 2; ++i) {
        const uint32_t bits = packed[i];
        result[2 * i] = sycl::bit_cast<half>(uint16_t(bits));
        result[2 * i + 1] = sycl::bit_cast<half>(uint16_t(bits >> 16));
      }
    }
  } else {
    const int lane = sg.get_local_linear_id();
#pragma unroll
    for (int i = 0; i < Count; ++i)
      result[i] = ptr[(i / 2) * (2 * kSg) + lane * 2 + (i & 1)];
  }
  return result;
}

inline float sigmoid(float value) { return 1.0f / (1.0f + sycl::exp(-value)); }

template <bool WithCombine>
class NormKernel;
template <bool WithCombine>
class PrefillNorm256Kernel;

template <bool WithCombine, int NormLocal>
void launch_norm_generic(
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
  static_assert(NormLocal == 256 || NormLocal == 512);
  static_assert(kHidden % NormLocal == 0);
  constexpr int subgroups = NormLocal / kSg;
  using Kernel = std::conditional_t<
      NormLocal == kNormLocal,
      NormKernel<WithCombine>,
      PrefillNorm256Kernel<WithCombine>>;
  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> partial(subgroups + 1, cgh);
    cgh.parallel_for<Kernel>(
        sycl::nd_range<1>(size_t(m * kStreams * NormLocal), NormLocal),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(kSg)]] {
          const int group = item.get_group_linear_id();
          const int row = group / kStreams;
          const int branch = group % kStreams;
          const int lane = item.get_local_linear_id();
          const auto sg = item.get_sub_group();
          const int subgroup = lane / kSg;
          const int subgroup_lane = sg.get_local_linear_id();
          const int offset = row * kWidth + branch * kHidden;
          const float injection_scale =
              WithCombine
                  ? 2.0f *
                        sigmoid(
                            float(injection[row * injection_stride + branch]) *
                            0.25f)
                  : 0.0f;
          half rounded[kHidden / NormLocal];
          float sum_sq = 0.0f;
#pragma unroll
          for (int i = 0; i < kHidden / NormLocal; ++i) {
            const int col = lane + i * NormLocal;
            const float h = float(hidden[offset + col]);
            const float b =
                WithCombine ? float(block[row * kHidden + col]) : 0.0f;
            half rounded_value = half(h + b * injection_scale);
            if constexpr (WithCombine) {
              volatile uint16_t rounded_bits =
                  sycl::bit_cast<uint16_t>(rounded_value);
              rounded_value = sycl::bit_cast<half>(uint16_t(rounded_bits));
              combined[offset + col] = rounded_value;
            }
            rounded[i] = rounded_value;
            const float value = float(rounded_value);
            sum_sq += value * value;
          }
          const float sg_sum =
              sycl::reduce_over_group(sg, sum_sq, sycl::plus<float>());
          if (subgroup_lane == 0) partial[subgroup] = sg_sum;
          item.barrier(sycl::access::fence_space::local_space);
          // WG256仅16个SG；WG512保持原来的两段partial相加顺序。
          float pair = partial[subgroup_lane];
          if constexpr (subgroups == 2 * kSg)
            pair += partial[subgroup_lane + kSg];
          const float total =
              sycl::reduce_over_group(sg, pair, sycl::plus<float>());
          const float inv = sycl::rsqrt(total / float(kHidden) + eps);
#pragma unroll
          for (int i = 0; i < kHidden / NormLocal; ++i) {
            const int col = lane + i * NormLocal;
            const float w = float(weight[branch * kHidden + col]);
            normed[offset + col] = half(float(rounded[i]) * inv * (1.0f + w));
          }
        });
  });
}

template <bool WithCombine>
void launch_norm(
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
  if constexpr (WithCombine) {
    if (m == 1) {
      manual_norm::launch(
          queue,
          hidden,
          block,
          injection,
          injection_stride,
          weight,
          combined,
          normed,
          eps);
      return;
    }
    if (manual_norm::launch_small_m(
            queue,
            hidden,
            block,
            injection,
            injection_stride,
            weight,
            combined,
            normed,
            m,
            eps)) {
      return;
    }
  }
  // combine+norm大M默认256；单独norm和旧small-M/manual仍默认512。
  if (m >= 9) {
    const char* local = std::getenv("QWEN38_HC_PREFILL_NORM_LOCAL");
    if ((!local && WithCombine) || (local && std::strcmp(local, "256") == 0)) {
      launch_norm_generic<WithCombine, 256>(
          queue,
          hidden,
          block,
          injection,
          injection_stride,
          weight,
          combined,
          normed,
          m,
          eps);
      return;
    }
  }
  launch_norm_generic<WithCombine, 512>(
      queue,
      hidden,
      block,
      injection,
      injection_stride,
      weight,
      combined,
      normed,
      m,
      eps);
}

class GateKernel;
void launch_gate(
    sycl::queue& queue,
    const half* input,
    const half* gate,
    half* output,
    int m) {
  queue.parallel_for<GateKernel>(
      sycl::nd_range<1>(size_t(m * kHidden), kLocal),
      [=](sycl::nd_item<1> item) {
        const int index = item.get_global_linear_id();
        const int row = index / kHidden;
        const int col = index % kHidden;
        float mixed = 0.0f;
#pragma unroll
        for (int branch = 0; branch < kStreams; ++branch) {
          const int offset = row * kWidth + branch * kHidden + col;
          mixed += float(input[offset]) * sigmoid(float(gate[offset]));
        }
        output[index] = half(mixed * 0.25f);
      });
}

class CombineKernel;
void launch_combine(
    sycl::queue& queue,
    const half* hidden,
    const half* block,
    const half* injection,
    int64_t injection_stride,
    half* output,
    int m) {
  queue.parallel_for<CombineKernel>(
      sycl::nd_range<1>(size_t(m * kWidth), kLocal),
      [=](sycl::nd_item<1> item) {
        const int index = item.get_global_linear_id();
        const int row = index / kWidth;
        const int branch = (index / kHidden) % kStreams;
        const int col = index % kHidden;
        const float scale =
            2.0f *
            sigmoid(float(injection[row * injection_stride + branch]) * 0.25f);
        output[index] = half(
            float(hidden[index]) + float(block[row * kHidden + col]) * scale);
      });
}

template <int Rows, int Splits>
class DownKernel;

template <int Rows, int Splits>
void launch_down(
    sycl::queue& queue,
    const half* input,
    const half* weight,
    half* output,
    int m,
    int n,
    bool x_aligned,
    bool w_aligned) {
  const int tiles = (m + Rows - 1) / Rows;
  const int groups = tiles * n;
  // Subgroup K-splits provide row-level parallelism without global scratch.
  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> partial(Rows * Splits, cgh);
    cgh.parallel_for<DownKernel<Rows, Splits>>(
        sycl::nd_range<1>(size_t(groups) * Splits * kSg, Splits * kSg),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(kSg)]] {
          const auto sg = item.get_sub_group();
          const int group = item.get_group_linear_id();
          const int row = group / n;
          const int out_col = group % n;
          const int token = row * Rows;
          const int lane = sg.get_local_linear_id();
          const int split = item.get_local_linear_id() / kSg;
          constexpr int elements_per_lane = 16;
          float acc[Rows][elements_per_lane] = {};
          const half* w = weight + int64_t(out_col) * kWidth;
#pragma unroll 2
          for (int k = split * (kWidth / Splits);
               k < (split + 1) * (kWidth / Splits);
               k += elements_per_lane * kSg) {
            const auto weight_v = load_vector < elements_per_lane,
                       Rows == 1 > (sg, w + k, w_aligned);
#pragma unroll
            for (int r = 0; r < Rows; ++r) {
              if (token + r >= m) continue;
              const auto x = load_vector < elements_per_lane,
                         Rows == 1 > (sg,
                                      input + int64_t(token + r) * kWidth + k,
                                      x_aligned);
#pragma unroll
              for (int i = 0; i < elements_per_lane; ++i)
                acc[r][i] =
                    sycl::fma(float(x[i]), float(weight_v[i]), acc[r][i]);
            }
          }
#pragma unroll
          for (int r = 0; r < Rows; ++r) {
            float lane_sum = 0.0f;
#pragma unroll
            for (int i = 0; i < elements_per_lane; ++i)
              lane_sum += acc[r][i];
            const float sum =
                sycl::reduce_over_group(sg, lane_sum, sycl::plus<float>());
            if (lane == 0) partial[r * Splits + split] = sum;
          }
          item.barrier(sycl::access::fence_space::local_space);
          if (item.get_local_linear_id() == 0) {
#pragma unroll
            for (int r = 0; r < Rows; ++r) {
              if (token + r >= m) continue;
              float sum = 0.0f;
#pragma unroll
              for (int split_index = 0; split_index < Splits; ++split_index)
                sum += partial[r * Splits + split_index];
              half value = half(sum);
              if (out_col < kRank) {
                const half scaled = half(float(value) * 0.25f);
                const float z = float(scaled);
                value = half(z * sigmoid(z));
              }
              output[(token + r) * n + out_col] = value;
            }
          }
        });
  });
}

template <bool FusedGate>
class UpKernel;

class UpGateM1PairKernel;

void launch_up_gate_m1_pair(
    sycl::queue& queue,
    const half* input,
    const half* weight,
    const half* normed,
    half* output,
    bool x_aligned,
    bool w_aligned) {
  static_assert(kHidden % 2 == 0);
  constexpr int groups = kHidden / 2;
  constexpr size_t global = size_t((groups + 7) / 8) * 128;
  // 每个子组顺序计算相邻两列，仅共享低秩输入，不同时保留八行权重。
  queue.parallel_for<UpGateM1PairKernel>(
      sycl::nd_range<1>(global, 128),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(kSg)]] {
        const auto sg = item.get_sub_group();
        const int group = item.get_global_linear_id() / kSg;
        if (group >= groups) return;
        const int h = group * 2;
        const int lane = sg.get_local_linear_id();
        const auto x0 = load_vector<16>(sg, input, x_aligned);
        const auto x1 = load_vector<4>(sg, input + 256, x_aligned);
        float mixed[2] = {};
#pragma unroll
        for (int branch = 0; branch < kStreams; ++branch) {
#pragma unroll
          for (int col = 0; col < 2; ++col) {
            const int out_col = branch * kHidden + h + col;
            const half* w = weight + int64_t(out_col) * kRank;
            const auto w0 = load_vector<16>(sg, w, w_aligned);
            const auto w1 = load_vector<4>(sg, w + 256, w_aligned);
            float lane_sum = 0.0f;
        // 每个输出仍按原来的 16+4 次 FMA 和子组归约顺序计算。
#pragma unroll
            for (int i = 0; i < 16; ++i)
              lane_sum = sycl::fma(float(x0[i]), float(w0[i]), lane_sum);
#pragma unroll
            for (int i = 0; i < 4; ++i)
              lane_sum = sycl::fma(float(x1[i]), float(w1[i]), lane_sum);
            const float total =
                sycl::reduce_over_group(sg, lane_sum, sycl::plus<float>());
            if (lane == 0) {
              // 保留可观察的 FP16 舍入边界，不能折叠成 FP32 sigmoid。
              volatile uint16_t gate_bits = sycl::bit_cast<uint16_t>(
                  sycl::ext::intel::math::float2half_rn(total));
              const half gate = sycl::bit_cast<half>(uint16_t(gate_bits));
              mixed[col] += float(normed[out_col]) * sigmoid(float(gate));
            }
          }
        }
        if (lane == 0) {
#pragma unroll
          for (int col = 0; col < 2; ++col)
            output[h + col] = half(mixed[col] * 0.25f);
        }
      });
}

class UpGateMultiPairKernel;

void launch_up_gate_multi_pair(
    sycl::queue& queue,
    const half* input,
    int64_t input_stride,
    const half* weight,
    const half* normed,
    half* output,
    int m,
    bool x_aligned,
    bool w_aligned) {
  static_assert(kHidden % 2 == 0);
  constexpr int pairs = kHidden / 2;
  const int groups = m * pairs;
  const size_t global = size_t((groups + 7) / 8) * 128;
  // 每个 SG 只复用本行的低秩输入；不做跨 token 权重复用。
  queue.parallel_for<UpGateMultiPairKernel>(
      sycl::nd_range<1>(global, 128),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(kSg)]] {
        const auto sg = item.get_sub_group();
        const int group = item.get_global_linear_id() / kSg;
        if (group >= groups) return;
        const int row = group / pairs;
        const int h = (group % pairs) * 2;
        const int lane = sg.get_local_linear_id();
        const half* x = input + int64_t(row) * input_stride;
        const auto x0 = load_vector<16>(sg, x, x_aligned);
        const auto x1 = load_vector<4>(sg, x + 256, x_aligned);
        float mixed[2] = {};
#pragma unroll
        for (int branch = 0; branch < kStreams; ++branch) {
#pragma unroll
          for (int col = 0; col < 2; ++col) {
            const int out_col = branch * kHidden + h + col;
            const half* w = weight + int64_t(out_col) * kRank;
            const auto w0 = load_vector<16>(sg, w, w_aligned);
            const auto w1 = load_vector<4>(sg, w + 256, w_aligned);
            float lane_sum = 0.0f;
        // 保留原16+4次FMA、SG归约及每个输出的四分支累加顺序。
#pragma unroll
            for (int i = 0; i < 16; ++i)
              lane_sum = sycl::fma(float(x0[i]), float(w0[i]), lane_sum);
#pragma unroll
            for (int i = 0; i < 4; ++i)
              lane_sum = sycl::fma(float(x1[i]), float(w1[i]), lane_sum);
            const float total =
                sycl::reduce_over_group(sg, lane_sum, sycl::plus<float>());
            if (lane == 0) {
              volatile uint16_t gate_bits = sycl::bit_cast<uint16_t>(
                  sycl::ext::intel::math::float2half_rn(total));
              const half gate = sycl::bit_cast<half>(uint16_t(gate_bits));
              mixed[col] +=
                  float(normed[row * kWidth + out_col]) * sigmoid(float(gate));
            }
          }
        }
        if (lane == 0) {
#pragma unroll
          for (int col = 0; col < 2; ++col)
            output[row * kHidden + h + col] = half(mixed[col] * 0.25f);
        }
      });
}

template <bool FusedGate>
void launch_up(
    sycl::queue& queue,
    const half* input,
    int64_t input_stride,
    const half* weight,
    const half* normed,
    half* output,
    int m,
    bool x_aligned,
    bool w_aligned) {
  if constexpr (FusedGate) {
    if (m == 1) {
      // M1默认复用相邻两列；显式0保留旧路径，图A/B需分别捕获。
      const char* pair = std::getenv("QWEN38_HC_M1_UP_GATE_PAIR");
      if (!pair || (pair[0] == '1' && pair[1] == '\0')) {
        launch_up_gate_m1_pair(
            queue, input, weight, normed, output, x_aligned, w_aligned);
        return;
      }
    }
    if (m >= 2 && m <= 8) {
      const char* pair = std::getenv("QWEN38_HC_MULTI_UP_GATE_PAIR");
      if (!pair || (pair[0] == '1' && pair[1] == '\0')) {
        launch_up_gate_multi_pair(
            queue,
            input,
            input_stride,
            weight,
            normed,
            output,
            m,
            x_aligned,
            w_aligned);
        return;
      }
    }
  }
  const int n = FusedGate ? kHidden : kWidth;
  const int groups = m * n;
  const size_t global = size_t((groups + 7) / 8) * 128;
  queue.parallel_for<UpKernel<FusedGate>>(
      sycl::nd_range<1>(global, 128),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(kSg)]] {
        const auto sg = item.get_sub_group();
        const int group = item.get_global_linear_id() / kSg;
        if (group >= groups) return;
        const int row = group / n;
        const int h = group % n;
        const int lane = sg.get_local_linear_id();
        const half* x = input + int64_t(row) * input_stride;
        const auto x0 = load_vector<16>(sg, x, x_aligned);
        const auto x1 = load_vector<4>(sg, x + 256, x_aligned);
        float mixed = 0.0f;
        constexpr int branches = FusedGate ? kStreams : 1;
#pragma unroll
        for (int branch = 0; branch < branches; ++branch) {
          const int out_col = FusedGate ? branch * kHidden + h : h;
          const half* w = weight + int64_t(out_col) * kRank;
          const auto w0 = load_vector<16>(sg, w, w_aligned);
          const auto w1 = load_vector<4>(sg, w + 256, w_aligned);
          float lane_sum = 0.0f;
#pragma unroll
          for (int i = 0; i < 16; ++i)
            lane_sum = sycl::fma(float(x0[i]), float(w0[i]), lane_sum);
#pragma unroll
          for (int i = 0; i < 4; ++i)
            lane_sum = sycl::fma(float(x1[i]), float(w1[i]), lane_sum);
          const float total =
              sycl::reduce_over_group(sg, lane_sum, sycl::plus<float>());
          if (lane == 0) {
            volatile uint16_t gate_bits = sycl::bit_cast<uint16_t>(
                sycl::ext::intel::math::float2half_rn(total));
            const half gate = sycl::bit_cast<half>(uint16_t(gate_bits));
            if constexpr (FusedGate)
              mixed +=
                  float(normed[row * kWidth + out_col]) * sigmoid(float(gate));
            else
              output[row * kWidth + out_col] = gate;
          }
        }
        if constexpr (FusedGate)
          if (lane == 0) output[row * kHidden + h] = half(mixed * 0.25f);
      });
}

}  // namespace

void grouped_norm(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    torch::Tensor& output,
    double eps) {
  check_norm(input, weight, output, eps);
  const c10::DeviceGuard guard(input.device());
  const auto stream = c10::xpu::getCurrentXPUStream(input.get_device());
  record({&input, &weight, &output}, stream);
  launch_norm<false>(
      stream.queue(),
      data(input),
      nullptr,
      nullptr,
      0,
      data(weight),
      nullptr,
      mutable_data(output),
      input.size(0),
      float(eps));
}

void gate_mix(
    const torch::Tensor& input,
    const torch::Tensor& gate,
    torch::Tensor& output) {
  check_gate(input, gate, output);
  const c10::DeviceGuard guard(input.device());
  const auto stream = c10::xpu::getCurrentXPUStream(input.get_device());
  record({&input, &gate, &output}, stream);
  launch_gate(
      stream.queue(),
      data(input),
      data(gate),
      mutable_data(output),
      input.size(0));
}

void combine(
    const torch::Tensor& hidden,
    const torch::Tensor& block,
    const torch::Tensor& injection,
    torch::Tensor& output) {
  check_combine(hidden, block, injection, output);
  const c10::DeviceGuard guard(hidden.device());
  const auto stream = c10::xpu::getCurrentXPUStream(hidden.get_device());
  record({&hidden, &block, &injection, &output}, stream);
  launch_combine(
      stream.queue(),
      data(hidden),
      data(block),
      data(injection),
      injection.stride(0),
      mutable_data(output),
      hidden.size(0));
}

void combine_norm(
    const torch::Tensor& hidden,
    const torch::Tensor& block,
    const torch::Tensor& injection,
    const torch::Tensor& weight,
    torch::Tensor& combined,
    torch::Tensor& normed,
    double eps) {
  check_combine(hidden, block, injection, combined);
  check_norm(hidden, weight, normed, eps);
  check_distinct_outputs(
      {&combined, &normed}, {&hidden, &block, &injection, &weight});
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
      mutable_data(combined),
      mutable_data(normed),
      hidden.size(0),
      float(eps));
}

void prefill_grouped_norm(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    torch::Tensor& output,
    double eps) {
  check_norm(input, weight, output, eps, kPrefillMaxM);
  TORCH_CHECK(input.size(0) >= 9, "HC prefill requires M=9..4096");
  const c10::DeviceGuard guard(input.device());
  const auto stream = c10::xpu::getCurrentXPUStream(input.get_device());
  auto& queue = stream.queue();
  TORCH_CHECK(
      queue.is_in_order(), "HC prefill requires an in-order current stream");
  record({&input, &weight, &output}, stream);
  launch_norm<false>(
      queue,
      data(input),
      nullptr,
      nullptr,
      0,
      data(weight),
      nullptr,
      mutable_data(output),
      input.size(0),
      float(eps));
}

void prefill_gate_mix(
    const torch::Tensor& input,
    const torch::Tensor& gate,
    torch::Tensor& output) {
  check_gate(input, gate, output, kPrefillMaxM);
  TORCH_CHECK(input.size(0) >= 9, "HC prefill requires M=9..4096");
  const c10::DeviceGuard guard(input.device());
  const auto stream = c10::xpu::getCurrentXPUStream(input.get_device());
  auto& queue = stream.queue();
  TORCH_CHECK(
      queue.is_in_order(), "HC prefill requires an in-order current stream");
  record({&input, &gate, &output}, stream);
  launch_gate(
      queue, data(input), data(gate), mutable_data(output), input.size(0));
}

void prefill_combine(
    const torch::Tensor& hidden,
    const torch::Tensor& block,
    const torch::Tensor& injection,
    torch::Tensor& output) {
  check_combine(hidden, block, injection, output, kPrefillMaxM);
  TORCH_CHECK(hidden.size(0) >= 9, "HC prefill requires M=9..4096");
  const c10::DeviceGuard guard(hidden.device());
  const auto stream = c10::xpu::getCurrentXPUStream(hidden.get_device());
  auto& queue = stream.queue();
  TORCH_CHECK(
      queue.is_in_order(), "HC prefill requires an in-order current stream");
  record({&hidden, &block, &injection, &output}, stream);
  launch_combine(
      queue,
      data(hidden),
      data(block),
      data(injection),
      injection.stride(0),
      mutable_data(output),
      hidden.size(0));
}

void prefill_combine_norm(
    const torch::Tensor& hidden,
    const torch::Tensor& block,
    const torch::Tensor& injection,
    const torch::Tensor& weight,
    torch::Tensor& combined,
    torch::Tensor& normed,
    double eps) {
  check_combine(hidden, block, injection, combined, kPrefillMaxM);
  check_norm(hidden, weight, normed, eps, kPrefillMaxM);
  TORCH_CHECK(hidden.size(0) >= 9, "HC prefill requires M=9..4096");
  check_distinct_outputs(
      {&combined, &normed}, {&hidden, &block, &injection, &weight});
  const c10::DeviceGuard guard(hidden.device());
  const auto stream = c10::xpu::getCurrentXPUStream(hidden.get_device());
  auto& queue = stream.queue();
  TORCH_CHECK(
      queue.is_in_order(), "HC prefill requires an in-order current stream");
  record({&hidden, &block, &injection, &weight, &combined, &normed}, stream);
  // M>8 自动避开 manual small-M 路径；真实 FP16 combine 边界仍由原 kernel
  // 保留。
  launch_norm<true>(
      queue,
      data(hidden),
      data(block),
      data(injection),
      injection.stride(0),
      data(weight),
      mutable_data(combined),
      mutable_data(normed),
      hidden.size(0),
      float(eps));
}

void down(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    torch::Tensor& output) {
  check_down(input, weight, output);
  const c10::DeviceGuard guard(input.device());
  const auto stream = c10::xpu::getCurrentXPUStream(input.get_device());
  record({&input, &weight, &output}, stream);
  const bool x_aligned = !(reinterpret_cast<uintptr_t>(input.data_ptr()) & 3);
  const bool w_aligned = !(reinterpret_cast<uintptr_t>(weight.data_ptr()) & 3);
  if (input.size(0) == 1)
    launch_down<1, 8>(
        stream.queue(),
        data(input),
        data(weight),
        mutable_data(output),
        input.size(0),
        weight.size(0),
        x_aligned,
        w_aligned);
  else
    launch_down<2, 4>(
        stream.queue(),
        data(input),
        data(weight),
        mutable_data(output),
        input.size(0),
        weight.size(0),
        x_aligned,
        w_aligned);
}

void up(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    torch::Tensor& output) {
  check_up_input(input, weight, output, kWidth);
  const c10::DeviceGuard guard(input.device());
  const auto stream = c10::xpu::getCurrentXPUStream(input.get_device());
  record({&input, &weight, &output}, stream);
  const bool x_aligned = !(reinterpret_cast<uintptr_t>(input.data_ptr()) & 3) &&
                         input.stride(0) % 2 == 0;
  const bool w_aligned = !(reinterpret_cast<uintptr_t>(weight.data_ptr()) & 3);
  launch_up<false>(
      stream.queue(),
      data(input),
      input.stride(0),
      data(weight),
      nullptr,
      mutable_data(output),
      input.size(0),
      x_aligned,
      w_aligned);
}

void up_gate_mix(
    const torch::Tensor& lowrank,
    const torch::Tensor& weight,
    const torch::Tensor& normed,
    torch::Tensor& output) {
  check_up_gate(lowrank, weight, normed, output);
  const c10::DeviceGuard guard(lowrank.device());
  const auto stream = c10::xpu::getCurrentXPUStream(lowrank.get_device());
  record({&lowrank, &weight, &normed, &output}, stream);
  const bool x_aligned =
      !(reinterpret_cast<uintptr_t>(lowrank.data_ptr()) & 3) &&
      lowrank.stride(0) % 2 == 0;
  const bool w_aligned = !(reinterpret_cast<uintptr_t>(weight.data_ptr()) & 3);
  launch_up<true>(
      stream.queue(),
      data(lowrank),
      lowrank.stride(0),
      data(weight),
      data(normed),
      mutable_data(output),
      lowrank.size(0),
      x_aligned,
      w_aligned);
}

static void combine_mix_impl(
    const torch::Tensor& hidden,
    const torch::Tensor& block,
    const torch::Tensor& injection,
    const torch::Tensor& norm_weight,
    const torch::Tensor& down_weight,
    const torch::Tensor& up_weight,
    torch::Tensor& combined,
    torch::Tensor& normed,
    torch::Tensor& down_output,
    torch::Tensor& mixed,
    double eps,
    bool inputs_checked) {
  // The whole transaction is checked before the first of three submissions.
  if (!inputs_checked) {
    check_combine(hidden, block, injection, combined);
    check_norm(hidden, norm_weight, normed, eps);
    check_down(hidden, down_weight, down_output);
    TORCH_CHECK(
        down_weight.size(0) == kDownMerged,
        "combine_mix requires 336 down rows for injection");
    check_matrix(up_weight, hidden, kWidth, kRank, "up weight");
  } else {
    const int64_t rows = hidden.size(0);
    check_matrix(combined, hidden, rows, kWidth, "combined output");
    check_matrix(normed, hidden, rows, kWidth, "normed output");
    check_matrix(down_output, hidden, rows, kDownMerged, "down output");
  }
  check_matrix(mixed, hidden, hidden.size(0), kHidden, "mixed output");
  check_distinct_outputs(
      {&combined, &normed, &down_output, &mixed},
      {&hidden, &block, &injection, &norm_weight, &down_weight, &up_weight});
  const c10::DeviceGuard guard(hidden.device());
  const auto stream = c10::xpu::getCurrentXPUStream(hidden.get_device());
  auto& queue = stream.queue();
  TORCH_CHECK(
      queue.is_in_order(),
      "HC transaction requires an in-order current stream");
  // A later submission may throw after norm/down already use these pointers.
  record(
      {&hidden,
       &block,
       &injection,
       &norm_weight,
       &down_weight,
       &up_weight,
       &combined,
       &normed,
       &down_output,
       &mixed},
      stream);
  launch_norm<true>(
      queue,
      data(hidden),
      data(block),
      data(injection),
      injection.stride(0),
      data(norm_weight),
      mutable_data(combined),
      mutable_data(normed),
      hidden.size(0),
      float(eps));
  const bool norm_aligned =
      !(reinterpret_cast<uintptr_t>(normed.data_ptr()) & 3);
  const bool down_w_aligned =
      !(reinterpret_cast<uintptr_t>(down_weight.data_ptr()) & 3);
  if (hidden.size(0) == 1)
    launch_down<1, 8>(
        queue,
        data(normed),
        data(down_weight),
        mutable_data(down_output),
        hidden.size(0),
        kDownMerged,
        norm_aligned,
        down_w_aligned);
  else
    launch_down<2, 4>(
        queue,
        data(normed),
        data(down_weight),
        mutable_data(down_output),
        hidden.size(0),
        kDownMerged,
        norm_aligned,
        down_w_aligned);
  const bool low_aligned =
      !(reinterpret_cast<uintptr_t>(down_output.data_ptr()) & 3);
  const bool up_w_aligned =
      !(reinterpret_cast<uintptr_t>(up_weight.data_ptr()) & 3);
  launch_up<true>(
      queue,
      data(down_output),
      kDownMerged,
      data(up_weight),
      data(normed),
      mutable_data(mixed),
      hidden.size(0),
      low_aligned,
      up_w_aligned);
}

void combine_mix(
    const torch::Tensor& hidden,
    const torch::Tensor& block,
    const torch::Tensor& injection,
    const torch::Tensor& norm_weight,
    const torch::Tensor& down_weight,
    const torch::Tensor& up_weight,
    torch::Tensor& combined,
    torch::Tensor& normed,
    torch::Tensor& down_output,
    torch::Tensor& mixed,
    double eps) {
  combine_mix_impl(
      hidden,
      block,
      injection,
      norm_weight,
      down_weight,
      up_weight,
      combined,
      normed,
      down_output,
      mixed,
      eps,
      false);
}

void combine_mix_prechecked_inputs(
    const torch::Tensor& hidden,
    const torch::Tensor& block,
    const torch::Tensor& injection,
    const torch::Tensor& norm_weight,
    const torch::Tensor& down_weight,
    const torch::Tensor& up_weight,
    torch::Tensor& combined,
    torch::Tensor& normed,
    torch::Tensor& down_output,
    torch::Tensor& mixed,
    double eps) {
  combine_mix_impl(
      hidden,
      block,
      injection,
      norm_weight,
      down_weight,
      up_weight,
      combined,
      normed,
      down_output,
      mixed,
      eps,
      true);
}

void project_mix(
    const torch::Tensor& normed,
    const torch::Tensor& down_weight,
    const torch::Tensor& up_weight,
    torch::Tensor& down_output,
    torch::Tensor& mixed) {
  // In particular, up_weight/mixed are checked before launching down.
  check_down(normed, down_weight, down_output);
  TORCH_CHECK(
      down_weight.size(0) == kDownMerged,
      "project_mix requires 336 down rows for injection");
  check_matrix(up_weight, normed, kWidth, kRank, "up weight");
  check_matrix(mixed, normed, normed.size(0), kHidden, "mixed output");
  check_distinct_outputs(
      {&down_output, &mixed}, {&normed, &down_weight, &up_weight});
  const c10::DeviceGuard guard(normed.device());
  const auto stream = c10::xpu::getCurrentXPUStream(normed.get_device());
  auto& queue = stream.queue();
  TORCH_CHECK(
      queue.is_in_order(),
      "HC transaction requires an in-order current stream");
  record({&normed, &down_weight, &up_weight, &down_output, &mixed}, stream);
  const bool norm_aligned =
      !(reinterpret_cast<uintptr_t>(normed.data_ptr()) & 3);
  const bool down_w_aligned =
      !(reinterpret_cast<uintptr_t>(down_weight.data_ptr()) & 3);
  if (normed.size(0) == 1)
    launch_down<1, 8>(
        queue,
        data(normed),
        data(down_weight),
        mutable_data(down_output),
        normed.size(0),
        kDownMerged,
        norm_aligned,
        down_w_aligned);
  else
    launch_down<2, 4>(
        queue,
        data(normed),
        data(down_weight),
        mutable_data(down_output),
        normed.size(0),
        kDownMerged,
        norm_aligned,
        down_w_aligned);
  const bool low_aligned =
      !(reinterpret_cast<uintptr_t>(down_output.data_ptr()) & 3);
  const bool up_w_aligned =
      !(reinterpret_cast<uintptr_t>(up_weight.data_ptr()) & 3);
  launch_up<true>(
      queue,
      data(down_output),
      kDownMerged,
      data(up_weight),
      data(normed),
      mutable_data(mixed),
      normed.size(0),
      low_aligned,
      up_w_aligned);
}

}  // namespace vllm::qwen38::hc

TORCH_LIBRARY(qwen38_hc_sycl, m) {
  m.def(
      "grouped_norm(Tensor input, Tensor weight, Tensor(a!) output, float eps) "
      "-> ()");
  m.impl("grouped_norm", torch::kXPU, &vllm::qwen38::hc::grouped_norm);
  m.def("gate_mix(Tensor input, Tensor gate, Tensor(a!) output) -> ()");
  m.impl("gate_mix", torch::kXPU, &vllm::qwen38::hc::gate_mix);
  m.def(
      "combine(Tensor hidden, Tensor block, Tensor injection, Tensor(a!) "
      "output) -> ()");
  m.impl("combine", torch::kXPU, &vllm::qwen38::hc::combine);
  m.def(
      "combine_norm(Tensor hidden, Tensor block, Tensor injection, Tensor "
      "weight, Tensor(a!) combined, Tensor(b!) normed, float eps) -> ()");
  m.impl("combine_norm", torch::kXPU, &vllm::qwen38::hc::combine_norm);
  m.def("down(Tensor input, Tensor weight, Tensor(a!) output) -> ()");
  m.impl("down", torch::kXPU, &vllm::qwen38::hc::down);
  m.def("up(Tensor input, Tensor weight, Tensor(a!) output) -> ()");
  m.impl("up", torch::kXPU, &vllm::qwen38::hc::up);
  m.def(
      "up_gate_mix(Tensor lowrank, Tensor weight, Tensor normed, Tensor(a!) "
      "output) -> ()");
  m.impl("up_gate_mix", torch::kXPU, &vllm::qwen38::hc::up_gate_mix);
  m.def(
      "combine_mix(Tensor hidden, Tensor block, Tensor injection, Tensor "
      "norm_weight, Tensor down_weight, Tensor up_weight, Tensor(a!) combined, "
      "Tensor(b!) normed, Tensor(c!) down_output, Tensor(d!) mixed, float eps) "
      "-> ()");
  m.impl("combine_mix", torch::kXPU, &vllm::qwen38::hc::combine_mix);
  m.def(
      "project_mix(Tensor normed, Tensor down_weight, Tensor up_weight, "
      "Tensor(a!) down_output, Tensor(b!) mixed) -> ()");
  m.impl("project_mix", torch::kXPU, &vllm::qwen38::hc::project_mix);
}
