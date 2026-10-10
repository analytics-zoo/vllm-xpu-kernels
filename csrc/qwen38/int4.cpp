// SPDX-License-Identifier: Apache-2.0
#include <ATen/MemoryOverlap.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/ext/oneapi/experimental/group_load_store.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/sycl.hpp>

#include <cstdint>
#include <initializer_list>
#include <limits>

#include "qwen38/int4_small_m.h"
#include "qwen38/ops.h"

namespace vllm::qwen38 {
namespace {

using half = sycl::half;
using bfloat16 = sycl::ext::oneapi::bfloat16;
constexpr int kGroup = 128;

void check_no_overlap(const torch::Tensor& a, const torch::Tensor& b) {
  at::assert_no_overlap(a, b);
  // All INT4 tensors are contiguous; ATen can miss physical overlap between
  // independent Storage owners (for example a DLPack round trip).
  const auto a_start = reinterpret_cast<uintptr_t>(a.const_data_ptr());
  const auto b_start = reinterpret_cast<uintptr_t>(b.const_data_ptr());
  const auto a_bytes = a.numel() * a.element_size();
  const auto b_bytes = b.numel() * b.element_size();
  TORCH_CHECK(
      a_start <= b_start ? b_start - a_start >= a_bytes
                         : a_start - b_start >= b_bytes,
      "INT4 tensors have overlapping physical memory");
}

void check_tensor(
    const torch::Tensor& t,
    const torch::Tensor& input,
    at::ScalarType dtype,
    const char* name) {
  TORCH_CHECK(
      t.device() == input.device(), name, " must be on the input device");
  TORCH_CHECK(t.scalar_type() == dtype, name, " has an invalid dtype");
  TORCH_CHECK(
      t.dim() == 2 && t.is_contiguous(), name, " must be contiguous 2D");
  TORCH_CHECK(!t.is_neg() && !t.is_conj(), name, " must not be a lazy view");
}

void check_input(const torch::Tensor& input) {
  TORCH_CHECK(input.is_xpu(), "input must be an XPU tensor");
  TORCH_CHECK(
      input.dim() == 2 && input.is_contiguous(), "input must be contiguous 2D");
  TORCH_CHECK(
      !input.is_neg() && !input.is_conj(), "input must not be a lazy view");
  TORCH_CHECK(
      input.size(0) > 0 && input.size(1) > 0 && input.size(1) % kGroup == 0,
      "input must have positive dimensions and K divisible by 128");
  TORCH_CHECK(
      input.size(0) <= std::numeric_limits<int>::max() &&
          input.size(1) <= std::numeric_limits<int>::max(),
      "input dimensions exceed the kernel index range");
}

void check_projection(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& scale,
    const torch::Tensor& output) {
  check_tensor(weight, input, at::kByte, "weight");
  check_tensor(scale, input, at::kHalf, "scale");
  check_tensor(output, input, at::kHalf, "output");
  const auto n = weight.size(0);
  TORCH_CHECK(n > 0 && n <= std::numeric_limits<int>::max(), "invalid N");
  TORCH_CHECK(weight.size(1) == input.size(1) / 2, "weight must be [N,K/2]");
  TORCH_CHECK(
      scale.size(0) == n && scale.size(1) == input.size(1) / kGroup,
      "scale must be [N,K/128]");
  TORCH_CHECK(
      output.size(0) == input.size(0) && output.size(1) == n,
      "output must be [M,N]");
  check_no_overlap(output, input);
  check_no_overlap(output, weight);
  check_no_overlap(output, scale);
}

// Preserve allocations when a caller drops tensors before a non-default
// stream completes. No host/device synchronization is needed here.
void record_stream(
    std::initializer_list<const torch::Tensor*> tensors,
    c10::xpu::XPUStream stream) {
  for (auto* t : tensors) {
    c10::xpu::XPUCachingAllocator::recordStream(
        t->storage().data_ptr(), stream);
  }
}

template <typename Scalar>
class QuantizeKernel;

template <typename Scalar>
void launch_quantize(
    sycl::queue& queue,
    const Scalar* input,
    uint32_t* qweight,
    half* scale,
    int64_t rows,
    int64_t k) {
  constexpr int sg_size = 32;
  constexpr int local_size = 128;
  const int64_t blocks = rows * (k / kGroup);
  const size_t global_size = ((blocks + 3) / 4) * local_size;
  queue.parallel_for<QuantizeKernel<Scalar>>(
      sycl::nd_range<1>(global_size, local_size),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(sg_size)]] {
        const auto sg = item.get_sub_group();
        const int lane = sg.get_local_linear_id();
        const int64_t block = item.get_global_linear_id() / sg_size;
        if (block >= blocks) {
          return;
        }
        float values[4];
        float max_pos = -std::numeric_limits<float>::infinity();
        float neg_max = -std::numeric_limits<float>::infinity();
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          values[i] = static_cast<float>(input[block * kGroup + lane * 4 + i]);
          max_pos = sycl::fmax(max_pos, values[i]);
          neg_max = sycl::fmax(neg_max, -values[i]);
        }
        max_pos = sycl::reduce_over_group(sg, max_pos, sycl::maximum<float>());
        neg_max = sycl::reduce_over_group(sg, neg_max, sycl::maximum<float>());
        const float signed_max = max_pos >= neg_max ? max_pos : -neg_max;
        const float d = signed_max / -8.0f;
        const float inv_d = d != 0.0f ? 1.0f / d : 0.0f;
        uint32_t packed = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float scaled = values[i] * inv_d;
          const int q = sycl::clamp(static_cast<int>(scaled + 8.5f), 0, 15);
          packed |= static_cast<uint32_t>(q) << (i * 4);
        }
        const uint32_t neighbor = sycl::select_from_group(sg, packed, lane ^ 1);
        if ((lane & 1) == 0) {
          qweight[block * 16 + lane / 2] = packed | (neighbor << 16);
        }
        if (lane == 0) {
          scale[block] = static_cast<half>(d);
        }
      });
}

struct Projection {
  const half* input;
  const uint8_t* weight0;
  const half* scale0;
  half* output0;
  const uint8_t* weight1;
  const half* scale1;
  half* output1;
  int m;
  int k;
  int n0;
  int n1;
};

template <int Rows, bool Fused, bool FixedTP4, int Subgroup>
class WideLinearKernel;

// Subgroup block reads cover four scale groups per iteration. Each input
// uint64 and weight uint16 contain four corresponding FP16/INT4 elements.
template <int Rows, bool Fused, bool FixedTP4 = false, int Subgroup = 16>
void launch_wide_linear(sycl::queue& queue, Projection p) {
  constexpr int sg_size = Subgroup;
  constexpr int words_per_load = 128 / sg_size;
  constexpr int accumulators = 128 / sg_size;
  const size_t row_tiles = (int64_t(p.n0) + p.n1 + Rows - 1) / Rows;
  queue.parallel_for<WideLinearKernel<Rows, Fused, FixedTP4, Subgroup>>(
      sycl::nd_range<1>(row_tiles * sg_size, sg_size),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(sg_size)]] {
        const auto sg = item.get_sub_group();
        const int lane = sg.get_local_linear_id();
        const int row_base = item.get_group_linear_id() * Rows;
        // Fused tiles do not straddle the two matrices. Resolve their
        // pointers once rather than in every K iteration and output row.
        const int n0 = FixedTP4 ? 4096 : p.n0;
        const bool second = Fused && row_base >= n0;
        const int n_base = second ? row_base - n0 : row_base;
        const int n_rows =
            FixedTP4 ? (second ? 24 : 4096) : (second ? p.n1 : p.n0);
        const int k = FixedTP4 ? 2560 : p.k;
        const auto* weights = second ? p.weight1 : p.weight0;
        const auto* scales_ptr = second ? p.scale1 : p.scale0;
        auto* output = second ? p.output1 : p.output0;
        const int groups = k / kGroup;
        uint32_t row_scale_pairs[Rows];
        if constexpr (FixedTP4) {
#pragma unroll
          for (int r = 0; r < Rows; ++r) {
            row_scale_pairs[r] =
                lane < 10 && n_base + r < n_rows
                    ? reinterpret_cast<const uint32_t*>(
                          scales_ptr + int64_t(n_base + r) * groups)[lane]
                    : 0;
          }
        }
        float accum[Rows][accumulators] = {};
        for (int block = 0; block < groups; block += 4) {
          namespace sx = sycl::ext::oneapi::experimental;
          constexpr auto props = sx::properties{
              sx::data_placement_striped,
              sx::contiguous_memory,
              sx::full_group,
              sx::alignment<4>};
          sycl::vec<uint64_t, words_per_load> inputs;
          const auto* input_words = reinterpret_cast<const uint64_t*>(p.input);
          auto input_block = sycl::address_space_cast<
                                 sycl::access::address_space::global_space,
                                 sycl::access::decorated::yes>(
                                 input_words + block * (kGroup / 4))
                                 .get_decorated();
          sx::group_load(sg, input_block, inputs, props);
          sycl::vec<uint16_t, words_per_load> raw[Rows];
          float scales[Rows][4];
#pragma unroll
          for (int r = 0; r < Rows; ++r) {
            const int n = n_base + r;
            if (n < n_rows) {
              sx::group_load(
                  sg,
                  sycl::address_space_cast<
                      sycl::access::address_space::global_space,
                      sycl::access::decorated::yes>(
                      reinterpret_cast<const uint16_t*>(
                          weights + int64_t(n) * (k / 2) +
                          block * (kGroup / 2)))
                      .get_decorated(),
                  raw[r],
                  props);
              if constexpr (FixedTP4) {
                const uint32_t s0 =
                    sycl::select_from_group(sg, row_scale_pairs[r], block / 2);
                const uint32_t s1 = sycl::select_from_group(
                    sg, row_scale_pairs[r], block / 2 + 1);
                scales[r][0] =
                    static_cast<float>(sycl::bit_cast<half>(uint16_t(s0)));
                scales[r][1] = static_cast<float>(
                    sycl::bit_cast<half>(uint16_t(s0 >> 16)));
                scales[r][2] =
                    static_cast<float>(sycl::bit_cast<half>(uint16_t(s1)));
                scales[r][3] = static_cast<float>(
                    sycl::bit_cast<half>(uint16_t(s1 >> 16)));
              } else {
#pragma unroll
                for (int g = 0; g < 4; ++g) {
                  scales[r][g] = static_cast<float>(
                      scales_ptr[int64_t(n) * groups + block + g]);
                }
              }
            }
          }
#pragma unroll
          for (int r = 0; r < Rows; ++r) {
            if (n_base + r >= n_rows) {
              continue;
            }
#pragma unroll
            for (int i = 0; i < words_per_load; ++i) {
#pragma unroll
              for (int j = 0; j < 4; ++j) {
                const float x = static_cast<float>(
                    sycl::bit_cast<half>(uint16_t(inputs[i] >> (16 * j))));
                const float weight = (int((raw[r][i] >> (4 * j)) & 15) - 8) *
                                     scales[r][i / (32 / sg_size)];
                accum[r][(i * 4 + j) % accumulators] =
                    sycl::fma(x, weight, accum[r][(i * 4 + j) % accumulators]);
              }
            }
          }
        }
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
          float lane_sum = 0;
#pragma unroll
          for (int i = 0; i < accumulators; ++i) {
            lane_sum += accum[r][i];
          }
          const float value =
              sycl::reduce_over_group(sg, lane_sum, sycl::plus<float>());
          if (lane == 0 && n_base + r < n_rows) {
            output[n_base + r] = static_cast<half>(value);
          }
        }
      });
}

template <int Rows, int Tokens>
class LinearKernel;

// A subgroup cooperates on K, keeps several output rows in registers, and
// reuses each dequantized weight across a small token tile. Reduction occurs
// only after the full K loop, not after each quantization group.
template <int Rows, int Tokens>
void launch_linear(sycl::queue& queue, Projection p) {
  constexpr int sg_size = 16;
  constexpr int local_size = 128;
  const int64_t row_tiles = (int64_t(p.n0) + p.n1 + Rows - 1) / Rows;
  const int64_t token_tiles = (p.m + Tokens - 1) / Tokens;
  const size_t subgroups = row_tiles * token_tiles;
  const size_t global_size = ((subgroups + 7) / 8) * local_size;
  queue.parallel_for<LinearKernel<Rows, Tokens>>(
      sycl::nd_range<1>(global_size, local_size),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(sg_size)]] {
        const auto sg = item.get_sub_group();
        const int lane = sg.get_local_linear_id();
        const int64_t tile = item.get_global_linear_id() / sg_size;
        if (tile >= int64_t(subgroups)) {
          return;
        }
        const int row_base = (tile % row_tiles) * Rows;
        const int token_base = (tile / row_tiles) * Tokens;
        float accum[Rows][Tokens] = {};
        const int groups = p.k / kGroup;
        for (int block = 0; block < groups; block += 4) {
#pragma unroll
          for (int b = 0; b < 4; ++b) {
            const int group = block + b;
            if (group >= groups) {
              break;
            }
            sycl::vec<float, 8> x[Tokens];
#pragma unroll
            for (int t = 0; t < Tokens; ++t) {
              sycl::vec<half, 8> values(half(0));
              if (token_base + t < p.m) {
                values.load(
                    lane,
                    p.input + int64_t(token_base + t) * p.k + group * kGroup);
              }
              x[t] = values.template convert<float>();
            }
#pragma unroll
            for (int r = 0; r < Rows; ++r) {
              const int row = row_base + r;
              if (row >= p.n0 + p.n1) {
                continue;
              }
              const bool second = row >= p.n0;
              const int n = second ? row - p.n0 : row;
              const auto* weights = second ? p.weight1 : p.weight0;
              const auto* scales = second ? p.scale1 : p.scale0;
              const float d =
                  static_cast<float>(scales[int64_t(n) * groups + group]);
              sycl::vec<uint8_t, 4> raw;
              raw.load(
                  lane,
                  weights + int64_t(n) * (p.k / 2) + group * (kGroup / 2));
              const uint32_t packed = raw.as<sycl::vec<uint32_t, 1>>()[0];
#pragma unroll
              for (int i = 0; i < 8; ++i) {
                float w = (int((packed >> (i * 4)) & 15) - 8) * d;
                if constexpr (Tokens > 1) {
                  // Match the small-M DPAS and original M>1 Q4_0 contract:
                  // dequantization rounds once to FP16 before FP32 MMA.
                  const auto rounded = sycl::vec<float, 1>(w)
                      .template convert<half, sycl::rounding_mode::rte>();
                  // The BMG/2025.3 lowering can eliminate a half->float
                  // round trip, even for explicit convert<half, rte>(). A
                  // volatile bit observation retains the required boundary.
                  const volatile uint16_t bits =
                      rounded.template as<sycl::vec<uint16_t, 1>>()[0];
                  w = float(sycl::bit_cast<half>(uint16_t(bits)));
                }
#pragma unroll
                for (int t = 0; t < Tokens; ++t) {
                  accum[r][t] = sycl::fma(x[t][i], w, accum[r][t]);
                }
              }
            }
          }
        }
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
#pragma unroll
          for (int t = 0; t < Tokens; ++t) {
            const float value =
                sycl::reduce_over_group(sg, accum[r][t], sycl::plus<float>());
            if (lane == 0 && row_base + r < p.n0 + p.n1 &&
                token_base + t < p.m) {
              const int row = row_base + r;
              const bool second = row >= p.n0;
              auto* output = second ? p.output1 : p.output0;
              const int n = second ? row - p.n0 : row;
              const int n_cols = second ? p.n1 : p.n0;
              output[int64_t(token_base + t) * n_cols + n] =
                  static_cast<half>(value);
            }
          }
        }
      });
}

Projection projection(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& scale,
    torch::Tensor& output) {
  return {
      reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
      weight.data_ptr<uint8_t>(),
      reinterpret_cast<const half*>(scale.data_ptr<at::Half>()),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      nullptr,
      nullptr,
      nullptr,
      int(input.size(0)),
      int(input.size(1)),
      int(weight.size(0)),
      0};
}

void dispatch_linear(sycl::queue& queue, Projection p) {
  if (p.n1 == 0 && launch_int4_small_m(
                       queue, p.input, p.weight0, p.scale0, p.output0,
                       p.m, p.n0, p.k)) {
    return;
  }
  const bool block_aligned = ((reinterpret_cast<uintptr_t>(p.input) |
                               reinterpret_cast<uintptr_t>(p.weight0) |
                               reinterpret_cast<uintptr_t>(p.weight1)) &
                              3u) == 0;
  if (p.m == 1 && p.k % 512 == 0 && p.n0 + p.n1 >= 128 && block_aligned) {
    if (p.n1 == 0) {
      launch_wide_linear<2, false>(queue, p);
    } else if (
        p.n0 == 4096 && p.n1 == 24 && p.k == 2560 &&
        ((reinterpret_cast<uintptr_t>(p.scale0) |
          reinterpret_cast<uintptr_t>(p.scale1)) &
         3u) == 0) {
      launch_wide_linear<1, true, true, 32>(queue, p);
    } else {
      launch_wide_linear<1, true>(queue, p);
    }
  } else if (p.m == 1) {
    launch_linear<4, 1>(queue, p);
  } else if (p.m <= 2) {
    launch_linear<2, 2>(queue, p);
  } else if (p.m <= 4) {
    launch_linear<2, 4>(queue, p);
  } else {
    launch_linear<1, 8>(queue, p);
  }
}

}  // namespace

void q4_0_quantize(
    const torch::Tensor& input, torch::Tensor& qweight, torch::Tensor& scale) {
  check_input(input);
  TORCH_CHECK(
      input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16,
      "quantization input must be FP16 or BF16");
  check_tensor(qweight, input, at::kInt, "qweight");
  check_tensor(scale, input, at::kHalf, "scale");
  TORCH_CHECK(
      qweight.size(0) == input.size(0) && qweight.size(1) == input.size(1) / 8,
      "qweight must be [N,K/8]");
  TORCH_CHECK(
      scale.size(0) == input.size(0) && scale.size(1) == input.size(1) / kGroup,
      "scale must be [N,K/128]");
  check_no_overlap(qweight, input);
  check_no_overlap(scale, input);
  check_no_overlap(qweight, scale);
  const c10::OptionalDeviceGuard guard(input.device());
  auto stream = c10::xpu::getCurrentXPUStream(input.get_device());
  record_stream({&input, &qweight, &scale}, stream);
  auto* q = reinterpret_cast<uint32_t*>(qweight.data_ptr<int32_t>());
  auto* d = reinterpret_cast<half*>(scale.data_ptr<at::Half>());
  if (input.scalar_type() == at::kHalf) {
    launch_quantize(
        stream.queue(),
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        q,
        d,
        input.size(0),
        input.size(1));
  } else {
    launch_quantize(
        stream.queue(),
        reinterpret_cast<const bfloat16*>(input.data_ptr<at::BFloat16>()),
        q,
        d,
        input.size(0),
        input.size(1));
  }
}

void int4_linear(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& scale,
    torch::Tensor& output) {
  check_input(input);
  TORCH_CHECK(
      input.scalar_type() == at::kHalf, "projection input must be FP16");
  check_projection(input, weight, scale, output);
  const c10::OptionalDeviceGuard guard(input.device());
  auto stream = c10::xpu::getCurrentXPUStream(input.get_device());
  record_stream({&input, &weight, &scale, &output}, stream);
  dispatch_linear(stream.queue(), projection(input, weight, scale, output));
}

void int4_linear_fused2(
    const torch::Tensor& input,
    const torch::Tensor& weight0,
    const torch::Tensor& scale0,
    torch::Tensor& output0,
    const torch::Tensor& weight1,
    const torch::Tensor& scale1,
    torch::Tensor& output1) {
  check_input(input);
  TORCH_CHECK(
      input.scalar_type() == at::kHalf && input.size(0) == 1,
      "fused projection requires M=1 FP16 input");
  check_projection(input, weight0, scale0, output0);
  check_projection(input, weight1, scale1, output1);
  check_no_overlap(output0, weight1);
  check_no_overlap(output0, scale1);
  check_no_overlap(output1, weight0);
  check_no_overlap(output1, scale0);
  check_no_overlap(output0, output1);
  TORCH_CHECK(
      weight0.size(0) + weight1.size(0) <= std::numeric_limits<int>::max(),
      "fused N exceeds the kernel index range");
  const c10::OptionalDeviceGuard guard(input.device());
  auto stream = c10::xpu::getCurrentXPUStream(input.get_device());
  record_stream(
      {&input, &weight0, &scale0, &output0, &weight1, &scale1, &output1},
      stream);
  auto p = projection(input, weight0, scale0, output0);
  p.weight1 = weight1.data_ptr<uint8_t>();
  p.scale1 = reinterpret_cast<const half*>(scale1.data_ptr<at::Half>());
  p.output1 = reinterpret_cast<half*>(output1.data_ptr<at::Half>());
  p.n1 = weight1.size(0);
  dispatch_linear(stream.queue(), p);
}

}  // namespace vllm::qwen38
