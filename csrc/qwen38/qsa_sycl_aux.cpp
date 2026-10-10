// SPDX-License-Identifier: Apache-2.0
#include "qwen38/qsa_sycl_aux.h"
#include "qwen38/qsa_sycl.h"

#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>
#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <type_traits>

namespace vllm::qwen38::qsa_sycl {
namespace {

using half = sycl::half;
constexpr int kQkvDim = 256;
constexpr int kIndexerDim = 128;
constexpr int kRotaryDim = 64;
constexpr int kRotaryHalf = 32;

template <bool Mrope, typename PosT>
class QkvPostprocessKernel;
template <bool Mrope, bool EagerFp16, typename PosT>
class IndexerNormRopeKernel;
class IndexerProjectionInt4Kernel;

void check_xpu(const at::Tensor& tensor, const at::Tensor& anchor,
               at::ScalarType dtype, const char* label) {
  TORCH_CHECK(tensor.device() == anchor.device(), label, " device mismatch");
  TORCH_CHECK(tensor.scalar_type() == dtype, label, " dtype mismatch");
  TORCH_CHECK(!tensor.is_neg() && !tensor.is_conj(),
              label, " cannot be a lazy view");
}

void record_tensors(std::initializer_list<const at::Tensor*> tensors,
                    c10::xpu::XPUStream stream) {
  for (const auto* tensor : tensors) {
    c10::xpu::XPUCachingAllocator::recordStream(
        tensor->storage().data_ptr(), stream);
  }
}

bool may_overlap(const at::Tensor& lhs, const at::Tensor& rhs) {
  if (lhs.numel() == 0 || rhs.numel() == 0) return false;
  auto bounds = [](const at::Tensor& tensor, uintptr_t& first,
                   uintptr_t& last) -> bool {
    constexpr auto max = std::numeric_limits<uintptr_t>::max();
    uintptr_t extent = 0;
    for (int d = 0; d < tensor.dim(); ++d) {
      if (tensor.stride(d) < 0 || tensor.size(d) < 0) return false;
      auto stride = static_cast<uintptr_t>(tensor.stride(d));
      auto count = static_cast<uintptr_t>(std::max<int64_t>(
          tensor.size(d) - 1, 0));
      if (stride && count > (max - extent) / stride) return false;
      extent += count * stride;
    }
    auto bytes_per_element = static_cast<uintptr_t>(tensor.element_size());
    if (extent >= max / bytes_per_element) return false;
    auto bytes = (extent + 1) * bytes_per_element;
    first = reinterpret_cast<uintptr_t>(tensor.data_ptr());
    if (first > max - bytes) return false;
    last = first + bytes;
    return true;
  };
  uintptr_t lf = 0, ll = 0, rf = 0, rl = 0;
  return !bounds(lhs, lf, ll) || !bounds(rhs, rf, rl) ||
      (lf < rl && rf < ll);
}

void validate_output_aliases(std::initializer_list<const at::Tensor*> inputs,
                             std::initializer_list<const at::Tensor*> outputs) {
  for (const auto* output : outputs) {
    for (const auto* input : inputs) {
      TORCH_CHECK(!may_overlap(*output, *input),
                  "QSA auxiliary output aliases an input");
    }
  }
  for (auto first = outputs.begin(); first != outputs.end(); ++first) {
    for (auto second = first + 1; second != outputs.end(); ++second) {
      TORCH_CHECK(!may_overlap(**first, **second),
                  "QSA auxiliary outputs overlap");
    }
  }
}

// A bit-level RNE boundary is necessary for the eager-FP16 v2 route: an
// optimizer must not fold the temporary FP16 multiply back into FP32 FMA.
inline float round_fp16(float value) {
  uint32_t bits = sycl::bit_cast<uint32_t>(value);
  uint32_t sign = bits & 0x80000000u;
  uint32_t magnitude = bits & 0x7fffffffu;
  if (magnitude > 0x7f800000u) {
    return sycl::bit_cast<float>(sign | 0x7fc00000u);
  }
  if (magnitude >= 0x47800000u) {
    return sycl::bit_cast<float>(sign | 0x7f800000u);
  }
  if (magnitude < 0x33000000u) {
    return sycl::bit_cast<float>(sign);
  }
  if (magnitude < 0x38800000u) {
    uint32_t exponent = sycl::clamp(magnitude >> 23, 102u, 112u);
    uint32_t shift = 126u - exponent;
    uint32_t significand = (magnitude & 0x7fffffu) | 0x800000u;
    uint32_t bias = (1u << (shift - 1u)) - 1u;
    uint32_t units = (significand + bias +
                      ((significand >> shift) & 1u)) >> shift;
    float subnormal = static_cast<float>(units) * 0x1p-24f;
    return sycl::bit_cast<float>(
        sycl::bit_cast<uint32_t>(subnormal) | sign);
  }
  uint32_t rounded = (magnitude + 0xfffu + ((magnitude >> 13) & 1u)) &
      0xffffe000u;
  if (rounded >= 0x47800000u) rounded = 0x7f800000u;
  return sycl::bit_cast<float>(rounded | sign);
}

template <typename PosT>
int64_t position_at(const PosT* positions, int row, int axis,
                    int64_t axis_stride, int64_t row_stride) {
  return static_cast<int64_t>(
      positions[static_cast<int64_t>(axis) * axis_stride +
                static_cast<int64_t>(row) * row_stride]);
}

template <bool Mrope, typename PosT>
void launch_qkv(sycl::queue& queue, const half* qkv, half* q_out,
                half* gate_out, half* k_out, half* v_out,
                const half* norm_wq, const half* norm_wk,
                const PosT* positions, const half* cache, int rows,
                int q_heads, int kv_heads, bool gate, int64_t pos_axis_stride,
                int64_t pos_row_stride, int cache_rows) {
  const int total_heads = (gate ? 2 * q_heads : q_heads) + 2 * kv_heads;
  const int packed_width = total_heads * kQkvDim;
  queue.parallel_for<QkvPostprocessKernel<Mrope, PosT>>(
      sycl::nd_range<1>(sycl::range<1>(rows * total_heads * kQkvDim),
                        sycl::range<1>(kQkvDim)),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
#pragma clang fp contract(off) reassociate(off)
        const int group = item.get_group_linear_id();
        const int row = group / total_heads;
        const int segment = group % total_heads;
        const int d = item.get_local_linear_id();
        int kind = 0;  // 0=Q, 1=gate, 2=K, 3=V
        int out_head = 0;
        if (gate && segment < 2 * q_heads) {
          kind = segment & 1;
          out_head = segment / 2;
        } else if (!gate && segment < q_heads) {
          out_head = segment;
        } else if (segment < (gate ? 2 * q_heads : q_heads) + kv_heads) {
          kind = 2;
          out_head = segment - (gate ? 2 * q_heads : q_heads);
        } else {
          kind = 3;
          out_head = segment - (gate ? 2 * q_heads : q_heads) - kv_heads;
        }
        const int64_t source = static_cast<int64_t>(row) * packed_width +
            segment * kQkvDim;
        const float value = static_cast<float>(qkv[source + d]);
        const int64_t destination =
            (static_cast<int64_t>(row) *
             ((kind == 0 || kind == 1) ? q_heads : kv_heads) + out_head) *
                kQkvDim + d;
        if (kind == 1) {
          gate_out[destination] =
              static_cast<half>(1.0f / (1.0f + sycl::exp(-value)));
          return;
        }
        if (kind == 3) {
          v_out[destination] = qkv[source + d];
          return;
        }
        const half* weight = kind == 0 ? norm_wq : norm_wk;
        const float variance = sycl::reduce_over_group(
            item.get_group(), value * value, sycl::plus<float>()) /
            static_cast<float>(kQkvDim);
        const float inverse_rms = sycl::rsqrt(variance + 1e-6f);
        float normalized = (value * (1.0f + static_cast<float>(weight[d]))) *
            inverse_rms;
        if (d < kRotaryDim) {
          const int pair = d % kRotaryHalf;
          const int axis = Mrope ? pair % 3 : 0;
          int64_t position = position_at(
              positions, row, axis, pos_axis_stride, pos_row_stride);
          if (position < 0 || position >= cache_rows) {
            normalized = 0.0f;
          } else {
            const int counterpart = d < kRotaryHalf
                ? d + kRotaryHalf : d - kRotaryHalf;
            const float other = static_cast<float>(qkv[source + counterpart]);
            const float other_normalized =
                (other * (1.0f + static_cast<float>(weight[counterpart]))) *
                inverse_rms;
            const int64_t cache_offset = position * kRotaryDim;
            const float cosine = static_cast<float>(cache[cache_offset + pair]);
            const float sine = static_cast<float>(
                cache[cache_offset + kRotaryHalf + pair]);
            normalized = d < kRotaryHalf
                ? normalized * cosine - other_normalized * sine
                : normalized * cosine + other_normalized * sine;
          }
        }
        if (kind == 0) {
          q_out[destination] = static_cast<half>(normalized);
        } else {
          k_out[destination] = static_cast<half>(normalized);
        }
      });
}

template <bool Mrope, bool EagerFp16, typename PosT>
void launch_indexer(sycl::queue& queue, const half* input, half* output,
                    const half* weight, const PosT* positions,
                    const half* cache, int rows, int heads,
                    int64_t input_row_stride, int64_t position_axis_stride,
                    int64_t position_row_stride, int cache_rows) {
  queue.parallel_for<IndexerNormRopeKernel<Mrope, EagerFp16, PosT>>(
      sycl::nd_range<1>(sycl::range<1>(rows * heads * kIndexerDim),
                        sycl::range<1>(kIndexerDim)),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
#pragma clang fp contract(off) reassociate(off)
        const int group = item.get_group_linear_id();
        const int row = group / heads;
        const int head = group % heads;
        const int d = item.get_local_linear_id();
        const int64_t source = static_cast<int64_t>(row) * input_row_stride +
            head * kIndexerDim;
        const float value = static_cast<float>(input[source + d]);
        const float variance = sycl::reduce_over_group(
            item.get_group(), value * value, sycl::plus<float>()) /
            static_cast<float>(kIndexerDim);
        const float inverse_rms = sycl::rsqrt(variance + 1e-6f);
        const float factor = 1.0f + static_cast<float>(weight[d]);
        float normalized = EagerFp16
            ? round_fp16((value * inverse_rms) * factor)
            : (value * factor) * inverse_rms;
        const int64_t target =
            (static_cast<int64_t>(row) * heads + head) * kIndexerDim + d;
        int64_t position = position_at(
            positions, row, Mrope ? (d % kRotaryHalf) % 3 : 0,
            position_axis_stride, position_row_stride);
        if (position < -cache_rows || position >= cache_rows) {
          output[target] = half(0.0f);
          return;
        }
        if (position < 0) position += cache_rows;
        if (d < kRotaryDim) {
          const int pair = d % kRotaryHalf;
          const int counterpart = d < kRotaryHalf
              ? d + kRotaryHalf : d - kRotaryHalf;
          const float other = static_cast<float>(
              input[source + counterpart]);
          const float other_factor =
              1.0f + static_cast<float>(weight[counterpart]);
          const float other_normalized = EagerFp16
              ? round_fp16((other * inverse_rms) * other_factor)
              : (other * other_factor) * inverse_rms;
          const int64_t cache_offset = position * kRotaryDim;
          const float cosine = static_cast<float>(cache[cache_offset + pair]);
          const float sine = static_cast<float>(
              cache[cache_offset + kRotaryHalf + pair]);
          if constexpr (EagerFp16) {
            normalized = d < kRotaryHalf
                ? round_fp16(round_fp16(normalized * cosine) -
                             round_fp16(other_normalized * sine))
                : round_fp16(round_fp16(normalized * cosine) +
                             round_fp16(other_normalized * sine));
          } else {
            normalized = d < kRotaryHalf
                ? normalized * cosine - other_normalized * sine
                : normalized * cosine + other_normalized * sine;
          }
        }
        output[target] = static_cast<half>(normalized);
      });
}

void launch_projection_int4(sycl::queue& queue, const half* x,
                            const uint8_t* packed, const half* scales,
                            half* out, int rows) {
  queue.parallel_for<IndexerProjectionInt4Kernel>(
      sycl::nd_range<1>(sycl::range<1>(rows * 640 * 128),
                        sycl::range<1>(128)),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
        const int group = item.get_group_linear_id();
        const int row = group / 640;
        const int column = group % 640;
        const int lane = item.get_local_linear_id();
        float sum = 0.0f;
#pragma unroll
        for (int scale_group = 0; scale_group < 20; ++scale_group) {
          const int k = scale_group * 128 + lane;
          const uint8_t byte = packed[column * 1280 + k / 2];
          const int quantized = lane & 1 ? (byte >> 4) & 15 : byte & 15;
          const float scale = static_cast<float>(
              scales[column * 20 + scale_group]);
          sum += static_cast<float>(x[row * 2560 + k]) *
              static_cast<float>(quantized - 8) * scale;
        }
        const float value = sycl::reduce_over_group(
            item.get_group(), sum, sycl::plus<float>());
        if (lane == 0) out[row * 640 + column] = static_cast<half>(value);
      });
}

}  // namespace

at::Tensor qkv_postprocess_v1(
    const at::Tensor& qkv, at::Tensor q_out, at::Tensor gate_out,
    at::Tensor k_out, at::Tensor v_out, const at::Tensor& norm_wq,
    const at::Tensor& norm_wk, const at::Tensor& positions,
    const at::Tensor& cos_sin_cache, int64_t q_heads, int64_t kv_heads,
    bool attn_output_gate, bool mrope, bool positions_bounds_proven) {
  TORCH_CHECK(qkv.is_xpu(), "QKV input must be XPU");
  check_xpu(qkv, qkv, at::kHalf, "QKV input");
  TORCH_CHECK(positions_bounds_proven,
              "QKV postprocess requires producer position-bound proof");
  const std::array<const at::Tensor*, 7> fp16_tensors = {
      &q_out, &gate_out, &k_out, &v_out, &norm_wq, &norm_wk,
      &cos_sin_cache};
  for (const auto* t : fp16_tensors) {
    check_xpu(*t, qkv, at::kHalf, "QKV tensor");
  }
  TORCH_CHECK(positions.device() == qkv.device() &&
                  (positions.scalar_type() == at::kInt ||
                   positions.scalar_type() == at::kLong) &&
                  !positions.is_neg() && !positions.is_conj(),
              "QKV positions must be XPU int32/int64");
  TORCH_CHECK(q_heads >= 1 && q_heads <= 6 && kv_heads == 1 &&
                  (!mrope || attn_output_gate),
              "unsupported QKV Q/K/gate geometry");
  TORCH_CHECK(qkv.scalar_type() == at::kHalf && qkv.dim() == 2 &&
                  qkv.size(0) >= 1 && qkv.size(0) <= 4096 &&
                  qkv.size(1) ==
                      (attn_output_gate ? 2 * q_heads + 2 * kv_heads
                                        : q_heads + 2 * kv_heads) * kQkvDim &&
                  qkv.is_contiguous(),
              "QKV input shape/strides are invalid");
  const int64_t rows = qkv.size(0);
  TORCH_CHECK(q_out.sizes() == at::IntArrayRef({rows, q_heads * kQkvDim}) &&
                  q_out.is_contiguous() &&
                  gate_out.sizes() == q_out.sizes() &&
                  gate_out.is_contiguous() &&
                  k_out.sizes() == at::IntArrayRef({rows, kv_heads * kQkvDim}) &&
                  k_out.is_contiguous() &&
                  v_out.sizes() == k_out.sizes() && v_out.is_contiguous(),
              "QKV caller outputs have wrong shape/strides");
  TORCH_CHECK(norm_wq.sizes() == at::IntArrayRef({kQkvDim}) &&
                  norm_wq.is_contiguous() &&
                  norm_wk.sizes() == norm_wq.sizes() &&
                  norm_wk.is_contiguous() &&
                  cos_sin_cache.dim() == 2 && cos_sin_cache.size(0) > 0 &&
                  cos_sin_cache.size(1) == kRotaryDim &&
                  cos_sin_cache.is_contiguous(),
              "QKV norm/cache shape/strides are invalid");
  if (mrope) {
    TORCH_CHECK(positions.sizes() == at::IntArrayRef({3, rows}) &&
                    positions.stride(1) == 1 && positions.stride(0) >= rows,
                "QKV MRoPE positions must be [3,M] row-strided");
  } else {
    TORCH_CHECK(positions.sizes() == at::IntArrayRef({rows}) &&
                    positions.is_contiguous(),
                "QKV plain positions must be contiguous [M]");
  }
  validate_output_aliases({&qkv, &norm_wq, &norm_wk, &positions,
                           &cos_sin_cache},
                          {&q_out, &gate_out, &k_out, &v_out});
  c10::OptionalDeviceGuard guard(qkv.device());
  auto stream = c10::xpu::getCurrentXPUStream(qkv.get_device());
  auto& queue = stream.queue();
  TORCH_CHECK(queue.is_in_order(), "QKV requires an in-order current stream");
  record_tensors({&qkv, &q_out, &gate_out, &k_out, &v_out, &norm_wq,
                  &norm_wk, &positions, &cos_sin_cache}, stream);
  const auto* packed = static_cast<const half*>(qkv.data_ptr());
  auto* q = static_cast<half*>(q_out.data_ptr());
  auto* g = static_cast<half*>(gate_out.data_ptr());
  auto* k = static_cast<half*>(k_out.data_ptr());
  auto* v = static_cast<half*>(v_out.data_ptr());
  const auto* wq = static_cast<const half*>(norm_wq.data_ptr());
  const auto* wk = static_cast<const half*>(norm_wk.data_ptr());
  const auto* cache = static_cast<const half*>(cos_sin_cache.data_ptr());
  auto launch = [&](auto mrope_tag, auto pos_tag) {
    using PosT = typename decltype(pos_tag)::type;
    launch_qkv<decltype(mrope_tag)::value, PosT>(
        queue, packed, q, g, k, v, wq, wk,
        positions.data_ptr<PosT>(), cache, rows, q_heads, kv_heads,
        attn_output_gate, mrope ? positions.stride(0) : 0,
        mrope ? positions.stride(1) : 1, cos_sin_cache.size(0));
  };
  if (positions.scalar_type() == at::kInt) {
    if (mrope) launch(std::true_type{}, std::type_identity<int32_t>{});
    else launch(std::false_type{}, std::type_identity<int32_t>{});
  } else if (mrope) {
    launch(std::true_type{}, std::type_identity<int64_t>{});
  } else {
    launch(std::false_type{}, std::type_identity<int64_t>{});
  }
  return q_out;
}

at::Tensor indexer_norm_rope_v2(
    const at::Tensor& input, at::Tensor output, const at::Tensor& weight,
    const at::Tensor& positions, const at::Tensor& cos_sin_cache,
    bool mrope, bool eager_fp16, bool positions_bounds_proven) {
  TORCH_CHECK(input.is_xpu(), "indexer input must be XPU");
  check_xpu(input, input, at::kHalf, "indexer input");
  // Legacy v1 supplies producer proof. Legacy v2 does not, but its eager
  // FP16 path guards every cache read in the device kernel, so it needs no
  // fabricated host proof. The non-eager path remains proof gated.
  TORCH_CHECK(positions_bounds_proven || eager_fp16,
              "indexer norm+RoPE requires producer position-bound proof");
  check_xpu(output, input, at::kHalf, "indexer output");
  check_xpu(weight, input, at::kHalf, "indexer weight");
  check_xpu(cos_sin_cache, input, at::kHalf, "indexer cache");
  TORCH_CHECK(positions.device() == input.device() &&
                  (positions.scalar_type() == at::kInt ||
                   positions.scalar_type() == at::kLong) &&
                  !positions.is_neg() && !positions.is_conj(),
              "indexer positions must be XPU int32/int64");
  TORCH_CHECK(input.scalar_type() == at::kHalf && input.dim() == 3 &&
                  input.size(0) >= 1 && input.size(0) <= 4096 &&
                  (input.size(1) == 1 || input.size(1) == 4) &&
                  input.size(2) == kIndexerDim &&
                  input.stride(2) == 1 && input.stride(1) == kIndexerDim &&
                  input.stride(0) >= input.size(1) * kIndexerDim &&
                  output.sizes() == input.sizes() && output.is_contiguous(),
              "indexer input/output shape or strides are invalid");
  const int64_t rows = input.size(0);
  TORCH_CHECK(weight.sizes() == at::IntArrayRef({kIndexerDim}) &&
                  weight.is_contiguous() &&
                  cos_sin_cache.dim() == 2 && cos_sin_cache.size(0) > 0 &&
                  cos_sin_cache.size(1) == kRotaryDim &&
                  cos_sin_cache.is_contiguous(),
              "indexer weight/cache shape or strides are invalid");
  if (mrope) {
    TORCH_CHECK(positions.sizes() == at::IntArrayRef({3, rows}) &&
                    positions.stride(0) > 0 && positions.stride(1) > 0,
                "indexer MRoPE positions must be [3,M] positive-strided");
  } else {
    TORCH_CHECK(positions.sizes() == at::IntArrayRef({rows}) &&
                    positions.is_contiguous(),
                "indexer plain positions must be contiguous [M]");
  }
  validate_output_aliases({&input, &weight, &positions, &cos_sin_cache},
                          {&output});
  c10::OptionalDeviceGuard guard(input.device());
  auto stream = c10::xpu::getCurrentXPUStream(input.get_device());
  auto& queue = stream.queue();
  TORCH_CHECK(queue.is_in_order(),
              "indexer norm+RoPE requires an in-order current stream");
  record_tensors({&input, &output, &weight, &positions, &cos_sin_cache},
                 stream);
  auto launch = [&](auto mrope_tag, auto eager_tag, auto pos_tag) {
    using PosT = typename decltype(pos_tag)::type;
    launch_indexer<decltype(mrope_tag)::value,
                   decltype(eager_tag)::value, PosT>(
        queue, static_cast<const half*>(input.data_ptr()),
        static_cast<half*>(output.data_ptr()),
        static_cast<const half*>(weight.data_ptr()),
        positions.data_ptr<PosT>(),
        static_cast<const half*>(cos_sin_cache.data_ptr()), rows,
        input.size(1), input.stride(0),
        mrope ? positions.stride(0) : 0,
        mrope ? positions.stride(1) : 1, cos_sin_cache.size(0));
  };
  if (positions.scalar_type() == at::kInt) {
    if (mrope && eager_fp16)
      launch(std::true_type{}, std::true_type{}, std::type_identity<int32_t>{});
    else if (mrope)
      launch(std::true_type{}, std::false_type{}, std::type_identity<int32_t>{});
    else if (eager_fp16)
      launch(std::false_type{}, std::true_type{}, std::type_identity<int32_t>{});
    else
      launch(std::false_type{}, std::false_type{}, std::type_identity<int32_t>{});
  } else {
    if (mrope && eager_fp16)
      launch(std::true_type{}, std::true_type{}, std::type_identity<int64_t>{});
    else if (mrope)
      launch(std::true_type{}, std::false_type{}, std::type_identity<int64_t>{});
    else if (eager_fp16)
      launch(std::false_type{}, std::true_type{}, std::type_identity<int64_t>{});
    else
      launch(std::false_type{}, std::false_type{}, std::type_identity<int64_t>{});
  }
  return output;
}

at::Tensor indexer_projection_int4_v1(
    const at::Tensor& input, const at::Tensor& packed_weight,
    const at::Tensor& group_scales, at::Tensor output) {
  TORCH_CHECK(input.is_xpu(), "indexer projection input must be XPU");
  check_xpu(input, input, at::kHalf, "projection input");
  check_xpu(packed_weight, input, at::kByte, "projection weight");
  check_xpu(group_scales, input, at::kHalf, "projection scale");
  check_xpu(output, input, at::kHalf, "projection output");
  TORCH_CHECK(input.dim() == 2 && input.size(0) >= 1 &&
                  input.size(0) <= 8 && input.size(1) == 2560 &&
                  input.is_contiguous() &&
                  packed_weight.sizes() == at::IntArrayRef({640, 1280}) &&
                  packed_weight.is_contiguous() &&
                  group_scales.sizes() == at::IntArrayRef({640, 20}) &&
                  group_scales.is_contiguous() &&
                  output.sizes() == at::IntArrayRef({input.size(0), 640}) &&
                  output.is_contiguous(),
              "QSA INT4 projection requires [M,2560] x [640,1280]"
              " with [640,20] scales and [M,640] output");
  validate_output_aliases({&input, &packed_weight, &group_scales}, {&output});
  c10::OptionalDeviceGuard guard(input.device());
  auto stream = c10::xpu::getCurrentXPUStream(input.get_device());
  auto& queue = stream.queue();
  TORCH_CHECK(queue.is_in_order(),
              "indexer projection requires an in-order current stream");
  record_tensors({&input, &packed_weight, &group_scales, &output}, stream);
  const auto* x = static_cast<const half*>(input.data_ptr());
  const auto* packed = packed_weight.data_ptr<uint8_t>();
  const auto* scales = static_cast<const half*>(group_scales.data_ptr());
  auto* out = static_cast<half*>(output.data_ptr());
  const int rows = input.size(0);
  launch_projection_int4(queue, x, packed, scales, out, rows);
  return output;
}

at::Tensor q_norm_rope_select_v1(
    const at::Tensor& projected_q, const at::Tensor& norm_weight,
    const at::Tensor& positions, const at::Tensor& cos_sin_cache,
    const at::Tensor& compressed_key_cache, const at::Tensor& page_table,
    const at::Tensor& token_to_req, const at::Tensor& query_positions,
    const at::Tensor& sequence_lengths, at::Tensor q_output, at::Tensor out,
    at::Tensor scores_a, at::Tensor indices_a, at::Tensor scores_b,
    at::Tensor indices_b, int64_t compressed_page_size, int64_t max_seq_len,
    bool mrope, bool eager_fp16, bool positions_bounds_proven) {
  // Validate the entire selection stage before indexer_norm_rope_v2 submits
  // anything. The core selection implementation repeats its own validation.
  TORCH_CHECK(projected_q.is_xpu() && projected_q.scalar_type() == at::kHalf &&
                  projected_q.dim() == 3 && projected_q.size(0) >= 1 &&
                  projected_q.size(0) <= 4096 && projected_q.size(1) == 4 &&
                  projected_q.size(2) == 128,
              "Q norm+RoPE+select requires projected [M,4,128]");
  const int64_t rows = projected_q.size(0);
  check_xpu(compressed_key_cache, projected_q, at::kHalf,
            "compressed key cache");
  check_xpu(page_table, projected_q, at::kInt, "selection page table");
  check_xpu(token_to_req, projected_q, at::kInt, "selection requests");
  check_xpu(query_positions, projected_q, at::kLong,
            "selection query positions");
  check_xpu(sequence_lengths, projected_q, at::kInt,
            "selection sequence lengths");
  check_xpu(q_output, projected_q, at::kHalf, "normalized Q output");
  check_xpu(out, projected_q, at::kInt, "selected indices output");
  for (const auto* tensor : {&scores_a, &scores_b}) {
    check_xpu(*tensor, projected_q, at::kFloat, "selection scores");
  }
  for (const auto* tensor : {&indices_a, &indices_b}) {
    check_xpu(*tensor, projected_q, at::kInt, "selection indices");
  }
  TORCH_CHECK(compressed_page_size == 64 || compressed_page_size == 128,
              "selection compressed page size must be 64 or 128");
  TORCH_CHECK(compressed_key_cache.dim() == 4 &&
                  compressed_key_cache.size(0) > 0 &&
                  compressed_key_cache.size(1) == compressed_page_size &&
                  compressed_key_cache.size(2) == 1 &&
                  compressed_key_cache.size(3) == 128 &&
                  compressed_key_cache.stride(3) == 1 &&
                  compressed_key_cache.stride(1) >= 128 &&
                  compressed_key_cache.stride(0) > 0,
              "selection compressed cache layout is invalid");
  TORCH_CHECK(page_table.dim() == 2 && page_table.size(0) > 0 &&
                  page_table.size(1) > 0 && page_table.is_contiguous() &&
                  token_to_req.sizes() == at::IntArrayRef({rows}) &&
                  token_to_req.is_contiguous() &&
                  query_positions.sizes() == at::IntArrayRef({rows}) &&
                  query_positions.is_contiguous() &&
                  sequence_lengths.sizes() ==
                      at::IntArrayRef({page_table.size(0)}) &&
                  sequence_lengths.is_contiguous(),
              "selection metadata layout is invalid");
  TORCH_CHECK(max_seq_len > 0 && max_seq_len <= 256000 &&
                  max_seq_len <= page_table.size(1) *
                      compressed_page_size * 4,
              "selection max_seq_len exceeds page-table capacity");
  TORCH_CHECK(q_output.sizes() == projected_q.sizes() &&
                  q_output.is_contiguous() &&
                  out.sizes() == at::IntArrayRef({rows, 2051}) &&
                  out.is_contiguous(),
              "selection caller output layout is invalid");
  const int64_t scratch_rows = scores_a.dim() == 3 ? scores_a.size(0) : -1;
  TORCH_CHECK(valid_selection_scratch_rows(rows, max_seq_len, scratch_rows),
              "selection scratch rows must be min(M,32), or 128 only for "
              "M>=128 and max_seq_len>=4096");
  const std::array<int64_t, 3> workspace_shape = {
      scratch_rows, 32, 512};
  for (const auto* workspace : {&scores_a, &indices_a, &scores_b,
                                &indices_b}) {
    TORCH_CHECK(workspace->sizes() == at::IntArrayRef(workspace_shape) &&
                    workspace->is_contiguous(),
                "selection workspace must be caller-owned "
                "[scratch_rows,32,512]");
  }
  validate_output_aliases(
      {&projected_q, &norm_weight, &positions, &cos_sin_cache,
       &compressed_key_cache, &page_table, &token_to_req,
       &query_positions, &sequence_lengths},
      {&q_output, &out, &scores_a, &indices_a, &scores_b, &indices_b});
  c10::OptionalDeviceGuard guard(projected_q.device());
  auto stream = c10::xpu::getCurrentXPUStream(projected_q.get_device());
  TORCH_CHECK(stream.queue().is_in_order(),
              "Q norm+RoPE+select requires an in-order current stream");
  indexer_norm_rope_v2(projected_q, q_output, norm_weight, positions,
                       cos_sin_cache, mrope, eager_fp16,
                       positions_bounds_proven);
  return select_paged_tokens_v2(
      q_output, compressed_key_cache, page_table, token_to_req,
      query_positions, sequence_lengths, compressed_page_size,
      max_seq_len, out, scores_a, indices_a, scores_b, indices_b);
}

void bind_qsa_sycl_aux(pybind11::module_& module) {
  module.def("qsa_sycl_qkv_postprocess_v1", &qkv_postprocess_v1);
  module.def("qsa_sycl_indexer_norm_rope_v2", &indexer_norm_rope_v2);
  module.def("qsa_sycl_indexer_projection_int4_v1",
             &indexer_projection_int4_v1);
  module.def("qsa_sycl_q_norm_rope_select_v1", &q_norm_rope_select_v1);
}

}  // namespace vllm::qwen38::qsa_sycl
