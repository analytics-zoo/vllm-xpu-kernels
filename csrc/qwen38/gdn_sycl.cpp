// SPDX-License-Identifier: Apache-2.0
#include "qwen38/gdn_sycl.h"

#include <ATen/MemoryOverlap.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUGraphsC10Utils.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/ext/oneapi/experimental/group_load_store.hpp>
#include <sycl/ext/oneapi/experimental/root_group.hpp>
#include <sycl/sycl.hpp>

#ifdef QWEN38_GDN_STANDALONE_TEST
  #include <torch/library.h>
#endif

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>

namespace vllm::qwen38 {
namespace {

using half = sycl::half;
constexpr int kDim = 128;
constexpr int kSubgroup = 16;
constexpr int kValuesPerLane = kDim / kSubgroup;
constexpr int kRowsPerGroup = 8;
constexpr int kConvGroup = 256;

struct Shape {
  int m, h, hv, dim;
  int64_t qkvz_stride, ba_stride, conv_stride, ssm_stride;
  int conv_slots, ssm_slots, conv_len;
  float scale;
};

void check_data(
    const torch::Tensor& t,
    const torch::Tensor& ref,
    at::ScalarType dtype,
    const char* name) {
  TORCH_CHECK(t.device() == ref.device(), name, ": XPU device mismatch");
  TORCH_CHECK(t.scalar_type() == dtype, name, ": wrong dtype");
  TORCH_CHECK(!t.is_neg() && !t.is_conj(), name, ": lazy view unsupported");
}

void check_indices(
    const torch::Tensor& t,
    const torch::Tensor& ref,
    int64_t count,
    const char* name) {
  check_data(t, ref, at::kInt, name);
  TORCH_CHECK(
      t.is_contiguous() && t.numel() == count,
      name,
      ": expected contiguous int32 with ",
      count,
      " entries");
}

struct MemoryRows {
  uintptr_t begin, end, width, pitch;
  int64_t rows;
};

MemoryRows memory_rows(const torch::Tensor& t) {
  const auto begin = reinterpret_cast<uintptr_t>(t.const_data_ptr());
  if (t.numel() == 0) return {begin, begin, 0, 0, 0};
  const auto item = uintptr_t(t.element_size());
  if (t.is_contiguous())
    return {
        begin,
        begin + uintptr_t(t.numel()) * item,
        uintptr_t(t.numel()) * item,
        uintptr_t(t.numel()) * item,
        1};
  // Every accepted non-dense GDN layout has a dense inner row. Conv and SSM
  // live in disjoint regions of each shared KV page, not distinct storages.
  int64_t inner = 1;
  for (int d = int(t.dim()) - 1; d >= 1; --d) {
    TORCH_CHECK(
        t.size(d) == 1 || t.stride(d) == inner,
        "GDN alias check requires dense inner rows");
    inner *= t.size(d);
  }
  TORCH_CHECK(t.stride(0) >= inner, "GDN alias check requires positive pitch");
  const auto width = uintptr_t(inner) * item;
  const auto pitch = uintptr_t(t.stride(0)) * item;
  const auto rows = t.size(0);
  constexpr auto max = std::numeric_limits<uintptr_t>::max();
  TORCH_CHECK(
      uintptr_t(rows - 1) <= (max - width) / pitch,
      "GDN address span overflows");
  const auto span = uintptr_t(rows - 1) * pitch + width;
  TORCH_CHECK(begin <= max - span, "GDN address range overflows");
  return {begin, begin + span, width, pitch, rows};
}

bool physical_overlap(MemoryRows left, MemoryRows right) {
  if (left.rows == 0 || right.rows == 0 || left.end <= right.begin ||
      right.end <= left.begin)
    return false;
  if (left.pitch == right.pitch) {
    // Common KV-page pitch: decide by offset/width in O(1), not by iterating
    // thousands of cache pages on every decode token.
    if (left.begin > right.begin) std::swap(left, right);
    const auto delta = right.begin - left.begin;
    const auto row = delta / left.pitch;
    const auto offset = delta % left.pitch;
    return (row < uintptr_t(left.rows) && offset < left.width) ||
           (row + 1 < uintptr_t(left.rows) &&
            right.width > left.pitch - offset);
  }
  int64_t i = 0, j = 0;
  while (i < left.rows && j < right.rows) {
    const auto x = left.begin + uintptr_t(i) * left.pitch;
    const auto y = right.begin + uintptr_t(j) * right.pitch;
    if (x <= y && y - x >= left.width)
      i += int64_t((y - x - left.width) / left.pitch + 1);
    else if (y <= x && x - y >= right.width)
      j += int64_t((x - y - right.width) / right.pitch + 1);
    else
      return true;
  }
  return false;
}

bool physical_overlap(const torch::Tensor& a, const torch::Tensor& b) {
  return physical_overlap(memory_rows(a), memory_rows(b));
}

void check_no_overlap(const torch::Tensor& a, const torch::Tensor& b) {
  at::assert_no_overlap(a, b);
  TORCH_CHECK(!physical_overlap(a, b), "GDN tensors have overlapping memory");
}

void check_no_cross_alias(
    std::initializer_list<const torch::Tensor*> writes,
    std::initializer_list<const torch::Tensor*> reads) {
  for (auto* a : writes) {
    for (auto* b : reads)
      check_no_overlap(*a, *b);
    for (auto* b : writes) {
      if (a != b) check_no_overlap(*a, *b);
    }
  }
}

void record_stream(
    std::initializer_list<const torch::Tensor*> tensors,
    c10::xpu::XPUStream stream) {
  for (auto* t : tensors)
    c10::xpu::XPUCachingAllocator::recordStream(
        t->storage().data_ptr(), stream);
}

Shape check_core(
    const torch::Tensor& qkvz,
    const torch::Tensor& conv,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& a_log,
    const torch::Tensor& dt_bias,
    const torch::Tensor& ba,
    const torch::Tensor& ssm,
    const torch::Tensor& out,
    const torch::Tensor& z,
    bool speculative,
    int64_t m,
    double scale) {
  TORCH_CHECK(qkvz.is_xpu(), "GDN requires XPU tensors");
  TORCH_CHECK(std::isfinite(scale) && scale > 0.0, "invalid GDN scale");
  const float scale_fp32 = static_cast<float>(scale);
  TORCH_CHECK(
      std::isfinite(scale_fp32) && scale_fp32 > 0.0f,
      "invalid GDN scale after FP32 conversion");
  TORCH_CHECK(
      m > 0 && m <= std::numeric_limits<int>::max(), "invalid token count");
  for (auto* t : {&qkvz, &conv, &weight, &bias, &dt_bias, &ba, &ssm, &out, &z})
    check_data(*t, qkvz, at::kHalf, "GDN data tensor");
  check_data(a_log, qkvz, speculative ? at::kFloat : at::kHalf, "A_log");
  TORCH_CHECK(
      qkvz.dim() == 2 && qkvz.size(0) == m && qkvz.stride(1) == 1 &&
          qkvz.stride(0) >= qkvz.size(1),
      "invalid qkvz layout");
  TORCH_CHECK(
      ssm.dim() == 4 && ssm.size(0) > 0 && ssm.size(2) == kDim &&
          ssm.size(3) == kDim && ssm.stride(3) == 1 && ssm.stride(2) == kDim &&
          ssm.stride(1) == kDim * kDim &&
          ssm.stride(0) >= ssm.size(1) * kDim * kDim,
      "invalid SSM state layout");
  const int hv = static_cast<int>(ssm.size(1));
  // Qwen3.8 config: 16 key heads and 48 value heads. TP4=4/12;
  // TP8=2/6. The old 'HV=8' comment describes another model.
  const int h = hv / 3;
  TORCH_CHECK(
      (h == 4 && hv == 12) || (h == 2 && hv == 6),
      "unsupported Qwen3.8 GDN geometry: H=",
      h,
      " HV=",
      hv);
  const int dim = (2 * h + hv) * kDim;
  TORCH_CHECK(qkvz.size(1) == dim + hv * kDim, "invalid [q|k|v|z] width");
  TORCH_CHECK(
      ba.dim() == 2 && ba.size(0) == m && ba.size(1) == 2 * hv &&
          ba.stride(1) == 1 && ba.stride(0) >= 2 * hv,
      "invalid [b|a] layout");
  TORCH_CHECK(
      conv.dim() == 3 && conv.size(0) > 0 && conv.size(2) == dim &&
          conv.stride(2) == 1 && conv.stride(1) == dim &&
          conv.stride(0) >= conv.size(1) * dim,
      "invalid convolution state layout");
  TORCH_CHECK(
      conv.size(0) <= std::numeric_limits<int>::max() &&
          ssm.size(0) <= std::numeric_limits<int>::max(),
      "state slot count exceeds kernel index range");
  TORCH_CHECK(
      weight.is_contiguous() && weight.numel() == int64_t(dim) * 4 &&
          bias.is_contiguous() && bias.numel() == dim &&
          dt_bias.is_contiguous() && dt_bias.numel() == hv &&
          a_log.is_contiguous() && a_log.numel() == hv,
      "invalid GDN parameter dimensions");
  for (auto* t : {&out, &z}) {
    TORCH_CHECK(
        t->is_contiguous() && t->dim() == 3 && t->size(0) == m &&
            t->size(1) == hv && t->size(2) == kDim,
        "invalid GDN output layout");
  }
  check_no_cross_alias(
      {&conv, &ssm, &out, &z}, {&qkvz, &weight, &bias, &a_log, &dt_bias, &ba});
  return {
      static_cast<int>(m),
      h,
      hv,
      dim,
      qkvz.stride(0),
      ba.stride(0),
      conv.stride(0),
      ssm.stride(0),
      static_cast<int>(conv.size(0)),
      static_cast<int>(ssm.size(0)),
      static_cast<int>(conv.size(1)),
      scale_fp32};
}

inline float sigmoid(float x) { return 1.0f / (1.0f + sycl::exp(-x)); }
inline float softplus(float x) {
  return x > 20.0f ? x : sycl::log(1.0f + sycl::exp(x));
}

inline float round_spec_conv_product(float product) {
  // IGC may promote float(half(product)) back to FP32. The volatile bit
  // boundary is needed only for speculative FP16 products accumulated in
  // FP32; recurrent state and normal decode intentionally stay FP32.
  const volatile uint16_t bits = sycl::bit_cast<uint16_t>(half(product));
  return float(sycl::bit_cast<half>(uint16_t(bits)));
}

// A feature has exactly one writer. Shifting it in the same work-item after
// loading all three history values cannot race with another work-group.
template <bool Spec>
class GdnConvKernel;

class GdnConvPackedParallelKernel;

template <bool Spec>
sycl::event launch_conv(
    sycl::queue& queue,
    const half* input,
    half* conv,
    const half* weight,
    const half* bias,
    const int* indices,
    const int* other_indices,
    const int* tokens,
    const int* accepted,
    half* qkv,
    half* z,
    Shape s,
    int sequences,
    int spec_tokens) {
  const int64_t features = int64_t(sequences) * s.dim;
  const size_t global = (features + kConvGroup - 1) / kConvGroup * kConvGroup;
  return queue.parallel_for<GdnConvKernel<Spec>>(
      sycl::nd_range<1>(global, kConvGroup), [=](sycl::nd_item<1> item) {
        const int64_t id = item.get_global_linear_id();
        if (id >= features) return;
        const int seq = id / s.dim;
        const int f = id % s.dim;
        const int row = seq * spec_tokens;
        int idx = indices[row];
        int64_t initial_col = 0;
        if constexpr (Spec) {
          initial_col = int64_t(accepted[seq]) - 1;
          if (initial_col < 0 || initial_col >= spec_tokens)
            idx = -1;
          else
            idx = indices[row + (s.conv_len == 3 ? initial_col : 0)];
        }
        const bool other_valid = Spec || (other_indices[row] >= 0 &&
                                          other_indices[row] < s.ssm_slots);
        const bool valid =
            other_valid && idx >= (Spec ? 1 : 0) && idx < s.conv_slots;
        const int initial_offset = Spec && s.conv_len != 3 ? initial_col : 0;
        half* base = valid ? conv + int64_t(idx) * s.conv_stride : nullptr;
        float a = valid ? float(base[(initial_offset + 0) * s.dim + f]) : 0.f;
        float b = valid ? float(base[(initial_offset + 1) * s.dim + f]) : 0.f;
        float c = valid ? float(base[(initial_offset + 2) * s.dim + f]) : 0.f;
        const float w0 = float(weight[4 * f + 0]);
        const float w1 = float(weight[4 * f + 1]);
        const float w2 = float(weight[4 * f + 2]);
        const float w3 = float(weight[4 * f + 3]);
        if constexpr (Spec) {
          if (valid && s.conv_len != 3) {
            // At this point a/b/c are in registers. The forward copy is
            // equivalent to Triton's retained-history shift for any accepted.
            base[0 * s.dim + f] = half(b);
            base[1 * s.dim + f] = half(c);
          }
        }
        for (int t = 0; t < spec_tokens; ++t) {
          const int global_t = Spec ? tokens[row + t] : seq;
          const bool token_valid = global_t >= 0 && global_t < s.m;
          const float x =
              valid && token_valid
                  ? float(input[int64_t(global_t) * s.qkvz_stride + f])
                  : 0.f;
          float value = 0.f;
          if (valid && token_valid) {
            if constexpr (Spec) {
              // v2 follows the Triton FP16 multiply / FP32 accumulate path.
              value = float(bias[f]) + round_spec_conv_product(a * w0) +
                      round_spec_conv_product(b * w1) +
                      round_spec_conv_product(c * w2) +
                      round_spec_conv_product(x * w3);
            } else {
              value = a * w0 + b * w1 + c * w2 + x * w3 + float(bias[f]);
            }
            value /= 1.0f + sycl::exp(-value);
          }
          qkv[int64_t(row + t) * s.dim + f] = half(value);
          const bool save_valid = !Spec || (indices[row + t] > 0 &&
                                            indices[row + t] < s.conv_slots);
          if (token_valid && f >= 2 * s.h * kDim) {
            const int vf = f - 2 * s.h * kDim;
            z[int64_t(global_t) * s.hv * kDim + vf] =
                valid && save_valid
                    ? input[int64_t(global_t) * s.qkvz_stride + s.dim + vf]
                    : half(0.f);
          }
          if (valid && token_valid) {
            if constexpr (Spec) {
              const int save_idx = s.conv_len == 3 ? indices[row + t] : idx;
              if (save_idx > 0 && save_idx < s.conv_slots) {
                half* dest = conv + int64_t(save_idx) * s.conv_stride;
                if (s.conv_len == 3) {
                  dest[0 * s.dim + f] = half(b);
                  dest[1 * s.dim + f] = half(c);
                }
                dest[(s.conv_len == 3 ? 2 : 2 + t) * s.dim + f] = half(x);
              }
            } else {
              base[0 * s.dim + f] = half(b);
              base[1 * s.dim + f] = half(c);
              base[2 * s.dim + f] = half(x);
            }
          }
          a = b;
          b = c;
          c = x;
        }
      });
}

// Packed speculative checkpoints make every convolution token independent:
// t's three inputs are initial history and/or already-present input tokens.
// All packed-state writes are deferred to the recurrent kernel, which depends
// on this event. Even a checkpoint row may contain accepted history, so
// writing it here would race with another token's read.
sycl::event launch_conv_packed_parallel(
    sycl::queue& queue,
    const half* input,
    half* conv,
    const half* weight,
    const half* bias,
    const int* indices,
    const int* tokens,
    const int* accepted,
    half* qkv,
    half* z,
    Shape s,
    int sequences,
    int spec_tokens) {
  const int64_t elements = int64_t(sequences) * spec_tokens * s.dim;
  const size_t global = (elements + kConvGroup - 1) / kConvGroup * kConvGroup;
  return queue.parallel_for<GdnConvPackedParallelKernel>(
      sycl::nd_range<1>(global, kConvGroup), [=](sycl::nd_item<1> item) {
        const int64_t id = item.get_global_linear_id();
        if (id >= elements) return;
        const int f = id % s.dim;
        const int linear_t = id / s.dim;
        const int t = linear_t % spec_tokens;
        const int row = (linear_t / spec_tokens) * spec_tokens;
        const int64_t initial_col = int64_t(accepted[row / spec_tokens]) - 1;
        const int idx = indices[row];
        const int global_t = tokens[row + t];
        const bool valid = initial_col >= 0 && initial_col < spec_tokens &&
                           idx > 0 && idx < s.conv_slots;
        const bool token_valid = global_t >= 0 && global_t < s.m;
        const half* base =
            valid ? conv + int64_t(idx) * s.conv_stride : nullptr;
        float history[3] = {0.f, 0.f, 0.f};
#pragma unroll
        for (int j = 0; j < 3; ++j) {
          const int prior = t - 3 + j;
          if (valid) {
            if (prior < 0) {
              const int col = initial_col + t + j;
              history[j] = float(base[int64_t(col) * s.dim + f]);
            } else {
              const int prior_token = tokens[row + prior];
              if (prior_token >= 0 && prior_token < s.m)
                history[j] =
                    float(input[int64_t(prior_token) * s.qkvz_stride + f]);
            }
          }
        }
        const float x =
            valid && token_valid
                ? float(input[int64_t(global_t) * s.qkvz_stride + f])
                : 0.f;
        float value = 0.f;
        if (valid && token_valid) {
          value =
              float(bias[f]) +
              round_spec_conv_product(history[0] * float(weight[4 * f + 0])) +
              round_spec_conv_product(history[1] * float(weight[4 * f + 1])) +
              round_spec_conv_product(history[2] * float(weight[4 * f + 2])) +
              round_spec_conv_product(x * float(weight[4 * f + 3]));
          value /= 1.0f + sycl::exp(-value);
        }
        qkv[int64_t(row + t) * s.dim + f] = half(value);
        const bool save_valid =
            indices[row + t] > 0 && indices[row + t] < s.conv_slots;
        if (token_valid && f >= 2 * s.h * kDim) {
          const int vf = f - 2 * s.h * kDim;
          z[int64_t(global_t) * s.hv * kDim + vf] =
              valid && save_valid
                  ? input[int64_t(global_t) * s.qkvz_stride + s.dim + vf]
                  : half(0.f);
        }
        // All packed-state writes occur after this kernel. The accepted
        // history may live in row 2+t, so writing checkpoints here races
        // another token's convolution input read.
      });
}

template <bool Spec, int RowsPerSubgroup, bool Aligned>
class GdnRecurrentKernel;

// Each lane owns four FP16 pairs, at columns 2*(lane + 16*j) and +1.
// A decorated global pointer and compile-time alignment avoid the generic
// address-space cast and dynamic alignment checks inside group_load/store.
inline void load_row128_block(
    sycl::sub_group sg, const half* ptr, float (&values)[kValuesPerLane]) {
  namespace sx = sycl::ext::oneapi::experimental;
  constexpr auto props = sx::properties{
      sx::data_placement_striped,
      sx::contiguous_memory,
      sx::full_group,
      sx::alignment<4>};
  sycl::vec<uint32_t, kValuesPerLane / 2> raw;
  auto* words =
      sycl::address_space_cast<
          sycl::access::address_space::global_space,
          sycl::access::decorated::yes>(reinterpret_cast<const uint32_t*>(ptr))
          .get_decorated();
  sx::group_load(sg, words, raw, props);
#pragma unroll
  for (int i = 0; i < kValuesPerLane / 2; ++i) {
    const uint32_t pair = raw[i];
    values[2 * i] = float(sycl::bit_cast<half>(uint16_t(pair)));
    values[2 * i + 1] = float(sycl::bit_cast<half>(uint16_t(pair >> 16)));
  }
}

inline void load_row128_scalar(
    sycl::sub_group sg, const half* ptr, float (&values)[kValuesPerLane]) {
  const int lane = sg.get_local_linear_id();
#pragma unroll
  for (int j = 0; j < kValuesPerLane; ++j) {
    const int col = 2 * (lane + (j / 2) * kSubgroup) + (j & 1);
    values[j] = float(ptr[col]);
  }
}

inline void store_row128_block(
    sycl::sub_group sg, half* ptr, const float (&values)[kValuesPerLane]) {
  namespace sx = sycl::ext::oneapi::experimental;
  constexpr auto props = sx::properties{
      sx::data_placement_striped,
      sx::contiguous_memory,
      sx::full_group,
      sx::alignment<4>};
  sycl::vec<uint32_t, kValuesPerLane / 2> raw;
#pragma unroll
  for (int i = 0; i < kValuesPerLane / 2; ++i) {
    const uint32_t lo = sycl::bit_cast<uint16_t>(half(values[2 * i]));
    const uint32_t hi = sycl::bit_cast<uint16_t>(half(values[2 * i + 1]));
    raw[i] = lo | (hi << 16);
  }
  auto* words =
      sycl::address_space_cast<
          sycl::access::address_space::global_space,
          sycl::access::decorated::yes>(reinterpret_cast<uint32_t*>(ptr))
          .get_decorated();
  sx::group_store(sg, raw, words, props);
}

inline void store_row128_scalar(
    sycl::sub_group sg, half* ptr, const float (&values)[kValuesPerLane]) {
  const int lane = sg.get_local_linear_id();
#pragma unroll
  for (int j = 0; j < kValuesPerLane; ++j) {
    const int col = 2 * (lane + (j / 2) * kSubgroup) + (j & 1);
    ptr[col] = half(values[j]);
  }
}

// The root barrier, rather than a one-wave scheduling assumption, protects
// the shared Q/K convolution history from another head's in-place shift.
// Each WG owns 32 V rows; its Q/K/V convolution values stay FP32 in SLM,
// matching the final fused reference's numerical boundary. The legacy two
// kernels remain the fallback for other geometries and unsupported devices.
class GdnDecodeRootKernel;
constexpr int kRootLocalSize = kSubgroup * kRowsPerGroup;
constexpr int kRootRowsPerSubgroup = 4;
constexpr int kRootRowsPerGroup =
    kRootLocalSize / kSubgroup * kRootRowsPerSubgroup;
constexpr int kRootSplits = kDim / kRootRowsPerGroup;
constexpr size_t kRootLocalBytes = 3 * kDim * sizeof(float);

template <class Body>
struct GdnRootBody {
  Body body;
  constexpr auto get(sycl::ext::oneapi::experimental::properties_tag) const {
    namespace sx = sycl::ext::oneapi::experimental;
    return sx::properties{sx::sub_group_size<kSubgroup>};
  }
  void operator()(sycl::nd_item<1> item) const { body(item); }
};

size_t root_kernel_group_limit(sycl::queue& queue) {
  namespace sx = sycl::ext::oneapi::experimental;
  struct CachedLimit {
    sycl::queue queue;
    size_t limit;
  };
  // Queue identity includes its context/device. No tensor, indices or shape
  // validation is cached; a new stream is queried before its first submit.
  static thread_local std::optional<CachedLimit> cached;
  if (cached && cached->queue == queue) return cached->limit;
  size_t limit = 0;
  try {
    const auto id = sycl::get_kernel_id<GdnDecodeRootKernel>();
    const auto bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
        queue.get_context(), {queue.get_device()}, {id});
    const auto kernel = bundle.get_kernel(id);
    limit = kernel.ext_oneapi_get_info<
        sx::info::kernel_queue_specific::max_num_work_groups>(
        queue, sycl::range<1>{kRootLocalSize}, kRootLocalBytes);
  } catch (const sycl::exception&) {
    // This is a pre-submit capability failure only. Errors from launching or
    // executing the cooperative kernel must propagate, never retry on state
    // that may already have been modified.
  }
  cached.emplace(CachedLimit{queue, limit});
  return limit;
}

void launch_decode_root(
    sycl::queue& queue,
    const half* input,
    half* conv,
    const half* weight,
    const half* bias,
    const int* conv_indices,
    const int* ssm_indices,
    const half* a_log,
    const half* dt_bias,
    const half* ba,
    half* state,
    half* output,
    half* z,
    Shape s) {
  namespace sx = sycl::ext::oneapi::experimental;
  const size_t groups = size_t(s.hv) * kRootSplits;
  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> qkv(sycl::range<1>{3 * kDim}, cgh);
    sx::nd_launch<GdnDecodeRootKernel>(
        cgh,
        sx::launch_config{
            sycl::nd_range<1>{groups * kRootLocalSize, kRootLocalSize},
            sx::properties{sx::use_root_sync}},
        GdnRootBody{[=](sycl::nd_item<1> item) {
          const int tid = item.get_local_linear_id();
          const int group = item.get_group_linear_id();
          const int hv = group / kRootSplits;
          const int split = group % kRootSplits;
          const int kh = hv / (s.hv / s.h);
          const int ci = conv_indices[0], si = ssm_indices[0];
          const bool valid =
              ci >= 0 && ci < s.conv_slots && si >= 0 && si < s.ssm_slots;
          half* history = valid ? conv + int64_t(ci) * s.conv_stride : nullptr;
          float saved_b[3], saved_c[3], saved_x[3];
#pragma unroll
          for (int part = 0; part < 3; ++part) {
            const int base = part == 0   ? kh * kDim
                             : part == 1 ? (s.h + kh) * kDim
                                         : (2 * s.h + hv) * kDim;
            const int f = base + tid;
            const float a = valid ? float(history[f]) : 0.f;
            const float b = valid ? float(history[s.dim + f]) : 0.f;
            const float c = valid ? float(history[2 * s.dim + f]) : 0.f;
            const float x = valid ? float(input[f]) : 0.f;
            float value = 0.f;
            if (valid) {
              value = a * float(weight[4 * f]) + b * float(weight[4 * f + 1]) +
                      c * float(weight[4 * f + 2]) +
                      x * float(weight[4 * f + 3]) + float(bias[f]);
              value /= 1.f + sycl::exp(-value);
            }
            qkv[part * kDim + tid] = value;
            saved_b[part] = b;
            saved_c[part] = c;
            saved_x[part] = x;
          }
          // Every WG participates, including invalid slots. All old history
          // reads and this WG's SLM writes precede the unique-owner shifts.
          sycl::group_barrier(item.ext_oneapi_get_root_group());
          if (valid && split == 0) {
#pragma unroll
            for (int part = 0; part < 3; ++part) {
              if (part < 2 && hv % (s.hv / s.h) != 0) continue;
              const int base = part == 0   ? kh * kDim
                               : part == 1 ? (s.h + kh) * kDim
                                           : (2 * s.h + hv) * kDim;
              const int f = base + tid;
              history[f] = half(saved_b[part]);
              history[s.dim + f] = half(saved_c[part]);
              history[2 * s.dim + f] = half(saved_x[part]);
            }
          }
          const auto sg = item.get_sub_group();
          const int lane = sg.get_local_linear_id();
          const int vi_base = split * kRootRowsPerGroup +
                              (tid / kSubgroup) * kRootRowsPerSubgroup;
          if (!valid) {
            if (lane == 0) {
#pragma unroll
              for (int r = 0; r < kRootRowsPerSubgroup; ++r) {
                output[hv * kDim + vi_base + r] = half(0.f);
                z[hv * kDim + vi_base + r] = half(0.f);
              }
            }
            return;
          }
          float q[kValuesPerLane], k[kValuesPerLane];
          float q2 = 0.f, k2 = 0.f;
#pragma unroll
          for (int j = 0; j < kValuesPerLane; ++j) {
            const int col = 2 * (lane + (j / 2) * kSubgroup) + (j & 1);
            q[j] = qkv[col];
            k[j] = qkv[kDim + col];
            q2 += q[j] * q[j];
            k2 += k[j] * k[j];
          }
          const float q_inv =
              s.scale *
              sycl::rsqrt(
                  sycl::reduce_over_group(sg, q2, sycl::plus<float>()) + 1e-6f);
          const float k_inv = sycl::rsqrt(
              sycl::reduce_over_group(sg, k2, sycl::plus<float>()) + 1e-6f);
#pragma unroll
          for (int j = 0; j < kValuesPerLane; ++j) {
            q[j] *= q_inv;
            k[j] *= k_inv;
          }
          const float decay = sycl::exp(
              -sycl::exp(float(a_log[hv])) *
              softplus(float(ba[s.hv + hv]) + float(dt_bias[hv])));
          const float beta = sigmoid(float(ba[hv]));
          half* head =
              state + int64_t(si) * s.ssm_stride + int64_t(hv) * kDim * kDim;
          float h[kRootRowsPerSubgroup][kValuesPerLane];
#pragma unroll
          for (int r = 0; r < kRootRowsPerSubgroup; ++r)
            load_row128_block(sg, head + (vi_base + r) * kDim, h[r]);
#pragma unroll
          for (int r = 0; r < kRootRowsPerSubgroup; ++r) {
            float kv = 0.f;
#pragma unroll
            for (int j = 0; j < kValuesPerLane; ++j) {
              h[r][j] *= decay;
              kv += h[r][j] * k[j];
            }
            kv = sycl::reduce_over_group(sg, kv, sycl::plus<float>());
            const int vi = vi_base + r;
            const float delta = (qkv[2 * kDim + vi] - kv) * beta;
            float dot = 0.f;
#pragma unroll
            for (int j = 0; j < kValuesPerLane; ++j) {
              h[r][j] += delta * k[j];
              dot += h[r][j] * q[j];
            }
            dot = sycl::reduce_over_group(sg, dot, sycl::plus<float>());
            if (lane == 0) {
              output[hv * kDim + vi] = half(dot);
              z[hv * kDim + vi] = input[s.dim + hv * kDim + vi];
            }
            store_row128_block(sg, head + vi * kDim, h[r]);
          }
        }});
  });
}

template <bool Spec, int RowsPerSubgroup, bool Aligned>
void launch_recurrent(
    sycl::queue& queue,
    const half* qkv,
    const half* ba,
    const half* a_half,
    const float* a_float,
    const half* dt_bias,
    half* state,
    const int* indices,
    const int* conv_indices,
    const int* tokens,
    const int* accepted,
    half* output,
    half* conv,
    const half* original_input,
    Shape s,
    int sequences,
    int spec_tokens,
    sycl::event ready) {
  constexpr int rows_per_workgroup = kRowsPerGroup * RowsPerSubgroup;
  const int64_t groups =
      int64_t(sequences) * s.hv * (kDim / rows_per_workgroup);
  queue.submit([&](sycl::handler& cgh) {
    cgh.depends_on(ready);
    cgh.parallel_for<GdnRecurrentKernel<Spec, RowsPerSubgroup, Aligned>>(
        sycl::nd_range<1>(
            groups * kSubgroup * kRowsPerGroup, kSubgroup * kRowsPerGroup),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(kSubgroup)]] {
          const auto sg = item.get_sub_group();
          const int lane = sg.get_local_linear_id();
          const int64_t group = item.get_group_linear_id();
          const int vi_base =
              (group % (kDim / rows_per_workgroup)) * rows_per_workgroup +
              (item.get_local_linear_id() / kSubgroup) * RowsPerSubgroup;
          const int hv = (group / (kDim / rows_per_workgroup)) % s.hv;
          const int seq = group / (int64_t(s.hv) * (kDim / rows_per_workgroup));
          const int row = seq * spec_tokens;
          int initial_idx = indices[row];
          if constexpr (Spec) {
            const int64_t col = int64_t(accepted[seq]) - 1;
            initial_idx =
                col >= 0 && col < spec_tokens ? indices[row + col] : -1;
          }
          const bool conv_valid = Spec || (conv_indices[row] >= 0 &&
                                           conv_indices[row] < s.conv_slots);
          const bool initial_valid = conv_valid &&
                                     initial_idx >= (Spec ? 1 : 0) &&
                                     initial_idx < s.ssm_slots;
          const int64_t head_offset = int64_t(hv) * kDim * kDim;
          float h[RowsPerSubgroup][kValuesPerLane];
#pragma unroll
          for (int r = 0; r < RowsPerSubgroup; ++r) {
            if (initial_valid) {
              const half* source = state + int64_t(initial_idx) * s.ssm_stride +
                                   head_offset + int64_t(vi_base + r) * kDim;
              if constexpr (Aligned)
                load_row128_block(sg, source, h[r]);
              else
                load_row128_scalar(sg, source, h[r]);
            } else {
#pragma unroll
              for (int j = 0; j < kValuesPerLane; ++j)
                h[r][j] = 0.f;
            }
          }
          const int kh = hv / (s.hv / s.h);
          const float neg_a =
              -sycl::exp(Spec ? a_float[hv] : float(a_half[hv]));
          const float bias = float(dt_bias[hv]);
          for (int t = 0; t < spec_tokens; ++t) {
            const int global_t = Spec ? tokens[row + t] : seq;
            const int save_idx = indices[row + t];
            const bool token_valid = global_t >= 0 && global_t < s.m;
            const bool save_valid =
                save_idx >= (Spec ? 1 : 0) && save_idx < s.ssm_slots;
            if (!token_valid) continue;
            if (!initial_valid) {
              if (lane == 0) {
#pragma unroll
                for (int r = 0; r < RowsPerSubgroup; ++r)
                  output
                      [int64_t(global_t) * s.hv * kDim + hv * kDim + vi_base +
                       r] = half(0.f);
              }
              continue;
            }
            const half* input = qkv + int64_t(row + t) * s.dim;
            float q[kValuesPerLane], k[kValuesPerLane];
            float q2 = 0.f, k2 = 0.f;
            if constexpr (Aligned) {
              load_row128_block(sg, input + kh * kDim, q);
              load_row128_block(sg, input + (s.h + kh) * kDim, k);
            } else {
              load_row128_scalar(sg, input + kh * kDim, q);
              load_row128_scalar(sg, input + (s.h + kh) * kDim, k);
            }
#pragma unroll
            for (int j = 0; j < kValuesPerLane; ++j) {
              q2 += q[j] * q[j];
              k2 += k[j] * k[j];
            }
            const float q_inv = s.scale * sycl::rsqrt(
                                              sycl::reduce_over_group(
                                                  sg, q2, sycl::plus<float>()) +
                                              1e-6f);
            const float k_inv = sycl::rsqrt(
                sycl::reduce_over_group(sg, k2, sycl::plus<float>()) + 1e-6f);
            const float b = float(ba[int64_t(global_t) * s.ba_stride + hv]);
            const float a =
                float(ba[int64_t(global_t) * s.ba_stride + s.hv + hv]);
            const float decay = sycl::exp(neg_a * softplus(a + bias));
            const float beta = sigmoid(b);
#pragma unroll
            for (int j = 0; j < kValuesPerLane; ++j) {
              k[j] *= k_inv;
              q[j] *= q_inv;
            }
#pragma unroll
            for (int r = 0; r < RowsPerSubgroup; ++r) {
              const int vi = vi_base + r;
              const float v = float(input[(2 * s.h + hv) * kDim + vi]);
              float kv = 0.f;
#pragma unroll
              for (int j = 0; j < kValuesPerLane; ++j) {
                h[r][j] *= decay;
                kv += h[r][j] * k[j];
              }
              kv = sycl::reduce_over_group(sg, kv, sycl::plus<float>());
              const float delta = (v - kv) * beta;
              float dot = 0.f;
#pragma unroll
              for (int j = 0; j < kValuesPerLane; ++j) {
                h[r][j] += delta * k[j];
                dot += h[r][j] * q[j];
              }
              dot = sycl::reduce_over_group(sg, dot, sycl::plus<float>());
              if (lane == 0)
                output[int64_t(global_t) * s.hv * kDim + hv * kDim + vi] =
                    Spec && !save_valid ? half(0.f) : half(dot);
              if (save_valid) {
                const int64_t dst = int64_t(save_idx) * s.ssm_stride +
                                    head_offset + int64_t(vi) * kDim;
                if constexpr (Aligned)
                  store_row128_block(sg, state + dst, h[r]);
                else
                  store_row128_scalar(sg, state + dst, h[r]);
              }
            }
          }
          if constexpr (Spec) {
            if (conv != nullptr) {
              // Every subgroup owns a disjoint pair of features. Load both
              // source rows before either store (accepted=1 aliases row 1).
              const int64_t subgroups_per_seq =
                  int64_t(s.hv) * (kDim / rows_per_workgroup) * kRowsPerGroup;
              const int64_t subgroup =
                  (group % (int64_t(s.hv) * (kDim / rows_per_workgroup))) *
                      kRowsPerGroup +
                  item.get_local_linear_id() / kSubgroup;
              const int64_t col = int64_t(accepted[seq]) - 1;
              const int conv_idx = indices[row];
              if (col >= 0 && col < spec_tokens && conv_idx > 0 &&
                  conv_idx < s.conv_slots) {
#pragma unroll
                for (int p = 0; p < 2; ++p) {
                  const int64_t f = subgroup * 2 + p;
                  if (lane == 0 && f < s.dim) {
                    half* dest = conv + int64_t(conv_idx) * s.conv_stride;
                    const half a = dest[int64_t(col + 1) * s.dim + f];
                    const half b = dest[int64_t(col + 2) * s.dim + f];
                    dest[f] = a;
                    dest[s.dim + f] = b;
                    for (int t = 0; t < spec_tokens; ++t) {
                      const int token = tokens[row + t];
                      if (token >= 0 && token < s.m)
                        dest[int64_t(2 + t) * s.dim + f] =
                            original_input[int64_t(token) * s.qkvz_stride + f];
                    }
                  }
                }
              }
            }
          }
        });
  });
}

class GdnNormGateKernel;

}  // namespace

bool tensors_disjoint_host(
    const std::vector<torch::Tensor>& writes,
    const std::vector<std::optional<torch::Tensor>>& reads) {
  struct Entry {
    c10::Device device;
    MemoryRows rows;
  };
  std::vector<Entry> entries;
  entries.reserve(writes.size() + reads.size());
  const auto append = [&](const torch::Tensor& tensor) {
    if (!tensor.defined() || tensor.layout() != at::kStrided ||
        tensor.is_meta())
      return false;
    entries.push_back({tensor.device(), memory_rows(tensor)});
    return true;
  };
  // All failure handling is confined to metadata inspection. This function
  // never allocates device storage, submits a kernel, or changes a tensor.
  try {
    for (const auto& tensor : writes)
      if (!append(tensor)) return false;
    for (const auto& tensor : reads)
      if (tensor.has_value() && !append(*tensor)) return false;
  } catch (const c10::Error&) {
    return false;
  }
  for (size_t i = 0; i < writes.size(); ++i)
    for (size_t j = i + 1; j < entries.size(); ++j)
      if (entries[i].device == entries[j].device &&
          physical_overlap(entries[i].rows, entries[j].rows))
        return false;
  return true;
}

void gdn_decode_sycl(
    const torch::Tensor& qkvz,
    torch::Tensor& conv_state,
    const torch::Tensor& conv_weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& conv_indices,
    const torch::Tensor& a_log,
    const torch::Tensor& dt_bias,
    const torch::Tensor& ba,
    torch::Tensor& ssm_state,
    const torch::Tensor& ssm_indices,
    torch::Tensor& output,
    torch::Tensor& z,
    double scale) {
  const Shape s = check_core(
      qkvz,
      conv_state,
      conv_weight,
      conv_bias,
      a_log,
      dt_bias,
      ba,
      ssm_state,
      output,
      z,
      false,
      qkvz.size(0),
      scale);
  TORCH_CHECK(s.conv_len == 3, "ordinary decode requires exactly 3 conv rows");
  check_indices(conv_indices, qkvz, s.m, "conv indices");
  check_indices(ssm_indices, qkvz, s.m, "SSM indices");
  check_no_cross_alias(
      {&conv_state, &ssm_state, &output, &z}, {&conv_indices, &ssm_indices});
  c10::OptionalDeviceGuard guard(qkvz.device());
  auto stream = c10::xpu::getCurrentXPUStream(qkvz.device().index());
  auto& queue = stream.queue();
  // Preserve ownership even if a later submit fails after conv mutates state.
  record_stream(
      {&qkvz,
       &conv_state,
       &conv_weight,
       &conv_bias,
       &conv_indices,
       &a_log,
       &dt_bias,
       &ba,
       &ssm_state,
       &ssm_indices,
       &output,
       &z},
      stream);
  // 当前 UR graph 无法 finalize use_root_sync；capture 保留两阶段数学路径。
  const bool root_eligible =
      s.m == 1 && s.h == 4 && queue.is_in_order() &&
      c10::xpu::currentStreamCaptureStatusMayInitCtx() !=
          c10::xpu::CaptureStatus::Recording &&
      (reinterpret_cast<uintptr_t>(ssm_state.data_ptr()) & 3U) == 0 &&
      (s.ssm_stride & 1) == 0;
  if (root_eligible &&
      root_kernel_group_limit(queue) >= size_t(s.hv) * kRootSplits) {
    launch_decode_root(
        queue,
        reinterpret_cast<const half*>(qkvz.data_ptr()),
        reinterpret_cast<half*>(conv_state.data_ptr()),
        reinterpret_cast<const half*>(conv_weight.data_ptr()),
        reinterpret_cast<const half*>(conv_bias.data_ptr()),
        conv_indices.data_ptr<int>(),
        ssm_indices.data_ptr<int>(),
        reinterpret_cast<const half*>(a_log.data_ptr()),
        reinterpret_cast<const half*>(dt_bias.data_ptr()),
        reinterpret_cast<const half*>(ba.data_ptr()),
        reinterpret_cast<half*>(ssm_state.data_ptr()),
        reinterpret_cast<half*>(output.data_ptr()),
        reinterpret_cast<half*>(z.data_ptr()),
        s);
    return;
  }
  // Per-call scratch is only q/k/v. The current stream and allocator own it.
  auto qkv = at::empty({s.m, s.dim}, qkvz.options());
  record_stream({&qkv}, stream);
  const auto ready = launch_conv<false>(
      queue,
      reinterpret_cast<const half*>(qkvz.data_ptr()),
      reinterpret_cast<half*>(conv_state.data_ptr()),
      reinterpret_cast<const half*>(conv_weight.data_ptr()),
      reinterpret_cast<const half*>(conv_bias.data_ptr()),
      conv_indices.data_ptr<int>(),
      ssm_indices.data_ptr<int>(),
      nullptr,
      nullptr,
      reinterpret_cast<half*>(qkv.data_ptr()),
      reinterpret_cast<half*>(z.data_ptr()),
      s,
      s.m,
      1);
  const auto recurrent = [&]<int Rows, bool Aligned>() {
    launch_recurrent<false, Rows, Aligned>(
        queue,
        reinterpret_cast<const half*>(qkv.data_ptr()),
        reinterpret_cast<const half*>(ba.data_ptr()),
        reinterpret_cast<const half*>(a_log.data_ptr()),
        nullptr,
        reinterpret_cast<const half*>(dt_bias.data_ptr()),
        reinterpret_cast<half*>(ssm_state.data_ptr()),
        ssm_indices.data_ptr<int>(),
        conv_indices.data_ptr<int>(),
        nullptr,
        nullptr,
        reinterpret_cast<half*>(output.data_ptr()),
        nullptr,
        nullptr,
        s,
        s.m,
        1,
        ready);
  };
  const bool aligned = ((reinterpret_cast<uintptr_t>(qkv.data_ptr()) |
                         reinterpret_cast<uintptr_t>(ssm_state.data_ptr())) &
                        3U) == 0 &&
                       (s.ssm_stride & 1) == 0;
  if (s.m <= 2) {
    if (aligned)
      recurrent.template operator()<2, true>();
    else
      recurrent.template operator()<2, false>();
  } else {
    if (aligned)
      recurrent.template operator()<4, true>();
    else
      recurrent.template operator()<4, false>();
  }
}

void gdn_spec_v2_sycl(
    const torch::Tensor& qkvz,
    torch::Tensor& conv_state,
    const torch::Tensor& conv_weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& spec_indices,
    const torch::Tensor& a_log,
    const torch::Tensor& dt_bias,
    const torch::Tensor& ba,
    torch::Tensor& ssm_state,
    torch::Tensor& output,
    torch::Tensor& z,
    const torch::Tensor& token_indices,
    const torch::Tensor& accepted,
    int64_t sequences,
    int64_t tokens_per_sequence,
    double scale) {
  TORCH_CHECK(
      sequences > 0 && tokens_per_sequence >= 2 && tokens_per_sequence <= 8 &&
          sequences <= std::numeric_limits<int>::max() / tokens_per_sequence,
      "spec v2 requires sequences>0 and M=2..8");
  const int64_t m = sequences * tokens_per_sequence;
  const Shape s = check_core(
      qkvz,
      conv_state,
      conv_weight,
      conv_bias,
      a_log,
      dt_bias,
      ba,
      ssm_state,
      output,
      z,
      true,
      m,
      scale);
  TORCH_CHECK(
      s.conv_len == 3 || s.conv_len >= tokens_per_sequence + 2,
      "spec v2 conv cache must be 3-row or packed M+2");
  check_indices(spec_indices, qkvz, m, "spec state indices");
  check_indices(token_indices, qkvz, m, "spec token indices");
  check_indices(accepted, qkvz, sequences, "accepted count");
  check_no_cross_alias(
      {&conv_state, &ssm_state, &output, &z},
      {&spec_indices, &token_indices, &accepted});
  c10::OptionalDeviceGuard guard(qkvz.device());
  auto stream = c10::xpu::getCurrentXPUStream(qkvz.device().index());
  auto& queue = stream.queue();
  record_stream(
      {&qkvz,
       &conv_state,
       &conv_weight,
       &conv_bias,
       &spec_indices,
       &a_log,
       &dt_bias,
       &ba,
       &ssm_state,
       &output,
       &z,
       &token_indices,
       &accepted},
      stream);
  auto qkv = at::empty({m, s.dim}, qkvz.options());
  record_stream({&qkv}, stream);
  const auto ready =
      s.conv_len == 3
          ? launch_conv<true>(
                queue,
                reinterpret_cast<const half*>(qkvz.data_ptr()),
                reinterpret_cast<half*>(conv_state.data_ptr()),
                reinterpret_cast<const half*>(conv_weight.data_ptr()),
                reinterpret_cast<const half*>(conv_bias.data_ptr()),
                spec_indices.data_ptr<int>(),
                nullptr,
                token_indices.data_ptr<int>(),
                accepted.data_ptr<int>(),
                reinterpret_cast<half*>(qkv.data_ptr()),
                reinterpret_cast<half*>(z.data_ptr()),
                s,
                static_cast<int>(sequences),
                static_cast<int>(tokens_per_sequence))
          : launch_conv_packed_parallel(
                queue,
                reinterpret_cast<const half*>(qkvz.data_ptr()),
                reinterpret_cast<half*>(conv_state.data_ptr()),
                reinterpret_cast<const half*>(conv_weight.data_ptr()),
                reinterpret_cast<const half*>(conv_bias.data_ptr()),
                spec_indices.data_ptr<int>(),
                token_indices.data_ptr<int>(),
                accepted.data_ptr<int>(),
                reinterpret_cast<half*>(qkv.data_ptr()),
                reinterpret_cast<half*>(z.data_ptr()),
                s,
                static_cast<int>(sequences),
                static_cast<int>(tokens_per_sequence));
  const auto recurrent = [&]<bool Aligned>() {
    launch_recurrent<true, 1, Aligned>(
        queue,
        reinterpret_cast<const half*>(qkv.data_ptr()),
        reinterpret_cast<const half*>(ba.data_ptr()),
        nullptr,
        a_log.data_ptr<float>(),
        reinterpret_cast<const half*>(dt_bias.data_ptr()),
        reinterpret_cast<half*>(ssm_state.data_ptr()),
        spec_indices.data_ptr<int>(),
        nullptr,
        token_indices.data_ptr<int>(),
        accepted.data_ptr<int>(),
        reinterpret_cast<half*>(output.data_ptr()),
        s.conv_len == 3 ? nullptr
                        : reinterpret_cast<half*>(conv_state.data_ptr()),
        s.conv_len == 3 ? nullptr
                        : reinterpret_cast<const half*>(qkvz.data_ptr()),
        s,
        static_cast<int>(sequences),
        static_cast<int>(tokens_per_sequence),
        ready);
  };
  const bool aligned = ((reinterpret_cast<uintptr_t>(qkv.data_ptr()) |
                         reinterpret_cast<uintptr_t>(ssm_state.data_ptr())) &
                        3U) == 0 &&
                       (s.ssm_stride & 1) == 0;
  if (aligned)
    recurrent.template operator()<true>();
  else
    recurrent.template operator()<false>();
}

void gdn_norm_gate_sycl(
    const torch::Tensor& x,
    const torch::Tensor& z,
    const torch::Tensor& weight,
    torch::Tensor& normalized,
    double eps,
    bool sigmoid_gate) {
  TORCH_CHECK(
      x.is_xpu() && x.is_contiguous() && !x.is_neg() && !x.is_conj() &&
          x.scalar_type() == at::kHalf && x.dim() == 3 && x.size(0) > 0 &&
          x.size(1) > 0 && x.size(2) == kDim && x.stride(2) == 1 &&
          x.stride(1) == kDim && x.stride(0) == x.size(1) * kDim,
      "x must be contiguous non-lazy [M,HV,128] FP16 XPU");
  for (auto* t :
       std::initializer_list<const torch::Tensor*>{&z, &weight, &normalized})
    check_data(*t, x, at::kHalf, "norm tensor");
  TORCH_CHECK(z.sizes() == x.sizes() && z.is_contiguous(), "invalid z");
  TORCH_CHECK(
      weight.is_contiguous() && weight.dim() == 1 && weight.numel() == kDim,
      "invalid norm weight");
  TORCH_CHECK(
      normalized.is_contiguous() && normalized.dim() == 2 &&
          normalized.size(0) == x.size(0) &&
          normalized.size(1) == x.size(1) * kDim,
      "normalized must be [M,HV*128]");
  const float epsilon = static_cast<float>(eps);
  TORCH_CHECK(
      std::isfinite(epsilon) && epsilon > 0.0f,
      "invalid norm eps after FP32 conversion");
  check_no_cross_alias({&normalized}, {&x, &z, &weight});
  c10::OptionalDeviceGuard guard(x.device());
  auto stream = c10::xpu::getCurrentXPUStream(x.device().index());
  auto& queue = stream.queue();
  record_stream({&x, &z, &weight, &normalized}, stream);
  const int64_t rows = x.size(0) * x.size(1);
  auto* px = reinterpret_cast<const half*>(x.data_ptr());
  auto* pz = reinterpret_cast<const half*>(z.data_ptr());
  auto* pw = reinterpret_cast<const half*>(weight.data_ptr());
  auto* po = reinterpret_cast<half*>(normalized.data_ptr());
  queue.parallel_for<GdnNormGateKernel>(
      sycl::nd_range<1>(rows * kSubgroup, kSubgroup),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(kSubgroup)]] {
        const auto sg = item.get_sub_group();
        const int lane = sg.get_local_linear_id();
        const int64_t row = item.get_group_linear_id();
        float xv[kValuesPerLane], zv[kValuesPerLane];
        float sum = 0.f;
#pragma unroll
        for (int j = 0; j < kValuesPerLane; ++j) {
          const int col = lane + j * kSubgroup;
          xv[j] = float(px[row * kDim + col]);
          zv[j] = float(pz[row * kDim + col]);
          sum += xv[j] * xv[j];
        }
        sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>());
        const float inv = sycl::rsqrt(sum / kDim + epsilon);
#pragma unroll
        for (int j = 0; j < kValuesPerLane; ++j) {
          const int col = lane + j * kSubgroup;
          const float denom = 1.0f + sycl::exp(-zv[j]);
          const float gate = sigmoid_gate ? 1.0f / denom : zv[j] / denom;
          po[row * kDim + col] = half(xv[j] * inv * float(pw[col]) * gate);
        }
      });
}

#ifdef QWEN38_GDN_STANDALONE_TEST
// Test-only view of the actual speculative convolution's FP16 output.
// It keeps the production ABI unchanged while exposing the rounding boundary.
void gdn_spec_conv_probe_sycl(
    const torch::Tensor& qkvz,
    torch::Tensor& conv,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& indices,
    const torch::Tensor& tokens,
    const torch::Tensor& accepted,
    torch::Tensor& qkv,
    torch::Tensor& z,
    int64_t h,
    int64_t hv) {
  const int64_t m = tokens.numel();
  TORCH_CHECK(
      ((h == 4 && hv == 12) || (h == 2 && hv == 6)) && m >= 2 && m <= 8,
      "probe requires TP4/TP8 speculative M=2..8");
  const int64_t dim = (2 * h + hv) * kDim;
  TORCH_CHECK(qkvz.is_xpu(), "probe requires XPU tensors");
  for (const auto* t : std::initializer_list<const torch::Tensor*>{
           &conv, &weight, &bias, &qkv, &z})
    check_data(*t, qkvz, at::kHalf, "probe tensor");
  check_indices(indices, qkvz, m, "probe indices");
  check_indices(tokens, qkvz, m, "probe tokens");
  check_indices(accepted, qkvz, 1, "probe accepted");
  TORCH_CHECK(
      qkvz.is_contiguous() && qkvz.dim() == 2 && qkvz.size(0) == m &&
          qkvz.size(1) == dim + hv * kDim && conv.is_contiguous() &&
          conv.dim() == 3 && conv.size(0) > 1 &&
          conv.size(0) <= std::numeric_limits<int>::max() &&
          (conv.size(1) == 3 || conv.size(1) >= m + 2) && conv.size(2) == dim &&
          weight.is_contiguous() && weight.dim() == 2 &&
          weight.size(0) == dim && weight.size(1) == 4 &&
          bias.is_contiguous() && bias.numel() == dim && qkv.is_contiguous() &&
          qkv.dim() == 2 && qkv.size(0) == m && qkv.size(1) == dim &&
          z.is_contiguous() && z.dim() == 3 && z.size(0) == m &&
          z.size(1) == hv && z.size(2) == kDim,
      "invalid probe layouts");
  check_no_cross_alias(
      {&conv, &qkv, &z}, {&qkvz, &weight, &bias, &indices, &tokens, &accepted});
  const Shape s{
      static_cast<int>(m),
      static_cast<int>(h),
      static_cast<int>(hv),
      static_cast<int>(dim),
      qkvz.stride(0),
      0,
      conv.stride(0),
      0,
      static_cast<int>(conv.size(0)),
      0,
      static_cast<int>(conv.size(1)),
      1.0f};
  c10::OptionalDeviceGuard guard(qkvz.device());
  auto stream = c10::xpu::getCurrentXPUStream(qkvz.device().index());
  record_stream(
      {&qkvz, &conv, &weight, &bias, &indices, &tokens, &accepted, &qkv, &z},
      stream);
  if (s.conv_len == 3) {
    launch_conv<true>(
        stream.queue(),
        reinterpret_cast<const half*>(qkvz.data_ptr()),
        reinterpret_cast<half*>(conv.data_ptr()),
        reinterpret_cast<const half*>(weight.data_ptr()),
        reinterpret_cast<const half*>(bias.data_ptr()),
        indices.data_ptr<int>(),
        nullptr,
        tokens.data_ptr<int>(),
        accepted.data_ptr<int>(),
        reinterpret_cast<half*>(qkv.data_ptr()),
        reinterpret_cast<half*>(z.data_ptr()),
        s,
        1,
        static_cast<int>(m));
  } else {
    launch_conv_packed_parallel(
        stream.queue(),
        reinterpret_cast<const half*>(qkvz.data_ptr()),
        reinterpret_cast<half*>(conv.data_ptr()),
        reinterpret_cast<const half*>(weight.data_ptr()),
        reinterpret_cast<const half*>(bias.data_ptr()),
        indices.data_ptr<int>(),
        tokens.data_ptr<int>(),
        accepted.data_ptr<int>(),
        reinterpret_cast<half*>(qkv.data_ptr()),
        reinterpret_cast<half*>(z.data_ptr()),
        s,
        1,
        static_cast<int>(m));
  }
}
#endif

}  // namespace vllm::qwen38

#ifdef QWEN38_GDN_STANDALONE_TEST
TORCH_LIBRARY(qwen38_gdn_sycl_test, m) {
  m.def(
      "decode(Tensor qkvz, Tensor(a!) conv, Tensor weight, Tensor bias, "
      "Tensor conv_idx, Tensor a_log, Tensor dt_bias, Tensor ba, "
      "Tensor(b!) ssm, Tensor ssm_idx, Tensor(c!) output, Tensor(d!) z, "
      "float scale) -> ()");
  m.impl("decode", torch::kXPU, &vllm::qwen38::gdn_decode_sycl);
  m.def(
      "spec_v2(Tensor qkvz, Tensor(a!) conv, Tensor weight, Tensor bias, "
      "Tensor idx, Tensor a_log, Tensor dt_bias, Tensor ba, "
      "Tensor(b!) ssm, Tensor(c!) output, Tensor(d!) z, Tensor token_idx, "
      "Tensor accepted, int sequences, int tokens_per_sequence, "
      "float scale) -> ()");
  m.impl("spec_v2", torch::kXPU, &vllm::qwen38::gdn_spec_v2_sycl);
  m.def(
      "norm_gate(Tensor x, Tensor z, Tensor weight, Tensor(a!) out, "
      "float eps, bool sigmoid_gate) -> ()");
  m.impl("norm_gate", torch::kXPU, &vllm::qwen38::gdn_norm_gate_sycl);
  m.def(
      "spec_conv_probe(Tensor qkvz, Tensor(a!) conv, Tensor weight, "
      "Tensor bias, Tensor indices, Tensor tokens, Tensor accepted, "
      "Tensor(b!) qkv, Tensor(c!) z, int h, int hv) -> ()");
  m.impl(
      "spec_conv_probe", torch::kXPU, &vllm::qwen38::gdn_spec_conv_probe_sycl);
}
#endif
