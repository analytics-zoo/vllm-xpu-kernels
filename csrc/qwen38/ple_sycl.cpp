// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <unordered_set>
#include <vector>

#include "qwen38/ple_sycl.h"

namespace vllm::qwen38::ple_sycl {
namespace {

using half = sycl::half;
constexpr int kMaxConvState = 128;
constexpr int kMaxSpecTokens = 128;
constexpr int kHcCount = 4;
constexpr int kHcHidden = 2560;
constexpr int kHcWidth = kHcCount * kHcHidden;
constexpr size_t kReduceLocal = 1024;
constexpr size_t kFusedLocal = 256;

void check_xpu(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.device().is_xpu(), name, " must be on XPU");
  TORCH_CHECK(!t.is_neg() && !t.is_conj(), name, " must not be a lazy view");
}

void check_tensor(
    const at::Tensor& base,
    const at::Tensor& t,
    at::ScalarType dtype,
    const char* name) {
  check_xpu(t, name);
  TORCH_CHECK(t.device() == base.device(), name, " must be on the same XPU");
  TORCH_CHECK(t.scalar_type() == dtype, name, " has an invalid dtype");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_index(
    const at::Tensor& base, const at::Tensor& t, const char* name) {
  check_xpu(t, name);
  TORCH_CHECK(t.device() == base.device(), name, " must be on the same XPU");
  TORCH_CHECK(
      t.scalar_type() == at::kInt || t.scalar_type() == at::kLong,
      name,
      " must have dtype int32 or int64");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

struct ByteRange {
  std::uintptr_t begin;
  std::uintptr_t end;
};

// Storage identity is insufficient when two tensors independently wrap the
// same allocation. A strided state's bounding range conservatively includes
// its padding; rejecting a false-positive overlap is safer than a data race.
ByteRange byte_range(const at::Tensor& t) {
  const auto begin = reinterpret_cast<std::uintptr_t>(t.data_ptr());
  if (t.numel() == 0) return {begin, begin};
  constexpr auto max = std::numeric_limits<std::uintptr_t>::max();
  std::uintptr_t last_element = 0;
  for (int64_t dim = 0; dim < t.dim(); ++dim) {
    TORCH_CHECK(t.stride(dim) >= 0, "negative tensor stride is unsupported");
    const auto extent = static_cast<std::uintptr_t>(t.size(dim) - 1);
    const auto stride = static_cast<std::uintptr_t>(t.stride(dim));
    TORCH_CHECK(
        stride == 0 || extent <= max / stride,
        "tensor address range overflows");
    const auto contribution = extent * stride;
    TORCH_CHECK(
        last_element <= max - contribution, "tensor address range overflows");
    last_element += contribution;
  }
  const auto item_bytes = static_cast<std::uintptr_t>(t.element_size());
  TORCH_CHECK(
      item_bytes > 0 && last_element < max / item_bytes,
      "tensor byte range overflows");
  const auto bytes = (last_element + 1) * item_bytes;
  TORCH_CHECK(begin <= max - bytes, "tensor byte range overflows");
  return {begin, begin + bytes};
}

void check_no_alias(
    const at::Tensor& output,
    std::initializer_list<const at::Tensor*> inputs,
    const char* name) {
  const auto output_range = byte_range(output);
  for (const auto* input : inputs) {
    const auto input_range = byte_range(*input);
    const bool address_overlap = output_range.begin < input_range.end &&
                                 input_range.begin < output_range.end;
    TORCH_CHECK(
        !output.is_alias_of(*input) && !address_overlap,
        name,
        " must not overlap an input or another output storage/address range");
  }
}

void record_stream(
    c10::xpu::XPUStream stream,
    std::initializer_list<const at::Tensor*> tensors) {
  for (const auto* t : tensors) {
    c10::xpu::XPUCachingAllocator::recordStream(
        t->storage().data_ptr(), stream);
  }
}

// The untrusted ABI must prove metadata before it submits a state-changing
// kernel. Only that path performs a host copy. The scheduler-proven trusted
// ABI never copies metadata, synchronizes or reads back an output.
std::vector<int64_t> index_values(const at::Tensor& t) {
  const at::Tensor host = t.to(at::kCPU).contiguous();
  std::vector<int64_t> result(host.numel());
  if (t.scalar_type() == at::kInt) {
    const auto* data = host.data_ptr<int32_t>();
    std::transform(
        data, data + result.size(), result.begin(), [](int32_t value) {
          return static_cast<int64_t>(value);
        });
  } else {
    const auto* data = host.data_ptr<int64_t>();
    std::copy(data, data + result.size(), result.begin());
  }
  return result;
}

void check_offsets(
    const at::Tensor& offsets, int64_t requests, int64_t tokens) {
  TORCH_CHECK(
      offsets.dim() == 1 && offsets.numel() == requests + 1,
      "query_start_loc must be [requests + 1]");
  const auto values = index_values(offsets);
  TORCH_CHECK(
      !values.empty() && values.front() == 0 && values.back() == tokens,
      "query_start_loc endpoints are invalid");
  for (size_t i = 1; i < values.size(); ++i) {
    TORCH_CHECK(
        values[i] >= values[i - 1] && values[i] <= tokens,
        "query_start_loc must be monotonic within token count");
  }
}

std::vector<int64_t> check_unique_slots(
    const at::Tensor& slots, int64_t n, int64_t null_id) {
  TORCH_CHECK(
      null_id < 0 || null_id >= n,
      "untrusted null_block_id must not be a real state slot");
  std::unordered_set<int64_t> seen;
  auto values = index_values(slots);
  for (const auto value : values) {
    if (value == null_id) continue;
    TORCH_CHECK(value >= 0 && value < n, "state index out of bounds");
    TORCH_CHECK(seen.insert(value).second, "duplicate state index");
  }
  return values;
}

void check_trusted_null(int64_t n, int64_t null_id) {
  TORCH_CHECK(
      null_id == 0 || null_id < 0 || null_id >= n,
      "trusted null_block_id must be reserved slot 0 or out of range");
}

bool positive_non_overlapping(const at::Tensor& state) {
  std::array<std::pair<int64_t, int64_t>, 3> dims{};
  for (int d = 0; d < 3; ++d) {
    dims[d] = {state.stride(d), state.size(d)};
    if (dims[d].first <= 0) return false;
  }
  std::sort(dims.begin(), dims.end());
  int64_t span = 1;
  for (const auto& [stride, size] : dims) {
    if (size <= 1) continue;
    if (stride < span ||
        (size - 1) > std::numeric_limits<int64_t>::max() / stride)
      return false;
    const int64_t extent = (size - 1) * stride;
    if (span > std::numeric_limits<int64_t>::max() - extent) return false;
    span += extent;
  }
  return true;
}

int64_t state_length(int64_t kernel_size, int64_t dilation) {
  TORCH_CHECK(
      kernel_size >= 1 && dilation > 0,
      "kernel_size and dilation must be positive");
  TORCH_CHECK(
      kernel_size - 1 <= kMaxConvState / dilation,
      "dilated state length exceeds 128");
  return (kernel_size - 1) * dilation;
}

void check_eps(double eps) {
  const float converted = static_cast<float>(eps);
  TORCH_CHECK(
      std::isfinite(eps) && eps > 0.0 && std::isfinite(converted) &&
          converted > 0.0f,
      "eps must remain finite and positive in FP32");
}

void check_state(
    const at::Tensor& input,
    const at::Tensor& state,
    const at::Tensor& weights,
    const at::Tensor& slots,
    const at::Tensor* initial,
    const at::Tensor& output,
    int64_t required_width,
    bool dim_first) {
  check_xpu(input, "input");
  check_xpu(state, "conv_state");
  TORCH_CHECK(
      input.scalar_type() == at::kHalf && input.dim() == 2 &&
          input.size(1) > 0 && input.is_contiguous(),
      "input must be contiguous float16 [tokens, channels]");
  check_tensor(input, weights, at::kHalf, "conv_weights");
  check_index(input, slots, "state_indices");
  check_tensor(input, output, at::kHalf, "output");
  TORCH_CHECK(
      weights.dim() == 2 && weights.size(0) == input.size(1) &&
          weights.size(1) >= 1 && output.sizes() == input.sizes(),
      "conv_weights/output shapes are invalid");
  TORCH_CHECK(slots.dim() == 1, "state_indices must be rank 1");
  TORCH_CHECK(
      state.dim() == 3 && state.size(0) > 0 &&
          state.device() == input.device() &&
          (state.scalar_type() == at::kHalf ||
           state.scalar_type() == at::kFloat) &&
          positive_non_overlapping(state),
      "conv_state must be non-overlapping float16/float32 "
      "[slots,channels,state] or [slots,state,channels]");
  const bool shape_ok =
      dim_first
          ? state.size(1) == input.size(1) && state.size(2) >= required_width
          : state.size(2) == input.size(1) && state.size(1) >= required_width;
  TORCH_CHECK(shape_ok, "conv_state shape/layout is invalid");
  if (initial) {
    check_tensor(input, *initial, at::kBool, "has_initial_state");
    TORCH_CHECK(
        initial->dim() == 1 && initial->numel() == slots.numel(),
        "has_initial_state must be [requests]");
    check_no_alias(state, {initial}, "conv_state");
    check_no_alias(output, {initial}, "output");
  }
  check_no_alias(state, {&input, &weights, &slots, &output}, "conv_state");
  check_no_alias(output, {&input, &weights, &slots}, "output");
}

inline float silu(float x) { return x / (1.0f + sycl::exp(-x)); }
inline float sigmoid(float x) { return 1.0f / (1.0f + sycl::exp(-x)); }

inline int64_t state_offset(
    int64_t slot,
    int64_t channel,
    int64_t position,
    int64_t s0,
    int64_t s1,
    int64_t s2,
    bool dim_first) {
  return dim_first ? slot * s0 + channel * s1 + position * s2
                   : slot * s0 + position * s1 + channel * s2;
}

template <typename State, typename Index>
void launch_decode(
    sycl::queue& queue,
    const half* input,
    State* state,
    const half* weight,
    const Index* slots,
    const bool* initial,
    half* output,
    int64_t rows,
    int64_t width,
    int64_t ksize,
    int64_t dilation,
    int64_t nslots,
    int64_t s0,
    int64_t s1,
    int64_t s2,
    bool dim_first,
    int64_t null_id) {
  const int64_t len = (ksize - 1) * dilation;
  queue.parallel_for(sycl::range<1>(rows * width), [=](sycl::id<1> id) {
    const int64_t linear = id[0];
    const int64_t row = linear / width;
    const int64_t channel = linear % width;
    const int64_t slot = slots[row];
    if (slot == null_id || slot < 0 || slot >= nslots) {
      output[linear] = half(0.0f);
      return;
    }
    float history[kMaxConvState + 1];
    for (int64_t p = 0; p < len; ++p) {
      const auto offset = state_offset(slot, channel, p, s0, s1, s2, dim_first);
      history[p] = initial[row] ? static_cast<float>(state[offset]) : 0.0f;
    }
    history[len] = static_cast<float>(input[linear]);
    float sum = 0.0f;
    for (int64_t k = 0; k < ksize; ++k)
      sum += static_cast<float>(weight[channel * ksize + k]) *
             history[k * dilation];
    output[linear] = half(silu(sum));
    for (int64_t p = 0; p < len; ++p)
      state[state_offset(slot, channel, p, s0, s1, s2, dim_first)] =
          static_cast<State>(history[p + 1]);
  });
}

template <typename State, typename Index, typename Offset>
void launch_prefill(
    sycl::queue& queue,
    const half* input,
    const Offset* starts,
    State* state,
    const half* weight,
    const Index* slots,
    const bool* initial,
    half* output,
    int64_t requests,
    int64_t width,
    int64_t ksize,
    int64_t dilation,
    int64_t nslots,
    int64_t s0,
    int64_t s1,
    int64_t s2,
    bool dim_first,
    int64_t null_id) {
  const int64_t len = (ksize - 1) * dilation;
  queue.parallel_for(sycl::range<1>(requests * width), [=](sycl::id<1> id) {
    const int64_t linear = id[0];
    const int64_t request = linear / width;
    const int64_t channel = linear % width;
    const int64_t begin = starts[request];
    const int64_t end = starts[request + 1];
    const int64_t slot = slots[request];
    if (slot == null_id || slot < 0 || slot >= nslots) {
      for (int64_t t = begin; t < end; ++t)
        output[t * width + channel] = half(0.0f);
      return;
    }
    if (begin == end) return;
    float history[kMaxConvState + 1];
    for (int64_t p = 0; p < len; ++p)
      history[p] = initial[request]
                       ? static_cast<float>(state[state_offset(
                             slot, channel, p, s0, s1, s2, dim_first)])
                       : 0.0f;
    for (int64_t t = begin; t < end; ++t) {
      history[len] = static_cast<float>(input[t * width + channel]);
      float sum = 0.0f;
      for (int64_t k = 0; k < ksize; ++k)
        sum += static_cast<float>(weight[channel * ksize + k]) *
               history[k * dilation];
      output[t * width + channel] = half(silu(sum));
      for (int64_t p = 0; p < len; ++p)
        history[p] = history[p + 1];
    }
    for (int64_t p = 0; p < len; ++p)
      state[state_offset(slot, channel, p, s0, s1, s2, dim_first)] =
          static_cast<State>(history[p]);
  });
}

template <typename State, typename Index, typename Offset, typename Accepted>
void launch_spec(
    sycl::queue& queue,
    const half* input,
    const Offset* starts,
    State* state,
    const half* weight,
    const Index* slots,
    const Accepted* accepted,
    half* output,
    int64_t requests,
    int64_t width,
    int64_t ksize,
    int64_t num_spec,
    int64_t dilation,
    int64_t nslots,
    int64_t s0,
    int64_t s1,
    int64_t s2,
    bool dim_first,
    int64_t null_id) {
  const int64_t len = (ksize - 1) * dilation;
  const int64_t capacity = len + num_spec;
  queue.parallel_for(sycl::range<1>(requests * width), [=](sycl::id<1> id) {
    const int64_t linear = id[0];
    const int64_t request = linear / width;
    const int64_t channel = linear % width;
    const int64_t begin = starts[request];
    const int64_t end = starts[request + 1];
    const int64_t slot = slots[request];
    if (slot == null_id || slot < 0 || slot >= nslots) {
      for (int64_t t = begin; t < end; ++t)
        output[t * width + channel] = half(0.0f);
      return;
    }
    if (begin == end) return;
    int64_t rollback = accepted[request] - 1;
    rollback = sycl::clamp(rollback, int64_t(0), num_spec);
    float history[kMaxConvState + 1];
    float extended[kMaxConvState + kMaxSpecTokens + 1];
    for (int64_t p = 0; p < len; ++p) {
      history[p] = static_cast<float>(state[state_offset(
          slot, channel, rollback + p, s0, s1, s2, dim_first)]);
      extended[p] = history[p];
    }
    int64_t count = 0;
    for (int64_t t = begin; t < end; ++t) {
      history[len] = static_cast<float>(input[t * width + channel]);
      float sum = 0.0f;
      for (int64_t k = 0; k < ksize; ++k)
        sum += static_cast<float>(weight[channel * ksize + k]) *
               history[k * dilation];
      output[t * width + channel] = half(silu(sum));
      if (len + count < kMaxConvState + kMaxSpecTokens + 1)
        extended[len + count] = history[len];
      ++count;
      for (int64_t p = 0; p < len; ++p)
        history[p] = history[p + 1];
    }
    if (len > 0) {
      int64_t keep = sycl::clamp(len + count - 1, int64_t(0), capacity);
      for (int64_t p = 0; p < keep; ++p)
        state[state_offset(slot, channel, p, s0, s1, s2, dim_first)] =
            static_cast<State>(extended[p + 1]);
    }
  });
}

}  // namespace

#ifdef QWEN38_PLE_SYCL_STANDALONE_TEST
bool test_no_address_alias(const at::Tensor& output, const at::Tensor& input) {
  check_no_alias(output, {&input}, "output");
  return true;
}
#endif

at::Tensor ngram_ids(
    at::Tensor input_ids,
    at::Tensor query_start_loc,
    at::Tensor ngram_context,
    at::Tensor layer_multipliers,
    at::Tensor vocab_sizes,
    at::Tensor offsets,
    at::Tensor output,
    int64_t eos_token_id,
    int64_t heads_per_ngram) {
  check_xpu(input_ids, "input_ids");
  check_tensor(input_ids, input_ids, at::kLong, "input_ids");
  check_tensor(input_ids, query_start_loc, at::kLong, "query_start_loc");
  check_tensor(input_ids, ngram_context, at::kLong, "ngram_context");
  check_tensor(input_ids, layer_multipliers, at::kLong, "layer_multipliers");
  check_tensor(input_ids, vocab_sizes, at::kLong, "vocab_sizes");
  check_tensor(input_ids, offsets, at::kLong, "offsets");
  check_tensor(input_ids, output, at::kLong, "output");
  TORCH_CHECK(
      input_ids.dim() == 1 && query_start_loc.dim() == 1 &&
          query_start_loc.numel() >= 1 && layer_multipliers.dim() == 1 &&
          layer_multipliers.numel() >= 2 && heads_per_ngram > 0,
      "invalid N-gram metadata dimensions");
  const int64_t requests = query_start_loc.numel() - 1;
  const int64_t context_width = layer_multipliers.numel() - 1;
  const int64_t heads = context_width * heads_per_ngram;
  TORCH_CHECK(
      ngram_context.sizes() == at::IntArrayRef({requests, context_width}) &&
          vocab_sizes.dim() == 1 && vocab_sizes.numel() == heads &&
          offsets.dim() == 1 && offsets.numel() == heads &&
          output.sizes() == at::IntArrayRef({input_ids.numel(), heads}),
      "invalid N-gram metadata/output shape");
  check_offsets(query_start_loc, requests, input_ids.numel());
  for (const auto size : index_values(vocab_sizes))
    TORCH_CHECK(size > 0, "N-gram vocab sizes must be positive");
  check_no_alias(
      output,
      {&input_ids,
       &query_start_loc,
       &ngram_context,
       &layer_multipliers,
       &vocab_sizes,
       &offsets},
      "output");
  if (input_ids.numel() == 0) return output;
  const c10::OptionalDeviceGuard guard(input_ids.device());
  auto stream = c10::xpu::getCurrentXPUStream(input_ids.device().index());
  auto& queue = stream.queue();
  const auto* ids = input_ids.data_ptr<int64_t>();
  const auto* starts = query_start_loc.data_ptr<int64_t>();
  const auto* context = ngram_context.data_ptr<int64_t>();
  const auto* multiplier = layer_multipliers.data_ptr<int64_t>();
  const auto* moduli = vocab_sizes.data_ptr<int64_t>();
  const auto* head_offsets = offsets.data_ptr<int64_t>();
  auto* out = output.data_ptr<int64_t>();
  const int64_t tokens = input_ids.numel();
  record_stream(
      stream,
      {&input_ids,
       &query_start_loc,
       &ngram_context,
       &layer_multipliers,
       &vocab_sizes,
       &offsets,
       &output});
  queue.parallel_for(sycl::range<1>(tokens * heads), [=](sycl::id<1> id) {
    const int64_t token = id[0] / heads;
    const int64_t head = id[0] % heads;
    int64_t request = requests - 1;
    for (int64_t r = 0; r < requests; ++r) {
      if (token < starts[r + 1]) {
        request = r;
        break;
      }
    }
    const int64_t local = token - starts[request];
    const int64_t position = context_width + local;
    const int64_t ngram = head / heads_per_ngram + 2;
    const int64_t current = ids[token];
    uint64_t mixed =
        static_cast<uint64_t>(current) * static_cast<uint64_t>(multiplier[0]);
    bool after_eos = current == eos_token_id;
    for (int64_t shift = 1; shift < ngram; ++shift) {
      int64_t prior = eos_token_id;
      const int64_t predecessor = position - shift;
      if (!after_eos && predecessor >= 0) {
        prior = predecessor < context_width
                    ? context[request * context_width + predecessor]
                    : ids[starts[request] + predecessor - context_width];
        after_eos = prior == eos_token_id;
      }
      mixed ^= static_cast<uint64_t>(prior) *
               static_cast<uint64_t>(multiplier[shift]);
    }
    const int64_t modulus = moduli[head];
    int64_t remainder = static_cast<int64_t>(mixed) % modulus;
    if (remainder < 0) remainder += modulus;
    out[token * heads + head] = remainder + head_offsets[head];
  });
  return output;
}

at::Tensor embedding_gather(
    at::Tensor ids,
    at::Tensor weight,
    at::Tensor local_start,
    at::Tensor local_rows,
    at::Tensor output) {
  check_xpu(ids, "ngram_ids");
  check_tensor(ids, ids, at::kLong, "ngram_ids");
  check_tensor(ids, weight, at::kHalf, "local_weight");
  check_tensor(ids, local_start, at::kLong, "local_vocab_start");
  check_tensor(ids, local_rows, at::kLong, "local_num_rows");
  check_tensor(ids, output, at::kHalf, "local_partial");
  TORCH_CHECK(
      ids.dim() == 2 && weight.dim() == 2 && local_start.numel() == 1 &&
          local_rows.numel() == 1 && output.dim() == 2 &&
          output.size(0) == ids.size(0) &&
          output.size(1) == ids.size(1) * weight.size(1),
      "embedding gather shapes are invalid");
  const int64_t start = index_values(local_start).front();
  const int64_t rows = index_values(local_rows).front();
  TORCH_CHECK(
      start >= 0 && rows >= 0 && rows <= weight.size(0) &&
          start <= std::numeric_limits<int64_t>::max() - rows,
      "embedding shard metadata is invalid");
  check_no_alias(
      output, {&ids, &weight, &local_start, &local_rows}, "local_partial");
  if (output.numel() == 0) return output;
  const c10::OptionalDeviceGuard guard(ids.device());
  auto stream = c10::xpu::getCurrentXPUStream(ids.device().index());
  auto& queue = stream.queue();
  const auto* id_ptr = ids.data_ptr<int64_t>();
  const auto* w = reinterpret_cast<const half*>(weight.data_ptr());
  const auto* start_ptr = local_start.data_ptr<int64_t>();
  const auto* rows_ptr = local_rows.data_ptr<int64_t>();
  auto* out = reinterpret_cast<half*>(output.data_ptr());
  const int64_t heads = ids.size(1);
  const int64_t width = weight.size(1);
  record_stream(stream, {&ids, &weight, &local_start, &local_rows, &output});
  queue.parallel_for(sycl::range<1>(output.numel()), [=](sycl::id<1> id) {
    const int64_t linear = id[0];
    const int64_t head = (linear / width) % heads;
    const int64_t token = linear / (heads * width);
    const int64_t row = id_ptr[token * heads + head] - start_ptr[0];
    out[linear] = row >= 0 && row < rows_ptr[0]
                      ? w[row * width + linear % width]
                      : half(0.0f);
  });
  return output;
}

at::Tensor grouped_norm(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor output,
    double eps,
    int64_t group_size) {
  check_xpu(input, "input");
  check_tensor(input, input, at::kHalf, "input");
  check_tensor(input, weight, at::kHalf, "weight");
  check_tensor(input, output, at::kHalf, "output");
  TORCH_CHECK(
      input.dim() >= 1 && weight.dim() == 1 &&
          weight.numel() == input.size(-1) && output.sizes() == input.sizes() &&
          group_size > 0 && input.size(-1) > 0 &&
          input.size(-1) % group_size == 0,
      "grouped_norm dimensions are invalid");
  check_eps(eps);
  check_no_alias(output, {&input, &weight}, "output");
  const int64_t groups = input.numel() / group_size;
  if (groups == 0) return output;
  const c10::OptionalDeviceGuard guard(input.device());
  auto stream = c10::xpu::getCurrentXPUStream(input.device().index());
  auto& queue = stream.queue();
  const auto* x = reinterpret_cast<const half*>(input.data_ptr());
  const auto* w = reinterpret_cast<const half*>(weight.data_ptr());
  auto* y = reinterpret_cast<half*>(output.data_ptr());
  const int64_t width = input.size(-1);
  const float epsilon = static_cast<float>(eps);
  record_stream(stream, {&input, &weight, &output});
  queue.parallel_for(
      sycl::nd_range<1>(groups * kReduceLocal, kReduceLocal),
      [=](sycl::nd_item<1> item) {
        const int64_t group = item.get_group_linear_id();
        const int64_t lane = item.get_local_linear_id();
        const int64_t row_begin = group / (width / group_size) * width;
        const int64_t within = group % (width / group_size) * group_size;
        float square = 0.0f;
        for (int64_t j = lane; j < group_size; j += kReduceLocal) {
          const float value = static_cast<float>(x[row_begin + within + j]);
          square += value * value;
        }
        square = sycl::reduce_over_group(
            item.get_group(), square, sycl::plus<float>());
        const float inverse = sycl::rsqrt(square / group_size + epsilon);
        for (int64_t j = lane; j < group_size; j += kReduceLocal) {
          const auto pos = row_begin + within + j;
          y[pos] = half(
              static_cast<float>(x[pos]) * inverse *
              (1.0f + static_cast<float>(w[within + j])));
        }
      });
  return output;
}

at::Tensor score_gate(
    at::Tensor key, at::Tensor query, at::Tensor output, int64_t hidden_size) {
  check_xpu(key, "key");
  check_tensor(key, key, at::kHalf, "key");
  check_tensor(key, query, at::kHalf, "query");
  check_tensor(key, output, at::kHalf, "output");
  TORCH_CHECK(
      key.dim() >= 1 && query.sizes() == key.sizes() && hidden_size > 0 &&
          key.size(-1) > 0 && key.size(-1) % hidden_size == 0 &&
          output.numel() == key.numel() / hidden_size,
      "score_gate dimensions are invalid");
  check_no_alias(output, {&key, &query}, "output");
  const int64_t groups = output.numel();
  if (groups == 0) return output;
  const c10::OptionalDeviceGuard guard(key.device());
  auto stream = c10::xpu::getCurrentXPUStream(key.device().index());
  auto& queue = stream.queue();
  const auto* a = reinterpret_cast<const half*>(key.data_ptr());
  const auto* b = reinterpret_cast<const half*>(query.data_ptr());
  auto* y = reinterpret_cast<half*>(output.data_ptr());
  record_stream(stream, {&key, &query, &output});
  queue.parallel_for(
      sycl::nd_range<1>(groups * kReduceLocal, kReduceLocal),
      [=](sycl::nd_item<1> item) {
        const int64_t group = item.get_group_linear_id();
        const int64_t lane = item.get_local_linear_id();
        float dot = 0.0f;
        for (int64_t j = lane; j < hidden_size; j += kReduceLocal)
          dot += static_cast<float>(a[group * hidden_size + j]) *
                 static_cast<float>(b[group * hidden_size + j]);
        dot =
            sycl::reduce_over_group(item.get_group(), dot, sycl::plus<float>());
        if (lane == 0) {
          dot /= sycl::sqrt(static_cast<float>(hidden_size));
          const float signed_root =
              dot == 0.0f ? 0.0f
                          : (dot < 0.0f ? -1.0f : 1.0f) *
                                sycl::sqrt(sycl::fmax(sycl::fabs(dot), 1e-6f));
          y[group] = half(sigmoid(signed_root));
        }
      });
  return output;
}

at::Tensor gated_value(
    at::Tensor gate, at::Tensor value, at::Tensor output, int64_t hc_count) {
  check_xpu(gate, "gate");
  check_tensor(gate, gate, at::kHalf, "gate");
  check_tensor(gate, value, at::kHalf, "value");
  check_tensor(gate, output, at::kHalf, "output");
  TORCH_CHECK(
      gate.dim() >= 1 && value.dim() >= 1 && hc_count > 0 &&
          value.size(-1) > 0 &&
          gate.numel() == (value.numel() / value.size(-1)) * hc_count &&
          output.numel() == gate.numel() * value.size(-1),
      "gated_value dimensions are invalid");
  check_no_alias(output, {&gate, &value}, "output");
  if (output.numel() == 0) return output;
  const c10::OptionalDeviceGuard guard(gate.device());
  auto stream = c10::xpu::getCurrentXPUStream(gate.device().index());
  auto& queue = stream.queue();
  const auto* g = reinterpret_cast<const half*>(gate.data_ptr());
  const auto* x = reinterpret_cast<const half*>(value.data_ptr());
  auto* y = reinterpret_cast<half*>(output.data_ptr());
  const int64_t width = value.size(-1);
  record_stream(stream, {&gate, &value, &output});
  queue.parallel_for(sycl::range<1>(output.numel()), [=](sycl::id<1> id) {
    const int64_t linear = id[0];
    const int64_t group = linear / width;
    const int64_t row = group / hc_count;
    y[linear] = half(
        static_cast<float>(g[group]) *
        static_cast<float>(x[row * width + linear % width]));
  });
  return output;
}

at::Tensor gated_value_grouped_norm(
    at::Tensor gate,
    at::Tensor value,
    at::Tensor weight,
    at::Tensor raw_output,
    at::Tensor normalized_output,
    double eps) {
  check_xpu(gate, "gate");
  check_tensor(gate, gate, at::kHalf, "gate");
  check_tensor(gate, value, at::kHalf, "value");
  check_tensor(gate, weight, at::kHalf, "weight");
  check_tensor(gate, raw_output, at::kHalf, "raw_output");
  check_tensor(gate, normalized_output, at::kHalf, "normalized_output");
  TORCH_CHECK(
      gate.sizes() == at::IntArrayRef({1, kHcCount}) &&
          value.sizes() == at::IntArrayRef({1, kHcHidden}) &&
          weight.sizes() == at::IntArrayRef({kHcWidth}) &&
          raw_output.sizes() == at::IntArrayRef({1, kHcWidth}) &&
          normalized_output.sizes() == raw_output.sizes(),
      "fused gated-value/grouped-norm expects [1,4], [1,2560], [10240], "
      "[1,10240]");
  check_eps(eps);
  check_no_alias(
      raw_output, {&gate, &value, &weight, &normalized_output}, "raw_output");
  check_no_alias(
      normalized_output, {&gate, &value, &weight}, "normalized_output");
  const c10::OptionalDeviceGuard guard(gate.device());
  auto stream = c10::xpu::getCurrentXPUStream(gate.device().index());
  auto& queue = stream.queue();
  const auto* g = reinterpret_cast<const half*>(gate.data_ptr());
  const auto* x = reinterpret_cast<const half*>(value.data_ptr());
  const auto* w = reinterpret_cast<const half*>(weight.data_ptr());
  auto* raw = reinterpret_cast<half*>(raw_output.data_ptr());
  auto* norm = reinterpret_cast<half*>(normalized_output.data_ptr());
  const float epsilon = static_cast<float>(eps);
  record_stream(
      stream, {&gate, &value, &weight, &raw_output, &normalized_output});
  queue.parallel_for(
      sycl::nd_range<1>(kHcCount * kFusedLocal, kFusedLocal),
      [=](sycl::nd_item<1> item) {
        const int64_t group = item.get_group_linear_id();
        const int64_t lane = item.get_local_linear_id();
        const float gate_value = static_cast<float>(g[group]);
        float square = 0.0f;
        half rounded_values[kHcHidden / kFusedLocal];
        for (int64_t j = lane; j < kHcHidden; j += kFusedLocal) {
          // The intermediate is physically FP16 in the original two-op path.
          const half rounded = half(gate_value * static_cast<float>(x[j]));
          rounded_values[j / kFusedLocal] = rounded;
          raw[group * kHcHidden + j] = rounded;
          const float value_float = static_cast<float>(rounded);
          square += value_float * value_float;
        }
        square = sycl::reduce_over_group(
            item.get_group(), square, sycl::plus<float>());
        const float inverse = sycl::rsqrt(square / kHcHidden + epsilon);
        for (int64_t j = lane; j < kHcHidden; j += kFusedLocal) {
          const float rounded =
              static_cast<float>(rounded_values[j / kFusedLocal]);
          norm[group * kHcHidden + j] = half(
              rounded * inverse *
              (1.0f + static_cast<float>(w[group * kHcHidden + j])));
        }
      });
  return raw_output;
}

at::Tensor
residual_add(at::Tensor first, at::Tensor second, at::Tensor output) {
  check_xpu(first, "first");
  check_tensor(first, first, at::kHalf, "first");
  check_tensor(first, second, at::kHalf, "second");
  check_tensor(first, output, at::kHalf, "output");
  TORCH_CHECK(
      first.sizes() == second.sizes() && first.sizes() == output.sizes(),
      "residual_add tensors must have identical shapes");
  check_no_alias(output, {&first, &second}, "output");
  if (output.numel() == 0) return output;
  const c10::OptionalDeviceGuard guard(first.device());
  auto stream = c10::xpu::getCurrentXPUStream(first.device().index());
  auto& queue = stream.queue();
  const auto* a = reinterpret_cast<const half*>(first.data_ptr());
  const auto* b = reinterpret_cast<const half*>(second.data_ptr());
  auto* y = reinterpret_cast<half*>(output.data_ptr());
  record_stream(stream, {&first, &second, &output});
  queue.parallel_for(sycl::range<1>(output.numel()), [=](sycl::id<1> id) {
    y[id[0]] =
        half(static_cast<float>(a[id[0]]) + static_cast<float>(b[id[0]]));
  });
  return output;
}

namespace {

enum class ConvMode { Decode, Prefill, Spec };

template <typename State, typename Index, typename Offset, typename Accepted>
void submit_conv(
    sycl::queue& queue,
    ConvMode mode,
    const at::Tensor& input,
    const at::Tensor& starts,
    const at::Tensor& state,
    const at::Tensor& weights,
    const at::Tensor& slots,
    const at::Tensor& initial,
    const at::Tensor& accepted,
    const at::Tensor& output,
    int64_t spec_tokens,
    int64_t dilation,
    bool dim_first,
    int64_t null_id) {
  const auto* x = reinterpret_cast<const half*>(input.data_ptr());
  auto* cache = reinterpret_cast<State*>(state.data_ptr());
  const auto* w = reinterpret_cast<const half*>(weights.data_ptr());
  const auto* index = slots.data_ptr<Index>();
  auto* y = reinterpret_cast<half*>(output.data_ptr());
  const int64_t width = input.size(1);
  const int64_t ksize = weights.size(1);
  const int64_t nslots = state.size(0);
  const int64_t s0 = state.stride(0);
  const int64_t s1 = state.stride(1);
  const int64_t s2 = state.stride(2);
  if (mode == ConvMode::Decode) {
    launch_decode<State, Index>(
        queue,
        x,
        cache,
        w,
        index,
        initial.data_ptr<bool>(),
        y,
        input.size(0),
        width,
        ksize,
        dilation,
        nslots,
        s0,
        s1,
        s2,
        dim_first,
        null_id);
  } else if (mode == ConvMode::Prefill) {
    launch_prefill<State, Index, Offset>(
        queue,
        x,
        starts.data_ptr<Offset>(),
        cache,
        w,
        index,
        initial.data_ptr<bool>(),
        y,
        slots.numel(),
        width,
        ksize,
        dilation,
        nslots,
        s0,
        s1,
        s2,
        dim_first,
        null_id);
  } else {
    launch_spec<State, Index, Offset, Accepted>(
        queue,
        x,
        starts.data_ptr<Offset>(),
        cache,
        w,
        index,
        accepted.data_ptr<Accepted>(),
        y,
        slots.numel(),
        width,
        ksize,
        spec_tokens,
        dilation,
        nslots,
        s0,
        s1,
        s2,
        dim_first,
        null_id);
  }
}

at::Tensor short_conv_impl(
    ConvMode mode,
    at::Tensor input,
    at::Tensor starts,
    at::Tensor state,
    at::Tensor weights,
    at::Tensor slots,
    at::Tensor initial,
    at::Tensor accepted,
    at::Tensor output,
    int64_t spec_tokens,
    int64_t dilation,
    bool dim_first,
    int64_t null_id,
    bool trusted) {
  TORCH_CHECK(weights.dim() == 2, "conv_weights must be rank 2");
  const int64_t len = state_length(weights.size(1), dilation);
  if (mode == ConvMode::Spec) {
    TORCH_CHECK(
        spec_tokens >= 0 && spec_tokens <= kMaxSpecTokens,
        "num_spec_tokens exceeds implementation limit");
  }
  const int64_t required =
      len + (mode == ConvMode::Spec ? spec_tokens : int64_t(0));
  const at::Tensor* initial_ptr = mode == ConvMode::Spec ? nullptr : &initial;
  check_state(
      input, state, weights, slots, initial_ptr, output, required, dim_first);
  const int64_t requests = slots.numel();
  if (mode == ConvMode::Decode) {
    TORCH_CHECK(
        requests == input.size(0),
        "decode state indices must match input rows");
  } else {
    check_index(input, starts, "query_start_loc");
    TORCH_CHECK(
        starts.dim() == 1 && starts.numel() == requests + 1,
        "query_start_loc must be [requests + 1]");
    check_no_alias(state, {&starts}, "conv_state");
    check_no_alias(output, {&starts}, "output");
    if (mode == ConvMode::Spec) {
      check_index(input, accepted, "num_accepted_tokens");
      TORCH_CHECK(
          accepted.dim() == 1 && accepted.numel() == requests,
          "num_accepted_tokens must be [requests]");
      check_no_alias(state, {&accepted}, "conv_state");
      check_no_alias(output, {&accepted}, "output");
      check_no_alias(accepted, {&starts}, "num_accepted_tokens");
    }
  }
  if (trusted) {
    // The caller must carry the scheduler's construction proof for the
    // *values* of starts, accepted and slots. Shape, dtype, capacity and
    // storage alias checks above remain unconditional.
    check_trusted_null(state.size(0), null_id);
  } else {
    const auto slot_values = check_unique_slots(slots, state.size(0), null_id);
    if (mode != ConvMode::Decode) {
      check_offsets(starts, requests, input.size(0));
      if (mode == ConvMode::Spec) {
        const auto boundaries = index_values(starts);
        const auto counts = index_values(accepted);
        for (int64_t r = 0; r < requests; ++r) {
          const int64_t query_length = boundaries[r + 1] - boundaries[r];
          TORCH_CHECK(
              query_length <= spec_tokens + 1,
              "spec query length exceeds num_spec_tokens + 1");
          // accepted describes the previous query, not this query's length.
          // Null/empty requests return before reading it in launch_spec.
          if (slot_values[r] != null_id && query_length > 0) {
            TORCH_CHECK(
                counts[r] >= 1 && counts[r] <= spec_tokens + 1,
                "num_accepted_tokens outside valid rollback capacity");
          }
        }
      }
    }
  }
  // No output or cache update has occurred before this point. In particular,
  // a rejected speculative request cannot partially write rollback state.
  if (input.size(0) == 0 || requests == 0) return output;
  const c10::OptionalDeviceGuard guard(input.device());
  auto stream = c10::xpu::getCurrentXPUStream(input.device().index());
  auto& queue = stream.queue();
  // Record every owner before submission, including exceptions from submit.
  record_stream(stream, {&input, &state, &weights, &slots, &output});
  if (mode != ConvMode::Spec) record_stream(stream, {&initial});
  if (mode != ConvMode::Decode) record_stream(stream, {&starts});
  if (mode == ConvMode::Spec) record_stream(stream, {&accepted});
  auto run =
      [&](auto state_tag, auto index_tag, auto offset_tag, auto accepted_tag) {
        using State = decltype(state_tag);
        using Index = decltype(index_tag);
        using Offset = decltype(offset_tag);
        using Accepted = decltype(accepted_tag);
        submit_conv<State, Index, Offset, Accepted>(
            queue,
            mode,
            input,
            starts,
            state,
            weights,
            slots,
            initial,
            accepted,
            output,
            spec_tokens,
            dilation,
            dim_first,
            null_id);
      };
  auto run_indices = [&](auto state_tag, auto index_tag) {
    if (mode == ConvMode::Decode) {
      run(state_tag, index_tag, int32_t{}, int32_t{});
    } else if (mode == ConvMode::Prefill) {
      if (starts.scalar_type() == at::kInt)
        run(state_tag, index_tag, int32_t{}, int32_t{});
      else
        run(state_tag, index_tag, int64_t{}, int32_t{});
    } else if (starts.scalar_type() == at::kInt) {
      if (accepted.scalar_type() == at::kInt)
        run(state_tag, index_tag, int32_t{}, int32_t{});
      else
        run(state_tag, index_tag, int32_t{}, int64_t{});
    } else if (accepted.scalar_type() == at::kInt) {
      run(state_tag, index_tag, int64_t{}, int32_t{});
    } else {
      run(state_tag, index_tag, int64_t{}, int64_t{});
    }
  };
  if (state.scalar_type() == at::kHalf) {
    if (slots.scalar_type() == at::kInt)
      run_indices(half{}, int32_t{});
    else
      run_indices(half{}, int64_t{});
  } else if (slots.scalar_type() == at::kInt) {
    run_indices(float{}, int32_t{});
  } else {
    run_indices(float{}, int64_t{});
  }
  return output;
}

}  // namespace

at::Tensor short_conv_decode(
    at::Tensor input,
    at::Tensor state,
    at::Tensor weights,
    at::Tensor slots,
    at::Tensor initial,
    at::Tensor output,
    int64_t dilation,
    bool dim_first,
    int64_t null_id) {
  return short_conv_impl(
      ConvMode::Decode,
      input,
      {},
      state,
      weights,
      slots,
      initial,
      {},
      output,
      0,
      dilation,
      dim_first,
      null_id,
      false);
}

at::Tensor short_conv_decode_trusted(
    at::Tensor input,
    at::Tensor state,
    at::Tensor weights,
    at::Tensor slots,
    at::Tensor initial,
    at::Tensor output,
    int64_t dilation,
    bool dim_first,
    int64_t null_id) {
  return short_conv_impl(
      ConvMode::Decode,
      input,
      {},
      state,
      weights,
      slots,
      initial,
      {},
      output,
      0,
      dilation,
      dim_first,
      null_id,
      true);
}

at::Tensor short_conv_prefill(
    at::Tensor input,
    at::Tensor starts,
    at::Tensor state,
    at::Tensor weights,
    at::Tensor slots,
    at::Tensor initial,
    at::Tensor output,
    int64_t dilation,
    bool dim_first,
    int64_t null_id) {
  return short_conv_impl(
      ConvMode::Prefill,
      input,
      starts,
      state,
      weights,
      slots,
      initial,
      {},
      output,
      0,
      dilation,
      dim_first,
      null_id,
      false);
}

at::Tensor short_conv_prefill_trusted(
    at::Tensor input,
    at::Tensor starts,
    at::Tensor state,
    at::Tensor weights,
    at::Tensor slots,
    at::Tensor initial,
    at::Tensor output,
    int64_t dilation,
    bool dim_first,
    int64_t null_id) {
  return short_conv_impl(
      ConvMode::Prefill,
      input,
      starts,
      state,
      weights,
      slots,
      initial,
      {},
      output,
      0,
      dilation,
      dim_first,
      null_id,
      true);
}

at::Tensor short_conv_spec(
    at::Tensor input,
    at::Tensor starts,
    at::Tensor state,
    at::Tensor weights,
    at::Tensor slots,
    at::Tensor accepted,
    at::Tensor output,
    int64_t spec_tokens,
    int64_t dilation,
    bool dim_first,
    int64_t null_id) {
  return short_conv_impl(
      ConvMode::Spec,
      input,
      starts,
      state,
      weights,
      slots,
      {},
      accepted,
      output,
      spec_tokens,
      dilation,
      dim_first,
      null_id,
      false);
}

at::Tensor short_conv_spec_trusted(
    at::Tensor input,
    at::Tensor starts,
    at::Tensor state,
    at::Tensor weights,
    at::Tensor slots,
    at::Tensor accepted,
    at::Tensor output,
    int64_t spec_tokens,
    int64_t dilation,
    bool dim_first,
    int64_t null_id) {
  return short_conv_impl(
      ConvMode::Spec,
      input,
      starts,
      state,
      weights,
      slots,
      {},
      accepted,
      output,
      spec_tokens,
      dilation,
      dim_first,
      null_id,
      true);
}

}  // namespace vllm::qwen38::ple_sycl

// Isolated test-only registration. Production registration belongs to the
// main agent's central bindings; this namespace cannot collide with ESIMD.
#ifdef QWEN38_PLE_SYCL_STANDALONE_TEST
  #include <torch/library.h>
TORCH_LIBRARY(_qwen38_ple_sycl_test, m) {
  namespace ple = vllm::qwen38::ple_sycl;
  m.def("ple_test_no_address_alias(Tensor output, Tensor input) -> bool");
  m.impl("ple_test_no_address_alias", at::kCPU, &ple::test_no_address_alias);
  m.def(
      "ple_ngram_ids(Tensor input_ids, Tensor query_start_loc, "
      "Tensor ngram_context, Tensor layer_multipliers, "
      "Tensor ngram_heads_vocab_sizes, Tensor ngram_heads_offsets, "
      "Tensor(a!) output, int eos_token_id, int heads_per_ngram) -> ()");
  m.impl(
      "ple_ngram_ids",
      at::kXPU,
      [](at::Tensor a,
         at::Tensor b,
         at::Tensor c,
         at::Tensor d,
         at::Tensor e,
         at::Tensor f,
         at::Tensor g,
         int64_t h,
         int64_t i) -> void { ple::ngram_ids(a, b, c, d, e, f, g, h, i); });
  m.def(
      "ple_embedding_gather(Tensor ngram_ids, Tensor local_weight, "
      "Tensor local_vocab_start, Tensor local_num_rows, "
      "Tensor(a!) local_partial) -> ()");
  m.impl(
      "ple_embedding_gather",
      at::kXPU,
      [](at::Tensor a, at::Tensor b, at::Tensor c, at::Tensor d, at::Tensor e)
          -> void { ple::embedding_gather(a, b, c, d, e); });
  m.def(
      "ple_grouped_norm(Tensor input, Tensor weight, Tensor(a!) output, "
      "float eps, int group_size) -> ()");
  m.impl(
      "ple_grouped_norm",
      at::kXPU,
      [](at::Tensor a, at::Tensor b, at::Tensor c, double d, int64_t e)
          -> void { ple::grouped_norm(a, b, c, d, e); });
  m.def(
      "ple_score_gate(Tensor key, Tensor query, Tensor(a!) output, "
      "int hidden_size) -> ()");
  m.impl(
      "ple_score_gate",
      at::kXPU,
      [](at::Tensor a, at::Tensor b, at::Tensor c, int64_t d) -> void {
        ple::score_gate(a, b, c, d);
      });
  m.def(
      "ple_gated_value(Tensor gate, Tensor value, Tensor(a!) output, "
      "int hc_count) -> ()");
  m.impl(
      "ple_gated_value",
      at::kXPU,
      [](at::Tensor a, at::Tensor b, at::Tensor c, int64_t d) -> void {
        ple::gated_value(a, b, c, d);
      });
  m.def(
      "ple_gated_value_grouped_norm(Tensor gate, Tensor value, Tensor weight, "
      "Tensor(a!) raw_output, Tensor(b!) normalized_output, float eps) -> ()");
  m.impl(
      "ple_gated_value_grouped_norm",
      at::kXPU,
      [](at::Tensor a,
         at::Tensor b,
         at::Tensor c,
         at::Tensor d,
         at::Tensor e,
         double f) -> void {
        ple::gated_value_grouped_norm(a, b, c, d, e, f);
      });
  m.def(
      "ple_residual_add(Tensor gated_value_flat, Tensor conv_output, "
      "Tensor(a!) output) -> ()");
  m.impl(
      "ple_residual_add",
      at::kXPU,
      [](at::Tensor a, at::Tensor b, at::Tensor c) -> void {
        ple::residual_add(a, b, c);
      });
  m.def(
      "ple_short_conv_decode(Tensor input, Tensor(a!) conv_state, "
      "Tensor conv_weights, Tensor state_indices, "
      "Tensor has_initial_state, Tensor(b!) output, int dilation, "
      "bool state_dim_first, int null_block_id) -> ()");
  m.impl(
      "ple_short_conv_decode",
      at::kXPU,
      [](at::Tensor a,
         at::Tensor b,
         at::Tensor c,
         at::Tensor d,
         at::Tensor e,
         at::Tensor f,
         int64_t g,
         bool h,
         int64_t i) -> void {
        ple::short_conv_decode(a, b, c, d, e, f, g, h, i);
      });
  m.def(
      "ple_short_conv_decode_trusted(Tensor input, Tensor(a!) conv_state, "
      "Tensor conv_weights, Tensor state_indices, "
      "Tensor has_initial_state, Tensor(b!) output, "
      "int dilation, bool state_dim_first, int null_block_id) -> ()");
  m.impl(
      "ple_short_conv_decode_trusted",
      at::kXPU,
      [](at::Tensor a,
         at::Tensor b,
         at::Tensor c,
         at::Tensor d,
         at::Tensor e,
         at::Tensor f,
         int64_t g,
         bool h,
         int64_t i) -> void {
        ple::short_conv_decode_trusted(a, b, c, d, e, f, g, h, i);
      });
  m.def(
      "ple_short_conv_prefill(Tensor input, Tensor query_start_loc, "
      "Tensor(a!) conv_state, Tensor conv_weights, Tensor state_indices, "
      "Tensor has_initial_state, Tensor(b!) output, "
      "int dilation, bool state_dim_first, int null_block_id) -> ()");
  m.impl(
      "ple_short_conv_prefill",
      at::kXPU,
      [](at::Tensor a,
         at::Tensor b,
         at::Tensor c,
         at::Tensor d,
         at::Tensor e,
         at::Tensor f,
         at::Tensor g,
         int64_t h,
         bool i,
         int64_t j) -> void {
        ple::short_conv_prefill(a, b, c, d, e, f, g, h, i, j);
      });
  m.def(
      "ple_short_conv_prefill_trusted(Tensor input, Tensor query_start_loc, "
      "Tensor(a!) conv_state, Tensor conv_weights, Tensor state_indices, "
      "Tensor has_initial_state, Tensor(b!) output, int dilation, "
      "bool state_dim_first, "
      "int null_block_id) -> ()");
  m.impl(
      "ple_short_conv_prefill_trusted",
      at::kXPU,
      [](at::Tensor a,
         at::Tensor b,
         at::Tensor c,
         at::Tensor d,
         at::Tensor e,
         at::Tensor f,
         at::Tensor g,
         int64_t h,
         bool i,
         int64_t j) -> void {
        ple::short_conv_prefill_trusted(a, b, c, d, e, f, g, h, i, j);
      });
  m.def(
      "ple_short_conv_spec(Tensor input, Tensor query_start_loc, "
      "Tensor(a!) conv_state, Tensor conv_weights, Tensor state_indices, "
      "Tensor num_accepted_tokens, Tensor(b!) output, "
      "int num_spec_tokens, int dilation, bool state_dim_first, "
      "int null_block_id) -> ()");
  m.impl(
      "ple_short_conv_spec",
      at::kXPU,
      [](at::Tensor a,
         at::Tensor b,
         at::Tensor c,
         at::Tensor d,
         at::Tensor e,
         at::Tensor f,
         at::Tensor g,
         int64_t h,
         int64_t i,
         bool j,
         int64_t k) -> void {
        ple::short_conv_spec(a, b, c, d, e, f, g, h, i, j, k);
      });
  m.def(
      "ple_short_conv_spec_trusted(Tensor input, Tensor query_start_loc, "
      "Tensor(a!) conv_state, Tensor conv_weights, Tensor state_indices, "
      "Tensor num_accepted_tokens, Tensor(b!) output, "
      "int num_spec_tokens, int dilation, "
      "bool state_dim_first, int null_block_id) -> ()");
  m.impl(
      "ple_short_conv_spec_trusted",
      at::kXPU,
      [](at::Tensor a,
         at::Tensor b,
         at::Tensor c,
         at::Tensor d,
         at::Tensor e,
         at::Tensor f,
         at::Tensor g,
         int64_t h,
         int64_t i,
         bool j,
         int64_t k) -> void {
        ple::short_conv_spec_trusted(a, b, c, d, e, f, g, h, i, j, k);
      });
}
#endif
