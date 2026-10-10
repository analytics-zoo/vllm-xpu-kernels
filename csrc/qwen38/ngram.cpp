// SPDX-License-Identifier: Apache-2.0
// Preserve the final PLE host-USM layout and allocation lifetime. This source
// uses only ordinary SYCL; no ESIMD package or extension is needed at runtime.
#include <ATen/MemoryOverlap.h>
#include <ATen/core/CachingHostAllocator.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>

#include <array>
#include <cstdint>
#include <limits>

#include "qwen38/ngram.h"

namespace vllm::qwen38 {
namespace {

constexpr std::array<uint64_t, 16> kVocabulary{
    20000003ULL,
    20000023ULL,
    20000033ULL,
    20000047ULL,
    20000059ULL,
    20000063ULL,
    20000069ULL,
    20000077ULL,
    20000081ULL,
    20000093ULL,
    20000107ULL,
    20000147ULL,
    20000153ULL,
    20000159ULL,
    20000161ULL,
    20000171ULL};
constexpr std::array<uint64_t, 16> kMagic{
    0xad7f2572edd5e756ULL,
    0xad7f094d2dd3860cULL,
    0xad7efb3a4f348292ULL,
    0xad7ee7864c48f7c0ULL,
    0xad7ed6a2dd81b3bfULL,
    0xad7ed101b8e02bbbULL,
    0xad7ec8900234b56bULL,
    0xad7ebd4db9d4513dULL,
    0xad7eb7ac95dcca18ULL,
    0xad7ea6c92ad8e05eULL,
    0xad7e93152facb510ULL,
    0xad7e5ac9d9b6d43aULL,
    0xad7e525827b16249ULL,
    0xad7e49e67600f083ULL,
    0xad7e4715e583ae2dULL,
    0xad7e3903139f0d61ULL};
constexpr std::array<uint64_t, 16> kOffsets{
    0ULL,
    20000003ULL,
    40000026ULL,
    60000059ULL,
    80000106ULL,
    100000165ULL,
    120000228ULL,
    140000297ULL,
    160000374ULL,
    180000455ULL,
    200000548ULL,
    220000655ULL,
    240000802ULL,
    260000955ULL,
    280001114ULL,
    300001275ULL};

void check_physical(const at::Tensor& t, const char* name) {
  TORCH_CHECK(
      t.layout() == at::kStrided && t.is_contiguous() && !t.is_neg() &&
          !t.is_conj(),
      name,
      " must be a contiguous physical tensor");
}

void check_device(
    const at::Tensor& t, const at::Tensor& reference, const char* name) {
  TORCH_CHECK(
      t.device() == reference.device(), name, " must be on the same XPU");
  check_physical(t, name);
}

// Both tensors have already passed the contiguous physical-tensor checks.
// ATen's storage-based check alone misses independently wrapped allocations.
void check_no_alias(const at::Tensor& output, const at::Tensor& input) {
  if (output.numel() == 0 || input.numel() == 0) return;
  at::assert_no_overlap(output, input);
  const auto output_begin =
      reinterpret_cast<uintptr_t>(output.const_data_ptr());
  const auto input_begin = reinterpret_cast<uintptr_t>(input.const_data_ptr());
  const bool overlap = output_begin >= input_begin
                           ? output_begin - input_begin < input.nbytes()
                           : input_begin - output_begin < output.nbytes();
  TORCH_CHECK(!overlap, "NGram output must not overlap an input address range");
}

void record_xpu(const at::Tensor& t, c10::xpu::XPUStream stream) {
  c10::xpu::XPUCachingAllocator::recordStream(t.storage().data_ptr(), stream);
}

template <bool ResetEos>
class NGramDecodeIdsKernel;
class NGramHostLookupKernel;

}  // namespace

template <bool ResetEos>
static void ngram_decode_ids_impl(
    const at::Tensor& input_ids,
    const at::Tensor& context,
    const at::Tensor& multipliers,
    at::Tensor& output,
    int64_t eos_token_id) {
  TORCH_CHECK(input_ids.is_xpu(), "input_ids must be on XPU");
  const std::array<const at::Tensor*, 4> tensors{
      &input_ids, &context, &multipliers, &output};
  for (const auto* t : tensors) {
    check_device(*t, input_ids, "NGram argument");
    TORCH_CHECK(t->scalar_type() == at::kLong, "NGram arguments must be int64");
  }
  const int64_t m = input_ids.numel();
  TORCH_CHECK(input_ids.dim() == 1 && m > 0, "input_ids must be [M], M>0");
  TORCH_CHECK(
      context.dim() == 2 && context.size(0) == m && context.size(1) == 2,
      "context must be [M,2]");
  TORCH_CHECK(
      multipliers.dim() == 1 && multipliers.size(0) == 3,
      "multipliers must be [3]");
  TORCH_CHECK(
      output.dim() == 2 && output.size(0) == m && output.size(1) == 16,
      "output must be [M,16]");
  for (const auto* t : {&input_ids, &context, &multipliers}) {
    check_no_alias(output, *t);
  }
  const c10::OptionalDeviceGuard guard(input_ids.device());
  auto stream = c10::xpu::getCurrentXPUStream(input_ids.get_device());
  for (const auto* t : tensors) {
    record_xpu(*t, stream);
  }
  const auto* input =
      reinterpret_cast<const uint64_t*>(input_ids.data_ptr<int64_t>());
  const auto* history =
      reinterpret_cast<const uint64_t*>(context.data_ptr<int64_t>());
  const auto* mult =
      reinterpret_cast<const uint64_t*>(multipliers.data_ptr<int64_t>());
  auto* out = output.data_ptr<int64_t>();
  const auto eos = static_cast<uint64_t>(eos_token_id);
  stream.queue().parallel_for<NGramDecodeIdsKernel<ResetEos>>(
      sycl::nd_range<1>(size_t(m) * 16, 16),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
        const int64_t row = item.get_group_linear_id();
        const int head = item.get_local_linear_id();
        const uint64_t current = input[row];
        uint64_t previous = history[row * 2 + 1];
        uint64_t previous_2 = history[row * 2];
        if constexpr (ResetEos) {
          // 当前 EOS 清空全部历史；前一 token 为 EOS 则截断更早历史。
          const bool reset = current == eos;
          previous_2 = reset || previous == eos ? eos : previous_2;
          previous = reset ? eos : previous;
        }
        uint64_t mixed = (current * mult[0]) ^ (previous * mult[1]);
        if (head >= 8) mixed ^= previous_2 * mult[2];
        const bool negative = (mixed >> 63) != 0;
        const uint64_t magnitude = negative ? uint64_t(0) - mixed : mixed;
        uint64_t quotient = sycl::mul_hi(magnitude, kMagic[head]);
        quotient = (quotient + ((magnitude - quotient) >> 1)) >> 24;
        uint64_t remainder = magnitude - quotient * kVocabulary[head];
        if (negative && remainder != 0)
          remainder = kVocabulary[head] - remainder;
        out[row * 16 + head] = remainder + kOffsets[head];
      });
}

void ngram_decode_ids(
    const at::Tensor& input_ids,
    const at::Tensor& context,
    const at::Tensor& multipliers,
    at::Tensor& output) {
  ngram_decode_ids_impl<false>(input_ids, context, multipliers, output, 0);
}

void ngram_decode_ids_eos(
    const at::Tensor& input_ids,
    const at::Tensor& context,
    const at::Tensor& multipliers,
    at::Tensor& output,
    int64_t eos_token_id) {
  ngram_decode_ids_impl<true>(
      input_ids, context, multipliers, output, eos_token_id);
}

at::Tensor ngram_host_lookup_chunked(
    at::TensorList weights,
    const at::Tensor& ids,
    at::Tensor output,
    int64_t vocab_start,
    int64_t vocab_end) {
  TORCH_CHECK(
      weights.size() >= 1 && weights.size() <= 8,
      "NGram lookup requires 1..8 pinned host chunks");
  TORCH_CHECK(ids.is_xpu(), "NGram ids must be on XPU");
  check_device(ids, ids, "ids");
  check_device(output, ids, "output");
  TORCH_CHECK(
      ids.scalar_type() == at::kLong && (output.scalar_type() == at::kHalf ||
                                         output.scalar_type() == at::kBFloat16),
      "NGram lookup requires int64 ids and FP16/BF16 output");
  TORCH_CHECK(
      ids.dim() == 2 && ids.size(1) == 16 && output.dim() == 2 &&
          output.size(0) == ids.size(0) && output.size(1) == 2560,
      "NGram lookup expects ids [M,16], output [M,2560]");
  check_no_alias(output, ids);
  std::array<const uint16_t*, 8> tables{};
  int64_t total_rows = 0;
  int64_t chunk_rows = 0;
  for (size_t index = 0; index < weights.size(); ++index) {
    const auto& weight = weights[index];
    check_physical(weight, "weight");
    TORCH_CHECK(
        weight.device().is_cpu() && weight.dim() == 2 && weight.size(1) == 160,
        "NGram weights must be CPU tensors [N,160]");
    TORCH_CHECK(
        weight.scalar_type() == output.scalar_type(),
        "NGram weights/output must have the same FP16/BF16 dtype");
    // Host-USM can also be wrapped as an XPU tensor; device labels alone do
    // not prove that the writable output is disjoint from a pinned table.
    check_no_alias(output, weight);
    if (index == 0) chunk_rows = weight.size(0);
    TORCH_CHECK(
        weight.size(0) > 0 && weight.size(0) <= chunk_rows &&
            (index + 1 == weights.size() || weight.size(0) == chunk_rows),
        "NGram chunks must have equal full sizes and a nonempty bounded last "
        "chunk");
    TORCH_CHECK(
        weight.size(0) <= std::numeric_limits<int64_t>::max() - total_rows,
        "NGram row count overflows int64");
    total_rows += weight.size(0);
    tables[index] = static_cast<const uint16_t*>(weight.data_ptr());
  }
  TORCH_CHECK(
      vocab_start >= 0 && vocab_end >= vocab_start &&
          vocab_end - vocab_start <= total_rows,
      "NGram lookup has invalid local vocabulary bounds");
  const c10::OptionalDeviceGuard guard(ids.device());
  auto stream = c10::xpu::getCurrentXPUStream(ids.get_device());
  auto& queue = stream.queue();
  TORCH_CHECK(
      queue.is_in_order() &&
          queue.get_device().has(sycl::aspect::usm_host_allocations),
      "NGram lookup requires the current in-order host-USM-capable stream");
  for (const auto& weight : weights) {
    TORCH_CHECK(
        sycl::get_pointer_type(weight.data_ptr(), queue.get_context()) ==
            sycl::usm::alloc::host,
        "NGram weights must be XPU-accessible pinned host memory");
  }
  if (ids.numel() == 0) return output;
  // Complete all data/metadata checks before recording any ownership or
  // submitting work. The host allocation cannot be recycled before this queue
  // finishes, including an offset view whose owner is freed by the caller.
  for (const auto& weight : weights) {
    TORCH_CHECK(
        at::getHostAllocator(at::kXPU)->record_event(
            weight.data_ptr(),
            weight.storage().data_ptr().get_context(),
            stream.unwrap()),
        "NGram pinned weights must belong to the PyTorch host allocator");
  }
  record_xpu(ids, stream);
  record_xpu(output, stream);
  const auto* indices = ids.data_ptr<int64_t>();
  auto* result = static_cast<uint16_t*>(output.data_ptr());
  const size_t rows = ids.numel();
  const int64_t chunk_count = weights.size();
  queue.parallel_for<NGramHostLookupKernel>(
      sycl::nd_range<1>(rows * 64, 64),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
        const size_t row = item.get_group_linear_id();
        const size_t lane = item.get_local_linear_id();
        const int64_t global = indices[row];
        const bool owned = global >= vocab_start && global < vocab_end;
        int64_t bank = 0, offset = 0;
        if (owned) {
          const int64_t local = global - vocab_start;
          if (chunk_count == 1) {
            offset = local;
          } else if (chunk_count == 2) {
            bank = local >= chunk_rows;
            offset = local - bank * chunk_rows;
          } else {
            bank = local / chunk_rows;
            offset = local % chunk_rows;
          }
        }
        for (size_t col = lane; col < 160; col += 64) {
          result[row * 160 + col] =
              owned ? tables[bank][offset * 160 + col] : uint16_t(0);
        }
      });
  return output;
}

at::Tensor ngram_host_lookup(
    const at::Tensor& weight,
    const at::Tensor& ids,
    at::Tensor output,
    int64_t vocab_start,
    int64_t vocab_end) {
  const std::array<at::Tensor, 1> weights{weight};
  return ngram_host_lookup_chunked(
      weights, ids, output, vocab_start, vocab_end);
}

}  // namespace vllm::qwen38
