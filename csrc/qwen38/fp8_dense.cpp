// SPDX-License-Identifier: Apache-2.0
// Dense Qwen4 TP2 decode: ordinary SYCL with the shared SYCL-TLA GEMV body.
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/ext/oneapi/experimental/device_architecture.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <torch/extension.h>

#include "../xpu/gemv/xe_2/fp8_gemv.hpp"

namespace {
namespace sx = sycl::ext::oneapi::experimental;
namespace gemv = vllm::fp8_gemv;
using half = sycl::half;

bool compatible(
    const at::Tensor& x, const at::Tensor& w, const at::Tensor& scale) {
  return x.is_xpu() && x.scalar_type() == at::kHalf && x.dim() == 2 &&
         x.size(0) == 1 && x.is_contiguous() && !x.is_neg() && !x.is_conj() &&
         w.device() == x.device() && w.dim() == 2 &&
         w.scalar_type() == at::ScalarType::Float8_e4m3fn &&
         w.size(1) == x.size(1) && w.is_contiguous() && !w.is_neg() &&
         !w.is_conj() && scale.device() == x.device() &&
         scale.scalar_type() == at::kFloat && scale.numel() == 1 &&
         scale.is_contiguous() && !scale.is_neg() && !scale.is_conj() &&
         reinterpret_cast<uintptr_t>(x.const_data_ptr()) % 64 == 0 &&
         reinterpret_cast<uintptr_t>(w.const_data_ptr()) % 64 == 0;
}

int splits(int64_t n, int64_t k) {
  if (k == 5120 && (n == 14336 || n == 8192 || n == 7168)) return 31;
  if (n == 5120 && k == 7168) return 62;
  if (n == 5120 && k == 3072) return 20;
  return 0;
}

bool is_b70(const c10::xpu::XPUStream& stream) {
  static thread_local int last_device = -1;
  static thread_local bool supported = false;
  if (last_device != stream.device_index()) {
    supported = stream.queue()
                    .get_device()
                    .get_info<sx::info::device::architecture>() ==
                sx::architecture::intel_gpu_bmg_g31;
    last_device = stream.device_index();
  }
  return supported;
}

void record(
    std::initializer_list<const at::Tensor*> tensors,
    c10::xpu::XPUStream stream) {
  for (const auto* tensor : tensors)
    c10::xpu::XPUCachingAllocator::recordStream(
        tensor->storage().data_ptr(), stream);
}

gemv::Seg<half>
segment(const at::Tensor& w, const at::Tensor& scale, at::Tensor& output) {
  return {
      reinterpret_cast<const uint8_t*>(w.const_data_ptr()),
      scale.const_data_ptr<float>(),
      reinterpret_cast<half*>(output.mutable_data_ptr<at::Half>()),
      int(w.size(0)),
      int(w.size(0)),
      0};
}

template <int KS, bool Pair>
struct DenseFp8Kernel {
  const half* input;
  int k;
  gemv::Seg<half> first;
  gemv::Seg<half> second;

  [[sycl::reqd_sub_group_size(gemv::kSg)]] void
  operator()(sycl::nd_item<1> item) const {
#ifdef __SYCL_DEVICE_ONLY__
    auto* partial = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[KS * gemv::kSg]>(item.get_group());
    const int group = item.get_group(0);
    const auto& seg = Pair && group >= second.wg_begin ? second : first;
    const int row = (group - seg.wg_begin) * gemv::kRowBlock;
    const gemv::GlobalA<1, half> a{input, k, 1, k * 2};
    const gemv::u32x8 unused[1][gemv::kChunkSteps] = {};
    gemv::gemv_body<1, KS, 1, gemv::GlobalA<1, half>, 1, 0, half, true>(
        item, seg, row, a, partial, 1, k, unused, false);
#endif
  }
};

template <int KS, bool Pair>
void launch(
    c10::xpu::XPUStream stream,
    const at::Tensor& x,
    gemv::Seg<half> first,
    gemv::Seg<half> second) {
  const int first_groups = (first.N + gemv::kRowBlock - 1) / gemv::kRowBlock;
  second.wg_begin = first_groups;
  const int groups =
      first_groups +
      (Pair ? (second.N + gemv::kRowBlock - 1) / gemv::kRowBlock : 0);
  sx::nd_launch(
      stream.queue(),
      sycl::nd_range<1>(size_t(groups * KS * gemv::kSg), KS * gemv::kSg),
      DenseFp8Kernel<KS, Pair>{
          reinterpret_cast<const half*>(x.const_data_ptr<at::Half>()),
          int(x.size(1)),
          first,
          second});
}

std::optional<at::Tensor> try_linear(
    const at::Tensor& x, const at::Tensor& weight, const at::Tensor& scale) {
  if (!compatible(x, weight, scale)) return std::nullopt;
  const int ks = splits(weight.size(0), weight.size(1));
  if (!ks) return std::nullopt;
  const c10::DeviceGuard guard(x.device());
  const auto stream = c10::xpu::getCurrentXPUStream(x.get_device());
  if (!is_b70(stream)) return std::nullopt;
  auto output = at::empty({1, weight.size(0)}, x.options());
  record({&x, &weight, &scale, &output}, stream);
  const auto seg = segment(weight, scale, output);
  if (ks == 31)
    launch<31, false>(stream, x, seg, seg);
  else if (ks == 62)
    launch<62, false>(stream, x, seg, seg);
  else
    launch<20, false>(stream, x, seg, seg);
  return output;
}

std::optional<std::tuple<at::Tensor, at::Tensor>> try_pair(
    const at::Tensor& x,
    const at::Tensor& w0,
    const at::Tensor& s0,
    const at::Tensor& w1,
    const at::Tensor& s1) {
  if (!compatible(x, w0, s0) || !compatible(x, w1, s1) || w0.size(0) != 8192 ||
      w1.size(0) != 48 || x.size(1) != 5120)
    return std::nullopt;
  const c10::DeviceGuard guard(x.device());
  const auto stream = c10::xpu::getCurrentXPUStream(x.get_device());
  if (!is_b70(stream)) return std::nullopt;
  auto output0 = at::empty({1, 8192}, x.options());
  auto output1 = at::empty({1, 48}, x.options());
  record({&x, &w0, &s0, &w1, &s1, &output0, &output1}, stream);
  // Schedule the tiny a/b projection first so it cannot form a kernel tail.
  launch<31, true>(
      stream, x, segment(w1, s1, output1), segment(w0, s0, output0));
  return std::make_tuple(output0, output1);
}
}  // namespace

void init_qwen4_dense_fp8(pybind11::module_& module) {
  module.def("try_qwen4_dense_fp8_linear", &try_linear);
  module.def("try_qwen4_dense_fp8_pair", &try_pair);
}
