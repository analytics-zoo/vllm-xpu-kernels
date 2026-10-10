// SPDX-License-Identifier: Apache-2.0
// Standalone HC UP experiment. Deliberately absent from central CMake and
// registered under a unique namespace; never loaded as the default HC path.

#include <ATen/MemoryOverlap.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUStream.h>
#include <cute/tensor.hpp>
#include <cute/arch/mma_xe.hpp>
#include <sycl/ext/intel/math/imf_fp_conversions.hpp>
#include <sycl/ext/oneapi/experimental/device_architecture.hpp>
#include <sycl/sycl.hpp>
#include <torch/library.h>
#include <torch/types.h>

#include <cstdint>
#include <limits>

namespace vllm::qwen38::hc_up_dpas_probe {
namespace {

using half = sycl::half;
constexpr int kRank = 320;
constexpr int kHidden = 2560;
constexpr int kStreams = 4;
constexpr int kWidth = kStreams * kHidden;
constexpr int kTileN = 16;
constexpr int kTileK = 16;

void check_no_overlap(const torch::Tensor& a, const torch::Tensor& b) {
  at::assert_no_overlap(a, b);
  const bool padded =
      at::get_overlap_status(a, b) == at::MemOverlapStatus::TooHard;
  const auto a_start = reinterpret_cast<uintptr_t>(
      padded ? a.storage().data() : a.const_data_ptr());
  const auto b_start = reinterpret_cast<uintptr_t>(
      padded ? b.storage().data() : b.const_data_ptr());
  const auto a_bytes = padded ? a.storage().nbytes() : a.numel() * a.element_size();
  const auto b_bytes = padded ? b.storage().nbytes() : b.numel() * b.element_size();
  TORCH_CHECK(
      a_start <= b_start ? b_start - a_start >= a_bytes
                         : a_start - b_start >= b_bytes,
      "probe tensors have overlapping storage or unprovable overlap");
}

template <int Rows>
class Kernel;

inline float sigmoid(float value) { return 1.0f / (1.0f + sycl::exp(-value)); }

template <int Rows>
void launch(
    sycl::queue& queue,
    const half* input,
    int input_stride,
    const half* weight,
    const half* normed,
    half* output,
    int m) {
  static_assert(Rows == 2 || Rows == 4 || Rows == 8);
  using Mma = cute::XE_DPAS_TT<Rows, float, cute::half_t>;
  static_assert(Mma::K == kTileK);

  // One SG owns 16 hidden columns across all rows and four branches. The
  // branches merge locally, so neither a second launch nor an inter-WG race
  // is needed. M5-8 pads the DPAS atom to eight rows.
  queue.parallel_for<Kernel<Rows>>(
      sycl::nd_range<1>(kHidden, kTileN),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(kTileN)]] {
        const int lane = int(item.get_local_linear_id());
        const int h_base = int(item.get_group_linear_id()) * kTileN;
        const int h = h_base + lane;
        float mixed[Rows] = {};

        for (int branch = 0; branch < kStreams; ++branch) {
          typename Mma::DVector accum = {};
          const int n_base = branch * kHidden + h_base;

          for (int kb = 0; kb < kRank; kb += kTileK) {
            typename Mma::AVector a_frag;
            typename Mma::BVector b_frag;

#pragma unroll
            for (int row = 0; row < Rows; ++row) {
              const half value =
                  row < m ? input[int64_t(row) * input_stride + kb + lane]
                          : half(0);
              a_frag[row] = sycl::bit_cast<uint16_t>(value);
            }

            // The INT4 small-M DPAS kernel has GPU-verified B layout:
            // lane/2 selects column 0, +8 selects column 1, lane parity
            // selects alternating K values, and b_frag interleaves columns.
            const int col0 = n_base + lane / 2;
            const int col1 = col0 + 8;
            const int parity = lane & 1;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
              b_frag[2 * j] = sycl::bit_cast<uint16_t>(
                  weight[int64_t(col0) * kRank + kb + 2 * j + parity]);
              b_frag[2 * j + 1] = sycl::bit_cast<uint16_t>(
                  weight[int64_t(col1) * kRank + kb + 2 * j + parity]);
            }
            Mma::fma(accum, a_frag, b_frag, accum);
          }

#pragma unroll
          for (int row = 0; row < Rows; ++row) {
            if (row < m) {
              // Match the required FP32 -> FP16 -> FP32 gate boundary. The
              // FP32 DPAS accumulation order may still differ from old HC.
              volatile uint16_t gate_bits = sycl::bit_cast<uint16_t>(
                  sycl::ext::intel::math::float2half_rn(accum[row]));
              const half gate = sycl::bit_cast<half>(uint16_t(gate_bits));
              mixed[row] +=
                  float(normed[int64_t(row) * kWidth + n_base + lane]) *
                  sigmoid(float(gate));
            }
          }
        }

#pragma unroll
        for (int row = 0; row < Rows; ++row) {
          if (row < m)
            output[int64_t(row) * kHidden + h] = half(mixed[row] * 0.25f);
        }
      });
}

void check_matrix(
    const torch::Tensor& value,
    const torch::Tensor& input,
    int rows,
    int columns,
    const char* name) {
  TORCH_CHECK(value.device() == input.device(), name, " must share the device");
  TORCH_CHECK(value.scalar_type() == at::kHalf, name, " must be FP16");
  TORCH_CHECK(!value.is_neg() && !value.is_conj(), name, " is a lazy view");
  TORCH_CHECK(
      value.dim() == 2 && value.size(0) == rows && value.size(1) == columns &&
          value.is_contiguous(),
      name,
      " has an invalid shape/layout");
}

void record(const torch::Tensor& value, c10::xpu::XPUStream stream) {
  c10::xpu::XPUCachingAllocator::recordStream(
      value.storage().data_ptr(), stream);
}

}  // namespace

void up_gate_mix(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& normed,
    torch::Tensor& output) {
  // Complete preflight before the first submit. This sidecar has no fallback:
  // unsupported inputs throw, leaving output and the GPU queue untouched.
  TORCH_CHECK(
      input.is_xpu() && input.scalar_type() == at::kHalf && input.dim() == 2 &&
          input.size(0) >= 2 && input.size(0) <= 8 && input.size(1) == kRank &&
          input.stride(1) == 1 && input.stride(0) >= kRank &&
          input.stride(0) <= std::numeric_limits<int>::max() &&
          !input.is_neg() && !input.is_conj(),
      "probe input must be nonoverlapping FP16 XPU [M=2..8,320]");
  const int m = int(input.size(0));
  check_matrix(weight, input, kWidth, kRank, "probe weight");
  check_matrix(normed, input, m, kWidth, "probe normed");
  check_matrix(output, input, m, kHidden, "probe output");
  TORCH_CHECK(
      (reinterpret_cast<uintptr_t>(weight.data_ptr()) & 63u) == 0,
      "probe weight must be 64-byte aligned for 2D VNNI load");
  check_no_overlap(output, input);
  check_no_overlap(output, weight);
  check_no_overlap(output, normed);

  const c10::DeviceGuard guard(input.device());
  const auto stream = c10::xpu::getCurrentXPUStream(input.get_device());
  auto& queue = stream.queue();
  TORCH_CHECK(queue.is_in_order(), "probe requires current in-order queue");
  namespace sx = sycl::ext::oneapi::experimental;
  TORCH_CHECK(
      queue.get_device().get_info<sx::info::device::architecture>() ==
          sx::architecture::intel_gpu_bmg_g31,
      "probe requires BMG G31");

  const auto* x = reinterpret_cast<const half*>(input.data_ptr<at::Half>());
  const auto* w = reinterpret_cast<const half*>(weight.data_ptr<at::Half>());
  const auto* n = reinterpret_cast<const half*>(normed.data_ptr<at::Half>());
  auto* y = reinterpret_cast<half*>(output.data_ptr<at::Half>());
  record(input, stream);
  record(weight, stream);
  record(normed, stream);
  record(output, stream);
  if (m <= 2)
    launch<2>(queue, x, int(input.stride(0)), w, n, y, m);
  else if (m <= 4)
    launch<4>(queue, x, int(input.stride(0)), w, n, y, m);
  else
    launch<8>(queue, x, int(input.stride(0)), w, n, y, m);
}

}  // namespace vllm::qwen38::hc_up_dpas_probe

TORCH_LIBRARY(qwen38_hc_dpas_probe, m) {
  m.def(
      "up_gate_mix(Tensor input, Tensor weight, Tensor normed, Tensor(a!) "
      "output) -> ()");
  m.impl(
      "up_gate_mix", torch::kXPU, &vllm::qwen38::hc_up_dpas_probe::up_gate_mix);
}
