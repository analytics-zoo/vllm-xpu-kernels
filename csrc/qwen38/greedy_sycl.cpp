// SPDX-License-Identifier: Apache-2.0
// Isolated M1 transport kernels; same NaN and first-index argmax semantics.
#include "qwen38/greedy_sycl.h"
#include <sycl/ext/oneapi/experimental/device_architecture.hpp>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <sycl/sycl.hpp>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace vllm::qwen38 {
namespace {
template <int WG>
class Local;
template <int WG>
void local(
    sycl::queue& q,
    const sycl::half* x,
    float* packet,
    int n,
    int start,
    int padding) {
  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<std::uint32_t> partial(WG / 32, cgh);
    cgh.parallel_for<Local<WG>>(
        sycl::nd_range<1>(WG, WG),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
          const int tid = it.get_local_linear_id(), lane = tid % 32,
                    sgid = tid / 32;
          if (tid < 64) packet[tid] = 0;
          std::uint32_t best = 0;
          for (int i = tid; i < n; i += WG) {
            std::uint16_t bits =
                i < n - padding ? sycl::bit_cast<std::uint16_t>(x[i]) : 0xfc00U;
            if ((bits & 0x7fffU) == 0) bits = 0;
            const std::uint16_t order =
                (bits & 0x7fffU) > 0x7c00U
                    ? 0xffffU
                    : (bits & 0x8000U ? std::uint16_t(~bits)
                                      : std::uint16_t(bits ^ 0x8000U));
            best = sycl::max(best, (std::uint32_t(order) << 16) | (65535U - i));
          }
          best = sycl::reduce_over_group(
              it.get_sub_group(), best, sycl::maximum<>());
          if (lane == 0) partial[sgid] = best;
          it.barrier(sycl::access::fence_space::local_space);
          if (sgid == 0) {
            best = lane < WG / 32 ? partial[lane] : 0;
            best = sycl::reduce_over_group(
                it.get_sub_group(), best, sycl::maximum<>());
            const int id = 65535 - int(best & 65535U);
            if (lane == 0)
              packet[0] = id < n - padding
                              ? float(x[id])
                              : -std::numeric_limits<float>::infinity();
            if (lane == 1) packet[1] = float(start + id);
          }
        });
  });
}
template <typename Token, int Ranks>
class Global;
template <typename Token, int Ranks>
void global(sycl::queue& q, const float* packets, Token* out) {
  q.parallel_for<Global<Token, Ranks>>(
      sycl::nd_range<1>(32, 32),
      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        const int lane = it.get_local_linear_id();
        std::uint64_t key = 0;
        if (lane < Ranks) {
          auto bits = sycl::bit_cast<std::uint32_t>(packets[lane * 64]);
          if ((bits & 0x7fffffffU) == 0) bits = 0;
          const std::uint32_t order =
              (bits & 0x7fffffffU) > 0x7f800000U
                  ? 0xffffffffU
                  : (bits & 0x80000000U ? ~bits : (bits ^ 0x80000000U));
          key = (std::uint64_t(order) << 32) | (0xffffffffU - lane);
        }
        key =
            sycl::reduce_over_group(it.get_sub_group(), key, sycl::maximum<>());
        const int rank = int(0xffffffffU - std::uint32_t(key));
        if (lane == 0) out[0] = Token(packets[rank * 64 + 1]);
      });
}

}  // namespace
namespace {
bool valid(const at::Tensor& x, at::ScalarType dtype) {
  return x.defined() && x.is_xpu() && x.layout() == c10::kStrided &&
         !x.is_nested() && x.scalar_type() == dtype && x.is_contiguous() &&
         !x.is_neg() && !x.is_conj() && !x.requires_grad();
}
bool supported(sycl::queue& q) {
  namespace sx = sycl::ext::oneapi::experimental;
  return q.is_in_order() &&
         q.get_device().get_info<sx::info::device::architecture>() ==
             sx::architecture::intel_gpu_bmg_g31;
}
void retain(const at::Tensor& x, c10::xpu::XPUStream stream) {
  c10::xpu::XPUCachingAllocator::recordStream(x.storage().data_ptr(), stream);
}
}  // namespace
std::optional<at::Tensor>
try_local_argmax_packet(const at::Tensor& x, int64_t start, int64_t padding) {
  if (!valid(x, at::kHalf) || x.dim() != 2 || x.size(0) != 1 || x.size(1) < 1 ||
      x.size(1) > 65535 || padding < 0 || padding > x.size(1) || start < 0 ||
      start >= (1 << 24) - x.size(1))
    return std::nullopt;
  c10::DeviceGuard guard(x.device());
  auto stream = c10::xpu::getCurrentXPUStream(x.device().index());
  auto& q = stream.queue();
  if (!supported(q)) return std::nullopt;
  auto out = at::empty({1, 64}, x.options().dtype(at::kFloat));
  retain(x, stream);
  retain(out, stream);
  local<1024>(
      q,
      (const sycl::half*)x.const_data_ptr(),
      out.data_ptr<float>(),
      x.size(1),
      start,
      padding);
  return out;
}
template <typename Token>
std::optional<at::Tensor> global_token(const at::Tensor& x) {
  if (!valid(x, at::kFloat) || x.dim() != 2 || x.size(1) != 64 ||
      (x.size(0) != 4 && x.size(0) != 8))
    return std::nullopt;
  c10::DeviceGuard guard(x.device());
  auto stream = c10::xpu::getCurrentXPUStream(x.device().index());
  auto& q = stream.queue();
  if (!supported(q)) return std::nullopt;
  auto out = at::empty(
      {1},
      x.options().dtype(
          std::is_same_v<Token, std::int32_t> ? at::kInt : at::kLong));
  retain(x, stream);
  retain(out, stream);
  if (x.size(0) == 4)
    global<Token, 4>(
        q, x.const_data_ptr<float>(), out.template data_ptr<Token>());
  else
    global<Token, 8>(
        q, x.const_data_ptr<float>(), out.template data_ptr<Token>());
  return out;
}
std::optional<at::Tensor> try_global_argmax_token(const at::Tensor& x) {
  return global_token<std::int64_t>(x);
}
std::optional<at::Tensor> try_global_argmax_token_int32(const at::Tensor& x) {
  return global_token<std::int32_t>(x);
}
}  // namespace vllm::qwen38
