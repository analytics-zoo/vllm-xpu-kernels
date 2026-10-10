// SPDX-License-Identifier: Apache-2.0
#include "qwen38/gdn_sycl_projection.h"

#include <ATen/MemoryOverlap.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/ext/oneapi/experimental/group_load_store.hpp>
#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <string_view>

#include "qwen38/gdn_sycl.h"
#include "qwen38/ops.h"

namespace vllm::qwen38 {
namespace {
using half = sycl::half;
constexpr int kHeads = 12;
constexpr int kHeadSize = 128;
constexpr int kWidth = kHeads * kHeadSize;
constexpr int kSubgroup = 16;
constexpr int kOutputRowsPerGroup = 32;
static_assert(kHeadSize / 2 == kSubgroup * sizeof(uint32_t));
template <int RowsPerGroup>
class GdnNormInt4FusedKernel;

void check_no_overlap(const torch::Tensor& a, const torch::Tensor& b) {
  at::assert_no_overlap(a, b);
  const auto a_start = reinterpret_cast<uintptr_t>(a.const_data_ptr());
  const auto b_start = reinterpret_cast<uintptr_t>(b.const_data_ptr());
  const auto a_bytes = a.numel() * a.element_size();
  const auto b_bytes = b.numel() * b.element_size();
  TORCH_CHECK(
      a_start <= b_start ? b_start - a_start >= a_bytes
                         : a_start - b_start >= b_bytes,
      "GDN INT4 tensors have overlapping physical memory");
}

// The final ESIMD path retains FP32 normalized values in work-group storage.
template <int RowsPerGroup>
void launch_fused(
    sycl::queue& queue,
    const half* x,
    const half* z,
    const half* norm_weight,
    const uint8_t* weight,
    const half* scale,
    half* output,
    float eps) {
  static_assert(RowsPerGroup == 16 || RowsPerGroup == kOutputRowsPerGroup);
  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> normalized(kWidth, cgh);
    cgh.parallel_for<GdnNormInt4FusedKernel<RowsPerGroup>>(
        sycl::nd_range<1>(2560 / RowsPerGroup * 256, 256),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(kSubgroup)]] {
          const auto sg = item.get_sub_group();
          const int lane = sg.get_local_linear_id();
          const int worker = item.get_local_linear_id() / kSubgroup;
          if (worker < kHeads) {
            float xv[8], zv[8], sq[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
              const int col = lane + j * kSubgroup;
              xv[j] = float(x[worker * kHeadSize + col]);
              zv[j] = float(z[worker * kHeadSize + col]);
              sq[j] = xv[j] * xv[j];
            }
        // Pairwise 128-element tree, apart from subgroup lowering.
#pragma unroll
            for (int j = 0; j < 4; ++j)
              sq[j] += sq[j + 4];
#pragma unroll
            for (int j = 0; j < 2; ++j)
              sq[j] += sq[j + 2];
            const float sum =
                sycl::reduce_over_group(sg, sq[0] + sq[1], sycl::plus<float>());
            const float inv = sycl::rsqrt(sum / kHeadSize + eps);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
              const int col = lane + j * kSubgroup;
              const float value = xv[j] * inv * float(norm_weight[col]);
              normalized[worker * kHeadSize + col] =
                  value * (1.0f / (1.0f + sycl::exp(-zv[j])));
            }
          }
          item.barrier(sycl::access::fence_space::local_space);

#pragma unroll
          for (int row_index = 0; row_index < RowsPerGroup / 16; ++row_index) {
            // 每个输出保留原有累加/归约顺序，只复用本WG的FP32 norm。
            const int row = item.get_group_linear_id() * RowsPerGroup +
                            row_index * 16 + worker;
            // Four pairwise accumulators per lane represent the 64 ESIMD
            // even/odd pairs; each position accumulates across all 12 heads.
            float acc[4] = {};
            const int64_t row_byte = int64_t(row) * (kWidth / 2);
            const int64_t row_scale = int64_t(row) * kHeads;
            namespace sx = sycl::ext::oneapi::experimental;
            constexpr auto props = sx::properties{
                sx::data_placement_striped,
                sx::contiguous_memory,
                sx::full_group,
                sx::alignment<4>};
            for (int head = 0; head < kHeads; ++head) {
              // 64 packed bytes/head: one consecutive uint32 per SG16 lane.
              sycl::vec<uint32_t, 1> raw;
              auto* words = sycl::address_space_cast<
                                sycl::access::address_space::global_space,
                                sycl::access::decorated::yes>(
                                reinterpret_cast<const uint32_t*>(
                                    weight + row_byte + head * (kHeadSize / 2)))
                                .get_decorated();
              sx::group_load(sg, words, raw, props);
              const uint32_t packed_word = raw[0];
              const float s = float(scale[row_scale + head]);
              const float neg8s = -8.0f * s;
#pragma unroll
              for (int j = 0; j < 4; ++j) {
                const int col = head * kHeadSize + 2 * (lane * 4 + j);
                const uint8_t packed = uint8_t(packed_word >> (8 * j));
                const float lo = float(packed & 15) * s + neg8s;
                const float hi = float(packed >> 4) * s + neg8s;
                acc[j] += normalized[col] * lo + normalized[col + 1] * hi;
              }
            }
        // 64->32->16->8->4->2->1 pairwise reduction.
#pragma unroll
            for (int offset = 8; offset > 0; offset /= 2) {
#pragma unroll
              for (int j = 0; j < 4; ++j) {
                const float other = sycl::shift_group_left(sg, acc[j], offset);
                if (lane < offset) acc[j] += other;
              }
            }
            if (lane == 0)
              output[row] = half((acc[0] + acc[2]) + (acc[1] + acc[3]));
          }
        });
  });
}
}  // namespace

void gdn_norm_int4_sycl(
    const torch::Tensor& x,
    const torch::Tensor& z,
    const torch::Tensor& norm_weight,
    const torch::Tensor& int4_weight,
    const torch::Tensor& int4_scale,
    torch::Tensor& output,
    int64_t hv,
    int64_t v,
    double eps,
    bool sigmoid_gate) {
  // A caller may use this as a guarded fast path. Check the entire projection
  // before norm_gate submits, so an unsupported weight layout cannot leave a
  // partially executed operation followed by a fallback state write.
  TORCH_CHECK(
      (hv == 12 || hv == 6) && v == 128,
      "native GDN INT4 output supports only TP4/TP8 Qwen3.8");
  const float epsilon = static_cast<float>(eps);
  TORCH_CHECK(
      std::isfinite(epsilon) && epsilon > 0.0f,
      "invalid GDN norm epsilon after FP32 conversion");
  TORCH_CHECK(x.is_xpu(), "GDN INT4 output requires XPU tensors");
  const auto device = x.device();
  for (const auto* t : std::initializer_list<const torch::Tensor*>{
           &x, &z, &norm_weight, &int4_scale, &output}) {
    TORCH_CHECK(
        t->device() == device && t->scalar_type() == at::kHalf &&
            t->is_contiguous() && !t->is_neg() && !t->is_conj(),
        "GDN INT4 output requires contiguous FP16 tensors on one XPU");
  }
  TORCH_CHECK(
      int4_weight.device() == device &&
          int4_weight.scalar_type() == at::kByte &&
          int4_weight.is_contiguous() && !int4_weight.is_neg() &&
          !int4_weight.is_conj(),
      "GDN INT4 output requires contiguous uint8 weight on the same XPU");
  const int64_t k = hv * v;
  TORCH_CHECK(
      x.dim() == 2 && x.size(0) == hv && x.size(1) == v &&
          z.sizes() == x.sizes() && norm_weight.dim() == 1 &&
          norm_weight.numel() == v && int4_weight.dim() == 2 &&
          int4_weight.size(0) > 0 &&
          int4_weight.size(0) <= std::numeric_limits<int>::max() &&
          int4_weight.size(1) == k / 2 && int4_scale.dim() == 2 &&
          int4_scale.size(0) == int4_weight.size(0) &&
          int4_scale.size(1) == k / 128 && output.dim() == 2 &&
          output.size(0) == 1 && output.size(1) == int4_weight.size(0),
      "invalid GDN INT4 output shapes");
  for (const auto* input : {&x, &z, &norm_weight, &int4_weight, &int4_scale})
    check_no_overlap(output, *input);

  c10::OptionalDeviceGuard guard(device);
  auto stream = c10::xpu::getCurrentXPUStream(device.index());
  auto& queue = stream.queue();
  TORCH_CHECK(
      queue.is_in_order(),
      "GDN INT4 projection requires an in-order current stream");

  // Preserve the established two-kernel path for offset/misaligned views.
  const uintptr_t alignment =
      reinterpret_cast<uintptr_t>(x.data_ptr()) |
      reinterpret_cast<uintptr_t>(z.data_ptr()) |
      reinterpret_cast<uintptr_t>(norm_weight.data_ptr()) |
      reinterpret_cast<uintptr_t>(int4_weight.data_ptr()) |
      reinterpret_cast<uintptr_t>(int4_scale.data_ptr()) |
      reinterpret_cast<uintptr_t>(output.data_ptr());
  if (hv == kHeads && sigmoid_gate && int4_weight.size(0) == 2560 &&
      (alignment & 3U) == 0) {
    // Record before submit for dropped inputs on a non-default stream.
    for (const auto* t : std::initializer_list<const torch::Tensor*>{
             &x, &z, &norm_weight, &int4_weight, &int4_scale, &output})
      c10::xpu::XPUCachingAllocator::recordStream(
          t->storage().data_ptr(), stream);
    const char* tile_env = std::getenv("VLLM_XPU_QWEN38_GDN_OUTPUT_TILE");
    const std::string_view tile = tile_env == nullptr ? "32" : tile_env;
    const auto launch = tile == "32" ? launch_fused<32> : launch_fused<16>;
    launch(
        queue,
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(z.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(norm_weight.data_ptr<at::Half>()),
        int4_weight.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(int4_scale.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        epsilon);
    return;
  }

  auto normalized = at::empty({1, k}, x.options());
  auto x3 = x.view({1, hv, v});
  auto z3 = z.view({1, hv, v});
  gdn_norm_gate_sycl(x3, z3, norm_weight, normalized, eps, sigmoid_gate);
  int4_linear(normalized, int4_weight, int4_scale, output);
}

}  // namespace vllm::qwen38
