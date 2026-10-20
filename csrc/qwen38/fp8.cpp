// SPDX-License-Identifier: Apache-2.0
#include <ATen/MemoryOverlap.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUStream.h>
#include <torch/extension.h>
#include <cute/tensor.hpp>
#include <cute/arch/mma_xe.hpp>
#include <cute/arch/mma_xe_legacy.hpp>
#include <cute/arch/copy_xe_legacy_U8.hpp>
#include <sycl/ext/oneapi/experimental/group_load_store.hpp>
#include <sycl/ext/oneapi/experimental/device_architecture.hpp>
#include <sycl/sycl.hpp>

#include <cstdint>
#include <cmath>
#include <limits>
#include <optional>
#include <unordered_map>
#include <vector>

#include "../xpu/gemv/xe_2/fp8_gemv.hpp"

namespace {
using half = sycl::half;
constexpr int H = 2560, E = 512, TopK = 10, Routes = 11, SG = 16;
namespace gemv = vllm::fp8_gemv;

// Scale the exact bit decode by 2^-8 so normals and subnormals use the same
// integer conversion. Restore 2^8 after the FP32 dot product, before rounding.
inline std::uint32_t decode_pair(std::uint32_t word, bool odd) {
  const auto raw = odd ? word >> 8 : word;
  auto bits = ((raw & 0x00800080U) << 8) | ((raw & 0x007f007fU) << 7);
  if ((raw & 0x7fU) == 0x7fU) bits = (bits & 0xffff0000U) | 0x7e00U;
  if (((raw >> 16) & 0x7fU) == 0x7fU) bits = (bits & 0x0000ffffU) | 0x7e000000U;
  return bits;
}

template <bool NK>
inline sycl::vec<std::uint32_t, 8>
load_weight(const std::uint8_t* w, int n, int k, int nb, int kb) {
  sycl::vec<std::uint32_t, 8> raw;
#ifdef __SYCL_DEVICE_ONLY__
  if constexpr (NK) {
    cute::intel::coord_t coord = {kb / 4, nb};
    cute::detail::XeSubgroup2DBlockLoadTranspose<4, 8, 16, 1>{}(
        w, k, n, k, coord, &raw);
  } else {
    cute::intel::coord_t coord = {nb, kb};
    cute::XE_2D_U8x32x16_LD_V::copy(
        w, n, k, n, coord, reinterpret_cast<std::uint8_t*>(&raw));
  }
#endif
  return raw;
}

template <int Rows>
inline void multiply(
    const sycl::vec<std::uint32_t, 8>& raw,
    const typename gemv::Dpas<Rows>::A& ae,
    const typename gemv::Dpas<Rows>::A& ao,
    typename gemv::Dpas<Rows>::C& ce,
    typename gemv::Dpas<Rows>::C& co) {
  cute::intel::int8 be, bo;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    be[i] = decode_pair(raw[i], false);
    bo[i] = decode_pair(raw[i], true);
  }
  gemv::Dpas<Rows>::mma(ce, ae, be);
  gemv::Dpas<Rows>::mma(co, ao, bo);
}

template <int Rows, int Splits>
class Linear;
template <int Rows, int Splits>
void linear(
    sycl::queue& q,
    const half* x,
    const std::uint8_t* weight,
    const float* scale,
    half* out,
    int m,
    int n,
    int k) {
  using D = gemv::Dpas<Rows>;
  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float> partial(Splits * SG * Rows, cgh);
    cgh.parallel_for<Linear<Rows, Splits>>(
        sycl::nd_range<1>(((n + SG - 1) / SG) * SG * Splits, SG * Splits),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG)]] {
          const int lane = item.get_local_linear_id() % SG;
          const int split = item.get_local_linear_id() / SG;
          const int nb = item.get_group_linear_id() * SG;
          typename D::C ce{}, co{};
          for (int kb = split * 32; kb < k; kb += Splits * 32) {
            typename D::A ae{}, ao{};
#pragma unroll
            for (int r = 0; r < Rows; ++r) {
              gemv::vset<Rows>(
                  ae,
                  r,
                  r < m && kb + 2 * lane < k
                      ? short(sycl::bit_cast<std::uint16_t>(
                            x[r * k + kb + 2 * lane]))
                      : short(0));
              gemv::vset<Rows>(
                  ao,
                  r,
                  r < m && kb + 2 * lane + 1 < k
                      ? short(sycl::bit_cast<std::uint16_t>(
                            x[r * k + kb + 2 * lane + 1]))
                      : short(0));
            }
            auto raw = load_weight<true>(weight, n, k, nb, kb);
            multiply<Rows>(raw, ae, ao, ce, co);
          }
#pragma unroll
          for (int r = 0; r < Rows; ++r)
            partial[(split * SG + lane) * Rows + r] =
                gemv::vget<Rows>(ce, r) + gemv::vget<Rows>(co, r);
          item.barrier(sycl::access::fence_space::local_space);
          if (split == 0 && nb + lane < n) {
#pragma unroll
            for (int r = 0; r < Rows; ++r) {
              if (r < m) {
                float sum = 0;
#pragma unroll
                for (int s = 0; s < Splits; ++s)
                  sum += partial[(s * SG + lane) * Rows + r];
                out[r * n + nb + lane] = half(sum * (scale[0] * 256));
              }
            }
          }
        });
  });
}

class Router;
void router_scalar(
    sycl::queue& q,
    const half* x,
    const half* rw,
    const half* sw,
    half* logits,
    half* gate,
    int m) {
  constexpr int WG = 128;
  const int tasks = m * (E + 1);
  q.parallel_for<Router>(
      sycl::nd_range<1>(((tasks + 7) / 8) * WG, WG),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG)]] {
        const int task = item.get_global_linear_id() / SG;
        if (task >= tasks) return;
        const int t = task / (E + 1), e = task % (E + 1);
        const int lane = item.get_local_linear_id() % SG;
        const half* w = e == E ? sw : rw + e * H;
        float acc[4] = {};
        for (int kb = lane; kb < H; kb += 4 * SG) {
#pragma unroll
          for (int j = 0; j < 4; ++j)
            acc[j] = sycl::fma(
                float(x[t * H + kb + j * SG]), float(w[kb + j * SG]), acc[j]);
        }
        float sum = acc[0] + acc[1] + acc[2] + acc[3];
        sum =
            sycl::reduce_over_group(item.get_sub_group(), sum, sycl::plus<>());
        if (lane == 0) {
          const half value = half(sum);
          if (e == E)
            gate[t] = half(1 / (1 + sycl::exp(-float(value))));
          else
            logits[t * E + e] = value;
        }
      });
}

inline cute::intel::int8
load_router_weight(const half* rw, int n, int nb, int kb) {
  cute::intel::int8 b{};
#ifdef __SYCL_DEVICE_ONLY__
  cute::intel::coord_t coord = {kb / 2, nb};
  cute::detail::XeSubgroup2DBlockLoadTranspose<4, 8, 16, 1>{}(
      rw, H * 2, n, H * 2, coord, &b);
#endif
  return b;
}
template <int Splits>
class RouterDpas;
template <int Splits>
void router_dpas(
    sycl::queue& q,
    const half* x,
    const half* rw,
    const half* sw,
    half* logits,
    half* gate) {
  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float> partial(Splits * SG, cgh);
    cgh.parallel_for<RouterDpas<Splits>>(
        sycl::nd_range<1>((E / SG + 1) * Splits * SG, Splits * SG),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
          const int lane = it.get_local_linear_id() % SG,
                    split = it.get_local_linear_id() / SG;
          const int nb = it.get_group_linear_id() * SG;
          const half* weight = nb == E ? sw : rw;
          const int weight_n = nb == E ? 1 : E, weight_nb = nb == E ? 0 : nb;
          float c = 0;
          for (int kb = split * SG; kb < H; kb += Splits * SG) {
            const short a = short(sycl::bit_cast<std::uint16_t>(x[kb + lane]));
            const auto b = load_router_weight(weight, weight_n, weight_nb, kb);
            cute::XE_1x16x16_F32F16F16F32_TT::fma(c, a, b, c);
          }
          partial[split * SG + lane] = c;
          it.barrier(sycl::access::fence_space::local_space);
          if (split == 0) {
            float sum = 0;
#pragma unroll
            for (int j = 0; j < Splits; ++j)
              sum += partial[j * SG + lane];
            if (nb == E) {
              if (lane == 0)
                gate[0] = half(1 / (1 + sycl::exp(-float(half(sum)))));
            } else
              logits[nb + lane] = half(sum);
          }
        });
  });
}
// M1 uses 16 independent K partitions. Matrix accumulation stays FP32 and
// each logit/shared gate is rounded to FP16 at the original consumer boundary.
// Larger batches retain the scalar path; no weight-content cache is introduced.
void router(
    sycl::queue& q,
    const half* x,
    const half* rw,
    const half* sw,
    half* logits,
    half* gate,
    int m) {
  if (m == 1 && std::uintptr_t(rw) % 64 == 0 && std::uintptr_t(sw) % 64 == 0)
    router_dpas<16>(q, x, rw, sw, logits, gate);
  else
    router_scalar(q, x, rw, sw, logits, gate, m);
}

class Top;
void topk(sycl::queue& q, const half* logits, int* ids, float* probs, int m) {
  q.parallel_for<Top>(
      sycl::nd_range<1>(m * 32, 32),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
        const int t = item.get_group_linear_id(),
                  lane = item.get_local_linear_id();
        const auto group = item.get_sub_group();
        std::uint32_t keys[16];
        int invalid = 0;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
          const int id = j * 32 + lane;
          const auto value = logits[t * E + id];
          invalid |= !sycl::isfinite(float(value));
          auto bits = sycl::bit_cast<std::uint16_t>(value);
          if ((bits & 0x7fffU) == 0) bits = 0;
          const std::uint16_t order = bits & 0x8000U
                                          ? std::uint16_t(~bits)
                                          : std::uint16_t(bits ^ 0x8000U);
          keys[j] = (std::uint32_t(order) << 9) | (511U - id);
        }
        invalid = sycl::reduce_over_group(group, invalid, sycl::maximum<>());
        float selected[TopK];
#pragma unroll
        for (int r = 0; r < TopK; ++r) {
          std::uint32_t best = 0;
#pragma unroll
          for (int j = 0; j < 16; ++j)
            best = sycl::max(best, keys[j]);
          best = sycl::reduce_over_group(group, best, sycl::maximum<>());
          const int id = 511 - (best & 511U);
          if (lane == 0) {
            ids[t * TopK + r] = invalid ? r : id;
            selected[r] = float(logits[t * E + id]);
          }
#pragma unroll
          for (int j = 0; j < 16; ++j)
            if (j * 32 + lane == id) keys[j] = 0;
        }
        if (lane == 0) {
          float sum = 0;
          const float maximum = selected[0];
#pragma unroll
          for (int r = 0; r < TopK; ++r) {
            selected[r] = sycl::exp(selected[r] - maximum);
            sum += selected[r];
          }
#pragma unroll
          for (int r = 0; r < TopK; ++r)
            probs[t * TopK + r] = invalid
                                      ? std::numeric_limits<float>::quiet_NaN()
                                      : selected[r] / sum;
        }
      });
}

template <int I, int Splits>
class Up;
template <int I, int Splits>
void up(
    sycl::queue& q,
    const half* x,
    const std::uint8_t* w,
    const float* scale,
    const std::uint8_t* shared,
    const float* ss,
    const int* ids,
    half* inter,
    int m) {
  using D = gemv::Dpas<1>;
  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float> partial(Splits * SG * 2, cgh);
    cgh.parallel_for<Up<I, Splits>>(
        sycl::nd_range<1>(m * Routes * (I / SG) * SG * Splits, SG * Splits),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG)]] {
          const int task = item.get_group_linear_id();
          const int row = task / (I / SG), nb = task % (I / SG) * SG;
          const int t = row / Routes, r = row % Routes;
          const int lane = item.get_local_linear_id() % SG;
          const int split = item.get_local_linear_id() / SG;
          const bool sh = r == TopK;
          const int e = sh ? 0 : ids[t * TopK + r];
          const auto* base = sh ? shared : w + std::size_t(e) * H * 2 * I;
          typename D::C gate_even{}, gate_odd{}, up_even{}, up_odd{};
          for (int kb = split * 32; kb < H; kb += Splits * 32) {
            const short ae =
                short(sycl::bit_cast<std::uint16_t>(x[t * H + kb + 2 * lane]));
            const short ao = short(
                sycl::bit_cast<std::uint16_t>(x[t * H + kb + 2 * lane + 1]));
            const auto gate_weight =
                sh ? load_weight<true>(base, 2 * I, H, nb, kb)
                   : load_weight<false>(base, 2 * I, H, nb, kb);
            const auto up_weight =
                sh ? load_weight<true>(base, 2 * I, H, nb + I, kb)
                   : load_weight<false>(base, 2 * I, H, nb + I, kb);
            multiply<1>(gate_weight, ae, ao, gate_even, gate_odd);
            multiply<1>(up_weight, ae, ao, up_even, up_odd);
          }
          partial[(split * SG + lane) * 2] = gate_even + gate_odd;
          partial[(split * SG + lane) * 2 + 1] = up_even + up_odd;
          item.barrier(sycl::access::fence_space::local_space);
          if (split == 0) {
            float g = 0, u = 0;
#pragma unroll
            for (int s = 0; s < Splits; ++s) {
              g += partial[(s * SG + lane) * 2];
              u += partial[(s * SG + lane) * 2 + 1];
            }
            const float sc = (sh ? ss[0] : scale[e]) * 256;
            g = float(half(g * sc));
            u = float(half(u * sc));
            inter[row * I + nb + lane] = half((g / (1 + sycl::exp(-g))) * u);
          }
        });
  });
}

template <int I, bool Merge>
class Down;
template <int I, bool Merge>
void down(
    sycl::queue& q,
    const half* inter,
    const std::uint8_t* w,
    const float* scale,
    const std::uint8_t* shared,
    const float* ss,
    const int* ids,
    const float* probs,
    const half* gate,
    half* partial,
    int m) {
  using D = gemv::Dpas<1>;
  constexpr int WG = Merge ? Routes * SG : 128;
  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<half> sums(Merge ? Routes * SG : 1, cgh);
    cgh.parallel_for<Down<I, Merge>>(
        sycl::nd_range<1>(m * Routes * H, WG),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG)]] {
          const int task = item.get_global_linear_id() / SG;
          const int row =
              Merge ? (task / (Routes * H / SG)) * Routes + task % Routes
                    : task / (H / SG);
          const int nb =
              Merge ? (task / Routes) % (H / SG) * SG : task % (H / SG) * SG;
          const int t = row / Routes, r = row % Routes;
          const int lane = item.get_local_linear_id() % SG;
          const bool sh = r == TopK;
          const int e = sh ? 0 : ids[t * TopK + r];
          const auto* base = sh ? shared : w + std::size_t(e) * I * H;
          typename D::C ce{}, co{};
#pragma unroll
          for (int kb = 0; kb < I; kb += 32) {
            const short ae = kb + 2 * lane < I
                                 ? short(sycl::bit_cast<std::uint16_t>(
                                       inter[row * I + kb + 2 * lane]))
                                 : short(0);
            const short ao = kb + 2 * lane + 1 < I
                                 ? short(sycl::bit_cast<std::uint16_t>(
                                       inter[row * I + kb + 2 * lane + 1]))
                                 : short(0);
            auto raw = sh ? load_weight<true>(base, H, I, nb, kb)
                          : load_weight<false>(base, H, I, nb, kb);
            multiply<1>(raw, ae, ao, ce, co);
          }
          const float acc = (ce + co) * 256;
          float value = sh ? float(half(acc * ss[0])) * float(gate[t])
                           : acc * (scale[e] * probs[t * TopK + r]);
          if constexpr (Merge) {
            sums[r * SG + lane] = half(value);
            item.barrier(sycl::access::fence_space::local_space);
            if (r == 0) {
              float sum = 0;
#pragma unroll
              for (int rank = 0; rank < TopK; ++rank)
                sum += float(sums[rank * SG + lane]);
              partial[t * H + nb + lane] =
                  half(float(half(sum)) + float(sums[TopK * SG + lane]));
            }
          } else {
            partial[row * H + nb + lane] = half(value);
          }
        });
  });
}

template <int I, int Pad>
class DownMapped;
template <int I, int Pad>
void down_mapped(
    sycl::queue& q,
    const half* inter,
    const std::uint8_t* w,
    const float* scale,
    const std::uint8_t* shared,
    const float* ss,
    const int* ids,
    const float* probs,
    const half* gate,
    half* partial,
    int m) {
  using D = gemv::Dpas<1>;
  constexpr int WG = Pad * SG;
  constexpr bool Merge = true;
  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<half> sums(Pad * SG, cgh);
    cgh.parallel_for<DownMapped<I, Pad>>(
        sycl::nd_range<1>(m * (H / SG) * WG, WG),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG)]] {
          const int t = item.get_group_linear_id() / (H / SG);
          const int nb = item.get_group_linear_id() % (H / SG) * SG;
          const int r = item.get_local_linear_id() / SG;
          const int row = t * Routes + r;
          const int lane = item.get_local_linear_id() % SG;
          float value = 0;
          if (r < Routes) {
            const bool sh = r == TopK;
            const int e = sh ? 0 : ids[t * TopK + r];
            const auto* base = sh ? shared : w + std::size_t(e) * I * H;
            typename D::C ce{}, co{};
#pragma unroll
            for (int kb = 0; kb < I; kb += 32) {
              const short ae = kb + 2 * lane < I
                                   ? short(sycl::bit_cast<std::uint16_t>(
                                         inter[row * I + kb + 2 * lane]))
                                   : short(0);
              const short ao = kb + 2 * lane + 1 < I
                                   ? short(sycl::bit_cast<std::uint16_t>(
                                         inter[row * I + kb + 2 * lane + 1]))
                                   : short(0);
              auto raw = sh ? load_weight<true>(base, H, I, nb, kb)
                            : load_weight<false>(base, H, I, nb, kb);
              multiply<1>(raw, ae, ao, ce, co);
            }
            const float acc = (ce + co) * 256;
            value = sh ? float(half(acc * ss[0])) * float(gate[t])
                       : acc * (scale[e] * probs[t * TopK + r]);
          }
          if constexpr (Merge) {
            sums[r * SG + lane] = half(value);
            item.barrier(sycl::access::fence_space::local_space);
            if (r == 0) {
              float sum = 0;
#pragma unroll
              for (int rank = 0; rank < TopK; ++rank)
                sum += float(sums[rank * SG + lane]);
              partial[t * H + nb + lane] =
                  half(float(half(sum)) + float(sums[TopK * SG + lane]));
            }
          } else {
            partial[row * H + nb + lane] = half(value);
          }
        });
  });
}

class Reduce;
void reduce(sycl::queue& q, const half* partial, half* out, int m) {
  q.parallel_for<Reduce>(sycl::range<1>(m * H), [=](sycl::id<1> id) {
    const int t = id[0] / H, n = id[0] % H;
    float sum = 0;
#pragma unroll
    for (int r = 0; r < TopK; ++r)
      sum += float(partial[(t * Routes + r) * H + n]);
    out[id[0]] =
        half(float(half(sum)) + float(partial[(t * Routes + TopK) * H + n]));
  });
}

bool valid(
    const torch::Tensor& t,
    c10::Device device,
    at::ScalarType dtype,
    std::size_t alignment = 1) {
  return t.defined() && t.device() == device && t.scalar_type() == dtype &&
         t.layout() == c10::kStrided && t.is_contiguous() && !t.is_neg() &&
         !t.is_conj() && std::uintptr_t(t.const_data_ptr()) % alignment == 0;
}
bool supported(sycl::queue& q) {
  namespace sx = sycl::ext::oneapi::experimental;
  return q.is_in_order() &&
         q.get_device().get_info<sx::info::device::architecture>() ==
             sx::architecture::intel_gpu_bmg_g31;
}
void retain(const torch::Tensor& t, c10::xpu::XPUStream stream) {
  c10::xpu::XPUCachingAllocator::recordStream(t.storage().data_ptr(), stream);
}

template <typename Weight>
class GdnNorm;
template <typename Weight>
void norm(
    sycl::queue& q,
    const half* x,
    const half* z,
    const Weight* weight,
    half* out,
    int heads,
    float eps,
    bool sigmoid) {
  q.parallel_for<GdnNorm<Weight>>(
      sycl::nd_range<1>(heads * 32, 32),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
        const int row = item.get_group_linear_id();
        const auto sg = item.get_sub_group();
        namespace sx = sycl::ext::oneapi::experimental;
        constexpr auto props = sx::properties{
            sx::data_placement_striped,
            sx::contiguous_memory,
            sx::full_group,
            sx::alignment<4>};
        auto pointer = [](auto* p) {
          return sycl::address_space_cast<
                     sycl::access::address_space::global_space,
                     sycl::access::decorated::yes>(p)
              .get_decorated();
        };
        sycl::vec<half, 4> xv, zv, result;
        sycl::vec<Weight, 4> wv;
        sx::group_load(sg, pointer(x + row * 128), xv, props);
        sx::group_load(sg, pointer(z + row * 128), zv, props);
        sx::group_load(sg, pointer(weight), wv, props);
        float values[4], squares[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          values[j] = float(xv[j]);
          squares[j] = values[j] * values[j];
        }
        const float sum = sycl::reduce_over_group(
            item.get_sub_group(),
            (squares[0] + squares[2]) + (squares[1] + squares[3]),
            sycl::plus<>());
        const float inv = sycl::rsqrt(sum / 128 + eps);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float gate = float(zv[j]);
          const float activated = 1 / (1 + sycl::native::exp(-gate));
          const float value = values[j] * inv * float(wv[j]);
          result[j] = half(value * (sigmoid ? activated : gate * activated));
        }
        sx::group_store(sg, result, pointer(out + row * 128), props);
      });
}

std::optional<torch::Tensor> try_gdn_norm(
    const torch::Tensor& x,
    const torch::Tensor& z,
    const torch::Tensor& weight,
    double eps,
    bool sigmoid) {
  if (!x.defined() || !x.is_xpu() || x.dim() != 3 || x.size(0) < 1 ||
      x.size(0) > 4096 ||
      (x.size(1) != 6 && x.size(1) != 12 &&
       !((x.size(1) == 24 || x.size(1) == 48) && x.size(0) <= 8)) ||
      x.size(2) != 128 || !weight.defined() ||
      !valid(x, x.device(), at::kHalf, 4) ||
      !valid(z, x.device(), at::kHalf, 4) || z.sizes() != x.sizes() ||
      weight.numel() != 128 || weight.dim() != 1 ||
      (!valid(weight, x.device(), at::kHalf, 4) &&
       !valid(weight, x.device(), at::kFloat, 4)) ||
      !std::isfinite(float(eps)) || float(eps) <= 0)
    return std::nullopt;
  c10::DeviceGuard guard(x.device());
  auto stream = c10::xpu::getCurrentXPUStream(x.device().index());
  if (!stream.queue().is_in_order()) return std::nullopt;
  auto out = torch::empty_like(x);
  retain(x, stream);
  retain(z, stream);
  retain(weight, stream);
  auto submit = [&]<typename Weight>() {
    norm<Weight>(
        stream.queue(),
        (const half*)x.const_data_ptr(),
        (const half*)z.const_data_ptr(),
        (const Weight*)weight.const_data_ptr(),
        (half*)out.data_ptr(),
        x.size(0) * x.size(1),
        float(eps),
        sigmoid);
  };
  if (weight.scalar_type() == at::kHalf)
    submit.template operator()<half>();
  else
    submit.template operator()<float>();
  return out;
}

std::optional<torch::Tensor> try_linear(
    const torch::Tensor& x,
    const torch::Tensor& w,
    const torch::Tensor& scale) {
  if (!x.defined() || !x.is_xpu() || x.dim() != 2 || !w.defined() ||
      w.dim() != 2 || x.size(0) < 1 || x.size(0) > 8 ||
      w.size(1) != x.size(1) || w.size(0) > std::numeric_limits<int>::max() ||
      x.size(1) > std::numeric_limits<int>::max())
    return std::nullopt;
  const int m = x.size(0), n = w.size(0), k = x.size(1);
  const bool shape = (k == H && (n == 12 || n == 24 || n == 160 || n == 320 ||
                                 n == 640 || n == 2048 || n == 3584 ||
                                 n == 4096 || n == H || n == 10240)) ||
                     (n == H && (k == 80 || k == 160 || k == 768 || k == 1536));
  if (!shape || !valid(x, x.device(), at::kHalf, 2) ||
      !valid(w, x.device(), at::kFloat8_e4m3fn, 64) ||
      !valid(scale, x.device(), at::kFloat, 4) || scale.numel() != 1)
    return std::nullopt;
  c10::DeviceGuard guard(x.device());
  auto stream = c10::xpu::getCurrentXPUStream(x.device().index());
  if (!supported(stream.queue())) return std::nullopt;
  auto out = torch::empty({m, n}, x.options());
  retain(x, stream);
  retain(w, stream);
  retain(scale, stream);
  auto run = [&]<int Rows>() {
    linear<Rows, 4>(
        stream.queue(),
        (const half*)x.const_data_ptr(),
        (const std::uint8_t*)w.const_data_ptr(),
        scale.const_data_ptr<float>(),
        (half*)out.data_ptr(),
        m,
        n,
        k);
  };
  if (m == 1)
    run.template operator()<1>();
  else if (m == 2)
    run.template operator()<2>();
  else if (m <= 4)
    run.template operator()<4>();
  else
    run.template operator()<8>();
  return out;
}

struct Scratch {
  int m, inter;
  torch::Tensor logits, gate, indices, probs, hidden, partial;
};
class Workspace {
  std::unordered_map<c10::xpu::XPUStream, std::vector<Scratch>> buffers;
  int up_splits;
  int fuse_down;

 public:
  Workspace(int splits = -1, int fuse = -1)
      : up_splits(splits), fuse_down(fuse) {
    TORCH_CHECK(
        splits == -1 || splits == 4 || splits == 8 || splits == 16 ||
            splits == 20,
        "up_splits must be -1, 4, 8, 16 or 20");
    TORCH_CHECK(fuse >= -1 && fuse <= 1, "fuse_down must be -1, 0 or 1");
  }
  std::optional<torch::Tensor> try_run(
      const torch::Tensor& x,
      const torch::Tensor& rw,
      const std::vector<torch::Tensor>& w) {
    if (!x.defined() || !x.is_xpu() || x.dim() != 2 || x.size(1) != H ||
        x.size(0) < 1 || x.size(0) > 8 || w.size() != 9)
      return std::nullopt;
    const auto dev = x.device();
    if (!valid(x, dev, at::kHalf, 16) || !valid(rw, dev, at::kHalf, 16) ||
        rw.sizes() != at::IntArrayRef({E, H}))
      return std::nullopt;
    for (int j = 0; j < 8; ++j)
      if (!valid(
              w[j],
              dev,
              j % 2 ? at::kFloat : at::kFloat8_e4m3fn,
              j % 2 ? 4 : 64))
        return std::nullopt;
    if (!valid(w[8], dev, at::kHalf, 16) ||
        w[8].sizes() != at::IntArrayRef({1, H}) || w[0].dim() != 3)
      return std::nullopt;
    const int i = w[0].size(2) / 2;
    if ((i != 80 && i != 160) ||
        w[0].sizes() != at::IntArrayRef({E, H, 2 * i}) ||
        w[2].sizes() != at::IntArrayRef({E, i, H}) ||
        w[4].sizes() != at::IntArrayRef({2 * i, H}) ||
        w[6].sizes() != at::IntArrayRef({H, i}) || w[1].numel() != E ||
        w[3].numel() != E || w[5].numel() != 1 || w[7].numel() != 1)
      return std::nullopt;
    c10::DeviceGuard guard(dev);
    const int m = x.size(0);
    const int splits = up_splits >= 0       ? up_splits
                       : m == 1             ? (i == 80 ? 20 : 8)
                       : i == 80 && m >= 5  ? 4
                       : i == 160 && m == 5 ? 16
                                            : 8;
    const bool merge = fuse_down >= 0 ? fuse_down != 0 : m == 1;
    auto stream = c10::xpu::getCurrentXPUStream(dev.index());
    if (!supported(stream.queue())) return std::nullopt;
    auto& list = buffers[stream];
    Scratch* b = nullptr;
    for (auto& candidate : list)
      if (candidate.m == m && candidate.inter == i) {
        b = &candidate;
        break;
      }
    if (!b) {
      auto opts = x.options();
      // Allocate the complete entry before publishing it. Each stream and
      // exact capture shape owns its scratch; replay never allocates it again.
      Scratch next{
          m,
          i,
          torch::empty({m, E}, opts),
          torch::empty({m}, opts),
          torch::empty({m, TopK}, opts.dtype(at::kInt)),
          torch::empty({m, TopK}, opts.dtype(at::kFloat)),
          torch::empty({m, Routes, i}, opts),
          torch::empty({m, Routes, H}, opts)};
      list.push_back(std::move(next));
      b = &list.back();
    }
    auto out = torch::empty_like(x);
    retain(x, stream);
    retain(rw, stream);
    for (const auto& weight : w)
      retain(weight, stream);
    auto hp = [](const torch::Tensor& t) { return (half*)t.data_ptr(); };
    auto bp = [](const torch::Tensor& t) {
      return (const std::uint8_t*)t.const_data_ptr();
    };
    auto& q = stream.queue();
    router(q, hp(x), hp(rw), hp(w[8]), hp(b->logits), hp(b->gate), m);
    topk(
        q,
        hp(b->logits),
        b->indices.data_ptr<int>(),
        b->probs.data_ptr<float>(),
        m);
    auto run = [&]<int I>() {
      auto submit_up = [&]<int Splits>() {
        up<I, Splits>(
            q,
            hp(x),
            bp(w[0]),
            w[1].const_data_ptr<float>(),
            bp(w[4]),
            w[5].const_data_ptr<float>(),
            b->indices.data_ptr<int>(),
            hp(b->hidden),
            m);
      };
      if (splits == 4)
        submit_up.template operator()<4>();
      else if (splits == 16)
        submit_up.template operator()<16>();
      else if (splits == 20)
        submit_up.template operator()<20>();
      else
        submit_up.template operator()<8>();
      auto submit_down = [&]<bool Merge>() {
        down<I, Merge>(
            q,
            hp(b->hidden),
            bp(w[2]),
            w[3].const_data_ptr<float>(),
            bp(w[6]),
            w[7].const_data_ptr<float>(),
            b->indices.data_ptr<int>(),
            b->probs.data_ptr<float>(),
            hp(b->gate),
            Merge ? hp(out) : hp(b->partial),
            m);
      };
      if constexpr (I == 80) {
        if (merge && m == 1) {
          down_mapped<I, 12>(
              q,
              hp(b->hidden),
              bp(w[2]),
              w[3].const_data_ptr<float>(),
              bp(w[6]),
              w[7].const_data_ptr<float>(),
              b->indices.data_ptr<int>(),
              b->probs.data_ptr<float>(),
              hp(b->gate),
              hp(out),
              m);
          return;
        }
      }
      if (merge)
        submit_down.template operator()<true>();
      else
        submit_down.template operator()<false>();
    };
    if (i == 80)
      run.template operator()<80>();
    else
      run.template operator()<160>();
    if (!merge) reduce(q, hp(b->partial), hp(out), m);
    return out;
  }
};
}  // namespace

void init_qwen4_dense_fp8(pybind11::module_& module);

PYBIND11_MODULE(_qwen38_fp8_C, module) {
  init_qwen4_dense_fp8(module);
  module.def("try_gdn_norm", &try_gdn_norm);
  module.def("try_qwen38_fp8_linear", &try_linear);
  pybind11::class_<Workspace>(module, "Workspace")
      .def(
          pybind11::init<int, int>(),
          pybind11::arg("up_splits") = -1,
          pybind11::arg("fuse_down") = -1)
      .def("try_run", &Workspace::try_run);
}
