// SPDX-License-Identifier: Apache-2.0
// Native subgroup block-read implementation for compact80/160. Include from
// moe_sycl.cpp only; this file deliberately does not register Torch ops.
#pragma once

#include <sycl/ext/oneapi/experimental/group_load_store.hpp>

namespace vllm::qwen38::moe_sycl::block {
namespace sx = sycl::ext::oneapi::experimental;

constexpr int kSG = 16;
constexpr int kWG = 128;
constexpr int kH = 2560;
constexpr int kE = 512;
constexpr int kR = 10;

template <typename T, int N, typename Group>
inline sycl::vec<T, N> load(Group group, const T* pointer) {
  constexpr auto props = sx::properties{
      sx::data_placement_striped,
      sx::contiguous_memory,
      sx::full_group,
      sx::alignment<4>};
  sycl::vec<T, N> data;
  sx::group_load(
      group,
      sycl::address_space_cast<
          sycl::access::address_space::global_space,
          sycl::access::decorated::yes>(pointer)
          .get_decorated(),
      data,
      props);
  return data;
}

inline float low_half(std::uint32_t pair) {
  return float(sycl::bit_cast<sycl::half>(std::uint16_t(pair)));
}

inline float high_half(std::uint32_t pair) {
  return float(sycl::bit_cast<sycl::half>(std::uint16_t(pair >> 16)));
}

inline int signed_low(std::uint8_t packed) {
  const int nibble = packed & 15;
  return nibble < 8 ? nibble : nibble - 16;
}

inline int signed_high(std::uint8_t packed) {
  const int nibble = packed >> 4;
  return nibble < 8 ? nibble : nibble - 16;
}

template <int Vec>
class RouterBlockKernel;
template <int I>
class UpBlockKernel;
template <int I, int Tile, int SG, bool Prefetch>
class DownBlockKernel;

template <int Vec = 4>
inline void launch_router(sycl::queue& queue, Inputs p, Workspace w) {
  static_assert(Vec == 4 || Vec == 8);
  constexpr int subgroups = kWG / kSG;
  queue.parallel_for<RouterBlockKernel<Vec>>(
      sycl::nd_range<1>((p.m * kE / subgroups) * kWG, kWG),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(kSG)]] {
        const int task = item.get_global_linear_id() / kSG;
        const int token = task / kE;
        const int expert = task % kE;
        const int lane = item.get_local_linear_id() % kSG;
        const auto group = item.get_sub_group();
        float accum[4] = {};
        const auto* input =
            reinterpret_cast<const std::uint32_t*>(p.x + token * kH);
        const auto* weight = p.router_weight + std::size_t(expert) * (kH / 2);
        const auto* scale = p.router_scale + expert * 20;
        for (int g = 0; g < 80 / Vec; ++g) {
          const auto x = load<std::uint32_t, Vec>(group, input + g * Vec * kSG);
          const auto q = load<std::uint8_t, Vec>(group, weight + g * Vec * kSG);
#pragma unroll
          for (int i = 0; i < Vec; ++i) {
            const float d = float(scale[g * (Vec / 4) + i / 4]);
            const float we = (int(q[i] & 15) - 8) * d;
            const float wo = (int(q[i] >> 4) - 8) * d;
            // 仍为四路累加器，按原20个scale组的顺序FMA，不能改为Vec路。
            accum[i % 4] = sycl::fma(low_half(x[i]), we, accum[i % 4]);
            accum[i % 4] = sycl::fma(high_half(x[i]), wo, accum[i % 4]);
          }
        }
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i)
          sum += accum[i];
        sum = sycl::reduce_over_group(group, sum, sycl::plus<float>());
        if (lane == 0) w.logits[token * kE + expert] = sycl::half(sum);
      });
}

template <int I>
inline void launch_up(sycl::queue& queue, Inputs p, Workspace w) {
#ifndef MOE_SYCL_UP_VEC
  #define MOE_SYCL_UP_VEC 8
#endif
  constexpr int vec = MOE_SYCL_UP_VEC;
  static_assert(vec == 4 || vec == 8 || vec == 16);
  const int routed_tasks = p.m * kR * I;
  const int total = routed_tasks + p.m * (I + 1);
  const std::size_t groups = (total + 7) / 8;
  queue.parallel_for<UpBlockKernel<I>>(
      sycl::nd_range<1>(groups * kWG, kWG),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(kSG)]] {
        const int task = item.get_global_linear_id() / kSG;
        if (task >= total) return;
        const int lane = item.get_local_linear_id() % kSG;
        const auto group = item.get_sub_group();
        const int route_row = task / I;
        const int token = task < routed_tasks ? route_row / kR
                                              : (task - routed_tasks) / (I + 1);
        const auto* input =
            reinterpret_cast<const std::uint32_t*>(p.x + token * kH);
        float gate[vec] = {};
        float up[vec] = {};
        if (task < routed_tasks) {
          const int col = task % I;
          const int expert = w.ids[route_row];
          const std::size_t gate_index = std::size_t(expert * (2 * I) + col);
          const std::size_t up_index = gate_index + I;
          const auto* gate_weight = p.w13 + gate_index * (kH / 2);
          const auto* up_weight = p.w13 + up_index * (kH / 2);
          const auto* gate_scale = p.s13 + gate_index * 20;
          const auto* up_scale = p.s13 + up_index * 20;
          for (int g = 0; g < 80 / vec; ++g) {
            const auto x =
                load<std::uint32_t, vec>(group, input + g * vec * kSG);
            const auto qg =
                load<std::uint8_t, vec>(group, gate_weight + g * vec * kSG);
            const auto qu =
                load<std::uint8_t, vec>(group, up_weight + g * vec * kSG);
#pragma unroll
            for (int i = 0; i < vec; ++i) {
              const float dg = float(gate_scale[g * (vec / 4) + i / 4]);
              const float du = float(up_scale[g * (vec / 4) + i / 4]);
              gate[i] =
                  sycl::fma(low_half(x[i]), signed_low(qg[i]) * dg, gate[i]);
              gate[i] =
                  sycl::fma(high_half(x[i]), signed_high(qg[i]) * dg, gate[i]);
              up[i] = sycl::fma(low_half(x[i]), signed_low(qu[i]) * du, up[i]);
              up[i] =
                  sycl::fma(high_half(x[i]), signed_high(qu[i]) * du, up[i]);
            }
          }
        } else {
          const int col = (task - routed_tasks) % (I + 1);
          const auto* gate_weight = reinterpret_cast<const std::uint32_t*>(
              col == I ? p.shared_gate : p.shared_up + col * kH);
          const auto* up_weight = reinterpret_cast<const std::uint32_t*>(
              col == I ? p.shared_gate : p.shared_up + (I + col) * kH);
          for (int g = 0; g < 80 / vec; ++g) {
            const auto x =
                load<std::uint32_t, vec>(group, input + g * vec * kSG);
            const auto wg =
                load<std::uint32_t, vec>(group, gate_weight + g * vec * kSG);
            sycl::vec<std::uint32_t, vec> wu;
            if (col != I)
              wu = load<std::uint32_t, vec>(group, up_weight + g * vec * kSG);
#pragma unroll
            for (int i = 0; i < vec; ++i) {
              gate[i] = sycl::fma(low_half(x[i]), low_half(wg[i]), gate[i]);
              gate[i] = sycl::fma(high_half(x[i]), high_half(wg[i]), gate[i]);
              if (col != I) {
                up[i] = sycl::fma(low_half(x[i]), low_half(wu[i]), up[i]);
                up[i] = sycl::fma(high_half(x[i]), high_half(wu[i]), up[i]);
              }
            }
          }
        }
        float gate_sum = 0.0f;
        float up_sum = 0.0f;
#pragma unroll
        for (int i = 0; i < vec; ++i) {
          gate_sum += gate[i];
          up_sum += up[i];
        }
        gate_sum =
            sycl::reduce_over_group(group, gate_sum, sycl::plus<float>());
        up_sum = sycl::reduce_over_group(group, up_sum, sycl::plus<float>());
        if (lane == 0) {
          if (task < routed_tasks) {
            w.routed[task] =
                sycl::half((gate_sum / (1.0f + sycl::exp(-gate_sum))) * up_sum);
          } else {
            const int shared_task = task - routed_tasks;
            const int col = shared_task % (I + 1);
            if (col == I)
              w.gates[token] = 1.0f / (1.0f + sycl::exp(-gate_sum));
            else
              w.shared[token * I + col] = sycl::half(
                  (gate_sum / (1.0f + sycl::exp(-gate_sum))) * up_sum);
          }
        }
      });
}

template <int I, int Tile = 8, int SG = 16, bool Prefetch = false>
inline void launch_down(sycl::queue& queue, Inputs p, Workspace w) {
  constexpr int tile = Tile;
  constexpr int subgroups = 12;
  constexpr int local = subgroups * SG;
  constexpr int head_vec = (I == 160 ? 64 : 32) / SG;
  constexpr int tail_lanes = I == 160 ? 16 : 8;
  constexpr int tiles = kH / tile;
  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> partial(sycl::range<1>(12 * tile), cgh);
    cgh.parallel_for<DownBlockKernel<I, Tile, SG, Prefetch>>(
        sycl::nd_range<1>(std::size_t(p.m) * tiles * local, local),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG)]] {
          const int lane = item.get_local_linear_id() % SG;
          const int route = item.get_local_linear_id() / SG;
          const int token = item.get_group_linear_id() / tiles;
          const int hbase = (item.get_group_linear_id() % tiles) * tile;
          const auto group = item.get_sub_group();
          if (route < kR) {
            const int expert = w.ids[token * kR + route];
            const auto* input = w.routed + (token * kR + route) * I;
            const auto x0 = load<std::uint32_t, head_vec>(
                group, reinterpret_cast<const std::uint32_t*>(input));
            std::uint32_t tail_x = 0;
            if constexpr (I == 160 && SG == 16) {
              tail_x = load<std::uint32_t, 1>(
                  group,
                  reinterpret_cast<const std::uint32_t*>(input + 128))[0];
            } else if (lane < tail_lanes) {
              tail_x = reinterpret_cast<const std::uint32_t*>(
                  input)[(I == 160 ? 64 : 32) + lane];
            }
            struct RoutedRow {
              sycl::vec<std::uint8_t, head_vec> q;
              std::uint8_t tail;
              float d0;
              float d1;
            };
            const auto fetch = [&](int h) {
              const std::size_t row = std::size_t(expert) * kH + hbase + h;
              const auto* weight = p.w2 + row * (I / 2);
              const auto q0 = load<std::uint8_t, head_vec>(group, weight);
              std::uint8_t tail_q = 0;
              if constexpr (I == 160 && SG == 16)
                tail_q = load<std::uint8_t, 1>(group, weight + 64)[0];
              else if (lane < tail_lanes)
                tail_q = weight[(I == 160 ? 64 : 32) + lane];
              const float d0 = float(p.s2[row * ((I + 127) / 128)]);
              const float d1 = I == 160 ? float(p.s2[row * 2 + 1]) : d0;
              return RoutedRow{q0, tail_q, d0, d1};
            };
            RoutedRow next;
            if constexpr (Prefetch) next = fetch(0);
#pragma unroll
            for (int h = 0; h < tile; ++h) {
              const RoutedRow data = Prefetch ? next : fetch(h);
              // 提前发出下一行的读取，不增加tile、不改变当前行累加顺序。
              if constexpr (Prefetch)
                if (h + 1 < tile) next = fetch(h + 1);
              const auto q0 = data.q;
              const auto tail_q = data.tail;
              const auto d0 = data.d0;
              const auto d1 = data.d1;
              float acc[head_vec] = {};
#pragma unroll
              for (int i = 0; i < head_vec; ++i) {
                acc[i] =
                    sycl::fma(low_half(x0[i]), signed_low(q0[i]) * d0, acc[i]);
                acc[i] = sycl::fma(
                    high_half(x0[i]), signed_high(q0[i]) * d0, acc[i]);
              }
              if constexpr (I == 160) {
                acc[0] = sycl::fma(
                    low_half(tail_x), signed_low(tail_q) * d1, acc[0]);
                acc[0] = sycl::fma(
                    high_half(tail_x), signed_high(tail_q) * d1, acc[0]);
              } else if (lane < tail_lanes) {
                acc[0] = sycl::fma(
                    low_half(tail_x), signed_low(tail_q) * d0, acc[0]);
                acc[0] = sycl::fma(
                    high_half(tail_x), signed_high(tail_q) * d0, acc[0]);
              }
              float sum = 0.0f;
#pragma unroll
              for (int i = 0; i < head_vec; ++i)
                sum += acc[i];
              sum = sycl::reduce_over_group(group, sum, sycl::plus<float>());
              if (lane == 0) partial[route * tile + h] = sum;
            }
          } else if (route < 12) {
            const auto* input = w.shared + token * I;
            sycl::vec<std::uint32_t, head_vec> x0;
            std::uint32_t tail_x = 0;
            if (route == kR) {
              x0 = load<std::uint32_t, head_vec>(
                  group, reinterpret_cast<const std::uint32_t*>(input));
            } else if constexpr (I == 160 && SG == 16) {
              tail_x = load<std::uint32_t, 1>(
                  group,
                  reinterpret_cast<const std::uint32_t*>(input + 128))[0];
            } else if (lane < tail_lanes) {
              tail_x = reinterpret_cast<const std::uint32_t*>(
                  input)[(I == 160 ? 64 : 32) + lane];
            }
#pragma unroll
            for (int h = 0; h < tile; ++h) {
              const auto* weight = reinterpret_cast<const std::uint32_t*>(
                  p.shared_down + (hbase + h) * I);
              float result = 0.0f;
              if (route == kR) {
                const auto w0 = load<std::uint32_t, head_vec>(group, weight);
                float first[head_vec] = {};
#pragma unroll
                for (int i = 0; i < head_vec; ++i) {
                  first[i] =
                      sycl::fma(low_half(x0[i]), low_half(w0[i]), first[i]);
                  first[i] =
                      sycl::fma(high_half(x0[i]), high_half(w0[i]), first[i]);
                }
#pragma unroll
                for (int i = 0; i < head_vec; ++i)
                  result += first[i];
              } else {
                std::uint32_t tail_w = 0;
                if constexpr (I == 160 && SG == 16)
                  tail_w = load<std::uint32_t, 1>(group, weight + 64)[0];
                else if (lane < tail_lanes)
                  tail_w = weight[(I == 160 ? 64 : 32) + lane];
                if constexpr (I == 160 && SG == 16)
                  result = low_half(tail_x) * low_half(tail_w) +
                           high_half(tail_x) * high_half(tail_w);
                else if (lane < tail_lanes)
                  result = low_half(tail_x) * low_half(tail_w) +
                           high_half(tail_x) * high_half(tail_w);
              }
              result =
                  sycl::reduce_over_group(group, result, sycl::plus<float>());
              if (lane == 0) partial[route * tile + h] = result;
            }
          }
          item.barrier(sycl::access::fence_space::local_space);
          if (route == 0 && lane < tile) {
            float sum = 0.0f;
#pragma unroll
            for (int r = 0; r < kR; ++r)
              sum +=
                  float(w.weights[token * kR + r]) * partial[r * tile + lane];
            sum += w.gates[token] * partial[10 * tile + lane];
            sum += w.gates[token] * partial[11 * tile + lane];
            p.output[token * kH + hbase + lane] = sycl::half(sum);
          }
        });
  });
}

}  // namespace vllm::qwen38::moe_sycl::block
