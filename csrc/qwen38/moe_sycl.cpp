// SPDX-License-Identifier: Apache-2.0
#include "qwen38/moe_sycl.h"
#include "qwen38/moe_sycl_block.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string_view>

namespace vllm::qwen38::moe_sycl {
namespace {

constexpr int kHidden = 2560;
constexpr int kExperts = 512;
constexpr int kTopK = 10;
constexpr int kSubgroup = 16;
constexpr int kLocal = 128;
constexpr int kTile = 8;

inline int signed_nibble(std::uint8_t packed, bool high) {
  const int q = high ? packed >> 4 : packed & 15;
  return q < 8 ? q : q - 16;
}

inline int q40_nibble(std::uint8_t packed, bool high) {
  return (high ? packed >> 4 : packed & 15) - 8;
}

class RouterKernel;
class TopKKernel;
class TopKSubgroupKernel;
class TopKPackedSubgroupKernel;
class UpSharedKernel;
class DownReduceKernel;

void launch_router(sycl::queue& queue, Inputs p, Workspace w) {
  constexpr int subgroups = kLocal / kSubgroup;
  queue.parallel_for<RouterKernel>(
      sycl::nd_range<1>((p.m * kExperts / subgroups) * kLocal, kLocal),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(kSubgroup)]] {
        const int lane = item.get_local_linear_id() % kSubgroup;
        const int task = item.get_global_linear_id() / kSubgroup;
        const int expert = task % kExperts;
        const int token = task / kExperts;
        const auto sg = item.get_sub_group();
        float accum = 0.0f;
        const std::size_t row = std::size_t(expert) * (kHidden / 2);
        for (int kb = lane; kb < kHidden / 2; kb += kSubgroup) {
          const std::uint8_t packed = p.router_weight[row + kb];
          const float scale = float(p.router_scale[expert * 20 + kb / 64]);
          const int k = 2 * kb;
          accum += float(p.x[token * kHidden + k]) *
                       (q40_nibble(packed, false) * scale) +
                   float(p.x[token * kHidden + k + 1]) *
                       (q40_nibble(packed, true) * scale);
        }
        const float value = sycl::reduce_over_group(sg, accum, sycl::plus<>());
        if (lane == 0) w.logits[token * kExperts + expert] = sycl::half(value);
      });
}

void launch_topk(
    sycl::queue& queue, Inputs p, Workspace w, const sycl::half* logits) {
  queue.parallel_for<TopKKernel>(
      sycl::nd_range<1>(std::size_t(p.m) * kLocal, kLocal),
      [=](sycl::nd_item<1> item) {
        const int token = item.get_group_linear_id();
        const int lane = item.get_local_linear_id();
        const auto group = item.get_group();
        float scores[4];
        float max_local = -std::numeric_limits<float>::infinity();
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          scores[j] = float(logits[token * kExperts + lane + j * kLocal]);
          max_local = sycl::fmax(max_local, scores[j]);
        }
        const float max_row =
            sycl::reduce_over_group(group, max_local, sycl::maximum<float>());
        float sum_local = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          scores[j] = sycl::exp(scores[j] - max_row);
          sum_local += scores[j];
        }
        const float sum =
            sycl::reduce_over_group(group, sum_local, sycl::plus<float>());
        const float inverse = 1.0f / sum;
#pragma unroll
        for (int j = 0; j < 4; ++j)
          scores[j] *= inverse;

        float selected[kTopK];
#pragma unroll
        for (int rank = 0; rank < kTopK; ++rank) {
          float local_best = -1.0f;
#pragma unroll
          for (int j = 0; j < 4; ++j)
            local_best = sycl::fmax(local_best, scores[j]);
          const float best = sycl::reduce_over_group(
              group, local_best, sycl::maximum<float>());
          int local_id = kExperts;
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            if (scores[j] == best)
              local_id = sycl::min(local_id, lane + j * kLocal);
          }
          const int id =
              sycl::reduce_over_group(group, local_id, sycl::minimum<int>());
          if (lane == 0) {
            selected[rank] = best;
            w.ids[token * kTopK + rank] = sycl::isfinite(sum) ? id : rank;
          }
#pragma unroll
          for (int j = 0; j < 4; ++j)
            if (lane + j * kLocal == id) scores[j] = -1.0f;
        }
        if (lane == 0) {
          float top_sum = 0.0f;
#pragma unroll
          for (int rank = 0; rank < kTopK; ++rank)
            top_sum += selected[rank];
          const float inv_top = 1.0f / top_sum;
#pragma unroll
          for (int rank = 0; rank < kTopK; ++rank)
            w.weights[token * kTopK + rank] = sycl::half(
                sycl::isfinite(sum) ? selected[rank] * inv_top
                                    : std::numeric_limits<float>::quiet_NaN());
        }
      });
}

void launch_topk_subgroup(
    sycl::queue& queue, Inputs p, Workspace w, const sycl::half* logits) {
  constexpr int lanes = 32;
  constexpr int values_per_lane = kExperts / lanes;
  queue.parallel_for<TopKSubgroupKernel>(
      sycl::nd_range<1>(std::size_t(p.m) * lanes, lanes),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(lanes)]] {
        const int token = item.get_group_linear_id();
        const int lane = item.get_local_linear_id();
        const auto group = item.get_sub_group();
        float scores[values_per_lane];
        int invalid_local = 0;
#pragma unroll
        for (int j = 0; j < values_per_lane; ++j) {
          const float value =
              float(logits[token * kExperts + j * lanes + lane]);
          scores[j] = value;
          invalid_local |= !sycl::isfinite(value);
        }
        const bool invalid =
            sycl::reduce_over_group(
                group, invalid_local, sycl::maximum<int>()) != 0;

        float selected[kTopK];
#pragma unroll
        for (int rank = 0; rank < kTopK; ++rank) {
          float local_best = -std::numeric_limits<float>::infinity();
#pragma unroll
          for (int j = 0; j < values_per_lane; ++j)
            local_best = sycl::fmax(local_best, scores[j]);
          const float best = sycl::reduce_over_group(
              group, local_best, sycl::maximum<float>());
          int local_id = kExperts;
#pragma unroll
          for (int j = 0; j < values_per_lane; ++j)
            if (scores[j] == best)
              local_id = sycl::min(local_id, j * lanes + lane);
          const int id =
              sycl::reduce_over_group(group, local_id, sycl::minimum<int>());
          if (lane == 0) {
            selected[rank] = best;
            w.ids[token * kTopK + rank] = invalid ? rank : id;
          }
#pragma unroll
          for (int j = 0; j < values_per_lane; ++j)
            if (j * lanes + lane == id)
              scores[j] = -std::numeric_limits<float>::infinity();
        }
        if (lane == 0) {
          float top_sum = 0.0f;
          float values[kTopK];
#pragma unroll
          for (int rank = 0; rank < kTopK; ++rank) {
            values[rank] = sycl::exp(selected[rank] - selected[0]);
            top_sum += values[rank];
          }
          const float inv_top = 1.0f / top_sum;
#pragma unroll
          for (int rank = 0; rank < kTopK; ++rank)
            w.weights[token * kTopK + rank] = sycl::half(
                invalid ? std::numeric_limits<float>::quiet_NaN()
                        : values[rank] * inv_top);
        }
      });
}

// FP16分数和同分时的小expert id共用一个有序整数，省掉每轮第二次归约。
// +/-0按原float比较视为同分；非有限输入沿用原来的NaN输出契约。
void launch_topk_packed_subgroup(
    sycl::queue& queue, Inputs p, Workspace w, const sycl::half* logits) {
  constexpr int lanes = 32;
  constexpr int values_per_lane = kExperts / lanes;
  queue.parallel_for<TopKPackedSubgroupKernel>(
      sycl::nd_range<1>(std::size_t(p.m) * lanes, lanes),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(lanes)]] {
        const int token = item.get_group_linear_id();
        const int lane = item.get_local_linear_id();
        const auto group = item.get_sub_group();
        std::uint32_t keys[values_per_lane];
        int invalid_local = 0;
#pragma unroll
        for (int j = 0; j < values_per_lane; ++j) {
          const int id = j * lanes + lane;
          const auto value = logits[token * kExperts + id];
          invalid_local |= !sycl::isfinite(float(value));
          auto bits = sycl::bit_cast<std::uint16_t>(value);
          if ((bits & 0x7fffU) == 0) bits = 0;
          const std::uint16_t ordered = (bits & 0x8000U)
                                            ? std::uint16_t(~bits)
                                            : std::uint16_t(bits ^ 0x8000U);
          keys[j] = (std::uint32_t(ordered) << 9) | (511U - id);
        }
        const bool invalid =
            sycl::reduce_over_group(
                group, invalid_local, sycl::maximum<int>()) != 0;
        float selected[kTopK];
#pragma unroll
        for (int rank = 0; rank < kTopK; ++rank) {
          std::uint32_t local_best = 0;
#pragma unroll
          for (int j = 0; j < values_per_lane; ++j)
            local_best = sycl::max(local_best, keys[j]);
          const auto best = sycl::reduce_over_group(
              group, local_best, sycl::maximum<std::uint32_t>());
          const int id = 511 - int(best & 511U);
          if (lane == 0) {
            selected[rank] = float(logits[token * kExperts + id]);
            w.ids[token * kTopK + rank] = invalid ? rank : id;
          }
#pragma unroll
          for (int j = 0; j < values_per_lane; ++j)
            if (j * lanes + lane == id) keys[j] = 0;
        }
        if (lane == 0) {
          float top_sum = 0.0f;
          float values[kTopK];
#pragma unroll
          for (int rank = 0; rank < kTopK; ++rank) {
            values[rank] = sycl::exp(selected[rank] - selected[0]);
            top_sum += values[rank];
          }
          const float inv_top = 1.0f / top_sum;
#pragma unroll
          for (int rank = 0; rank < kTopK; ++rank)
            w.weights[token * kTopK + rank] = sycl::half(
                invalid ? std::numeric_limits<float>::quiet_NaN()
                        : values[rank] * inv_top);
        }
      });
}

void launch_up_shared(sycl::queue& queue, Inputs p, Workspace w) {
  const int routed_tasks = p.m * kTopK * p.intermediate;
  const int shared_tasks = p.m * (p.intermediate + 1);
  const int total = routed_tasks + shared_tasks;
  const std::size_t groups = (total + 7) / 8;
  queue.parallel_for<UpSharedKernel>(
      sycl::nd_range<1>(groups * kLocal, kLocal),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(kSubgroup)]] {
        const int task = item.get_global_linear_id() / kSubgroup;
        if (task >= total) return;
        const int lane = item.get_local_linear_id() % kSubgroup;
        const auto sg = item.get_sub_group();
        if (task < routed_tasks) {
          const int route_row = task / p.intermediate;
          const int col = task % p.intermediate;
          const int token = route_row / kTopK;
          const int expert = w.ids[route_row];
          const std::size_t gate_row =
              std::size_t(expert * (2 * p.intermediate) + col) * (kHidden / 2);
          const std::size_t up_row =
              gate_row + std::size_t(p.intermediate) * (kHidden / 2);
          const std::size_t gate_scale =
              std::size_t(expert * (2 * p.intermediate) + col) * 20;
          const std::size_t up_scale =
              gate_scale + std::size_t(p.intermediate) * 20;
          float gate = 0.0f;
          float up = 0.0f;
          for (int kb = lane; kb < kHidden / 2; kb += kSubgroup) {
            const int k = 2 * kb;
            const float x0 = float(p.x[token * kHidden + k]);
            const float x1 = float(p.x[token * kHidden + k + 1]);
            const auto qg = p.w13[gate_row + kb];
            const auto qu = p.w13[up_row + kb];
            const float sg_scale = float(p.s13[gate_scale + kb / 64]);
            const float su_scale = float(p.s13[up_scale + kb / 64]);
            gate += x0 * (signed_nibble(qg, false) * sg_scale) +
                    x1 * (signed_nibble(qg, true) * sg_scale);
            up += x0 * (signed_nibble(qu, false) * su_scale) +
                  x1 * (signed_nibble(qu, true) * su_scale);
          }
          gate = sycl::reduce_over_group(sg, gate, sycl::plus<float>());
          up = sycl::reduce_over_group(sg, up, sycl::plus<float>());
          if (lane == 0)
            w.routed[route_row * p.intermediate + col] =
                sycl::half((gate / (1.0f + sycl::exp(-gate))) * up);
        } else {
          const int local_task = task - routed_tasks;
          const int token = local_task / (p.intermediate + 1);
          const int col = local_task % (p.intermediate + 1);
          float gate = 0.0f;
          float up = 0.0f;
          if (col < p.intermediate) {
            const std::size_t gate_row = std::size_t(col) * kHidden;
            const std::size_t up_row =
                std::size_t(p.intermediate + col) * kHidden;
            for (int k = lane; k < kHidden; k += kSubgroup) {
              const float x = float(p.x[token * kHidden + k]);
              gate += x * float(p.shared_up[gate_row + k]);
              up += x * float(p.shared_up[up_row + k]);
            }
          } else {
            for (int k = lane; k < kHidden; k += kSubgroup)
              gate += float(p.x[token * kHidden + k]) * float(p.shared_gate[k]);
          }
          gate = sycl::reduce_over_group(sg, gate, sycl::plus<float>());
          if (col < p.intermediate)
            up = sycl::reduce_over_group(sg, up, sycl::plus<float>());
          if (lane == 0) {
            if (col < p.intermediate)
              w.shared[token * p.intermediate + col] =
                  sycl::half((gate / (1.0f + sycl::exp(-gate))) * up);
            else
              w.gates[token] = 1.0f / (1.0f + sycl::exp(-gate));
          }
        }
      });
}

void launch_down(sycl::queue& queue, Inputs p, Workspace w) {
  const int tiles = kHidden / kTile;
  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> partial(sycl::range<1>(12 * kTile), cgh);
    cgh.parallel_for<DownReduceKernel>(
        sycl::nd_range<1>(std::size_t(p.m) * tiles * 256, 256),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(kSubgroup)]] {
          const int lane = item.get_local_linear_id() % kSubgroup;
          const int route = item.get_local_linear_id() / kSubgroup;
          const int token = item.get_group_linear_id() / tiles;
          const int hbase = (item.get_group_linear_id() % tiles) * kTile;
          const auto sg = item.get_sub_group();
          if (route < kTopK) {
            const int expert = w.ids[token * kTopK + route];
            const std::size_t row_base = std::size_t(expert) * kHidden + hbase;
            const int bytes = p.intermediate / 2;
            const int groups = (p.intermediate + 127) / 128;
            const sycl::half* input =
                w.routed + (token * kTopK + route) * p.intermediate;
#pragma unroll
            for (int h = 0; h < kTile; ++h) {
              const std::size_t row = row_base + h;
              float accum = 0.0f;
              for (int kb = lane; kb < bytes; kb += kSubgroup) {
                const std::uint8_t packed = p.w2[row * bytes + kb];
                const float scale = float(p.s2[row * groups + kb / 64]);
                const int k = 2 * kb;
                accum +=
                    float(input[k]) * (signed_nibble(packed, false) * scale) +
                    float(input[k + 1]) * (signed_nibble(packed, true) * scale);
              }
              const float sum =
                  sycl::reduce_over_group(sg, accum, sycl::plus<float>());
              if (lane == 0) partial[route * kTile + h] = sum;
            }
          } else if (route == kTopK) {
            const sycl::half* input = w.shared + token * p.intermediate;
#pragma unroll
            for (int h = 0; h < kTile; ++h) {
              const sycl::half* weight =
                  p.shared_down + std::size_t(hbase + h) * p.intermediate;
              float first = 0.0f;
              float tail = 0.0f;
              for (int k = lane; k < p.intermediate; k += kSubgroup) {
                const float product = float(input[k]) * float(weight[k]);
                if (k < 128 || p.intermediate == 80)
                  first += product;
                else
                  tail += product;
              }
              first = sycl::reduce_over_group(sg, first, sycl::plus<float>());
              tail = sycl::reduce_over_group(sg, tail, sycl::plus<float>());
              if (lane == 0) {
                partial[10 * kTile + h] = first;
                partial[11 * kTile + h] = tail;
              }
            }
          }
          item.barrier(sycl::access::fence_space::local_space);
          if (route == 0 && lane < kTile) {
            float result = 0.0f;
#pragma unroll
            for (int r = 0; r < kTopK; ++r)
              result += float(w.weights[token * kTopK + r]) *
                        partial[r * kTile + lane];
            result += w.gates[token] * partial[10 * kTile + lane];
            result += w.gates[token] * partial[11 * kTile + lane];
            p.output[token * kHidden + hbase + lane] = sycl::half(result);
          }
        });
  });
}

}  // namespace

bool try_forward(sycl::queue& queue, const Inputs& p, const Workspace& w) {
  if (!queue.is_in_order() || p.m < 1 || p.m > 8 ||
      (p.intermediate != 80 && p.intermediate != 160) || p.x == nullptr ||
      p.w13 == nullptr || p.s13 == nullptr || p.w2 == nullptr ||
      p.s2 == nullptr || p.shared_up == nullptr || p.shared_down == nullptr ||
      p.shared_gate == nullptr || p.output == nullptr || w.ids == nullptr ||
      w.weights == nullptr || w.routed == nullptr || w.shared == nullptr ||
      w.gates == nullptr)
    return false;

  const bool router = p.logits == nullptr;
  if (router && (p.router_weight == nullptr || p.router_scale == nullptr ||
                 w.logits == nullptr))
    return false;
  if (!router && (p.router_weight != nullptr || p.router_scale != nullptr))
    return false;

  // Compact80 and compact160 row strides are divisible by four. A caller may
  // still supply an offset view, so check its base first.
  const auto aligned4 = [](const void* pointer) {
    return reinterpret_cast<std::uintptr_t>(pointer) % 4 == 0;
  };
  const bool native_block =
      aligned4(p.x) && aligned4(p.w13) && aligned4(p.w2) &&
      aligned4(p.shared_up) && aligned4(p.shared_down) &&
      aligned4(p.shared_gate) && aligned4(w.routed) && aligned4(w.shared) &&
      (!router || aligned4(p.router_weight));

  // The validation boundary is above this line; never return false after it.
  if (router) {
    if (native_block) {
      const bool wide_router = p.m == 1 && p.intermediate == 160;
      const char* vec_env =
          wide_router ? std::getenv("QWEN38_MOE_M1_ROUTER_VEC") : nullptr;
      const std::string_view vec =
          vec_env ? vec_env : (wide_router ? "8" : "4");
      if (vec == "8")
        block::launch_router<8>(queue, p, w);
      else
        block::launch_router(queue, p, w);
    } else
      launch_router(queue, p, w);
  }
  if (p.m == 1) {
    const char* packed = std::getenv("QWEN38_MOE_M1_PACKED_TOPK");
    if (!packed || (packed[0] == '1' && packed[1] == '\0'))
      launch_topk_packed_subgroup(queue, p, w, router ? w.logits : p.logits);
    else
      launch_topk_subgroup(queue, p, w, router ? w.logits : p.logits);
  } else
    launch_topk(queue, p, w, router ? w.logits : p.logits);
  if (native_block) {
    p.intermediate == 160 ? block::launch_up<160>(queue, p, w)
                          : block::launch_up<80>(queue, p, w);
    if (p.intermediate == 160) {
      if (p.m == 1) {
        const char* prefetch = std::getenv("QWEN38_MOE_M1_DOWN_PREFETCH");
        if (!prefetch || (prefetch[0] == '1' && prefetch[1] == '\0'))
          block::launch_down<160, 16, 16, true>(queue, p, w);
        else
          block::launch_down<160, 16>(queue, p, w);
      } else if (p.m == 2)
        block::launch_down<160, 8>(queue, p, w);
      else
        block::launch_down<160, 16>(queue, p, w);
    } else if (p.m <= 2)
      block::launch_down<80, 8>(queue, p, w);
    else
      block::launch_down<80, 16>(queue, p, w);
  } else {
    launch_up_shared(queue, p, w);
    launch_down(queue, p, w);
  }
  return true;
}

}  // namespace vllm::qwen38::moe_sycl
