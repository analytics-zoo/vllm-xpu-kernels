// SPDX-License-Identifier: Apache-2.0
#include "qwen38/qsa_sycl_owner.h"

#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>
#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

namespace vllm::qwen38::qsa_sycl {
namespace {

constexpr int kStoreWg = 128;
template <typename Scalar, int Width, bool Unique>
class RowStoreKernel;
template <typename PositionT, typename SlotT>
class FirstPositionsKernel;
class CheckHistoryKernel;
template <bool Unique>
class AcknowledgeHistoryKernel;
class ResetHistoryBlocksKernel;

void record_tensors(std::initializer_list<const at::Tensor*> tensors,
                    c10::xpu::XPUStream stream) {
  for (const auto* tensor : tensors) {
    c10::xpu::XPUCachingAllocator::recordStream(
        tensor->storage().data_ptr(), stream);
  }
}

bool positive_non_overlapping(const at::Tensor& tensor) {
  std::array<std::pair<int64_t, int64_t>, 4> dimensions{};
  int count = 0;
  for (int d = 0; d < tensor.dim(); ++d) {
    if (tensor.size(d) <= 1) continue;
    if (tensor.stride(d) <= 0) return false;
    dimensions[count++] = {tensor.stride(d), tensor.size(d)};
  }
  std::sort(dimensions.begin(), dimensions.begin() + count,
            [](const auto& a, const auto& b) { return a.first < b.first; });
  int64_t span = 1;
  for (int d = 0; d < count; ++d) {
    const auto [stride, size] = dimensions[d];
    if (stride < span || (size - 1) >
        (std::numeric_limits<int64_t>::max() - span) / stride) return false;
    span += (size - 1) * stride;
  }
  return true;
}

bool may_overlap(const at::Tensor& lhs, const at::Tensor& rhs) {
  if (lhs.numel() == 0 || rhs.numel() == 0) return false;
  auto bounds = [](const at::Tensor& t, uintptr_t& first,
                   uintptr_t& last) -> bool {
    constexpr auto max = std::numeric_limits<uintptr_t>::max();
    uintptr_t span = 0;
    for (int d = 0; d < t.dim(); ++d) {
      if (t.size(d) < 0 || t.stride(d) < 0) return false;
      auto count = static_cast<uintptr_t>(std::max<int64_t>(t.size(d)-1, 0));
      auto stride = static_cast<uintptr_t>(t.stride(d));
      if (stride && count > (max - span) / stride) return false;
      span += count * stride;
    }
    auto item = static_cast<uintptr_t>(t.element_size());
    if (span >= max / item) return false;
    auto bytes = (span + 1) * item;
    first = reinterpret_cast<uintptr_t>(t.data_ptr());
    if (first > max - bytes) return false;
    last = first + bytes;
    return true;
  };
  uintptr_t lf = 0, ll = 0, rf = 0, rl = 0;
  return !bounds(lhs, lf, ll) || !bounds(rhs, rf, rl) ||
      (lf < rl && rf < ll);
}

void check_xpu(const at::Tensor& tensor, const at::Tensor& anchor,
               at::ScalarType dtype, const char* label) {
  TORCH_CHECK(tensor.device() == anchor.device() &&
                  tensor.scalar_type() == dtype &&
                  !tensor.is_neg() && !tensor.is_conj(),
              label, " device/dtype/lazy-view mismatch");
}

void validate_store(const at::Tensor& cache,
                    const at::Tensor& slot_mapping,
                    const at::Tensor& rows) {
  TORCH_CHECK(cache.is_xpu(), "row-store cache must be XPU");
  TORCH_CHECK(
      !cache.is_neg() && !cache.is_conj(),
      "row-store cache lazy-view unsupported");
  TORCH_CHECK(cache.scalar_type() == at::kHalf ||
                  cache.scalar_type() == at::kBFloat16 ||
                  cache.scalar_type() == at::kLong,
              "row-store cache dtype unsupported");
  check_xpu(slot_mapping, cache, at::kLong, "row-store slots");
  check_xpu(rows, cache, cache.scalar_type(), "row-store rows");
  TORCH_CHECK(cache.dim() == 4 && cache.size(0) > 0 &&
                  cache.size(1) > 0 && cache.size(2) == 1 &&
                  cache.size(3) ==
                      (cache.scalar_type() == at::kLong ? 3 : 128) &&
                  cache.stride(3) == 1 &&
                  positive_non_overlapping(cache),
              "row-store cache shape/strides invalid");
  TORCH_CHECK(slot_mapping.dim() == 1 && slot_mapping.is_contiguous() &&
                  (rows.dim() == 2 ||
                   (rows.dim() == 3 && rows.size(1) == 1)) &&
                  rows.size(0) == slot_mapping.numel() &&
                  rows.size(rows.dim()-1) == cache.size(3) &&
                  positive_non_overlapping(rows) &&
                  (cache.scalar_type() == at::kLong ||
                   rows.stride(rows.dim()-1) == 1),
              "row-store slots/rows shape or strides invalid");
  TORCH_CHECK(cache.size(0) <=
                  std::numeric_limits<int64_t>::max() / cache.size(1),
              "row-store capacity overflows int64");
  TORCH_CHECK(!may_overlap(cache, slot_mapping) && !may_overlap(cache, rows),
              "row-store cache aliases input");
}

template <typename Scalar, int Width, bool Unique>
void launch_store_raw(sycl::queue& queue, Scalar* destination,
                      const int64_t* slots, const Scalar* source,
                      int64_t count, int64_t page_size, int64_t capacity,
                      int64_t page_stride, int64_t token_stride,
                      int64_t row_stride, int64_t column_stride) {
  if (count == 0) return;
  const int64_t groups = Unique ? count : 1;
  queue.parallel_for<RowStoreKernel<Scalar, Width, Unique>>(
      sycl::nd_range<1>(sycl::range<1>(groups * kStoreWg),
                        sycl::range<1>(kStoreWg)),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
        const int lane = item.get_local_linear_id();
        const int64_t first = Unique ? item.get_group_linear_id() : 0;
        const int64_t end = Unique ? first + 1 : count;
        for (int64_t row = first; row < end; ++row) {
          const int64_t slot = slots[row];
          if (slot >= 0 && slot < capacity) {
            const int64_t page = slot / page_size;
            const int64_t offset = slot - page * page_size;
            const int64_t target = page * page_stride + offset * token_stride;
            for (int column = lane; column < Width; column += kStoreWg) {
              destination[target + column] =
                  source[row * row_stride + column * column_stride];
            }
          }
          if constexpr (!Unique) {
            item.barrier(sycl::access::fence_space::global_and_local);
          }
        }
      });
}

void launch_store_typed(sycl::queue& queue, const at::Tensor& cache,
                        const at::Tensor& slots, const at::Tensor& rows,
                        bool unique_slots_proven) {
  const int64_t count = slots.numel();
  const int64_t page_size = cache.size(1);
  const int64_t capacity = cache.size(0) * page_size;
  const int64_t page_stride = cache.stride(0);
  const int64_t token_stride = cache.stride(1);
  const int64_t row_stride = rows.stride(0);
  const int64_t column_stride = rows.stride(rows.dim() - 1);
  const auto* slot_data = slots.data_ptr<int64_t>();
  if (cache.scalar_type() == at::kLong) {
    if (unique_slots_proven)
      launch_store_raw<int64_t, 3, true>(
          queue, cache.data_ptr<int64_t>(), slot_data,
          rows.data_ptr<int64_t>(), count, page_size, capacity,
          page_stride, token_stride, row_stride, column_stride);
    else
      launch_store_raw<int64_t, 3, false>(
          queue, cache.data_ptr<int64_t>(), slot_data,
          rows.data_ptr<int64_t>(), count, page_size, capacity,
          page_stride, token_stride, row_stride, column_stride);
  } else if (unique_slots_proven) {
    launch_store_raw<uint16_t, 128, true>(
        queue, static_cast<uint16_t*>(cache.data_ptr()), slot_data,
        static_cast<const uint16_t*>(rows.data_ptr()), count, page_size,
        capacity, page_stride, token_stride, row_stride, column_stride);
  } else {
    launch_store_raw<uint16_t, 128, false>(
        queue, static_cast<uint16_t*>(cache.data_ptr()), slot_data,
        static_cast<const uint16_t*>(rows.data_ptr()), count, page_size,
        capacity, page_stride, token_stride, row_stride, column_stride);
  }
}

using M1Stores = std::vector<std::tuple<at::Tensor, at::Tensor, at::Tensor>>;

bool preflight_m1_transaction(const M1Stores& stores) {
  if (stores.empty() || stores.size() > 3) return false;
  const auto device = std::get<0>(stores.front()).device();
  for (const auto& [cache, slots, rows] : stores) {
    if (!cache.is_xpu() || cache.device() != device ||
        slots.device() != device || rows.device() != device ||
        slots.dim() != 1 || slots.numel() != 1 || rows.dim() < 2 ||
        rows.size(0) != 1)
      return false;
    try {
      validate_store(cache, slots, rows);
    } catch (const c10::Error&) {
      // No store has been submitted; the complete legacy fallback is safe.
      return false;
    }
  }
  return true;
}

struct M1StoreDescriptor {
  void* cache;
  const void* rows;
  const int64_t* slot;
  int64_t capacity;
  int64_t page_size;
  int64_t page_stride;
  int64_t token_stride;
  int64_t column_stride;
  int page_shift;
  int width;
};

class M1StoreTransactionFusedKernel;

void launch_store_m1_transaction_fused(
    sycl::queue& queue,
    const std::array<M1StoreDescriptor, 3>& descriptors,
    int count) {
  queue.parallel_for<M1StoreTransactionFusedKernel>(
      sycl::nd_range<1>(sycl::range<1>(kStoreWg), sycl::range<1>(kStoreWg)),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
        const int lane = item.get_local_linear_id();
        for (int i = 0; i < count; ++i) {
          const auto d = descriptors[i];
          const int64_t slot = *d.slot;
          if (slot >= 0 && slot < d.capacity) {
            const int64_t page =
                d.page_shift >= 0 ? slot >> d.page_shift : slot / d.page_size;
            const int64_t token = d.page_shift >= 0 ? slot & (d.page_size - 1)
                                                    : slot - page * d.page_size;
            const int64_t target =
                page * d.page_stride + token * d.token_stride;
            if (d.width == 128) {
              auto* cache = static_cast<uint16_t*>(d.cache);
              const auto* rows = static_cast<const uint16_t*>(d.rows);
              cache[target + lane] = rows[lane];
            } else if (lane < 3) {
              auto* cache = static_cast<int64_t*>(d.cache);
              const auto* rows = static_cast<const int64_t*>(d.rows);
              cache[target + lane] = rows[lane * d.column_stride];
            }
          }
          // Later rows or slots may alias this cache. Every lane reaches the
          // fence even for an invalid slot; the last store needs no fence.
          if (i + 1 < count)
            item.barrier(sycl::access::fence_space::global_and_local);
        }
      });
}

}  // namespace

at::Tensor store_cache_rows_v3(at::Tensor cache,
                               const at::Tensor& slot_mapping,
                               const at::Tensor& rows,
                               bool unique_slots_proven) {
  validate_store(cache, slot_mapping, rows);
  if (slot_mapping.numel() == 0) return cache;
  c10::OptionalDeviceGuard guard(cache.device());
  auto stream = c10::xpu::getCurrentXPUStream(cache.get_device());
  TORCH_CHECK(stream.queue().is_in_order(),
              "row-store requires an in-order current stream");
  record_tensors({&cache, &slot_mapping, &rows}, stream);
  launch_store_typed(stream.queue(), cache, slot_mapping, rows,
                     unique_slots_proven);
  return cache;
}

bool try_store_m1_transaction_v1(
    const std::vector<std::tuple<at::Tensor, at::Tensor, at::Tensor>>& stores) {
  if (!preflight_m1_transaction(stores)) return false;
  const auto device = std::get<0>(stores.front()).device();
  c10::OptionalDeviceGuard guard(device);
  auto stream = c10::xpu::getCurrentXPUStream(device.index());
  TORCH_CHECK(stream.queue().is_in_order(),
              "row-store transaction requires an in-order stream");
  // No receipt ABI here: the model defaults it off. Never fall back after
  // one of these submissions, even if a later queue operation throws.
  for (const auto& [cache, slots, rows] : stores) {
    record_tensors({&cache, &slots, &rows}, stream);
  }
  for (const auto& [cache, slots, rows] : stores) {
    launch_store_typed(stream.queue(), cache, slots, rows, false);
  }
  return true;
}

bool try_store_m1_transaction_fused_v1(const M1Stores& stores) {
  if (!preflight_m1_transaction(stores)) return false;
  const auto device = std::get<0>(stores.front()).device();
  c10::OptionalDeviceGuard guard(device);
  auto stream = c10::xpu::getCurrentXPUStream(device.index());
  TORCH_CHECK(
      stream.queue().is_in_order(),
      "row-store transaction requires an in-order stream");

  std::array<M1StoreDescriptor, 3> descriptors{};
  for (size_t i = 0; i < stores.size(); ++i) {
    const auto& [cache, slots, rows] = stores[i];
    const int64_t page_size = cache.size(1);
    int page_shift = -1;
    if ((page_size & (page_size - 1)) == 0) {
      page_shift = 0;
      for (int64_t remaining = page_size; remaining > 1; remaining >>= 1)
        ++page_shift;
    }
    descriptors[i] = {
        cache.data_ptr(),
        rows.data_ptr(),
        slots.data_ptr<int64_t>(),
        cache.size(0) * page_size,
        page_size,
        cache.stride(0),
        cache.stride(1),
        rows.stride(rows.dim() - 1),
        page_shift,
        static_cast<int>(cache.size(3))};
  }
  // Record all owners before submitting the transaction; exception unwinding
  // must not recycle any allocation after a partial submission failure.
  for (const auto& [cache, slots, rows] : stores) {
    record_tensors({&cache, &slots, &rows}, stream);
  }
  launch_store_m1_transaction_fused(
      stream.queue(), descriptors, static_cast<int>(stores.size()));
  return true;
}

at::Tensor compressed_first_positions_v1(
    const at::Tensor& logical_positions, const at::Tensor& slots,
    at::Tensor out, int64_t ratio) {
  TORCH_CHECK(logical_positions.is_xpu(), "logical positions must be XPU");
  TORCH_CHECK(ratio == 4, "QSA compressed first positions require ratio=4");
  TORCH_CHECK(logical_positions.scalar_type() == at::kInt ||
                  logical_positions.scalar_type() == at::kLong,
              "logical positions must be int32/int64");
  check_xpu(slots, logical_positions, slots.scalar_type(), "slots");
  TORCH_CHECK(slots.scalar_type() == at::kInt ||
                  slots.scalar_type() == at::kLong,
              "slots must be int32/int64");
  check_xpu(out, logical_positions, logical_positions.scalar_type(), "out");
  TORCH_CHECK(logical_positions.dim() == 1 &&
                  logical_positions.is_contiguous() &&
                  slots.sizes() == logical_positions.sizes() &&
                  slots.is_contiguous() &&
                  out.sizes() == logical_positions.sizes() &&
                  out.is_contiguous() &&
                  !may_overlap(out, logical_positions) &&
                  !may_overlap(out, slots),
              "compressed first-position tensor contract invalid");
  if (out.numel() == 0) return out;
  c10::OptionalDeviceGuard guard(out.device());
  auto stream = c10::xpu::getCurrentXPUStream(out.get_device());
  auto& queue = stream.queue();
  const int64_t count = out.numel();
  auto launch = [&](auto position_tag, auto slot_tag) {
    using P = typename decltype(position_tag)::type;
    using S = typename decltype(slot_tag)::type;
    const auto* position = logical_positions.data_ptr<P>();
    const auto* slot = slots.data_ptr<S>();
    auto* result = out.data_ptr<P>();
    queue.parallel_for<FirstPositionsKernel<P, S>>(
        sycl::range<1>(count), [=](sycl::id<1> index) {
          const int64_t row = index[0];
          result[row] = slot[row] >= 0 && position[row] >= 3
              ? static_cast<P>(position[row] - 3) : P(0);
        });
  };
  if (logical_positions.scalar_type() == at::kInt) {
    if (slots.scalar_type() == at::kInt)
      launch(std::type_identity<int32_t>{}, std::type_identity<int32_t>{});
    else
      launch(std::type_identity<int32_t>{}, std::type_identity<int64_t>{});
  } else if (slots.scalar_type() == at::kInt) {
    launch(std::type_identity<int64_t>{}, std::type_identity<int32_t>{});
  } else {
    launch(std::type_identity<int64_t>{}, std::type_identity<int64_t>{});
  }
  record_tensors({&logical_positions, &slots, &out}, stream);
  return out;
}

at::Tensor check_history_v1(
    const at::Tensor& tags, const at::Tensor& token_to_req,
    const at::Tensor& query_start_loc, const at::Tensor& logical_positions,
    const at::Tensor& block_table, const at::Tensor& compressed_slots,
    at::Tensor result, int64_t ring_size, int64_t num_blocks,
    int64_t compress_ratio) {
  TORCH_CHECK(tags.is_xpu(), "history tags must be XPU");
  check_xpu(token_to_req, tags, at::kInt, "history requests");
  check_xpu(query_start_loc, tags, at::kInt, "history starts");
  check_xpu(logical_positions, tags, at::kLong, "history positions");
  check_xpu(block_table, tags, at::kInt, "history table");
  check_xpu(compressed_slots, tags, at::kLong, "history slots");
  check_xpu(result, tags, at::kBool, "history result");
  const int64_t rows = token_to_req.numel();
  const int64_t requests = query_start_loc.numel() - 1;
  TORCH_CHECK(tags.scalar_type() == at::kLong && tags.dim() == 1 &&
                  tags.is_contiguous() && num_blocks > 0 && ring_size >= 4 &&
                  num_blocks <=
                      (std::numeric_limits<int64_t>::max() - 1) / ring_size &&
                  tags.numel() == num_blocks * ring_size + 1 &&
                  compress_ratio == 4 && rows >= 1 && rows <= 64 &&
                  requests >= 1 && token_to_req.is_contiguous() &&
                  query_start_loc.is_contiguous() &&
                  logical_positions.sizes() == at::IntArrayRef({rows}) &&
                  logical_positions.is_contiguous() &&
                  compressed_slots.sizes() == at::IntArrayRef({rows}) &&
                  compressed_slots.is_contiguous() &&
                  block_table.dim() == 2 && block_table.size(0) >= requests &&
                  block_table.size(1) >= 1 && block_table.is_contiguous() &&
                  result.numel() == 1 && result.is_contiguous(),
              "history check metadata shape/strides invalid");
  for (const auto* input : {&tags, &token_to_req, &query_start_loc,
                            &logical_positions, &block_table,
                            &compressed_slots}) {
    TORCH_CHECK(!may_overlap(result, *input),
                "history result aliases an input");
  }
  c10::OptionalDeviceGuard guard(tags.device());
  auto stream = c10::xpu::getCurrentXPUStream(tags.get_device());
  const auto* tag = tags.data_ptr<int64_t>();
  const auto* req = token_to_req.data_ptr<int32_t>();
  const auto* start = query_start_loc.data_ptr<int32_t>();
  const auto* pos = logical_positions.data_ptr<int64_t>();
  const auto* table = block_table.data_ptr<int32_t>();
  const auto* slots = compressed_slots.data_ptr<int64_t>();
  auto* ok = result.data_ptr<bool>();
  const int table_stride = block_table.size(1);
  stream.queue().parallel_for<CheckHistoryKernel>(
      sycl::nd_range<1>(sycl::range<1>(128), sycl::range<1>(128)),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
        const int row = item.get_local_linear_id();
        bool valid = true;
        if (row < rows) {
          const int request = req[row];
          const bool request_ok = request >= 0 && request < requests;
          const int safe_req = sycl::clamp(request, 0,
                                           static_cast<int>(requests - 1));
          const int first = start[safe_req];
          const int last = start[safe_req + 1];
          const int64_t position = pos[row];
          const bool live = request_ok && row >= first && row < last &&
              slots[row] >= 0 && position >= 3;
          const int block = table[safe_req * table_stride];
          if (live && (block < 0 || block >= num_blocks)) {
            valid = false;
          } else if (live) {
            const int64_t chunk_start = position - (row - first);
            for (int d = 0; d < 4; ++d) {
              const int64_t historical = position - 3 + d;
              if (historical < chunk_start) {
                const int64_t offset =
                    ((historical % ring_size) + ring_size) % ring_size;
                valid &= tag[static_cast<int64_t>(block) * ring_size +
                             offset] == historical;
              }
            }
          }
        }
        const bool all = sycl::reduce_over_group(
            item.get_group(), valid, sycl::logical_and<bool>());
        if (row == 0) ok[0] = all;
      });
  record_tensors({&tags, &token_to_req, &query_start_loc,
                  &logical_positions, &block_table, &compressed_slots,
                  &result}, stream);
  return result;
}

at::Tensor acknowledge_history_v1(
    at::Tensor tags, const at::Tensor& slot_mapping,
    const at::Tensor& logical_positions, int64_t capacity,
    bool unique_slots_proven) {
  TORCH_CHECK(tags.is_xpu() && tags.scalar_type() == at::kLong &&
                  tags.dim() == 1 && tags.is_contiguous() && capacity > 0 &&
                  tags.numel() == capacity + 1,
              "history tag capacity invalid");
  check_xpu(slot_mapping, tags, at::kLong, "history slots");
  check_xpu(logical_positions, tags, at::kLong, "history positions");
  TORCH_CHECK(slot_mapping.dim() == 1 && slot_mapping.is_contiguous() &&
                  logical_positions.sizes() == slot_mapping.sizes() &&
                  logical_positions.is_contiguous() &&
                  !may_overlap(tags, slot_mapping) &&
                  !may_overlap(tags, logical_positions),
              "history acknowledgement inputs invalid");
  const int64_t count = slot_mapping.numel();
  if (!count) return tags;
  c10::OptionalDeviceGuard guard(tags.device());
  auto stream = c10::xpu::getCurrentXPUStream(tags.get_device());
  auto* tag = tags.data_ptr<int64_t>();
  const auto* slots = slot_mapping.data_ptr<int64_t>();
  const auto* positions = logical_positions.data_ptr<int64_t>();
  if (unique_slots_proven) {
    stream.queue().parallel_for<AcknowledgeHistoryKernel<true>>(
        sycl::range<1>(count), [=](sycl::id<1> index) {
          const int64_t row = index[0];
          const int64_t slot = slots[row];
          if (slot >= 0 && slot < capacity) tag[slot] = positions[row];
        });
  } else {
    stream.queue().parallel_for<AcknowledgeHistoryKernel<false>>(
        sycl::range<1>(1), [=](sycl::id<1>) {
          for (int64_t row = 0; row < count; ++row) {
            const int64_t slot = slots[row];
            if (slot >= 0 && slot < capacity) tag[slot] = positions[row];
          }
        });
  }
  record_tensors({&tags, &slot_mapping, &logical_positions}, stream);
  return tags;
}

at::Tensor reset_history_blocks_v1(
    at::Tensor tags, const at::Tensor& changed_blocks,
    int64_t ring_size, int64_t num_blocks) {
  TORCH_CHECK(tags.is_xpu() && tags.scalar_type() == at::kLong &&
                  tags.dim() == 1 && tags.is_contiguous() &&
                  ring_size >= 4 && num_blocks > 0 &&
                  num_blocks <=
                      (std::numeric_limits<int64_t>::max() - 1) / ring_size &&
                  tags.numel() == num_blocks * ring_size + 1,
              "history reset tag geometry invalid");
  check_xpu(changed_blocks, tags, at::kLong, "changed blocks");
  TORCH_CHECK(changed_blocks.dim() == 1 && changed_blocks.is_contiguous() &&
                  !may_overlap(tags, changed_blocks),
              "changed blocks must be disjoint contiguous int64");
  const int64_t count = changed_blocks.numel();
  if (!count) return tags;
  TORCH_CHECK(count <= std::numeric_limits<int64_t>::max() / ring_size,
              "history reset grid overflows int64");
  c10::OptionalDeviceGuard guard(tags.device());
  auto stream = c10::xpu::getCurrentXPUStream(tags.get_device());
  auto* tag = tags.data_ptr<int64_t>();
  const auto* blocks = changed_blocks.data_ptr<int64_t>();
  stream.queue().parallel_for<ResetHistoryBlocksKernel>(
      sycl::range<1>(1), [=](sycl::id<1>) {
        // A changed-block list may contain duplicates. Serial reset avoids
        // concurrent writes to the same tag without asking the host to read
        // device block IDs or to trust a uniqueness hint.
        for (int64_t row = 0; row < count; ++row) {
          const int64_t block = blocks[row];
          if (block >= 0 && block < num_blocks) {
            for (int64_t offset = 0; offset < ring_size; ++offset) {
              tag[block * ring_size + offset] = -1;
            }
          }
        }
      });
  record_tensors({&tags, &changed_blocks}, stream);
  return tags;
}

void bind_qsa_sycl_owner(pybind11::module_& module) {
  module.def("qsa_sycl_store_cache_rows_v3", &store_cache_rows_v3);
  module.def("qsa_sycl_try_store_m1_transaction_v1",
             &try_store_m1_transaction_v1);
  module.def("qsa_sycl_try_store_m1_transaction_fused_v1",
             &try_store_m1_transaction_fused_v1);
}

}  // namespace vllm::qwen38::qsa_sycl

#ifdef QWEN38_QSA_OWNER_STANDALONE
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  vllm::qwen38::qsa_sycl::bind_qsa_sycl_owner(module);
}
#endif
