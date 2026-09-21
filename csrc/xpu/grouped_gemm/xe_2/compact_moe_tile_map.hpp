// SPDX-License-Identifier: BSD-3-Clause
// Independent Gemma small-row scheduler experiment. The arithmetic xe_gemm
// implementation and existing public grouped-GEMM ABI remain unchanged.
#pragma once

namespace MoE {

template <class Policy>
class GemmaCompactTileMap;
template <class Policy>
class GemmaCompactDirectGemm;

template <class Policy>
void launch_gemma_compact_gemm(
    sycl::queue& stream,
    const cutlass::half_t* activations,
    const cutlass::float_e4m3_t* weights,
    const float* scales,
    cutlass::half_t* outputs,
    const int* rows_per_expert,
    int total_rows,
    int gemm_n,
    int gemm_k,
    int32_t* tile_map) {
  using namespace cute;
  using WGTile = typename Policy::WGTile;
  constexpr int tile_m = int(get<0>(WGTile{}));
  constexpr int tile_n = int(get<1>(WGTile{}));
  static_assert(tile_m == 8 || tile_m == 16);
  constexpr int tile_shift = tile_m == 8 ? 3 : 4;

  // Existing rows_per_expert contract: 128 nonnegative counts summing to R.
  // sum(ceil(count/tile_m)) <= R, so 1 + 4*R int32 slots suffice.
  // Every entry is {expert, expert_row_base, expert_rows, local_m_tile}.
  // Scratch belongs to this invocation and this current Torch stream only.
  const sycl::event map_ready = stream.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<GemmaCompactTileMap<Policy>>(
        sycl::nd_range<1>{sycl::range<1>(128), sycl::range<1>(128)},
        [=](sycl::nd_item<1> item) {
          const int expert = static_cast<int>(item.get_local_id(0));
          const int rows = rows_per_expert[expert];
          const int tiles = (rows + tile_m - 1) >> tile_shift;
          const auto group = item.get_group();
          const int row_base = sycl::exclusive_scan_over_group(
              group, rows, sycl::plus<int>());
          const int tile_base = sycl::exclusive_scan_over_group(
              group, tiles, sycl::plus<int>());
          if (expert == 127) {
            tile_map[0] = tile_base + tiles;
          }
          for (int tile = 0; tile < tiles; ++tile) {
            // Bounds check protects the map allocation even if a caller
            // violates the pre-existing counts contract. Such invalid counts
            // are not a supported operation and do not trigger async fallback.
            const int entry = tile_base + tile;
            if (entry >= 0 && entry < total_rows) {
              int32_t* tuple = tile_map + 1 + 4 * entry;
              tuple[0] = expert;
              tuple[1] = row_base;
              tuple[2] = rows;
              tuple[3] = tile;
            }
          }
        });
  });

  using Atom = MMA_Atom<XE_DPAS_TT<8, float, cutlass::half_t>>;
  using MMA = typename TiledMMAHelper<
      Atom, Layout<WGTile>, typename Policy::SGLayout>::TiledMMA;
  const auto mma = MMA{};
  const int threads = size(mma);
  const int n_tiles = (gemm_n + tile_n - 1) / tile_n;
  const sycl::range<3> local(1, 1, threads);
  const sycl::range<3> groups(1, total_rows, n_tiles);
  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;
  syclex::properties props{
      syclex::sub_group_size<16>, intelex::grf_size<256>};

  stream.submit([&](sycl::handler& cgh) {
    cgh.depends_on(map_ready);
    cgh.parallel_for<GemmaCompactDirectGemm<Policy>>(
        sycl::nd_range<3>{groups * local, local}, props,
        [=](sycl::nd_item<3> item) {
          const int m_tile = static_cast<int>(item.get_group(1));
          const int n_tile = static_cast<int>(item.get_group(2));
          // Uniform workgroup early-exit before xe_gemm's barriers.
          if (m_tile >= tile_map[0]) {
            return;
          }
          const int32_t* tuple = tile_map + 1 + 4 * m_tile;
          const int expert = tuple[0];
          const int row_base = tuple[1];
          const int expert_rows = tuple[2];
          const int local_m_tile = tuple[3];
          const int64_t weight_offset =
              static_cast<int64_t>(expert) * gemm_n * gemm_k;
          auto A = make_moe_tensor<cutlass::half_t, 'R'>(
              const_cast<cutlass::half_t*>(activations) + row_base * gemm_k,
              expert_rows, gemm_k);
          // Public weight layout [E,K,N] is viewed by xe_gemm as column-major
          // [N,K], exactly as the original LayoutKindB='R' launcher.
          auto B = make_moe_tensor<cutlass::float_e4m3_t, 'C'>(
              const_cast<cutlass::float_e4m3_t*>(weights) + weight_offset,
              gemm_n, gemm_k);
          auto D = make_moe_tensor<cutlass::half_t, 'R'>(
              outputs + row_base * gemm_n, expert_rows, gemm_n);
          const cutlass::half_t* bias = nullptr;
          xe_gemm<typename Policy::GmemTiledCopyA,
                  typename Policy::GmemTiledCopyB,
                  typename Policy::GmemTiledCopyD>(
              A, B, scales + expert, bias, D,
              make_coord(local_m_tile, n_tile, _, 0), mma);
        });
  });
}

// Pure metadata check. Unsupported cases submit nothing and continue through
// the unchanged V1 m8/native dispatcher. No model-specific Python integration.
inline bool gemma_compact_eligible(
    const at::Tensor& a,
    const at::Tensor& b,
    const c10::optional<at::Tensor>& scale,
    const c10::optional<at::Tensor>& bias,
    const at::Tensor& d,
    const at::Tensor& counts,
    int64_t n,
    int64_t k,
    int64_t experts) {
  if (!a.is_xpu() || experts != 128 || !scale.has_value() || bias.has_value()) {
    return false;
  }
  if (a.dim() != 2 || b.dim() != 3 || d.dim() != 2 || counts.dim() != 1 ||
      scale->dim() != 1 || a.size(0) < 1 || a.size(0) > 64 ||
      !((k == 2816 && n == 704) || (k == 352 && n == 2816))) {
    return false;
  }
  if (a.scalar_type() != at::kHalf || d.scalar_type() != at::kHalf ||
      b.scalar_type() != at::kFloat8_e4m3fn || scale->scalar_type() != at::kFloat ||
      counts.scalar_type() != at::kInt || b.size(0) != 128 ||
      counts.numel() != 128 || scale->numel() != 128 ||
      a.size(1) != k || b.size(1) != k || b.size(2) != n ||
      d.size(0) != a.size(0) || d.size(1) != n) {
    return false;
  }
  if (!a.is_contiguous() || !b.is_contiguous() || !d.is_contiguous() ||
      !counts.is_contiguous() || !scale->is_contiguous() ||
      b.device() != a.device() || d.device() != a.device() ||
      counts.device() != a.device() || scale->device() != a.device()) {
    return false;
  }
  return !d.is_alias_of(a) && !d.is_alias_of(b) &&
         !d.is_alias_of(*scale) && !d.is_alias_of(counts);
}

}  // namespace MoE
