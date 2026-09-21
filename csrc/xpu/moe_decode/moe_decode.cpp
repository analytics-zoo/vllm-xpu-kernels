// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <c10/xpu/XPUFunctions.h>
#include <torch/library.h>
#include <Python.h>
#include <cute/tensor.hpp>
#include "gemm_xe2_policy.hpp"
#include "grouped_gemm_xe2.hpp"
#include "gate_up.hpp"

namespace vllm::moe_decode {
using namespace cute;
template <int Variant>
class GateUpPolicy : public MoE::xe_gemm_policy_base {
 public:
  using WGTile = Shape<_8, _32, _32>;
  using SGLayout = Layout<Shape<_1, _2, _1>, Stride<_2, _1, _0>>;
};
template <bool Gate, bool Grouped, int Variant>
class RoutedGemm;
template <bool Grouped, int Variant>
class RoutedGather;
template <int Variant>
class Activation;
struct alignas(16) Half8 {
  sycl::half v[8];
};
struct alignas(8) Half4 {
  sycl::half v[4];
};

// The M5 tiles fit in 128 GRFs. Preserve the established 256-GRF kernels
// for other batch sizes and use the same GRF mode for the M5 vector helpers.
template <int Dimensions, typename Kernel>
struct Register128 {
  Kernel kernel;
  auto get(sycl::ext::oneapi::experimental::properties_tag) const {
    return sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::sub_group_size<32>,
        sycl::ext::intel::experimental::grf_size<128>};
  }
  void operator()(sycl::nd_item<Dimensions> item) const { kernel(item); }
};

template <typename Name, bool Narrow, int Dimensions, typename Kernel>
void submit_vector(
    sycl::queue& q, sycl::nd_range<Dimensions> range, Kernel kernel) {
  if constexpr (Narrow)
    q.parallel_for<Name>(range, Register128<Dimensions, Kernel>{kernel});
  else
    q.parallel_for<Name>(range, kernel);
}

template <bool Gate, bool Grouped, int Variant>
void gemm(
    sycl::queue& q,
    const at::Tensor& input,
    const at::Tensor& weights,
    const at::Tensor& scales,
    const at::Tensor& ids,
    at::Tensor& output,
    const int* tile_map) {
  constexpr bool FusedGate = Gate && Variant < 4;
  using Policy = std::
      conditional_t<FusedGate, GateUpPolicy<Variant>, MoE::w8a16_policy_m_8>;
  using Tile = typename Policy::WGTile;
  using MMA = typename TiledMMAHelper<
      MMA_Atom<XE_DPAS_TT<8, float, cutlass::half_t>>,
      Layout<Tile>,
      typename Policy::SGLayout>::TiledMMA;
  const auto mma = MMA{};
  constexpr int K = Gate ? 2816 : 352;
  constexpr int N = Gate ? 704 : 2816;
  constexpr int D = Gate ? (Variant >= 4 ? 704 : 352) : 2816;
  constexpr int TN = Gate && Variant < 4 ? 32 : 64;
  const int routes = ids.numel(), threads = size(mma);
  const auto* a = reinterpret_cast<const cutlass::half_t*>(input.data_ptr());
  const auto* b =
      reinterpret_cast<const cutlass::float_e4m3_t*>(weights.data_ptr());
  const auto* s = scales.data_ptr<float>();
  const auto* experts = ids.data_ptr<int>();
  auto* d = reinterpret_cast<cutlass::half_t*>(output.data_ptr());
  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;
  constexpr int GRF = Variant == 5 ? 128 : 256;
  syclex::properties props{syclex::sub_group_size<16>, intelex::grf_size<GRF>};
  const sycl::range<3> local(1, 1, threads), groups(1, routes, D / TN);
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<RoutedGemm<Gate, Grouped, Variant>>(
        sycl::nd_range<3>(groups * local, local),
        props,
        [=](sycl::nd_item<3> it) {
          const int route = it.get_group(1), tile = it.get_group(2);
          int expert, row_base, rows, m_tile;
          if constexpr (Grouped) {
            if (route >= tile_map[0]) return;
            const int* tuple = tile_map + 1 + 4 * route;
            expert = tuple[0];
            row_base = tuple[1];
            rows = tuple[2];
            m_tile = tuple[3];
          } else {
            expert = experts[route];
            row_base = route;
            rows = 1;
            m_tile = 0;
          }
          if (expert < 0 || expert >= 128) {
            for (int n = it.get_local_linear_id(); n < TN; n += threads)
              d[route * D + tile * TN + n] = cutlass::half_t(0.f);
            return;
          }
          const int a_row = Grouped ? row_base : (Gate ? route / 8 : route);
          auto A = MoE::make_moe_tensor<cutlass::half_t, 'R'>(
              const_cast<cutlass::half_t*>(a) + a_row * K, rows, K);
          auto B = MoE::make_moe_tensor<cutlass::float_e4m3_t, 'C'>(
              const_cast<cutlass::float_e4m3_t*>(b) + int64_t(expert) * K * N,
              N,
              K);
          auto C = MoE::make_moe_tensor<cutlass::half_t, 'R'>(
              d + row_base * D, rows, D);
          const cutlass::half_t* bias = nullptr;
          if constexpr (Gate && Variant < 4) {
            MoE::xe_gemm_gate_up<
                typename Policy::GmemTiledCopyA,
                typename Policy::GmemTiledCopyB,
                typename Policy::GmemTiledCopyD>(
                A, B, s + expert, bias, C, make_coord(m_tile, tile, _, 0), mma);
          } else {
            MoE::xe_gemm<
                typename Policy::GmemTiledCopyA,
                typename Policy::GmemTiledCopyB,
                typename Policy::GmemTiledCopyD>(
                A, B, s + expert, bias, C, make_coord(m_tile, tile, _, 0), mma);
          }
        });
  });
}

struct PrepareMap {
  const Half8* input;
  const int* ids;
  Half8* packed;
  int* map;
  int* reverse;
  int routes;
  void operator()
      [[sycl::reqd_sub_group_size(32)]] (sycl::nd_item<1> it) const {
    const int group = it.get_group(0), lane = it.get_local_id(0);
    if (group == 0) {
      int rows = 0;
      for (int j = 0; j < routes; ++j)
        rows += ids[j] == lane;
      const int tiles = (rows + 7) / 8;
      const int row_base = sycl::exclusive_scan_over_group(
          it.get_group(), rows, sycl::plus<int>());
      const int tile_base = sycl::exclusive_scan_over_group(
          it.get_group(), tiles, sycl::plus<int>());
      if (lane == 127) map[0] = tile_base + tiles;
      for (int j = 0; j < tiles; ++j) {
        int* p = map + 1 + 4 * (tile_base + j);
        p[0] = lane;
        p[1] = row_base;
        p[2] = rows;
        p[3] = j;
      }
      return;
    }
    const int route = group - 1, expert = ids[route];
    if (expert < 0 || expert >= 128) {
      if (lane == 0) reverse[route] = -1;
      return;
    }
    int before = 0;
    for (int j = lane; j < routes; j += 128)
      before += ids[j] >= 0 && ids[j] < 128 &&
                (ids[j] < expert || (ids[j] == expert && j < route));
    const int dest =
        sycl::reduce_over_group(it.get_group(), before, sycl::plus<int>());
    if (lane == 0) reverse[route] = dest;
    for (int col = lane; col < 352; col += 128)
      packed[dest * 352 + col] = input[(route / 8) * 352 + col];
  }
};

template <bool Grouped, int Variant>
void launch(
    at::Tensor output,
    const at::Tensor& input,
    const at::Tensor& w13,
    const at::Tensor& s13,
    const at::Tensor& w2,
    const at::Tensor& s2,
    const at::Tensor& weights,
    const at::Tensor& ids) {
  auto& q = c10::xpu::getCurrentXPUStream(input.get_device()).queue();
  const int m = input.size(0);
  auto activation = at::empty({m * 8, 352}, input.options());
  auto partial = at::empty({m * 8, 2816}, input.options());
  at::Tensor packed = input, map, reverse;
  int* map_ptr = nullptr;
  int* reverse_ptr = nullptr;
  if constexpr (Grouped) {
    packed = at::empty({m * 8, 2816}, input.options());
    map = at::empty({1 + 4 * m * 8}, ids.options());
    reverse = at::empty_like(ids);
    map_ptr = map.data_ptr<int>();
    reverse_ptr = reverse.data_ptr<int>();
    const PrepareMap prepare{
        reinterpret_cast<const Half8*>(input.data_ptr()),
        ids.data_ptr<int>(),
        reinterpret_cast<Half8*>(packed.data_ptr()),
        map_ptr,
        reverse_ptr,
        m * 8};
    const sycl::nd_range<1> range((1 + m * 8) * 128, 128);
    if constexpr (Variant == 5)
      q.parallel_for(range, Register128<1, PrepareMap>{prepare});
    else
      q.parallel_for(range, prepare);
  }
  if constexpr (Variant >= 4) {
    auto gate_up = at::empty({m * 8, 704}, input.options());
    gemm<true, Grouped, Variant>(q, packed, w13, s13, ids, gate_up, map_ptr);
    const auto* in = reinterpret_cast<const Half4*>(gate_up.data_ptr());
    auto* out = reinterpret_cast<Half4*>(activation.data_ptr());
    submit_vector<Activation<Variant>, Variant == 5>(
        q,
        sycl::nd_range<2>({size_t(m * 8), 96}, {1, 32}),
        [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(32)]] {
          const int row = it.get_group(0), col = it.get_global_id(1);
          if (col >= 88) return;
          const Half4 gate = in[row * 176 + col], up = in[row * 176 + 88 + col];
          Half4 value;
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const float x = float(gate.v[i]);
            float exponent =
                2.f * 0.7978845608028654f * (x + 0.044715f * (x * x * x));
            if (exponent > 30.f) exponent = 30.f;
            if (exponent < -30.f) exponent = -30.f;
            const float e = sycl::native::exp(exponent);
            const float t = (e - 1.f) * sycl::native::recip(e + 1.f);
            const sycl::half gelu = sycl::half(0.5f * x * (1.f + t));
            value.v[i] = sycl::half(float(gelu) * float(up.v[i]));
          }
          out[row * 88 + col] = value;
        });
  } else {
    gemm<true, Grouped, Variant>(q, packed, w13, s13, ids, activation, map_ptr);
  }
  gemm<false, Grouped, Variant>(q, activation, w2, s2, ids, partial, map_ptr);
  const auto* in = reinterpret_cast<const Half8*>(partial.data_ptr());
  const auto* w = weights.data_ptr<float>();
  auto* out = reinterpret_cast<Half8*>(output.data_ptr());
  submit_vector<RoutedGather<Grouped, Variant>, Variant == 5>(
      q,
      sycl::nd_range<2>({size_t(m), 352}, {1, 32}),
      [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(32)]] {
        const int row = it.get_group(0), col = it.get_global_id(1);
        float sum[8] = {};
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          const int src = Grouped ? reverse_ptr[row * 8 + k] : row * 8 + k;
          if (src < 0) continue;
          const Half8 value = in[src * 352 + col];
#pragma unroll
          for (int i = 0; i < 8; ++i)
            sum[i] += float(value.v[i]) * w[row * 8 + k];
        }
        Half8 value;
#pragma unroll
        for (int i = 0; i < 8; ++i)
          value.v[i] = sycl::half(sum[i]);
        out[row * 352 + col] = value;
      });
}

bool experts(
    at::Tensor output,
    const at::Tensor& input,
    const at::Tensor& w13,
    const at::Tensor& s13,
    const at::Tensor& w2,
    const at::Tensor& s2,
    const at::Tensor& weights,
    const at::Tensor& ids) {
  if (!input.is_xpu() || input.dim() != 2 || input.size(0) < 1 ||
      input.size(0) > 8 || input.size(1) != 2816 ||
      input.scalar_type() != at::kHalf || output.scalar_type() != at::kHalf ||
      output.sizes() != input.sizes() ||
      w13.scalar_type() != at::kFloat8_e4m3fn ||
      w2.scalar_type() != at::kFloat8_e4m3fn ||
      s13.scalar_type() != at::kFloat || s2.scalar_type() != at::kFloat ||
      weights.scalar_type() != at::kFloat || ids.scalar_type() != at::kInt ||
      w13.sizes() != at::IntArrayRef({128, 2816, 704}) ||
      w2.sizes() != at::IntArrayRef({128, 352, 2816}) ||
      s13.sizes() != at::IntArrayRef({128}) ||
      s2.sizes() != at::IntArrayRef({128}) ||
      ids.sizes() != at::IntArrayRef({input.size(0), 8}) ||
      weights.sizes() != ids.sizes())
    return false;
  for (const auto& t : {input, output, w13, s13, w2, s2, weights, ids})
    if (!t.is_contiguous() || t.device() != input.device() ||
        reinterpret_cast<uintptr_t>(t.data_ptr()) % 16)
      return false;
  for (const auto& t : {input, w13, s13, w2, s2, weights, ids})
    if (output.is_alias_of(t)) return false;
  // AOT is BMG-G31 only. Cache immutable architecture metadata, never tensors.
  static thread_local int cached_device = -1;
  static thread_local bool supported_arch = false;
  if (cached_device != input.get_device()) {
    namespace sx = sycl::ext::oneapi::experimental;
    supported_arch = c10::xpu::get_raw_device(input.get_device())
                         .get_info<sx::info::device::architecture>() ==
                     sx::architecture::intel_gpu_bmg_g31;
    cached_device = input.get_device();
  }
  if (!supported_arch || reinterpret_cast<uintptr_t>(input.data_ptr()) % 64 ||
      reinterpret_cast<uintptr_t>(w13.data_ptr()) % 64 ||
      reinterpret_cast<uintptr_t>(w2.data_ptr()) % 64)
    return false;
  const c10::DeviceGuard guard(input.device());
  // M<=4: direct routes avoid all packing. M>=5: group common experts so their
  // weight tiles can be reused; one prepare builds the map for both GEMMs.
  if (input.size(0) <= 4)
    launch<false, 0>(output, input, w13, s13, w2, s2, weights, ids);
  else if (input.size(0) == 5)
    launch<true, 5>(output, input, w13, s13, w2, s2, weights, ids);
  else
    launch<true, 4>(output, input, w13, s13, w2, s2, weights, ids);
  return true;
}
TORCH_LIBRARY_FRAGMENT(_xpu_C, m) {
  m.def(
      "fp8_moe_decode(Tensor(a!) output, Tensor input, Tensor w13, Tensor s13, "
      "Tensor w2, Tensor s2, Tensor weights, Tensor ids) -> bool");
}
TORCH_LIBRARY_IMPL(_xpu_C, XPU, m) { m.impl("fp8_moe_decode", &experts); }
// This try-op permits the pre-existing experts path during FakeTensor tracing.
TORCH_LIBRARY_IMPL(_xpu_C, Meta, m) {
  m.impl(
      "fp8_moe_decode",
      [](at::Tensor,
         const at::Tensor&,
         const at::Tensor&,
         const at::Tensor&,
         const at::Tensor&,
         const at::Tensor&,
         const at::Tensor&,
         const at::Tensor&) { return false; });
}
}  // namespace vllm::moe_decode
static PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_moe_decode_C",
    nullptr,
    -1,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr};
PyMODINIT_FUNC PyInit__moe_decode_C() { return PyModule_Create(&module); }
