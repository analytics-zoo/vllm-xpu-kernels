// SPDX-License-Identifier: Apache-2.0
// Decode helpers: deterministic route packing, vector gather/GELU and
// head-parallel RoPE.
#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <Python.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>
#include <torch/library.h>
#include <ATen/core/dispatch/Dispatcher.h>

namespace vllm::decode_aux {
namespace {
struct alignas(16) Half8 {
  sycl::half v[8];
};
struct alignas(8) Half4 {
  sycl::half v[4];
};

void tensor(
    const at::Tensor& t, const at::Tensor& anchor, at::ScalarType dtype) {
  TORCH_CHECK(
      t.device() == anchor.device() && t.scalar_type() == dtype &&
          t.is_contiguous(),
      "Expected contiguous tensor with matching device and dtype");
}
void separate(const at::Tensor& out, const at::Tensor& in) {
  at::assert_no_internal_overlap(out);
  at::assert_no_overlap(out, in);
}

struct Prepare {
  const sycl::half* input;
  const int* ids;
  sycl::half* packed;
  int* counts;
  int* reverse;
  int routes;
  void operator()
      [[sycl::reqd_sub_group_size(32)]] (sycl::nd_item<1> it) const {
    const int group = it.get_group(0), lane = it.get_local_id(0);
    if (group < 4) {
      const int expert = group * 32 + lane;
      int count = 0;
      for (int j = 0; j < routes; ++j)
        count += ids[j] == expert;
      counts[expert] =
          count;  // Overwrite every expert, including empty experts.
      return;
    }
    const int route = group - 4, expert = ids[route];
    if (expert < 0 || expert >= 128) {
      if (lane == 0) reverse[route] = -1;
      return;
    }
    int before = 0;
    for (int j = lane; j < routes; j += 32) {
      const int other = ids[j];
      before += other >= 0 && other < 128 &&
                (other < expert || (other == expert && j < route));
    }
    const int destination =
        sycl::reduce_over_group(it.get_sub_group(), before, sycl::plus<int>());
    if (lane == 0) reverse[route] = destination;
    const auto* src =
        reinterpret_cast<const Half8*>(input + (route / 8) * 2816);
    auto* dst = reinterpret_cast<Half8*>(packed + destination * 2816);
    for (int col = lane; col < 352; col += 32)
      dst[col] = src[col];
  }
};

void prepare(
    at::Tensor packed,
    at::Tensor counts,
    at::Tensor reverse,
    const at::Tensor& input,
    const at::Tensor& ids) {
  tensor(input, input, at::kHalf);
  tensor(ids, input, at::kInt);
  tensor(packed, input, at::kHalf);
  tensor(counts, input, at::kInt);
  tensor(reverse, input, at::kInt);
  TORCH_CHECK(
      input.dim() == 2 && input.size(0) >= 1 && input.size(0) <= 8 &&
          input.size(1) == 2816,
      "Expected input [M=1..8, 2816]");
  const int m = input.size(0);
  TORCH_CHECK(
      ids.sizes() == at::IntArrayRef({m, 8}) &&
          reverse.sizes() == ids.sizes() &&
          packed.sizes() == at::IntArrayRef({m * 8, 2816}) &&
          counts.sizes() == at::IntArrayRef({128}),
      "Expected topk=8, E=128 and matching outputs");
  for (const auto& out : {packed, counts, reverse}) {
    separate(out, input);
    separate(out, ids);
  }
  separate(packed, counts);
  separate(packed, reverse);
  separate(counts, reverse);
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(input.data_ptr()) % 16 == 0 &&
          reinterpret_cast<uintptr_t>(packed.data_ptr()) % 16 == 0,
      "Expected 16-byte aligned FP16 data");
  const c10::DeviceGuard guard(input.device());
  auto& q = c10::xpu::getCurrentXPUStream(input.get_device()).queue();
  q.parallel_for(
      sycl::nd_range<1>((4 + m * 8) * 32, 32),
      Prepare{
          reinterpret_cast<const sycl::half*>(input.data_ptr()),
          ids.data_ptr<int>(),
          reinterpret_cast<sycl::half*>(packed.data_ptr()),
          counts.data_ptr<int>(),
          reverse.data_ptr<int>(),
          m * 8});
}

struct Gather {
  const sycl::half* input;
  const float* weights;
  const int* reverse;
  sycl::half* output;
  int routes;
  void operator()
      [[sycl::reqd_sub_group_size(32)]] (sycl::nd_item<2> it) const {
    const int row = it.get_group(0), col = it.get_global_id(1);
    float accum[8] = {};
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int src = reverse[row * 8 + k];
      if (src < 0 || src >= routes) continue;
      const auto values =
          reinterpret_cast<const Half8*>(input + src * 2816)[col];
      const float weight = weights[row * 8 + k];
#pragma unroll
      for (int i = 0; i < 8; ++i)
        accum[i] += float(values.v[i]) * weight;
    }
    Half8 result;
#pragma unroll
    for (int i = 0; i < 8; ++i)
      result.v[i] = sycl::half(accum[i]);
    reinterpret_cast<Half8*>(output + row * 2816)[col] = result;
  }
};

void gather(
    at::Tensor output,
    const at::Tensor& input,
    const at::Tensor& weights,
    const at::Tensor& reverse) {
  tensor(output, input, at::kHalf);
  tensor(input, input, at::kHalf);
  tensor(weights, input, at::kFloat);
  tensor(reverse, input, at::kInt);
  TORCH_CHECK(
      output.dim() == 2 && output.size(0) >= 1 && output.size(0) <= 8 &&
          output.size(1) == 2816,
      "Expected output [M=1..8,2816]");
  const int m = output.size(0);
  TORCH_CHECK(
      input.sizes() == at::IntArrayRef({m * 8, 2816}) &&
          weights.sizes() == at::IntArrayRef({m, 8}) &&
          reverse.sizes() == weights.sizes(),
      "Expected topk=8 and matching tensors");
  separate(output, input);
  separate(output, weights);
  separate(output, reverse);
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(input.data_ptr()) % 16 == 0 &&
          reinterpret_cast<uintptr_t>(output.data_ptr()) % 16 == 0,
      "Expected 16-byte aligned FP16 data");
  const c10::DeviceGuard guard(input.device());
  auto& q = c10::xpu::getCurrentXPUStream(input.get_device()).queue();
  q.parallel_for(
      sycl::nd_range<2>({size_t(m), 352}, {1, 32}),
      Gather{
          reinterpret_cast<const sycl::half*>(input.data_ptr()),
          weights.data_ptr<float>(),
          reverse.data_ptr<int>(),
          reinterpret_cast<sycl::half*>(output.data_ptr()),
          m * 8});
}

struct Gelu {
  const sycl::half* input;
  sycl::half* output;
  int width;
  void operator()
      [[sycl::reqd_sub_group_size(32)]] (sycl::nd_item<2> it) const {
    const int row = it.get_group(0), col = it.get_global_id(1);
    if (col * 4 >= width) return;
    const auto gate =
        reinterpret_cast<const Half4*>(input + row * width * 2)[col];
    const auto up =
        reinterpret_cast<const Half4*>(input + row * width * 2 + width)[col];
    Half4 result;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float x = float(gate.v[i]);
      const float inner = 0.7978845608028654f * (x + 0.044715f * (x * x * x));
      // ESIMD uses an exponential form. Saturate the exponent to avoid inf/inf
      // for large finite gates; preserve NaN propagation with explicit
      // comparisons.
      float exponent = 2.0f * inner;
      if (exponent > 30.0f) exponent = 30.0f;
      if (exponent < -30.0f) exponent = -30.0f;
      const float e = sycl::native::exp(exponent);
      const float tanh_value = (e - 1.0f) * sycl::native::recip(e + 1.0f);
      const sycl::half activated = sycl::half(0.5f * x * (1.0f + tanh_value));
      result.v[i] = sycl::half(float(activated) * float(up.v[i]));
    }
    reinterpret_cast<Half4*>(output + row * width)[col] = result;
  }
};

void gelu(at::Tensor output, const at::Tensor& input) {
  tensor(input, input, at::kHalf);
  tensor(output, input, at::kHalf);
  TORCH_CHECK(
      input.dim() == 2 && output.dim() == 2 &&
          input.size(0) == output.size(0) &&
          input.size(1) == 2 * output.size(1) && input.size(0) >= 1 &&
          input.size(0) <= 64 &&
          (output.size(1) == 352 || output.size(1) == 1056),
      "Expected FP16 GELU [1..64, 2*(352|1056)]");
  separate(output, input);
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(input.data_ptr()) % 8 == 0 &&
          reinterpret_cast<uintptr_t>(output.data_ptr()) % 8 == 0,
      "Expected 8-byte aligned FP16 data");
  const int rows = input.size(0), width = output.size(1),
            threads = ((width / 4 + 31) / 32) * 32;
  const c10::DeviceGuard guard(input.device());
  auto& q = c10::xpu::getCurrentXPUStream(input.get_device()).queue();
  q.parallel_for(
      sycl::nd_range<2>({size_t(rows), size_t(threads)}, {1, 32}),
      Gelu{
          reinterpret_cast<const sycl::half*>(input.data_ptr()),
          reinterpret_cast<sycl::half*>(output.data_ptr()),
          width});
}

struct Rope {
  const int64_t* positions;
  sycl::half* query;
  sycl::half* key;
  const sycl::half* cache;
  int qheads, kheads, head, rotary;
  int64_t qstride, kstride, cache_rows;
  void operator()
      [[sycl::reqd_sub_group_size(32)]] (sycl::nd_item<2> it) const {
    const int row = it.get_group(0), h = it.get_group(1),
              lane = it.get_local_id(1);
    const auto pos = positions[row];
    if (pos < 0 || pos >= cache_rows) return;
    auto* data = h < qheads ? query + row * qstride + h * head
                            : key + row * kstride + (h - qheads) * head;
    const auto* cs = cache + pos * rotary;
    const int half = rotary / 2;
    for (int col = lane; col < half; col += 32) {
      const sycl::half x = data[col], y = data[col + half], c = cs[col],
                       s = cs[col + half];
      // Preserve the FP16 products and add/subtract of the upstream operator.
      const sycl::half xc = sycl::half(float(x) * float(c)),
                       ys = sycl::half(float(y) * float(s));
      const sycl::half yc = sycl::half(float(y) * float(c)),
                       xs = sycl::half(float(x) * float(s));
      data[col] = sycl::half(float(xc) - float(ys));
      data[col + half] = sycl::half(float(yc) + float(xs));
    }
  }
};

void rope(
    const at::Tensor& positions,
    at::Tensor query,
    at::Tensor key,
    int64_t head,
    const at::Tensor& cache) {
  tensor(positions, query, at::kLong);
  tensor(cache, query, at::kHalf);
  TORCH_CHECK(
      query.scalar_type() == at::kHalf && key.scalar_type() == at::kHalf &&
          query.device() == key.device() && query.dim() == 2 &&
          key.dim() == 2 && query.stride(1) == 1 && key.stride(1) == 1 &&
          positions.dim() == 1 && positions.numel() >= 1 &&
          positions.numel() <= 8 && query.size(0) == positions.numel() &&
          key.size(0) == positions.numel() && (head == 256 || head == 512) &&
          query.size(1) % head == 0 && key.size(1) % head == 0 &&
          query.size(1) > 0 && key.size(1) > 0 &&
          query.stride(0) >= query.size(1) && key.stride(0) >= key.size(1) &&
          cache.dim() == 2 && cache.size(1) > 0 && cache.size(1) <= head &&
          cache.size(1) % 64 == 0,
      "Expected small-M FP16 NeoX RoPE with head=256|512 and rotary multiple "
      "of 64");
  // Q and K may be disjoint slices of a single QKV tensor with a common row
  // stride.
  TORCH_CHECK(
      query.data_ptr() != key.data_ptr(),
      "Q/K must not alias the same elements");
  const int m = positions.numel(), qh = query.size(1) / head,
            kh = key.size(1) / head;
  const c10::DeviceGuard guard(query.device());
  auto& q = c10::xpu::getCurrentXPUStream(query.get_device()).queue();
  q.parallel_for(
      sycl::nd_range<2>({size_t(m), size_t(qh + kh) * 32}, {1, 32}),
      Rope{
          positions.data_ptr<int64_t>(),
          reinterpret_cast<sycl::half*>(query.data_ptr()),
          reinterpret_cast<sycl::half*>(key.data_ptr()),
          reinterpret_cast<const sycl::half*>(cache.data_ptr()),
          qh,
          kh,
          int(head),
          int(cache.size(1)),
          query.stride(0),
          key.stride(0),
          cache.size(0)});
}
bool try_rope(
    const at::Tensor& positions,
    at::Tensor query,
    at::Tensor key,
    int64_t head,
    const at::Tensor& cache) {
  if (!query.is_xpu() || query.scalar_type() != at::kHalf ||
      key.scalar_type() != at::kHalf || cache.scalar_type() != at::kHalf ||
      positions.scalar_type() != at::kLong || key.device() != query.device() ||
      cache.device() != query.device() ||
      positions.device() != query.device() || !positions.is_contiguous() ||
      !cache.is_contiguous() || query.dim() != 2 || key.dim() != 2 ||
      positions.dim() != 1 || positions.numel() < 1 || positions.numel() > 8 ||
      query.size(0) != positions.numel() || key.size(0) != positions.numel() ||
      query.stride(1) != 1 || key.stride(1) != 1 ||
      (head != 256 && head != 512) || query.size(1) <= 0 || key.size(1) <= 0 ||
      query.size(1) % head || key.size(1) % head ||
      query.stride(0) < query.size(1) || key.stride(0) < key.size(1) ||
      cache.dim() != 2 || cache.size(1) <= 0 || cache.size(1) > head ||
      cache.size(1) % 64 || query.data_ptr() == key.data_ptr())
    return false;
  rope(positions, query, key, head, cache);
  return true;
}

at::Tensor gelu_dispatch(const at::Tensor& x) {
  auto shape = x.sizes().vec();
  TORCH_CHECK(!shape.empty(), "GELU requires at least one dimension");
  shape.back() /= 2;
  auto out = at::empty(shape, x.options());
  if (x.is_xpu() && x.scalar_type() == at::kHalf && x.dim() == 2 &&
      x.size(0) >= 1 && x.size(0) <= 64 &&
      (x.size(1) == 704 || x.size(1) == 2112) && x.is_contiguous() &&
      reinterpret_cast<uintptr_t>(x.data_ptr()) % 8 == 0) {
    gelu(out, x);
  } else {
    using F = void(at::Tensor&, at::Tensor&);
    static auto old = c10::Dispatcher::singleton()
                          .findSchemaOrThrow("_C::gelu_tanh_and_mul", "")
                          .typed<F>();
    at::Tensor input = x;
    old.call(out, input);
  }
  return out;
}

at::Tensor rms_dispatch(
    const at::Tensor& x, const std::optional<at::Tensor>& weight, double eps) {
  if (x.is_xpu() && x.scalar_type() == at::kHalf && x.dim() >= 2 &&
      x.dim() <= 4 && x.size(0) >= 1 && x.size(0) <= 8 &&
      (x.size(-1) == 256 || x.size(-1) == 512 || x.size(-1) == 2816 ||
       x.size(-1) == 5376) &&
      x.numel() / x.size(-1) >= 1 && x.numel() / x.size(-1) <= 128 &&
      x.stride(-1) == 1 && eps >= 0 &&
      (!weight || (weight->device() == x.device() &&
                   weight->scalar_type() == at::kHalf && weight->dim() == 1 &&
                   weight->numel() == x.size(-1) && weight->is_contiguous()))) {
    using F =
        at::Tensor(const at::Tensor&, const std::optional<at::Tensor>&, double);
    static auto fast = c10::Dispatcher::singleton()
                           .findSchemaOrThrow("_xpu_C::rms_norm_small_m", "")
                           .typed<F>();
    return fast.call(x, weight, eps);
  }
  auto w = weight ? *weight : at::ones({x.size(-1)}, x.options());
  auto out = at::empty(x.sizes(), x.options());
  using F = void(at::Tensor&, at::Tensor&, at::Tensor&, double);
  static auto old = c10::Dispatcher::singleton()
                        .findSchemaOrThrow("_C::rms_norm", "")
                        .typed<F>();
  at::Tensor input = x;
  old.call(out, input, w, eps);
  return out;
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(_xpu_C, m) {
  m.def("gelu_tanh_and_mul_decode(Tensor input) -> Tensor");
  m.def(
      "rms_norm_decode_dispatch(Tensor input, Tensor? weight, float epsilon) "
      "-> Tensor");
  m.def(
      "try_rotary_embedding_small_m(Tensor positions, Tensor(a!) query, "
      "Tensor(b!) key, int head, Tensor cache) -> bool");
  m.def(
      "moe_prepare_small_m(Tensor(a!) packed, Tensor(b!) counts, Tensor(c!) "
      "reverse, Tensor input, Tensor ids) -> ()");
  m.def(
      "moe_gather_small_m(Tensor(a!) output, Tensor input, Tensor weights, "
      "Tensor reverse) -> ()");
  m.def("gelu_tanh_and_mul_small_m(Tensor(a!) output, Tensor input) -> ()");
  m.def(
      "rotary_embedding_small_m(Tensor positions, Tensor(a!) query, "
      "Tensor(b!) key, int head, Tensor cache) -> ()");
}
TORCH_LIBRARY_IMPL(_xpu_C, XPU, m) {
  m.impl("gelu_tanh_and_mul_decode", &gelu_dispatch);
  m.impl("rms_norm_decode_dispatch", &rms_dispatch);
  m.impl("try_rotary_embedding_small_m", &try_rope);
  m.impl("moe_prepare_small_m", &prepare);
  m.impl("moe_gather_small_m", &gather);
  m.impl("gelu_tanh_and_mul_small_m", &gelu);
  m.impl("rotary_embedding_small_m", &rope);
}
TORCH_LIBRARY_IMPL(_xpu_C, Meta, m) {
  m.impl("gelu_tanh_and_mul_decode", [](const at::Tensor& x) {
    auto s = x.sym_sizes().vec();
    s.back() /= 2;
    return at::empty_symint(s, x.options());
  });
  m.impl(
      "rms_norm_decode_dispatch",
      [](const at::Tensor& x, const std::optional<at::Tensor>&, double) {
        return at::empty_symint(x.sym_sizes(), x.options());
      });
  m.impl(
      "try_rotary_embedding_small_m",
      [](const at::Tensor&,
         at::Tensor,
         at::Tensor,
         int64_t,
         const at::Tensor&) { return false; });
  m.impl(
      "moe_prepare_small_m",
      [](at::Tensor,
         at::Tensor,
         at::Tensor,
         const at::Tensor&,
         const at::Tensor&) {});
  m.impl(
      "moe_gather_small_m",
      [](at::Tensor, const at::Tensor&, const at::Tensor&, const at::Tensor&) {
      });
  m.impl("gelu_tanh_and_mul_small_m", [](at::Tensor, const at::Tensor&) {});
  m.impl(
      "rotary_embedding_small_m",
      [](const at::Tensor&,
         at::Tensor,
         at::Tensor,
         int64_t,
         const at::Tensor&) {});
}
}  // namespace vllm::decode_aux

static PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_decode_aux_C", nullptr, -1, nullptr};
PyMODINIT_FUNC PyInit__decode_aux_C() { return PyModule_Create(&module); }
