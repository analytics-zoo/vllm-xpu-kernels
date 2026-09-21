#include "gemma_decode_fusions.hpp"

#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>

namespace vllm::gemma_decode {
namespace {

constexpr int kHeadSize = 256;
constexpr int kQueryHeads = 8;
constexpr int kKvHeads = 4;
constexpr int kAllHeads = kQueryHeads + 2 * kKvHeads;
constexpr int kQkvWidth = kAllHeads * kHeadSize;

const sycl::half* ptr(const at::Tensor& tensor) {
  return reinterpret_cast<const sycl::half*>(tensor.data_ptr<at::Half>());
}

sycl::half* mutable_ptr(at::Tensor& tensor) {
  return reinterpret_cast<sycl::half*>(tensor.data_ptr<at::Half>());
}

void check_half_tensor(const at::Tensor& tensor, const at::Device& device) {
  TORCH_CHECK(tensor.device() == device && tensor.scalar_type() == at::kHalf &&
                  tensor.is_contiguous(),
              "Expected contiguous FP16 tensor on the input XPU device");
}

template <typename Position>
struct QkvNormRopeKernel {
  const sycl::half* qkv;
  const sycl::half* q_weight;
  const sycl::half* k_weight;
  const Position* positions;
  const sycl::half* cache;
  sycl::half* query;
  sycl::half* key;
  sycl::half* value;
  int64_t cache_rows;
  float q_epsilon;
  float k_epsilon;
  float v_epsilon;

  void operator()(sycl::nd_item<1> item) const {
    const int lane = item.get_local_linear_id();
    const int group = item.get_group_linear_id();
    const int token = group / kAllHeads;
    const int head = group % kAllHeads;
    const int64_t source = int64_t(token) * kQkvWidth + head * kHeadSize;
    const float x = float(qkv[source + lane]);
    const float y = float(qkv[source + lane + kHeadSize / 2]);
    const float sum = sycl::reduce_over_group(
        item.get_group(), x * x + y * y, sycl::plus<float>());
    const bool is_query = head < kQueryHeads;
    const bool is_value = head >= kQueryHeads + kKvHeads;
    const float epsilon = is_query ? q_epsilon : is_value ? v_epsilon : k_epsilon;
    const float inverse_rms = sycl::rsqrt(sum / kHeadSize + epsilon);
    sycl::half a = sycl::half(x * inverse_rms);
    sycl::half b = sycl::half(y * inverse_rms);
    sycl::half* output;
    int output_head;
    int output_heads;
    if (is_query) {
      a = a * q_weight[lane];
      b = b * q_weight[lane + kHeadSize / 2];
      output = query;
      output_head = head;
      output_heads = kQueryHeads;
    } else if (!is_value) {
      a = a * k_weight[lane];
      b = b * k_weight[lane + kHeadSize / 2];
      output = key;
      output_head = head - kQueryHeads;
      output_heads = kKvHeads;
    } else {
      output = value;
      output_head = head - kQueryHeads - kKvHeads;
      output_heads = kKvHeads;
    }
    if (!is_value) {
      const int64_t position = positions[token];
      if (position >= 0 && position < cache_rows) {
        const sycl::half cosine = cache[position * kHeadSize + lane];
        const sycl::half sine =
            cache[position * kHeadSize + lane + kHeadSize / 2];
        // Match upstream half operators, including their storage rounding.
        const sycl::half rotated_a = a * cosine - b * sine;
        const sycl::half rotated_b = b * cosine + a * sine;
        a = rotated_a;
        b = rotated_b;
      } else {
        a = b = sycl::half(sycl::nan(0u));
      }
    }
    const int64_t destination =
        (int64_t(token) * output_heads + output_head) * kHeadSize;
    output[destination + lane] = a;
    output[destination + lane + kHeadSize / 2] = b;
  }
};

template <typename Position>
void launch_qkv(
    sycl::queue& queue,
    const at::Tensor& qkv,
    const at::Tensor& q_weight,
    const at::Tensor& k_weight,
    const at::Tensor& positions,
    const at::Tensor& cache,
    at::Tensor& query,
    at::Tensor& key,
    at::Tensor& value,
    double q_epsilon,
    double k_epsilon,
    double v_epsilon) {
  constexpr int group_size = kHeadSize / 2;
  queue.parallel_for(
      sycl::nd_range<1>(qkv.size(0) * kAllHeads * group_size, group_size),
      QkvNormRopeKernel<Position>{
          ptr(qkv), ptr(q_weight), ptr(k_weight), positions.data_ptr<Position>(),
          ptr(cache), mutable_ptr(query), mutable_ptr(key), mutable_ptr(value),
          cache.size(0), float(q_epsilon), float(k_epsilon), float(v_epsilon)});
}

struct RouterNormKernel {
  const sycl::half* input;
  const sycl::half* router_scale;
  const void* root_size;
  const sycl::half* moe_weight;
  sycl::half* router_input;
  sycl::half* moe_input;
  int width;
  bool root_is_float;
  float router_epsilon;
  float moe_epsilon;

  void operator()(sycl::nd_item<1> item) const {
    const int lane = item.get_local_linear_id();
    const int64_t row = int64_t(item.get_group_linear_id()) * width;
    float squared_sum = 0.0f;
    for (int k = lane; k < width; k += 256) {
      const float value = float(input[row + k]);
      squared_sum += value * value;
    }
    squared_sum = sycl::reduce_over_group(
        item.get_group(), squared_sum, sycl::plus<float>());
    const float inverse_router = sycl::rsqrt(squared_sum / width + router_epsilon);
    const float inverse_moe = sycl::rsqrt(squared_sum / width + moe_epsilon);
    const sycl::half root = root_is_float
                                ? sycl::half(static_cast<const float*>(root_size)[0])
                                : static_cast<const sycl::half*>(root_size)[0];
    for (int k = lane; k < width; k += 256) {
      const float value = float(input[row + k]);
      const sycl::half router_normalized = sycl::half(value * inverse_router);
      const sycl::half rooted = router_normalized * root;
      router_input[row + k] = rooted * router_scale[k];
      const sycl::half moe_normalized = sycl::half(value * inverse_moe);
      moe_input[row + k] = moe_normalized * moe_weight[k];
    }
  }
};

struct RouterGemvKernel {
  const sycl::half* router_input;
  const sycl::half* projection;
  float* output;
  int width;
  int experts;

  void operator()(sycl::nd_item<1> item) const {
    const int lane = item.get_local_linear_id();
    const int group = item.get_group_linear_id();
    const int row = group / experts;
    const int expert = group % experts;
    float dot = 0.0f;
    for (int k = lane; k < width; k += 64) {
      dot += float(router_input[int64_t(row) * width + k]) *
             float(projection[int64_t(expert) * width + k]);
    }
    dot = sycl::reduce_over_group(item.get_group(), dot, sycl::plus<float>());
    if (lane == 0) {
      // GateLinear on XPU uses FP16 F.linear, then casts its result to FP32.
      output[int64_t(row) * experts + expert] = float(sycl::half(dot));
    }
  }
};

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> qkv_norm_rope(
    const at::Tensor& qkv,
    const at::Tensor& q_weight,
    const at::Tensor& k_weight,
    const at::Tensor& positions,
    const at::Tensor& cos_sin_cache,
    double q_epsilon,
    double k_epsilon,
    double v_epsilon) {
  TORCH_CHECK(qkv.is_xpu(), "Expected XPU qkv input");
  check_half_tensor(qkv, qkv.device());
  TORCH_CHECK(qkv.dim() == 2 && qkv.size(0) >= 1 && qkv.size(0) <= 8 &&
                  qkv.size(1) == kQkvWidth,
              "This path requires [M,4096], 1 <= M <= 8, Q8/KV4/D256");
  check_half_tensor(q_weight, qkv.device());
  check_half_tensor(k_weight, qkv.device());
  TORCH_CHECK(q_weight.dim() == 1 && q_weight.numel() == kHeadSize &&
                  k_weight.dim() == 1 && k_weight.numel() == kHeadSize,
              "Expected Q/K norm weights [256]");
  check_half_tensor(cos_sin_cache, qkv.device());
  TORCH_CHECK(cos_sin_cache.dim() == 2 && cos_sin_cache.size(1) == kHeadSize,
              "Expected full NeoX rotary cache [P,256]");
  TORCH_CHECK(positions.device() == qkv.device() && positions.dim() == 1 &&
                  positions.numel() == qkv.size(0) && positions.is_contiguous() &&
                  (positions.scalar_type() == at::kInt ||
                   positions.scalar_type() == at::kLong),
              "Expected contiguous int32/int64 positions [M]");
  TORCH_CHECK(q_epsilon >= 0 && k_epsilon >= 0 && v_epsilon >= 0,
              "Expected nonnegative norm epsilons");
  const c10::DeviceGuard guard(qkv.device());
  auto query = at::empty({qkv.size(0), kQueryHeads * kHeadSize}, qkv.options());
  auto key = at::empty({qkv.size(0), kKvHeads * kHeadSize}, qkv.options());
  auto value = at::empty_like(key);
  auto& queue = c10::xpu::getCurrentXPUStream(qkv.get_device()).queue();
  if (positions.scalar_type() == at::kInt) {
    launch_qkv<int32_t>(queue, qkv, q_weight, k_weight, positions, cos_sin_cache,
                        query, key, value, q_epsilon, k_epsilon, v_epsilon);
  } else {
    launch_qkv<int64_t>(queue, qkv, q_weight, k_weight, positions, cos_sin_cache,
                        query, key, value, q_epsilon, k_epsilon, v_epsilon);
  }
  return {query, key, value};
}

std::tuple<at::Tensor, at::Tensor> norm_router_norm(
    const at::Tensor& input,
    const at::Tensor& router_scale,
    const at::Tensor& root_size,
    const at::Tensor& projection,
    const at::Tensor& moe_weight,
    double router_epsilon,
    double moe_epsilon) {
  TORCH_CHECK(input.is_xpu(), "Expected XPU input");
  check_half_tensor(input, input.device());
  TORCH_CHECK(input.dim() == 2 && input.size(0) >= 1 && input.size(0) <= 8 &&
                  input.size(1) == 2816,
              "This path requires input [M,2816], 1 <= M <= 8");
  check_half_tensor(router_scale, input.device());
  check_half_tensor(moe_weight, input.device());
  check_half_tensor(projection, input.device());
  TORCH_CHECK(router_scale.dim() == 1 && router_scale.numel() == input.size(1) &&
                  moe_weight.dim() == 1 && moe_weight.numel() == input.size(1),
              "Expected router_scale and moe_weight [2816]");
  TORCH_CHECK(projection.dim() == 2 && projection.size(0) == 128 &&
                  projection.size(1) == input.size(1),
              "Expected projection weight [128,2816]");
  TORCH_CHECK(root_size.device() == input.device() && root_size.numel() == 1 &&
                  root_size.is_contiguous() &&
                  (root_size.scalar_type() == at::kFloat ||
                   root_size.scalar_type() == at::kHalf),
              "Expected root_size scalar, FP16 or FP32 on the input device");
  TORCH_CHECK(router_epsilon >= 0 && moe_epsilon >= 0,
              "Expected nonnegative norm epsilons");
  const c10::DeviceGuard guard(input.device());
  auto router_input = at::empty_like(input);
  auto moe_input = at::empty_like(input);
  auto logits = at::empty({input.size(0), projection.size(0)},
                          input.options().dtype(at::kFloat));
  auto& queue = c10::xpu::getCurrentXPUStream(input.get_device()).queue();
  queue.parallel_for(
      sycl::nd_range<1>(input.size(0) * 256, 256),
      RouterNormKernel{ptr(input), ptr(router_scale), root_size.data_ptr(),
                       ptr(moe_weight), mutable_ptr(router_input),
                       mutable_ptr(moe_input), int(input.size(1)),
                       root_size.scalar_type() == at::kFloat,
                       float(router_epsilon), float(moe_epsilon)});
  queue.parallel_for(
      sycl::nd_range<1>(input.size(0) * projection.size(0) * 64, 64),
      RouterGemvKernel{ptr(router_input), ptr(projection), logits.data_ptr<float>(),
                       int(input.size(1)), int(projection.size(0))});
  return {logits, moe_input};
}

}  // namespace vllm::gemma_decode
