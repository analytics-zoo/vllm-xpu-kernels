// SPDX-License-Identifier: Apache-2.0
// Explicit CPU selector: this executable must never acquire a GPU device.
// Include the owned implementation so its raw SYCL launchers, rather than a
// duplicated Python reference, are what this CPU-only test executes.
#include "../csrc/qwen38/qsa_sycl_aux.cpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace vllm::qwen38::qsa_sycl {
at::Tensor select_paged_tokens_v2(
    const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t,
    int64_t, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor) {
  throw std::runtime_error("CPU test must not call the GPU selection wrapper");
}
}  // namespace vllm::qwen38::qsa_sycl

using vllm::qwen38::qsa_sycl::launch_indexer;
using vllm::qwen38::qsa_sycl::launch_qkv;
using vllm::qwen38::qsa_sycl::launch_projection_int4;
using vllm::qwen38::qsa_sycl::round_fp16;

void close(float actual, float expected, float tolerance, const char* label) {
  if (std::abs(actual - expected) > tolerance) {
    std::cerr << label << ": actual=" << actual
              << ", expected=" << expected << '\n';
    throw std::runtime_error(label);
  }
}

int main() {
  sycl::queue queue(sycl::cpu_selector_v);
  using half = sycl::half;
  for (float value : {0.0f, 1.00048828125f, -0.3759765625f,
                      65504.0f, 65520.0f}) {
    close(round_fp16(value), static_cast<float>(half(value)), 0.0f,
          "FP16 RNE boundary");
  }
  // Intel's CPU SYCL half conversion may flush subnormals; use IEEE-754
  // binary16 bit-pattern expectations for this boundary instead.
  close(round_fp16(0x1p-24f), 0x1p-24f, 0.0f, "FP16 min subnormal");
  close(round_fp16(0x1.8p-25f), 0x1p-24f, 0.0f,
        "FP16 subnormal round-up");
  constexpr int rows = 2;
  constexpr int q_heads = 3;
  constexpr int segments = 2 * q_heads + 2;
  auto* qkv = sycl::malloc_shared<half>(rows * segments * 256, queue);
  auto* q = sycl::malloc_shared<half>(rows * q_heads * 256, queue);
  auto* gate = sycl::malloc_shared<half>(rows * q_heads * 256, queue);
  auto* k = sycl::malloc_shared<half>(rows * 256, queue);
  auto* v = sycl::malloc_shared<half>(rows * 256, queue);
  auto* weight = sycl::malloc_shared<half>(256, queue);
  auto* cache = sycl::malloc_shared<half>(3 * 64, queue);
  auto* positions = sycl::malloc_shared<int32_t>(3 * rows, queue);
  for (int i = 0; i < rows * segments * 256; ++i) {
    qkv[i] = half(static_cast<float>((i % 31) - 15) * 0.0625f);
  }
  for (int i = 0; i < 256; ++i) weight[i] = half(0.0f);
  for (int position = 0; position < 3; ++position) {
    const float cosine = position == 0 ? 1.0f :
                         position == 1 ? 0.75f : 0.5f;
    const float sine = position == 0 ? 0.0f :
                       position == 1 ? 0.25f : 0.5f;
    for (int d = 0; d < 32; ++d) {
      cache[position * 64 + d] = half(cosine);
      cache[position * 64 + 32 + d] = half(sine);
    }
  }
  for (int row = 0; row < rows; ++row) {
    for (int axis = 0; axis < 3; ++axis) {
      positions[axis * rows + row] = axis;
    }
  }
  launch_qkv<true, int32_t>(queue, qkv, q, gate, k, v, weight, weight,
                             positions, cache, rows, q_heads, 1, true,
                             rows, 1, 3);
  queue.wait_and_throw();
  for (int row = 0; row < rows; ++row) {
    const int64_t source = row * segments * 256;
    float sum = 0.0f;
    for (int d = 0; d < 256; ++d) {
      const float value = static_cast<float>(qkv[source + d]);
      sum += value * value;
    }
    const float inv = 1.0f / std::sqrt(sum / 256.0f + 1e-6f);
    for (int pair = 0; pair < 3; ++pair) {
      const int axis = pair % 3;
      const float first = static_cast<float>(qkv[source + pair]) * inv;
      const float second = static_cast<float>(qkv[source + 32 + pair]) * inv;
      const float cosine = static_cast<float>(cache[axis * 64 + pair]);
      const float sine = static_cast<float>(cache[axis * 64 + 32 + pair]);
      close(static_cast<float>(q[row * q_heads * 256 + pair]),
            first * cosine - second * sine, 0.002f, "QKV MRoPE first");
      close(static_cast<float>(q[row * q_heads * 256 + 32 + pair]),
            second * cosine + first * sine, 0.002f, "QKV MRoPE second");
    }
    const float gate_input = static_cast<float>(qkv[source + 256]);
    close(static_cast<float>(gate[row * q_heads * 256]),
          1.0f / (1.0f + std::exp(-gate_input)), 0.001f,
          "interleaved gate sigmoid");
    close(static_cast<float>(v[row * 256]),
          static_cast<float>(qkv[source + (segments - 1) * 256]),
          0.0f, "V copy");
  }

  auto* index_input = sycl::malloc_shared<half>(rows * 4 * 128, queue);
  auto* index_output = sycl::malloc_shared<half>(rows * 4 * 128, queue);
  auto* index_eager = sycl::malloc_shared<half>(rows * 4 * 128, queue);
  auto* index_weight = sycl::malloc_shared<half>(128, queue);
  for (int i = 0; i < rows * 4 * 128; ++i)
    index_input[i] = half(static_cast<float>((i % 37) - 18) * 0.03125f);
  for (int i = 0; i < 128; ++i) index_weight[i] = half(0.0f);
  launch_indexer<true, false, int32_t>(
      queue, index_input, index_output, index_weight, positions, cache,
      rows, 4, 4 * 128, rows, 1, 3);
  launch_indexer<true, true, int32_t>(
      queue, index_input, index_eager, index_weight, positions, cache,
      rows, 4, 4 * 128, rows, 1, 3);
  queue.wait_and_throw();
  float square_sum = 0.0f;
  for (int d = 0; d < 128; ++d) {
    const float value = static_cast<float>(index_input[d]);
    square_sum += value * value;
  }
  const float inv = 1.0f / std::sqrt(square_sum / 128.0f + 1e-6f);
  for (int pair = 0; pair < 3; ++pair) {
    const float first = static_cast<float>(index_input[pair]) * inv;
    const float second = static_cast<float>(index_input[32 + pair]) * inv;
    const float cosine = static_cast<float>(cache[pair * 64 + pair]);
    const float sine = static_cast<float>(cache[pair * 64 + 32 + pair]);
    close(static_cast<float>(index_output[pair]),
          first * cosine - second * sine, 0.002f,
          "indexer FP32 norm+MRoPE");
    const float eager_first = round_fp16(first);
    const float eager_second = round_fp16(second);
    const float eager_result = round_fp16(
        round_fp16(eager_first * cosine) -
        round_fp16(eager_second * sine));
    close(static_cast<float>(index_eager[pair]), eager_result, 0.001f,
          "indexer eager FP16 boundary");
  }
  auto* projection_input = sycl::malloc_shared<half>(rows * 2560, queue);
  auto* projection_weight = sycl::malloc_shared<uint8_t>(640 * 1280, queue);
  auto* projection_scales = sycl::malloc_shared<half>(640 * 20, queue);
  auto* projection_output = sycl::malloc_shared<half>(rows * 640, queue);
  for (int row = 0; row < rows; ++row) {
    for (int index = 0; index < 2560; ++index) {
      projection_input[row * 2560 + index] =
          half(index & 1 ? static_cast<float>(row + 1) : 0.0f);
    }
  }
  for (int index = 0; index < 640 * 1280; ++index)
    projection_weight[index] = 0x98;  // low even=8/zero, high odd=9/+1
  for (int index = 0; index < 640 * 20; ++index)
    projection_scales[index] = half(0.1f);
  launch_projection_int4(queue, projection_input, projection_weight,
                         projection_scales, projection_output, rows);
  queue.wait_and_throw();
  for (int row = 0; row < rows; ++row) {
    for (int column : {0, 319, 639}) {
      close(static_cast<float>(projection_output[row * 640 + column]),
            1280.0f * static_cast<float>(projection_scales[column * 20]) *
                static_cast<float>(row + 1),
            0.15f, "GGML q4_0 indexer projection");
    }
  }
  for (void* pointer : {static_cast<void*>(qkv), static_cast<void*>(q),
                        static_cast<void*>(gate), static_cast<void*>(k),
                        static_cast<void*>(v), static_cast<void*>(weight),
                        static_cast<void*>(cache), static_cast<void*>(positions),
                        static_cast<void*>(index_input),
                        static_cast<void*>(index_output),
                        static_cast<void*>(index_eager),
                        static_cast<void*>(index_weight),
                        static_cast<void*>(projection_input),
                        static_cast<void*>(projection_weight),
                        static_cast<void*>(projection_scales),
                        static_cast<void*>(projection_output)}) {
    sycl::free(pointer, queue);
  }
  std::cout << "CPU SYCL QKV/indexer math passed\n";
}
