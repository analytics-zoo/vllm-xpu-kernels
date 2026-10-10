// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>

namespace vllm::qwen38::moe_sycl {

// Routed MoE INT4 uses signed, two's-complement nibbles. The optional router
// Q4_0 uses (unsigned nibble - 8) * scale instead. The physical intermediate
// is exactly 80 (TP8) or 160 (TP4); group size stays 128.
// Every pointer must be device-accessible, contiguous and remain alive until
// the current queue completes. The caller owns stream/allocator recording.
struct Inputs {
  const sycl::half* x;                // [M,2560]
  const sycl::half* logits;           // [M,512], null for router hostchain
  const std::uint8_t* router_weight;  // [512,1280], optional Q4_0
  const sycl::half* router_scale;     // [512,20], router hostchain only
  const std::uint8_t* w13;            // [512,2*I,1280]
  const sycl::half* s13;              // [512,2*I,20]
  const std::uint8_t* w2;             // [512,2560,I/2]
  const sycl::half* s2;               // [512,2560,ceil(I/128)]
  const sycl::half* shared_up;        // [2*I,2560]
  const sycl::half* shared_down;      // [2560,I]
  const sycl::half* shared_gate;      // [1,2560]
  sycl::half* output;                 // [M,2560], caller-owned
  int m;                              // 1..8
  int intermediate;                   // 80 or 160
};

struct Workspace {
  sycl::half* logits;   // [M,512], needed only for router hostchain
  std::int32_t* ids;    // [M,10]
  sycl::half* weights;  // [M,10]
  sycl::half* routed;   // [M*10,I]
  sycl::half* shared;   // [M,I]
  float* gates;         // [M]
};

// Returns false only before any enqueue for an unsupported contract. After
// the first enqueue, errors propagate. No allocation or host synchronization.
// All four stages use the supplied queue, which must be the current in-order
// XPU queue. The caller must validate tensor shape/dtype/alias and retain all
// owners (including this workspace) for async execution.
bool try_forward(
    sycl::queue& queue, const Inputs& input, const Workspace& workspace);

}  // namespace vllm::qwen38::moe_sycl
