// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <ATen/core/dispatch/Dispatcher.h>
#include <optional>

namespace vllm::decode {

// Optional extensions are loaded by the package before kernels are called.
// Keep the original implementation available when an extension is absent.
template <typename Signature>
std::optional<c10::TypedOperatorHandle<Signature>>
optional_operator(const char* name, const char* overload = "") {
  const auto schema = c10::Dispatcher::singleton().findSchema({name, overload});
  if (!schema) return std::nullopt;
  return schema->typed<Signature>();
}

}  // namespace vllm::decode
