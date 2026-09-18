#include "upstream_weightless.hpp"

#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include "core/registration.h"

namespace vllm_weightless_rms {

at::Tensor rms_norm_no_weight(at::Tensor input, double epsilon) {
  TORCH_CHECK(input.is_xpu() && input.scalar_type() == at::kHalf,
              "Expected FP16 XPU input");
  TORCH_CHECK(input.dim() >= 2 && input.dim() <= 4,
              "Expected rank 2, 3, or 4 input");
  const int64_t width = input.size(-1);
  TORCH_CHECK(width == 256 || width == 512 || width == 2816,
              "Expected hidden width 256, 512, or 2816");
  const int64_t rows = input.numel() / width;
  TORCH_CHECK(input.size(0) >= 1 && input.size(0) <= 8 && rows >= 1 && rows <= 64,
              "Expected batch 1..8 and total normalized rows 1..64");
  TORCH_CHECK(input.stride(-1) == 1,
              "Expected unit final stride; outer dimensions may be strided");
  TORCH_CHECK(epsilon >= 0, "Expected a nonnegative epsilon");
  const c10::DeviceGuard guard(input.device());
  auto output = at::empty(input.sizes(), input.options());
  call_rms_norm_kernel<at::Half>(output, input, float(epsilon));
  return output;
}

}  // namespace vllm_weightless_rms

TORCH_LIBRARY_FRAGMENT(_xpu_C, m) {
  m.def("rms_norm_no_weight(Tensor input, float epsilon) -> Tensor");
  m.impl("rms_norm_no_weight", torch::kXPU,
         &vllm_weightless_rms::rms_norm_no_weight);
  m.impl("rms_norm_no_weight", torch::kMeta,
         [](at::Tensor input, double epsilon) {
           return at::empty_symint(input.sym_sizes(), input.options());
         });
}

REGISTER_EXTENSION(_rms_C)
