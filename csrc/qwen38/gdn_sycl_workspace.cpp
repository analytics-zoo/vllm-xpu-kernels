// SPDX-License-Identifier: Apache-2.0
#include "qwen38/gdn_sycl_workspace.h"

#include <c10/core/DeviceGuard.h>
#include <c10/core/GradMode.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>

#include <cmath>
#include <limits>

#include "qwen38/gdn_sycl.h"
#include "qwen38/gdn_sycl_projection.h"
#include "qwen38/ops.h"

namespace vllm::qwen38 {
namespace {

bool plain(const at::Tensor& t, at::Device device, at::ScalarType dtype) {
  return t.defined() && t.device() == device && t.layout() == at::kStrided &&
         t.scalar_type() == dtype && !t.is_conj() && !t.is_neg();
}

bool dense(
    const at::Tensor& t,
    at::Device device,
    at::ScalarType dtype,
    at::IntArrayRef shape) {
  return plain(t, device, dtype) && t.sizes() == shape && t.is_contiguous();
}

bool convertible(
    const at::Tensor& t, at::Device device, at::IntArrayRef shape) {
  return t.defined() &&
         (t.scalar_type() == at::kHalf || t.scalar_type() == at::kBFloat16 ||
          t.scalar_type() == at::kFloat) &&
         dense(t, device, t.scalar_type(), shape);
}

at::Tensor half_parameter(const at::Tensor& t) {
  return t.detach().to(at::kHalf).contiguous();
}

}  // namespace

std::optional<at::Tensor> GdnSyclM1Workspace::try_run(
    at::Tensor input,
    const std::vector<at::Tensor>& w,
    const std::optional<at::Tensor>& bias,
    at::Tensor conv,
    at::Tensor ssm,
    at::Tensor indices,
    int64_t hv,
    double eps,
    bool sigmoid_gate) {
  return try_run_impl(
      input,
      w,
      bias,
      conv,
      ssm,
      indices,
      std::nullopt,
      std::nullopt,
      hv,
      eps,
      sigmoid_gate);
}

std::optional<at::Tensor> GdnSyclSpecWorkspace::try_run(
    at::Tensor input,
    const std::vector<at::Tensor>& w,
    const std::optional<at::Tensor>& bias,
    at::Tensor conv,
    at::Tensor ssm,
    at::Tensor indices,
    at::Tensor token_indices,
    at::Tensor accepted,
    int64_t hv,
    double eps,
    bool sigmoid_gate) {
  return workspace_.try_run_impl(
      input,
      w,
      bias,
      conv,
      ssm,
      indices,
      token_indices,
      accepted,
      hv,
      eps,
      sigmoid_gate);
}

std::optional<at::Tensor> GdnSyclM1Workspace::try_run_impl(
    at::Tensor input,
    const std::vector<at::Tensor>& w,
    const std::optional<at::Tensor>& bias,
    at::Tensor conv,
    at::Tensor ssm,
    at::Tensor indices,
    const std::optional<at::Tensor>& token_indices,
    const std::optional<at::Tensor>& accepted,
    int64_t hv,
    double eps,
    bool sigmoid_gate) {
  if (at::GradMode::is_enabled() || !input.defined() || !input.is_xpu() ||
      input.dim() != 2 || w.size() != 10 || (hv != 6 && hv != 12))
    return std::nullopt;
  const bool spec = token_indices.has_value();
  const int64_t m = input.size(0);
  if (spec != accepted.has_value() || (spec ? (m < 2 || m > 8) : m != 1))
    return std::nullopt;
  const auto device = input.device();
  const int64_t h = hv / 3, dim = (2 * h + hv) * 128;
  const int64_t qkvz_width = dim + hv * 128;
  const float epsilon = static_cast<float>(eps);
  if (!std::isfinite(eps) || !std::isfinite(epsilon) || epsilon <= 0.0f ||
      !dense(input, device, at::kHalf, {m, 2560}) ||
      !dense(w[0], device, at::kByte, {qkvz_width, 1280}) ||
      !dense(w[1], device, at::kHalf, {qkvz_width, 20}) ||
      !dense(w[2], device, at::kByte, {2 * hv, 1280}) ||
      !dense(w[3], device, at::kHalf, {2 * hv, 20}) ||
      !dense(w[4], device, at::kByte, {2560, hv * 64}) ||
      !dense(w[5], device, at::kHalf, {2560, hv}) ||
      !convertible(w[6], device, {dim, 1, 4}) ||
      !convertible(w[7], device, {hv}) || !convertible(w[8], device, {hv}) ||
      !convertible(w[9], device, {128}) ||
      (bias.has_value() && !convertible(*bias, device, {dim})) ||
      !plain(indices, device, at::kInt) || !indices.is_contiguous() ||
      (spec ? !((indices.dim() == 1 && indices.size(0) >= m) ||
                (indices.dim() == 2 && indices.size(0) >= 1 &&
                 indices.size(1) == m))
            : indices.sizes() != at::IntArrayRef({1})) ||
      (spec &&
       (!plain(*token_indices, device, at::kInt) || token_indices->dim() != 1 ||
        !token_indices->is_contiguous() || token_indices->size(0) < m ||
        !plain(*accepted, device, at::kInt) || accepted->dim() != 1 ||
        !accepted->is_contiguous() || accepted->size(0) < 1)) ||
      !plain(conv, device, at::kHalf) || !plain(ssm, device, at::kHalf) ||
      conv.dim() != 3 || conv.size(0) <= 0 ||
      conv.size(0) > std::numeric_limits<int>::max() ||
      conv.size(1) > std::numeric_limits<int>::max() ||
      (spec ? !(conv.size(1) == 3 || conv.size(1) >= m + 2)
            : conv.size(1) != 3) ||
      conv.size(2) != dim || conv.stride(2) != 1 || conv.stride(1) != dim ||
      conv.stride(0) < conv.size(1) * dim || ssm.dim() != 4 ||
      ssm.size(0) <= 0 || ssm.size(0) > std::numeric_limits<int>::max() ||
      ssm.size(1) != hv || ssm.size(2) != 128 || ssm.size(3) != 128 ||
      ssm.stride(3) != 1 || ssm.stride(2) != 128 ||
      ssm.stride(1) != 128 * 128 || ssm.stride(0) < hv * 128 * 128)
    return std::nullopt;

  // Raw sources must not be overwritten even when the derived FP16 copy
  // happens to be disjoint. This preserves the Python guard's live contract.
  std::vector<std::optional<at::Tensor>> reads(w.begin(), w.end());
  reads.emplace_back(input);
  reads.emplace_back(indices);
  reads.emplace_back(bias);
  reads.emplace_back(token_indices);
  reads.emplace_back(accepted);
  if (!tensors_disjoint_host({conv, ssm}, reads)) return std::nullopt;

  const c10::OptionalDeviceGuard device_guard(device);
  auto stream = c10::xpu::getCurrentXPUStream(device.index());
  if (!stream.queue().is_in_order()) return std::nullopt;
  std::lock_guard<std::mutex> lock(mutex_);
  auto& s = by_stream_[stream.unwrap()];
  if (!s.output.defined() || s.hv != hv || s.m != m) {
    // An allocation failure must not publish a partially built workspace.
    // In particular, output alone is not a witness that normalized exists.
    Scratch fresh;
    fresh.hv = hv;
    fresh.m = m;
    const auto options = input.options().requires_grad(false);
    fresh.qkvz = at::empty({m, qkvz_width}, options);
    fresh.ba = at::empty({m, 2 * hv}, options);
    fresh.core = at::empty({m, hv, 128}, options);
    fresh.z = at::empty({m, hv, 128}, options);
    fresh.core_rows = fresh.core.view({m * hv, 128});
    fresh.z_rows = fresh.z.view({m * hv, 128});
    fresh.output = at::empty({m, 2560}, options);
    if (spec) fresh.normalized = at::empty({m, hv * 128}, options);
    s = std::move(fresh);
  }
  std::vector<at::Tensor> writes = {
      conv, ssm, s.qkvz, s.ba, s.core, s.z, s.output};
  if (spec) writes.push_back(s.normalized);
  if (!tensors_disjoint_host(writes, reads)) return std::nullopt;

  std::vector<at::Tensor> sources = {w[6], w[7], w[8], w[9]};
  if (bias.has_value()) sources.push_back(*bias);
  const bool refresh = !s.binding || !s.binding->matches(sources);
  if (!refresh) {
    // Check cached derived buffers too, before the first device submission.
    auto derived_reads = reads;
    for (const auto* t :
         {&s.conv_weight, &s.conv_bias, &s.a_log, &s.dt_bias, &s.norm_weight})
      derived_reads.emplace_back(*t);
    if (!tensors_disjoint_host(writes, derived_reads)) return std::nullopt;
  }

  // First possible submission below (parameter conversion/zeroing). Never
  // catch/replay anything after this boundary, including partial failures.
  if (refresh) {
    // A failed conversion must not leave the old witness attached to a
    // partially refreshed set of derived tensors. Any later reuse rebuilds
    // the whole set, even if the caller has restored the previous weights.
    s.binding.reset();
    auto binding = std::make_unique<TensorBindingSnapshotV1>(sources);
    s.conv_weight = half_parameter(w[6]).view({dim, 4});
    s.conv_bias = bias.has_value()
                      ? half_parameter(*bias)
                      : at::zeros({dim}, input.options().requires_grad(false));
    s.a_log =
        spec ? w[7].detach().to(at::kFloat).contiguous() : half_parameter(w[7]);
    s.dt_bias = half_parameter(w[8]);
    s.norm_weight = half_parameter(w[9]);
    s.binding = std::move(binding);
  }
  if (spec) {
    auto state_ids = indices.flatten().narrow(0, 0, m);
    auto token_ids = token_indices->narrow(0, 0, m);
    auto accepted_count = accepted->narrow(0, 0, 1);
    int4_linear(input, w[0], w[1], s.qkvz);
    int4_linear(input, w[2], w[3], s.ba);
    gdn_spec_v2_sycl(
        s.qkvz,
        conv,
        s.conv_weight,
        s.conv_bias,
        state_ids,
        s.a_log,
        s.dt_bias,
        s.ba,
        ssm,
        s.core,
        s.z,
        token_ids,
        accepted_count,
        1,
        m,
        1.0 / std::sqrt(128.0));
    gdn_norm_gate_sycl(
        s.core, s.z, s.norm_weight, s.normalized, eps, sigmoid_gate);
    int4_linear(s.normalized, w[4], w[5], s.output);
    return s.output.view({m, 2560});
  }
  int4_linear_fused2(input, w[0], w[1], s.qkvz, w[2], w[3], s.ba);
  gdn_decode_sycl(
      s.qkvz,
      conv,
      s.conv_weight,
      s.conv_bias,
      indices,
      s.a_log,
      s.dt_bias,
      s.ba,
      ssm,
      indices,
      s.core,
      s.z,
      1.0 / std::sqrt(128.0));
  gdn_norm_int4_sycl(
      s.core_rows,
      s.z_rows,
      s.norm_weight,
      w[4],
      w[5],
      s.output,
      hv,
      128,
      eps,
      sigmoid_gate);
  // A fresh TensorImpl prevents consumer metadata edits from changing the
  // private scratch layout; backing storage follows the current stream.
  return s.output.view({1, 2560});
}

void GdnSyclM1Workspace::invalidate() {
  std::lock_guard<std::mutex> lock(mutex_);
  by_stream_.clear();
}

}  // namespace vllm::qwen38
