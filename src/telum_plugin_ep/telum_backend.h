// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include <gsl/span>

#include "kernels/op_kernel.h"

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

struct TelumBackendConfig {
  std::string backend_kind = "stub";
  bool stub_support_mul = true;
};

// Optional fast-path function pointer for Mul. Allows the EP to avoid per-call virtual dispatch and
// redundant span size checks when shapes are already validated.
struct TelumMulTrustedFn {
  using Fn = OrtStatus* (*)(void* ctx,
                            const float* input0,
                            const float* input1,
                            float* output,
                            size_t num_elems) noexcept;

  Fn fn{};
  void* ctx{};

  constexpr explicit operator bool() const noexcept { return fn != nullptr; }
};

class TelumBackend {
 public:
  virtual ~TelumBackend() = default;

  virtual std::string BackendKind() const = 0;

  virtual bool SupportsMul() const noexcept = 0;
  virtual bool SupportsOp(telum::OpKind op_kind) const noexcept = 0;

  virtual OrtStatus* Mul(gsl::span<const float> input0,
                         gsl::span<const float> input1,
                         gsl::span<float> output) noexcept = 0;

  // Return a "trusted" Mul implementation that assumes inputs/outputs are already validated.
  // This is intended for perf-only fast paths in the plugin EP scaffold.
  virtual TelumMulTrustedFn GetMulTrustedFn() noexcept { return {}; }
};

using TelumBackendFactoryFn = std::unique_ptr<TelumBackend> (*)(const OrtApi& api,
                                                                const TelumBackendConfig& config);

void RegisterTelumBackendFactory(TelumBackendFactoryFn factory_fn);
std::unique_ptr<TelumBackend> CreateTelumBackend(const OrtApi& api, const TelumBackendConfig& config);
