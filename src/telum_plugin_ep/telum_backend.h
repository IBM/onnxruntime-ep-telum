// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include <gsl/span>

#include "kernels/op_kernel.h"

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

struct TelumBackendConfig {
  std::string backend_kind = "zdnn";
};

struct TelumMatMulRequest {
  gsl::span<const float> input_a;
  gsl::span<const float> input_b;
  gsl::span<float> output;
  gsl::span<const int64_t> a_shape;
  gsl::span<const int64_t> b_shape;
  gsl::span<const int64_t> output_shape;
};

struct TelumBinaryRequest {
  telum::OpKind op_kind{telum::OpKind::kUnknown};
  gsl::span<const float> input_a;
  gsl::span<const float> input_b;
  gsl::span<float> output;
};

struct TelumUnaryRequest {
  telum::OpKind op_kind{telum::OpKind::kUnknown};
  gsl::span<const float> input;
  gsl::span<float> output;
};

struct TelumSoftmaxRequest {
  gsl::span<const float> input;
  gsl::span<float> output;
  gsl::span<const int64_t> input_shape;
  int64_t axis{-1};
};

struct TelumLayerNormRequest {
  gsl::span<const float> input;
  gsl::span<const float> scale;
  gsl::span<const float> bias;
  gsl::span<float> output;
  gsl::span<const int64_t> input_shape;
  gsl::span<const int64_t> scale_shape;
  gsl::span<const int64_t> bias_shape;
  bool has_bias{false};
  int64_t axis{-1};
  float epsilon{1e-5f};
};

struct TelumGemmRequest {
  gsl::span<const float> input_a;
  gsl::span<const float> input_b;
  gsl::span<const float> input_c;
  gsl::span<float> output;
  gsl::span<const int64_t> a_shape;
  gsl::span<const int64_t> b_shape;
  gsl::span<const int64_t> c_shape;
  gsl::span<const int64_t> output_shape;
  bool has_c{false};
  bool trans_a{false};
  bool trans_b{false};
  float alpha{1.0f};
  float beta{1.0f};
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
  virtual bool IsRuntimeReady() const noexcept = 0;
  virtual std::string RuntimeStatusMessage() const = 0;
  virtual uint32_t MaxDimIdxSize() const noexcept = 0;

  virtual bool SupportsMul() const noexcept = 0;
  virtual bool SupportsOp(telum::OpKind op_kind) const noexcept = 0;

  virtual OrtStatus* Mul(gsl::span<const float> input0,
                         gsl::span<const float> input1,
                         gsl::span<float> output) noexcept = 0;

  virtual OrtStatus* Binary(const TelumBinaryRequest& request) noexcept = 0;
  virtual OrtStatus* Unary(const TelumUnaryRequest& request) noexcept = 0;
  virtual OrtStatus* Softmax(const TelumSoftmaxRequest& request) noexcept = 0;
  virtual OrtStatus* LayerNormalization(const TelumLayerNormRequest& request) noexcept = 0;
  virtual OrtStatus* MatMul(const TelumMatMulRequest& request) noexcept = 0;
  virtual OrtStatus* Gemm(const TelumGemmRequest& request) noexcept = 0;

  // Return a "trusted" Mul implementation that assumes inputs/outputs are already validated.
  // This is intended for perf-only fast paths in the plugin EP scaffold.
  virtual TelumMulTrustedFn GetMulTrustedFn() noexcept { return {}; }
};

using TelumBackendFactoryFn = std::unique_ptr<TelumBackend> (*)(const OrtApi& api,
                                                                const TelumBackendConfig& config);

void RegisterTelumBackendFactory(TelumBackendFactoryFn factory_fn);
std::unique_ptr<TelumBackend> CreateTelumBackend(const OrtApi& api, const TelumBackendConfig& config);
