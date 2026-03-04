// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../plugin_ep_utils.h"

class TelumBackend;

namespace telum {

enum class OpKind : uint8_t {
  kUnknown = 0,
  kMatMul,
  kGemm,
  kAdd,
  kSub,
  kMul,
  kDiv,
  kMin,
  kMax,
  kRelu,
  kGelu,
  kTanh,
  kSigmoid,
  kExp,
  kLog,
  kSqrt,
  kSoftmax,
  kLayerNormalization,
  kReshape,
  kTranspose,
  kSqueeze,
  kUnsqueeze,
  kReduceMean,
  kCast,
  kWhere,
  kExpand,
  kConcat,
  kGather,
  kSlice,
};

struct TensorInitializer {
  std::vector<int64_t> shape;
  ONNXTensorElementDataType elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  std::vector<uint8_t> raw_data;

  size_t ElementCount() const noexcept;
  size_t ElementSize() const noexcept;

  template <typename T>
  const T* Data() const noexcept {
    return reinterpret_cast<const T*>(raw_data.data());
  }
};

using TensorInitializerMap = std::unordered_map<std::string, TensorInitializer>;

struct KernelConfig {
  bool log_fallbacks = false;
  bool strict_mode = false;
};

class CompiledNodeKernel {
 public:
  virtual ~CompiledNodeKernel() = default;

  virtual OrtStatus* Compute(OrtKernelContext* kernel_ctx) noexcept = 0;
  virtual const std::string& OpType() const noexcept = 0;
  virtual OpKind GetOpKind() const noexcept = 0;
  virtual const std::vector<std::string>& InputNames() const noexcept = 0;
  virtual std::string SerializeAttributes() const = 0;
};

bool TryGetOpKind(const std::string& op_type, const std::string& domain, OpKind& op_kind);
std::string OpKindToString(OpKind op_kind);

bool OpUsesNnpaGating(OpKind op_kind);

OrtStatus* ConvertConstValueToInitializer(const OrtApi& ort_api,
                                          Ort::ConstValue value,
                                          TensorInitializer& initializer);

std::unique_ptr<CompiledNodeKernel> CreateCompiledNodeKernel(const OrtApi& ort_api,
                                                             const OrtLogger& logger,
                                                             TelumBackend& backend,
                                                             const KernelConfig& kernel_config,
                                                             const Ort::ConstNode& node,
                                                             const TensorInitializerMap& initializers,
                                                             bool drop_constant_initializers,
                                                             OrtStatus*& error_status);

std::unique_ptr<CompiledNodeKernel> CreateCompiledNodeKernelFromEpContext(const OrtApi& ort_api,
                                                                          const OrtLogger& logger,
                                                                          TelumBackend& backend,
                                                                          const KernelConfig& kernel_config,
                                                                          const std::string& op_type,
                                                                          const std::string& attributes_blob,
                                                                          const std::vector<std::string>& input_names,
                                                                          const TensorInitializerMap& initializers,
                                                                          bool drop_constant_initializers,
                                                                          OrtStatus*& error_status);

}  // namespace telum
