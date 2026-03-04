// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "onnxruntime_c_api.h"

#include "../plugin_ep_utils.h"
#include "telum_backend.h"

class TelumEpFactory;

struct CustomMulKernel {
  CustomMulKernel(const OrtApi& ort_api,
                  const OrtLogger& logger,
                  TelumBackend& backend)
      : ort_api_(ort_api), logger_(logger), backend_(backend) {}

  OrtStatusPtr ComputeV2(OrtKernelContext* kernel_ctx) {
    try {
      Ort::KernelContext context{kernel_ctx};
      if (context.GetInputCount() != 2) {
        return Ort::Status("Custom_Mul expects exactly 2 inputs", ORT_INVALID_ARGUMENT).release();
      }

      Ort::ConstValue x = context.GetInput(0);
      Ort::ConstValue y = context.GetInput(1);

      auto x_info = x.GetTensorTypeAndShapeInfo();
      auto y_info = y.GetTensorTypeAndShapeInfo();
      if (x_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
          y_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        return Ort::Status("Custom_Mul supports float tensors only", ORT_INVALID_ARGUMENT).release();
      }

      const auto x_shape = x_info.GetShape();
      const auto y_shape = y_info.GetShape();
      if (x_shape != y_shape) {
        return Ort::Status("Custom_Mul requires equal input shapes", ORT_INVALID_ARGUMENT).release();
      }

      const size_t num_elements = x_info.GetElementCount();
      Ort::UnownedValue out = context.GetOutput(0, x_shape);

      const float* x_data = x.GetTensorData<float>();
      const float* y_data = y.GetTensorData<float>();
      float* out_data = out.GetTensorMutableData<float>();

      OrtStatus* st = backend_.Mul(gsl::span<const float>(x_data, num_elements),
                                   gsl::span<const float>(y_data, num_elements),
                                   gsl::span<float>(out_data, num_elements));
      if (st == nullptr) {
        return nullptr;
      }

      Ort::Status backend_status{st};
      // CPU fallback for custom op if backend path is disabled/unavailable.
      for (size_t i = 0; i < num_elements; ++i) {
        out_data[i] = x_data[i] * y_data[i];
      }

      IGNORE_ORTSTATUS(ort_api_.Logger_LogMessage(
          &logger_, ORT_LOGGING_LEVEL_WARNING,
          ("Custom_Mul backend fallback: " + std::string(backend_status.GetErrorMessage())).c_str(),
          ORT_FILE, __LINE__, __FUNCTION__));

      return nullptr;

    } catch (const Ort::Exception& ex) {
      Ort::Status status(ex);
      return status.release();
    } catch (const std::exception& ex) {
      return Ort::Status(ex.what(), ORT_EP_FAIL).release();
    }
  }

 private:
  const OrtApi& ort_api_;
  const OrtLogger& logger_;
  TelumBackend& backend_;
};

struct TelumEpCustomOp : Ort::CustomOpBase<TelumEpCustomOp, CustomMulKernel, /*WithStatus*/ true> {
  explicit TelumEpCustomOp(const char* provider, TelumEpFactory* factory)
      : provider_(provider), factory_(factory) {}

  OrtStatusPtr CreateKernelV2(const OrtApi& api, const OrtKernelInfo* info, void** op_kernel) const;
  OrtStatusPtr KernelComputeV2(void* op_kernel, OrtKernelContext* context) const;

  const char* GetName() const { return name_; }
  void SetName(const char* name) { name_ = name; }

  const char* GetExecutionProviderType() const { return provider_; }

  size_t GetInputTypeCount() const { return num_inputs_; }
  void SetInputTypeCount(size_t num) { num_inputs_ = num; }

  ONNXTensorElementDataType GetInputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }

  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t) const {
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC;
  }

  size_t GetOutputTypeCount() const { return num_outputs_; }
  void SetOutputTypeCount(size_t num) { num_outputs_ = num; }

  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }

  OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(size_t) const {
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC;
  }

  bool GetVariadicInputHomogeneity() const { return false; }
  bool GetVariadicOutputHomogeneity() const { return false; }

 private:
  const char* provider_ = nullptr;
  const char* name_ = nullptr;
  size_t num_inputs_ = 1;
  size_t num_outputs_ = 1;
  TelumEpFactory* factory_ = nullptr;
};
