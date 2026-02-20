// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "op_kernel.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../telum_backend.h"

namespace telum {
namespace {

size_t GetElementSize(ONNXTensorElementDataType elem_type) {
  switch (elem_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return sizeof(bool);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return sizeof(float);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return sizeof(int32_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return sizeof(int64_t);
    default:
      return 0;
  }
}

OrtStatus* MakeStatus(const OrtApi& ort_api, OrtErrorCode code, const std::string& msg) {
  return ort_api.CreateStatus(code, msg.c_str());
}

std::optional<size_t> TryComputeElementCount(gsl::span<const int64_t> shape) {
  size_t count = 1;
  for (int64_t dim : shape) {
    if (dim < 0) {
      return std::nullopt;
    }

    const size_t dim_size = static_cast<size_t>(dim);
    if (dim_size == 0) {
      return static_cast<size_t>(0);
    }

    if (count > std::numeric_limits<size_t>::max() / dim_size) {
      return std::nullopt;
    }

    count *= dim_size;
  }

  return count;
}

int64_t NormalizeAxis(int64_t axis, size_t rank, bool allow_end = false) {
  const int64_t rank_i64 = static_cast<int64_t>(rank);
  if (axis < 0) {
    axis += rank_i64;
  }

  const int64_t max_axis = allow_end ? rank_i64 : (rank_i64 - 1);
  if (axis < 0 || axis > max_axis) {
    throw Ort::Exception("axis out of range", ORT_INVALID_ARGUMENT);
  }

  return axis;
}

bool IsStaticShape(gsl::span<const int64_t> shape) {
  return std::all_of(shape.begin(), shape.end(), [](int64_t d) { return d >= 0; });
}

std::vector<int64_t> ComputeContiguousStrides(gsl::span<const int64_t> dims) {
  std::vector<int64_t> strides(dims.size(), 1);
  for (size_t i = dims.size(); i-- > 0;) {
    if (i + 1 < dims.size()) {
      strides[i] = strides[i + 1] * dims[i + 1];
    }
  }

  return strides;
}

void AlignDimsToRank(gsl::span<const int64_t> dims, size_t out_rank, std::vector<int64_t>& aligned_dims) {
  aligned_dims.assign(out_rank, 1);
  const size_t offset = out_rank - dims.size();
  for (size_t i = 0; i < dims.size(); ++i) {
    aligned_dims[offset + i] = dims[i];
  }
}

bool ComputeBroadcastShape(const std::vector<std::vector<int64_t>>& input_shapes,
                           std::vector<int64_t>& output_shape,
                           std::string& error) {
  output_shape.clear();
  error.clear();

  size_t out_rank = 0;
  for (const auto& shape : input_shapes) {
    out_rank = std::max(out_rank, shape.size());
  }

  output_shape.assign(out_rank, 1);

  for (size_t axis = 0; axis < out_rank; ++axis) {
    int64_t dim = 1;
    for (const auto& shape : input_shapes) {
      const size_t shift = out_rank - shape.size();
      const int64_t cur = axis < shift ? 1 : shape[axis - shift];
      if (cur < 0) {
        error = "broadcast shape has dynamic dimension";
        return false;
      }

      if (cur == 1) {
        continue;
      }

      if (dim == 1 || dim == cur) {
        dim = cur;
      } else {
        error = "incompatible broadcast dimensions";
        return false;
      }
    }

    output_shape[axis] = dim;
  }

  return true;
}

struct TensorArg {
  bool present = false;
  ONNXTensorElementDataType elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  std::vector<int64_t> shape;
  size_t element_count = 0;
  const void* data = nullptr;
};

bool IsTensorType(ONNXTensorElementDataType type) {
  return type != ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
}

template <typename T>
const T* GetTensorData(const TensorArg& tensor, ONNXTensorElementDataType expected) {
  if (!tensor.present || tensor.elem_type != expected || tensor.data == nullptr) {
    return nullptr;
  }

  return reinterpret_cast<const T*>(tensor.data);
}

bool ReadIntVector(const TensorArg& tensor, std::vector<int64_t>& values) {
  values.clear();
  if (!tensor.present) {
    return false;
  }

  if (tensor.elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    const int64_t* data = reinterpret_cast<const int64_t*>(tensor.data);
    values.assign(data, data + tensor.element_count);
    return true;
  }

  if (tensor.elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    const int32_t* data = reinterpret_cast<const int32_t*>(tensor.data);
    values.reserve(tensor.element_count);
    for (size_t i = 0; i < tensor.element_count; ++i) {
      values.push_back(static_cast<int64_t>(data[i]));
    }
    return true;
  }

  return false;
}

OrtStatus* ValidateStaticTensor(const OrtApi& ort_api, const TensorArg& tensor, const std::string& input_name) {
  if (!tensor.present) {
    return MakeStatus(ort_api, ORT_INVALID_ARGUMENT, "Missing required input '" + input_name + "'");
  }

  if (!IsTensorType(tensor.elem_type)) {
    return MakeStatus(ort_api, ORT_INVALID_ARGUMENT, "Input '" + input_name + "' is not a tensor");
  }

  if (!IsStaticShape(tensor.shape)) {
    return MakeStatus(ort_api, ORT_INVALID_ARGUMENT,
                      "Input '" + input_name + "' has dynamic shape; Telum plugin requires static shapes");
  }

  return nullptr;
}

bool ParseBoolAttr(const Ort::ConstNode& node, const char* attr_name, bool default_value) {
  Ort::ConstOpAttr attr;
  if (!node.GetAttributeByName(attr_name, attr).IsOK()) {
    return default_value;
  }

  int64_t raw = 0;
  if (!attr.GetValue(raw).IsOK()) {
    return default_value;
  }

  return raw != 0;
}

template <typename T>
bool TryParseScalarAttr(const Ort::ConstNode& node, const char* attr_name, T& value) {
  Ort::ConstOpAttr attr;
  if (!node.GetAttributeByName(attr_name, attr).IsOK()) {
    return false;
  }

  return attr.GetValue(value).IsOK();
}

bool TryParseIntVectorAttr(const Ort::ConstNode& node, const char* attr_name, std::vector<int64_t>& values) {
  Ort::ConstOpAttr attr;
  if (!node.GetAttributeByName(attr_name, attr).IsOK()) {
    return false;
  }

  return attr.GetValueArray(values).IsOK();
}

std::string JoinInt64Pipe(const std::vector<int64_t>& values) {
  std::ostringstream os;
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      os << "|";
    }
    os << values[i];
  }
  return os.str();
}

bool ParseInt64Pipe(const std::string& value, std::vector<int64_t>& values) {
  values.clear();
  if (value.empty()) {
    return true;
  }

  std::stringstream ss(value);
  std::string token;
  while (std::getline(ss, token, '|')) {
    if (token.empty()) {
      return false;
    }
    try {
      size_t parsed = 0;
      const long long num = std::stoll(token, &parsed, 10);
      if (parsed != token.size()) {
        return false;
      }
      values.push_back(static_cast<int64_t>(num));
    } catch (...) {
      return false;
    }
  }
  return true;
}

struct KernelAttributes {
  float alpha = 1.0f;   // Gemm
  float beta = 1.0f;    // Gemm
  int64_t trans_a = 0;  // Gemm
  int64_t trans_b = 0;  // Gemm

  int64_t axis = -1;  // Softmax/Gather/LayerNorm
  float epsilon = 1e-5f;

  std::vector<int64_t> perm;
  bool has_perm = false;

  std::vector<int64_t> axes;
  bool has_axes = false;

  bool keepdims = true;  // ReduceMean
  bool noop_with_empty_axes = false;

  int64_t cast_to = static_cast<int64_t>(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

  int64_t concat_axis = 0;

  bool allow_zero = false;  // Reshape
};

std::string SerializeKernelAttributesBlob(const KernelAttributes& attrs) {
  std::ostringstream os;
  os << "alpha=" << attrs.alpha
     << ",beta=" << attrs.beta
     << ",trans_a=" << attrs.trans_a
     << ",trans_b=" << attrs.trans_b
     << ",axis=" << attrs.axis
     << ",epsilon=" << attrs.epsilon
     << ",perm=" << JoinInt64Pipe(attrs.perm)
     << ",has_perm=" << (attrs.has_perm ? 1 : 0)
     << ",axes=" << JoinInt64Pipe(attrs.axes)
     << ",has_axes=" << (attrs.has_axes ? 1 : 0)
     << ",keepdims=" << (attrs.keepdims ? 1 : 0)
     << ",noop_with_empty_axes=" << (attrs.noop_with_empty_axes ? 1 : 0)
     << ",cast_to=" << attrs.cast_to
     << ",concat_axis=" << attrs.concat_axis
     << ",allow_zero=" << (attrs.allow_zero ? 1 : 0);
  return os.str();
}

bool ParseKernelAttributesBlob(const std::string& blob, KernelAttributes& attrs) {
  attrs = KernelAttributes{};
  if (blob.empty()) {
    return true;
  }

  std::stringstream ss(blob);
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (token.empty()) {
      continue;
    }

    const size_t eq = token.find('=');
    if (eq == std::string::npos || eq == 0) {
      return false;
    }
    const std::string key = token.substr(0, eq);
    const std::string value = token.substr(eq + 1);

    try {
      if (key == "alpha") attrs.alpha = std::stof(value);
      else if (key == "beta") attrs.beta = std::stof(value);
      else if (key == "trans_a") attrs.trans_a = std::stoll(value);
      else if (key == "trans_b") attrs.trans_b = std::stoll(value);
      else if (key == "axis") attrs.axis = std::stoll(value);
      else if (key == "epsilon") attrs.epsilon = std::stof(value);
      else if (key == "perm") { if (!ParseInt64Pipe(value, attrs.perm)) return false; }
      else if (key == "has_perm") attrs.has_perm = (value == "1");
      else if (key == "axes") { if (!ParseInt64Pipe(value, attrs.axes)) return false; }
      else if (key == "has_axes") attrs.has_axes = (value == "1");
      else if (key == "keepdims") attrs.keepdims = (value == "1");
      else if (key == "noop_with_empty_axes") attrs.noop_with_empty_axes = (value == "1");
      else if (key == "cast_to") attrs.cast_to = std::stoll(value);
      else if (key == "concat_axis") attrs.concat_axis = std::stoll(value);
      else if (key == "allow_zero") attrs.allow_zero = (value == "1");
    } catch (...) {
      return false;
    }
  }

  return true;
}

class GenericNodeKernel final : public CompiledNodeKernel {
 public:
  GenericNodeKernel(const OrtApi& ort_api,
                    const OrtLogger& logger,
                    TelumBackend& backend,
                    KernelConfig kernel_config,
                    OpKind op_kind,
                    std::string op_type,
                    std::vector<std::string> input_names,
                    KernelAttributes attributes,
                    const TensorInitializerMap& initializers,
                    bool drop_constant_initializers,
                    int since_version)
      : ort_api_(ort_api),
        logger_(logger),
        backend_(backend),
        kernel_config_(kernel_config),
        op_kind_(op_kind),
        op_type_(std::move(op_type)),
        input_names_(std::move(input_names)),
        attributes_(std::move(attributes)),
        initializers_(initializers),
        drop_constant_initializers_(drop_constant_initializers),
        since_version_(since_version) {}

  OrtStatus* Compute(OrtKernelContext* kernel_ctx) noexcept override {
    try {
      Ort::KernelContext context{kernel_ctx};

      std::vector<TensorArg> inputs;
      RETURN_IF_ERROR(ResolveInputs(context, inputs));

      switch (op_kind_) {
        case OpKind::kMatMul:
          return ComputeMatMul(context, inputs);
        case OpKind::kGemm:
          return ComputeGemm(context, inputs);
        case OpKind::kAdd:
        case OpKind::kSub:
        case OpKind::kMul:
        case OpKind::kDiv:
        case OpKind::kMin:
        case OpKind::kMax:
          return ComputeBinaryElementwise(context, inputs);
        case OpKind::kRelu:
        case OpKind::kGelu:
        case OpKind::kTanh:
        case OpKind::kSigmoid:
        case OpKind::kExp:
        case OpKind::kLog:
        case OpKind::kSqrt:
          return ComputeUnary(context, inputs);
        case OpKind::kSoftmax:
          return ComputeSoftmax(context, inputs);
        case OpKind::kLayerNormalization:
          return ComputeLayerNormalization(context, inputs);
        case OpKind::kReshape:
          return ComputeReshape(context, inputs);
        case OpKind::kTranspose:
          return ComputeTranspose(context, inputs);
        case OpKind::kSqueeze:
          return ComputeSqueeze(context, inputs);
        case OpKind::kUnsqueeze:
          return ComputeUnsqueeze(context, inputs);
        case OpKind::kReduceMean:
          return ComputeReduceMean(context, inputs);
        case OpKind::kCast:
          return ComputeCast(context, inputs);
        case OpKind::kWhere:
          return ComputeWhere(context, inputs);
        case OpKind::kExpand:
          return ComputeExpand(context, inputs);
        case OpKind::kConcat:
          return ComputeConcat(context, inputs);
        case OpKind::kGather:
          return ComputeGather(context, inputs);
        case OpKind::kSlice:
          return ComputeSlice(context, inputs);
        default:
          return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Unsupported op in Telum kernel: " + op_type_);
      }
    } catch (const Ort::Exception& ex) {
      Ort::Status status(ex);
      return status.release();
    } catch (const std::exception& ex) {
      return MakeStatus(ort_api_, ORT_EP_FAIL,
                        std::string("Telum kernel exception for ") + op_type_ + ": " + ex.what());
    }
  }

  const std::string& OpType() const noexcept override {
    return op_type_;
  }

  OpKind GetOpKind() const noexcept override {
    return op_kind_;
  }

  const std::vector<std::string>& InputNames() const noexcept override {
    return input_names_;
  }

  std::string SerializeAttributes() const override {
    return SerializeKernelAttributesBlob(attributes_);
  }

 private:
  OrtStatus* ResolveInputs(Ort::KernelContext& context, std::vector<TensorArg>& inputs) const {
    inputs.clear();
    inputs.reserve(input_names_.size());

    size_t runtime_input_index = 0;
    const size_t runtime_input_count = context.GetInputCount();

    for (const auto& input_name : input_names_) {
      TensorArg arg{};

      if (input_name.empty()) {
        inputs.push_back(std::move(arg));
        continue;
      }

      auto it = initializers_.find(input_name);
      const bool has_initializer = it != initializers_.end();
      if (drop_constant_initializers_ && has_initializer) {
        const TensorInitializer& initializer = it->second;
        arg.present = true;
        arg.elem_type = initializer.elem_type;
        arg.shape = initializer.shape;
        arg.element_count = initializer.ElementCount();
        arg.data = initializer.raw_data.empty() ? nullptr : initializer.raw_data.data();
        inputs.push_back(std::move(arg));
        continue;
      }

      if (runtime_input_index < runtime_input_count) {
        Ort::ConstValue input = context.GetInput(runtime_input_index++);
        auto type_shape = input.GetTensorTypeAndShapeInfo();
        arg.present = true;
        arg.elem_type = type_shape.GetElementType();
        arg.shape = type_shape.GetShape();
        arg.element_count = type_shape.GetElementCount();

        switch (arg.elem_type) {
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            arg.data = input.GetTensorData<float>();
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            arg.data = input.GetTensorData<bool>();
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            arg.data = input.GetTensorData<int32_t>();
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            arg.data = input.GetTensorData<int64_t>();
            break;
          default:
            arg.data = nullptr;
            break;
        }

        inputs.push_back(std::move(arg));
        continue;
      }

      if (has_initializer) {
        const TensorInitializer& initializer = it->second;
        arg.present = true;
        arg.elem_type = initializer.elem_type;
        arg.shape = initializer.shape;
        arg.element_count = initializer.ElementCount();
        arg.data = initializer.raw_data.empty() ? nullptr : initializer.raw_data.data();
        inputs.push_back(std::move(arg));
        continue;
      }

      // Optional input not present.
      inputs.push_back(std::move(arg));
    }

    return nullptr;
  }

  OrtStatus* AllocateOutput(Ort::KernelContext& context,
                            size_t output_index,
                            gsl::span<const int64_t> shape,
                            Ort::UnownedValue& output) const {
    std::vector<int64_t> out_shape(shape.begin(), shape.end());
    output = context.GetOutput(output_index, out_shape);
    return nullptr;
  }

  OrtStatus* ComputeBinaryElementwise(Ort::KernelContext& context,
                                      const std::vector<TensorArg>& inputs) const {
    if (inputs.size() < 2) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, op_type_ + " expects 2 inputs");
    }

    const TensorArg& a = inputs[0];
    const TensorArg& b = inputs[1];
    RETURN_IF_ERROR(ValidateStaticTensor(ort_api_, a, input_names_[0]));
    RETURN_IF_ERROR(ValidateStaticTensor(ort_api_, b, input_names_[1]));

    if (a.elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        b.elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT,
                        op_type_ + " currently supports float tensors only");
    }

    std::vector<int64_t> out_shape;
    std::string err;
    if (!ComputeBroadcastShape({a.shape, b.shape}, out_shape, err)) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT,
                        op_type_ + " broadcast failure: " + err);
    }

    const auto output_elems_opt = TryComputeElementCount(out_shape);
    if (!output_elems_opt.has_value()) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, op_type_ + " output element count overflow");
    }

    Ort::UnownedValue output;
    RETURN_IF_ERROR(AllocateOutput(context, 0, out_shape, output));
    float* y = output.GetTensorMutableData<float>();
    if (*output_elems_opt == 0) {
      return nullptr;
    }

    const float* a_data = GetTensorData<float>(a, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    const float* b_data = GetTensorData<float>(b, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    if (a_data == nullptr || b_data == nullptr) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, op_type_ + " input data is null");
    }

    // Fast path for Mul using backend when no broadcast is required.
    if (op_kind_ == OpKind::kMul && a.shape == out_shape && b.shape == out_shape &&
        backend_.SupportsOp(telum::OpKind::kMul)) {
      OrtStatus* st = backend_.Mul(gsl::span<const float>(a_data, *output_elems_opt),
                                   gsl::span<const float>(b_data, *output_elems_opt),
                                   gsl::span<float>(y, *output_elems_opt));
      if (st == nullptr) {
        return nullptr;
      }

      Ort::Status backend_status{st};
      if (!kernel_config_.log_fallbacks) {
        return backend_status.release();
      }

      IGNORE_ORTSTATUS(ort_api_.Logger_LogMessage(
          &logger_, ORT_LOGGING_LEVEL_WARNING,
          ("Telum EP fallback to CPU path for " + op_type_ + ": " + backend_status.GetErrorMessage()).c_str(),
          ORT_FILE, __LINE__, __FUNCTION__));
      // Continue with CPU compute below.
    }

    const size_t out_rank = out_shape.size();
    std::vector<int64_t> a_aligned;
    std::vector<int64_t> b_aligned;
    AlignDimsToRank(a.shape, out_rank, a_aligned);
    AlignDimsToRank(b.shape, out_rank, b_aligned);

    auto out_strides = ComputeContiguousStrides(out_shape);
    auto a_strides = ComputeContiguousStrides(a_aligned);
    auto b_strides = ComputeContiguousStrides(b_aligned);

    for (size_t i = 0; i < out_rank; ++i) {
      if (a_aligned[i] == 1) {
        a_strides[i] = 0;
      }
      if (b_aligned[i] == 1) {
        b_strides[i] = 0;
      }
    }

    const size_t out_size = *output_elems_opt;
    for (size_t out_idx = 0; out_idx < out_size; ++out_idx) {
      size_t remainder = out_idx;
      size_t a_offset = 0;
      size_t b_offset = 0;
      for (size_t axis = 0; axis < out_rank; ++axis) {
        const int64_t stride = out_strides[axis];
        const int64_t coord = stride == 0 ? 0 : static_cast<int64_t>(remainder / static_cast<size_t>(stride));
        remainder = stride == 0 ? 0 : (remainder % static_cast<size_t>(stride));
        a_offset += static_cast<size_t>(coord * a_strides[axis]);
        b_offset += static_cast<size_t>(coord * b_strides[axis]);
      }

      const float av = a_data[a_offset];
      const float bv = b_data[b_offset];
      float out = 0.0f;
      switch (op_kind_) {
        case OpKind::kAdd:
          out = av + bv;
          break;
        case OpKind::kSub:
          out = av - bv;
          break;
        case OpKind::kMul:
          out = av * bv;
          break;
        case OpKind::kDiv:
          out = av / bv;
          break;
        case OpKind::kMin:
          out = std::min(av, bv);
          break;
        case OpKind::kMax:
          out = std::max(av, bv);
          break;
        default:
          return MakeStatus(ort_api_, ORT_EP_FAIL, "invalid binary op kind");
      }

      y[out_idx] = out;
    }

    return nullptr;
  }

  OrtStatus* ComputeUnary(Ort::KernelContext& context,
                          const std::vector<TensorArg>& inputs) const {
    if (inputs.empty()) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, op_type_ + " expects 1 input");
    }

    const TensorArg& x = inputs[0];
    RETURN_IF_ERROR(ValidateStaticTensor(ort_api_, x, input_names_[0]));
    if (x.elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, op_type_ + " supports float only");
    }

    const float* x_data = GetTensorData<float>(x, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    if (x_data == nullptr) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, op_type_ + " input data is null");
    }

    Ort::UnownedValue output;
    RETURN_IF_ERROR(AllocateOutput(context, 0, x.shape, output));
    float* y_data = output.GetTensorMutableData<float>();

    for (size_t i = 0; i < x.element_count; ++i) {
      const float v = x_data[i];
      float out = 0.0f;
      switch (op_kind_) {
        case OpKind::kRelu:
          out = std::max(0.0f, v);
          break;
        case OpKind::kGelu: {
          const float c = std::sqrt(2.0f / 3.14159265358979323846f);
          out = 0.5f * v * (1.0f + std::tanh(c * (v + 0.044715f * v * v * v)));
          break;
        }
        case OpKind::kTanh:
          out = std::tanh(v);
          break;
        case OpKind::kSigmoid:
          out = 1.0f / (1.0f + std::exp(-v));
          break;
        case OpKind::kExp:
          out = std::exp(v);
          break;
        case OpKind::kLog:
          out = std::log(v);
          break;
        case OpKind::kSqrt:
          out = std::sqrt(v);
          break;
        default:
          return MakeStatus(ort_api_, ORT_EP_FAIL, "invalid unary op kind");
      }
      y_data[i] = out;
    }

    return nullptr;
  }

  OrtStatus* ComputeMatMul(Ort::KernelContext& context,
                           const std::vector<TensorArg>& inputs) const {
    if (inputs.size() < 2) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "MatMul expects 2 inputs");
    }

    const TensorArg& a = inputs[0];
    const TensorArg& b = inputs[1];
    RETURN_IF_ERROR(ValidateStaticTensor(ort_api_, a, input_names_[0]));
    RETURN_IF_ERROR(ValidateStaticTensor(ort_api_, b, input_names_[1]));

    if (a.elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || b.elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "MatMul supports float only");
    }

    if (a.shape.size() < 2 || b.shape.size() < 2) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "MatMul requires input rank >= 2");
    }

    const int64_t m = a.shape[a.shape.size() - 2];
    const int64_t k_a = a.shape[a.shape.size() - 1];
    const int64_t k_b = b.shape[b.shape.size() - 2];
    const int64_t n = b.shape[b.shape.size() - 1];
    if (k_a != k_b) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "MatMul K dimensions must match");
    }

    std::vector<int64_t> a_batch(a.shape.begin(), a.shape.end() - 2);
    std::vector<int64_t> b_batch(b.shape.begin(), b.shape.end() - 2);
    std::vector<int64_t> out_batch;
    std::string err;
    if (!ComputeBroadcastShape({a_batch, b_batch}, out_batch, err)) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "MatMul batch broadcast failure: " + err);
    }

    std::vector<int64_t> out_shape = out_batch;
    out_shape.push_back(m);
    out_shape.push_back(n);

    Ort::UnownedValue output;
    RETURN_IF_ERROR(AllocateOutput(context, 0, out_shape, output));
    float* y = output.GetTensorMutableData<float>();

    const float* a_data = GetTensorData<float>(a, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    const float* b_data = GetTensorData<float>(b, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    if (a_data == nullptr || b_data == nullptr) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "MatMul input data is null");
    }

    std::vector<int64_t> a_aligned;
    std::vector<int64_t> b_aligned;
    AlignDimsToRank(a_batch, out_batch.size(), a_aligned);
    AlignDimsToRank(b_batch, out_batch.size(), b_aligned);

    std::vector<int64_t> a_dims = a_aligned;
    a_dims.push_back(m);
    a_dims.push_back(k_a);
    std::vector<int64_t> b_dims = b_aligned;
    b_dims.push_back(k_b);
    b_dims.push_back(n);

    const auto y_strides = ComputeContiguousStrides(out_shape);
    auto a_strides = ComputeContiguousStrides(a_dims);
    auto b_strides = ComputeContiguousStrides(b_dims);

    for (size_t i = 0; i < out_batch.size(); ++i) {
      if (a_aligned[i] == 1) {
        a_strides[i] = 0;
      }
      if (b_aligned[i] == 1) {
        b_strides[i] = 0;
      }
    }

    size_t batch_count = 1;
    for (int64_t d : out_batch) {
      batch_count *= static_cast<size_t>(d);
    }

    const size_t m_us = static_cast<size_t>(m);
    const size_t n_us = static_cast<size_t>(n);
    const size_t k_us = static_cast<size_t>(k_a);

    std::vector<int64_t> out_batch_strides = ComputeContiguousStrides(out_batch);

    for (size_t batch = 0; batch < batch_count; ++batch) {
      size_t tmp = batch;
      size_t a_base = 0;
      size_t b_base = 0;
      size_t y_base = 0;

      for (size_t axis = 0; axis < out_batch.size(); ++axis) {
        const int64_t stride = out_batch_strides.empty() ? 0 : out_batch_strides[axis];
        const int64_t coord = out_batch.empty() ? 0 : static_cast<int64_t>(tmp / static_cast<size_t>(stride));
        tmp = out_batch.empty() ? 0 : (tmp % static_cast<size_t>(stride));

        a_base += static_cast<size_t>(coord * a_strides[axis]);
        b_base += static_cast<size_t>(coord * b_strides[axis]);
        y_base += static_cast<size_t>(coord * y_strides[axis]);
      }

      for (size_t mi = 0; mi < m_us; ++mi) {
        for (size_t ni = 0; ni < n_us; ++ni) {
          float sum = 0.0f;
          for (size_t ki = 0; ki < k_us; ++ki) {
            const size_t a_off = a_base + mi * static_cast<size_t>(a_strides[a_strides.size() - 2]) +
                                 ki * static_cast<size_t>(a_strides[a_strides.size() - 1]);
            const size_t b_off = b_base + ki * static_cast<size_t>(b_strides[b_strides.size() - 2]) +
                                 ni * static_cast<size_t>(b_strides[b_strides.size() - 1]);
            sum += a_data[a_off] * b_data[b_off];
          }

          const size_t y_off = y_base + mi * static_cast<size_t>(y_strides[y_strides.size() - 2]) +
                               ni * static_cast<size_t>(y_strides[y_strides.size() - 1]);
          y[y_off] = sum;
        }
      }
    }

    return nullptr;
  }

  OrtStatus* ComputeGemm(Ort::KernelContext& context,
                         const std::vector<TensorArg>& inputs) const {
    if (inputs.size() < 2) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Gemm expects at least 2 inputs");
    }

    const TensorArg& a = inputs[0];
    const TensorArg& b = inputs[1];
    RETURN_IF_ERROR(ValidateStaticTensor(ort_api_, a, input_names_[0]));
    RETURN_IF_ERROR(ValidateStaticTensor(ort_api_, b, input_names_[1]));

    if (a.elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || b.elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Gemm supports float only");
    }

    if (a.shape.size() != 2 || b.shape.size() != 2) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Gemm expects 2D A and B");
    }

    const bool trans_a = attributes_.trans_a != 0;
    const bool trans_b = attributes_.trans_b != 0;

    const int64_t a_rows = trans_a ? a.shape[1] : a.shape[0];
    const int64_t a_cols = trans_a ? a.shape[0] : a.shape[1];
    const int64_t b_rows = trans_b ? b.shape[1] : b.shape[0];
    const int64_t b_cols = trans_b ? b.shape[0] : b.shape[1];

    if (a_cols != b_rows) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Gemm K dimensions mismatch");
    }

    const int64_t m = a_rows;
    const int64_t n = b_cols;
    const int64_t k = a_cols;

    std::vector<int64_t> out_shape{m, n};
    Ort::UnownedValue output;
    RETURN_IF_ERROR(AllocateOutput(context, 0, out_shape, output));

    const float* a_data = GetTensorData<float>(a, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    const float* b_data = GetTensorData<float>(b, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    float* y_data = output.GetTensorMutableData<float>();

    const TensorArg* c = (inputs.size() > 2 && inputs[2].present) ? &inputs[2] : nullptr;
    const float* c_data = nullptr;
    std::vector<int64_t> c_shape;
    if (c != nullptr) {
      if (c->elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Gemm C must be float");
      }
      c_data = GetTensorData<float>(*c, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
      c_shape = c->shape;
    }

    const auto get_a = [&](int64_t row, int64_t col) -> float {
      if (trans_a) {
        return a_data[col * a.shape[1] + row];
      }
      return a_data[row * a.shape[1] + col];
    };

    const auto get_b = [&](int64_t row, int64_t col) -> float {
      if (trans_b) {
        return b_data[col * b.shape[1] + row];
      }
      return b_data[row * b.shape[1] + col];
    };

    const auto get_c = [&](int64_t row, int64_t col) -> float {
      if (c_data == nullptr) {
        return 0.0f;
      }

      if (c_shape.empty()) {
        return c_data[0];
      }

      if (c_shape.size() == 1) {
        if (c_shape[0] == 1) {
          return c_data[0];
        }
        return c_data[static_cast<size_t>(col)];
      }

      if (c_shape.size() == 2) {
        const int64_t c_rows = c_shape[0];
        const int64_t c_cols = c_shape[1];
        const int64_t r = c_rows == 1 ? 0 : row;
        const int64_t cidx = c_cols == 1 ? 0 : col;
        return c_data[static_cast<size_t>(r * c_cols + cidx)];
      }

      return 0.0f;
    };

    for (int64_t mi = 0; mi < m; ++mi) {
      for (int64_t ni = 0; ni < n; ++ni) {
        float sum = 0.0f;
        for (int64_t ki = 0; ki < k; ++ki) {
          sum += get_a(mi, ki) * get_b(ki, ni);
        }

        const float c_val = get_c(mi, ni);
        y_data[static_cast<size_t>(mi * n + ni)] = attributes_.alpha * sum + attributes_.beta * c_val;
      }
    }

    return nullptr;
  }

  OrtStatus* ComputeSoftmax(Ort::KernelContext& context,
                            const std::vector<TensorArg>& inputs) const {
    if (inputs.empty()) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Softmax expects 1 input");
    }

    const TensorArg& x = inputs[0];
    RETURN_IF_ERROR(ValidateStaticTensor(ort_api_, x, input_names_[0]));
    if (x.elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Softmax supports float only");
    }

    const float* x_data = GetTensorData<float>(x, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

    const int64_t axis = NormalizeAxis(attributes_.axis, x.shape.size());
    size_t outer = 1;
    size_t axis_dim = static_cast<size_t>(x.shape[static_cast<size_t>(axis)]);
    size_t inner = 1;
    for (size_t i = 0; i < static_cast<size_t>(axis); ++i) {
      outer *= static_cast<size_t>(x.shape[i]);
    }
    for (size_t i = static_cast<size_t>(axis) + 1; i < x.shape.size(); ++i) {
      inner *= static_cast<size_t>(x.shape[i]);
    }

    Ort::UnownedValue output;
    RETURN_IF_ERROR(AllocateOutput(context, 0, x.shape, output));
    float* y_data = output.GetTensorMutableData<float>();

    for (size_t o = 0; o < outer; ++o) {
      for (size_t in = 0; in < inner; ++in) {
        const size_t base = o * axis_dim * inner + in;

        float max_v = -std::numeric_limits<float>::infinity();
        for (size_t a = 0; a < axis_dim; ++a) {
          max_v = std::max(max_v, x_data[base + a * inner]);
        }

        float sum = 0.0f;
        for (size_t a = 0; a < axis_dim; ++a) {
          const float ev = std::exp(x_data[base + a * inner] - max_v);
          y_data[base + a * inner] = ev;
          sum += ev;
        }

        const float inv_sum = 1.0f / sum;
        for (size_t a = 0; a < axis_dim; ++a) {
          y_data[base + a * inner] *= inv_sum;
        }
      }
    }

    return nullptr;
  }

  OrtStatus* ComputeLayerNormalization(Ort::KernelContext& context,
                                       const std::vector<TensorArg>& inputs) const {
    if (inputs.size() < 2) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "LayerNormalization expects at least 2 inputs");
    }

    const TensorArg& x = inputs[0];
    const TensorArg& scale = inputs[1];
    RETURN_IF_ERROR(ValidateStaticTensor(ort_api_, x, input_names_[0]));
    RETURN_IF_ERROR(ValidateStaticTensor(ort_api_, scale, input_names_[1]));

    if (x.elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        scale.elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "LayerNormalization supports float only");
    }

    const TensorArg* bias = (inputs.size() > 2 && inputs[2].present) ? &inputs[2] : nullptr;
    if (bias != nullptr && bias->elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "LayerNormalization bias must be float");
    }

    const int64_t axis = NormalizeAxis(attributes_.axis, x.shape.size());
    size_t outer = 1;
    size_t inner = 1;
    for (size_t i = 0; i < static_cast<size_t>(axis); ++i) {
      outer *= static_cast<size_t>(x.shape[i]);
    }
    for (size_t i = static_cast<size_t>(axis); i < x.shape.size(); ++i) {
      inner *= static_cast<size_t>(x.shape[i]);
    }

    if (scale.element_count != inner) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "LayerNormalization scale size mismatch");
    }

    const float* x_data = GetTensorData<float>(x, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    const float* scale_data = GetTensorData<float>(scale, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    const float* bias_data = bias == nullptr ? nullptr : GetTensorData<float>(*bias, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

    Ort::UnownedValue output;
    RETURN_IF_ERROR(AllocateOutput(context, 0, x.shape, output));
    float* y_data = output.GetTensorMutableData<float>();

    for (size_t o = 0; o < outer; ++o) {
      const size_t base = o * inner;

      double mean = 0.0;
      for (size_t i = 0; i < inner; ++i) {
        mean += static_cast<double>(x_data[base + i]);
      }
      mean /= static_cast<double>(inner);

      double var = 0.0;
      for (size_t i = 0; i < inner; ++i) {
        const double diff = static_cast<double>(x_data[base + i]) - mean;
        var += diff * diff;
      }
      var /= static_cast<double>(inner);
      const float inv_std = 1.0f / std::sqrt(static_cast<float>(var) + attributes_.epsilon);

      for (size_t i = 0; i < inner; ++i) {
        float v = (x_data[base + i] - static_cast<float>(mean)) * inv_std;
        v *= scale_data[i];
        if (bias_data != nullptr) {
          v += bias_data[i];
        }
        y_data[base + i] = v;
      }
    }

    return nullptr;
  }

  OrtStatus* ComputeReshape(Ort::KernelContext& context,
                            const std::vector<TensorArg>& inputs) const {
    if (inputs.size() < 2) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Reshape expects 2 inputs");
    }

    const TensorArg& x = inputs[0];
    const TensorArg& shape_tensor = inputs[1];
    RETURN_IF_ERROR(ValidateStaticTensor(ort_api_, x, input_names_[0]));
    RETURN_IF_ERROR(ValidateStaticTensor(ort_api_, shape_tensor, input_names_[1]));

    std::vector<int64_t> requested_shape;
    if (!ReadIntVector(shape_tensor, requested_shape)) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Reshape shape input must be int32/int64 tensor");
    }

    std::vector<int64_t> out_shape = requested_shape;
    int64_t infer_idx = -1;
    size_t known_prod = 1;
    const size_t input_size = x.element_count;

    for (size_t i = 0; i < out_shape.size(); ++i) {
      int64_t dim = out_shape[i];
      if (dim == 0) {
        if (!attributes_.allow_zero) {
          if (i >= x.shape.size()) {
            return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Reshape 0-dim refers to invalid axis");
          }
          dim = x.shape[i];
          out_shape[i] = dim;
        }
      }

      if (dim == -1) {
        if (infer_idx != -1) {
          return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Reshape has multiple -1 dimensions");
        }
        infer_idx = static_cast<int64_t>(i);
      } else if (dim < -1) {
        return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Reshape has invalid negative dimension");
      } else {
        known_prod *= static_cast<size_t>(dim == 0 ? 0 : dim);
      }
    }

    if (infer_idx >= 0) {
      if (known_prod == 0 || input_size % known_prod != 0) {
        return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Reshape cannot infer dimension");
      }
      out_shape[static_cast<size_t>(infer_idx)] = static_cast<int64_t>(input_size / known_prod);
    }

    const auto output_size_opt = TryComputeElementCount(out_shape);
    if (!output_size_opt.has_value() || *output_size_opt != input_size) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Reshape output size mismatch");
    }

    Ort::UnownedValue output;
    RETURN_IF_ERROR(AllocateOutput(context, 0, out_shape, output));
    void* out_data = output.GetTensorMutableRawData();

    const size_t elem_size = GetElementSize(x.elem_type);
    if (elem_size == 0) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Reshape unsupported input type");
    }

    std::memcpy(out_data, x.data, input_size * elem_size);
    return nullptr;
  }

  OrtStatus* ComputeTranspose(Ort::KernelContext& context,
                              const std::vector<TensorArg>& inputs) const {
    if (inputs.empty()) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Transpose expects 1 input");
    }

    const TensorArg& x = inputs[0];
    RETURN_IF_ERROR(ValidateStaticTensor(ort_api_, x, input_names_[0]));

    const size_t rank = x.shape.size();
    std::vector<size_t> perm;
    if (attributes_.has_perm) {
      perm.reserve(attributes_.perm.size());
      for (int64_t v : attributes_.perm) {
        if (v < 0 || static_cast<size_t>(v) >= rank) {
          return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Transpose perm has invalid axis");
        }
        perm.push_back(static_cast<size_t>(v));
      }
      if (perm.size() != rank) {
        return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Transpose perm rank mismatch");
      }
      std::vector<bool> seen(rank, false);
      for (size_t axis : perm) {
        if (seen[axis]) {
          return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Transpose perm contains duplicate axis");
        }
        seen[axis] = true;
      }
    } else {
      perm.resize(rank);
      for (size_t i = 0; i < rank; ++i) {
        perm[i] = rank - 1 - i;
      }
    }

    std::vector<int64_t> out_shape(rank);
    for (size_t i = 0; i < rank; ++i) {
      out_shape[i] = x.shape[perm[i]];
    }

    Ort::UnownedValue output;
    RETURN_IF_ERROR(AllocateOutput(context, 0, out_shape, output));
    void* y_data = output.GetTensorMutableRawData();

    if (x.element_count == 0) {
      return nullptr;
    }

    const size_t elem_size = GetElementSize(x.elem_type);
    if (elem_size == 0) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Transpose unsupported element type");
    }

    if (rank == 0) {
      std::memcpy(y_data, x.data, elem_size);
      return nullptr;
    }

    const auto in_strides = ComputeContiguousStrides(x.shape);
    const auto out_strides = ComputeContiguousStrides(out_shape);

    uint8_t* y_bytes = reinterpret_cast<uint8_t*>(y_data);
    const uint8_t* x_bytes = reinterpret_cast<const uint8_t*>(x.data);

    for (size_t out_idx = 0; out_idx < x.element_count; ++out_idx) {
      size_t remainder = out_idx;
      size_t in_offset = 0;
      for (size_t axis = 0; axis < rank; ++axis) {
        const int64_t stride = out_strides[axis];
        const size_t coord = stride == 0 ? 0 : (remainder / static_cast<size_t>(stride));
        remainder = stride == 0 ? 0 : (remainder % static_cast<size_t>(stride));
        in_offset += coord * static_cast<size_t>(in_strides[perm[axis]]);
      }

      std::memcpy(y_bytes + out_idx * elem_size, x_bytes + in_offset * elem_size, elem_size);
    }

    return nullptr;
  }

  OrtStatus* ComputeSqueeze(Ort::KernelContext& context,
                            const std::vector<TensorArg>& inputs) const {
    return ComputeSqueezeUnsqueeze(context, inputs, /*is_unsqueeze*/ false);
  }

  OrtStatus* ComputeUnsqueeze(Ort::KernelContext& context,
                              const std::vector<TensorArg>& inputs) const {
    return ComputeSqueezeUnsqueeze(context, inputs, /*is_unsqueeze*/ true);
  }

  OrtStatus* ComputeSqueezeUnsqueeze(Ort::KernelContext& context,
                                     const std::vector<TensorArg>& inputs,
                                     bool is_unsqueeze) const {
    if (inputs.empty()) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT,
                        std::string(is_unsqueeze ? "Unsqueeze" : "Squeeze") + " expects input X");
    }

    const TensorArg& x = inputs[0];
    RETURN_IF_ERROR(ValidateStaticTensor(ort_api_, x, input_names_[0]));

    std::vector<int64_t> axes;
    if (inputs.size() > 1 && inputs[1].present) {
      if (!ReadIntVector(inputs[1], axes)) {
        return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT,
                          std::string(is_unsqueeze ? "Unsqueeze" : "Squeeze") + " axes input must be int tensor");
      }
    } else if (attributes_.has_axes) {
      axes = attributes_.axes;
    }

    std::vector<int64_t> out_shape;
    if (is_unsqueeze) {
      if (axes.empty()) {
        return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Unsqueeze requires axes");
      }

      const size_t out_rank = x.shape.size() + axes.size();
      std::vector<int64_t> axis_flags(out_rank, 0);
      for (int64_t axis : axes) {
        int64_t a = NormalizeAxis(axis, out_rank, /*allow_end*/ true);
        if (axis_flags[static_cast<size_t>(a)] != 0) {
          return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Unsqueeze axes contains duplicate axis");
        }
        axis_flags[static_cast<size_t>(a)] = 1;
      }

      out_shape.assign(out_rank, 0);
      size_t in_idx = 0;
      for (size_t i = 0; i < out_rank; ++i) {
        if (axis_flags[i] != 0) {
          out_shape[i] = 1;
        } else {
          out_shape[i] = x.shape[in_idx++];
        }
      }
    } else {
      std::unordered_set<int64_t> axis_set;
      if (!axes.empty()) {
        for (int64_t axis : axes) {
          axis_set.insert(NormalizeAxis(axis, x.shape.size()));
        }
      }

      for (size_t i = 0; i < x.shape.size(); ++i) {
        const bool remove = axes.empty() ? (x.shape[i] == 1) : (axis_set.count(static_cast<int64_t>(i)) != 0);
        if (remove) {
          if (x.shape[i] != 1) {
            return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Squeeze axis dimension must be 1");
          }
          continue;
        }

        out_shape.push_back(x.shape[i]);
      }
    }

    Ort::UnownedValue output;
    RETURN_IF_ERROR(AllocateOutput(context, 0, out_shape, output));
    void* out_data = output.GetTensorMutableRawData();

    const size_t elem_size = GetElementSize(x.elem_type);
    if (elem_size == 0) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Squeeze/Unsqueeze unsupported element type");
    }

    std::memcpy(out_data, x.data, x.element_count * elem_size);
    return nullptr;
  }

  OrtStatus* ComputeReduceMean(Ort::KernelContext& context,
                               const std::vector<TensorArg>& inputs) const {
    if (inputs.empty()) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "ReduceMean expects input X");
    }

    const TensorArg& x = inputs[0];
    RETURN_IF_ERROR(ValidateStaticTensor(ort_api_, x, input_names_[0]));
    if (x.elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "ReduceMean supports float only");
    }

    std::vector<int64_t> axes;
    if (inputs.size() > 1 && inputs[1].present) {
      if (!ReadIntVector(inputs[1], axes)) {
        return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "ReduceMean axes input must be int tensor");
      }
    } else if (attributes_.has_axes) {
      axes = attributes_.axes;
    }

    std::unordered_set<int64_t> axis_set;
    if (!axes.empty()) {
      for (int64_t axis : axes) {
        axis_set.insert(NormalizeAxis(axis, x.shape.size()));
      }
    } else if (!attributes_.noop_with_empty_axes) {
      for (size_t i = 0; i < x.shape.size(); ++i) {
        axis_set.insert(static_cast<int64_t>(i));
      }
    }

    std::vector<int64_t> out_shape;
    for (size_t i = 0; i < x.shape.size(); ++i) {
      const bool reduced = axis_set.count(static_cast<int64_t>(i)) != 0;
      if (reduced) {
        if (attributes_.keepdims) {
          out_shape.push_back(1);
        }
      } else {
        out_shape.push_back(x.shape[i]);
      }
    }

    if (out_shape.empty() && attributes_.keepdims) {
      out_shape.push_back(1);
    }

    const auto out_count_opt = TryComputeElementCount(out_shape);
    if (!out_count_opt.has_value()) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "ReduceMean output size overflow");
    }

    Ort::UnownedValue output;
    RETURN_IF_ERROR(AllocateOutput(context, 0, out_shape, output));
    float* y_data = output.GetTensorMutableData<float>();

    const float* x_data = GetTensorData<float>(x, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

    std::vector<double> sums(*out_count_opt, 0.0);
    std::vector<size_t> counts(*out_count_opt, 0);

    const auto in_strides = ComputeContiguousStrides(x.shape);
    const auto out_strides = ComputeContiguousStrides(out_shape);

    for (size_t idx = 0; idx < x.element_count; ++idx) {
      size_t remainder = idx;
      size_t out_offset = 0;
      size_t out_axis = 0;

      for (size_t axis = 0; axis < x.shape.size(); ++axis) {
        const int64_t stride = in_strides[axis];
        const int64_t coord = stride == 0 ? 0 : static_cast<int64_t>(remainder / static_cast<size_t>(stride));
        remainder = stride == 0 ? 0 : (remainder % static_cast<size_t>(stride));

        const bool reduced = axis_set.count(static_cast<int64_t>(axis)) != 0;
        if (reduced) {
          if (attributes_.keepdims) {
            ++out_axis;
          }
          continue;
        }

        out_offset += static_cast<size_t>(coord * out_strides[out_axis]);
        ++out_axis;
      }

      sums[out_offset] += static_cast<double>(x_data[idx]);
      counts[out_offset] += 1;
    }

    for (size_t i = 0; i < *out_count_opt; ++i) {
      y_data[i] = counts[i] == 0 ? 0.0f : static_cast<float>(sums[i] / static_cast<double>(counts[i]));
    }

    return nullptr;
  }

  template <typename SrcT, typename DstT>
  static void ConvertArray(const SrcT* src, DstT* dst, size_t count) {
    for (size_t i = 0; i < count; ++i) {
      dst[i] = static_cast<DstT>(src[i]);
    }
  }

  OrtStatus* ComputeCast(Ort::KernelContext& context,
                         const std::vector<TensorArg>& inputs) const {
    if (inputs.empty()) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Cast expects 1 input");
    }

    const TensorArg& x = inputs[0];
    RETURN_IF_ERROR(ValidateStaticTensor(ort_api_, x, input_names_[0]));

    Ort::UnownedValue output;
    RETURN_IF_ERROR(AllocateOutput(context, 0, x.shape, output));

    auto out_type = output.GetTensorTypeAndShapeInfo().GetElementType();
    const auto to_type = static_cast<ONNXTensorElementDataType>(attributes_.cast_to);
    if (to_type != out_type) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Cast output type mismatch with model output");
    }

    const size_t count = x.element_count;

    if (x.elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && to_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      std::memcpy(output.GetTensorMutableData<float>(), x.data, count * sizeof(float));
      return nullptr;
    }

    if (to_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      float* dst = output.GetTensorMutableData<float>();
      if (x.elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        ConvertArray(GetTensorData<int64_t>(x, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64), dst, count);
        return nullptr;
      }
      if (x.elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
        ConvertArray(GetTensorData<int32_t>(x, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32), dst, count);
        return nullptr;
      }
      if (x.elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
        ConvertArray(GetTensorData<bool>(x, ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL), dst, count);
        return nullptr;
      }
    }

    if (to_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      int64_t* dst = output.GetTensorMutableData<int64_t>();
      if (x.elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        ConvertArray(GetTensorData<float>(x, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT), dst, count);
        return nullptr;
      }
      if (x.elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
        ConvertArray(GetTensorData<int32_t>(x, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32), dst, count);
        return nullptr;
      }
      if (x.elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
        ConvertArray(GetTensorData<bool>(x, ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL), dst, count);
        return nullptr;
      }
    }

    if (to_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
      int32_t* dst = output.GetTensorMutableData<int32_t>();
      if (x.elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        ConvertArray(GetTensorData<float>(x, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT), dst, count);
        return nullptr;
      }
      if (x.elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        ConvertArray(GetTensorData<int64_t>(x, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64), dst, count);
        return nullptr;
      }
      if (x.elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
        ConvertArray(GetTensorData<bool>(x, ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL), dst, count);
        return nullptr;
      }
    }

    if (to_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
      bool* dst = output.GetTensorMutableData<bool>();
      if (x.elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        const float* src = GetTensorData<float>(x, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
        for (size_t i = 0; i < count; ++i) dst[i] = src[i] != 0.0f;
        return nullptr;
      }
      if (x.elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        const int64_t* src = GetTensorData<int64_t>(x, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
        for (size_t i = 0; i < count; ++i) dst[i] = src[i] != 0;
        return nullptr;
      }
      if (x.elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
        const int32_t* src = GetTensorData<int32_t>(x, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
        for (size_t i = 0; i < count; ++i) dst[i] = src[i] != 0;
        return nullptr;
      }
    }

    return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Unsupported Cast type combination");
  }

  OrtStatus* ComputeWhere(Ort::KernelContext& context,
                          const std::vector<TensorArg>& inputs) const {
    if (inputs.size() < 3) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Where expects 3 inputs");
    }

    const TensorArg& cond = inputs[0];
    const TensorArg& x = inputs[1];
    const TensorArg& y = inputs[2];
    RETURN_IF_ERROR(ValidateStaticTensor(ort_api_, cond, input_names_[0]));
    RETURN_IF_ERROR(ValidateStaticTensor(ort_api_, x, input_names_[1]));
    RETURN_IF_ERROR(ValidateStaticTensor(ort_api_, y, input_names_[2]));

    if (cond.elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Where condition must be bool");
    }
    if (x.elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || y.elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Where supports float data tensors only");
    }

    std::vector<int64_t> out_shape;
    std::string err;
    if (!ComputeBroadcastShape({cond.shape, x.shape, y.shape}, out_shape, err)) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Where broadcast failure: " + err);
    }

    const auto out_count_opt = TryComputeElementCount(out_shape);
    if (!out_count_opt.has_value()) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Where output size overflow");
    }

    Ort::UnownedValue output;
    RETURN_IF_ERROR(AllocateOutput(context, 0, out_shape, output));
    float* out = output.GetTensorMutableData<float>();

    const size_t out_rank = out_shape.size();
    std::vector<int64_t> cond_dims;
    std::vector<int64_t> x_dims;
    std::vector<int64_t> y_dims;
    AlignDimsToRank(cond.shape, out_rank, cond_dims);
    AlignDimsToRank(x.shape, out_rank, x_dims);
    AlignDimsToRank(y.shape, out_rank, y_dims);

    auto cond_strides = ComputeContiguousStrides(cond_dims);
    auto x_strides = ComputeContiguousStrides(x_dims);
    auto y_strides = ComputeContiguousStrides(y_dims);
    auto out_strides = ComputeContiguousStrides(out_shape);

    for (size_t i = 0; i < out_rank; ++i) {
      if (cond_dims[i] == 1) cond_strides[i] = 0;
      if (x_dims[i] == 1) x_strides[i] = 0;
      if (y_dims[i] == 1) y_strides[i] = 0;
    }

    const bool* cond_data = GetTensorData<bool>(cond, ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);
    const float* x_data = GetTensorData<float>(x, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    const float* y_data = GetTensorData<float>(y, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

    for (size_t out_idx = 0; out_idx < *out_count_opt; ++out_idx) {
      size_t rem = out_idx;
      size_t c_off = 0;
      size_t x_off = 0;
      size_t y_off = 0;
      for (size_t axis = 0; axis < out_rank; ++axis) {
        const int64_t stride = out_strides[axis];
        const int64_t coord = stride == 0 ? 0 : static_cast<int64_t>(rem / static_cast<size_t>(stride));
        rem = stride == 0 ? 0 : (rem % static_cast<size_t>(stride));
        c_off += static_cast<size_t>(coord * cond_strides[axis]);
        x_off += static_cast<size_t>(coord * x_strides[axis]);
        y_off += static_cast<size_t>(coord * y_strides[axis]);
      }

      out[out_idx] = cond_data[c_off] ? x_data[x_off] : y_data[y_off];
    }

    return nullptr;
  }

  OrtStatus* ComputeExpand(Ort::KernelContext& context,
                           const std::vector<TensorArg>& inputs) const {
    if (inputs.size() < 2) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Expand expects 2 inputs");
    }

    const TensorArg& x = inputs[0];
    const TensorArg& shape_tensor = inputs[1];
    RETURN_IF_ERROR(ValidateStaticTensor(ort_api_, x, input_names_[0]));
    RETURN_IF_ERROR(ValidateStaticTensor(ort_api_, shape_tensor, input_names_[1]));

    if (x.elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Expand supports float only");
    }

    std::vector<int64_t> target_shape;
    if (!ReadIntVector(shape_tensor, target_shape)) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Expand shape input must be int tensor");
    }

    std::vector<int64_t> out_shape;
    std::string err;
    if (!ComputeBroadcastShape({x.shape, target_shape}, out_shape, err)) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Expand broadcast failure: " + err);
    }

    if (out_shape != target_shape) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Expand target shape is not broadcast-compatible");
    }

    const auto out_count_opt = TryComputeElementCount(out_shape);
    if (!out_count_opt.has_value()) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Expand output size overflow");
    }

    Ort::UnownedValue output;
    RETURN_IF_ERROR(AllocateOutput(context, 0, out_shape, output));
    float* out = output.GetTensorMutableData<float>();

    const size_t out_rank = out_shape.size();
    std::vector<int64_t> x_dims;
    AlignDimsToRank(x.shape, out_rank, x_dims);
    auto x_strides = ComputeContiguousStrides(x_dims);
    auto out_strides = ComputeContiguousStrides(out_shape);
    for (size_t i = 0; i < out_rank; ++i) {
      if (x_dims[i] == 1) {
        x_strides[i] = 0;
      }
    }

    const float* x_data = GetTensorData<float>(x, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

    for (size_t out_idx = 0; out_idx < *out_count_opt; ++out_idx) {
      size_t rem = out_idx;
      size_t x_off = 0;
      for (size_t axis = 0; axis < out_rank; ++axis) {
        const int64_t stride = out_strides[axis];
        const int64_t coord = stride == 0 ? 0 : static_cast<int64_t>(rem / static_cast<size_t>(stride));
        rem = stride == 0 ? 0 : (rem % static_cast<size_t>(stride));
        x_off += static_cast<size_t>(coord * x_strides[axis]);
      }
      out[out_idx] = x_data[x_off];
    }

    return nullptr;
  }

  OrtStatus* ComputeConcat(Ort::KernelContext& context,
                           const std::vector<TensorArg>& inputs) const {
    if (inputs.empty()) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Concat requires at least one input");
    }

    std::vector<const TensorArg*> present_inputs;
    present_inputs.reserve(inputs.size());
    for (const auto& input : inputs) {
      if (input.present) {
        present_inputs.push_back(&input);
      }
    }

    if (present_inputs.empty()) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Concat has no present inputs");
    }

    const TensorArg& first = *present_inputs[0];
    RETURN_IF_ERROR(ValidateStaticTensor(ort_api_, first, "input0"));
    if (first.elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Concat supports float only");
    }

    const int64_t axis = NormalizeAxis(attributes_.concat_axis, first.shape.size());
    std::vector<int64_t> out_shape = first.shape;

    for (size_t i = 1; i < present_inputs.size(); ++i) {
      const TensorArg& cur = *present_inputs[i];
      RETURN_IF_ERROR(ValidateStaticTensor(ort_api_, cur, "input" + std::to_string(i)));
      if (cur.elem_type != first.elem_type) {
        return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Concat inputs must share the same element type");
      }
      if (cur.shape.size() != first.shape.size()) {
        return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Concat input rank mismatch");
      }
      for (size_t d = 0; d < cur.shape.size(); ++d) {
        if (static_cast<int64_t>(d) == axis) {
          continue;
        }
        if (cur.shape[d] != out_shape[d]) {
          return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Concat non-axis dimensions must match");
        }
      }
      out_shape[static_cast<size_t>(axis)] += cur.shape[static_cast<size_t>(axis)];
    }

    Ort::UnownedValue output;
    RETURN_IF_ERROR(AllocateOutput(context, 0, out_shape, output));
    float* out_data = output.GetTensorMutableData<float>();

    size_t outer = 1;
    size_t inner = 1;
    for (size_t i = 0; i < static_cast<size_t>(axis); ++i) {
      outer *= static_cast<size_t>(out_shape[i]);
    }
    for (size_t i = static_cast<size_t>(axis) + 1; i < out_shape.size(); ++i) {
      inner *= static_cast<size_t>(out_shape[i]);
    }

    size_t axis_offset = 0;
    for (const TensorArg* input : present_inputs) {
      const size_t axis_dim = static_cast<size_t>(input->shape[static_cast<size_t>(axis)]);
      const size_t block = axis_dim * inner;
      const float* src = GetTensorData<float>(*input, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

      for (size_t o = 0; o < outer; ++o) {
        const size_t dst_off = (o * static_cast<size_t>(out_shape[static_cast<size_t>(axis)]) + axis_offset) * inner;
        const size_t src_off = o * block;
        std::memcpy(out_data + dst_off, src + src_off, block * sizeof(float));
      }

      axis_offset += axis_dim;
    }

    return nullptr;
  }

  OrtStatus* ComputeGather(Ort::KernelContext& context,
                           const std::vector<TensorArg>& inputs) const {
    if (inputs.size() < 2) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Gather expects 2 inputs");
    }

    const TensorArg& data = inputs[0];
    const TensorArg& indices = inputs[1];
    RETURN_IF_ERROR(ValidateStaticTensor(ort_api_, data, input_names_[0]));
    RETURN_IF_ERROR(ValidateStaticTensor(ort_api_, indices, input_names_[1]));

    if (data.elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Gather supports float data only");
    }

    if (indices.elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 &&
        indices.elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Gather indices must be int32/int64");
    }

    const int64_t axis = NormalizeAxis(attributes_.axis, data.shape.size());
    const size_t axis_us = static_cast<size_t>(axis);

    std::vector<int64_t> out_shape;
    out_shape.insert(out_shape.end(), data.shape.begin(), data.shape.begin() + static_cast<ptrdiff_t>(axis_us));
    out_shape.insert(out_shape.end(), indices.shape.begin(), indices.shape.end());
    out_shape.insert(out_shape.end(), data.shape.begin() + static_cast<ptrdiff_t>(axis_us + 1), data.shape.end());

    Ort::UnownedValue output;
    RETURN_IF_ERROR(AllocateOutput(context, 0, out_shape, output));
    float* out = output.GetTensorMutableData<float>();

    size_t prefix = 1;
    size_t suffix = 1;
    for (size_t i = 0; i < axis_us; ++i) {
      prefix *= static_cast<size_t>(data.shape[i]);
    }
    for (size_t i = axis_us + 1; i < data.shape.size(); ++i) {
      suffix *= static_cast<size_t>(data.shape[i]);
    }

    const int64_t axis_dim = data.shape[axis_us];
    const size_t indices_count = indices.element_count;

    const float* data_ptr = GetTensorData<float>(data, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

    auto get_index = [&](size_t idx_pos) -> int64_t {
      if (indices.elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        return reinterpret_cast<const int64_t*>(indices.data)[idx_pos];
      }
      return static_cast<int64_t>(reinterpret_cast<const int32_t*>(indices.data)[idx_pos]);
    };

    size_t out_offset = 0;
    for (size_t p = 0; p < prefix; ++p) {
      for (size_t i = 0; i < indices_count; ++i) {
        int64_t idx = get_index(i);
        if (idx < 0) {
          idx += axis_dim;
        }
        if (idx < 0 || idx >= axis_dim) {
          return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Gather index out of range");
        }

        const size_t src_off = (p * static_cast<size_t>(axis_dim) + static_cast<size_t>(idx)) * suffix;
        std::memcpy(out + out_offset, data_ptr + src_off, suffix * sizeof(float));
        out_offset += suffix;
      }
    }

    return nullptr;
  }

  OrtStatus* ComputeSlice(Ort::KernelContext& context,
                          const std::vector<TensorArg>& inputs) const {
    if (inputs.size() < 3) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Slice expects at least 3 inputs");
    }

    const TensorArg& data = inputs[0];
    const TensorArg& starts = inputs[1];
    const TensorArg& ends = inputs[2];
    RETURN_IF_ERROR(ValidateStaticTensor(ort_api_, data, input_names_[0]));
    RETURN_IF_ERROR(ValidateStaticTensor(ort_api_, starts, input_names_[1]));
    RETURN_IF_ERROR(ValidateStaticTensor(ort_api_, ends, input_names_[2]));

    if (data.elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Slice supports float data only");
    }

    std::vector<int64_t> starts_vec;
    std::vector<int64_t> ends_vec;
    if (!ReadIntVector(starts, starts_vec) || !ReadIntVector(ends, ends_vec)) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Slice starts/ends must be integer tensors");
    }

    if (starts_vec.size() != ends_vec.size()) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Slice starts/ends length mismatch");
    }

    std::vector<int64_t> axes_vec;
    if (inputs.size() > 3 && inputs[3].present) {
      if (!ReadIntVector(inputs[3], axes_vec)) {
        return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Slice axes must be integer tensor");
      }
    } else {
      axes_vec.resize(starts_vec.size());
      for (size_t i = 0; i < axes_vec.size(); ++i) {
        axes_vec[i] = static_cast<int64_t>(i);
      }
    }

    std::vector<int64_t> steps_vec;
    if (inputs.size() > 4 && inputs[4].present) {
      if (!ReadIntVector(inputs[4], steps_vec)) {
        return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Slice steps must be integer tensor");
      }
    } else {
      steps_vec.assign(starts_vec.size(), 1);
    }

    if (axes_vec.size() != starts_vec.size() || steps_vec.size() != starts_vec.size()) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Slice axes/steps length mismatch");
    }

    const size_t rank = data.shape.size();
    std::vector<int64_t> starts_full(rank, 0);
    std::vector<int64_t> ends_full = data.shape;
    std::vector<int64_t> steps_full(rank, 1);

    for (size_t i = 0; i < axes_vec.size(); ++i) {
      const size_t axis = static_cast<size_t>(NormalizeAxis(axes_vec[i], rank));
      const int64_t dim = data.shape[axis];
      int64_t step = steps_vec[i];
      if (step == 0) {
        return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Slice step must be non-zero");
      }

      int64_t start = starts_vec[i];
      int64_t end = ends_vec[i];

      if (start < 0) start += dim;
      if (end < 0) end += dim;

      if (step > 0) {
        start = std::clamp(start, static_cast<int64_t>(0), dim);
        end = std::clamp(end, static_cast<int64_t>(0), dim);
      } else {
        start = std::clamp(start, static_cast<int64_t>(-1), dim - 1);
        end = std::clamp(end, static_cast<int64_t>(-1), dim - 1);
      }

      starts_full[axis] = start;
      ends_full[axis] = end;
      steps_full[axis] = step;
    }

    std::vector<int64_t> out_shape(rank, 0);
    for (size_t axis = 0; axis < rank; ++axis) {
      const int64_t step = steps_full[axis];
      const int64_t start = starts_full[axis];
      const int64_t end = ends_full[axis];
      int64_t len = 0;
      if (step > 0) {
        if (end > start) {
          len = (end - start + step - 1) / step;
        }
      } else {
        if (start > end) {
          len = (start - end - step - 1) / (-step);
        }
      }
      out_shape[axis] = std::max<int64_t>(len, 0);
    }

    Ort::UnownedValue output;
    RETURN_IF_ERROR(AllocateOutput(context, 0, out_shape, output));
    float* out = output.GetTensorMutableData<float>();

    const auto out_count_opt = TryComputeElementCount(out_shape);
    if (!out_count_opt.has_value()) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "Slice output size overflow");
    }

    const auto in_strides = ComputeContiguousStrides(data.shape);
    const auto out_strides = ComputeContiguousStrides(out_shape);
    const float* in = GetTensorData<float>(data, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

    for (size_t out_idx = 0; out_idx < *out_count_opt; ++out_idx) {
      size_t rem = out_idx;
      size_t in_off = 0;
      for (size_t axis = 0; axis < rank; ++axis) {
        const int64_t stride = out_strides[axis];
        const int64_t coord = stride == 0 ? 0 : static_cast<int64_t>(rem / static_cast<size_t>(stride));
        rem = stride == 0 ? 0 : (rem % static_cast<size_t>(stride));
        const int64_t in_coord = starts_full[axis] + coord * steps_full[axis];
        in_off += static_cast<size_t>(in_coord * in_strides[axis]);
      }

      out[out_idx] = in[in_off];
    }

    return nullptr;
  }

  OrtStatus* ComputeReduceLikeCopy(Ort::KernelContext& context,
                                   const std::vector<TensorArg>& inputs,
                                   const std::vector<int64_t>& output_shape) const {
    if (inputs.empty()) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "missing input");
    }

    const TensorArg& x = inputs[0];
    Ort::UnownedValue output;
    RETURN_IF_ERROR(AllocateOutput(context, 0, output_shape, output));

    const size_t elem_size = GetElementSize(x.elem_type);
    if (elem_size == 0) {
      return MakeStatus(ort_api_, ORT_INVALID_ARGUMENT, "unsupported element type");
    }

    std::memcpy(output.GetTensorMutableRawData(), x.data, x.element_count * elem_size);
    return nullptr;
  }

  const OrtApi& ort_api_;
  const OrtLogger& logger_;
  TelumBackend& backend_;
  KernelConfig kernel_config_;
  OpKind op_kind_ = OpKind::kUnknown;
  std::string op_type_;
  std::vector<std::string> input_names_;
  KernelAttributes attributes_;
  const TensorInitializerMap& initializers_;
  bool drop_constant_initializers_ = true;
  int since_version_ = 0;
};

KernelAttributes ParseKernelAttributes(OpKind op_kind, const Ort::ConstNode& node) {
  KernelAttributes attrs{};

  switch (op_kind) {
    case OpKind::kGemm:
      TryParseScalarAttr(node, "alpha", attrs.alpha);
      TryParseScalarAttr(node, "beta", attrs.beta);
      TryParseScalarAttr(node, "transA", attrs.trans_a);
      TryParseScalarAttr(node, "transB", attrs.trans_b);
      break;
    case OpKind::kSoftmax:
      TryParseScalarAttr(node, "axis", attrs.axis);
      if (attrs.axis == -1 && node.GetSinceVersion() < 13) {
        attrs.axis = 1;
      }
      break;
    case OpKind::kLayerNormalization:
      TryParseScalarAttr(node, "axis", attrs.axis);
      TryParseScalarAttr(node, "epsilon", attrs.epsilon);
      break;
    case OpKind::kTranspose:
      attrs.has_perm = TryParseIntVectorAttr(node, "perm", attrs.perm);
      break;
    case OpKind::kSqueeze:
    case OpKind::kUnsqueeze:
      attrs.has_axes = TryParseIntVectorAttr(node, "axes", attrs.axes);
      break;
    case OpKind::kReduceMean:
      attrs.has_axes = TryParseIntVectorAttr(node, "axes", attrs.axes);
      attrs.keepdims = ParseBoolAttr(node, "keepdims", true);
      attrs.noop_with_empty_axes = ParseBoolAttr(node, "noop_with_empty_axes", false);
      break;
    case OpKind::kCast:
      TryParseScalarAttr(node, "to", attrs.cast_to);
      break;
    case OpKind::kConcat:
      TryParseScalarAttr(node, "axis", attrs.concat_axis);
      break;
    case OpKind::kGather:
      TryParseScalarAttr(node, "axis", attrs.axis);
      break;
    case OpKind::kReshape:
      attrs.allow_zero = ParseBoolAttr(node, "allowzero", false);
      break;
    default:
      break;
  }

  return attrs;
}

}  // namespace

size_t TensorInitializer::ElementCount() const noexcept {
  const auto count = TryComputeElementCount(shape);
  return count.value_or(0);
}

size_t TensorInitializer::ElementSize() const noexcept {
  return GetElementSize(elem_type);
}

bool TryGetOpKind(const std::string& op_type, const std::string& domain, OpKind& op_kind) {
  if (!domain.empty() && domain != "ai.onnx" && domain != "com.microsoft") {
    op_kind = OpKind::kUnknown;
    return false;
  }

  if (op_type == "MatMul") op_kind = OpKind::kMatMul;
  else if (op_type == "Gemm") op_kind = OpKind::kGemm;
  else if (op_type == "Add") op_kind = OpKind::kAdd;
  else if (op_type == "Sub") op_kind = OpKind::kSub;
  else if (op_type == "Mul") op_kind = OpKind::kMul;
  else if (op_type == "Div") op_kind = OpKind::kDiv;
  else if (op_type == "Min") op_kind = OpKind::kMin;
  else if (op_type == "Max") op_kind = OpKind::kMax;
  else if (op_type == "Relu") op_kind = OpKind::kRelu;
  else if (op_type == "Gelu" && domain == "com.microsoft") op_kind = OpKind::kGelu;
  else if (op_type == "Tanh") op_kind = OpKind::kTanh;
  else if (op_type == "Sigmoid") op_kind = OpKind::kSigmoid;
  else if (op_type == "Exp") op_kind = OpKind::kExp;
  else if (op_type == "Log") op_kind = OpKind::kLog;
  else if (op_type == "Sqrt") op_kind = OpKind::kSqrt;
  else if (op_type == "Softmax") op_kind = OpKind::kSoftmax;
  else if (op_type == "LayerNormalization") op_kind = OpKind::kLayerNormalization;
  else if (op_type == "Reshape") op_kind = OpKind::kReshape;
  else if (op_type == "Transpose") op_kind = OpKind::kTranspose;
  else if (op_type == "Squeeze") op_kind = OpKind::kSqueeze;
  else if (op_type == "Unsqueeze") op_kind = OpKind::kUnsqueeze;
  else if (op_type == "ReduceMean") op_kind = OpKind::kReduceMean;
  else if (op_type == "Cast") op_kind = OpKind::kCast;
  else if (op_type == "Where") op_kind = OpKind::kWhere;
  else if (op_type == "Expand") op_kind = OpKind::kExpand;
  else if (op_type == "Concat") op_kind = OpKind::kConcat;
  else if (op_type == "Gather") op_kind = OpKind::kGather;
  else if (op_type == "Slice") op_kind = OpKind::kSlice;
  else {
    op_kind = OpKind::kUnknown;
    return false;
  }

  return true;
}

std::string OpKindToString(OpKind op_kind) {
  switch (op_kind) {
    case OpKind::kMatMul: return "MatMul";
    case OpKind::kGemm: return "Gemm";
    case OpKind::kAdd: return "Add";
    case OpKind::kSub: return "Sub";
    case OpKind::kMul: return "Mul";
    case OpKind::kDiv: return "Div";
    case OpKind::kMin: return "Min";
    case OpKind::kMax: return "Max";
    case OpKind::kRelu: return "Relu";
    case OpKind::kGelu: return "Gelu";
    case OpKind::kTanh: return "Tanh";
    case OpKind::kSigmoid: return "Sigmoid";
    case OpKind::kExp: return "Exp";
    case OpKind::kLog: return "Log";
    case OpKind::kSqrt: return "Sqrt";
    case OpKind::kSoftmax: return "Softmax";
    case OpKind::kLayerNormalization: return "LayerNormalization";
    case OpKind::kReshape: return "Reshape";
    case OpKind::kTranspose: return "Transpose";
    case OpKind::kSqueeze: return "Squeeze";
    case OpKind::kUnsqueeze: return "Unsqueeze";
    case OpKind::kReduceMean: return "ReduceMean";
    case OpKind::kCast: return "Cast";
    case OpKind::kWhere: return "Where";
    case OpKind::kExpand: return "Expand";
    case OpKind::kConcat: return "Concat";
    case OpKind::kGather: return "Gather";
    case OpKind::kSlice: return "Slice";
    default: return "Unknown";
  }
}

bool OpUsesNnpaGating(OpKind op_kind) {
  switch (op_kind) {
    case OpKind::kMul:
    case OpKind::kRelu:
    case OpKind::kGelu:
    case OpKind::kTanh:
    case OpKind::kSigmoid:
    case OpKind::kExp:
    case OpKind::kLog:
    case OpKind::kSqrt:
    case OpKind::kSoftmax:
    case OpKind::kLayerNormalization:
    case OpKind::kMatMul:
    case OpKind::kGemm:
      return true;
    default:
      return false;
  }
}

OrtStatus* ConvertConstValueToInitializer(const OrtApi& ort_api,
                                          Ort::ConstValue value,
                                          TensorInitializer& initializer) {
  try {
    auto type_shape = value.GetTensorTypeAndShapeInfo();
    initializer.elem_type = type_shape.GetElementType();
    initializer.shape = type_shape.GetShape();

    const size_t elem_size = GetElementSize(initializer.elem_type);
    if (elem_size == 0) {
      return MakeStatus(ort_api, ORT_INVALID_ARGUMENT,
                        "Unsupported initializer type for Telum plugin");
    }

    const size_t count = type_shape.GetElementCount();
    initializer.raw_data.resize(count * elem_size);
    if (count == 0) {
      return nullptr;
    }

    const void* src = nullptr;
    switch (initializer.elem_type) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        src = value.GetTensorData<float>();
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        src = value.GetTensorData<bool>();
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        src = value.GetTensorData<int32_t>();
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        src = value.GetTensorData<int64_t>();
        break;
      default:
        return MakeStatus(ort_api, ORT_INVALID_ARGUMENT,
                          "Unsupported initializer element type for Telum plugin");
    }

    std::memcpy(initializer.raw_data.data(), src, initializer.raw_data.size());
    return nullptr;
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    return MakeStatus(ort_api, ORT_EP_FAIL, ex.what());
  }
}

std::unique_ptr<CompiledNodeKernel> CreateCompiledNodeKernel(const OrtApi& ort_api,
                                                             const OrtLogger& logger,
                                                             TelumBackend& backend,
                                                             const KernelConfig& kernel_config,
                                                             const Ort::ConstNode& node,
                                                             const TensorInitializerMap& initializers,
                                                             bool drop_constant_initializers,
                                                             OrtStatus*& error_status) {
  error_status = nullptr;

  try {
    const std::string op_type = node.GetOperatorType();
    const std::string domain = node.GetDomain();
    OpKind op_kind = OpKind::kUnknown;
    if (!TryGetOpKind(op_type, domain, op_kind)) {
      error_status = MakeStatus(ort_api, ORT_INVALID_ARGUMENT,
                                "Unsupported Telum op in CreateCompiledNodeKernel: " + op_type +
                                    " (domain=" + domain + ")");
      return nullptr;
    }

    std::vector<Ort::ConstValueInfo> node_inputs = node.GetInputs();
    std::vector<std::string> input_names;
    input_names.reserve(node_inputs.size());
    for (const auto& input : node_inputs) {
      input_names.push_back(input.GetName());
    }

    KernelAttributes attrs = ParseKernelAttributes(op_kind, node);

    return std::make_unique<GenericNodeKernel>(
        ort_api, logger, backend, kernel_config,
        op_kind, op_type, std::move(input_names), std::move(attrs),
        initializers, drop_constant_initializers, node.GetSinceVersion());

  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    error_status = status.release();
    return nullptr;
  } catch (const std::exception& ex) {
    error_status = MakeStatus(ort_api, ORT_EP_FAIL, ex.what());
    return nullptr;
  }
}

std::unique_ptr<CompiledNodeKernel> CreateCompiledNodeKernelFromEpContext(const OrtApi& ort_api,
                                                                          const OrtLogger& logger,
                                                                          TelumBackend& backend,
                                                                          const KernelConfig& kernel_config,
                                                                          const std::string& op_type,
                                                                          const std::string& attributes_blob,
                                                                          const std::vector<std::string>& input_names,
                                                                          const TensorInitializerMap& initializers,
                                                                          bool drop_constant_initializers,
                                                                          OrtStatus*& error_status) {
  error_status = nullptr;

  try {
    OpKind op_kind = OpKind::kUnknown;
    if (!TryGetOpKind(op_type, "ai.onnx", op_kind) &&
        !TryGetOpKind(op_type, "com.microsoft", op_kind)) {
      error_status = MakeStatus(ort_api, ORT_INVALID_ARGUMENT,
                                "Unsupported EPContext op_type for Telum plugin: " + op_type);
      return nullptr;
    }

    KernelAttributes attrs{};
    if (!ParseKernelAttributesBlob(attributes_blob, attrs)) {
      error_status = MakeStatus(ort_api, ORT_INVALID_ARGUMENT,
                                "Failed to parse EPContext attributes blob for op " + op_type);
      return nullptr;
    }

    return std::make_unique<GenericNodeKernel>(
        ort_api, logger, backend, kernel_config,
        op_kind, op_type, input_names, std::move(attrs),
        initializers, drop_constant_initializers, /*since_version*/ 0);
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    error_status = status.release();
    return nullptr;
  } catch (const std::exception& ex) {
    error_status = MakeStatus(ort_api, ORT_EP_FAIL, ex.what());
    return nullptr;
  }
}

}  // namespace telum
