// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "telum_capability_policy.h"

#include <algorithm>
#include <array>
#include <limits>
#include <set>
#include <sstream>
#include <utility>
#include <vector>

namespace {

bool TryGetTensorShapeAndType(Ort::ConstValueInfo value_info,
                              ONNXTensorElementDataType& elem_type,
                              std::vector<int64_t>& shape,
                              bool& is_tensor) {
  is_tensor = false;
  elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  shape.clear();

  auto type_info = value_info.TypeInfo();
  if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
    return true;
  }

  auto tensor_type = type_info.GetTensorTypeAndShapeInfo();
  elem_type = tensor_type.GetElementType();
  shape = tensor_type.GetShape();
  is_tensor = true;
  return true;
}

bool IsStaticShape(gsl::span<const int64_t> shape) {
  return std::all_of(shape.begin(), shape.end(), [](int64_t d) { return d >= 0; });
}

bool IsSupportedCastType(ONNXTensorElementDataType type) {
  return type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
         type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 ||
         type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ||
         type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
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

bool ComputeBroadcastShape(const std::vector<std::vector<int64_t>>& input_shapes,
                           std::vector<int64_t>& output_shape) {
  output_shape.clear();

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
        return false;
      }
      if (cur == 1) {
        continue;
      }
      if (dim == 1 || dim == cur) {
        dim = cur;
      } else {
        return false;
      }
    }
    output_shape[axis] = dim;
  }

  return true;
}

int64_t NormalizeAxis(int64_t axis, size_t rank, bool allow_end = false) {
  const int64_t rank_i64 = static_cast<int64_t>(rank);
  if (axis < 0) {
    axis += rank_i64;
  }
  const int64_t max_axis = allow_end ? rank_i64 : (rank_i64 - 1);
  if (axis < 0 || axis > max_axis) {
    return std::numeric_limits<int64_t>::min();
  }

  return axis;
}

TelumCapabilityNodeDecision Reject(TelumCapabilityRejectReason reason, std::string detail) {
  TelumCapabilityNodeDecision decision;
  decision.reject_reason = reason;
  decision.reject_detail = std::move(detail);
  return decision;
}

bool IsTensorStaticAndType(Ort::ConstValueInfo value_info,
                           ONNXTensorElementDataType expected_type,
                           std::string& error) {
  ONNXTensorElementDataType actual = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  std::vector<int64_t> shape;
  bool is_tensor = false;
  TryGetTensorShapeAndType(value_info, actual, shape, is_tensor);
  if (!is_tensor) {
    error = "expected tensor";
    return false;
  }

  if (!IsStaticShape(shape)) {
    error = "requires static shape";
    return false;
  }

  if (actual != expected_type) {
    std::ostringstream os;
    os << "expected type " << static_cast<int>(expected_type)
       << ", got " << static_cast<int>(actual);
    error = os.str();
    return false;
  }

  return true;
}

bool CheckDefaultFloatInputsOutputs(const std::vector<Ort::ConstValueInfo>& inputs,
                                    const std::vector<Ort::ConstValueInfo>& outputs,
                                    std::string& error) {
  for (const auto& input : inputs) {
    if (input.GetName().empty()) {
      continue;
    }
    if (!IsTensorStaticAndType(input, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, error)) {
      return false;
    }
  }

  for (const auto& output : outputs) {
    if (!IsTensorStaticAndType(output, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, error)) {
      return false;
    }
  }

  return true;
}

}  // namespace

TelumCapabilityNodeDecision EvaluateTelumCapabilityNode(const OrtApi& ort_api,
                                                        const Ort::ConstNode& node,
                                                        const std::string& ep_name,
                                                        const TelumBackend& backend,
                                                        bool /*strict_mode*/) {
  TelumCapabilityNodeDecision decision{};

  const OrtNode* node_ptr = node;
  const char* op_type_c = nullptr;
  const char* domain_c = nullptr;
  if (OrtStatus* st = ort_api.Node_GetOperatorType(node_ptr, &op_type_c); st != nullptr) {
    Ort::Status status(st);
    decision.error = status.GetErrorMessage();
    return decision;
  }
  if (OrtStatus* st = ort_api.Node_GetDomain(node_ptr, &domain_c); st != nullptr) {
    Ort::Status status(st);
    decision.error = status.GetErrorMessage();
    return decision;
  }

  const std::string op_type = op_type_c ? op_type_c : "";
  const std::string domain = domain_c ? domain_c : "";

  // EPContext compatibility path.
  if (op_type == "EPContext" && domain == "com.microsoft") {
    Ort::ConstOpAttr source_attr;
    Ort::Status status = node.GetAttributeByName("source", source_attr);
    if (!status.IsOK()) {
      return Reject(TelumCapabilityRejectReason::kEpContextSourceMismatch,
                    "EPContext node is missing source attribute");
    }

    std::string source;
    status = source_attr.GetValue(source);
    if (!status.IsOK() || source != ep_name) {
      return Reject(TelumCapabilityRejectReason::kEpContextSourceMismatch,
                    "EPContext source does not match this EP");
    }

    decision.disposition = TelumCapabilityNodeDisposition::kFuseNode;
    return decision;
  }

  telum::OpKind op_kind = telum::OpKind::kUnknown;
  if (!telum::TryGetOpKind(op_type, domain, op_kind)) {
    return Reject(TelumCapabilityRejectReason::kUnsupportedOp,
                  "Operator not supported by Telum plugin");
  }

  decision.op_kind = op_kind;

  if (telum::OpUsesNnpaGating(op_kind)) {
    if (!backend.SupportsOp(op_kind)) {
      return Reject(TelumCapabilityRejectReason::kBackendCapabilityUnavailable,
                    "NNPA/zDNN capability unavailable for op " + op_type);
    }
  }

  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();

  auto reject_type = [&](const std::string& reason) {
    return Reject(TelumCapabilityRejectReason::kNonTensorOrTypeMismatch, reason);
  };

  auto reject_shape = [&](const std::string& reason) {
    return Reject(TelumCapabilityRejectReason::kShapeOrDynamic, reason);
  };

  auto reject_constraint = [&](const std::string& reason) {
    return Reject(TelumCapabilityRejectReason::kOpConstraint, reason);
  };

  std::string type_error;

  switch (op_kind) {
    case telum::OpKind::kMatMul: {
      if (inputs.size() != 2 || outputs.size() != 1) {
        return reject_constraint("MatMul requires 2 inputs and 1 output");
      }
      if (!CheckDefaultFloatInputsOutputs(inputs, outputs, type_error)) {
        return reject_type(type_error);
      }

      auto a_shape = GetTensorShape(inputs[0]);
      auto b_shape = GetTensorShape(inputs[1]);
      if (!a_shape.has_value() || !b_shape.has_value() || !IsStaticShape(*a_shape) || !IsStaticShape(*b_shape)) {
        return reject_shape("MatMul requires static input shapes");
      }
      if (a_shape->size() < 2 || b_shape->size() < 2) {
        return reject_constraint("MatMul requires rank >= 2");
      }
      if ((*a_shape)[a_shape->size() - 1] != (*b_shape)[b_shape->size() - 2]) {
        return reject_constraint("MatMul K dimensions must match");
      }
      std::vector<int64_t> out_batch;
      std::vector<int64_t> a_batch(a_shape->begin(), a_shape->end() - 2);
      std::vector<int64_t> b_batch(b_shape->begin(), b_shape->end() - 2);
      if (!ComputeBroadcastShape({a_batch, b_batch}, out_batch)) {
        return reject_constraint("MatMul batch dimensions are not broadcast-compatible");
      }
      break;
    }

    case telum::OpKind::kGemm: {
      if (inputs.size() < 2 || outputs.size() != 1) {
        return reject_constraint("Gemm requires at least 2 inputs and 1 output");
      }

      ONNXTensorElementDataType t{};
      std::vector<int64_t> s;
      bool is_tensor = false;
      TryGetTensorShapeAndType(inputs[0], t, s, is_tensor);
      if (!is_tensor || t != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || !IsStaticShape(s) || s.size() != 2) {
        return reject_type("Gemm input A must be static 2D float tensor");
      }
      TryGetTensorShapeAndType(inputs[1], t, s, is_tensor);
      if (!is_tensor || t != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || !IsStaticShape(s) || s.size() != 2) {
        return reject_type("Gemm input B must be static 2D float tensor");
      }
      if (!CheckDefaultFloatInputsOutputs({inputs[0], inputs[1]}, outputs, type_error)) {
        return reject_type(type_error);
      }
      if (inputs.size() > 2 && !inputs[2].GetName().empty()) {
        TryGetTensorShapeAndType(inputs[2], t, s, is_tensor);
        if (!is_tensor || t != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || !IsStaticShape(s)) {
          return reject_type("Gemm input C must be static float tensor");
        }
      }
      break;
    }

    case telum::OpKind::kAdd:
    case telum::OpKind::kSub:
    case telum::OpKind::kMul:
    case telum::OpKind::kDiv:
    case telum::OpKind::kMin:
    case telum::OpKind::kMax: {
      if (inputs.size() != 2 || outputs.size() != 1) {
        return reject_constraint("Elementwise op requires 2 inputs and 1 output");
      }
      if (!CheckDefaultFloatInputsOutputs(inputs, outputs, type_error)) {
        return reject_type(type_error);
      }
      auto a_shape = GetTensorShape(inputs[0]);
      auto b_shape = GetTensorShape(inputs[1]);
      if (!a_shape.has_value() || !b_shape.has_value() || !IsStaticShape(*a_shape) || !IsStaticShape(*b_shape)) {
        return reject_shape("Elementwise op requires static input shapes");
      }
      std::vector<int64_t> out_shape;
      if (!ComputeBroadcastShape({*a_shape, *b_shape}, out_shape)) {
        return reject_constraint("Elementwise op inputs are not ONNX-broadcast-compatible");
      }
      if (out_shape.size() > 4) {
        return reject_constraint("Elementwise op output rank > 4 is not supported");
      }
      break;
    }

    case telum::OpKind::kRelu:
    case telum::OpKind::kGelu:
    case telum::OpKind::kTanh:
    case telum::OpKind::kSigmoid:
    case telum::OpKind::kExp:
    case telum::OpKind::kLog:
    case telum::OpKind::kSqrt:
    case telum::OpKind::kSoftmax: {
      if (inputs.size() != 1 || outputs.size() != 1) {
        return reject_constraint("Unary activation op requires 1 input and 1 output");
      }
      if (!CheckDefaultFloatInputsOutputs(inputs, outputs, type_error)) {
        return reject_type(type_error);
      }
      if (op_kind == telum::OpKind::kSoftmax) {
        int64_t axis = -1;
        TryParseScalarAttr(node, "axis", axis);
        const auto shape = GetTensorShape(inputs[0]);
        if (!shape.has_value() || shape->empty()) {
          return reject_shape("Softmax requires static rank >= 1");
        }
        const int64_t normalized = NormalizeAxis(axis, shape->size());
        if (normalized != static_cast<int64_t>(shape->size() - 1)) {
          return reject_constraint("Softmax currently supports axis on the last dimension only");
        }
      }
      break;
    }

    case telum::OpKind::kLayerNormalization: {
      if (inputs.size() < 2 || outputs.empty()) {
        return reject_constraint("LayerNormalization requires at least 2 inputs");
      }
      ONNXTensorElementDataType t{};
      std::vector<int64_t> shape;
      bool is_tensor = false;
      for (size_t i = 0; i < std::min<size_t>(inputs.size(), 3); ++i) {
        if (inputs[i].GetName().empty()) continue;
        TryGetTensorShapeAndType(inputs[i], t, shape, is_tensor);
        if (!is_tensor || t != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || !IsStaticShape(shape)) {
          return reject_type("LayerNormalization inputs must be static float tensors");
        }
      }
      for (const auto& output : outputs) {
        TryGetTensorShapeAndType(output, t, shape, is_tensor);
        if (!is_tensor || t != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || !IsStaticShape(shape)) {
          return reject_type("LayerNormalization outputs must be static float tensors");
        }
      }

      int64_t axis = -1;
      TryParseScalarAttr(node, "axis", axis);
      auto x_shape = GetTensorShape(inputs[0]);
      if (!x_shape.has_value() || x_shape->empty()) {
        return reject_shape("LayerNormalization requires rank >= 1");
      }
      const int64_t normalized = NormalizeAxis(axis, x_shape->size());
      if (normalized != static_cast<int64_t>(x_shape->size() - 1)) {
        return reject_constraint("LayerNormalization currently supports last-dimension axis only");
      }
      break;
    }

    case telum::OpKind::kReshape:
    case telum::OpKind::kExpand: {
      if (inputs.size() < 2 || outputs.size() != 1) {
        return reject_constraint(op_kind == telum::OpKind::kReshape ?
                                 "Reshape requires 2 inputs" : "Expand requires 2 inputs");
      }
      if (!IsTensorStaticAndType(inputs[0], ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, type_error)) {
        return reject_type(type_error);
      }
      ONNXTensorElementDataType t{};
      std::vector<int64_t> s;
      bool is_tensor = false;
      TryGetTensorShapeAndType(inputs[1], t, s, is_tensor);
      if (!is_tensor || !IsStaticShape(s) ||
          (t != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 && t != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)) {
        return reject_type("Shape input must be static int32/int64 tensor");
      }
      if (!IsTensorStaticAndType(outputs[0], ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, type_error)) {
        return reject_type(type_error);
      }
      break;
    }

    case telum::OpKind::kTranspose: {
      if (inputs.size() != 1 || outputs.size() != 1) {
        return reject_constraint("Transpose requires 1 input and 1 output");
      }
      if (!CheckDefaultFloatInputsOutputs(inputs, outputs, type_error)) {
        return reject_type(type_error);
      }
      auto in_shape = GetTensorShape(inputs[0]);
      if (!in_shape.has_value() || !IsStaticShape(*in_shape)) {
        return reject_shape("Transpose requires static input shape");
      }
      std::vector<int64_t> perm;
      if (TryParseIntVectorAttr(node, "perm", perm)) {
        if (perm.size() != in_shape->size()) {
          return reject_constraint("Transpose perm length must equal input rank");
        }
        std::set<int64_t> seen;
        for (int64_t axis : perm) {
          if (axis < 0 || static_cast<size_t>(axis) >= in_shape->size() || !seen.insert(axis).second) {
            return reject_constraint("Transpose perm must be a full non-duplicated permutation");
          }
        }
      }
      break;
    }

    case telum::OpKind::kSqueeze:
    case telum::OpKind::kUnsqueeze: {
      if (inputs.empty() || outputs.size() != 1) {
        return reject_constraint("Squeeze/Unsqueeze requires at least input X and 1 output");
      }
      if (!IsTensorStaticAndType(inputs[0], ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, type_error) ||
          !IsTensorStaticAndType(outputs[0], ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, type_error)) {
        return reject_type(type_error);
      }
      if (inputs.size() > 1 && !inputs[1].GetName().empty()) {
        ONNXTensorElementDataType t{};
        std::vector<int64_t> s;
        bool is_tensor = false;
        TryGetTensorShapeAndType(inputs[1], t, s, is_tensor);
        if (!is_tensor || !IsStaticShape(s) ||
            (t != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 && t != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)) {
          return reject_type("Squeeze/Unsqueeze axes input must be static int tensor");
        }
      }
      break;
    }

    case telum::OpKind::kReduceMean: {
      if (inputs.empty() || outputs.size() != 1) {
        return reject_constraint("ReduceMean requires input X and one output");
      }
      if (!IsTensorStaticAndType(inputs[0], ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, type_error) ||
          !IsTensorStaticAndType(outputs[0], ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, type_error)) {
        return reject_type(type_error);
      }
      if (inputs.size() > 1 && !inputs[1].GetName().empty()) {
        ONNXTensorElementDataType t{};
        std::vector<int64_t> s;
        bool is_tensor = false;
        TryGetTensorShapeAndType(inputs[1], t, s, is_tensor);
        if (!is_tensor || !IsStaticShape(s) ||
            (t != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 && t != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)) {
          return reject_type("ReduceMean axes input must be static int tensor");
        }
      }
      break;
    }

    case telum::OpKind::kCast: {
      if (inputs.size() != 1 || outputs.size() != 1) {
        return reject_constraint("Cast requires 1 input and 1 output");
      }
      ONNXTensorElementDataType in_t{};
      ONNXTensorElementDataType out_t{};
      std::vector<int64_t> shape;
      bool is_tensor = false;
      TryGetTensorShapeAndType(inputs[0], in_t, shape, is_tensor);
      if (!is_tensor || !IsStaticShape(shape) || !IsSupportedCastType(in_t)) {
        return reject_type("Cast input type is unsupported");
      }
      TryGetTensorShapeAndType(outputs[0], out_t, shape, is_tensor);
      if (!is_tensor || !IsStaticShape(shape) || !IsSupportedCastType(out_t)) {
        return reject_type("Cast output type is unsupported");
      }
      break;
    }

    case telum::OpKind::kWhere: {
      if (inputs.size() != 3 || outputs.size() != 1) {
        return reject_constraint("Where requires 3 inputs and 1 output");
      }
      if (!IsTensorStaticAndType(inputs[0], ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, type_error)) {
        return reject_type("Where condition must be static bool tensor");
      }
      if (!IsTensorStaticAndType(inputs[1], ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, type_error) ||
          !IsTensorStaticAndType(inputs[2], ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, type_error) ||
          !IsTensorStaticAndType(outputs[0], ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, type_error)) {
        return reject_type("Where data tensors must be static float");
      }
      auto c = GetTensorShape(inputs[0]);
      auto x = GetTensorShape(inputs[1]);
      auto y = GetTensorShape(inputs[2]);
      std::vector<int64_t> out;
      if (!c.has_value() || !x.has_value() || !y.has_value() ||
          !ComputeBroadcastShape({*c, *x, *y}, out) || out.size() > 4) {
        return reject_constraint("Where inputs must be broadcast-compatible with rank <= 4");
      }
      break;
    }

    case telum::OpKind::kConcat: {
      if (inputs.empty() || outputs.size() != 1) {
        return reject_constraint("Concat requires at least 1 input and 1 output");
      }
      int64_t axis = 0;
      TryParseScalarAttr(node, "axis", axis);
      std::vector<int64_t> base_shape;
      bool have_shape = false;
      for (const auto& input : inputs) {
        if (input.GetName().empty()) {
          continue;
        }
        if (!IsTensorStaticAndType(input, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, type_error)) {
          return reject_type(type_error);
        }
        auto shape = GetTensorShape(input);
        if (!shape.has_value() || !IsStaticShape(*shape)) {
          return reject_shape("Concat requires static input shapes");
        }
        if (!have_shape) {
          base_shape = *shape;
          have_shape = true;
          continue;
        }
        if (shape->size() != base_shape.size()) {
          return reject_constraint("Concat input ranks must match");
        }
      }
      if (!have_shape) {
        return reject_constraint("Concat has no real inputs");
      }
      const int64_t normalized_axis = NormalizeAxis(axis, base_shape.size());
      if (normalized_axis == std::numeric_limits<int64_t>::min()) {
        return reject_constraint("Concat axis is out of range");
      }
      for (const auto& input : inputs) {
        if (input.GetName().empty()) continue;
        auto shape = GetTensorShape(input);
        for (size_t i = 0; i < shape->size(); ++i) {
          if (static_cast<int64_t>(i) == normalized_axis) continue;
          if ((*shape)[i] != base_shape[i]) {
            return reject_constraint("Concat non-axis dimensions must match");
          }
        }
      }
      if (!IsTensorStaticAndType(outputs[0], ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, type_error)) {
        return reject_type(type_error);
      }
      break;
    }

    case telum::OpKind::kGather: {
      if (inputs.size() != 2 || outputs.size() != 1) {
        return reject_constraint("Gather requires 2 inputs and 1 output");
      }
      if (!IsTensorStaticAndType(inputs[0], ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, type_error)) {
        return reject_type("Gather data must be static float tensor");
      }
      ONNXTensorElementDataType idx_t{};
      std::vector<int64_t> idx_shape;
      bool is_tensor = false;
      TryGetTensorShapeAndType(inputs[1], idx_t, idx_shape, is_tensor);
      if (!is_tensor || !IsStaticShape(idx_shape) ||
          (idx_t != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 && idx_t != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)) {
        return reject_type("Gather indices must be static int32/int64 tensor");
      }
      if (!IsTensorStaticAndType(outputs[0], ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, type_error)) {
        return reject_type(type_error);
      }
      break;
    }

    case telum::OpKind::kSlice: {
      if (inputs.size() < 3 || outputs.size() != 1) {
        return reject_constraint("Slice requires at least 3 inputs and 1 output");
      }
      if (!IsTensorStaticAndType(inputs[0], ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, type_error) ||
          !IsTensorStaticAndType(outputs[0], ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, type_error)) {
        return reject_type(type_error);
      }
      for (size_t i = 1; i < std::min<size_t>(inputs.size(), 5); ++i) {
        if (inputs[i].GetName().empty()) continue;
        ONNXTensorElementDataType t{};
        std::vector<int64_t> s;
        bool is_tensor = false;
        TryGetTensorShapeAndType(inputs[i], t, s, is_tensor);
        if (!is_tensor || !IsStaticShape(s) ||
            (t != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 && t != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)) {
          return reject_type("Slice starts/ends/axes/steps inputs must be static int tensors");
        }
      }
      break;
    }

    default:
      return reject_constraint("Unhandled op in capability policy");
  }

  decision.disposition = TelumCapabilityNodeDisposition::kFuseNode;
  return decision;
}

void RecordTelumCapabilityDecision(const TelumCapabilityNodeDecision& decision,
                                   TelumCapabilityStats& stats) {
  ++stats.num_nodes_considered;
  switch (decision.reject_reason) {
    case TelumCapabilityRejectReason::kNone:
      break;
    case TelumCapabilityRejectReason::kUnsupportedOp:
      ++stats.num_rejected_unsupported_op;
      break;
    case TelumCapabilityRejectReason::kNonTensorOrTypeMismatch:
      ++stats.num_rejected_non_tensor_or_type;
      break;
    case TelumCapabilityRejectReason::kShapeOrDynamic:
      ++stats.num_rejected_shape_or_dynamic;
      break;
    case TelumCapabilityRejectReason::kOpConstraint:
      ++stats.num_rejected_op_constraint;
      break;
    case TelumCapabilityRejectReason::kEpContextSourceMismatch:
      ++stats.num_rejected_epcontext_source_mismatch;
      break;
    case TelumCapabilityRejectReason::kBackendCapabilityUnavailable:
      ++stats.num_rejected_backend_capability;
      break;
  }
}
