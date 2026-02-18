// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "telum_capability_policy.h"

#include <array>
#include <cstring>

TelumCapabilityNodeDecision EvaluateTelumCapabilityNode(const OrtApi& ort_api,
                                                        const Ort::ConstNode& node,
                                                        const std::string& ep_name,
                                                        const TelumBackend& backend) {
  TelumCapabilityNodeDecision decision{};

  const OrtNode* node_ptr = node;
  const char* op_type = nullptr;
  const char* domain = nullptr;
  if (OrtStatus* st = ort_api.Node_GetOperatorType(node_ptr, &op_type); st != nullptr) {
    Ort::Status status(st);
    decision.error = status.GetErrorMessage();
    return decision;
  }
  if (OrtStatus* st = ort_api.Node_GetDomain(node_ptr, &domain); st != nullptr) {
    Ort::Status status(st);
    decision.error = status.GetErrorMessage();
    return decision;
  }

  // EPContext node support for pre-compiled model loading.
  if (std::strcmp(op_type, "EPContext") == 0 && std::strcmp(domain, "com.microsoft") == 0) {
    Ort::ConstOpAttr source_attr;
    Ort::Status status = node.GetAttributeByName("source", source_attr);
    if (status.IsOK()) {
      std::string source_value;
      status = source_attr.GetValue(source_value);
      if (status.IsOK() && source_value == ep_name) {
        decision.disposition = TelumCapabilityNodeDisposition::kFuseNode;
      } else {
        decision.reject_reason = TelumCapabilityRejectReason::kEpContextSourceMismatch;
      }
    } else {
      decision.reject_reason = TelumCapabilityRejectReason::kEpContextSourceMismatch;
    }
    return decision;
  }

  if (std::strcmp(op_type, "Mul") == 0) {
    const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
    const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
    if (inputs.size() != 2 || outputs.size() != 1) {
      decision.error = "Mul should have 2 inputs and 1 output";
      return decision;
    }

    std::array<bool, 3> is_float = {false, false, false};
    IsFloatTensor(inputs[0], is_float[0]);
    IsFloatTensor(inputs[1], is_float[1]);
    IsFloatTensor(outputs[0], is_float[2]);
    if (!is_float[0] || !is_float[1] || !is_float[2]) {
      decision.reject_reason = TelumCapabilityRejectReason::kNonFloatType;
      return decision;
    }

    const auto input_0_shape = GetTensorShape(inputs[0]);
    const auto input_1_shape = GetTensorShape(inputs[1]);
    if (!input_0_shape.has_value() || !input_1_shape.has_value()) {
      decision.reject_reason = TelumCapabilityRejectReason::kShapeOrDynamic;
      return decision;
    }

    // Current scaffold supports static equal-shape Mul only.
    if (!AreShapesStaticAndEqual(*input_0_shape, *input_1_shape)) {
      decision.reject_reason = TelumCapabilityRejectReason::kShapeOrDynamic;
      return decision;
    }

    if (!backend.SupportsMul()) {
      decision.reject_reason = TelumCapabilityRejectReason::kBackendCapabilityUnavailable;
      return decision;
    }

    decision.disposition = TelumCapabilityNodeDisposition::kFuseNode;
    return decision;
  }

  if (std::strcmp(op_type, "Custom_Mul") == 0 && std::strcmp(domain, "test") == 0) {
    decision.disposition = TelumCapabilityNodeDisposition::kSingleNode;
    return decision;
  }

  decision.reject_reason = TelumCapabilityRejectReason::kUnsupportedOp;
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
    case TelumCapabilityRejectReason::kNonFloatType:
      ++stats.num_rejected_non_float;
      break;
    case TelumCapabilityRejectReason::kShapeOrDynamic:
      ++stats.num_rejected_shape_or_dynamic;
      break;
    case TelumCapabilityRejectReason::kEpContextSourceMismatch:
      ++stats.num_rejected_epcontext_source_mismatch;
      break;
    case TelumCapabilityRejectReason::kBackendCapabilityUnavailable:
      ++stats.num_rejected_backend_capability;
      break;
  }
}
