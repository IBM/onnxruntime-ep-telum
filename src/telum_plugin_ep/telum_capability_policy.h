// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

#include "../plugin_ep_utils.h"
#include "kernels/op_kernel.h"
#include "telum_backend.h"

enum class TelumCapabilityNodeDisposition {
  kUnsupported = 0,
  kFuseNode,
  kSingleNode,
};

enum class TelumCapabilityRejectReason {
  kNone = 0,
  kUnsupportedOp,
  kNonTensorOrTypeMismatch,
  kShapeOrDynamic,
  kOpConstraint,
  kEpContextSourceMismatch,
  kBackendCapabilityUnavailable,
};

struct TelumCapabilityNodeDecision {
  TelumCapabilityNodeDisposition disposition = TelumCapabilityNodeDisposition::kUnsupported;
  TelumCapabilityRejectReason reject_reason = TelumCapabilityRejectReason::kNone;
  std::string reject_detail;
  std::string error;
  telum::OpKind op_kind = telum::OpKind::kUnknown;
};

struct TelumCapabilityStats {
  size_t num_nodes_considered = 0;
  size_t num_rejected_unsupported_op = 0;
  size_t num_rejected_non_tensor_or_type = 0;
  size_t num_rejected_shape_or_dynamic = 0;
  size_t num_rejected_op_constraint = 0;
  size_t num_rejected_epcontext_source_mismatch = 0;
  size_t num_rejected_backend_capability = 0;
};

TelumCapabilityNodeDecision EvaluateTelumCapabilityNode(const OrtApi& ort_api,
                                                        const Ort::ConstNode& node,
                                                        const std::string& ep_name,
                                                        const TelumBackend& backend,
                                                        bool strict_mode);
void RecordTelumCapabilityDecision(const TelumCapabilityNodeDecision& decision,
                                   TelumCapabilityStats& stats);

