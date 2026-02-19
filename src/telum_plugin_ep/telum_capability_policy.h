// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

#include "../plugin_ep_utils.h"
#include "telum_backend.h"

enum class TelumCapabilityNodeDisposition {
  kUnsupported = 0,
  kFuseNode,
  kSingleNode,
};

enum class TelumCapabilityRejectReason {
  kNone = 0,
  kUnsupportedOp,
  kNonFloatType,
  kShapeOrDynamic,
  kEpContextSourceMismatch,
  kBackendCapabilityUnavailable,
};

struct TelumCapabilityNodeDecision {
  TelumCapabilityNodeDisposition disposition = TelumCapabilityNodeDisposition::kUnsupported;
  TelumCapabilityRejectReason reject_reason = TelumCapabilityRejectReason::kNone;
  std::string error;
};

struct TelumCapabilityStats {
  size_t num_nodes_considered = 0;
  size_t num_rejected_unsupported_op = 0;
  size_t num_rejected_non_float = 0;
  size_t num_rejected_shape_or_dynamic = 0;
  size_t num_rejected_epcontext_source_mismatch = 0;
  size_t num_rejected_backend_capability = 0;

  size_t NumSupported() const {
    return num_nodes_considered - num_rejected_unsupported_op - num_rejected_non_float -
           num_rejected_shape_or_dynamic - num_rejected_epcontext_source_mismatch -
           num_rejected_backend_capability;
  }
};

TelumCapabilityNodeDecision EvaluateTelumCapabilityNode(const OrtApi& ort_api,
                                                        const Ort::ConstNode& node,
                                                        const std::string& ep_name,
                                                        const TelumBackend& backend);
void RecordTelumCapabilityDecision(const TelumCapabilityNodeDecision& decision,
                                   TelumCapabilityStats& stats);
