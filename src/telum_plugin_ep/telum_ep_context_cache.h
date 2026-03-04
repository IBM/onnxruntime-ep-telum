// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>

#include "kernels/op_kernel.h"

namespace telum_ep_context {

struct EpCacheContext {
  std::string op_type;
  std::string attributes_blob;
  std::vector<std::string> input_names;
  telum::TensorInitializerMap initializers;
};

std::string SerializeEpCacheContext(const std::string& op_type,
                                    const std::string& attributes_blob,
                                    gsl::span<const std::string> input_names,
                                    const telum::TensorInitializerMap& initializers);

OrtStatus* ParseEpCacheContext(const OrtApi& ort_api,
                               const std::string& serialized,
                               EpCacheContext& parsed);

}  // namespace telum_ep_context
