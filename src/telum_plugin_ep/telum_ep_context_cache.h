// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <unordered_map>

#include "../plugin_ep_utils.h"

namespace telum_ep_context {

struct MulEpCacheContext {
  std::string input0_name;
  std::string input1_name;
  std::unordered_map<std::string, FloatInitializer> float_initializers;
};

std::string SerializeMulEpCacheContext(
    const std::string& input0_name,
    const std::string& input1_name,
    const std::unordered_map<std::string, FloatInitializer>& float_initializers);

OrtStatus* ParseMulEpCacheContext(const OrtApi& ort_api,
                                  const std::string& serialized,
                                  MulEpCacheContext& parsed);

}  // namespace telum_ep_context

