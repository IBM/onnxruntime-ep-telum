// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <optional>
#include <string>

#include "telum_backend.h"

namespace telum_compat {

constexpr const char* kFieldVersion = "version";
constexpr const char* kFieldOrtApiVersion = "ort_api_version";
constexpr const char* kFieldBackend = "backend";
constexpr const char* kFieldStubSupportMul = "stub_support_mul";

struct Info {
  std::string ep_name;
  std::optional<std::string> ep_version;
  std::optional<std::string> ort_api_version;
  std::optional<std::string> backend_kind;
  std::optional<std::string> stub_support_mul;
};

std::string BuildCompatibilityInfo(const std::string& ep_name,
                                   const std::string& ep_version,
                                   int ort_api_version,
                                   const TelumBackendConfig& backend_config);

bool TryParseCompatibilityInfo(const std::string& raw_info,
                               Info& parsed_info,
                               std::string& error);

bool TryParseBoolToken(const std::string& raw_value, bool& parsed_value);

}  // namespace telum_compat
