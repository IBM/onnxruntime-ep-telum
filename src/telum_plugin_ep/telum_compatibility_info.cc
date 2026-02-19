// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "telum_compatibility_info.h"

#include <cctype>
#include <unordered_map>
#include <string_view>
#include <vector>

#include "../plugin_ep_utils.h"

namespace {

std::vector<std::string> Split(const std::string& value, char delimiter) {
  std::vector<std::string> parts;
  size_t start = 0;
  while (start <= value.size()) {
    const size_t end = value.find(delimiter, start);
    if (end == std::string::npos) {
      parts.push_back(value.substr(start));
      break;
    }
    parts.push_back(value.substr(start, end - start));
    start = end + 1;
  }
  return parts;
}

bool EqualsIgnoreCase(std::string_view value, std::string_view expected_lowercase) {
  if (value.size() != expected_lowercase.size()) {
    return false;
  }

  for (size_t i = 0; i < value.size(); ++i) {
    const unsigned char c = static_cast<unsigned char>(value[i]);
    const char lower = static_cast<char>(std::tolower(c));
    if (lower != expected_lowercase[i]) {
      return false;
    }
  }

  return true;
}

}  // namespace

namespace telum_compat {

std::string BuildCompatibilityInfo(const std::string& ep_name,
                                   const std::string& ep_version,
                                   int ort_api_version,
                                   const TelumBackendConfig& backend_config) {
  return ep_name + ";" + kFieldVersion + "=" + ep_version + ";" + kFieldOrtApiVersion + "=" +
         std::to_string(ort_api_version) + ";" + kFieldBackend + "=" + backend_config.backend_kind + ";" +
         kFieldStubSupportMul + "=" + (backend_config.stub_support_mul ? "1" : "0");
}

bool TryParseCompatibilityInfo(const std::string& raw_info,
                               Info& parsed_info,
                               std::string& error) {
  parsed_info = {};
  error.clear();

  if (raw_info.empty()) {
    error = "compatibility info is empty";
    return false;
  }

  const auto tokens = Split(raw_info, ';');
  if (tokens.empty() || tokens[0].empty()) {
    error = "compatibility info is missing EP name";
    return false;
  }

  parsed_info.ep_name = tokens[0];

  std::unordered_map<std::string, std::string> fields;
  for (size_t i = 1; i < tokens.size(); ++i) {
    const std::string& token = tokens[i];
    if (token.empty()) {
      continue;
    }

    const size_t eq_pos = token.find('=');
    if (eq_pos == std::string::npos || eq_pos == 0) {
      error = "invalid compatibility token '" + token + "'";
      return false;
    }

    const std::string key = token.substr(0, eq_pos);
    const std::string val = token.substr(eq_pos + 1);
    fields[key] = val;
  }

  if (auto it = fields.find(kFieldVersion); it != fields.end()) {
    parsed_info.ep_version = it->second;
  }
  if (auto it = fields.find(kFieldOrtApiVersion); it != fields.end()) {
    parsed_info.ort_api_version = it->second;
  }
  if (auto it = fields.find(kFieldBackend); it != fields.end()) {
    parsed_info.backend_kind = it->second;
  }
  if (auto it = fields.find(kFieldStubSupportMul); it != fields.end()) {
    parsed_info.stub_support_mul = it->second;
  }

  return true;
}

bool TryParseBoolToken(const std::string& raw_value, bool& parsed_value) {
  if (raw_value.size() == 1) {
    if (raw_value[0] == '1') {
      parsed_value = true;
      return true;
    }
    if (raw_value[0] == '0') {
      parsed_value = false;
      return true;
    }
  }

  const std::string_view v{raw_value};
  if (EqualsIgnoreCase(v, "true") || EqualsIgnoreCase(v, "yes") || EqualsIgnoreCase(v, "on")) {
    parsed_value = true;
    return true;
  }

  if (EqualsIgnoreCase(v, "false") || EqualsIgnoreCase(v, "no") || EqualsIgnoreCase(v, "off")) {
    parsed_value = false;
    return true;
  }

  return false;
}

}  // namespace telum_compat
