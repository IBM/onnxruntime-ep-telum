// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "telum_ep_context_cache.h"

#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr const char* kTelumEpCacheContextFormat = "telum_mul_v1";

std::string JoinInt64Csv(const std::vector<int64_t>& values) {
  std::ostringstream os;
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      os << ",";
    }
    os << values[i];
  }
  return os.str();
}

std::string JoinFloatCsv(const std::vector<float>& values) {
  std::ostringstream os;
  os << std::setprecision(std::numeric_limits<float>::max_digits10);
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      os << ",";
    }
    os << values[i];
  }
  return os.str();
}

bool ParseKeyValuePairs(const std::string& serialized, std::unordered_map<std::string, std::string>& result) {
  result.clear();

  std::stringstream ss(serialized);
  std::string token;
  while (std::getline(ss, token, ';')) {
    if (token.empty()) {
      continue;
    }

    const size_t eq_pos = token.find('=');
    if (eq_pos == std::string::npos || eq_pos == 0 || eq_pos + 1 > token.size()) {
      return false;
    }

    const std::string key = token.substr(0, eq_pos);
    const std::string value = token.substr(eq_pos + 1);
    result[key] = value;
  }

  return !result.empty();
}

bool TryParseInt64Csv(const std::string& csv, std::vector<int64_t>& values) {
  values.clear();
  if (csv.empty()) {
    return true;
  }

  std::stringstream ss(csv);
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (token.empty()) {
      return false;
    }

    try {
      size_t parsed_chars = 0;
      const long long value = std::stoll(token, &parsed_chars, 10);
      if (parsed_chars != token.size()) {
        return false;
      }
      values.push_back(static_cast<int64_t>(value));
    } catch (...) {
      return false;
    }
  }

  return true;
}

bool TryParseFloatCsv(const std::string& csv, std::vector<float>& values) {
  values.clear();
  if (csv.empty()) {
    return true;
  }

  std::stringstream ss(csv);
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (token.empty()) {
      return false;
    }

    try {
      size_t parsed_chars = 0;
      const float value = std::stof(token, &parsed_chars);
      if (parsed_chars != token.size()) {
        return false;
      }
      values.push_back(value);
    } catch (...) {
      return false;
    }
  }

  return true;
}

bool TryParseBool01(const std::string& value, bool& parsed) {
  if (value == "1") {
    parsed = true;
    return true;
  }
  if (value == "0") {
    parsed = false;
    return true;
  }

  return false;
}

std::optional<size_t> TryGetTensorElementCount(const std::vector<int64_t>& shape) {
  size_t count = 1;
  for (int64_t dim : shape) {
    if (dim < 0) {
      return std::nullopt;
    }

    if (dim == 0) {
      count = 0;
      continue;
    }

    const size_t dim_size = static_cast<size_t>(dim);
    if (count > std::numeric_limits<size_t>::max() / dim_size) {
      return std::nullopt;
    }
    count *= dim_size;
  }

  return count;
}

OrtStatus* CreateInvalidEpCacheContextStatus(const OrtApi& ort_api, const std::string& detail) {
  const std::string message = "Invalid Telum EPContext ep_cache_context: " + detail;
  return ort_api.CreateStatus(ORT_INVALID_ARGUMENT, message.c_str());
}

}  // namespace

namespace telum_ep_context {

std::string SerializeMulEpCacheContext(
    const std::string& input0_name,
    const std::string& input1_name,
    const std::unordered_map<std::string, FloatInitializer>& float_initializers) {
  std::ostringstream os;
  os << "format=" << kTelumEpCacheContextFormat
     << ";op=Mul"
     << ";input0=" << input0_name
     << ";input1=" << input1_name;

  auto append_initializer = [&](size_t input_index, const std::string& input_name) {
    const std::string const_key = "const" + std::to_string(input_index);
    auto initializer_it = float_initializers.find(input_name);
    if (initializer_it == float_initializers.end()) {
      os << ";" << const_key << "=0";
      return;
    }

    os << ";" << const_key << "=1";
    os << ";" << const_key << "_shape=" << JoinInt64Csv(initializer_it->second.shape);
    os << ";" << const_key << "_data=" << JoinFloatCsv(initializer_it->second.data);
  };

  append_initializer(0, input0_name);
  append_initializer(1, input1_name);
  return os.str();
}

OrtStatus* ParseMulEpCacheContext(const OrtApi& ort_api,
                                  const std::string& serialized,
                                  MulEpCacheContext& parsed) {
  std::unordered_map<std::string, std::string> key_values;
  if (!ParseKeyValuePairs(serialized, key_values)) {
    return CreateInvalidEpCacheContextStatus(ort_api, "malformed key-value pairs");
  }

  auto format_it = key_values.find("format");
  if (format_it == key_values.end() || format_it->second != kTelumEpCacheContextFormat) {
    return CreateInvalidEpCacheContextStatus(ort_api, "missing or unsupported format");
  }

  auto op_it = key_values.find("op");
  if (op_it == key_values.end() || op_it->second != "Mul") {
    return CreateInvalidEpCacheContextStatus(ort_api, "missing or unsupported op");
  }

  auto input0_it = key_values.find("input0");
  auto input1_it = key_values.find("input1");
  if (input0_it == key_values.end() || input1_it == key_values.end() ||
      input0_it->second.empty() || input1_it->second.empty()) {
    return CreateInvalidEpCacheContextStatus(ort_api, "missing input names");
  }

  parsed.input0_name = input0_it->second;
  parsed.input1_name = input1_it->second;
  parsed.float_initializers.clear();

  auto parse_initializer = [&](size_t input_index, const std::string& input_name) -> OrtStatus* {
    const std::string const_key = "const" + std::to_string(input_index);
    auto is_const_it = key_values.find(const_key);
    if (is_const_it == key_values.end()) {
      return CreateInvalidEpCacheContextStatus(ort_api, "missing '" + const_key + "' flag");
    }

    bool is_const = false;
    if (!TryParseBool01(is_const_it->second, is_const)) {
      return CreateInvalidEpCacheContextStatus(ort_api, "invalid '" + const_key + "' flag");
    }

    if (!is_const) {
      return nullptr;
    }

    const std::string shape_key = const_key + "_shape";
    const std::string data_key = const_key + "_data";
    auto shape_it = key_values.find(shape_key);
    auto data_it = key_values.find(data_key);
    if (shape_it == key_values.end() || data_it == key_values.end()) {
      return CreateInvalidEpCacheContextStatus(ort_api, "missing constant initializer payload");
    }

    FloatInitializer initializer{};
    if (!TryParseInt64Csv(shape_it->second, initializer.shape)) {
      return CreateInvalidEpCacheContextStatus(ort_api, "invalid constant initializer shape");
    }
    if (!TryParseFloatCsv(data_it->second, initializer.data)) {
      return CreateInvalidEpCacheContextStatus(ort_api, "invalid constant initializer data");
    }

    auto element_count = TryGetTensorElementCount(initializer.shape);
    if (!element_count.has_value() || *element_count != initializer.data.size()) {
      return CreateInvalidEpCacheContextStatus(ort_api, "constant initializer shape/data mismatch");
    }

    parsed.float_initializers.emplace(input_name, std::move(initializer));
    return nullptr;
  };

  RETURN_IF_ERROR(parse_initializer(0, parsed.input0_name));
  RETURN_IF_ERROR(parse_initializer(1, parsed.input1_name));
  return nullptr;
}

}  // namespace telum_ep_context

