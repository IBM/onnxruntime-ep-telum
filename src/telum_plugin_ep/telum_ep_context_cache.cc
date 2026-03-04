// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "telum_ep_context_cache.h"

#include <cstring>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "../plugin_ep_utils.h"

namespace {

constexpr const char* kTelumEpCacheContextFormatV2 = "telum_ep_v2";
constexpr const char* kLegacyFormatMulV1 = "telum_mul_v1";

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

std::string BytesToHex(const std::vector<uint8_t>& bytes) {
  std::ostringstream os;
  os << std::hex << std::setfill('0');
  for (uint8_t byte : bytes) {
    os << std::setw(2) << static_cast<int>(byte);
  }
  return os.str();
}

bool HexToBytes(const std::string& hex, std::vector<uint8_t>& bytes) {
  if (hex.size() % 2 != 0) {
    return false;
  }

  bytes.clear();
  bytes.reserve(hex.size() / 2);
  for (size_t i = 0; i < hex.size(); i += 2) {
    const std::string token = hex.substr(i, 2);
    try {
      size_t parsed_chars = 0;
      const int value = std::stoi(token, &parsed_chars, 16);
      if (parsed_chars != token.size() || value < 0 || value > 255) {
        return false;
      }
      bytes.push_back(static_cast<uint8_t>(value));
    } catch (...) {
      return false;
    }
  }

  return true;
}

bool ParseKeyValuePairs(const std::string& serialized,
                        std::unordered_map<std::string, std::string>& result) {
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

    result[token.substr(0, eq_pos)] = token.substr(eq_pos + 1);
  }

  return !result.empty();
}

bool TryParseInt64(const std::string& value, int64_t& parsed) {
  try {
    size_t parsed_chars = 0;
    const long long num = std::stoll(value, &parsed_chars, 10);
    if (parsed_chars != value.size()) {
      return false;
    }
    parsed = static_cast<int64_t>(num);
    return true;
  } catch (...) {
    return false;
  }
}

bool TryParseInt64Csv(const std::string& csv, std::vector<int64_t>& values) {
  values.clear();
  if (csv.empty()) {
    return true;
  }

  std::stringstream ss(csv);
  std::string token;
  while (std::getline(ss, token, ',')) {
    int64_t value = 0;
    if (!TryParseInt64(token, value)) {
      return false;
    }
    values.push_back(value);
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

OrtStatus* CreateInvalidEpCacheContextStatus(const OrtApi& ort_api, const std::string& detail) {
  const std::string message = "Invalid Telum EPContext ep_cache_context: " + detail;
  return ort_api.CreateStatus(ORT_INVALID_ARGUMENT, message.c_str());
}

OrtStatus* ParseLegacyMulFormat(const OrtApi& ort_api,
                                const std::unordered_map<std::string, std::string>& key_values,
                                telum_ep_context::EpCacheContext& parsed) {
  auto op_it = key_values.find("op");
  if (op_it == key_values.end() || op_it->second != "Mul") {
    return CreateInvalidEpCacheContextStatus(ort_api, "legacy format missing op=Mul");
  }

  auto input0_it = key_values.find("input0");
  auto input1_it = key_values.find("input1");
  if (input0_it == key_values.end() || input1_it == key_values.end() ||
      input0_it->second.empty() || input1_it->second.empty()) {
    return CreateInvalidEpCacheContextStatus(ort_api, "legacy format missing input names");
  }

  parsed.op_type = "Mul";
  parsed.attributes_blob.clear();
  parsed.input_names = {input0_it->second, input1_it->second};
  parsed.initializers.clear();

  auto parse_initializer = [&](size_t input_index, const std::string& input_name) -> OrtStatus* {
    const std::string const_key = "const" + std::to_string(input_index);
    auto is_const_it = key_values.find(const_key);
    if (is_const_it == key_values.end()) {
      return CreateInvalidEpCacheContextStatus(ort_api, "legacy format missing '" + const_key + "'");
    }

    bool is_const = false;
    if (!TryParseBool01(is_const_it->second, is_const)) {
      return CreateInvalidEpCacheContextStatus(ort_api, "legacy format invalid '" + const_key + "'");
    }

    if (!is_const) {
      return nullptr;
    }

    const std::string shape_key = const_key + "_shape";
    const std::string data_key = const_key + "_data";
    auto shape_it = key_values.find(shape_key);
    auto data_it = key_values.find(data_key);
    if (shape_it == key_values.end() || data_it == key_values.end()) {
      return CreateInvalidEpCacheContextStatus(ort_api, "legacy format missing constant payload");
    }

    std::vector<int64_t> shape;
    if (!TryParseInt64Csv(shape_it->second, shape)) {
      return CreateInvalidEpCacheContextStatus(ort_api, "legacy format invalid constant shape");
    }

    std::vector<float> float_data;
    if (!TryParseFloatCsv(data_it->second, float_data)) {
      return CreateInvalidEpCacheContextStatus(ort_api, "legacy format invalid constant data");
    }

    auto element_count = TryGetTensorElementCount(shape);
    if (!element_count.has_value() || *element_count != float_data.size()) {
      return CreateInvalidEpCacheContextStatus(ort_api, "legacy format shape/data mismatch");
    }

    telum::TensorInitializer initializer{};
    initializer.elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    initializer.shape = std::move(shape);
    initializer.raw_data.resize(float_data.size() * sizeof(float));
    if (!float_data.empty()) {
      std::memcpy(initializer.raw_data.data(), float_data.data(), initializer.raw_data.size());
    }

    parsed.initializers.emplace(input_name, std::move(initializer));
    return nullptr;
  };

  RETURN_IF_ERROR(parse_initializer(0, parsed.input_names[0]));
  RETURN_IF_ERROR(parse_initializer(1, parsed.input_names[1]));
  return nullptr;
}

}  // namespace

namespace telum_ep_context {

std::string SerializeEpCacheContext(const std::string& op_type,
                                    const std::string& attributes_blob,
                                    gsl::span<const std::string> input_names,
                                    const telum::TensorInitializerMap& initializers) {
  std::ostringstream os;
  os << "format=" << kTelumEpCacheContextFormatV2
     << ";op=" << op_type
     << ";attrs=" << attributes_blob
     << ";input_count=" << input_names.size();

  for (size_t i = 0; i < input_names.size(); ++i) {
    os << ";input" << i << "=" << input_names[i];
  }

  size_t init_index = 0;
  for (const auto& kv : initializers) {
    const auto& name = kv.first;
    const auto& init = kv.second;

    os << ";init" << init_index << "_name=" << name;
    os << ";init" << init_index << "_type=" << static_cast<int>(init.elem_type);
    os << ";init" << init_index << "_shape=" << JoinInt64Csv(init.shape);
    os << ";init" << init_index << "_data=" << BytesToHex(init.raw_data);
    ++init_index;
  }

  os << ";init_count=" << init_index;
  return os.str();
}

OrtStatus* ParseEpCacheContext(const OrtApi& ort_api,
                               const std::string& serialized,
                               EpCacheContext& parsed) {
  std::unordered_map<std::string, std::string> key_values;
  if (!ParseKeyValuePairs(serialized, key_values)) {
    return CreateInvalidEpCacheContextStatus(ort_api, "malformed key-value payload");
  }

  auto format_it = key_values.find("format");
  if (format_it == key_values.end()) {
    return CreateInvalidEpCacheContextStatus(ort_api, "missing format");
  }

  parsed = {};

  if (format_it->second == kLegacyFormatMulV1) {
    return ParseLegacyMulFormat(ort_api, key_values, parsed);
  }

  if (format_it->second != kTelumEpCacheContextFormatV2) {
    return CreateInvalidEpCacheContextStatus(ort_api, "unsupported format");
  }

  auto op_it = key_values.find("op");
  if (op_it == key_values.end() || op_it->second.empty()) {
    return CreateInvalidEpCacheContextStatus(ort_api, "missing op");
  }
  parsed.op_type = op_it->second;
  auto attrs_it = key_values.find("attrs");
  if (attrs_it != key_values.end()) {
    parsed.attributes_blob = attrs_it->second;
  }

  auto input_count_it = key_values.find("input_count");
  if (input_count_it == key_values.end()) {
    return CreateInvalidEpCacheContextStatus(ort_api, "missing input_count");
  }

  int64_t input_count = 0;
  if (!TryParseInt64(input_count_it->second, input_count) || input_count < 0) {
    return CreateInvalidEpCacheContextStatus(ort_api, "invalid input_count");
  }

  parsed.input_names.clear();
  parsed.input_names.reserve(static_cast<size_t>(input_count));
  for (int64_t i = 0; i < input_count; ++i) {
    const std::string key = "input" + std::to_string(i);
    auto it = key_values.find(key);
    if (it == key_values.end()) {
      return CreateInvalidEpCacheContextStatus(ort_api, "missing input name entry");
    }
    parsed.input_names.push_back(it->second);
  }

  auto init_count_it = key_values.find("init_count");
  if (init_count_it == key_values.end()) {
    return CreateInvalidEpCacheContextStatus(ort_api, "missing init_count");
  }

  int64_t init_count = 0;
  if (!TryParseInt64(init_count_it->second, init_count) || init_count < 0) {
    return CreateInvalidEpCacheContextStatus(ort_api, "invalid init_count");
  }

  parsed.initializers.clear();
  for (int64_t i = 0; i < init_count; ++i) {
    const std::string prefix = "init" + std::to_string(i);

    auto name_it = key_values.find(prefix + "_name");
    auto type_it = key_values.find(prefix + "_type");
    auto shape_it = key_values.find(prefix + "_shape");
    auto data_it = key_values.find(prefix + "_data");

    if (name_it == key_values.end() || type_it == key_values.end() ||
        shape_it == key_values.end() || data_it == key_values.end()) {
      return CreateInvalidEpCacheContextStatus(ort_api, "missing initializer fields");
    }

    int64_t type_value = 0;
    if (!TryParseInt64(type_it->second, type_value)) {
      return CreateInvalidEpCacheContextStatus(ort_api, "invalid initializer type");
    }

    telum::TensorInitializer initializer{};
    initializer.elem_type = static_cast<ONNXTensorElementDataType>(type_value);
    if (initializer.ElementSize() == 0) {
      return CreateInvalidEpCacheContextStatus(ort_api, "unsupported initializer element type");
    }

    if (!TryParseInt64Csv(shape_it->second, initializer.shape)) {
      return CreateInvalidEpCacheContextStatus(ort_api, "invalid initializer shape");
    }

    if (!HexToBytes(data_it->second, initializer.raw_data)) {
      return CreateInvalidEpCacheContextStatus(ort_api, "invalid initializer data encoding");
    }

    auto element_count = TryGetTensorElementCount(initializer.shape);
    if (!element_count.has_value()) {
      return CreateInvalidEpCacheContextStatus(ort_api, "invalid initializer shape dimensions");
    }

    const size_t expected_bytes = *element_count * initializer.ElementSize();
    if (initializer.raw_data.size() != expected_bytes) {
      return CreateInvalidEpCacheContextStatus(ort_api, "initializer byte length mismatch");
    }

    parsed.initializers.emplace(name_it->second, std::move(initializer));
  }

  return nullptr;
}

}  // namespace telum_ep_context
