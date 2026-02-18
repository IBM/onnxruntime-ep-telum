// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <gsl/span>

#include <optional>

#include "../plugin_ep_utils.h"
#include "telum_backend.h"

class TelumEpFactory;

/// <summary>
/// Telum scaffold implementation of ONNX Mul. Does not handle many things like broadcasting.
/// </summary>
struct MulKernel {
  enum class InputMode : uint8_t {
    kTwoRuntime = 0,
    kConst0_Runtime1,
    kRuntime0_Const1,
    kTwoConst,
  };

  MulKernel(const OrtApi& ort_api, const OrtLogger& logger,
            TelumBackend& backend,
            const std::unordered_map<std::string, FloatInitializer>& float_initializers,
            std::string input0_name, std::string input1_name,
            std::optional<std::vector<int64_t>> expected_output_shape = std::nullopt)
      : ort_api(ort_api),
        logger(logger),
        backend(backend),
        float_initializers(float_initializers),
        input0_name(std::move(input0_name)),
        input1_name(std::move(input1_name)) {
    // Optional fast path: a backend can provide a trusted Mul function pointer that avoids per-call virtual dispatch.
    mul_trusted_fn = backend.GetMulTrustedFn();

    // Cache output shape/size derived from compile-time type info when available.
    // Note that a scalar shape is represented by an empty shape vector, so we cannot use "empty" as a sentinel.
    if (expected_output_shape.has_value()) {
      bool all_static = true;
      for (int64_t dim : *expected_output_shape) {
        if (dim < 0) {
          all_static = false;
          break;
        }
      }

      if (all_static) {
        has_cached_output_shape = true;
        cached_output_shape = std::move(*expected_output_shape);
        cached_output_num_elems = 1;
        for (int64_t dim : cached_output_shape) {
          cached_output_num_elems *= static_cast<size_t>(dim);
        }
      }
    }

    // Cache initializer pointers and output shape/size for the common case where ORT drops constant initializers
    // from runtime inputs (NodeFusionOptions_DropConstantInitializers()).
    //
    // This is a perf-only optimization for the scaffold: it avoids per-Compute() unordered_map lookups and
    // repeated shape vector copies when one input is an initializer.
    //
    // NOTE: The pointers are only safe because this scaffold compiles a single node and does not mutate
    // float_initializers after creating kernels.
    saved_input0_initializer = TryGetSavedInitializer(this->input0_name);
    saved_input1_initializer = TryGetSavedInitializer(this->input1_name);
    if (saved_input0_initializer != nullptr && saved_input1_initializer != nullptr) {
      input_mode = InputMode::kTwoConst;
    } else if (saved_input0_initializer != nullptr) {
      input_mode = InputMode::kConst0_Runtime1;
    } else if (saved_input1_initializer != nullptr) {
      input_mode = InputMode::kRuntime0_Const1;
    } else {
      input_mode = InputMode::kTwoRuntime;
    }

    if (!has_cached_output_shape) {
      if (saved_input0_initializer != nullptr) {
        has_cached_output_shape = true;
        cached_output_shape = saved_input0_initializer->shape;
        cached_output_num_elems = saved_input0_initializer->data.size();
      } else if (saved_input1_initializer != nullptr) {
        has_cached_output_shape = true;
        cached_output_shape = saved_input1_initializer->shape;
        cached_output_num_elems = saved_input1_initializer->data.size();
      }
    }
  }

  const FloatInitializer* TryGetSavedInitializer(const std::string& name) const;

  void GetInputDataAndShape(Ort::KernelContext kernel_context, size_t index,
                            /*out*/ gsl::span<const float>& data,
                            /*out*/ std::vector<int64_t>& shape) const;

  OrtStatus* Compute(OrtKernelContext* kernel_ctx);

  const OrtApi& ort_api;
  const OrtLogger& logger;
  TelumBackend& backend;
  TelumMulTrustedFn mul_trusted_fn{};
  const std::unordered_map<std::string, FloatInitializer>& float_initializers;
  std::string input0_name;
  std::string input1_name;
  const FloatInitializer* saved_input0_initializer{};
  const FloatInitializer* saved_input1_initializer{};
  InputMode input_mode{InputMode::kTwoRuntime};
  bool has_cached_output_shape{};
  std::vector<int64_t> cached_output_shape;
  size_t cached_output_num_elems{};
};

/// <summary>
/// Kernel for EPContext nodes loaded from compiled models.
///
/// This implementation restores a Mul execution context from the EPContext node's
/// ep_cache_context attribute and reuses MulKernel execution.
/// </summary>
struct EpContextKernel {
  EpContextKernel(const OrtApi& ort_api, const OrtLogger& logger,
                  TelumBackend& backend,
                  std::unordered_map<std::string, FloatInitializer> float_initializers,
                  std::string input0_name, std::string input1_name)
      : ort_api(ort_api),
        logger(logger),
        float_initializers(std::move(float_initializers)),
        mul_kernel(ort_api, logger, backend, this->float_initializers,
                   std::move(input0_name), std::move(input1_name)) {}

  OrtStatus* Compute(OrtKernelContext* kernel_ctx);

  const OrtApi& ort_api;
  const OrtLogger& logger;
  std::unordered_map<std::string, FloatInitializer> float_initializers;
  MulKernel mul_kernel;
};

/// <summary>
/// Telum scaffold EP that can compile a single Mul operator.
/// </summary>
class TelumEp : public OrtEp, public ApiPtrs {
 public:
  struct Config {
    bool enable_ep_context = false;
    std::string backend_kind = "stub";
    bool stub_support_mul = true;
    // When true, ORT will drop constant initializer inputs from the compiled node's runtime interface, and the EP
    // must save the initializer values during Compile(). When false, constant initializers are provided as runtime
    // inputs (no EP-side copy), but ORT will retain them for the session lifetime.
    bool drop_constant_initializers = true;
    // Other EP configs (typically extracted from OrtSessionOptions or OrtHardwareDevice(s))
  };

  TelumEp(TelumEpFactory& factory, const std::string& name, const Config& config, const OrtLogger& logger);

  ~TelumEp();

  std::unordered_map<std::string, std::unique_ptr<MulKernel>>& MulKernels() {
    return mul_kernels_;
  }

  std::unordered_map<std::string, std::unique_ptr<EpContextKernel>>& EpContextKernels() {
    return ep_context_kernels_;
  }

  TelumBackend& Backend() {
    return *backend_;
  }

 private:
  static const char* ORT_API_CALL GetNameImpl(const OrtEp* this_ptr) noexcept;

  static OrtStatus* ORT_API_CALL CreateAllocatorImpl(_In_ OrtEp* this_ptr,
                                                     _In_ const OrtMemoryInfo* memory_info,
                                                     _Outptr_result_maybenull_ OrtAllocator** allocator) noexcept;

  static OrtStatus* ORT_API_CALL CreateSyncStreamForDeviceImpl(_In_ OrtEp* this_ptr,
                                                               _In_ const OrtMemoryDevice* memory_device,
                                                               _Outptr_ OrtSyncStreamImpl** stream) noexcept;

  static OrtStatus* ORT_API_CALL GetCapabilityImpl(OrtEp* this_ptr, const OrtGraph* graph,
                                                   OrtEpGraphSupportInfo* graph_support_info) noexcept;

  static OrtStatus* ORT_API_CALL CompileImpl(_In_ OrtEp* this_ptr, _In_ const OrtGraph** graphs,
                                             _In_ const OrtNode** fused_nodes, _In_ size_t count,
                                             _Out_writes_all_(count) OrtNodeComputeInfo** node_compute_infos,
                                             _Out_writes_(count) OrtNode** ep_context_nodes) noexcept;

  static void ORT_API_CALL ReleaseNodeComputeInfosImpl(OrtEp* this_ptr,
                                                       OrtNodeComputeInfo** node_compute_infos,
                                                       size_t num_node_compute_infos) noexcept;

  static const char* ORT_API_CALL GetCompiledModelCompatibilityInfoImpl(OrtEp* this_ptr,
                                                                        const OrtGraph* graph) noexcept;

  OrtStatus* CreateEpContextNodes(gsl::span<const OrtNode*> fused_nodes,
                                  /*out*/ gsl::span<OrtNode*> ep_context_nodes);

  // Save a subset of constant initializers needed by the compiled partition.
  // We intentionally avoid scanning all graph initializers as it can be expensive for large models.
  OrtStatus* SaveConstantInitializers(gsl::span<const Ort::ConstValueInfo> value_infos);

  TelumEpFactory& factory_;
  std::string name_;
  Config config_{};
  const OrtLogger& logger_;
  std::unique_ptr<TelumBackend> backend_;
  std::unordered_map<std::string, std::unique_ptr<MulKernel>> mul_kernels_;
  std::unordered_map<std::string, std::unique_ptr<EpContextKernel>> ep_context_kernels_;
  std::unordered_map<std::string, FloatInitializer> float_initializers_;
  std::string compatibility_info_;  // Cached compatibility string returned by GetCompiledModelCompatibilityInfo
};
