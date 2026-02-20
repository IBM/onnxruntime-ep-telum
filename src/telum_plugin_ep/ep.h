// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "../plugin_ep_utils.h"
#include "kernels/op_kernel.h"
#include "telum_backend.h"

class TelumEpFactory;

class TelumEp : public OrtEp, public ApiPtrs {
 public:
  struct Config {
    bool enable_ep_context = false;
    std::string backend_kind = "stub";
    bool stub_support_mul = true;

    bool strict_mode = false;
    bool log_fallbacks = true;
    bool log_partition_summary = true;
    bool verbose_partition_trace = false;
    bool enable_fusion = true;

    bool drop_constant_initializers = true;
  };

  TelumEp(TelumEpFactory& factory, const std::string& name, const Config& config, const OrtLogger& logger);
  ~TelumEp();

  std::unordered_map<std::string, std::unique_ptr<telum::CompiledNodeKernel>>& CompiledKernels() {
    return compiled_kernels_;
  }

  const std::unordered_map<std::string, std::unique_ptr<telum::CompiledNodeKernel>>& CompiledKernels() const {
    return compiled_kernels_;
  }

  telum::TensorInitializerMap& SavedInitializers() {
    return initializers_;
  }

  const telum::TensorInitializerMap& SavedInitializers() const {
    return initializers_;
  }

  TelumBackend& Backend() {
    return *backend_;
  }

  const Config& GetConfig() const {
    return config_;
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

  OrtStatus* SaveConstantInitializers(gsl::span<const Ort::ConstValueInfo> value_infos);

  TelumEpFactory& factory_;
  std::string name_;
  Config config_{};
  const OrtLogger& logger_;
  std::unique_ptr<TelumBackend> backend_;

  std::unordered_map<std::string, std::unique_ptr<telum::CompiledNodeKernel>> compiled_kernels_;
  telum::TensorInitializerMap initializers_;

  std::string compatibility_info_;
};
