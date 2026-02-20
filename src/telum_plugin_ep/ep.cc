// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ep.h"

#include <array>
#include <cassert>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "ep_factory.h"
#include "ep_stream_support.h"
#include "kernels/op_kernel.h"
#include "telum_capability_policy.h"
#include "telum_compatibility_info.h"
#include "telum_ep_context_cache.h"
#include "telum_profile.h"

namespace {

struct NodeComputeInfoBase : OrtNodeComputeInfo {
  virtual ~NodeComputeInfoBase() = default;
};

struct TelumNodeComputeInfo final : NodeComputeInfoBase {
  TelumNodeComputeInfo(TelumEp& ep, telum::CompiledNodeKernel* kernel)
      : ep(ep), kernel(kernel) {
    ort_version_supported = ORT_API_VERSION;
    CreateState = CreateStateImpl;
    Compute = ComputeImpl;
    ReleaseState = ReleaseStateImpl;
  }

  static OrtStatus* ORT_API_CALL CreateStateImpl(OrtNodeComputeInfo* this_ptr,
                                                  OrtNodeComputeContext* /*compute_context*/,
                                                  void** compute_state) {
    auto* node_compute_info = static_cast<TelumNodeComputeInfo*>(this_ptr);
    if (node_compute_info->kernel == nullptr) {
      return node_compute_info->ep.ort_api.CreateStatus(ORT_EP_FAIL, "TelumNodeComputeInfo missing kernel");
    }

    *compute_state = node_compute_info->kernel;
    return nullptr;
  }

  static OrtStatus* ORT_API_CALL ComputeImpl(OrtNodeComputeInfo* /*this_ptr*/,
                                             void* compute_state,
                                             OrtKernelContext* kernel_context) {
    auto* kernel = reinterpret_cast<telum::CompiledNodeKernel*>(compute_state);
    return kernel->Compute(kernel_context);
  }

  static void ORT_API_CALL ReleaseStateImpl(OrtNodeComputeInfo* /*this_ptr*/, void* /*compute_state*/) {
    // No-op. Kernel lifetime is owned by TelumEp::compiled_kernels_.
  }

  TelumEp& ep;
  telum::CompiledNodeKernel* kernel{};
};

std::string BoolToStr(bool v) {
  return v ? "1" : "0";
}

}  // namespace

TelumEp::TelumEp(TelumEpFactory& factory,
                 const std::string& name,
                 const Config& config,
                 const OrtLogger& logger)
    : OrtEp{},
      ApiPtrs{static_cast<const ApiPtrs&>(factory)},
      factory_(factory),
      name_(name),
      config_(config),
      logger_(logger),
      backend_(CreateTelumBackend(ort_api, TelumBackendConfig{config_.backend_kind, config_.stub_support_mul})) {
  telum_profile::ScopedEvent profile_ctor{telum_profile::Event::kTelumEpCtor};

  ort_version_supported = ORT_API_VERSION;

  GetName = GetNameImpl;
  GetCapability = GetCapabilityImpl;
  Compile = CompileImpl;
  ReleaseNodeComputeInfos = ReleaseNodeComputeInfosImpl;
  CreateAllocator = CreateAllocatorImpl;
  CreateSyncStreamForDevice = CreateSyncStreamForDeviceImpl;
  GetCompiledModelCompatibilityInfo = GetCompiledModelCompatibilityInfoImpl;

  std::ostringstream os;
  os << "Telum plugin EP created with name=" << name_
     << ", backend=" << config_.backend_kind
     << ", strict_mode=" << BoolToStr(config_.strict_mode)
     << ", log_fallbacks=" << BoolToStr(config_.log_fallbacks)
     << ", log_partition_summary=" << BoolToStr(config_.log_partition_summary)
     << ", verbose_partition_trace=" << BoolToStr(config_.verbose_partition_trace)
     << ", enable_fusion=" << BoolToStr(config_.enable_fusion)
     << ", drop_constant_initializers=" << BoolToStr(config_.drop_constant_initializers)
     << ", enable_ep_context=" << BoolToStr(config_.enable_ep_context)
     << ", stub_support_mul=" << BoolToStr(config_.stub_support_mul);

  IGNORE_ORTSTATUS(ort_api.Logger_LogMessage(
      &logger_, ORT_LOGGING_LEVEL_INFO, os.str().c_str(), ORT_FILE, __LINE__, __FUNCTION__));
}

TelumEp::~TelumEp() = default;

/*static*/
const char* ORT_API_CALL TelumEp::GetNameImpl(const OrtEp* this_ptr) noexcept {
  const auto* ep = static_cast<const TelumEp*>(this_ptr);
  return ep->name_.c_str();
}

OrtStatus* TelumEp::SaveConstantInitializers(gsl::span<const Ort::ConstValueInfo> value_infos) {
  telum_profile::ScopedEvent profile{telum_profile::Event::kSaveConstantInitializers};
  initializers_.clear();

  try {
    for (const auto& value_info : value_infos) {
      if (!value_info.IsConstantInitializer()) {
        continue;
      }

      const std::string name = value_info.GetName();
      Ort::ConstValue value;
      Ort::Status status = value_info.GetInitializer(value);
      if (!status.IsOK()) {
        return status.release();
      }

      telum::TensorInitializer initializer{};
      RETURN_IF_ERROR(telum::ConvertConstValueToInitializer(ort_api, value, initializer));
      initializers_.insert_or_assign(name, std::move(initializer));
    }

    return nullptr;
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    return ort_api.CreateStatus(ORT_EP_FAIL, ex.what());
  }
}

/*static*/
OrtStatus* ORT_API_CALL TelumEp::GetCapabilityImpl(OrtEp* this_ptr,
                                                   const OrtGraph* ort_graph,
                                                   OrtEpGraphSupportInfo* graph_support_info) noexcept {
  try {
    telum_profile::ScopedEvent profile{telum_profile::Event::kGetCapabilityImpl};
    auto* ep = static_cast<TelumEp*>(this_ptr);

    Ort::ConstGraph graph{ort_graph};
    std::vector<Ort::ConstNode> nodes = graph.GetNodes();
    if (nodes.empty()) {
      return nullptr;
    }

    const OrtNode* supported_node = nullptr;
    TelumCapabilityNodeDisposition selected_disposition = TelumCapabilityNodeDisposition::kUnsupported;
    TelumCapabilityNodeDecision selected_decision{};
    std::string selected_op_type;

    TelumCapabilityStats stats{};

    for (const auto& node : nodes) {
      const auto decision = EvaluateTelumCapabilityNode(ep->ort_api, node, ep->name_, ep->Backend(),
                                                        ep->config_.strict_mode);
      RecordTelumCapabilityDecision(decision, stats);

      if (!decision.error.empty()) {
        return ep->ort_api.CreateStatus(ORT_EP_FAIL, decision.error.c_str());
      }

      if (decision.disposition != TelumCapabilityNodeDisposition::kUnsupported) {
        supported_node = node;
        selected_disposition = decision.disposition;
        selected_decision = decision;
        selected_op_type = node.GetOperatorType();
        break;
      }

      if (ep->config_.strict_mode && decision.op_kind != telum::OpKind::kUnknown &&
          decision.reject_reason != TelumCapabilityRejectReason::kUnsupportedOp) {
        std::ostringstream strict_error;
        strict_error << "Telum strict_mode rejection: op='" << node.GetOperatorType() << "'"
                     << ", reason='" << decision.reject_detail << "'";
        return ep->ort_api.CreateStatus(ORT_EP_FAIL, strict_error.str().c_str());
      }
    }

    const size_t num_supported = supported_node != nullptr ? 1 : 0;
    const size_t num_fallback =
        stats.num_nodes_considered >= num_supported ? (stats.num_nodes_considered - num_supported) : 0;

    if (ep->config_.log_partition_summary &&
        stats.num_nodes_considered > 0 && (num_supported == 0 || num_fallback > 0)) {
      std::ostringstream os;
      os << "TelumEp::GetCapability selected " << num_supported << " of " << stats.num_nodes_considered
         << " node(s); CPU fallback node(s)=" << num_fallback
         << " [unsupported_op=" << stats.num_rejected_unsupported_op
         << ", non_tensor_or_type=" << stats.num_rejected_non_tensor_or_type
         << ", shape_or_dynamic=" << stats.num_rejected_shape_or_dynamic
         << ", op_constraint=" << stats.num_rejected_op_constraint
         << ", epcontext_source_mismatch=" << stats.num_rejected_epcontext_source_mismatch
         << ", backend_capability=" << stats.num_rejected_backend_capability
         << "]";

      IGNORE_ORTSTATUS(ep->ort_api.Logger_LogMessage(
          &ep->logger_,
          num_supported == 0 ? ORT_LOGGING_LEVEL_WARNING : ORT_LOGGING_LEVEL_INFO,
          os.str().c_str(), ORT_FILE, __LINE__, __FUNCTION__));
    }

    if (supported_node == nullptr) {
      return nullptr;
    }

    if (ep->config_.verbose_partition_trace) {
      std::ostringstream trace;
      trace << "Telum partition selection: op='" << selected_op_type << "'"
            << ", reason='" << selected_decision.reject_detail << "'";
      IGNORE_ORTSTATUS(ep->ort_api.Logger_LogMessage(
          &ep->logger_, ORT_LOGGING_LEVEL_INFO, trace.str().c_str(), ORT_FILE, __LINE__, __FUNCTION__));
    }

    if (selected_disposition == TelumCapabilityNodeDisposition::kSingleNode) {
      RETURN_IF_ERROR(ep->ep_api.EpGraphSupportInfo_AddSingleNode(graph_support_info, supported_node));
    } else {
      OrtNodeFusionOptions node_fusion_options = {};
      node_fusion_options.ort_version_supported = ORT_API_VERSION;
      node_fusion_options.drop_constant_initializers = ep->config_.drop_constant_initializers;

      const OrtNode* nodes_to_fuse[] = {supported_node};
      RETURN_IF_ERROR(ep->ep_api.EpGraphSupportInfo_AddNodesToFuse(
          graph_support_info, nodes_to_fuse, 1, &node_fusion_options));
    }

    return nullptr;
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    return Ort::Status(ex.what(), ORT_EP_FAIL).release();
  }
}

/*static*/
OrtStatus* ORT_API_CALL TelumEp::CompileImpl(_In_ OrtEp* this_ptr,
                                             _In_ const OrtGraph** ort_graphs,
                                             _In_ const OrtNode** fused_nodes,
                                             _In_ size_t count,
                                             _Out_writes_all_(count) OrtNodeComputeInfo** node_compute_infos,
                                             _Out_writes_(count) OrtNode** ep_context_nodes) noexcept {
  try {
    telum_profile::ScopedEvent profile{telum_profile::Event::kCompileImpl};
    auto* ep = static_cast<TelumEp*>(this_ptr);

    if (count != 1) {
      return Ort::Status("Expected to compile exactly one graph partition", ORT_EP_FAIL).release();
    }

    Ort::ConstGraph graph{ort_graphs[0]};
    std::vector<Ort::ConstNode> nodes = graph.GetNodes();
    if (nodes.size() != 1) {
      return Ort::Status("Expected to compile a single node graph", ORT_EP_FAIL).release();
    }

    const Ort::ConstNode node = nodes[0];
    const std::string node_op_type = node.GetOperatorType();
    const std::string node_domain = node.GetDomain();

    const bool is_ep_context_node = (node_op_type == "EPContext" && node_domain == "com.microsoft");
    if (ep->config_.enable_ep_context && is_ep_context_node) {
      return Ort::Status(
                 "Invalid configuration: 'enable_ep_context=1' cannot be used when compiling existing EPContext nodes",
                 ORT_INVALID_ARGUMENT)
          .release();
    }

    Ort::ConstNode fused_node{fused_nodes[0]};
    if (fused_node.GetEpName() != ep->name_) {
      return Ort::Status("Compiled node is not assigned to this EP", ORT_EP_FAIL).release();
    }

    const std::string fused_node_name = fused_node.GetName();

    OrtStatus* kernel_status = nullptr;
    std::unique_ptr<telum::CompiledNodeKernel> compiled_kernel;

    if (is_ep_context_node) {
      Ort::ConstOpAttr ep_cache_context_attr;
      Ort::Status attr_status = node.GetAttributeByName("ep_cache_context", ep_cache_context_attr);
      if (!attr_status.IsOK()) {
        return ep->ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                        "EPContext node is missing required ep_cache_context attribute");
      }

      std::string ep_cache_context;
      attr_status = ep_cache_context_attr.GetValue(ep_cache_context);
      if (!attr_status.IsOK()) {
        return ep->ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                        "EPContext node has invalid ep_cache_context attribute value");
      }

      telum_ep_context::EpCacheContext parsed_ctx{};
      RETURN_IF_ERROR(telum_ep_context::ParseEpCacheContext(ep->ort_api, ep_cache_context, parsed_ctx));

      compiled_kernel = telum::CreateCompiledNodeKernelFromEpContext(
          ep->ort_api, ep->logger_, ep->Backend(),
          telum::KernelConfig{ep->config_.log_fallbacks, ep->config_.strict_mode},
          parsed_ctx.op_type, parsed_ctx.attributes_blob, parsed_ctx.input_names, parsed_ctx.initializers,
          ep->config_.drop_constant_initializers, kernel_status);
      if (kernel_status != nullptr) {
        return kernel_status;
      }
    } else {
      std::vector<Ort::ConstValueInfo> node_inputs = node.GetInputs();
      if (ep->config_.drop_constant_initializers) {
        RETURN_IF_ERROR(ep->SaveConstantInitializers(
            gsl::span<const Ort::ConstValueInfo>{node_inputs.data(), node_inputs.size()}));
      } else {
        ep->initializers_.clear();
      }

      compiled_kernel = telum::CreateCompiledNodeKernel(
          ep->ort_api, ep->logger_, ep->Backend(),
          telum::KernelConfig{ep->config_.log_fallbacks, ep->config_.strict_mode},
          node, ep->initializers_, ep->config_.drop_constant_initializers, kernel_status);
      if (kernel_status != nullptr) {
        return kernel_status;
      }
    }

    if (!compiled_kernel) {
      return ep->ort_api.CreateStatus(ORT_EP_FAIL, "Failed to create compiled Telum kernel");
    }

    telum::CompiledNodeKernel* kernel_ptr = compiled_kernel.get();
    ep->compiled_kernels_[fused_node_name] = std::move(compiled_kernel);

    auto node_compute_info = std::make_unique<TelumNodeComputeInfo>(*ep, kernel_ptr);
    node_compute_infos[0] = node_compute_info.release();

    if (!is_ep_context_node && ep->config_.enable_ep_context) {
      assert(ep_context_nodes != nullptr);
      RETURN_IF_ERROR(ep->CreateEpContextNodes(gsl::span<const OrtNode*>(fused_nodes, count),
                                               gsl::span<OrtNode*>(ep_context_nodes, count)));
    }

    return nullptr;

  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    return Ort::Status(ex.what(), ORT_EP_FAIL).release();
  }
}

/*static*/
void ORT_API_CALL TelumEp::ReleaseNodeComputeInfosImpl(OrtEp* /*this_ptr*/,
                                                       OrtNodeComputeInfo** node_compute_infos,
                                                       size_t num_node_compute_infos) noexcept {
  for (size_t i = 0; i < num_node_compute_infos; ++i) {
    delete static_cast<NodeComputeInfoBase*>(node_compute_infos[i]);
  }
}

OrtStatus* TelumEp::CreateEpContextNodes(gsl::span<const OrtNode*> fused_nodes,
                                         /*out*/ gsl::span<OrtNode*> ep_context_nodes) {
  try {
    telum_profile::ScopedEvent profile{telum_profile::Event::kCreateEpContextNodes};
    assert(fused_nodes.size() == ep_context_nodes.size());

    for (size_t i = 0; i < fused_nodes.size(); ++i) {
      Ort::ConstNode fused_node{fused_nodes[i]};
      const std::string fused_node_name = fused_node.GetName();

      std::vector<Ort::ConstValueInfo> fused_node_inputs = fused_node.GetInputs();
      std::vector<Ort::ConstValueInfo> fused_node_outputs = fused_node.GetOutputs();

      std::vector<std::string> input_names;
      input_names.reserve(fused_node_inputs.size());
      for (const auto& vi : fused_node_inputs) {
        input_names.push_back(vi.GetName());
      }

      auto kernel_it = compiled_kernels_.find(fused_node_name);
      if (kernel_it == compiled_kernels_.end() || kernel_it->second == nullptr) {
        return ort_api.CreateStatus(ORT_EP_FAIL,
                                    "Missing compiled kernel while creating EPContext node");
      }
      input_names = kernel_it->second->InputNames();

      std::vector<std::string> output_names;
      output_names.reserve(fused_node_outputs.size());
      for (const auto& vi : fused_node_outputs) {
        output_names.push_back(vi.GetName());
      }

      const int64_t is_main_context = (i == 0) ? 1 : 0;
      const int64_t embed_mode = 1;

      std::array<Ort::OpAttr, 6> attributes = {};
      const auto& kernel_ref = *kernel_it->second;
      const std::string ep_ctx = telum_ep_context::SerializeEpCacheContext(
          kernel_ref.OpType(), kernel_ref.SerializeAttributes(), kernel_ref.InputNames(), initializers_);

      attributes[0] = Ort::OpAttr("ep_cache_context", ep_ctx.data(), static_cast<int>(ep_ctx.size()),
                                  ORT_OP_ATTR_STRING);
      attributes[1] = Ort::OpAttr("main_context", &is_main_context, 1, ORT_OP_ATTR_INT);
      attributes[2] = Ort::OpAttr("embed_mode", &embed_mode, 1, ORT_OP_ATTR_INT);
      attributes[3] = Ort::OpAttr("ep_sdk_version", "2", 1, ORT_OP_ATTR_STRING);
      attributes[4] = Ort::OpAttr("partition_name", fused_node_name.data(), static_cast<int>(fused_node_name.size()),
                                  ORT_OP_ATTR_STRING);
      attributes[5] = Ort::OpAttr("source", name_.data(), static_cast<int>(name_.size()), ORT_OP_ATTR_STRING);

      std::vector<const char*> c_input_names;
      c_input_names.reserve(input_names.size());
      for (const auto& name : input_names) {
        c_input_names.push_back(name.c_str());
      }

      std::vector<const char*> c_output_names;
      c_output_names.reserve(output_names.size());
      for (const auto& name : output_names) {
        c_output_names.push_back(name.c_str());
      }

      OrtOpAttr** op_attrs = reinterpret_cast<OrtOpAttr**>(attributes.data());
      RETURN_IF_ERROR(model_editor_api.CreateNode("EPContext", "com.microsoft", fused_node_name.c_str(),
                                                  c_input_names.data(), c_input_names.size(),
                                                  c_output_names.data(), c_output_names.size(),
                                                  op_attrs, attributes.size(),
                                                  &ep_context_nodes[i]));
    }

    return nullptr;

  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    return Ort::Status(ex.what(), ORT_EP_FAIL).release();
  }
}

/*static*/
OrtStatus* ORT_API_CALL TelumEp::CreateAllocatorImpl(_In_ OrtEp* this_ptr,
                                                     _In_ const OrtMemoryInfo* memory_info,
                                                     _Outptr_result_maybenull_ OrtAllocator** allocator) noexcept {
  auto* ep = static_cast<TelumEp*>(this_ptr);
  return ep->factory_.CreateAllocator(&ep->factory_, memory_info, nullptr, allocator);
}

/*static*/
OrtStatus* ORT_API_CALL TelumEp::CreateSyncStreamForDeviceImpl(_In_ OrtEp* this_ptr,
                                                               _In_ const OrtMemoryDevice* memory_device,
                                                               _Outptr_ OrtSyncStreamImpl** stream) noexcept {
  auto* ep = static_cast<TelumEp*>(this_ptr);

  if (auto mem_type = ep->factory_.ep_api.MemoryDevice_GetMemoryType(memory_device);
      mem_type != OrtDeviceMemoryType_DEFAULT) {
    std::string error = "Invalid OrtMemoryDevice. Expected OrtDeviceMemoryType_DEFAULT(0). Got ";
    error += std::to_string(mem_type);
    return ep->ort_api.CreateStatus(ORT_INVALID_ARGUMENT, error.c_str());
  }

  auto sync_stream = std::make_unique<StreamImpl>(ep->factory_, ep, nullptr);
  *stream = sync_stream.release();
  return nullptr;
}

/*static*/
const char* ORT_API_CALL TelumEp::GetCompiledModelCompatibilityInfoImpl(OrtEp* this_ptr,
                                                                         const OrtGraph* graph) noexcept {
  (void)graph;
  auto* ep = static_cast<TelumEp*>(this_ptr);

  const TelumBackendConfig backend_config{ep->config_.backend_kind, ep->config_.stub_support_mul};
  ep->compatibility_info_ = telum_compat::BuildCompatibilityInfo(
      ep->name_, ep->factory_.GetEpVersionString(), ORT_API_VERSION, backend_config,
      ep->config_.strict_mode, ep->config_.drop_constant_initializers);

  IGNORE_ORTSTATUS(ep->ort_api.Logger_LogMessage(
      &ep->logger_, ORT_LOGGING_LEVEL_INFO,
      ("GetCompiledModelCompatibilityInfo returning: " + ep->compatibility_info_).c_str(),
      ORT_FILE, __LINE__, __FUNCTION__));

  return ep->compatibility_info_.c_str();
}
