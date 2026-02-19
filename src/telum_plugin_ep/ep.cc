// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ep.h"

#include <array>
#include <cassert>
#include <cstring>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "ep_factory.h"
#include "ep_stream_support.h"
#include "telum_backend.h"
#include "telum_capability_policy.h"
#include "telum_compatibility_info.h"
#include "telum_ep_context_cache.h"
#include "telum_profile.h"

const FloatInitializer* MulKernel::TryGetSavedInitializer(const std::string& name) const {
  auto iter = float_initializers.find(name);
  return iter != float_initializers.end() ? &iter->second : nullptr;
}

void MulKernel::GetInputDataAndShape(Ort::KernelContext kernel_context, size_t index,
                                     /*out*/ gsl::span<const float>& data,
                                     /*out*/ std::vector<int64_t>& shape) const {
  Ort::ConstValue input = kernel_context.GetInput(index);
  auto type_shape = input.GetTensorTypeAndShapeInfo();

  ONNXTensorElementDataType elem_type = type_shape.GetElementType();
  if (elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    throw Ort::Exception("EP Expected float32 inputs", ORT_EP_FAIL);

  const float* float_data = input.GetTensorData<float>();
  size_t num_elems = type_shape.GetElementCount();
  data = gsl::span<const float>(float_data, num_elems);
  shape = type_shape.GetShape();
}

OrtStatus* MulKernel::Compute(OrtKernelContext* kernel_ctx) {
  // Fast path before constructing C++ wrappers and entering exception-based error handling.
  // This is a perf-only optimization for the scaffold.
  if (has_cached_output_shape) {
    // GetCapability() already enforces float32 + static equal shapes, and ORT validates user inputs against the
    // model schema before kernel execution. That means we can safely use the cached element count here without
    // re-querying input type/shape info on every Compute() call.
    const size_t expected_num_elems = cached_output_num_elems;

    auto get_input_float_data = [&](size_t index, const float** out) -> OrtStatus* {
      const OrtValue* input = nullptr;
      OrtStatus* st = ort_api.KernelContext_GetInput(kernel_ctx, index, &input);
      if (st != nullptr) {
        return st;
      }

      const void* data = nullptr;
      st = ort_api.GetTensorData(input, &data);
      if (st != nullptr) {
        return st;
      }

      *out = static_cast<const float*>(data);
      return nullptr;
    };

    const float* input0_data = nullptr;
    const float* input1_data = nullptr;

    // Determine runtime inputs without calling KernelContext_GetInputCount().
    // MulKernel precomputes a compact "input mode" at compile-time based on whether ORT dropped initializers.
    switch (input_mode) {
      case InputMode::kTwoConst: {
        // Both inputs are constant.
        input0_data = saved_input0_initializer->data.data();
        input1_data = saved_input1_initializer->data.data();
        break;
      }
      case InputMode::kConst0_Runtime1: {
        // input0 is constant initializer, runtime provides input1 at index 0.
        const float* runtime_data = nullptr;
        OrtStatus* st = get_input_float_data(/*index*/ 0, &runtime_data);
        if (st != nullptr) {
          return st;
        }

        input0_data = saved_input0_initializer->data.data();
        input1_data = runtime_data;
        break;
      }
      case InputMode::kRuntime0_Const1: {
        // input1 is constant initializer, runtime provides input0 at index 0.
        const float* runtime_data = nullptr;
        OrtStatus* st = get_input_float_data(/*index*/ 0, &runtime_data);
        if (st != nullptr) {
          return st;
        }

        input0_data = runtime_data;
        input1_data = saved_input1_initializer->data.data();
        break;
      }
      case InputMode::kTwoRuntime: {
        // Both runtime inputs provided at indices 0 and 1.
        OrtStatus* st = get_input_float_data(/*index*/ 0, &input0_data);
        if (st != nullptr) {
          return st;
        }
        st = get_input_float_data(/*index*/ 1, &input1_data);
        if (st != nullptr) {
          return st;
        }
        break;
      }
    }

    OrtValue* output = nullptr;
    OrtStatus* st = ort_api.KernelContext_GetOutput(kernel_ctx, 0,
                                                    cached_output_shape.data(), cached_output_shape.size(),
                                                    &output);
    if (st != nullptr) {
      return st;
    }

    void* output_data = nullptr;
    st = ort_api.GetTensorMutableData(output, &output_data);
    if (st != nullptr) {
      return st;
    }

    float* out = static_cast<float*>(output_data);
    if (mul_trusted_fn) {
      return mul_trusted_fn.fn(mul_trusted_fn.ctx, input0_data, input1_data, out, expected_num_elems);
    }

    return backend.Mul(gsl::span<const float>(input0_data, expected_num_elems),
                       gsl::span<const float>(input1_data, expected_num_elems),
                       gsl::span<float>(out, expected_num_elems));
  }

  Ort::KernelContext kernel_context(kernel_ctx);
  try {
    gsl::span<const float> input0;
    gsl::span<const float> input1;
    std::vector<int64_t> shape0;
    std::vector<int64_t> shape1;

    size_t num_inputs = kernel_context.GetInputCount();
    if (num_inputs == 2) {
      // Both inputs are non-constant. Get them from ORT's KernelContext.
      GetInputDataAndShape(kernel_context, 0, input0, shape0);
      GetInputDataAndShape(kernel_context, 1, input1, shape1);
    } else if (num_inputs == 1) {
      // ORT is only providing one non-constant input because this EP chose not to request constant initializer inputs.
      // Get the constant input from the initializers saved by the EP.
      // Refer to "NodeFusionOptions_DropConstantInitializers()".

      if (const FloatInitializer* const_input0 = saved_input0_initializer; const_input0 != nullptr) {
        GetInputDataAndShape(kernel_context, 0, input1, shape1);
        input0 = gsl::span<const float>(const_input0->data);
        shape0 = const_input0->shape;
      } else if (const FloatInitializer* const_input1 = saved_input1_initializer; const_input1 != nullptr) {
        GetInputDataAndShape(kernel_context, 0, input0, shape0);
        input1 = gsl::span<const float>(const_input1->data);
        shape1 = const_input1->shape;
      }
    } else {
      // Both inputs are constant. Should never happen unless all ORT optimizations (specifically constant-folding)
      // are disabled.
      const FloatInitializer* const_input0 = TryGetSavedInitializer(input0_name);
      const FloatInitializer* const_input1 = TryGetSavedInitializer(input1_name);
      RETURN_IF(const_input0 == nullptr || const_input1 == nullptr, ort_api,
                "Expected 2 initializer inputs to be saved by EP");

      input0 = gsl::span<const float>(const_input0->data);
      input1 = gsl::span<const float>(const_input1->data);
      shape0 = const_input0->shape;
      shape1 = const_input1->shape;
    }

    if (shape0 != shape1) {
      throw Ort::Exception("Expected same dimensions for both inputs", ORT_INVALID_ARGUMENT);
    }

    size_t num_outputs = kernel_context.GetOutputCount();
    if (num_outputs != 1) {
      throw Ort::Exception("Expected 1 output for MulKernel", ORT_INVALID_ARGUMENT);
    }

    auto output = kernel_context.GetOutput(0, shape0);
    float* output_data = output.GetTensorMutableData<float>();
    RETURN_IF_ERROR(backend.Mul(input0, input1, gsl::span<float>(output_data, input0.size())));
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    Ort::Status status(ex.what(), ORT_EP_FAIL);
    return status.release();
  }

  return nullptr;
}

OrtStatus* EpContextKernel::Compute(OrtKernelContext* kernel_ctx) {
  return mul_kernel.Compute(kernel_ctx);
}

/// <summary>
/// Intermediate base class with virtual destructor for proper polymorphic deletion.
/// This allows ReleaseNodeComputeInfosImpl to delete any derived type correctly
/// without manual type dispatch.
/// </summary>
struct NodeComputeInfoBase : OrtNodeComputeInfo {
  virtual ~NodeComputeInfoBase() = default;
};

/// <summary>
/// Telum OrtNodeComputeInfo that represents the computation function for a compiled OrtGraph.
/// </summary>
struct TelumNodeComputeInfo : NodeComputeInfoBase {
  TelumNodeComputeInfo(TelumEp& ep, MulKernel* kernel);

  static OrtStatus* ORT_API_CALL CreateStateImpl(OrtNodeComputeInfo* this_ptr,
                                                 OrtNodeComputeContext* compute_context,
                                                 void** compute_state);
  static OrtStatus* ORT_API_CALL ComputeImpl(OrtNodeComputeInfo* this_ptr, void* compute_state,
                                             OrtKernelContext* kernel_context);
  static void ORT_API_CALL ReleaseStateImpl(OrtNodeComputeInfo* this_ptr, void* compute_state);

  TelumEp& ep;
  MulKernel* kernel{};
};

/// <summary>
/// OrtNodeComputeInfo for EPContext nodes - delegates to EpContextKernel.
/// </summary>
struct EpContextNodeComputeInfo : NodeComputeInfoBase {
  EpContextNodeComputeInfo(TelumEp& ep, EpContextKernel* kernel);

  static OrtStatus* ORT_API_CALL CreateStateImpl(OrtNodeComputeInfo* this_ptr,
                                                 OrtNodeComputeContext* compute_context,
                                                 void** compute_state);
  static OrtStatus* ORT_API_CALL ComputeImpl(OrtNodeComputeInfo* this_ptr, void* compute_state,
                                             OrtKernelContext* kernel_context);
  static void ORT_API_CALL ReleaseStateImpl(OrtNodeComputeInfo* this_ptr, void* compute_state);

  TelumEp& ep;
  EpContextKernel* kernel{};
};

TelumEp::TelumEp(TelumEpFactory& factory, const std::string& name, const Config& config, const OrtLogger& logger)
    : OrtEp{},  // explicitly call the struct ctor to ensure all optional values are default initialized
      ApiPtrs{static_cast<const ApiPtrs&>(factory)},
      factory_{factory},
      name_{name},
      config_{config},
      logger_{logger},
      backend_{CreateTelumBackend(ort_api, TelumBackendConfig{config_.backend_kind, config_.stub_support_mul})} {
  telum_profile::ScopedEvent profile_ctor{telum_profile::Event::kTelumEpCtor};
  ort_version_supported = ORT_API_VERSION;  // set to the ORT version we were compiled with.

  // Initialize the execution provider's function table
  GetName = GetNameImpl;
  GetCapability = GetCapabilityImpl;
  Compile = CompileImpl;
  ReleaseNodeComputeInfos = ReleaseNodeComputeInfosImpl;
  CreateAllocator = CreateAllocatorImpl;                                      // optional. can be nullptr
  CreateSyncStreamForDevice = CreateSyncStreamForDeviceImpl;                  // optional. can be nullptr
  GetCompiledModelCompatibilityInfo = GetCompiledModelCompatibilityInfoImpl;  // compatibility info for compiled models

  IGNORE_ORTSTATUS(ort_api.Logger_LogMessage(&logger_,
                                             OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                              ("Telum plugin EP created with name " + name_ +
                                               ", backend=" + config_.backend_kind +
                                               ", stub_support_mul=" + (config_.stub_support_mul ? "1" : "0") +
                                               ", drop_constant_initializers=" +
                                               (config_.drop_constant_initializers ? "1" : "0"))
                                                  .c_str(),
                                             ORT_FILE, __LINE__, __FUNCTION__));
}

TelumEp::~TelumEp() = default;

/*static*/
const char* ORT_API_CALL TelumEp ::GetNameImpl(const OrtEp* this_ptr) noexcept {
  const auto* ep = static_cast<const TelumEp*>(this_ptr);
  return ep->name_.c_str();
}

OrtStatus* TelumEp::SaveConstantInitializers(gsl::span<const Ort::ConstValueInfo> value_infos) {
  telum_profile::ScopedEvent profile{telum_profile::Event::kSaveConstantInitializers};
  float_initializers_.clear();

  try {
    for (const auto& value_info : value_infos) {
      if (!value_info.IsConstantInitializer()) {
        continue;
      }

      auto name = value_info.GetName();
      Ort::ConstValue value;
      auto status = value_info.GetInitializer(value);
      if (!status.IsOK()) {
        return status.release();
      }

      auto type_shape = value.GetTensorTypeAndShapeInfo();
      const size_t num_elems = type_shape.GetElementCount();
      const ONNXTensorElementDataType elem_type = type_shape.GetElementType();
      if (elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        return Ort::Status("Expected float32 initializers", ORT_INVALID_ARGUMENT).release();
      }

      std::vector<int64_t> dims = type_shape.GetShape();
      const float* data = value.GetTensorData<float>();

      FloatInitializer ep_initializer;
      ep_initializer.shape = std::move(dims);
      ep_initializer.data.resize(num_elems);
      if (num_elems > 0) {
        std::memcpy(ep_initializer.data.data(), data, num_elems * sizeof(float));
      }
      float_initializers_.insert_or_assign(std::move(name), std::move(ep_initializer));
    }
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    Ort::Status status(ex.what(), ORT_EP_FAIL);
    return status.release();
  }

  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL TelumEp::GetCapabilityImpl(OrtEp* this_ptr, const OrtGraph* ort_graph,
                                                     OrtEpGraphSupportInfo* graph_support_info) noexcept {
  try {
    telum_profile::ScopedEvent profile{telum_profile::Event::kGetCapabilityImpl};
    TelumEp* ep = static_cast<TelumEp*>(this_ptr);

    Ort::ConstGraph graph{ort_graph};
    std::vector<Ort::ConstNode> nodes = graph.GetNodes();
    if (nodes.empty()) {
      return nullptr;  // No nodes to process
    }

    const OrtNode* supported_node = nullptr;
    TelumCapabilityStats stats{};
    TelumCapabilityNodeDisposition selected_disposition = TelumCapabilityNodeDisposition::kUnsupported;

    for (const auto& node : nodes) {
      const auto decision = EvaluateTelumCapabilityNode(ep->ort_api, node, ep->name_, ep->Backend());
      RecordTelumCapabilityDecision(decision, stats);

      if (!decision.error.empty()) {
        return ep->ort_api.CreateStatus(ORT_EP_FAIL, decision.error.c_str());
      }

      if (decision.disposition != TelumCapabilityNodeDisposition::kUnsupported) {
        supported_node = node;
        selected_disposition = decision.disposition;
        break;  // This EP only supports compiling one node at a time.
      }
    }

    const size_t num_supported = supported_node != nullptr ? 1 : 0;
    const size_t num_fallback =
        stats.num_nodes_considered >= num_supported ? (stats.num_nodes_considered - num_supported) : 0;
    if (stats.num_nodes_considered > 0 && (num_supported == 0 || num_fallback > 0)) {
      std::ostringstream os;
      os << "TelumEp::GetCapability selected " << num_supported << " of " << stats.num_nodes_considered
         << " node(s); CPU fallback node(s)=" << num_fallback
         << " [unsupported_op=" << stats.num_rejected_unsupported_op
         << ", non_float=" << stats.num_rejected_non_float
         << ", shape_or_dynamic=" << stats.num_rejected_shape_or_dynamic
         << ", epcontext_source_mismatch=" << stats.num_rejected_epcontext_source_mismatch
         << ", backend_capability=" << stats.num_rejected_backend_capability
         << "]";
      IGNORE_ORTSTATUS(ep->ort_api.Logger_LogMessage(
          &ep->logger_,
          num_supported == 0 ? OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING : OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
          os.str().c_str(), ORT_FILE, __LINE__, __FUNCTION__));
    }

    // Return early if no supported nodes
    if (supported_node == nullptr) {
      return nullptr;
    }

    if (selected_disposition == TelumCapabilityNodeDisposition::kSingleNode) {
      // Custom_Mul has concrete kernel implementation - no fusion needed.
      // Calls EpGraphSupportInfo_AddSingleNode() to inform ORT that the custom node should NOT be fused or compiled.
      RETURN_IF_ERROR(ep->ep_api.EpGraphSupportInfo_AddSingleNode(graph_support_info, supported_node));
    } else {
      // Both EPContext and Mul use AddNodesToFuse
      OrtNodeFusionOptions node_fusion_options = {};
      node_fusion_options.ort_version_supported = ORT_API_VERSION;

      // Set "drop constant initializers" to true if the compiling EP doesn't need ORT to provide constant initializers
      // as inputs to the fused/compiled node at inference time. This allows ORT to release unused initializers.
      // This EP defaults to dropping initializers and saving them during Compile(), but can be configured to keep
      // initializer inputs to avoid EP-side copies in session creation benchmarks.
      node_fusion_options.drop_constant_initializers = ep->config_.drop_constant_initializers;
      const OrtNode* nodes_to_fuse[] = {supported_node};
      RETURN_IF_ERROR(ep->ep_api.EpGraphSupportInfo_AddNodesToFuse(
          graph_support_info,
          nodes_to_fuse,
          1,
          &node_fusion_options));
    }

  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    Ort::Status status(ex.what(), ORT_EP_FAIL);
    return status.release();
  }

  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL TelumEp::CompileImpl(_In_ OrtEp* this_ptr, _In_ const OrtGraph** ort_graphs,
                                               _In_ const OrtNode** fused_nodes, _In_ size_t count,
                                               _Out_writes_all_(count) OrtNodeComputeInfo** node_compute_infos,
                                               _Out_writes_(count) OrtNode** ep_context_nodes) noexcept {
  try {
    telum_profile::ScopedEvent profile{telum_profile::Event::kCompileImpl};
    if (count != 1) {
      Ort::Status status("Expected to compile a single graph", ORT_EP_FAIL);
      return status.release();
    }

    TelumEp* ep = static_cast<TelumEp*>(this_ptr);

    Ort::ConstGraph graph{ort_graphs[0]};

    std::vector<Ort::ConstNode> nodes = graph.GetNodes();
    if (nodes.size() != 1) {
      Ort::Status status("Expected to compile a single node", ORT_EP_FAIL);
      return status.release();
    }

    auto node_op_type = nodes[0].GetOperatorType();
    auto node_domain = nodes[0].GetDomain();

    // Check if this is an EPContext node (from loading a pre-compiled model)
    bool is_ep_context_node = (node_op_type == "EPContext" && node_domain == "com.microsoft");

    // Validate configuration: cannot enable EPContext generation when loading a compiled model.
    // This is a configuration error - you cannot re-compile an already compiled model.
    if (ep->config_.enable_ep_context && is_ep_context_node) {
      Ort::Status status(
          "Invalid configuration: 'enable_ep_context' is true but model already contains "
          "EPContext nodes. Cannot re-compile an already compiled model. Either:\n"
          "  1. Use the original (uncompiled) model as input, or\n"
          "  2. Disable ep_context generation when loading a compiled model.",
          ORT_INVALID_ARGUMENT);
      return status.release();
    }

    if (node_op_type != "Mul" && !is_ep_context_node) {
      Ort::Status status("Expected to compile a Mul node or EPContext node", ORT_EP_FAIL);
      return status.release();
    }

    Ort::ConstNode fused_node{fused_nodes[0]};
    auto ep_name = fused_node.GetEpName();
    if (ep_name != ep->name_) {
      Ort::Status status("The fused node is expected to assigned to this EP to run on", ORT_EP_FAIL);
      return status.release();
    }

    auto fused_node_name = fused_node.GetName();

    if (is_ep_context_node) {
      Ort::ConstOpAttr ep_cache_context_attr;
      Ort::Status attr_status = nodes[0].GetAttributeByName("ep_cache_context", ep_cache_context_attr);
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

      telum_ep_context::MulEpCacheContext parsed_ep_context{};
      RETURN_IF_ERROR(telum_ep_context::ParseMulEpCacheContext(ep->ort_api, ep_cache_context, parsed_ep_context));

      // Create EpContextKernel using restored execution context from ep_cache_context.
      auto kernel = std::make_unique<EpContextKernel>(ep->ort_api, ep->logger_,
                                                      ep->Backend(),
                                                      std::move(parsed_ep_context.float_initializers),
                                                      std::move(parsed_ep_context.input0_name),
                                                      std::move(parsed_ep_context.input1_name));
      EpContextKernel* kernel_ptr = kernel.get();
      ep->ep_context_kernels_.emplace(fused_node_name, std::move(kernel));

      // Use EpContextNodeComputeInfo for EPContext nodes
      auto node_compute_info = std::make_unique<EpContextNodeComputeInfo>(*ep, kernel_ptr);
      node_compute_infos[0] = node_compute_info.release();
    } else {
      // For Mul nodes during initial compilation, we need exactly 2 inputs
      std::vector<Ort::ConstValueInfo> node_inputs = nodes[0].GetInputs();
      if (node_inputs.size() != 2) {
        std::string err_msg = "Mul node should have 2 inputs, got " + std::to_string(node_inputs.size());
        Ort::Status status(err_msg.c_str(), ORT_EP_FAIL);
        return status.release();
      }

      // In GetCapability(), this EP specified that it doesn't need ORT to provide constant initializers during
      // inference. Save any constant initializer inputs required by this compiled partition.
      if (ep->config_.drop_constant_initializers) {
        RETURN_IF_ERROR(ep->SaveConstantInitializers(
            gsl::span<const Ort::ConstValueInfo>{node_inputs.data(), node_inputs.size()}));
      } else {
        // Ensure kernels don't accidentally observe stale initializer values from a prior compilation.
        ep->float_initializers_.clear();
      }

      // Create MulKernel for Mul nodes.
      //
      // Avoid calling GetTensorShape() when we already have a constant initializer input, as MulKernel can derive and
      // cache the static output shape from the saved initializer metadata.
      //
      // Note: initialize via a lambda to avoid GCC maybe-uninitialized false positives with std::optional assignment.
      std::optional<std::vector<int64_t>> expected_output_shape = [&]() {
        if (ep->config_.drop_constant_initializers &&
            (node_inputs[0].IsConstantInitializer() || node_inputs[1].IsConstantInitializer())) {
          return std::optional<std::vector<int64_t>>{};
        }
        return GetTensorShape(node_inputs[0]);
      }();
      auto kernel = std::make_unique<MulKernel>(ep->ort_api, ep->logger_,
                                                ep->Backend(),
                                                ep->float_initializers_,
                                                node_inputs[0].GetName(),
                                                node_inputs[1].GetName(),
                                                std::move(expected_output_shape));
      MulKernel* kernel_ptr = kernel.get();
      ep->mul_kernels_.emplace(fused_node_name, std::move(kernel));

      // Use TelumNodeComputeInfo for Mul nodes
      auto node_compute_info = std::make_unique<TelumNodeComputeInfo>(*ep, kernel_ptr);
      node_compute_infos[0] = node_compute_info.release();

      // Create EpContext nodes for the fused nodes we compiled (only for Mul, not EPContext).
      if (ep->config_.enable_ep_context) {
        assert(ep_context_nodes != nullptr);
        RETURN_IF_ERROR(ep->CreateEpContextNodes(gsl::span<const OrtNode*>(fused_nodes, count),
                                                 gsl::span<OrtNode*>(ep_context_nodes, count)));
      }
    }
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    Ort::Status status(ex.what(), ORT_EP_FAIL);
    return status.release();
  }

  return nullptr;
}

/*static*/
void ORT_API_CALL TelumEp::ReleaseNodeComputeInfosImpl(OrtEp* this_ptr,
                                                         OrtNodeComputeInfo** node_compute_infos,
                                                         size_t num_node_compute_infos) noexcept {
  (void)this_ptr;
  for (size_t i = 0; i < num_node_compute_infos; i++) {
    // All node compute info types derive from NodeComputeInfoBase which has a virtual destructor.
    // This ensures correct polymorphic deletion without manual type dispatch.
    delete static_cast<NodeComputeInfoBase*>(node_compute_infos[i]);
  }
}

// Creates EPContext nodes from the given fused nodes.
// This serializes minimal Mul execution state into ep_cache_context so EPContext models can run.
OrtStatus* TelumEp::CreateEpContextNodes(gsl::span<const OrtNode*> fused_nodes,
                                           /*out*/ gsl::span<OrtNode*> ep_context_nodes) {
  try {
    telum_profile::ScopedEvent profile{telum_profile::Event::kCreateEpContextNodes};
    assert(fused_nodes.size() == ep_context_nodes.size());

    // Helper to collect input or output names from an array of OrtValueInfo instances.
    auto collect_input_output_names = [&](gsl::span<Ort::ConstValueInfo const> value_infos,
                                          std::vector<std::string>& result) {
      std::vector<std::string> value_names;
      value_names.reserve(value_infos.size());

      for (const auto& vi : value_infos) {
        value_names.push_back(vi.GetName());
      }

      result = std::move(value_names);
    };

    // Create an "EPContext" node for every fused node.
    for (size_t i = 0; i < fused_nodes.size(); ++i) {
      Ort::ConstNode fused_node{fused_nodes[i]};
      auto fused_node_name = fused_node.GetName();

      std::vector<Ort::ConstValueInfo> fused_node_inputs = fused_node.GetInputs();
      std::vector<Ort::ConstValueInfo> fused_node_outputs = fused_node.GetOutputs();

      std::vector<std::string> input_names;
      std::vector<std::string> output_names;

      collect_input_output_names(fused_node_inputs, /*out*/ input_names);
      collect_input_output_names(fused_node_outputs, /*out*/ output_names);

      std::string serialized_input0_name;
      std::string serialized_input1_name;
      if (input_names.size() == 2) {
        serialized_input0_name = input_names[0];
        serialized_input1_name = input_names[1];
      } else {
        // Fused node inputs may exclude dropped constant initializers.
        // Rehydrate canonical Mul input names from the just-created MulKernel for serialization.
        auto kernel_it = mul_kernels_.find(fused_node_name);
        if (kernel_it != mul_kernels_.end() && kernel_it->second != nullptr) {
          serialized_input0_name = kernel_it->second->input0_name;
          serialized_input1_name = kernel_it->second->input1_name;
        }
      }

      if (serialized_input0_name.empty() || serialized_input1_name.empty()) {
        return ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                    "Expected valid Mul input names when creating Telum EPContext nodes");
      }

      int64_t is_main_context = (i == 0);
      int64_t embed_mode = 1;

      // Create node attributes. The CreateNode() function copies the attributes.
      std::array<Ort::OpAttr, 6> attributes = {};
      std::string ep_ctx = telum_ep_context::SerializeMulEpCacheContext(
          serialized_input0_name, serialized_input1_name, float_initializers_);
      attributes[0] = Ort::OpAttr("ep_cache_context", ep_ctx.data(), static_cast<int>(ep_ctx.size()),
                                  ORT_OP_ATTR_STRING);

      attributes[1] = Ort::OpAttr("main_context", &is_main_context, 1, ORT_OP_ATTR_INT);
      attributes[2] = Ort::OpAttr("embed_mode", &embed_mode, 1, ORT_OP_ATTR_INT);
      attributes[3] = Ort::OpAttr("ep_sdk_version", "1", 1, ORT_OP_ATTR_STRING);
      attributes[4] = Ort::OpAttr("partition_name", fused_node_name.data(), static_cast<int>(fused_node_name.size()),
                                  ORT_OP_ATTR_STRING);

      attributes[5] = Ort::OpAttr("source", this->name_.data(), static_cast<int>(this->name_.size()),
                                  ORT_OP_ATTR_STRING);

      std::vector<const char*> c_input_names;
      std::transform(input_names.begin(), input_names.end(), std::back_inserter(c_input_names),
                     [](const std::string& s) { return s.c_str(); });
      std::vector<const char*> c_output_names;
      std::transform(output_names.begin(), output_names.end(), std::back_inserter(c_output_names),
                     [](const std::string& s) { return s.c_str(); });

      OrtOpAttr** op_attrs = reinterpret_cast<OrtOpAttr**>(attributes.data());
      RETURN_IF_ERROR(model_editor_api.CreateNode("EPContext", "com.microsoft", fused_node_name.c_str(),
                                                  c_input_names.data(), c_input_names.size(),
                                                  c_output_names.data(), c_output_names.size(),
                                                  op_attrs, attributes.size(),
                                                  &ep_context_nodes[i]));
    }
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    Ort::Status status(ex.what(), ORT_EP_FAIL);
    return status.release();
  }

  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL TelumEp::CreateAllocatorImpl(_In_ OrtEp* this_ptr,
                                                       _In_ const OrtMemoryInfo* memory_info,
                                                       _Outptr_result_maybenull_ OrtAllocator** allocator) noexcept {
  // A per-session allocator could be created here.
  // Logging of any issues should use ep->logger_ which is the session logger.

  TelumEp* ep = static_cast<TelumEp*>(this_ptr);

  // for simplicity in this example we use the factory implementation.
  return ep->factory_.CreateAllocator(&ep->factory_, memory_info, nullptr, allocator);
}

/*static*/
OrtStatus* ORT_API_CALL TelumEp::CreateSyncStreamForDeviceImpl(_In_ OrtEp* this_ptr,
                                                                 _In_ const OrtMemoryDevice* memory_device,
                                                                 _Outptr_ OrtSyncStreamImpl** stream) noexcept {
  // A per-session OrtSyncStreamImpl can be created here if the session options affect the implementation.
  // Logging of any issues should use logger_ which is the session logger.

  TelumEp* ep = static_cast<TelumEp*>(this_ptr);

  // we only create streams for the default device memory.
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

//
// Implementation of TelumNodeComputeInfo
//
TelumNodeComputeInfo::TelumNodeComputeInfo(TelumEp& ep, MulKernel* kernel) : ep(ep), kernel(kernel) {
  ort_version_supported = ORT_API_VERSION;
  CreateState = CreateStateImpl;
  Compute = ComputeImpl;
  ReleaseState = ReleaseStateImpl;
}

OrtStatus* TelumNodeComputeInfo::CreateStateImpl(OrtNodeComputeInfo* this_ptr,
                                                   OrtNodeComputeContext* compute_context,
                                                   void** compute_state) {
  auto* node_compute_info = static_cast<TelumNodeComputeInfo*>(this_ptr);
  (void)compute_context;
  if (node_compute_info->kernel == nullptr) {
    return node_compute_info->ep.ort_api.CreateStatus(ORT_EP_FAIL, "TelumNodeComputeInfo missing kernel");
  }

  *compute_state = node_compute_info->kernel;
  return nullptr;
}

OrtStatus* TelumNodeComputeInfo::ComputeImpl(OrtNodeComputeInfo* this_ptr, void* compute_state,
                                               OrtKernelContext* kernel_context) {
  (void)this_ptr;
  MulKernel& kernel = *reinterpret_cast<MulKernel*>(compute_state);
  return kernel.Compute(kernel_context);
}

void TelumNodeComputeInfo::ReleaseStateImpl(OrtNodeComputeInfo* this_ptr, void* compute_state) {
  (void)this_ptr;
  MulKernel& kernel = *reinterpret_cast<MulKernel*>(compute_state);
  (void)kernel;
  // Do nothing for this example.
}

//
// Implementation of EpContextNodeComputeInfo
//
EpContextNodeComputeInfo::EpContextNodeComputeInfo(TelumEp& ep, EpContextKernel* kernel) : ep(ep), kernel(kernel) {
  ort_version_supported = ORT_API_VERSION;
  CreateState = CreateStateImpl;
  Compute = ComputeImpl;
  ReleaseState = ReleaseStateImpl;
}

OrtStatus* EpContextNodeComputeInfo::CreateStateImpl(OrtNodeComputeInfo* this_ptr,
                                                     OrtNodeComputeContext* compute_context,
                                                     void** compute_state) {
  auto* node_compute_info = static_cast<EpContextNodeComputeInfo*>(this_ptr);
  (void)compute_context;
  if (node_compute_info->kernel == nullptr) {
    return node_compute_info->ep.ort_api.CreateStatus(ORT_EP_FAIL, "EpContextNodeComputeInfo missing kernel");
  }

  *compute_state = node_compute_info->kernel;
  return nullptr;
}

OrtStatus* EpContextNodeComputeInfo::ComputeImpl(OrtNodeComputeInfo* this_ptr, void* compute_state,
                                                 OrtKernelContext* kernel_context) {
  (void)this_ptr;
  EpContextKernel& kernel = *reinterpret_cast<EpContextKernel*>(compute_state);
  return kernel.Compute(kernel_context);
}

void EpContextNodeComputeInfo::ReleaseStateImpl(OrtNodeComputeInfo* this_ptr, void* compute_state) {
  (void)this_ptr;
  (void)compute_state;
  // Do nothing for this example.
}

//
// Implementation of GetCompiledModelCompatibilityInfo
//
/*static*/
const char* ORT_API_CALL TelumEp::GetCompiledModelCompatibilityInfoImpl(OrtEp* this_ptr,
                                                                          const OrtGraph* graph) noexcept {
  // Suppress unused parameter warning. The ORT_UNUSED_PARAMETER macro is in internal headers
  // (core/common/common.h) which are not available to plugin EPs using only public APIs.
  // A real EP would inspect the graph for model-specific compatibility info.
  (void)graph;
  auto* ep = static_cast<TelumEp*>(this_ptr);

  // Generate a compatibility string that includes:
  // - EP name
  // - EP version (from factory)
  // - ORT API version
  // - backend kind used for compilation
  // - backend capability gate values that affect partitioning behavior
  //
  // In a real EP, this might include driver versions, hardware IDs, ISA levels, etc.
  // The string format is EP-defined and should be parseable by ValidateCompiledModelCompatibilityInfo.
  const TelumBackendConfig backend_config{ep->config_.backend_kind, ep->config_.stub_support_mul};
  ep->compatibility_info_ = telum_compat::BuildCompatibilityInfo(
      ep->name_, ep->factory_.GetEpVersionString(), ORT_API_VERSION, backend_config);

  IGNORE_ORTSTATUS(ep->ort_api.Logger_LogMessage(&ep->logger_,
                                                 OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                                 ("GetCompiledModelCompatibilityInfo returning: " + ep->compatibility_info_).c_str(),
                                                 ORT_FILE, __LINE__, __FUNCTION__));

  return ep->compatibility_info_.c_str();
}
