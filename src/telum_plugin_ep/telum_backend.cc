// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "telum_backend.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(ORT_TELUM_PLUGIN_EP_ZDNN) && defined(__linux__)
#include <dlfcn.h>
#endif

#include "telum_profile.h"

namespace {

constexpr const char* kTelumBackendKindZdnn = "zdnn";

class UnavailableTelumBackend final : public TelumBackend {
 public:
  explicit UnavailableTelumBackend(const OrtApi& api, std::string reason)
      : api_(api), reason_(std::move(reason)) {}

  std::string BackendKind() const override { return "unavailable"; }

  bool IsRuntimeReady() const noexcept override { return false; }

  std::string RuntimeStatusMessage() const override { return reason_; }

  uint32_t MaxDimIdxSize() const noexcept override { return 0; }

  bool SupportsMul() const noexcept override { return false; }

  bool SupportsOp(telum::OpKind /*op_kind*/) const noexcept override { return false; }

  OrtStatus* Mul(gsl::span<const float> /*input0*/,
                 gsl::span<const float> /*input1*/,
                 gsl::span<float> /*output*/) noexcept override {
    const std::string msg = "Telum backend is unavailable: " + reason_;
    return api_.CreateStatus(ORT_FAIL, msg.c_str());
  }

  OrtStatus* Binary(const TelumBinaryRequest& /*request*/) noexcept override {
    const std::string msg = "Telum backend is unavailable: " + reason_;
    return api_.CreateStatus(ORT_FAIL, msg.c_str());
  }

  OrtStatus* Unary(const TelumUnaryRequest& /*request*/) noexcept override {
    const std::string msg = "Telum backend is unavailable: " + reason_;
    return api_.CreateStatus(ORT_FAIL, msg.c_str());
  }

  OrtStatus* Softmax(const TelumSoftmaxRequest& /*request*/) noexcept override {
    const std::string msg = "Telum backend is unavailable: " + reason_;
    return api_.CreateStatus(ORT_FAIL, msg.c_str());
  }

  OrtStatus* LayerNormalization(const TelumLayerNormRequest& /*request*/) noexcept override {
    const std::string msg = "Telum backend is unavailable: " + reason_;
    return api_.CreateStatus(ORT_FAIL, msg.c_str());
  }

  OrtStatus* MatMul(const TelumMatMulRequest& /*request*/) noexcept override {
    const std::string msg = "Telum backend is unavailable: " + reason_;
    return api_.CreateStatus(ORT_FAIL, msg.c_str());
  }

  OrtStatus* Gemm(const TelumGemmRequest& /*request*/) noexcept override {
    const std::string msg = "Telum backend is unavailable: " + reason_;
    return api_.CreateStatus(ORT_FAIL, msg.c_str());
  }

 private:
  const OrtApi& api_;
  std::string reason_;
};

#if defined(ORT_TELUM_PLUGIN_EP_ZDNN) && defined(__linux__) && (defined(__s390x__) || defined(__s390__))

// Minimal zDNN ABI declarations for runtime dynamic loading via dlopen/dlsym.
// We intentionally avoid compile-time dependency on zDNN headers/libraries so upstream CI can still build this plugin.

constexpr int kZdnnOk = 0;
constexpr uint32_t kZdnnLayout1D = 0;   // ZDNN_1D
constexpr uint32_t kZdnnLayout2D = 1;   // ZDNN_2D
constexpr uint32_t kZdnnLayout2DS = 2;  // ZDNN_2DS
constexpr uint32_t kZdnnLayout3DS = 4;  // ZDNN_3DS
constexpr uint32_t kZdnnLayoutNhwc = 8; // ZDNN_NHWC
constexpr uint32_t kZdnnTypeFp32 = 255; // FP32
constexpr int kMatmulOpAddition = 0;    // MATMUL_OP_ADDITION
constexpr int kMatmulBcastOpAddition = 0;  // MATMUL_BCAST_OP_ADDITION
constexpr int kSoftmaxActNone = 0;      // SOFTMAX_ACT_NONE
constexpr int kMomentsBesselPopulation = 0;  // MOMENTS_BESSEL_POPULATION

constexpr int kNnpaAdd = 16;            // NNPA_ADD
constexpr int kNnpaSub = 17;            // NNPA_SUB
constexpr int kNnpaMul = 18;            // NNPA_MUL
constexpr int kNnpaDiv = 19;            // NNPA_DIV
constexpr int kNnpaMin = 20;            // NNPA_MIN
constexpr int kNnpaMax = 21;            // NNPA_MAX
constexpr int kNnpaLog = 32;            // NNPA_LOG
constexpr int kNnpaExp = 33;            // NNPA_EXP
constexpr int kNnpaSqrt = 34;           // NNPA_SQRT
constexpr int kNnpaRelu = 49;           // NNPA_RELU
constexpr int kNnpaTanh = 50;           // NNPA_TANH
constexpr int kNnpaSigmoid = 51;        // NNPA_SIGMOID
constexpr int kNnpaGelu = 53;           // NNPA_GELU
constexpr int kNnpaSoftmax = 52;        // NNPA_SOFTMAX
constexpr int kNnpaMoments = 65;        // NNPA_MOMENTS
constexpr int kNnpaLayernorm = 66;      // NNPA_LAYERNORM
constexpr int kNnpaMatmulOp = 113;      // NNPA_MATMUL_OP
constexpr int kNnpaMatmulOpBcast23 = 114; // NNPA_MATMUL_OP_BCAST23
constexpr int kNnpaMatmulOpBcast1 = 115;  // NNPA_MATMUL_OP_BCAST1

struct ZdnnTensorDesc {
  uint32_t layout;
  uint32_t format;
  uint32_t type;
  uint32_t dim4;
  uint32_t dim3;
  uint32_t dim2;
  uint32_t dim1;
};

struct ZdnnZTensor {
  ZdnnTensorDesc* pre_transformed_desc;
  ZdnnTensorDesc* transformed_desc;
  uint64_t buffer_size;
  void* buffer;
  bool is_transformed;
  char reserved[3];
  float rec_scale;
  float offset;
  char reserved2[20];
};

using ZdnnInitFn = void (*)();
using ZdnnIsNnpaInstalledFn = bool (*)();
using ZdnnIsNnpaFunctionInstalledFn = bool (*)(int, ...);
using ZdnnGetStatusMessageFn = const char* (*)(int);
using ZdnnGetNnpaMaxDimIdxSizeFn = uint32_t (*)();
using ZdnnInitPreTransformedDescFn = void (*)(uint32_t, uint32_t, ZdnnTensorDesc*, ...);
using ZdnnGenerateTransformedDescFn = int (*)(const ZdnnTensorDesc*, ZdnnTensorDesc*);
using ZdnnInitZTensorFn = void (*)(ZdnnTensorDesc*, ZdnnTensorDesc*, ZdnnZTensor*);
using ZdnnAllocHelperZTensorFn = int (*)(ZdnnZTensor*);
using ZdnnFreeZTensorBufferFn = int (*)(const ZdnnZTensor*);
using ZdnnTransformZTensorFn = int (*)(ZdnnZTensor*, ...);
using ZdnnTransformOrigTensorFn = int (*)(const ZdnnZTensor*, void*);
using ZdnnBinaryFn = int (*)(const ZdnnZTensor*, const ZdnnZTensor*, ZdnnZTensor*);
using ZdnnUnaryFn = int (*)(const ZdnnZTensor*, ZdnnZTensor*);
using ZdnnReluFn = int (*)(const ZdnnZTensor*, const void*, ZdnnZTensor*);
using ZdnnSoftmaxFn = int (*)(const ZdnnZTensor*, void*, int, ZdnnZTensor*);
using ZdnnMomentsFn = int (*)(const ZdnnZTensor*, int, ZdnnZTensor*, ZdnnZTensor*);
using ZdnnLayernormFn = int (*)(const ZdnnZTensor*, const ZdnnZTensor*, const ZdnnZTensor*, float, float, float,
                                ZdnnZTensor*);
using ZdnnMatmulOpFn = int (*)(const ZdnnZTensor*, const ZdnnZTensor*, const ZdnnZTensor*, int, ZdnnZTensor*);
using ZdnnMatmulBcastOpFn = int (*)(const ZdnnZTensor*, const ZdnnZTensor*, const ZdnnZTensor*, int, ZdnnZTensor*);
using ZdnnMatmulTransposeOpFn = int (*)(const ZdnnZTensor*, const ZdnnZTensor*, const ZdnnZTensor*, bool, bool, int,
                                        ZdnnZTensor*);

struct ZdnnDynApi final {
  void* handle{};

  ZdnnInitFn init{};
  ZdnnIsNnpaInstalledFn is_nnpa_installed{};
  ZdnnIsNnpaFunctionInstalledFn is_nnpa_function_installed{};
  ZdnnGetStatusMessageFn get_status_message{};
  ZdnnGetNnpaMaxDimIdxSizeFn get_nnpa_max_dim_idx_size{};
  ZdnnInitPreTransformedDescFn init_pre_transformed_desc{};
  ZdnnGenerateTransformedDescFn generate_transformed_desc{};
  ZdnnInitZTensorFn init_ztensor{};
  ZdnnAllocHelperZTensorFn allochelper_ztensor{};
  ZdnnFreeZTensorBufferFn free_ztensor_buffer{};
  ZdnnTransformZTensorFn transform_ztensor{};
  ZdnnTransformOrigTensorFn transform_origtensor{};

  ZdnnBinaryFn add{};
  ZdnnBinaryFn sub{};
  ZdnnBinaryFn mul{};
  ZdnnBinaryFn div{};
  ZdnnBinaryFn min{};
  ZdnnBinaryFn max{};
  ZdnnReluFn relu{};
  ZdnnUnaryFn gelu{};
  ZdnnUnaryFn tanh{};
  ZdnnUnaryFn sigmoid{};
  ZdnnUnaryFn exp{};
  ZdnnUnaryFn log{};
  ZdnnUnaryFn sqrt{};
  ZdnnSoftmaxFn softmax{};
  ZdnnMomentsFn moments{};
  ZdnnLayernormFn layernorm{};

  ZdnnMatmulOpFn matmul_op{};
  ZdnnMatmulBcastOpFn matmul_bcast_op{};
  ZdnnMatmulTransposeOpFn matmul_transpose_op{};

  bool IsLoaded() const noexcept { return handle != nullptr; }
};

template <typename Fn>
void LoadZdnnSymbol(void* handle, const char* symbol_name, Fn& fn) {
  fn = reinterpret_cast<Fn>(::dlsym(handle, symbol_name));
}

const ZdnnDynApi& GetZdnnApi() noexcept {
  // Keep libzdnn loaded for process lifetime to avoid teardown ordering issues.
  static const ZdnnDynApi api = []() {
    ZdnnDynApi a{};

    a.handle = ::dlopen("libzdnn.so", RTLD_NOW | RTLD_LOCAL);
    if (a.handle == nullptr) {
      return a;
    }

    LoadZdnnSymbol(a.handle, "zdnn_init", a.init);
    LoadZdnnSymbol(a.handle, "zdnn_is_nnpa_installed", a.is_nnpa_installed);
    LoadZdnnSymbol(a.handle, "zdnn_is_nnpa_function_installed", a.is_nnpa_function_installed);
    LoadZdnnSymbol(a.handle, "zdnn_get_status_message", a.get_status_message);
    LoadZdnnSymbol(a.handle, "zdnn_get_nnpa_max_dim_idx_size", a.get_nnpa_max_dim_idx_size);
    LoadZdnnSymbol(a.handle, "zdnn_init_pre_transformed_desc", a.init_pre_transformed_desc);
    LoadZdnnSymbol(a.handle, "zdnn_generate_transformed_desc", a.generate_transformed_desc);
    LoadZdnnSymbol(a.handle, "zdnn_init_ztensor", a.init_ztensor);
    LoadZdnnSymbol(a.handle, "zdnn_allochelper_ztensor", a.allochelper_ztensor);
    LoadZdnnSymbol(a.handle, "zdnn_free_ztensor_buffer", a.free_ztensor_buffer);
    LoadZdnnSymbol(a.handle, "zdnn_transform_ztensor", a.transform_ztensor);
    LoadZdnnSymbol(a.handle, "zdnn_transform_origtensor", a.transform_origtensor);

    LoadZdnnSymbol(a.handle, "zdnn_add", a.add);
    LoadZdnnSymbol(a.handle, "zdnn_sub", a.sub);
    LoadZdnnSymbol(a.handle, "zdnn_mul", a.mul);
    LoadZdnnSymbol(a.handle, "zdnn_div", a.div);
    LoadZdnnSymbol(a.handle, "zdnn_min", a.min);
    LoadZdnnSymbol(a.handle, "zdnn_max", a.max);
    LoadZdnnSymbol(a.handle, "zdnn_relu", a.relu);
    LoadZdnnSymbol(a.handle, "zdnn_gelu", a.gelu);
    LoadZdnnSymbol(a.handle, "zdnn_tanh", a.tanh);
    LoadZdnnSymbol(a.handle, "zdnn_sigmoid", a.sigmoid);
    LoadZdnnSymbol(a.handle, "zdnn_exp", a.exp);
    LoadZdnnSymbol(a.handle, "zdnn_log", a.log);
    LoadZdnnSymbol(a.handle, "zdnn_sqrt", a.sqrt);
    LoadZdnnSymbol(a.handle, "zdnn_softmax", a.softmax);
    LoadZdnnSymbol(a.handle, "zdnn_moments", a.moments);
    LoadZdnnSymbol(a.handle, "zdnn_layernorm", a.layernorm);

    LoadZdnnSymbol(a.handle, "zdnn_matmul_op", a.matmul_op);
    LoadZdnnSymbol(a.handle, "zdnn_matmul_bcast_op", a.matmul_bcast_op);
    LoadZdnnSymbol(a.handle, "zdnn_matmul_transpose_op", a.matmul_transpose_op);

    const bool ok = a.init && a.is_nnpa_installed && a.is_nnpa_function_installed &&
                    a.get_status_message && a.get_nnpa_max_dim_idx_size &&
                    a.init_pre_transformed_desc && a.generate_transformed_desc &&
                    a.init_ztensor && a.allochelper_ztensor && a.free_ztensor_buffer &&
                    a.transform_ztensor && a.transform_origtensor &&
                    a.add && a.sub && a.mul && a.div && a.min && a.max &&
                    a.relu && a.gelu && a.tanh && a.sigmoid && a.exp && a.log && a.sqrt &&
                    a.softmax && a.moments && a.layernorm &&
                    a.matmul_op && a.matmul_bcast_op && a.matmul_transpose_op;
    if (!ok) {
      ::dlclose(a.handle);
      a.handle = nullptr;
      return a;
    }

    a.init();
    return a;
  }();

  return api;
}

class ZdnnTelumBackend final : public TelumBackend {
 public:
  explicit ZdnnTelumBackend(const OrtApi& api)
      : api_(api), zdnn_(GetZdnnApi()) {
    if (!zdnn_.IsLoaded()) {
      reason_ = "libzdnn.so not available (dlopen/dlsym failed)";
      return;
    }

    const bool nnpa_installed = zdnn_.is_nnpa_installed();
    if (!nnpa_installed) {
      reason_ = "NNPA is not installed/enabled on this host";
      return;
    }

    max_dim_idx_size_ = zdnn_.get_nnpa_max_dim_idx_size();

    supports_add_ = IsNnpaFunctionInstalled(kNnpaAdd);
    supports_sub_ = IsNnpaFunctionInstalled(kNnpaSub);
    supports_mul_ = IsNnpaFunctionInstalled(kNnpaMul);
    supports_div_ = IsNnpaFunctionInstalled(kNnpaDiv);
    supports_min_ = IsNnpaFunctionInstalled(kNnpaMin);
    supports_max_ = IsNnpaFunctionInstalled(kNnpaMax);
    supports_relu_ = IsNnpaFunctionInstalled(kNnpaRelu);
    supports_gelu_ = IsNnpaFunctionInstalled(kNnpaGelu);
    supports_tanh_ = IsNnpaFunctionInstalled(kNnpaTanh);
    supports_sigmoid_ = IsNnpaFunctionInstalled(kNnpaSigmoid);
    supports_exp_ = IsNnpaFunctionInstalled(kNnpaExp);
    supports_log_ = IsNnpaFunctionInstalled(kNnpaLog);
    supports_sqrt_ = IsNnpaFunctionInstalled(kNnpaSqrt);
    supports_softmax_ = IsNnpaFunctionInstalled(kNnpaSoftmax);
    supports_layernorm_ = IsNnpaFunctionInstalled(kNnpaMoments) && IsNnpaFunctionInstalled(kNnpaLayernorm);
    supports_matmul_ = IsNnpaFunctionInstalled(kNnpaMatmulOp) ||
                       IsNnpaFunctionInstalled(kNnpaMatmulOpBcast23) ||
                       IsNnpaFunctionInstalled(kNnpaMatmulOpBcast1);

    runtime_ready_ = supports_add_ || supports_sub_ || supports_mul_ || supports_div_ ||
                     supports_min_ || supports_max_ || supports_relu_ || supports_gelu_ ||
                     supports_tanh_ || supports_sigmoid_ || supports_exp_ || supports_log_ ||
                     supports_sqrt_ || supports_softmax_ || supports_layernorm_ || supports_matmul_;

    if (!runtime_ready_) {
      reason_ = "zDNN runtime loaded but required NNPA functions are unavailable";
      return;
    }

    reason_ = "ready";
  }

  std::string BackendKind() const override { return kTelumBackendKindZdnn; }

  bool IsRuntimeReady() const noexcept override { return runtime_ready_; }

  std::string RuntimeStatusMessage() const override { return reason_; }

  uint32_t MaxDimIdxSize() const noexcept override { return max_dim_idx_size_; }

  bool SupportsMul() const noexcept override { return supports_mul_; }

  bool SupportsOp(telum::OpKind op_kind) const noexcept override {
    switch (op_kind) {
      case telum::OpKind::kAdd:
        return supports_add_;
      case telum::OpKind::kSub:
        return supports_sub_;
      case telum::OpKind::kMul:
        return supports_mul_;
      case telum::OpKind::kDiv:
        return supports_div_;
      case telum::OpKind::kMin:
        return supports_min_;
      case telum::OpKind::kMax:
        return supports_max_;
      case telum::OpKind::kRelu:
        return supports_relu_;
      case telum::OpKind::kGelu:
        return supports_gelu_;
      case telum::OpKind::kTanh:
        return supports_tanh_;
      case telum::OpKind::kSigmoid:
        return supports_sigmoid_;
      case telum::OpKind::kExp:
        return supports_exp_;
      case telum::OpKind::kLog:
        return supports_log_;
      case telum::OpKind::kSqrt:
        return supports_sqrt_;
      case telum::OpKind::kSoftmax:
        return supports_softmax_;
      case telum::OpKind::kLayerNormalization:
        return supports_layernorm_;
      case telum::OpKind::kMatMul:
      case telum::OpKind::kGemm:
        return supports_matmul_;
      default:
        return false;
    }
  }

  TelumMulTrustedFn GetMulTrustedFn() noexcept override {
    if (!supports_mul_) {
      return {};
    }

    return TelumMulTrustedFn{&MulTrusted, this};
  }

  OrtStatus* Mul(gsl::span<const float> input0,
                 gsl::span<const float> input1,
                 gsl::span<float> output) noexcept override {
    return Binary(TelumBinaryRequest{telum::OpKind::kMul, input0, input1, output});
  }

  OrtStatus* Binary(const TelumBinaryRequest& request) noexcept override {
    if (!runtime_ready_) {
      const std::string msg = "zDNN backend is not ready: " + reason_;
      return api_.CreateStatus(ORT_FAIL, msg.c_str());
    }

    const size_t num_elems = request.input_a.size();
    if (num_elems != request.input_b.size() || num_elems != request.output.size()) {
      return api_.CreateStatus(ORT_INVALID_ARGUMENT,
                               "zDNN backend expected same number of elements for binary op");
    }

    const auto resolved = ResolveBinaryFn(request.op_kind);
    if (resolved.fn == nullptr) {
      const std::string msg = "zDNN backend does not support binary op kind " +
                              std::to_string(static_cast<int>(request.op_kind));
      return api_.CreateStatus(ORT_FAIL, msg.c_str());
    }

    return BinaryImpl(request.input_a.data(), request.input_b.data(), request.output.data(),
                      num_elems, resolved.fn, resolved.fn_name);
  }

  OrtStatus* Unary(const TelumUnaryRequest& request) noexcept override {
    if (!runtime_ready_) {
      const std::string msg = "zDNN backend is not ready: " + reason_;
      return api_.CreateStatus(ORT_FAIL, msg.c_str());
    }

    const size_t num_elems = request.input.size();
    if (num_elems != request.output.size()) {
      return api_.CreateStatus(ORT_INVALID_ARGUMENT,
                               "zDNN backend expected same number of elements for unary op");
    }

    const auto resolved = ResolveUnaryFn(request.op_kind);
    if (resolved.fn == nullptr && resolved.relu_fn == nullptr) {
      const std::string msg = "zDNN backend does not support unary op kind " +
                              std::to_string(static_cast<int>(request.op_kind));
      return api_.CreateStatus(ORT_FAIL, msg.c_str());
    }

    return UnaryImpl(request.input.data(), request.output.data(), num_elems, resolved);
  }

  OrtStatus* Softmax(const TelumSoftmaxRequest& request) noexcept override {
    if (!supports_softmax_) {
      const std::string msg = "zDNN backend cannot execute Softmax: " + reason_;
      return api_.CreateStatus(ORT_FAIL, msg.c_str());
    }

    if (request.input_shape.empty()) {
      return api_.CreateStatus(ORT_INVALID_ARGUMENT, "zDNN Softmax requires rank >= 1");
    }

    int64_t axis = request.axis;
    if (axis < 0) {
      axis += static_cast<int64_t>(request.input_shape.size());
    }
    if (axis < 0 || axis >= static_cast<int64_t>(request.input_shape.size())) {
      return api_.CreateStatus(ORT_INVALID_ARGUMENT, "zDNN Softmax axis is out of range");
    }
    if (axis != static_cast<int64_t>(request.input_shape.size() - 1)) {
      return api_.CreateStatus(ORT_INVALID_ARGUMENT,
                               "zDNN Softmax supports axis on the last dimension only");
    }

    size_t num_elems = 1;
    size_t batch = 1;
    for (size_t i = 0; i < request.input_shape.size(); ++i) {
      const int64_t d = request.input_shape[i];
      if (d <= 0) {
        return api_.CreateStatus(ORT_INVALID_ARGUMENT, "zDNN Softmax requires positive static dimensions");
      }
      if (num_elems > std::numeric_limits<size_t>::max() / static_cast<size_t>(d)) {
        return api_.CreateStatus(ORT_INVALID_ARGUMENT, "zDNN Softmax element count overflow");
      }
      num_elems *= static_cast<size_t>(d);
      if (i + 1 < request.input_shape.size()) {
        if (batch > std::numeric_limits<size_t>::max() / static_cast<size_t>(d)) {
          return api_.CreateStatus(ORT_INVALID_ARGUMENT, "zDNN Softmax batch size overflow");
        }
        batch *= static_cast<size_t>(d);
      }
    }

    const int64_t vector_len = request.input_shape.back();
    if (request.input.size() != num_elems || request.output.size() != num_elems) {
      return api_.CreateStatus(ORT_INVALID_ARGUMENT,
                               "zDNN Softmax tensor size does not match input shape");
    }
    if (!ValidateDimsAgainstLimit({static_cast<int64_t>(batch), 1, vector_len})) {
      return api_.CreateStatus(ORT_INVALID_ARGUMENT,
                               "zDNN Softmax dimensions exceed host NNPA max_dim_idx_size");
    }

    OwnedZTensor z_x(&zdnn_);
    OwnedZTensor z_y(&zdnn_);

    const uint32_t softmax_dims[] = {static_cast<uint32_t>(batch), 1U, static_cast<uint32_t>(vector_len)};
    RETURN_IF_ERROR(InitTensor(kZdnnLayout3DS, gsl::span<const uint32_t>(softmax_dims, 3), z_x));
    RETURN_IF_ERROR(InitTensor(kZdnnLayout3DS, gsl::span<const uint32_t>(softmax_dims, 3), z_y));

    z_x.ztensor.is_transformed = false;
    z_y.ztensor.is_transformed = false;

    int st = zdnn_.transform_ztensor(&z_x.ztensor, request.input.data());
    if (st != kZdnnOk) {
      const std::string msg = std::string("zdnn_transform_ztensor(Softmax input) failed: ") +
                              GetZdnnStatusMessage(st);
      return api_.CreateStatus(ORT_FAIL, msg.c_str());
    }

    st = zdnn_.softmax(&z_x.ztensor, nullptr, kSoftmaxActNone, &z_y.ztensor);
    if (st != kZdnnOk) {
      const std::string msg = std::string("zdnn_softmax failed: ") + GetZdnnStatusMessage(st);
      return api_.CreateStatus(ORT_FAIL, msg.c_str());
    }

    st = zdnn_.transform_origtensor(&z_y.ztensor, request.output.data());
    if (st != kZdnnOk) {
      const std::string msg = std::string("zdnn_transform_origtensor(Softmax output) failed: ") +
                              GetZdnnStatusMessage(st);
      return api_.CreateStatus(ORT_FAIL, msg.c_str());
    }

    return nullptr;
  }

  OrtStatus* LayerNormalization(const TelumLayerNormRequest& request) noexcept override {
    if (!supports_layernorm_) {
      const std::string msg = "zDNN backend cannot execute LayerNormalization: " + reason_;
      return api_.CreateStatus(ORT_FAIL, msg.c_str());
    }

    if (request.input_shape.empty()) {
      return api_.CreateStatus(ORT_INVALID_ARGUMENT, "zDNN LayerNormalization requires rank >= 1");
    }

    int64_t axis = request.axis;
    if (axis < 0) {
      axis += static_cast<int64_t>(request.input_shape.size());
    }
    if (axis < 0 || axis >= static_cast<int64_t>(request.input_shape.size())) {
      return api_.CreateStatus(ORT_INVALID_ARGUMENT, "zDNN LayerNormalization axis is out of range");
    }
    if (axis != static_cast<int64_t>(request.input_shape.size() - 1)) {
      return api_.CreateStatus(ORT_INVALID_ARGUMENT,
                               "zDNN LayerNormalization supports axis on the last dimension only");
    }

    int64_t n = 1;
    for (size_t i = 0; i + 1 < request.input_shape.size(); ++i) {
      const int64_t d = request.input_shape[i];
      if (d <= 0) {
        return api_.CreateStatus(ORT_INVALID_ARGUMENT,
                                 "zDNN LayerNormalization requires positive static dimensions");
      }
      if (n > std::numeric_limits<int64_t>::max() / d) {
        return api_.CreateStatus(ORT_INVALID_ARGUMENT, "zDNN LayerNormalization batch size overflow");
      }
      n *= d;
    }
    const int64_t c = request.input_shape.back();
    if (c <= 0) {
      return api_.CreateStatus(ORT_INVALID_ARGUMENT, "zDNN LayerNormalization hidden size must be positive");
    }
    if (!ValidateDimsAgainstLimit({n, c})) {
      return api_.CreateStatus(ORT_INVALID_ARGUMENT,
                               "zDNN LayerNormalization dimensions exceed host NNPA max_dim_idx_size");
    }

    const size_t expected_x = static_cast<size_t>(n) * static_cast<size_t>(c);
    if (request.input.size() != expected_x || request.output.size() != expected_x) {
      return api_.CreateStatus(ORT_INVALID_ARGUMENT,
                               "zDNN LayerNormalization tensor size does not match input shape");
    }

    if (request.scale_shape.size() != 1 || request.scale_shape[0] != c ||
        request.scale.size() != static_cast<size_t>(c)) {
      return api_.CreateStatus(ORT_INVALID_ARGUMENT,
                               "zDNN LayerNormalization requires scale shape [C]");
    }
    if (request.has_bias &&
        (request.bias_shape.size() != 1 || request.bias_shape[0] != c ||
         request.bias.size() != static_cast<size_t>(c))) {
      return api_.CreateStatus(ORT_INVALID_ARGUMENT,
                               "zDNN LayerNormalization requires bias shape [C] when provided");
    }

    OwnedZTensor z_x(&zdnn_);
    OwnedZTensor z_mean(&zdnn_);
    OwnedZTensor z_var(&zdnn_);
    OwnedZTensor z_y(&zdnn_);

    const uint32_t ln_x_dims[] = {static_cast<uint32_t>(n), 1U, 1U, static_cast<uint32_t>(c)};
    const uint32_t ln_moments_dims[] = {static_cast<uint32_t>(n), 1U, 1U, 1U};
    RETURN_IF_ERROR(InitTensor(kZdnnLayoutNhwc, gsl::span<const uint32_t>(ln_x_dims, 4), z_x));
    RETURN_IF_ERROR(InitTensor(kZdnnLayoutNhwc, gsl::span<const uint32_t>(ln_moments_dims, 4), z_mean));
    RETURN_IF_ERROR(InitTensor(kZdnnLayoutNhwc, gsl::span<const uint32_t>(ln_moments_dims, 4), z_var));
    RETURN_IF_ERROR(InitTensor(kZdnnLayoutNhwc, gsl::span<const uint32_t>(ln_x_dims, 4), z_y));

    z_x.ztensor.is_transformed = false;
    z_mean.ztensor.is_transformed = false;
    z_var.ztensor.is_transformed = false;
    z_y.ztensor.is_transformed = false;

    int st = zdnn_.transform_ztensor(&z_x.ztensor, request.input.data());
    if (st != kZdnnOk) {
      const std::string msg = std::string("zdnn_transform_ztensor(LayerNorm input) failed: ") +
                              GetZdnnStatusMessage(st);
      return api_.CreateStatus(ORT_FAIL, msg.c_str());
    }

    st = zdnn_.moments(&z_x.ztensor, kMomentsBesselPopulation, &z_mean.ztensor, &z_var.ztensor);
    if (st != kZdnnOk) {
      const std::string msg = std::string("zdnn_moments failed: ") + GetZdnnStatusMessage(st);
      return api_.CreateStatus(ORT_FAIL, msg.c_str());
    }

    st = zdnn_.layernorm(&z_x.ztensor, &z_mean.ztensor, &z_var.ztensor,
                         /*beta=*/0.0f, /*gamma=*/1.0f, request.epsilon, &z_y.ztensor);
    if (st != kZdnnOk) {
      const std::string msg = std::string("zdnn_layernorm failed: ") + GetZdnnStatusMessage(st);
      return api_.CreateStatus(ORT_FAIL, msg.c_str());
    }

    st = zdnn_.transform_origtensor(&z_y.ztensor, request.output.data());
    if (st != kZdnnOk) {
      const std::string msg = std::string("zdnn_transform_origtensor(LayerNorm output) failed: ") +
                              GetZdnnStatusMessage(st);
      return api_.CreateStatus(ORT_FAIL, msg.c_str());
    }

    const float* bias = request.has_bias ? request.bias.data() : nullptr;
    for (int64_t i = 0; i < n; ++i) {
      const size_t base = static_cast<size_t>(i) * static_cast<size_t>(c);
      for (int64_t j = 0; j < c; ++j) {
        float v = request.output[base + static_cast<size_t>(j)] * request.scale[static_cast<size_t>(j)];
        if (bias != nullptr) {
          v += bias[static_cast<size_t>(j)];
        }
        request.output[base + static_cast<size_t>(j)] = v;
      }
    }

    return nullptr;
  }

  OrtStatus* MatMul(const TelumMatMulRequest& request) noexcept override {
    if (!supports_matmul_) {
      const std::string msg = "zDNN backend cannot execute MatMul: " + reason_;
      return api_.CreateStatus(ORT_FAIL, msg.c_str());
    }

    MatMulPlan plan{};
    OrtStatus* st = BuildMatMulPlan(request.a_shape, request.b_shape, request.output_shape, plan);
    if (st != nullptr) {
      return st;
    }

    const size_t expected_a = ComputeElementCountOrZero(request.a_shape);
    const size_t expected_b = ComputeElementCountOrZero(request.b_shape);
    const size_t expected_y = ComputeElementCountOrZero(request.output_shape);
    if (request.input_a.size() != expected_a || request.input_b.size() != expected_b ||
        request.output.size() != expected_y) {
      return api_.CreateStatus(ORT_INVALID_ARGUMENT,
                               "zDNN MatMul tensor size does not match provided shape");
    }

    const size_t bias_size = plan.kind == MatMulPlan::Kind::kStacked || plan.kind == MatMulPlan::Kind::kBcast1
                                 ? static_cast<size_t>(plan.stack) * static_cast<size_t>(plan.n)
                                 : static_cast<size_t>(plan.n);
    std::vector<float> zero_bias(bias_size, 0.0f);

    return RunMatMulPlan(plan, request.input_a.data(), request.input_b.data(), zero_bias.data(), request.output.data());
  }

  OrtStatus* Gemm(const TelumGemmRequest& request) noexcept override {
    if (!supports_matmul_) {
      const std::string msg = "zDNN backend cannot execute Gemm: " + reason_;
      return api_.CreateStatus(ORT_FAIL, msg.c_str());
    }

    if (request.a_shape.size() != 2 || request.b_shape.size() != 2 || request.output_shape.size() != 2) {
      return api_.CreateStatus(ORT_INVALID_ARGUMENT, "zDNN Gemm currently supports rank-2 tensors only");
    }

    if (request.alpha != 1.0f || request.beta != 1.0f) {
      return api_.CreateStatus(ORT_INVALID_ARGUMENT,
                               "zDNN Gemm backend supports alpha=1 and beta=1 only");
    }

    const int64_t a_rows = request.trans_a ? request.a_shape[1] : request.a_shape[0];
    const int64_t a_cols = request.trans_a ? request.a_shape[0] : request.a_shape[1];
    const int64_t b_rows = request.trans_b ? request.b_shape[1] : request.b_shape[0];
    const int64_t b_cols = request.trans_b ? request.b_shape[0] : request.b_shape[1];

    if (a_rows <= 0 || a_cols <= 0 || b_rows <= 0 || b_cols <= 0) {
      return api_.CreateStatus(ORT_INVALID_ARGUMENT, "zDNN Gemm requires positive dimensions");
    }
    if (a_cols != b_rows) {
      return api_.CreateStatus(ORT_INVALID_ARGUMENT, "zDNN Gemm K dimensions mismatch");
    }

    const int64_t m = a_rows;
    const int64_t k = a_cols;
    const int64_t n = b_cols;

    if (request.output_shape[0] != m || request.output_shape[1] != n) {
      return api_.CreateStatus(ORT_INVALID_ARGUMENT, "zDNN Gemm output shape mismatch");
    }

    if (!ValidateDimsAgainstLimit({m, k, n})) {
      return api_.CreateStatus(ORT_INVALID_ARGUMENT,
                               "zDNN Gemm dimensions exceed host NNPA max_dim_idx_size");
    }

    const size_t expected_a = static_cast<size_t>(request.a_shape[0]) * static_cast<size_t>(request.a_shape[1]);
    const size_t expected_b = static_cast<size_t>(request.b_shape[0]) * static_cast<size_t>(request.b_shape[1]);
    const size_t expected_y = static_cast<size_t>(m) * static_cast<size_t>(n);
    if (request.input_a.size() != expected_a || request.input_b.size() != expected_b ||
        request.output.size() != expected_y) {
      return api_.CreateStatus(ORT_INVALID_ARGUMENT,
                               "zDNN Gemm tensor size does not match provided shape");
    }

    std::vector<float> bias(static_cast<size_t>(n), 0.0f);
    if (request.has_c) {
      OrtStatus* st = BuildBiasVector(request, static_cast<size_t>(n), bias);
      if (st != nullptr) {
        return st;
      }
    }

    return RunMatMul(request.input_a.data(), request.input_b.data(), bias.data(), request.output.data(),
                     static_cast<uint32_t>(m), static_cast<uint32_t>(k), static_cast<uint32_t>(n),
                     request.trans_a, request.trans_b);
  }

 private:
  struct BinaryFnInfo {
    ZdnnBinaryFn fn{};
    const char* fn_name{};
  };

  struct UnaryFnInfo {
    ZdnnUnaryFn fn{};
    ZdnnReluFn relu_fn{};
    const char* fn_name{};
  };

  struct BinaryCtx final {
    const ZdnnDynApi* zdnn{};
    ZdnnTensorDesc pre_desc{};
    ZdnnTensorDesc tfrmd_desc{};
    ZdnnZTensor a{};
    ZdnnZTensor b{};
    ZdnnZTensor c{};
    bool initialized{};

    explicit BinaryCtx(const ZdnnDynApi* zdnn_api)
        : zdnn(zdnn_api) {}

    BinaryCtx(const BinaryCtx&) = delete;
    BinaryCtx& operator=(const BinaryCtx&) = delete;

    ~BinaryCtx() {
      if (!initialized || zdnn == nullptr || zdnn->free_ztensor_buffer == nullptr) {
        return;
      }

      (void)zdnn->free_ztensor_buffer(&a);
      (void)zdnn->free_ztensor_buffer(&b);
      (void)zdnn->free_ztensor_buffer(&c);
    }
  };

  struct UnaryCtx final {
    const ZdnnDynApi* zdnn{};
    ZdnnTensorDesc pre_desc{};
    ZdnnTensorDesc tfrmd_desc{};
    ZdnnZTensor a{};
    ZdnnZTensor c{};
    bool initialized{};

    explicit UnaryCtx(const ZdnnDynApi* zdnn_api)
        : zdnn(zdnn_api) {}

    UnaryCtx(const UnaryCtx&) = delete;
    UnaryCtx& operator=(const UnaryCtx&) = delete;

    ~UnaryCtx() {
      if (!initialized || zdnn == nullptr || zdnn->free_ztensor_buffer == nullptr) {
        return;
      }

      (void)zdnn->free_ztensor_buffer(&a);
      (void)zdnn->free_ztensor_buffer(&c);
    }
  };

  struct OwnedZTensor final {
    const ZdnnDynApi* zdnn{};
    ZdnnTensorDesc pre_desc{};
    ZdnnTensorDesc tfrmd_desc{};
    ZdnnZTensor ztensor{};
    bool allocated{};

    explicit OwnedZTensor(const ZdnnDynApi* api)
        : zdnn(api) {}

    OwnedZTensor(const OwnedZTensor&) = delete;
    OwnedZTensor& operator=(const OwnedZTensor&) = delete;

    ~OwnedZTensor() {
      if (allocated && zdnn != nullptr && zdnn->free_ztensor_buffer != nullptr) {
        (void)zdnn->free_ztensor_buffer(&ztensor);
      }
    }
  };

  struct MatMulPlan final {
    enum class Kind { kUnstacked, kStacked, kBcast1, kBcast23 };

    Kind kind{Kind::kUnstacked};
    uint32_t m{};
    uint32_t k{};
    uint32_t n{};
    uint32_t stack{1};

    uint32_t layout_a{kZdnnLayout2D};
    uint32_t layout_b{kZdnnLayout2D};
    uint32_t layout_c{kZdnnLayout1D};
    uint32_t layout_y{kZdnnLayout2D};

    std::vector<uint32_t> logical_shape_a;
    std::vector<uint32_t> logical_shape_b;
    std::vector<uint32_t> logical_shape_c;
    std::vector<uint32_t> logical_shape_y;
  };

  static bool ComputeElementCount(gsl::span<const int64_t> shape, size_t& count) noexcept {
    count = 1;
    for (int64_t d : shape) {
      if (d <= 0) {
        return false;
      }
      if (count > std::numeric_limits<size_t>::max() / static_cast<size_t>(d)) {
        return false;
      }
      count *= static_cast<size_t>(d);
    }
    return true;
  }

  static size_t ComputeElementCountOrZero(gsl::span<const int64_t> shape) noexcept {
    size_t count = 0;
    return ComputeElementCount(shape, count) ? count : 0;
  }

  static bool AllOnes(const std::vector<int64_t>& dims) noexcept {
    return std::all_of(dims.begin(), dims.end(), [](int64_t d) { return d == 1; });
  }

  static std::vector<int64_t> AlignBatchDims(const std::vector<int64_t>& batch_dims, size_t out_rank) {
    if (batch_dims.size() >= out_rank) {
      return batch_dims;
    }
    std::vector<int64_t> aligned(out_rank - batch_dims.size(), 1);
    aligned.insert(aligned.end(), batch_dims.begin(), batch_dims.end());
    return aligned;
  }

  static bool BroadcastBatchDims(const std::vector<int64_t>& a_batch,
                                 const std::vector<int64_t>& b_batch,
                                 std::vector<int64_t>& out_batch) noexcept {
    const size_t rank_a = a_batch.size();
    const size_t rank_b = b_batch.size();
    const size_t out_rank = std::max(rank_a, rank_b);
    out_batch.assign(out_rank, 1);

    for (size_t i = 0; i < out_rank; ++i) {
      const int64_t dim_a = (i < rank_a) ? a_batch[rank_a - 1 - i] : 1;
      const int64_t dim_b = (i < rank_b) ? b_batch[rank_b - 1 - i] : 1;
      if (dim_a <= 0 || dim_b <= 0) {
        return false;
      }

      if (dim_a == dim_b) {
        out_batch[out_rank - 1 - i] = dim_a;
      } else if (dim_a == 1) {
        out_batch[out_rank - 1 - i] = dim_b;
      } else if (dim_b == 1) {
        out_batch[out_rank - 1 - i] = dim_a;
      } else {
        return false;
      }
    }

    return true;
  }

  OrtStatus* BuildMatMulPlan(gsl::span<const int64_t> a_shape,
                             gsl::span<const int64_t> b_shape,
                             gsl::span<const int64_t> output_shape,
                             MatMulPlan& plan) noexcept {
    if (a_shape.size() < 2 || b_shape.size() < 2) {
      return api_.CreateStatus(ORT_INVALID_ARGUMENT,
                               "zDNN MatMul requires rank >= 2 tensors");
    }

    for (int64_t d : a_shape) {
      if (d <= 0) {
        return api_.CreateStatus(ORT_INVALID_ARGUMENT, "zDNN MatMul A shape must be positive/static");
      }
    }
    for (int64_t d : b_shape) {
      if (d <= 0) {
        return api_.CreateStatus(ORT_INVALID_ARGUMENT, "zDNN MatMul B shape must be positive/static");
      }
    }

    const int64_t m = a_shape[a_shape.size() - 2];
    const int64_t k_a = a_shape[a_shape.size() - 1];
    const int64_t k_b = b_shape[b_shape.size() - 2];
    const int64_t n = b_shape[b_shape.size() - 1];
    if (k_a != k_b) {
      return api_.CreateStatus(ORT_INVALID_ARGUMENT, "zDNN MatMul requires matching K dimensions");
    }

    std::vector<int64_t> a_batch(a_shape.begin(), a_shape.end() - 2);
    std::vector<int64_t> b_batch(b_shape.begin(), b_shape.end() - 2);
    std::vector<int64_t> out_batch;
    if (!BroadcastBatchDims(a_batch, b_batch, out_batch)) {
      return api_.CreateStatus(ORT_INVALID_ARGUMENT,
                               "zDNN MatMul batch dimensions are not broadcast-compatible");
    }

    std::vector<int64_t> expected_output(out_batch);
    expected_output.push_back(m);
    expected_output.push_back(n);
    if (output_shape.size() != expected_output.size() ||
        !std::equal(output_shape.begin(), output_shape.end(), expected_output.begin())) {
      return api_.CreateStatus(ORT_INVALID_ARGUMENT, "zDNN MatMul output shape mismatch");
    }

    int64_t stack_i64 = 1;
    for (int64_t d : out_batch) {
      if (stack_i64 > std::numeric_limits<int64_t>::max() / d) {
        return api_.CreateStatus(ORT_INVALID_ARGUMENT, "zDNN MatMul stack dimension overflow");
      }
      stack_i64 *= d;
    }

    if (!ValidateDimsAgainstLimit({m, k_a, n, stack_i64})) {
      return api_.CreateStatus(ORT_INVALID_ARGUMENT,
                               "zDNN MatMul dimensions exceed host NNPA max_dim_idx_size");
    }

    const auto a_aligned = AlignBatchDims(a_batch, out_batch.size());
    const auto b_aligned = AlignBatchDims(b_batch, out_batch.size());
    const bool a_matches = (a_aligned == out_batch);
    const bool b_matches = (b_aligned == out_batch);
    const bool a_all_ones = AllOnes(a_aligned);
    const bool b_all_ones = AllOnes(b_aligned);

    plan = MatMulPlan{};
    plan.m = static_cast<uint32_t>(m);
    plan.k = static_cast<uint32_t>(k_a);
    plan.n = static_cast<uint32_t>(n);
    plan.stack = static_cast<uint32_t>(stack_i64);

    if (stack_i64 == 1) {
      if (!a_matches || !b_matches) {
        return api_.CreateStatus(ORT_INVALID_ARGUMENT,
                                 "zDNN MatMul rank-2 mapping requires non-partial batch broadcast");
      }
      plan.kind = MatMulPlan::Kind::kUnstacked;
      plan.layout_a = kZdnnLayout2D;
      plan.layout_b = kZdnnLayout2D;
      plan.layout_c = kZdnnLayout1D;
      plan.layout_y = kZdnnLayout2D;
      plan.logical_shape_a = {plan.m, plan.k};
      plan.logical_shape_b = {plan.k, plan.n};
      plan.logical_shape_c = {plan.n};
      plan.logical_shape_y = {plan.m, plan.n};
      return nullptr;
    }

    if (a_matches && b_matches) {
      plan.kind = MatMulPlan::Kind::kStacked;
      plan.layout_a = kZdnnLayout3DS;
      plan.layout_b = kZdnnLayout3DS;
      plan.layout_c = kZdnnLayout2DS;
      plan.layout_y = kZdnnLayout3DS;
      plan.logical_shape_a = {plan.stack, plan.m, plan.k};
      plan.logical_shape_b = {plan.stack, plan.k, plan.n};
      plan.logical_shape_c = {plan.stack, plan.n};
      plan.logical_shape_y = {plan.stack, plan.m, plan.n};
      return nullptr;
    }

    if (a_matches && b_all_ones) {
      plan.kind = MatMulPlan::Kind::kBcast23;
      plan.layout_a = kZdnnLayout3DS;
      plan.layout_b = kZdnnLayout2D;
      plan.layout_c = kZdnnLayout1D;
      plan.layout_y = kZdnnLayout3DS;
      plan.logical_shape_a = {plan.stack, plan.m, plan.k};
      plan.logical_shape_b = {plan.k, plan.n};
      plan.logical_shape_c = {plan.n};
      plan.logical_shape_y = {plan.stack, plan.m, plan.n};
      return nullptr;
    }

    if (a_all_ones && b_matches) {
      plan.kind = MatMulPlan::Kind::kBcast1;
      plan.layout_a = kZdnnLayout2D;
      plan.layout_b = kZdnnLayout3DS;
      plan.layout_c = kZdnnLayout2DS;
      plan.layout_y = kZdnnLayout3DS;
      plan.logical_shape_a = {plan.m, plan.k};
      plan.logical_shape_b = {plan.stack, plan.k, plan.n};
      plan.logical_shape_c = {plan.stack, plan.n};
      plan.logical_shape_y = {plan.stack, plan.m, plan.n};
      return nullptr;
    }

    return api_.CreateStatus(
        ORT_INVALID_ARGUMENT,
        "zDNN MatMul supports unstacked, stacked, or fully-unstacked broadcast operand patterns only");
  }

  bool IsNnpaFunctionInstalled(int fn) const noexcept {
    return zdnn_.is_nnpa_function_installed(1, fn);
  }

  bool ValidateDimsAgainstLimit(std::initializer_list<int64_t> dims) const noexcept {
    if (max_dim_idx_size_ == 0) {
      return true;
    }

    for (int64_t d : dims) {
      if (d < 0 || static_cast<uint64_t>(d) > static_cast<uint64_t>(max_dim_idx_size_)) {
        return false;
      }
    }

    return true;
  }

  static bool Choose2DDims(size_t num_elems, uint32_t max_dim_idx_size,
                           uint32_t& dim2, uint32_t& dim1) noexcept {
    constexpr uint32_t kDefaultDimLimit = 32768;
    const uint32_t dim_limit = max_dim_idx_size == 0 ? kDefaultDimLimit : max_dim_idx_size;

    if (num_elems == 0) {
      return false;
    }

    size_t candidate_dim1 = std::min(num_elems, static_cast<size_t>(dim_limit));
    while (candidate_dim1 > 1 && (num_elems % candidate_dim1) != 0) {
      --candidate_dim1;
    }

    if ((num_elems % candidate_dim1) != 0) {
      return false;
    }

    const size_t candidate_dim2 = num_elems / candidate_dim1;
    if (candidate_dim2 > static_cast<size_t>(std::numeric_limits<uint32_t>::max()) ||
        candidate_dim1 > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
      return false;
    }

    if (candidate_dim2 > dim_limit || candidate_dim1 > dim_limit) {
      return false;
    }

    dim2 = static_cast<uint32_t>(candidate_dim2);
    dim1 = static_cast<uint32_t>(candidate_dim1);
    return true;
  }

  static std::unordered_map<size_t, std::unique_ptr<BinaryCtx>>& BinaryCtxCache() {
    static thread_local std::unordered_map<size_t, std::unique_ptr<BinaryCtx>> cache;
    return cache;
  }

  static std::unordered_map<size_t, std::unique_ptr<UnaryCtx>>& UnaryCtxCache() {
    static thread_local std::unordered_map<size_t, std::unique_ptr<UnaryCtx>> cache;
    return cache;
  }

  BinaryCtx* GetOrInitBinaryCtx(size_t num_elems, std::string& error) noexcept {
    auto& cache = BinaryCtxCache();
    auto it = cache.find(num_elems);
    if (it != cache.end()) {
      return it->second.get();
    }

    uint32_t dim2 = 0;
    uint32_t dim1 = 0;
    if (!Choose2DDims(num_elems, max_dim_idx_size_, dim2, dim1)) {
      error = "unable to represent tensor shape for zDNN 2D descriptor";
      return nullptr;
    }

    auto ctx = std::make_unique<BinaryCtx>(&zdnn_);
    zdnn_.init_pre_transformed_desc(kZdnnLayout2D, kZdnnTypeFp32, &ctx->pre_desc, dim2, dim1);

    int st = zdnn_.generate_transformed_desc(&ctx->pre_desc, &ctx->tfrmd_desc);
    if (st != kZdnnOk) {
      error = std::string("zdnn_generate_transformed_desc failed: ") + GetZdnnStatusMessage(st);
      return nullptr;
    }

    zdnn_.init_ztensor(&ctx->pre_desc, &ctx->tfrmd_desc, &ctx->a);
    zdnn_.init_ztensor(&ctx->pre_desc, &ctx->tfrmd_desc, &ctx->b);
    zdnn_.init_ztensor(&ctx->pre_desc, &ctx->tfrmd_desc, &ctx->c);

    st = zdnn_.allochelper_ztensor(&ctx->a);
    if (st != kZdnnOk) {
      error = std::string("zdnn_allochelper_ztensor(input_a) failed: ") + GetZdnnStatusMessage(st);
      return nullptr;
    }

    st = zdnn_.allochelper_ztensor(&ctx->b);
    if (st != kZdnnOk) {
      error = std::string("zdnn_allochelper_ztensor(input_b) failed: ") + GetZdnnStatusMessage(st);
      return nullptr;
    }

    st = zdnn_.allochelper_ztensor(&ctx->c);
    if (st != kZdnnOk) {
      error = std::string("zdnn_allochelper_ztensor(output) failed: ") + GetZdnnStatusMessage(st);
      return nullptr;
    }

    ctx->initialized = true;
    BinaryCtx* out = ctx.get();
    cache.emplace(num_elems, std::move(ctx));
    return out;
  }

  UnaryCtx* GetOrInitUnaryCtx(size_t num_elems, std::string& error) noexcept {
    auto& cache = UnaryCtxCache();
    auto it = cache.find(num_elems);
    if (it != cache.end()) {
      return it->second.get();
    }

    uint32_t dim2 = 0;
    uint32_t dim1 = 0;
    if (!Choose2DDims(num_elems, max_dim_idx_size_, dim2, dim1)) {
      error = "unable to represent tensor shape for zDNN 2D descriptor";
      return nullptr;
    }

    auto ctx = std::make_unique<UnaryCtx>(&zdnn_);
    zdnn_.init_pre_transformed_desc(kZdnnLayout2D, kZdnnTypeFp32, &ctx->pre_desc, dim2, dim1);

    int st = zdnn_.generate_transformed_desc(&ctx->pre_desc, &ctx->tfrmd_desc);
    if (st != kZdnnOk) {
      error = std::string("zdnn_generate_transformed_desc failed: ") + GetZdnnStatusMessage(st);
      return nullptr;
    }

    zdnn_.init_ztensor(&ctx->pre_desc, &ctx->tfrmd_desc, &ctx->a);
    zdnn_.init_ztensor(&ctx->pre_desc, &ctx->tfrmd_desc, &ctx->c);

    st = zdnn_.allochelper_ztensor(&ctx->a);
    if (st != kZdnnOk) {
      error = std::string("zdnn_allochelper_ztensor(input) failed: ") + GetZdnnStatusMessage(st);
      return nullptr;
    }

    st = zdnn_.allochelper_ztensor(&ctx->c);
    if (st != kZdnnOk) {
      error = std::string("zdnn_allochelper_ztensor(output) failed: ") + GetZdnnStatusMessage(st);
      return nullptr;
    }

    ctx->initialized = true;
    UnaryCtx* out = ctx.get();
    cache.emplace(num_elems, std::move(ctx));
    return out;
  }

  BinaryFnInfo ResolveBinaryFn(telum::OpKind op_kind) const noexcept {
    switch (op_kind) {
      case telum::OpKind::kAdd:
        return supports_add_ ? BinaryFnInfo{zdnn_.add, "zdnn_add"} : BinaryFnInfo{};
      case telum::OpKind::kSub:
        return supports_sub_ ? BinaryFnInfo{zdnn_.sub, "zdnn_sub"} : BinaryFnInfo{};
      case telum::OpKind::kMul:
        return supports_mul_ ? BinaryFnInfo{zdnn_.mul, "zdnn_mul"} : BinaryFnInfo{};
      case telum::OpKind::kDiv:
        return supports_div_ ? BinaryFnInfo{zdnn_.div, "zdnn_div"} : BinaryFnInfo{};
      case telum::OpKind::kMin:
        return supports_min_ ? BinaryFnInfo{zdnn_.min, "zdnn_min"} : BinaryFnInfo{};
      case telum::OpKind::kMax:
        return supports_max_ ? BinaryFnInfo{zdnn_.max, "zdnn_max"} : BinaryFnInfo{};
      default:
        return BinaryFnInfo{};
    }
  }

  UnaryFnInfo ResolveUnaryFn(telum::OpKind op_kind) const noexcept {
    switch (op_kind) {
      case telum::OpKind::kRelu:
        return supports_relu_ ? UnaryFnInfo{nullptr, zdnn_.relu, "zdnn_relu"} : UnaryFnInfo{};
      case telum::OpKind::kGelu:
        return supports_gelu_ ? UnaryFnInfo{zdnn_.gelu, nullptr, "zdnn_gelu"} : UnaryFnInfo{};
      case telum::OpKind::kTanh:
        return supports_tanh_ ? UnaryFnInfo{zdnn_.tanh, nullptr, "zdnn_tanh"} : UnaryFnInfo{};
      case telum::OpKind::kSigmoid:
        return supports_sigmoid_ ? UnaryFnInfo{zdnn_.sigmoid, nullptr, "zdnn_sigmoid"} : UnaryFnInfo{};
      case telum::OpKind::kExp:
        return supports_exp_ ? UnaryFnInfo{zdnn_.exp, nullptr, "zdnn_exp"} : UnaryFnInfo{};
      case telum::OpKind::kLog:
        return supports_log_ ? UnaryFnInfo{zdnn_.log, nullptr, "zdnn_log"} : UnaryFnInfo{};
      case telum::OpKind::kSqrt:
        return supports_sqrt_ ? UnaryFnInfo{zdnn_.sqrt, nullptr, "zdnn_sqrt"} : UnaryFnInfo{};
      default:
        return UnaryFnInfo{};
    }
  }

  std::string GetZdnnStatusMessage(int st) const {
    const char* msg = zdnn_.get_status_message(st);
    return msg != nullptr ? std::string(msg) : std::string("unknown zDNN error");
  }

  OrtStatus* BinaryImpl(const float* input0_data,
                        const float* input1_data,
                        float* output_data,
                        size_t num_elems,
                        ZdnnBinaryFn fn,
                        const char* fn_name) noexcept {
    std::string error;
    BinaryCtx* ctx = GetOrInitBinaryCtx(num_elems, error);
    if (ctx == nullptr) {
      const std::string msg = "zDNN binary setup failed: " + error;
      return api_.CreateStatus(ORT_FAIL, msg.c_str());
    }

    ctx->a.is_transformed = false;
    ctx->b.is_transformed = false;
    ctx->c.is_transformed = false;

    int st = zdnn_.transform_ztensor(&ctx->a, input0_data);
    if (st != kZdnnOk) {
      const std::string msg = std::string("zdnn_transform_ztensor(input0) failed: ") + GetZdnnStatusMessage(st);
      return api_.CreateStatus(ORT_FAIL, msg.c_str());
    }

    st = zdnn_.transform_ztensor(&ctx->b, input1_data);
    if (st != kZdnnOk) {
      const std::string msg = std::string("zdnn_transform_ztensor(input1) failed: ") + GetZdnnStatusMessage(st);
      return api_.CreateStatus(ORT_FAIL, msg.c_str());
    }

    st = fn(&ctx->a, &ctx->b, &ctx->c);
    if (st != kZdnnOk) {
      const std::string msg = std::string(fn_name) + " failed: " + GetZdnnStatusMessage(st);
      return api_.CreateStatus(ORT_FAIL, msg.c_str());
    }

    st = zdnn_.transform_origtensor(&ctx->c, output_data);
    if (st != kZdnnOk) {
      const std::string msg = std::string("zdnn_transform_origtensor(output) failed: ") + GetZdnnStatusMessage(st);
      return api_.CreateStatus(ORT_FAIL, msg.c_str());
    }

    return nullptr;
  }

  OrtStatus* UnaryImpl(const float* input_data,
                       float* output_data,
                       size_t num_elems,
                       const UnaryFnInfo& info) noexcept {
    std::string error;
    UnaryCtx* ctx = GetOrInitUnaryCtx(num_elems, error);
    if (ctx == nullptr) {
      const std::string msg = "zDNN unary setup failed: " + error;
      return api_.CreateStatus(ORT_FAIL, msg.c_str());
    }

    ctx->a.is_transformed = false;
    ctx->c.is_transformed = false;

    int st = zdnn_.transform_ztensor(&ctx->a, input_data);
    if (st != kZdnnOk) {
      const std::string msg = std::string("zdnn_transform_ztensor(input) failed: ") + GetZdnnStatusMessage(st);
      return api_.CreateStatus(ORT_FAIL, msg.c_str());
    }

    if (info.relu_fn != nullptr) {
      st = info.relu_fn(&ctx->a, nullptr, &ctx->c);
    } else {
      st = info.fn(&ctx->a, &ctx->c);
    }
    if (st != kZdnnOk) {
      const std::string msg = std::string(info.fn_name) + " failed: " + GetZdnnStatusMessage(st);
      return api_.CreateStatus(ORT_FAIL, msg.c_str());
    }

    st = zdnn_.transform_origtensor(&ctx->c, output_data);
    if (st != kZdnnOk) {
      const std::string msg = std::string("zdnn_transform_origtensor(output) failed: ") + GetZdnnStatusMessage(st);
      return api_.CreateStatus(ORT_FAIL, msg.c_str());
    }

    return nullptr;
  }

  OrtStatus* InitTensor(uint32_t layout,
                        gsl::span<const uint32_t> logical_shape,
                        OwnedZTensor& t) noexcept {
    if (logical_shape.empty() || logical_shape.size() > 4) {
      return api_.CreateStatus(ORT_INVALID_ARGUMENT, "zDNN tensor rank must be between 1 and 4");
    }

    switch (logical_shape.size()) {
      case 1:
        zdnn_.init_pre_transformed_desc(layout, kZdnnTypeFp32, &t.pre_desc,
                                        logical_shape[0]);
        break;
      case 2:
        zdnn_.init_pre_transformed_desc(layout, kZdnnTypeFp32, &t.pre_desc,
                                        logical_shape[0], logical_shape[1]);
        break;
      case 3:
        zdnn_.init_pre_transformed_desc(layout, kZdnnTypeFp32, &t.pre_desc,
                                        logical_shape[0], logical_shape[1], logical_shape[2]);
        break;
      case 4:
        zdnn_.init_pre_transformed_desc(layout, kZdnnTypeFp32, &t.pre_desc,
                                        logical_shape[0], logical_shape[1], logical_shape[2],
                                        logical_shape[3]);
        break;
      default:
        return api_.CreateStatus(ORT_INVALID_ARGUMENT, "Unsupported zDNN tensor rank");
    }

    int st = zdnn_.generate_transformed_desc(&t.pre_desc, &t.tfrmd_desc);
    if (st != kZdnnOk) {
      const std::string msg = std::string("zdnn_generate_transformed_desc failed: ") +
                              GetZdnnStatusMessage(st);
      return api_.CreateStatus(ORT_FAIL, msg.c_str());
    }

    zdnn_.init_ztensor(&t.pre_desc, &t.tfrmd_desc, &t.ztensor);
    st = zdnn_.allochelper_ztensor(&t.ztensor);
    if (st != kZdnnOk) {
      const std::string msg = std::string("zdnn_allochelper_ztensor failed: ") +
                              GetZdnnStatusMessage(st);
      return api_.CreateStatus(ORT_FAIL, msg.c_str());
    }

    t.allocated = true;
    return nullptr;
  }

  OrtStatus* InitTensor2D(uint32_t rows, uint32_t cols, OwnedZTensor& t) noexcept {
    const uint32_t dims[2] = {rows, cols};
    return InitTensor(kZdnnLayout2D, gsl::span<const uint32_t>(dims, 2), t);
  }

  OrtStatus* InitTensor1D(uint32_t n, OwnedZTensor& t) noexcept {
    return InitTensor(kZdnnLayout1D, gsl::span<const uint32_t>(&n, 1), t);
  }

  OrtStatus* RunMatMulPlan(const MatMulPlan& plan,
                           const float* a_data,
                           const float* b_data,
                           const float* c_bias,
                           float* y_data) noexcept {
    OwnedZTensor z_a(&zdnn_);
    OwnedZTensor z_b(&zdnn_);
    OwnedZTensor z_c(&zdnn_);
    OwnedZTensor z_y(&zdnn_);

    RETURN_IF_ERROR(InitTensor(plan.layout_a, plan.logical_shape_a, z_a));
    RETURN_IF_ERROR(InitTensor(plan.layout_b, plan.logical_shape_b, z_b));
    RETURN_IF_ERROR(InitTensor(plan.layout_c, plan.logical_shape_c, z_c));
    RETURN_IF_ERROR(InitTensor(plan.layout_y, plan.logical_shape_y, z_y));

    z_a.ztensor.is_transformed = false;
    z_b.ztensor.is_transformed = false;
    z_c.ztensor.is_transformed = false;
    z_y.ztensor.is_transformed = false;

    int st = zdnn_.transform_ztensor(&z_a.ztensor, a_data);
    if (st != kZdnnOk) {
      const std::string msg = std::string("zdnn_transform_ztensor(MatMul A) failed: ") +
                              GetZdnnStatusMessage(st);
      return api_.CreateStatus(ORT_FAIL, msg.c_str());
    }

    st = zdnn_.transform_ztensor(&z_b.ztensor, b_data);
    if (st != kZdnnOk) {
      const std::string msg = std::string("zdnn_transform_ztensor(MatMul B) failed: ") +
                              GetZdnnStatusMessage(st);
      return api_.CreateStatus(ORT_FAIL, msg.c_str());
    }

    st = zdnn_.transform_ztensor(&z_c.ztensor, c_bias);
    if (st != kZdnnOk) {
      const std::string msg = std::string("zdnn_transform_ztensor(MatMul C) failed: ") +
                              GetZdnnStatusMessage(st);
      return api_.CreateStatus(ORT_FAIL, msg.c_str());
    }

    if (plan.kind == MatMulPlan::Kind::kUnstacked || plan.kind == MatMulPlan::Kind::kStacked) {
      st = zdnn_.matmul_op(&z_a.ztensor, &z_b.ztensor, &z_c.ztensor, kMatmulOpAddition, &z_y.ztensor);
      if (st != kZdnnOk) {
        const std::string msg = std::string("zdnn_matmul_op failed: ") + GetZdnnStatusMessage(st);
        return api_.CreateStatus(ORT_FAIL, msg.c_str());
      }
    } else {
      st = zdnn_.matmul_bcast_op(&z_a.ztensor, &z_b.ztensor, &z_c.ztensor,
                                 kMatmulBcastOpAddition, &z_y.ztensor);
      if (st != kZdnnOk) {
        const std::string msg = std::string("zdnn_matmul_bcast_op failed: ") + GetZdnnStatusMessage(st);
        return api_.CreateStatus(ORT_FAIL, msg.c_str());
      }
    }

    st = zdnn_.transform_origtensor(&z_y.ztensor, y_data);
    if (st != kZdnnOk) {
      const std::string msg = std::string("zdnn_transform_origtensor(MatMul Y) failed: ") +
                              GetZdnnStatusMessage(st);
      return api_.CreateStatus(ORT_FAIL, msg.c_str());
    }

    return nullptr;
  }

  OrtStatus* RunMatMul(const float* a_data,
                       const float* b_data,
                       const float* c_bias,
                       float* y_data,
                       uint32_t m,
                       uint32_t k,
                       uint32_t n,
                       bool trans_a,
                       bool trans_b) noexcept {
    const uint32_t a_rows = trans_a ? k : m;
    const uint32_t a_cols = trans_a ? m : k;
    const uint32_t b_rows = trans_b ? n : k;
    const uint32_t b_cols = trans_b ? k : n;

    if (!ValidateDimsAgainstLimit({a_rows, a_cols, b_rows, b_cols, n, m, k})) {
      return api_.CreateStatus(ORT_INVALID_ARGUMENT,
                               "zDNN MatMul dimensions exceed host NNPA max_dim_idx_size");
    }

    OwnedZTensor z_a(&zdnn_);
    OwnedZTensor z_b(&zdnn_);
    OwnedZTensor z_c(&zdnn_);
    OwnedZTensor z_y(&zdnn_);

    RETURN_IF_ERROR(InitTensor2D(a_rows, a_cols, z_a));
    RETURN_IF_ERROR(InitTensor2D(b_rows, b_cols, z_b));
    RETURN_IF_ERROR(InitTensor1D(n, z_c));
    RETURN_IF_ERROR(InitTensor2D(m, n, z_y));

    z_a.ztensor.is_transformed = false;
    z_b.ztensor.is_transformed = false;
    z_c.ztensor.is_transformed = false;
    z_y.ztensor.is_transformed = false;

    int st = zdnn_.transform_ztensor(&z_a.ztensor, a_data);
    if (st != kZdnnOk) {
      const std::string msg = std::string("zdnn_transform_ztensor(A) failed: ") + GetZdnnStatusMessage(st);
      return api_.CreateStatus(ORT_FAIL, msg.c_str());
    }

    st = zdnn_.transform_ztensor(&z_b.ztensor, b_data);
    if (st != kZdnnOk) {
      const std::string msg = std::string("zdnn_transform_ztensor(B) failed: ") + GetZdnnStatusMessage(st);
      return api_.CreateStatus(ORT_FAIL, msg.c_str());
    }

    st = zdnn_.transform_ztensor(&z_c.ztensor, c_bias);
    if (st != kZdnnOk) {
      const std::string msg = std::string("zdnn_transform_ztensor(C) failed: ") + GetZdnnStatusMessage(st);
      return api_.CreateStatus(ORT_FAIL, msg.c_str());
    }

    if (trans_a || trans_b) {
      st = zdnn_.matmul_transpose_op(&z_a.ztensor, &z_b.ztensor, &z_c.ztensor,
                                     trans_a, trans_b, kMatmulOpAddition, &z_y.ztensor);
      if (st != kZdnnOk) {
        const std::string msg = std::string("zdnn_matmul_transpose_op failed: ") + GetZdnnStatusMessage(st);
        return api_.CreateStatus(ORT_FAIL, msg.c_str());
      }
    } else {
      st = zdnn_.matmul_op(&z_a.ztensor, &z_b.ztensor, &z_c.ztensor,
                           kMatmulOpAddition, &z_y.ztensor);
      if (st != kZdnnOk) {
        const std::string msg = std::string("zdnn_matmul_op failed: ") + GetZdnnStatusMessage(st);
        return api_.CreateStatus(ORT_FAIL, msg.c_str());
      }
    }

    st = zdnn_.transform_origtensor(&z_y.ztensor, y_data);
    if (st != kZdnnOk) {
      const std::string msg = std::string("zdnn_transform_origtensor(Y) failed: ") + GetZdnnStatusMessage(st);
      return api_.CreateStatus(ORT_FAIL, msg.c_str());
    }

    return nullptr;
  }

  OrtStatus* BuildBiasVector(const TelumGemmRequest& request,
                             size_t n,
                             std::vector<float>& bias) noexcept {
    if (!request.has_c) {
      std::fill(bias.begin(), bias.end(), 0.0f);
      return nullptr;
    }

    const auto c_count = request.input_c.size();
    const auto& c_shape = request.c_shape;

    auto shape_count = [&]() -> std::optional<size_t> {
      size_t out = 1;
      for (int64_t d : c_shape) {
        if (d < 0) {
          return std::nullopt;
        }
        if (d == 0) {
          return static_cast<size_t>(0);
        }
        if (out > std::numeric_limits<size_t>::max() / static_cast<size_t>(d)) {
          return std::nullopt;
        }
        out *= static_cast<size_t>(d);
      }
      return out;
    };

    const auto shape_count_opt = shape_count();
    if (!shape_count_opt.has_value() || *shape_count_opt != c_count) {
      return api_.CreateStatus(ORT_INVALID_ARGUMENT,
                               "zDNN Gemm C tensor size does not match provided C shape");
    }

    if (c_shape.empty() || (c_shape.size() == 1 && c_shape[0] == 1) ||
        (c_shape.size() == 2 && c_shape[0] == 1 && c_shape[1] == 1)) {
      const float scalar = request.input_c[0];
      std::fill(bias.begin(), bias.end(), scalar);
      return nullptr;
    }

    if (c_shape.size() == 1 && static_cast<size_t>(c_shape[0]) == n) {
      std::copy(request.input_c.begin(), request.input_c.end(), bias.begin());
      return nullptr;
    }

    if (c_shape.size() == 2 && c_shape[0] == 1 && static_cast<size_t>(c_shape[1]) == n) {
      std::copy_n(request.input_c.begin(), n, bias.begin());
      return nullptr;
    }

    return api_.CreateStatus(ORT_INVALID_ARGUMENT,
                             "zDNN Gemm backend supports C as scalar, [N], or [1,N] only");
  }

  static OrtStatus* MulTrusted(void* ctx,
                               const float* input0_data,
                               const float* input1_data,
                               float* output_data,
                               size_t num_elems) noexcept {
    auto* self = static_cast<ZdnnTelumBackend*>(ctx);
    return self->BinaryImpl(input0_data, input1_data, output_data, num_elems,
                            self->zdnn_.mul, "zdnn_mul");
  }

  const OrtApi& api_;
  const ZdnnDynApi& zdnn_;

  bool runtime_ready_{false};
  bool supports_add_{false};
  bool supports_sub_{false};
  bool supports_mul_{false};
  bool supports_div_{false};
  bool supports_min_{false};
  bool supports_max_{false};
  bool supports_relu_{false};
  bool supports_gelu_{false};
  bool supports_tanh_{false};
  bool supports_sigmoid_{false};
  bool supports_exp_{false};
  bool supports_log_{false};
  bool supports_sqrt_{false};
  bool supports_softmax_{false};
  bool supports_layernorm_{false};
  bool supports_matmul_{false};
  uint32_t max_dim_idx_size_{0};
  std::string reason_;
};

#endif  // ORT_TELUM_PLUGIN_EP_ZDNN && __linux__ && s390x

std::atomic<TelumBackendFactoryFn>& ExternalBackendFactory() {
  static std::atomic<TelumBackendFactoryFn> factory_fn{nullptr};
  return factory_fn;
}

}  // namespace

void RegisterTelumBackendFactory(TelumBackendFactoryFn factory_fn) {
  ExternalBackendFactory().store(factory_fn, std::memory_order_release);
}

std::unique_ptr<TelumBackend> CreateTelumBackend(const OrtApi& api, const TelumBackendConfig& config) {
  telum_profile::ScopedEvent profile{telum_profile::Event::kCreateTelumBackend};
  TelumBackendFactoryFn factory_fn = ExternalBackendFactory().load(std::memory_order_acquire);

  if (factory_fn != nullptr) {
    if (auto backend = factory_fn(api, config); backend != nullptr) {
      return backend;
    }
  }

  if (config.backend_kind.empty() || config.backend_kind == kTelumBackendKindZdnn) {
#if defined(ORT_TELUM_PLUGIN_EP_ZDNN) && defined(__linux__) && (defined(__s390x__) || defined(__s390__))
    return std::make_unique<ZdnnTelumBackend>(api);
#else
    return std::make_unique<UnavailableTelumBackend>(
        api, "zDNN backend was requested but this build does not include zDNN support");
#endif
  }

  return std::make_unique<UnavailableTelumBackend>(
      api, "unknown backend kind '" + config.backend_kind + "'");
}
