// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "telum_backend.h"

#include <atomic>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

#if defined(ORT_TELUM_PLUGIN_EP_ZDNN) && defined(__linux__)
#include <dlfcn.h>
#endif

#include "telum_profile.h"

namespace {

#if defined(_MSC_VER)
#define TELUM_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define TELUM_RESTRICT __restrict__
#else
#define TELUM_RESTRICT
#endif

constexpr const char* kTelumBackendKindStub = "stub";
constexpr const char* kTelumBackendKindZdnn = "zdnn";

class StubTelumBackend final : public TelumBackend {
 public:
  explicit StubTelumBackend(const OrtApi& api, bool supports_mul)
      : api_(api), supports_mul_(supports_mul) {}

  bool SupportsMul() const noexcept override {
    return supports_mul_;
  }

  TelumMulTrustedFn GetMulTrustedFn() noexcept override {
    if (!supports_mul_) {
      return {};
    }
    return TelumMulTrustedFn{&StubMulTrusted, this};
  }

  OrtStatus* Mul(gsl::span<const float> input0,
                 gsl::span<const float> input1,
                 gsl::span<float> output) noexcept override {
    if (!supports_mul_) {
      return api_.CreateStatus(ORT_FAIL, "StubTelumBackend Mul is disabled by configuration");
    }

    const size_t num_elems = input0.size();
    if (num_elems != input1.size() || num_elems != output.size()) {
      return api_.CreateStatus(ORT_INVALID_ARGUMENT,
                               "StubTelumBackend expected same number of elements for Mul");
    }

    const float* input0_data = input0.data();
    const float* input1_data = input1.data();
    float* output_data = output.data();
    for (size_t i = 0; i < num_elems; ++i) {
      output_data[i] = input0_data[i] * input1_data[i];
    }

    return nullptr;
  }

 private:
  static OrtStatus* StubMulTrusted(void* ctx,
                                   const float* input0_data,
                                   const float* input1_data,
                                   float* output_data,
                                   size_t num_elems) noexcept {
    (void)ctx;
    // These pointers are expected to be non-aliasing for ORT tensors. Mark them restrict to help auto-vectorization.
    const float* TELUM_RESTRICT a = input0_data;
    const float* TELUM_RESTRICT b = input1_data;
    float* TELUM_RESTRICT c = output_data;

#if defined(__clang__)
#pragma clang loop vectorize(enable) interleave(enable)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
    for (size_t i = 0; i < num_elems; ++i) {
      c[i] = a[i] * b[i];
    }
    return nullptr;
  }

  const OrtApi& api_;
  bool supports_mul_;
};

class UnavailableTelumBackend final : public TelumBackend {
 public:
  explicit UnavailableTelumBackend(const OrtApi& api, std::string reason)
      : api_(api), reason_(std::move(reason)) {}

  bool SupportsMul() const noexcept override { return false; }

  OrtStatus* Mul(gsl::span<const float> /*input0*/,
                 gsl::span<const float> /*input1*/,
                 gsl::span<float> /*output*/) noexcept override {
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
constexpr uint32_t kZdnnLayout2D = 1;  // ZDNN_2D
constexpr uint32_t kZdnnTypeFp32 = 255;  // FP32
constexpr int kNnpaMul = 18;  // NNPA_MUL

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
using ZdnnInitPreTransformedDescFn = void (*)(uint32_t, uint32_t, ZdnnTensorDesc*, ...);
using ZdnnGenerateTransformedDescFn = int (*)(const ZdnnTensorDesc*, ZdnnTensorDesc*);
using ZdnnInitZTensorFn = void (*)(ZdnnTensorDesc*, ZdnnTensorDesc*, ZdnnZTensor*);
using ZdnnAllocHelperZTensorFn = int (*)(ZdnnZTensor*);
using ZdnnFreeZTensorBufferFn = int (*)(const ZdnnZTensor*);
using ZdnnTransformZTensorFn = int (*)(ZdnnZTensor*, ...);
using ZdnnTransformOrigTensorFn = int (*)(const ZdnnZTensor*, void*);
using ZdnnMulFn = int (*)(const ZdnnZTensor*, const ZdnnZTensor*, ZdnnZTensor*);

struct ZdnnDynApi final {
  void* handle{};

  ZdnnInitFn init{};
  ZdnnIsNnpaInstalledFn is_nnpa_installed{};
  ZdnnIsNnpaFunctionInstalledFn is_nnpa_function_installed{};
  ZdnnGetStatusMessageFn get_status_message{};
  ZdnnInitPreTransformedDescFn init_pre_transformed_desc{};
  ZdnnGenerateTransformedDescFn generate_transformed_desc{};
  ZdnnInitZTensorFn init_ztensor{};
  ZdnnAllocHelperZTensorFn allochelper_ztensor{};
  ZdnnFreeZTensorBufferFn free_ztensor_buffer{};
  ZdnnTransformZTensorFn transform_ztensor{};
  ZdnnTransformOrigTensorFn transform_origtensor{};
  ZdnnMulFn mul{};

  bool IsLoaded() const noexcept { return handle != nullptr; }
};

template <typename Fn>
void LoadZdnnSymbol(void* handle, const char* symbol_name, Fn& fn) {
  fn = reinterpret_cast<Fn>(::dlsym(handle, symbol_name));
}

const ZdnnDynApi& GetZdnnApi() noexcept {
  // Intentionally keep libzdnn loaded for process lifetime to avoid teardown ordering issues.
  static const ZdnnDynApi api = []() {
    ZdnnDynApi a{};

    // Resolve via the platform loader search path, so caller can set LD_LIBRARY_PATH if needed.
    a.handle = ::dlopen("libzdnn.so", RTLD_NOW | RTLD_LOCAL);
    if (a.handle == nullptr) {
      return a;
    }

    LoadZdnnSymbol(a.handle, "zdnn_init", a.init);
    LoadZdnnSymbol(a.handle, "zdnn_is_nnpa_installed", a.is_nnpa_installed);
    LoadZdnnSymbol(a.handle, "zdnn_is_nnpa_function_installed", a.is_nnpa_function_installed);
    LoadZdnnSymbol(a.handle, "zdnn_get_status_message", a.get_status_message);
    LoadZdnnSymbol(a.handle, "zdnn_init_pre_transformed_desc", a.init_pre_transformed_desc);
    LoadZdnnSymbol(a.handle, "zdnn_generate_transformed_desc", a.generate_transformed_desc);
    LoadZdnnSymbol(a.handle, "zdnn_init_ztensor", a.init_ztensor);
    LoadZdnnSymbol(a.handle, "zdnn_allochelper_ztensor", a.allochelper_ztensor);
    LoadZdnnSymbol(a.handle, "zdnn_free_ztensor_buffer", a.free_ztensor_buffer);
    LoadZdnnSymbol(a.handle, "zdnn_transform_ztensor", a.transform_ztensor);
    LoadZdnnSymbol(a.handle, "zdnn_transform_origtensor", a.transform_origtensor);
    LoadZdnnSymbol(a.handle, "zdnn_mul", a.mul);

    const bool ok = a.init && a.is_nnpa_installed && a.is_nnpa_function_installed &&
                    a.get_status_message && a.init_pre_transformed_desc &&
                    a.generate_transformed_desc && a.init_ztensor && a.allochelper_ztensor &&
                    a.free_ztensor_buffer && a.transform_ztensor && a.transform_origtensor &&
                    a.mul;
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
      supports_mul_ = false;
      reason_ = "libzdnn.so not available (dlopen/dlsym failed)";
      return;
    }

    const bool nnpa_installed = zdnn_.is_nnpa_installed();
    const bool mul_supported = nnpa_installed && zdnn_.is_nnpa_function_installed(1, kNnpaMul);
    supports_mul_ = mul_supported;

    if (!mul_supported) {
      reason_ = nnpa_installed ? "NNPA MUL function not installed" : "NNPA not installed/enabled";
    }
  }

  bool SupportsMul() const noexcept override {
    return supports_mul_;
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
    if (!supports_mul_) {
      const std::string msg = "zDNN backend cannot execute Mul: " + reason_;
      return api_.CreateStatus(ORT_FAIL, msg.c_str());
    }

    const size_t num_elems = input0.size();
    if (num_elems != input1.size() || num_elems != output.size()) {
      return api_.CreateStatus(ORT_INVALID_ARGUMENT,
                               "zDNN backend expected same number of elements for Mul");
    }

    return MulImpl(input0.data(), input1.data(), output.data(), num_elems);
  }

 private:
  static bool Choose2DDims(size_t num_elems, uint32_t& dim2, uint32_t& dim1) noexcept {
    // Conservative split that keeps the innermost dimension <= 32768 (observed limit for this environment).
    constexpr size_t kMaxDim1 = 32768;
    if (num_elems == 0) {
      return false;
    }

    size_t candidate_dim1 = std::min(num_elems, kMaxDim1);
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

    dim2 = static_cast<uint32_t>(candidate_dim2);
    dim1 = static_cast<uint32_t>(candidate_dim1);
    return true;
  }

  struct MulCtx final {
    const ZdnnDynApi* zdnn{};
    uint32_t dim2{};
    uint32_t dim1{};

    ZdnnTensorDesc pre_desc{};
    ZdnnTensorDesc tfrmd_desc{};
    ZdnnZTensor a{};
    ZdnnZTensor b{};
    ZdnnZTensor c{};

    bool initialized{};

    explicit MulCtx(const ZdnnDynApi* zdnn_api)
        : zdnn(zdnn_api) {}

    MulCtx(const MulCtx&) = delete;
    MulCtx& operator=(const MulCtx&) = delete;

    ~MulCtx() {
      if (!initialized || zdnn == nullptr || zdnn->free_ztensor_buffer == nullptr) {
        return;
      }
      (void)zdnn->free_ztensor_buffer(&a);
      (void)zdnn->free_ztensor_buffer(&b);
      (void)zdnn->free_ztensor_buffer(&c);
    }
  };

  static thread_local std::unordered_map<size_t, std::unique_ptr<MulCtx>>& MulCtxCache() {
    static thread_local std::unordered_map<size_t, std::unique_ptr<MulCtx>> cache;
    return cache;
  }

  MulCtx* GetOrInitMulCtx(size_t num_elems, std::string& error) noexcept {
    auto& cache = MulCtxCache();
    auto it = cache.find(num_elems);
    if (it != cache.end()) {
      return it->second.get();
    }

    uint32_t dim2 = 0;
    uint32_t dim1 = 0;
    if (!Choose2DDims(num_elems, dim2, dim1)) {
      error = "unable to represent tensor shape for zDNN 2D descriptor";
      return nullptr;
    }

    auto ctx = std::make_unique<MulCtx>(&zdnn_);
    ctx->dim2 = dim2;
    ctx->dim1 = dim1;

    // Elementwise Mul is shape-agnostic for this scaffold, so we use a contiguous 2D descriptor.
    // 2D avoids invalid-shape failures seen with large flattened 1D descriptors on current NNPA/zDNN.
    zdnn_.init_pre_transformed_desc(kZdnnLayout2D, kZdnnTypeFp32, &ctx->pre_desc, ctx->dim2, ctx->dim1);

    int st = zdnn_.generate_transformed_desc(&ctx->pre_desc, &ctx->tfrmd_desc);
    if (st != kZdnnOk) {
      error = std::string("zdnn_generate_transformed_desc failed: ") + zdnn_.get_status_message(st);
      return nullptr;
    }

    zdnn_.init_ztensor(&ctx->pre_desc, &ctx->tfrmd_desc, &ctx->a);
    zdnn_.init_ztensor(&ctx->pre_desc, &ctx->tfrmd_desc, &ctx->b);
    zdnn_.init_ztensor(&ctx->pre_desc, &ctx->tfrmd_desc, &ctx->c);

    st = zdnn_.allochelper_ztensor(&ctx->a);
    if (st != kZdnnOk) {
      error = std::string("zdnn_allochelper_ztensor(input0) failed: ") + zdnn_.get_status_message(st);
      return nullptr;
    }

    st = zdnn_.allochelper_ztensor(&ctx->b);
    if (st != kZdnnOk) {
      error = std::string("zdnn_allochelper_ztensor(input1) failed: ") + zdnn_.get_status_message(st);
      return nullptr;
    }

    st = zdnn_.allochelper_ztensor(&ctx->c);
    if (st != kZdnnOk) {
      error = std::string("zdnn_allochelper_ztensor(output) failed: ") + zdnn_.get_status_message(st);
      return nullptr;
    }

    ctx->initialized = true;
    MulCtx* out = ctx.get();
    cache.emplace(num_elems, std::move(ctx));
    return out;
  }

  OrtStatus* MulImpl(const float* input0_data,
                     const float* input1_data,
                     float* output_data,
                     size_t num_elems) noexcept {
    std::string error;
    MulCtx* ctx = GetOrInitMulCtx(num_elems, error);
    if (ctx == nullptr) {
      const std::string msg = "zDNN Mul setup failed: " + error;
      return api_.CreateStatus(ORT_FAIL, msg.c_str());
    }

    ctx->a.is_transformed = false;
    ctx->b.is_transformed = false;
    ctx->c.is_transformed = false;

    int st = zdnn_.transform_ztensor(&ctx->a, input0_data);
    if (st != kZdnnOk) {
      const std::string msg = std::string("zdnn_transform_ztensor(input0) failed: ") +
                              zdnn_.get_status_message(st);
      return api_.CreateStatus(ORT_FAIL, msg.c_str());
    }

    st = zdnn_.transform_ztensor(&ctx->b, input1_data);
    if (st != kZdnnOk) {
      const std::string msg = std::string("zdnn_transform_ztensor(input1) failed: ") +
                              zdnn_.get_status_message(st);
      return api_.CreateStatus(ORT_FAIL, msg.c_str());
    }

    st = zdnn_.mul(&ctx->a, &ctx->b, &ctx->c);
    if (st != kZdnnOk) {
      const std::string msg = std::string("zdnn_mul failed: ") + zdnn_.get_status_message(st);
      return api_.CreateStatus(ORT_FAIL, msg.c_str());
    }

    st = zdnn_.transform_origtensor(&ctx->c, output_data);
    if (st != kZdnnOk) {
      const std::string msg = std::string("zdnn_transform_origtensor(output) failed: ") +
                              zdnn_.get_status_message(st);
      return api_.CreateStatus(ORT_FAIL, msg.c_str());
    }

    return nullptr;
  }

  static OrtStatus* MulTrusted(void* ctx,
                               const float* input0_data,
                               const float* input1_data,
                               float* output_data,
                               size_t num_elems) noexcept {
    auto* self = static_cast<ZdnnTelumBackend*>(ctx);
    return self->MulImpl(input0_data, input1_data, output_data, num_elems);
  }

  const OrtApi& api_;
  const ZdnnDynApi& zdnn_;
  bool supports_mul_{false};
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

  if (config.backend_kind == kTelumBackendKindStub || config.backend_kind.empty()) {
    return std::make_unique<StubTelumBackend>(api, config.stub_support_mul);
  }

  if (config.backend_kind == kTelumBackendKindZdnn) {
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
