# User Guide

This guide explains how to use the Telum plugin EP at runtime.

## Overview

The repository provides a standalone ONNX Runtime plugin EP shared library.

- Registration name: `TelumPluginExecutionProvider`
- Primary integration API: plugin EP runtime registration
- Current scaffold focus: `Mul` execution path and EPContext support

## No ONNX Runtime Rebuild Required

You do not rebuild ONNX Runtime to use this plugin.

- Keep your existing ONNX Runtime binary/package.
- Build or download this plugin shared library separately.
- Load/register the plugin at runtime from your host application.

## Runtime Integration (C++)

Register the plugin library, then append the EP:

```cpp
Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "app"};
Ort::ThrowOnError(env.RegisterExecutionProviderLibrary(
    library_path, ORT_TSTR("TelumPluginExecutionProvider")));

Ort::SessionOptions so;
Ort::ThrowOnError(Ort::GetApi().SessionOptionsAppendExecutionProvider_V2(
    so, "TelumPluginExecutionProvider", /*provider_options=*/nullptr));
```

`library_path` must point to the built `telum_plugin_ep` shared library.

## Where This Code Goes

The registration/append code belongs in your host application, not in this plugin repository.

Typical placement in your app:

- Startup/bootstrap code creates `Ort::Env` once for process lifetime.
- Session-construction code creates `Ort::SessionOptions`.
- In that same session-construction path:
  - register the plugin library on the `Ort::Env`
  - append `TelumPluginExecutionProvider` to `Ort::SessionOptions`
  - create `Ort::Session`

Call order requirement:

1. Create `Ort::Env`
2. Register plugin library on `Ort::Env`
3. Configure and append EP on `Ort::SessionOptions`
4. Create `Ort::Session`

### Example Host-Side Placement

Example file in your app: `src/inference/session_factory.cc`

```cpp
#include <onnxruntime_cxx_api.h>

#include <memory>
#include <string>

struct OrtRuntimeContext {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "my-app"};
  bool telum_registered{false};
};

std::unique_ptr<Ort::Session> CreateSessionWithTelum(
    OrtRuntimeContext& ctx,
    const std::string& model_path,
    const std::string& telum_library_path) {
  if (!ctx.telum_registered) {
    Ort::ThrowOnError(ctx.env.RegisterExecutionProviderLibrary(
        telum_library_path.c_str(), ORT_TSTR("TelumPluginExecutionProvider")));
    ctx.telum_registered = true;
  }

  Ort::SessionOptions so;
  so.SetConfigEntry("ep.TelumPluginExecutionProvider.backend", "stub");
  Ort::ThrowOnError(Ort::GetApi().SessionOptionsAppendExecutionProvider_V2(
      so, "TelumPluginExecutionProvider", nullptr));

  return std::make_unique<Ort::Session>(ctx.env, model_path.c_str(), so);
}
```

If you build sessions in multiple places, centralize this in one helper/factory and reuse it.

## Runtime Configuration Keys

Set these in `OrtSessionOptions` config entries before creating the session.

Preferred keys use the registration name prefix:

- `ep.<registration_name>.<key>`

Example registration name:

- `TelumPluginExecutionProvider`

### Backend selection

- `ep.TelumPluginExecutionProvider.backend`
  - Values: `stub`, `zdnn`
  - Default: `stub`
- Legacy alias (still accepted): `telum.backend`

### Stub Mul switch

- `ep.TelumPluginExecutionProvider.stub_support_mul`
  - Values: boolean tokens (`0/1`, `false/true`, `no/yes`, `off/on`)
  - Default: `1`
- Legacy alias: `telum.stub_support_mul`

### Constant initializer handling

- `ep.TelumPluginExecutionProvider.drop_constant_initializers`
  - Values: boolean tokens
  - Default: `1`
- Legacy alias: `telum.drop_constant_initializers`

### EPContext generation

- `ep.context_enable`
  - Values: `0` or `1`
  - Default: `0`
  - Constraint: if `ep.context_enable=1`, then `drop_constant_initializers` must be enabled

## Supported Graph Patterns (Current Scaffold)

- `Mul` nodes:
  - `float32` inputs/outputs only
  - exactly 2 inputs + 1 output
  - static, equal input shapes required
- `EPContext` nodes in domain `com.microsoft` with matching `source` attribute
- `Custom_Mul` in domain `test` for sample custom-op flow

Unsupported nodes remain on other execution providers.

## zDNN Backend Notes

- zDNN path is compile-gated via `TELUM_EP_ENABLE_ZDNN`
- Runtime selection is config-based (`backend=zdnn`)
- On Linux s390x, the implementation dynamically loads `libzdnn.so`
- If zDNN cannot be loaded or NNPA MUL capability is unavailable, backend capability is rejected

## Distribution Helper APIs

### PyPI helper APIs

Defined in:

- `python/onnxruntime_ep_telum/__init__.py`

Helpers:

- `get_ep_name()`
- `get_ep_names()`
- `get_library_path()`

### NuGet helper APIs

Defined in:

- `packaging/nuget/src/OnnxRuntimeEpTelum/PluginEpHelpers.cs`

Helpers:

- `PluginEpHelpers.GetEpName()`
- `PluginEpHelpers.GetEpNames()`
- `PluginEpHelpers.GetLibraryPath()`

These helpers resolve EP name(s) and plugin-library path. Your host still performs the actual runtime registration.

## Profiling and Diagnostics

Set environment variable:

- `ORT_TELUM_PLUGIN_EP_PROFILE=1`

When enabled, the plugin emits lightweight timing summaries on unload.

## Common Failure Modes

- Plugin fails to load:
  - Check shared library path and dynamic library dependencies
- EP not selected for nodes:
  - Verify node shapes/types satisfy current scaffold constraints
  - Verify backend choice (`stub` or `zdnn`) and availability
- Invalid config value:
  - Use supported boolean tokens or allowed enum values
