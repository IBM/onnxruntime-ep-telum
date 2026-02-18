# User Guide

This guide explains how to use the Telum plugin EP at runtime.

## Overview

The repository provides a standalone ONNX Runtime plugin EP shared library.

- Registration name: `TelumPluginExecutionProvider`
- Primary integration API: plugin EP runtime registration
- Current scaffold focus: `Mul` execution path and EPContext support

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

## Python Package Helper APIs

The Python module provides helper utilities:

- `get_ep_name()`
- `get_ep_names()`
- `get_library_path()`

These helpers locate metadata and plugin library paths. Runtime EP registration still depends on host ONNX Runtime APIs.

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
