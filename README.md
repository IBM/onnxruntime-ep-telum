# ONNX Runtime Telum Plugin Execution Provider

Standalone plugin Execution Provider (EP) for ONNX Runtime targeting IBM Telum workflows.

This repository builds a shared library that is loaded at runtime through ONNX Runtime's plugin EP API. It is
maintained outside the main ONNX Runtime repository so the EP can evolve on its own cadence.

No ONNX Runtime rebuild is required: keep your existing ORT runtime and load this plugin library at runtime.

## Quick Start

Build:

```bash
make build
```

Register and append the EP in your C++ host:

```cpp
Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "app"};
Ort::ThrowOnError(env.RegisterExecutionProviderLibrary(
    library_path, ORT_TSTR("TelumPluginExecutionProvider")));

Ort::SessionOptions so;
Ort::ThrowOnError(Ort::GetApi().SessionOptionsAppendExecutionProvider_V2(
    so, "TelumPluginExecutionProvider", /*provider_options=*/nullptr));
```

`library_path` should point to the built plugin shared library (`telum_plugin_ep`).

## Documentation

- Install guide: `docs/INSTALL.md`
- General usage guide: `docs/USER_GUIDE.md`
- Contributing guide: `CONTRIBUTING.md`

## Current Capability Snapshot

This is an active scaffold with concrete runtime paths, not a fully broad operator backend yet.

- EP registration name: `TelumPluginExecutionProvider`
- Supported graph patterns:
  - `Mul` (`float32`, static equal-shape inputs)
  - `EPContext` nodes for compiled-model loading paths
  - sample custom-op flow (`Custom_Mul` in `test` domain)
- Backends:
  - `stub` (default)
  - optional `zdnn` runtime path on Linux s390x

## Build Notes

- Requires ONNX Runtime public headers (`onnxruntime_cxx_api.h`, `onnxruntime_ep_c_api.h`)
- Requires CMake `>= 3.20` and C++17 compiler
- By default, `make build` fetches ONNX Runtime headers into `.ort`
- zDNN compile switch: `TELUM_EP_ENABLE_ZDNN=ON|OFF`

## Runtime Configuration

Configuration is provided through `OrtSessionOptions` config entries.

Preferred key style:

- `ep.<registration_name>.<key>`

Examples for registration name `TelumPluginExecutionProvider`:

- `ep.TelumPluginExecutionProvider.backend` = `stub|zdnn`
- `ep.TelumPluginExecutionProvider.stub_support_mul` = boolean token
- `ep.TelumPluginExecutionProvider.drop_constant_initializers` = boolean token
- `ep.context_enable` = `0|1`

Legacy aliases (`telum.backend`, `telum.stub_support_mul`, `telum.drop_constant_initializers`) are still accepted.

## Python Packaging Helpers

The Python package `onnxruntime_ep_telum` provides convenience helpers:

- `get_ep_name()`
- `get_ep_names()`
- `get_library_path()`

These helpers expose EP metadata/library path; host-side ONNX Runtime APIs still perform the actual plugin registration.

## CI and Release

- PR CI workflow: `.github/workflows/pr-ci.yml`
- PyPI release workflow: `.github/workflows/release-pypi.yml`
- NuGet release workflow: `.github/workflows/release-nuget.yml`

## Repository Layout

- `src/telum_plugin_ep/`: plugin EP implementation
- `src/plugin_ep_utils.h`: shared helper utilities
- `python/onnxruntime_ep_telum/`: Python helper package and library staging path
- `packaging/nuget/`: NuGet package spec

## License

Apache-2.0. See `LICENSE`.
