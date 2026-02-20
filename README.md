# ONNX Runtime Telum Plugin Execution Provider

Standalone plugin Execution Provider (EP) for ONNX Runtime targeting IBM Telum workflows.

This repository builds a shared library that is loaded at runtime through ONNX Runtime's plugin EP API. It is
maintained outside the main ONNX Runtime repository so the EP can evolve on its own cadence.

No ONNX Runtime rebuild is required: keep your existing ORT runtime and load this plugin library at runtime.

## ONNX Runtime Version Compatibility

- Plugin EP architecture first appears in ONNX Runtime `1.23.0`.
- This repository currently uses plugin EP APIs introduced in ONNX Runtime API version `1.24`.
- Effective minimum supported released base runtime for this codebase: ONNX Runtime `1.24.1+`.

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

## Contributing

Contributions are welcome through pull requests.

- Open a branch from `main` and submit changes via PR.
- Keep PRs scoped and include validation evidence.
- Code owner review is required before merge.
- Follow the full contribution workflow in `CONTRIBUTING.md`.

## Current Capability Snapshot

- EP registration name: `TelumPluginExecutionProvider`
- Static-shape-first partitioning policy with explicit fallback diagnostics and optional strict mode
- Supported operators (plugin kernel path):
  - Math: `MatMul`, `Gemm`, `Add`, `Sub`, `Mul`, `Div`, `Min`, `Max`
  - Activations: `Relu`, `Gelu`, `Tanh`, `Sigmoid`, `Exp`, `Log`, `Sqrt`, `Softmax`
  - Normalization: `LayerNormalization`
  - Tensor: `Reshape`, `Transpose`, `Squeeze`, `Unsqueeze`, `ReduceMean`, `Cast`, `Where`, `Expand`, `Concat`, `Gather`, `Slice`
- EPContext compatibility:
  - EPContext node handling for compiled-model loading paths
  - v2 EPContext serialization format with compatibility path for legacy Mul-only format
- Custom-op flow example:
  - `Custom_Mul` in `test` domain
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
- `ep.TelumPluginExecutionProvider.strict_mode` = boolean token
- `ep.TelumPluginExecutionProvider.log_fallbacks` = boolean token
- `ep.TelumPluginExecutionProvider.log_partition_summary` = boolean token
- `ep.TelumPluginExecutionProvider.verbose_partition_trace` = boolean token
- `ep.TelumPluginExecutionProvider.enable_fusion` = boolean token
- `ep.TelumPluginExecutionProvider.drop_constant_initializers` = boolean token
- `ep.context_enable` = `0|1`

Legacy aliases (`telum.backend`, `telum.stub_support_mul`, `telum.drop_constant_initializers`,
`telum.strict_mode`, `telum.log_fallbacks`, `telum.log_partition_summary`,
`telum.verbose_partition_trace`, `telum.enable_fusion`) are still accepted.

## Validation Helpers

Validation helpers are included for parity runs and result capture:

- `tools/validation/run_functional_suite.sh`
- `tools/validation/run_perf_suite.sh`
- report templates under `reports/parity/`

## Python Packaging Helpers

The Python package `onnxruntime_ep_telum` provides convenience helpers:

- `get_ep_name()`
- `get_ep_names()`
- `get_library_path()`

These helpers expose EP metadata/library path; host-side ONNX Runtime APIs still perform the actual plugin registration.

## CI and Release

- PR CI workflow: `.github/workflows/pr-ci.yml`
  - Includes `packaging-sanity`, `linux-smoke-build`, and `s390x-qemu-build` (compile-only).
  - Keeps `s390x-full-build` gated to self-hosted runner flow.
- PyPI release workflow: `.github/workflows/release-pypi.yml`
- NuGet release workflow: `.github/workflows/release-nuget.yml`

## Repository Layout

- `src/telum_plugin_ep/`: plugin EP implementation
- `src/plugin_ep_utils.h`: shared helper utilities
- `python/onnxruntime_ep_telum/`: Python helper package and library staging path
- `packaging/nuget/`: NuGet package spec

## License

Apache-2.0. See `LICENSE`.
