# ONNX Runtime Telum Plugin Execution Provider (Standalone)

This repository contains a standalone ONNX Runtime **plugin Execution Provider (EP)** shared library for IBM Telum
(s390x). It is intended to be built and packaged outside of the ONNX Runtime repository, and loaded at runtime via the
ONNX Runtime plugin EP API.

If distributed as a Python package, the recommended package name format is:
- `onnxruntime-ep-<ep-identifier>` (this repo uses `onnxruntime-ep-telum`).

## Build

Prerequisites:
- CMake (>= 3.20)
- A C++17 compiler
- ONNX Runtime public headers (must include `onnxruntime_cxx_api.h` and `onnxruntime_ep_c_api.h`)

Configure and build:

```bash
cmake -S . -B build \
  -DONNXRUNTIME_INCLUDE_DIR=/path/to/onnxruntime/include/onnxruntime/core/session
cmake --build build -j
```

Notes:
- This library does **not** ship the ONNX Runtime shared library. ORT must be provided by the consuming application.
- On Linux s390x, the build enables an optional zDNN-backed path (dynamic `dlopen("libzdnn.so")`) by default.
  You can disable it with `-DTELUM_EP_ENABLE_ZDNN=OFF`.

## Runtime Usage (C++)

Register the plugin EP library:

```cpp
Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "app"};
env.RegisterExecutionProviderLibrary(library_path, ORT_TSTR("TelumPluginExecutionProvider"));
```

Then append the EP:

```cpp
Ort::SessionOptions so;
Ort::ThrowOnError(Ort::GetApi().SessionOptionsAppendExecutionProvider_V2(
    so, "TelumPluginExecutionProvider", /*provider_options=*/nullptr));
```

Exact APIs and calling patterns are described in the ONNX Runtime plugin EP documentation.

## Repo Layout

- `src/plugin_ep_utils.h`: helper utilities (adapted from ONNX Runtime plugin EP samples)
- `src/telum_plugin_ep/*`: Telum plugin EP implementation and entrypoints
- `python/onnxruntime_ep_telum/*`: optional Python helpers:
  - `get_library_path()`, `get_ep_names()`, `get_ep_name()`

## CI

This repo includes GitHub Actions PR CI in `.github/workflows/pr-ci.yml`:

- `packaging-sanity`: builds the Python source distribution.
- `linux-smoke-build`: configures/builds the plugin on `ubuntu-latest` with `TELUM_EP_ENABLE_ZDNN=OFF`.
- `s390x-full-build`: full build on `[self-hosted, linux, s390x]`, gated by default.

The s390x job only runs when both conditions are met:
- Repository variable `ENABLE_S390X_FULL_BUILD` is set to `true`.
- Trigger is explicit:
  - PR labeled `ci/s390x-full`, or
  - Manual `workflow_dispatch` run with `run_s390x=true`.
