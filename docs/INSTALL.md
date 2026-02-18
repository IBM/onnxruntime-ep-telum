# Install Guide

This guide covers building and staging the Telum plugin EP from source.

## What You Install

The primary deliverable is a shared library named `telum_plugin_ep`:

- Linux: `libtelum_plugin_ep.so`
- macOS: `libtelum_plugin_ep.dylib`
- Windows: `telum_plugin_ep.dll`

You register this library at runtime with ONNX Runtime's plugin EP API.

## Prerequisites

- CMake `>= 3.20`
- C++17 compiler toolchain
- Git
- ONNX Runtime public headers containing:
  - `onnxruntime_cxx_api.h`
  - `onnxruntime_ep_c_api.h`

Optional:

- `ninja` generator
- Linux s390x + `libzdnn.so` when using the `zdnn` backend path

## Option A: Build With Makefile (Recommended)

From repo root:

```bash
make build
```

What this does:

- Clones ONNX Runtime into `.ort` (default ref: `main`)
- Configures CMake with:
  - `ONNXRUNTIME_INCLUDE_DIR=.ort/include/onnxruntime/core/session`
  - `TELUM_EP_ENABLE_ZDNN=OFF` by default
- Builds into `build/`

To enable zDNN at compile time:

```bash
make build TELUM_EP_ENABLE_ZDNN=ON
```

## Option B: Build With CMake Directly

If you already have ONNX Runtime headers:

```bash
cmake -S . -B build -G Ninja \
  -DONNXRUNTIME_INCLUDE_DIR=/path/to/onnxruntime/include/onnxruntime/core/session \
  -DTELUM_EP_ENABLE_ZDNN=OFF
cmake --build build --parallel
```

## Verify Build Output

Check for the shared library in your build tree:

```bash
find build -type f \( -name 'libtelum_plugin_ep.so' -o -name 'libtelum_plugin_ep.dylib' -o -name 'telum_plugin_ep.dll' \)
```

## Stage for Application Use

Copy the built shared library to a location your application can read, then register that absolute path via:

- `Ort::Env::RegisterExecutionProviderLibrary(...)` in C/C++

See `docs/USER_GUIDE.md` for full runtime usage.

### Where to Put the Built Library

The library should be staged with your host application artifacts. A common layout is:

```text
my-app/
  bin/my_app
  models/model.onnx
  plugins/libtelum_plugin_ep.so
```

Then your app can resolve and pass:

- `/absolute/path/to/my-app/plugins/libtelum_plugin_ep.so`

into `RegisterExecutionProviderLibrary(...)` during startup/session setup.

## Optional: Build Python Source Package

Build sdist:

```bash
make python-sdist
```

If you want the Python package to carry the built plugin library, place the shared library under:

- `python/onnxruntime_ep_telum/lib/`

before packaging.

## Troubleshooting

- Configure fails on missing ORT headers:
  - Verify `-DONNXRUNTIME_INCLUDE_DIR` points to `.../include/onnxruntime/core/session`
- zDNN backend selected but unavailable at runtime:
  - Ensure build used `TELUM_EP_ENABLE_ZDNN=ON`
  - Ensure `libzdnn.so` is discoverable by the dynamic loader on Linux s390x
