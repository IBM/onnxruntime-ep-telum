# Telum Plugin EP Test Plan

This repository validates behavior via integration-style runtime tests:

1. `tools/validation/run_functional_suite.sh`
2. `tools/validation/run_perf_suite.sh`
3. `tools/validation/run_op_coverage_suite.sh`
4. `tests/run_parity_suite.sh`

`tests/run_parity_suite.sh` is the end-to-end gate used for parity validation. It checks:

1. fail-fast behavior for `ep.TelumPluginExecutionProvider.drop_constant_initializers=1`
2. functional smoke coverage on static benchmark models with `drop_constant_initializers=0`
3. perf suite output sanity (CPU and Telum rows in generated CSV)

`tools/validation/run_op_coverage_suite.sh` is the broader operator matrix runner. It uses:

- `tools/validation/op_coverage_matrix.psv` for test definitions and per-test focus notes
- `tools/validation/generate_op_coverage_models.py` for deterministic model generation

The op-coverage suite emits:

- `reports/parity/op_coverage_<timestamp>.csv`
- `reports/parity/op_coverage_<timestamp>.md`
- `reports/parity/op_coverage_<timestamp>.log`

Example:

```bash
python3 tools/validation/generate_op_coverage_models.py \
  --out-dir ~/onnx-proj/bench_models/op_coverage \
  --seed 42

tools/validation/run_op_coverage_suite.sh \
  --perf-test ~/onnx-proj/onnxruntime/build/s390x_telum/Release/onnxruntime_perf_test \
  --model-root ~/onnx-proj/bench_models/op_coverage \
  --plugin-lib ~/onnx-proj/onnxruntime-ep-telum/build/libtelum_plugin_ep.so \
  --out reports/parity
```

Example:

```bash
./tests/run_parity_suite.sh \
  --perf-test ~/onnx-proj/onnxruntime/build/s390x_telum/Release/onnxruntime_perf_test \
  --model-root ~/onnx-proj/bench_models \
  --plugin-lib ~/onnx-proj/onnxruntime-ep-telum/build/libtelum_plugin_ep.so \
  --out reports/parity
```
