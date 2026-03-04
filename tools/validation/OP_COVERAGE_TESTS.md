# Op-Coverage Test Matrix

This matrix defines the broader op-coverage harness used by:

- `tools/validation/generate_op_coverage_models.py`
- `tools/validation/run_op_coverage_suite.sh`

The canonical machine-readable source is:

- `tools/validation/op_coverage_matrix.psv`

## Coverage Focus

| Test ID | Model | Expected Path | Focus |
|---|---|---|---|
| `backend_matmul_chain` | `backend_matmul_chain.onnx` | `backend-zdnn` | Chained MatMul+Add+Relu flow to exercise multi-op backend acceleration candidate. |
| `backend_gemm_bias` | `backend_gemm_bias.onnx` | `backend-zdnn` | Gemm path with fused bias shape to validate Gemm constraints and runtime behavior. |
| `backend_add_equal` | `backend_add_equal.onnx` | `backend-zdnn` | Equal-shape binary Add fast path for backend elementwise handling. |
| `backend_tanh_unary` | `backend_tanh_unary.onnx` | `backend-zdnn` | Unary activation coverage for backend unary dispatch. |
| `backend_softmax_last_axis` | `backend_softmax_last_axis.onnx` | `backend-zdnn` | Softmax axis-last constraint path validation. |
| `backend_layernorm_last_axis` | `backend_layernorm_last_axis.onnx` | `backend-zdnn` | LayerNormalization axis-last with scale and bias constraints. |
| `cpu_reshape_static` | `cpu_reshape_static.onnx` | `plugin-cpu` | Static reshape correctness on plugin CPU kernel path. |
| `cpu_transpose_perm` | `cpu_transpose_perm.onnx` | `plugin-cpu` | Transpose with explicit permutation on plugin CPU kernel path. |
| `cpu_squeeze_axes` | `cpu_squeeze_axes.onnx` | `plugin-cpu` | Squeeze behavior using axes tensor input form. |
| `cpu_unsqueeze_axes` | `cpu_unsqueeze_axes.onnx` | `plugin-cpu` | Unsqueeze behavior using axes tensor input form. |
| `cpu_reduce_mean_axes` | `cpu_reduce_mean_axes.onnx` | `plugin-cpu` | ReduceMean correctness with explicit axes and keepdims behavior. |
| `cpu_cast_to_fp16` | `cpu_cast_to_fp16.onnx` | `plugin-cpu` | Cast conversion path to float16 output. |
| `cpu_where_broadcast` | `cpu_where_broadcast.onnx` | `plugin-cpu` | Where broadcast semantics with bool condition initializer. |
| `cpu_expand_static` | `cpu_expand_static.onnx` | `plugin-cpu` | Expand shape broadcast behavior on plugin CPU kernel path. |
| `cpu_concat_axis1` | `cpu_concat_axis1.onnx` | `plugin-cpu` | Concat axis handling and non-axis shape compatibility checks. |
| `cpu_gather_axis1` | `cpu_gather_axis1.onnx` | `plugin-cpu` | Gather indexing semantics with static int64 indices. |
| `cpu_slice_basic` | `cpu_slice_basic.onnx` | `plugin-cpu` | Slice starts/ends/axes/steps semantics with static integer tensors. |

## Typical Flow

1. Generate deterministic models.
2. Run matrix suite (CPU EP and Telum plugin EP).
3. Review generated CSV + Markdown report in `reports/parity/`.

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
