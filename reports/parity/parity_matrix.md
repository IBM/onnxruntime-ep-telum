# In-Tree to Plugin Parity Matrix

Source baseline: `feature/telum-op-backlog-phase1` in-tree Telum EP.

| In-tree area | Plugin target area | Current status | Notes |
|---|---|---|---|
| `telum_execution_provider.cc` capability scan | `src/telum_plugin_ep/ep.cc` + `src/telum_plugin_ep/telum_capability_policy.cc` | partial | Multi-node graph scan restored. Capability now rejects pseudo-offload. |
| Provider factory fail-fast | `src/telum_plugin_ep/ep_factory.cc` | parity | `backend=zdnn` now fails EP creation if runtime is unavailable. |
| MatMul kernel (`kernels/math/matmul.cc`) | `src/telum_plugin_ep/telum_backend.cc` + `src/telum_plugin_ep/kernels/op_kernel.cc` | partial | Real zDNN MatMul path added for unstacked + stacked/full-broadcast patterns (float path). |
| Gemm kernel (`kernels/math/gemm.cc`) | `src/telum_plugin_ep/telum_backend.cc` + `src/telum_plugin_ep/kernels/op_kernel.cc` | partial | Real zDNN Gemm path added for rank-2 static float, `alpha=beta=1`, constrained `C`. |
| Elementwise kernel (`kernels/math/elementwise.cc`) | `src/telum_plugin_ep/telum_backend.cc` + `src/telum_plugin_ep/kernels/op_kernel.cc` | partial | zDNN offload for Add/Sub/Mul/Div/Min/Max equal-shape static float. |
| Activation kernel (`kernels/activation/activation.cc`) | `src/telum_plugin_ep/telum_backend.cc` + `src/telum_plugin_ep/kernels/op_kernel.cc` | partial | zDNN unary offload for Relu/Gelu/Tanh/Sigmoid/Exp/Log/Sqrt static float. |
| Softmax kernel (`kernels/activation/softmax.cc`) | `src/telum_plugin_ep/telum_backend.cc` + `src/telum_plugin_ep/kernels/op_kernel.cc` | partial | zDNN softmax backend path enabled with axis-last constraints. |
| LayerNorm kernel (`kernels/nn/layer_norm.cc`) | `src/telum_plugin_ep/telum_backend.cc` + `src/telum_plugin_ep/kernels/op_kernel.cc` | partial | zDNN moments/layernorm backend path enabled with axis-last + `[C]` scale/bias constraints. |
| Tensor-manipulation kernels | `src/telum_plugin_ep/kernels/op_kernel.cc` | partial | Admitted on plugin CPU kernel path (no zDNN backend offload); dtype support includes float/fp16/bf16 for data-tensor ops. |
| EPContext serialization/replay | `src/telum_plugin_ep/telum_ep_context_cache.cc` + `src/telum_plugin_ep/ep.cc` | partial | Compatibility path retained; compile path now handles multiple partitions. |

## Immediate Next Parity Targets

1. Extend backend-offloaded dtype coverage (fp16/bf16 on selected zDNN-backed ops).
2. Expand plugin-native automated tests beyond integration scripts (capability semantics + kernel correctness by op).
3. Re-run functional + perf parity on s390x and refresh report artifacts.
