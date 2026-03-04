# Operator Coverage (Current Branch Snapshot)

Legend:
- `parity`: matches intended backend-offloaded behavior for current scope.
- `partial`: implemented but with narrower constraints than in-tree parity target.
- `plugin-cpu`: admitted to Telum plugin and executed via plugin CPU kernel path (no zDNN backend offload).

| Operator | Capability Gate | Kernel Path | Status | Notes |
|---|---|---|---|---|
| MatMul | static + float + rank>=2 + K match + NNPA dim limit + constrained broadcast pattern | zDNN `matmul_op` / `matmul_bcast_op` | partial | Supports unstacked, stacked, and fully-unstacked broadcast operand patterns. |
| Gemm | static + float + rank2 + alpha=1 + beta=1 + constrained C + NNPA dim limit | zDNN `matmul_op`/`matmul_transpose_op` | partial | `C` supports scalar / `[N]` / `[1,N]` only. |
| Add/Sub/Mul/Div/Min/Max | static + float + equal-shape + rank<=4 + NNPA dim limit | zDNN elementwise binary | partial | Broadcast path is explicitly not offloaded. |
| Relu/Gelu/Tanh/Sigmoid/Exp/Log/Sqrt | static + float + backend available | zDNN unary | partial | Uses flattened 2D descriptor path. |
| Softmax | static + float + axis-last + NNPA dim limit | zDNN `softmax` | partial | Axis must be last dimension. |
| LayerNormalization | static + float + axis-last + scale `[C]` + optional bias `[C]` + NNPA dim limit | zDNN `moments` + `layernorm` (+ CPU scale/bias apply) | partial | Float-only path in plugin branch. |
| Reshape | static + data tensor (float/fp16/bf16) + static int shape input | plugin CPU kernel | plugin-cpu | Not backend-offloaded in current branch. |
| Transpose | static + data tensor (float/fp16/bf16) | plugin CPU kernel | plugin-cpu | Not backend-offloaded in current branch. |
| Squeeze/Unsqueeze | static + data tensor (float/fp16/bf16) + static int axes (if input form used) | plugin CPU kernel | plugin-cpu | Not backend-offloaded in current branch. |
| ReduceMean | static + data tensor (float/fp16/bf16) + output type match + static int axes (if input form used) | plugin CPU kernel | plugin-cpu | Not backend-offloaded in current branch. |
| Cast | static + cast types in {float,fp16,bf16,int32,int64,bool} | plugin CPU kernel | plugin-cpu | Cast conversions route through plugin CPU path. |
| Where | static bool condition + static data tensors (float/fp16/bf16) + type match | plugin CPU kernel | plugin-cpu | Not backend-offloaded in current branch. |
| Expand | static data tensor (float/fp16/bf16) + static int shape input | plugin CPU kernel | plugin-cpu | Not backend-offloaded in current branch. |
| Concat | static data tensors (float/fp16/bf16) + type/rank compatibility | plugin CPU kernel | plugin-cpu | Not backend-offloaded in current branch. |
| Gather | static data tensor (float/fp16/bf16) + static int indices | plugin CPU kernel | plugin-cpu | Not backend-offloaded in current branch. |
| Slice | static data tensor (float/fp16/bf16) + static int starts/ends(/axes/steps) | plugin CPU kernel | plugin-cpu | Not backend-offloaded in current branch. |
