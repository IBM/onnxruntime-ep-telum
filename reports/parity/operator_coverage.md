# Operator Coverage Report Template

| Operator | Capability Gate | Kernel Path | Status | Notes |
|---|---|---|---|---|
| MatMul | static + dtype + K match | plugin kernel | pending | |
| Gemm | static + dtype + 2D checks | plugin kernel | pending | |
| Add/Sub/Mul/Div/Min/Max | static + broadcast | plugin kernel | pending | |
| Relu/Gelu/Tanh/Sigmoid/Exp/Log/Sqrt | static + dtype + NNPA gate | plugin kernel | pending | |
| Softmax | axis-last + NNPA gate | plugin kernel | pending | |
| LayerNormalization | axis-last + scale/bias shape + NNPA gate | plugin kernel | pending | |
| Reshape | static input + int shape tensor | plugin kernel | pending | |
| Transpose | static + valid perm | plugin kernel | pending | |
| Squeeze/Unsqueeze | static + valid axes | plugin kernel | pending | |
| ReduceMean | static + valid axes | plugin kernel | pending | |
| Cast | static + supported type pairs | plugin kernel | pending | |
| Where | static + bool cond + broadcast | plugin kernel | pending | |
| Expand | static + int shape tensor + broadcast | plugin kernel | pending | |
| Concat | static + axis + shape checks | plugin kernel | pending | |
| Gather | static + int indices | plugin kernel | pending | |
| Slice | static + int starts/ends/axes/steps | plugin kernel | pending | |
