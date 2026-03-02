# Environment Manifest

- Date (UTC): `2026-03-02T22:46:10Z`
- Host: `ubuntu-24` (`quantexa_vm_2`)
- Architecture: `s390x`
- Kernel: `6.8.0-100-generic`
- Compiler: `g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0`
- CMake: `3.28.3`
- ONNX Runtime perf binary: `/home/ubuntu/qtests/work/ort1242_build/Release/onnxruntime_perf_test`
- ONNX Runtime baseline: `v1.24.2`
- Plugin branch/commit: `k8ika0s/full-parity-plugin-port-ibm-sync` @ `2b1f9ac` (dirty worktree parity run)
- Backend mode: `zdnn`
- zDNN library path: `/home/ubuntu/zdnn-proj/zdnn/lib/libzdnn.so`
- NNPA capability summary: backend runtime ready for current admitted zDNN op set (validated by successful Telum placement/execution in functional run)

## Validation Artifacts

- Fail-fast log: `reports/parity/test_failfast_20260302_224349.log`
- Functional log: `reports/parity/functional_20260302_224349_zdnn.log`
- Perf CSV: `reports/parity/perf_parity_20260302_224349.csv`
- Perf raw log: `reports/parity/perf_parity_20260302_224349.log`
