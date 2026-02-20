# Telum Plugin EP Test Plan

This repository currently validates behavior via integration-style runtime tests:

1. `tools/validation/run_functional_suite.sh`
2. `tools/validation/run_perf_suite.sh`

Planned additions:

- Capability policy unit tests
- Strict-mode negative tests
- EPContext replay compatibility tests
- Backend-gating simulation tests (stub-based)
