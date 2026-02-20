# Telum Parity Reports

This directory stores parity validation artifacts for the standalone Telum plugin EP port.

Expected artifacts:

- `functional_*.log`: functional/placement logs.
- `perf_parity_*.csv`: like-for-like CPU vs Telum timing summary.
- `perf_parity_*.log`: raw perf output capture.
- `operator_coverage.md`: per-op pass/fail matrix.
- `environment_manifest.md`: host/build/runtime details.

Generate artifacts using:

- `tools/validation/run_functional_suite.sh`
- `tools/validation/run_perf_suite.sh`
