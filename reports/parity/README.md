# Telum Parity Reports

This directory stores parity validation artifacts for the standalone Telum plugin EP port.

Expected artifacts:

- `functional_*.log`: functional/placement logs.
- `perf_parity_*.csv`: like-for-like CPU vs Telum timing summary.
- `perf_parity_*.log`: raw perf output capture.
- `test_failfast_*.log`: fail-fast guard logs for disabled `drop_constant_initializers=1`.
- `parity_run_*.md`: per-run validation summary with key outcomes and perf snapshot.
- `parity_matrix.md`: in-tree vs plugin parity mapping and status.
- `operator_coverage.md`: per-op pass/fail matrix.
- `environment_manifest.md`: host/build/runtime details.

Generate artifacts using:

- `tools/validation/run_functional_suite.sh`
- `tools/validation/run_perf_suite.sh`
- `tests/run_parity_suite.sh`
