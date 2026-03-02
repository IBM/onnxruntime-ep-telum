#!/usr/bin/env bash
set -euo pipefail

# End-to-end parity-oriented validation runner for the Telum plugin EP.
#
# Validates:
# 1) fail-fast behavior for drop_constant_initializers=1
# 2) functional smoke runs (drop_constant_initializers=0)
# 3) perf suite output shape/sanity (CPU + Telum rows present)
#
# Usage:
#   ./tests/run_parity_suite.sh \
#     --perf-test /path/to/onnxruntime_perf_test \
#     --model-root /path/to/models \
#     --plugin-lib /path/to/libtelum_plugin_ep.so \
#     --runs 20 \
#     --out reports/parity

PERF_TEST=""
MODEL_ROOT=""
PLUGIN_LIB=""
ZDNN_LIB_DIR=""
RUNS=20
OUT_DIR="reports/parity"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --perf-test) PERF_TEST="$2"; shift 2 ;;
    --model-root) MODEL_ROOT="$2"; shift 2 ;;
    --plugin-lib) PLUGIN_LIB="$2"; shift 2 ;;
    --zdnn-lib-dir) ZDNN_LIB_DIR="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    --out) OUT_DIR="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$PERF_TEST" || -z "$MODEL_ROOT" ]]; then
  echo "Missing required args --perf-test and --model-root" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -z "$PLUGIN_LIB" ]]; then
  PLUGIN_LIB="${REPO_ROOT}/build/libtelum_plugin_ep.so"
fi

if [[ ! -f "$PLUGIN_LIB" ]]; then
  echo "Plugin library not found: ${PLUGIN_LIB}" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
FAILFAST_LOG="${OUT_DIR}/test_failfast_$(date +%Y%m%d_%H%M%S).log"
MODEL_PATH="${MODEL_ROOT}/matmul_chain_512_l4.onnx"
EXTRA_ZDNN_ARGS=()
if [[ -n "${ZDNN_LIB_DIR}" ]]; then
  EXTRA_ZDNN_ARGS+=(--zdnn-lib-dir "${ZDNN_LIB_DIR}")
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Missing required model for fail-fast check: ${MODEL_PATH}" >&2
  exit 2
fi

echo "[1/3] Validating fail-fast guard for drop_constant_initializers=1"
set +e
"${PERF_TEST}" \
  -I -m times -r 1 -o 0 -x 1 \
  --plugin_ep_libs "TelumPluginExecutionProvider|${PLUGIN_LIB}" \
  --plugin_eps "TelumPluginExecutionProvider" \
  -C "ep.TelumPluginExecutionProvider.backend|zdnn ep.TelumPluginExecutionProvider.drop_constant_initializers|1" \
  "${MODEL_PATH}" >"${FAILFAST_LOG}" 2>&1
RC=$?
set -e

if [[ $RC -eq 0 ]]; then
  echo "Expected fail-fast error for drop_constant_initializers=1, but run succeeded" >&2
  exit 1
fi
if ! grep -q "Unsupported configuration: 'ep.TelumPluginExecutionProvider.drop_constant_initializers=1'" "${FAILFAST_LOG}"; then
  echo "Fail-fast check did not produce the expected error message" >&2
  echo "See ${FAILFAST_LOG}" >&2
  exit 1
fi

echo "[2/3] Running functional suite with drop_constant_initializers=0"
"${REPO_ROOT}/tools/validation/run_functional_suite.sh" \
  --perf-test "${PERF_TEST}" \
  --model-root "${MODEL_ROOT}" \
  --backend zdnn \
  --plugin-lib "${PLUGIN_LIB}" \
  --drop-constant-initializers 0 \
  --out "${OUT_DIR}" \
  "${EXTRA_ZDNN_ARGS[@]}"

echo "[3/3] Running perf suite and validating CSV shape"
"${REPO_ROOT}/tools/validation/run_perf_suite.sh" \
  --perf-test "${PERF_TEST}" \
  --model-root "${MODEL_ROOT}" \
  --telum-backend zdnn \
  --plugin-lib "${PLUGIN_LIB}" \
  --drop-constant-initializers 0 \
  --runs "${RUNS}" \
  --out "${OUT_DIR}" \
  "${EXTRA_ZDNN_ARGS[@]}"

LATEST_CSV="$(ls -1t "${OUT_DIR}"/perf_parity_*.csv | head -n1)"
if [[ -z "${LATEST_CSV}" || ! -f "${LATEST_CSV}" ]]; then
  echo "Perf suite did not produce a CSV artifact" >&2
  exit 1
fi

LINE_COUNT="$(wc -l < "${LATEST_CSV}")"
if [[ "${LINE_COUNT}" -lt 5 ]]; then
  echo "Perf CSV has too few rows (${LINE_COUNT}); expected header + at least 4 data rows" >&2
  exit 1
fi

if ! awk -F',' 'NR > 1 && $2 == "cpu" {found=1} END {exit found ? 0 : 1}' "${LATEST_CSV}"; then
  echo "Perf CSV missing CPU rows: ${LATEST_CSV}" >&2
  exit 1
fi
if ! awk -F',' 'NR > 1 && $2 == "telum" {found=1} END {exit found ? 0 : 1}' "${LATEST_CSV}"; then
  echo "Perf CSV missing Telum rows: ${LATEST_CSV}" >&2
  exit 1
fi

echo "Parity suite completed successfully."
echo "Fail-fast log: ${FAILFAST_LOG}"
echo "Perf CSV: ${LATEST_CSV}"
