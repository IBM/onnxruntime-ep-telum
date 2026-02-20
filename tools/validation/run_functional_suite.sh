#!/usr/bin/env bash
set -euo pipefail

# Functional validation runner for Telum plugin EP.
#
# Usage:
#   ./tools/validation/run_functional_suite.sh \
#     --perf-test /path/to/onnxruntime_perf_test \
#     --model-root /path/to/models \
#     --backend stub|zdnn \
#     --out reports/parity

PERF_TEST=""
MODEL_ROOT=""
BACKEND="stub"
OUT_DIR="reports/parity"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --perf-test) PERF_TEST="$2"; shift 2 ;;
    --model-root) MODEL_ROOT="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --out) OUT_DIR="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$PERF_TEST" || -z "$MODEL_ROOT" ]]; then
  echo "Missing required args --perf-test and --model-root" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/functional_$(date +%Y%m%d_%H%M%S)_${BACKEND}.log"

{
  echo "# Telum Functional Validation"
  echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "backend=${BACKEND}"
  echo "perf_test=${PERF_TEST}"
  echo "model_root=${MODEL_ROOT}"
  echo

  declare -a MODELS=(
    "matmul_chain_512_l4.onnx"
    "matmul_chain_1024_l3.onnx"
  )

  for model in "${MODELS[@]}"; do
    model_path="${MODEL_ROOT}/${model}"
    if [[ ! -f "$model_path" ]]; then
      echo "SKIP missing model: ${model_path}"
      continue
    fi

    echo "===== MODEL ${model} ====="
    "${PERF_TEST}" \
      -e telum -I -m times -r 1 -o 0 -x 1 -v \
      --session_option "ep.TelumPluginExecutionProvider.backend=${BACKEND}" \
      "${model_path}" || true
    echo
  done
} | tee "$LOG"

echo "Wrote functional log: $LOG"
