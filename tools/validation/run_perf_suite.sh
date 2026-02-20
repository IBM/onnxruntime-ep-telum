#!/usr/bin/env bash
set -euo pipefail

# Perf comparison runner: CPU EP vs Telum plugin EP.
#
# Usage:
#   ./tools/validation/run_perf_suite.sh \
#     --perf-test /path/to/onnxruntime_perf_test \
#     --model-root /path/to/models \
#     --telum-backend zdnn \
#     --runs 20 \
#     --out reports/parity

PERF_TEST=""
MODEL_ROOT=""
TELUM_BACKEND="zdnn"
RUNS=20
OUT_DIR="reports/parity"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --perf-test) PERF_TEST="$2"; shift 2 ;;
    --model-root) MODEL_ROOT="$2"; shift 2 ;;
    --telum-backend) TELUM_BACKEND="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    --out) OUT_DIR="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$PERF_TEST" || -z "$MODEL_ROOT" ]]; then
  echo "Missing required args --perf-test and --model-root" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
CSV="$OUT_DIR/perf_parity_${STAMP}.csv"
RAW="$OUT_DIR/perf_parity_${STAMP}.log"

echo "model,provider,backend,runs,avg_ms" > "$CSV"

run_case() {
  local provider="$1"
  local backend="$2"
  local model_path="$3"

  local output
  if [[ "$provider" == "cpu" ]]; then
    output="$(${PERF_TEST} -e cpu -I -m times -r "${RUNS}" -o 0 -x 1 -S 42 "${model_path}" 2>&1 || true)"
  else
    output="$(${PERF_TEST} -e telum -I -m times -r "${RUNS}" -o 0 -x 1 -S 42 \
      --session_option "ep.TelumPluginExecutionProvider.backend=${backend}" \
      "${model_path}" 2>&1 || true)"
  fi

  echo "$output" >> "$RAW"

  local avg
  avg="$(echo "$output" | awk -F':' '/Average inference time cost/{gsub(/ /, "", $2); print $2}' | tail -n1)"
  if [[ -z "$avg" ]]; then
    avg="NA"
  fi

  echo "$(basename "$model_path"),${provider},${backend},${RUNS},${avg}" >> "$CSV"
}

declare -a MODELS=(
  "matmul_chain_512_l4.onnx"
  "matmul_chain_1024_l3.onnx"
)

for model in "${MODELS[@]}"; do
  model_path="${MODEL_ROOT}/${model}"
  if [[ ! -f "$model_path" ]]; then
    echo "SKIP missing model: ${model_path}" | tee -a "$RAW"
    continue
  fi

  run_case "cpu" "cpu" "$model_path"
  run_case "telum" "$TELUM_BACKEND" "$model_path"
done

echo "Wrote perf CSV: $CSV"
echo "Wrote perf RAW: $RAW"
