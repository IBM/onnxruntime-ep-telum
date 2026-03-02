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
PLUGIN_LIB=""
ZDNN_LIB_DIR=""
DROP_CONSTANT_INITIALIZERS="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --perf-test) PERF_TEST="$2"; shift 2 ;;
    --model-root) MODEL_ROOT="$2"; shift 2 ;;
    --telum-backend) TELUM_BACKEND="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    --out) OUT_DIR="$2"; shift 2 ;;
    --plugin-lib) PLUGIN_LIB="$2"; shift 2 ;;
    --zdnn-lib-dir) ZDNN_LIB_DIR="$2"; shift 2 ;;
    --drop-constant-initializers) DROP_CONSTANT_INITIALIZERS="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$PERF_TEST" || -z "$MODEL_ROOT" ]]; then
  echo "Missing required args --perf-test and --model-root" >&2
  exit 2
fi

if [[ "$TELUM_BACKEND" != "zdnn" ]]; then
  echo "Unsupported --telum-backend '${TELUM_BACKEND}'. Supported backend: zdnn" >&2
  exit 2
fi

if [[ "$DROP_CONSTANT_INITIALIZERS" != "0" && "$DROP_CONSTANT_INITIALIZERS" != "1" ]]; then
  echo "Invalid --drop-constant-initializers '${DROP_CONSTANT_INITIALIZERS}' (expected 0 or 1)" >&2
  exit 2
fi

if [[ -z "$PLUGIN_LIB" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
  PLUGIN_LIB="${REPO_ROOT}/build/libtelum_plugin_ep.so"
fi

if [[ ! -f "$PLUGIN_LIB" ]]; then
  echo "Plugin library not found: ${PLUGIN_LIB}" >&2
  exit 2
fi

resolve_zdnn_lib_dir() {
  if ldconfig -p 2>/dev/null | grep -q "libzdnn\\.so"; then
    return 0
  fi

  if [[ -n "$ZDNN_LIB_DIR" ]]; then
    if [[ ! -f "${ZDNN_LIB_DIR}/libzdnn.so" && ! -f "${ZDNN_LIB_DIR}/libzdnn.so.0" ]]; then
      echo "Provided --zdnn-lib-dir does not contain libzdnn.so: ${ZDNN_LIB_DIR}" >&2
      return 1
    fi
    export LD_LIBRARY_PATH="${ZDNN_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    return 0
  fi

  local candidates=(
    "/usr/local/lib"
    "/usr/lib"
    "/usr/lib64"
    "/opt/ibm/zdnn/lib"
    "$HOME/zdnn-proj/zdnn/lib"
    "$HOME/zdnn-proj-baseline-main-clean/zdnn/lib"
    "$HOME/zdnn-proj-baseline-main-regtests/zdnn/lib"
  )

  local dir
  for dir in "${candidates[@]}"; do
    if [[ -f "${dir}/libzdnn.so" || -f "${dir}/libzdnn.so.0" ]]; then
      export LD_LIBRARY_PATH="${dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
      return 0
    fi
  done

  echo "Unable to find libzdnn.so in ldconfig or known directories. Use --zdnn-lib-dir." >&2
  return 1
}

resolve_zdnn_lib_dir

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
  local rc=0
  if [[ "$provider" == "cpu" ]]; then
    set +e
    output="$(${PERF_TEST} -e cpu -I -m times -r "${RUNS}" -o 0 -x 1 -S 42 "${model_path}" 2>&1)"
    rc=$?
    set -e
  else
    set +e
    output="$(${PERF_TEST} -I -m times -r "${RUNS}" -o 0 -x 1 -S 42 \
      --plugin_ep_libs "TelumPluginExecutionProvider|${PLUGIN_LIB}" \
      --plugin_eps "TelumPluginExecutionProvider" \
      -C "ep.TelumPluginExecutionProvider.backend|${backend} ep.TelumPluginExecutionProvider.drop_constant_initializers|${DROP_CONSTANT_INITIALIZERS}" \
      "${model_path}" 2>&1)"
    rc=$?
    set -e
  fi

  echo "$output" >> "$RAW"

  if [[ $rc -ne 0 ]]; then
    echo "ERROR: ${provider} run failed for $(basename "$model_path") (rc=${rc})" >> "$RAW"
    return 1
  fi

  local avg
  avg="$(echo "$output" | awk -F':' '/Average inference time cost/{gsub(/ /, "", $2); print $2}' | tail -n1)"
  if [[ -z "$avg" || "$avg" == "NA" ]]; then
    echo "ERROR: unable to parse average latency for ${provider} $(basename "$model_path")" >> "$RAW"
    return 1
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

{
  echo "# Telum Perf Validation Metadata"
  echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "perf_test=${PERF_TEST}"
  echo "model_root=${MODEL_ROOT}"
  echo "plugin_lib=${PLUGIN_LIB}"
  echo "telum_backend=${TELUM_BACKEND}"
  echo "drop_constant_initializers=${DROP_CONSTANT_INITIALIZERS}"
  echo "ld_library_path=${LD_LIBRARY_PATH:-}"
  echo
} >> "$RAW"

echo "Wrote perf CSV: $CSV"
echo "Wrote perf RAW: $RAW"
