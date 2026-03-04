#!/usr/bin/env bash
set -euo pipefail

# Functional validation runner for Telum plugin EP.
#
# Usage:
#   ./tools/validation/run_functional_suite.sh \
#     --perf-test /path/to/onnxruntime_perf_test \
#     --model-root /path/to/models \
#     --backend zdnn \
#     --out reports/parity

PERF_TEST=""
MODEL_ROOT=""
BACKEND="zdnn"
OUT_DIR="reports/parity"
PLUGIN_LIB=""
ZDNN_LIB_DIR=""
DROP_CONSTANT_INITIALIZERS="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --perf-test) PERF_TEST="$2"; shift 2 ;;
    --model-root) MODEL_ROOT="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
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

if [[ "$BACKEND" != "zdnn" ]]; then
  echo "Unsupported backend '${BACKEND}'. Supported backend: zdnn" >&2
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
LOG="$OUT_DIR/functional_$(date +%Y%m%d_%H%M%S)_${BACKEND}.log"

{
  echo "# Telum Functional Validation"
  echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "backend=${BACKEND}"
  echo "perf_test=${PERF_TEST}"
  echo "model_root=${MODEL_ROOT}"
  echo "plugin_lib=${PLUGIN_LIB}"
  echo "drop_constant_initializers=${DROP_CONSTANT_INITIALIZERS}"
  echo "ld_library_path=${LD_LIBRARY_PATH:-}"
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
      -I -m times -r 1 -o 0 -x 1 -v \
      --plugin_ep_libs "TelumPluginExecutionProvider|${PLUGIN_LIB}" \
      --plugin_eps "TelumPluginExecutionProvider" \
      -C "ep.TelumPluginExecutionProvider.backend|${BACKEND} ep.TelumPluginExecutionProvider.drop_constant_initializers|${DROP_CONSTANT_INITIALIZERS}" \
      "${model_path}"
    echo
  done
} | tee "$LOG"

echo "Wrote functional log: $LOG"
