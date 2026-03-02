#!/usr/bin/env bash
set -euo pipefail

# Matrix-driven op-coverage validation runner: CPU EP vs Telum plugin EP.
#
# Usage:
#   ./tools/validation/run_op_coverage_suite.sh \
#     --perf-test /path/to/onnxruntime_perf_test \
#     --model-root /path/to/op_coverage_models \
#     --plugin-lib /path/to/libtelum_plugin_ep.so \
#     --out reports/parity

PERF_TEST=""
MODEL_ROOT=""
PLUGIN_LIB=""
BACKEND="zdnn"
MATRIX=""
OUT_DIR="reports/parity"
SEED=42
ZDNN_LIB_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --perf-test) PERF_TEST="$2"; shift 2 ;;
    --model-root) MODEL_ROOT="$2"; shift 2 ;;
    --plugin-lib) PLUGIN_LIB="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --matrix) MATRIX="$2"; shift 2 ;;
    --out) OUT_DIR="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --zdnn-lib-dir) ZDNN_LIB_DIR="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${PERF_TEST}" || -z "${MODEL_ROOT}" ]]; then
  echo "Missing required args --perf-test and --model-root" >&2
  exit 2
fi

if [[ "${BACKEND}" != "zdnn" ]]; then
  echo "Unsupported backend '${BACKEND}'. Supported backend: zdnn" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -z "${PLUGIN_LIB}" ]]; then
  PLUGIN_LIB="${REPO_ROOT}/build/libtelum_plugin_ep.so"
fi
if [[ -z "${MATRIX}" ]]; then
  MATRIX="${SCRIPT_DIR}/op_coverage_matrix.psv"
fi

if [[ ! -f "${PLUGIN_LIB}" ]]; then
  echo "Plugin library not found: ${PLUGIN_LIB}" >&2
  exit 2
fi
if [[ ! -f "${MATRIX}" ]]; then
  echo "Coverage matrix not found: ${MATRIX}" >&2
  exit 2
fi

resolve_zdnn_lib_dir() {
  if ldconfig -p 2>/dev/null | grep -q "libzdnn\\.so"; then
    return 0
  fi

  if [[ -n "${ZDNN_LIB_DIR}" ]]; then
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

mkdir -p "${OUT_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
CSV="${OUT_DIR}/op_coverage_${STAMP}.csv"
RAW="${OUT_DIR}/op_coverage_${STAMP}.log"
MD="${OUT_DIR}/op_coverage_${STAMP}.md"

echo "id,category,model,expected_path,cpu_runs,telum_runs,cpu_avg_ms,telum_avg_ms,speedup_cpu_div_telum,status,focus" > "${CSV}"

escape_csv_field() {
  local value="$1"
  value="${value//\"/\"\"}"
  printf "\"%s\"" "${value}"
}

run_case() {
  local provider="$1"
  local model_path="$2"
  local runs="$3"

  local output rc avg
  set +e
  if [[ "${provider}" == "cpu" ]]; then
    output="$("${PERF_TEST}" -e cpu -I -m times -r "${runs}" -o 0 -x 1 -S "${SEED}" "${model_path}" 2>&1)"
  else
    output="$("${PERF_TEST}" -I -m times -r "${runs}" -o 0 -x 1 -S "${SEED}" \
      --plugin_ep_libs "TelumPluginExecutionProvider|${PLUGIN_LIB}" \
      --plugin_eps "TelumPluginExecutionProvider" \
      -C "ep.TelumPluginExecutionProvider.backend|${BACKEND} ep.TelumPluginExecutionProvider.drop_constant_initializers|0" \
      "${model_path}" 2>&1)"
  fi
  rc=$?
  set -e

  echo "----- provider=${provider} model=$(basename "${model_path}") runs=${runs} -----" >> "${RAW}"
  echo "${output}" >> "${RAW}"
  echo >> "${RAW}"

  if [[ ${rc} -ne 0 ]]; then
    echo "NA"
    return 1
  fi

  avg="$(echo "${output}" | awk -F':' '/Average inference time cost/{gsub(/ /, "", $2); print $2}' | tail -n1)"
  if [[ -z "${avg}" ]]; then
    echo "NA"
    return 1
  fi

  echo "${avg}"
  return 0
}

{
  echo "# Telum Op-Coverage Validation"
  echo
  echo "- Date (UTC): $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "- perf_test: ${PERF_TEST}"
  echo "- model_root: ${MODEL_ROOT}"
  echo "- plugin_lib: ${PLUGIN_LIB}"
  echo "- backend: ${BACKEND}"
  echo "- matrix: ${MATRIX}"
  echo "- seed: ${SEED}"
  echo "- ld_library_path: ${LD_LIBRARY_PATH:-}"
  echo
  echo "| Test ID | Category | Model | Expected Path | CPU avg (ms) | Telum avg (ms) | CPU/Telum | Status | Focus |"
  echo "|---|---|---|---|---:|---:|---:|---|---|"
} > "${MD}"

tail -n +2 "${MATRIX}" | while IFS='|' read -r test_id model category expected_path cpu_runs telum_runs focus; do
  if [[ -z "${test_id}" ]]; then
    continue
  fi
  model_path="${MODEL_ROOT}/${model}"
  if [[ ! -f "${model_path}" ]]; then
    status="missing-model"
    cpu_avg="NA"
    telum_avg="NA"
    speedup="NA"
    echo "SKIP missing model: ${model_path}" >> "${RAW}"
  else
    status="pass"
    cpu_avg="$(run_case "cpu" "${model_path}" "${cpu_runs}")" || status="fail"
    telum_avg="$(run_case "telum" "${model_path}" "${telum_runs}")" || status="fail"
    if [[ "${cpu_avg}" == "NA" || "${telum_avg}" == "NA" ]]; then
      speedup="NA"
      status="fail"
    else
      speedup="$(awk -v c="${cpu_avg}" -v t="${telum_avg}" 'BEGIN { if (t == 0) print "NA"; else printf "%.4f", c/t }')"
    fi
  fi

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${test_id}" \
    "${category}" \
    "${model}" \
    "${expected_path}" \
    "${cpu_runs}" \
    "${telum_runs}" \
    "${cpu_avg}" \
    "${telum_avg}" \
    "${speedup}" \
    "${status}" \
    "$(escape_csv_field "${focus}")" >> "${CSV}"

  printf '| %s | %s | `%s` | `%s` | %s | %s | %s | %s | %s |\n' \
    "${test_id}" \
    "${category}" \
    "${model}" \
    "${expected_path}" \
    "${cpu_avg}" \
    "${telum_avg}" \
    "${speedup}" \
    "${status}" \
    "${focus}" >> "${MD}"
done

echo >> "${MD}"
echo "Artifacts:" >> "${MD}"
echo "- CSV: \`${CSV}\`" >> "${MD}"
echo "- Raw log: \`${RAW}\`" >> "${MD}"

if awk -F',' 'NR>1 && $10 != "pass" { found=1 } END { exit found ? 0 : 1 }' "${CSV}"; then
  echo "Op-coverage suite completed with failures. See ${CSV} and ${RAW}" >&2
  exit 1
fi

echo "Wrote op-coverage CSV: ${CSV}"
echo "Wrote op-coverage log: ${RAW}"
echo "Wrote op-coverage report: ${MD}"
