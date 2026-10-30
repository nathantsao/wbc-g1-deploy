#!/usr/bin/env bash
# Compare policy outputs on OpenVINO CPU/GPU/NPU with ONNX Runtime CPU on seeded synthetic inputs.
# Prints the max action difference, the max joint target difference (action difference times
# the per-joint action scale from config.yaml, in rad) over seeds 1..SEEDS, and the inference
# precision of each config, and saves them with the settings and the stderr log under OUT.
#
# Usage (from anywhere):
#   benchmark/compare_outputs.sh config/policy/wbc/params
#   CONFIGS="openvino:GPU openvino-ep:GPU" benchmark/compare_outputs.sh config/policy/karate/params
#
# Environment variables (default in brackets):
#   CONFIGS  runtime:device pairs to compare [openvino:CPU openvino:GPU openvino:NPU]
#   SEEDS    number of seeded inputs [200]
#   OUT      output directory, must not hold an earlier comparison
#            [new build/benchmark/output_diff/<timestamp>, or <timestamp>-2, -3, ... if it exists]
#   BIN      output diff executable, used as is when set
#            [build/benchmark/policy_output_diff, rebuilt before the comparison]
#   OVEP_VERSION  OpenVINO EP plugin version installed by scripts/bootstrap_thirdparty_intel.sh [1.8.1]
#   EP_LIB   OpenVINO EP plugin library, used for openvino-ep
#            [thirdparty/onnxruntime-ep-openvino-<OVEP_VERSION>-install/Release/lib/...]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "${ROOT}/benchmark/provenance.sh"
BIN_REBUILT=false
[[ -z "${BIN:-}" ]] && BIN_REBUILT=true
BIN="${BIN:-${ROOT}/build/benchmark/policy_output_diff}"
# The plugin path is fixed by version, as in run_all.sh.
OVEP_VERSION="${OVEP_VERSION:-1.8.1}"
DEFAULT_EP_LIB="${ROOT}/thirdparty/onnxruntime-ep-openvino-${OVEP_VERSION}-install/Release/lib/libonnxruntime_providers_openvino_plugin.so"
if [[ -n "${EP_LIB:-}" && ! -f "${EP_LIB}" ]]; then
  echo "EP_LIB not found: ${EP_LIB}" >&2
  exit 1
fi
# ONNX Runtime resolves a relative plugin path from its own lib directory, not the working directory.
EP_LIB="${EP_LIB:+$(realpath -s "${EP_LIB}")}"
[[ -z "${EP_LIB}" && -f "${DEFAULT_EP_LIB}" ]] && EP_LIB="${DEFAULT_EP_LIB}"
# Without OUT, a new directory under build/benchmark/output_diff is made just before the comparison.
OUT="${OUT:-}"
CONFIGS="${CONFIGS:-openvino:CPU openvino:GPU openvino:NPU}"
SEEDS="${SEEDS:-200}"

if [[ $# -ne 1 || ! -f "$1/policy.onnx" || ! -f "$1/config.yaml" ]]; then
  echo "Usage: $0 POLICY_PARAMS_DIR (with policy.onnx and config.yaml)" >&2
  exit 1
fi
[[ "${BIN_REBUILT}" == true ]] && build_default_bin "${ROOT}" policy_output_diff
if [[ ! -x "${BIN}" ]]; then
  echo "Output diff binary not found: ${BIN}" >&2
  exit 1
fi

# karate/wbc use action.scale; velocity uses actions.JointPositionAction.scale.
scale="$(python3 - "$1/config.yaml" <<'EOF'
import sys, yaml
config = yaml.safe_load(open(sys.argv[1]))
action = config.get("action") or config.get("actions", {}).get("JointPositionAction") or {}
if "scale" not in action:
    sys.exit("No action.scale or actions.JointPositionAction.scale in " + sys.argv[1])
print(",".join(repr(float(s)) for s in action["scale"]))
EOF
)"

if [[ -n "${OUT}" ]]; then
  if [[ -e "${OUT}/output_diff.csv" ]]; then
    echo "${OUT} already has a comparison. Set OUT to a new directory." >&2
    exit 1
  fi
  mkdir -p "${OUT}"
else
  OUT="$(make_out_dir "${ROOT}/build/benchmark/output_diff")"
fi

args=(--model "$1/policy.onnx" --seeds "${SEEDS}" --action-scale "${scale}")
[[ -n "${EP_LIB}" ]] && args+=(--provider-library "${EP_LIB}")
read -ra configs <<< "${CONFIGS}"
{
  echo "date=$(date -Iseconds)"
  echo "params=$(cd "$1" && pwd)"
  echo "seeds=${SEEDS}"
  echo "configs=${configs[*]}"
  echo "action_scale=${scale}"
  echo "model_sha256=$(sha256sum "$1/policy.onnx" | cut -d' ' -f1)"
  echo "bin=${BIN}"
  echo "bin_rebuilt=${BIN_REBUILT}"
  print_provenance "${ROOT}" "${EP_LIB}"
} > "${OUT}/environment.txt"

# The CSV goes to the console and to output_diff.csv. stderr is kept apart so log lines cannot be
# read as CSV rows, and is shown after the run.
status=0
"${BIN}" "${args[@]}" "${configs[@]}" 2> "${OUT}/output_diff.err" \
  | tee "${OUT}/output_diff.csv" || status=$?
[[ -s "${OUT}/output_diff.err" ]] && cat "${OUT}/output_diff.err" >&2
echo "Results: ${OUT}" >&2
exit "${status}"
