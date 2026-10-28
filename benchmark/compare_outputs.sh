#!/usr/bin/env bash
# Compare policy outputs on OpenVINO CPU/GPU/NPU with ONNX Runtime CPU on seeded synthetic inputs.
# Prints the max action difference and the max joint target difference (action difference times
# the per-joint action scale from config.yaml, in rad) over seeds 1..SEEDS.
#
# Usage (from anywhere):
#   benchmark/compare_outputs.sh config/policy/wbc/params
#   CONFIGS="openvino:GPU openvino-ep:GPU" benchmark/compare_outputs.sh config/policy/karate/params
#
# Environment variables (default in brackets):
#   CONFIGS  runtime:device pairs to compare [openvino:CPU openvino:GPU openvino:NPU]
#   SEEDS    number of seeded inputs [200]
#   BIN      output diff executable [build/benchmark/policy_output_diff]
#   EP_LIB   OpenVINO EP plugin library, used for openvino-ep [searched under thirdparty/]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BIN:-${ROOT}/build/benchmark/policy_output_diff}"
EP_LIB="${EP_LIB:-$(find "${ROOT}/thirdparty" -path '*-install/*' \
  -name 'libonnxruntime_providers_openvino_plugin.so' 2>/dev/null | head -n1 || true)}"
CONFIGS="${CONFIGS:-openvino:CPU openvino:GPU openvino:NPU}"
SEEDS="${SEEDS:-200}"

if [[ $# -ne 1 || ! -f "$1/policy.onnx" || ! -f "$1/config.yaml" ]]; then
  echo "Usage: $0 POLICY_PARAMS_DIR (with policy.onnx and config.yaml)" >&2
  exit 1
fi
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

args=(--model "$1/policy.onnx" --seeds "${SEEDS}" --action-scale "${scale}")
[[ -n "${EP_LIB}" ]] && args+=(--provider-library "${EP_LIB}")
read -ra configs <<< "${CONFIGS}"
"${BIN}" "${args[@]}" "${configs[@]}"
