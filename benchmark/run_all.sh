#!/usr/bin/env bash
# Run the policy benchmark on ONNX Runtime CPU and OpenVINO CPU/GPU/NPU, directly and
# through the OpenVINO EP, then print a table of medians and write summary.csv.
#
# A round runs every config once. A run is one benchmark execution of one config.
#
# Usage (from anywhere):
#   benchmark/run_all.sh
#   REPEATS=5 ITERATIONS=50000 benchmark/run_all.sh --model config/policy/wbc/params/policy.onnx
#   CONFIGS="onnx:CPU openvino:CPU openvino-ep:CPU" benchmark/run_all.sh --threads 2
#
# Extra arguments are passed to every run. The script rejects the options it sets for
# each configuration: --runtime, --device, --warmup, --iterations, --csv,
# --provider-library, and --list-devices.
#
# Environment variables (default in brackets):
#   CONFIGS     runtime:device pairs to run [all 7, see DEFAULT_CONFIGS]
#   REPEATS     number of rounds [3]
#   WARMUP      warm-up calls per run [500]
#   ITERATIONS  measured calls per run [20000]
#   OUT         output directory, must not hold earlier runs [build/benchmark/results/<timestamp>]
#   BIN         benchmark executable [build/benchmark/onnx_policy_cpu_benchmark]
#   EP_LIB      OpenVINO EP plugin library [searched under thirdparty/]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BIN:-${ROOT}/build/benchmark/onnx_policy_cpu_benchmark}"
# Without "|| true", a missing thirdparty/ would end the script with no message.
EP_LIB="${EP_LIB:-$(find "${ROOT}/thirdparty" -path '*-install/*' \
  -name 'libonnxruntime_providers_openvino_plugin.so' 2>/dev/null | head -n1 || true)}"
OUT="${OUT:-${ROOT}/build/benchmark/results/$(date +%Y%m%d-%H%M%S)}"
REPEATS="${REPEATS:-3}"
WARMUP="${WARMUP:-500}"
ITERATIONS="${ITERATIONS:-20000}"
DEFAULT_CONFIGS="onnx:CPU openvino:CPU openvino:GPU openvino:NPU openvino-ep:CPU openvino-ep:GPU openvino-ep:NPU"

if [[ ! -x "${BIN}" ]]; then
  echo "Benchmark binary not found: ${BIN}" >&2
  exit 1
fi

# These options are set for each configuration. Passing them again would mislabel the results.
for arg in "$@"; do
  case "${arg}" in
    --runtime|--device|--warmup|--iterations|--csv|--provider-library|--list-devices)
      echo "${arg} is set by run_all.sh for each configuration. Use CONFIGS, WARMUP, ITERATIONS, OUT, or EP_LIB instead." >&2
      exit 1
      ;;
  esac
done

read -ra configs <<< "${CONFIGS:-${DEFAULT_CONFIGS}}"
if [[ -z "${EP_LIB}" ]]; then
  echo "OpenVINO EP plugin not found. Skipping openvino-ep runs."
  kept=()
  for config in "${configs[@]}"; do
    [[ "${config}" == openvino-ep:* ]] || kept+=("${config}")
  done
  configs=("${kept[@]}")
fi
if [[ ${#configs[@]} -eq 0 ]]; then
  echo "No configurations to run." >&2
  exit 1
fi

# Old runs in a reused OUT would be mixed into the summary.
if compgen -G "${OUT}/*_round*.txt" > /dev/null; then
  echo "${OUT} already has benchmark runs. Set OUT to a new directory." >&2
  exit 1
fi
mkdir -p "${OUT}"
# CPU power settings change the results, so record them with each result set.
{
  echo "date=$(date -Iseconds)"
  echo "cpu=$(lscpu | sed -n 's/^Model name: *//p')"
  echo "kernel=$(uname -r)"
  echo "governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown)"
  echo "platform_profile=$(cat /sys/firmware/acpi/platform_profile 2>/dev/null || echo unknown)"
  echo "repeats=${REPEATS} warmup=${WARMUP} iterations=${ITERATIONS} extra_args=$*"
  echo "configs=${configs[*]}"
} > "${OUT}/environment.txt"

# Run every config once per round, so CPU heat and frequency changes affect all configs equally.
for ((round = 1; round <= REPEATS; round++)); do
  for config in "${configs[@]}"; do
    runtime="${config%%:*}"
    device="${config#*:}"
    name="${runtime}_${device}_round${round}"
    args=(--runtime "${runtime}" --device "${device}"
          --warmup "${WARMUP}" --iterations "${ITERATIONS}" --csv "${OUT}/${name}.csv")
    [[ "${runtime}" == "openvino-ep" ]] && args+=(--provider-library "${EP_LIB}")
    echo "[round ${round}/${REPEATS}] ${runtime} ${device}"
    if ! "${BIN}" "${args[@]}" "$@" > "${OUT}/${name}.txt" 2>&1; then
      echo "  Failed. See ${OUT}/${name}.txt"
    fi
  done
done

# Summarize: median of each metric over the successful runs of each config.
python3 - "${OUT}" "${DEFAULT_CONFIGS}" <<'EOF'
import glob, os, re, statistics, sys

out, default_configs = sys.argv[1], sys.argv[2]
keys = ["latency_p50_us", "latency_mean_us", "latency_p99_us", "latency_max_us",
        "throughput_inferences_per_sec", "initialization_ms",
        "process_cpu_utilization_pct", "rss_after_kib"]

# config name -> list of parsed reports from successful runs (empty if every run failed)
runs_by_config = {}
for path in sorted(glob.glob(os.path.join(out, "*_round*.txt"))):
    config = re.sub(r"_round\d+\.txt$", "", os.path.basename(path))
    report = dict(line.strip().split("=", 1) for line in open(path) if "=" in line)
    runs_by_config.setdefault(config, [])
    if "latency_p50_us" in report:
        runs_by_config[config].append(report)

# Show configs in DEFAULT_CONFIGS order, e.g. "onnx:CPU" -> "onnx_CPU".
order = [c.replace(":", "_") for c in default_configs.split()]
# Speedup is ONNX Runtime CPU p50 / this config's p50. It is blank if onnx:CPU was not run.
baseline = None
rows = []
for config in sorted(runs_by_config, key=lambda c: order.index(c) if c in order else len(order)):
    runs = runs_by_config[config]
    if not runs:
        rows.append([config, "FAILED"] + [""] * (len(keys) + 1))
        continue
    median = {k: statistics.median(float(r[k]) for r in runs) for k in keys}
    if config == "onnx_CPU":
        baseline = median["latency_p50_us"]
    speedup = f"{baseline / median['latency_p50_us']:.2f}x" if baseline else ""
    rows.append([config, str(len(runs))] + [f"{median[k]:.1f}" for k in keys] + [speedup])

csv_header = ["config", "runs"] + keys + ["p50_speedup_vs_onnx_cpu"]
with open(os.path.join(out, "summary.csv"), "w") as f:
    f.write(",".join(csv_header) + "\n")
    for row in rows:
        f.write(",".join(row) + "\n")

# Shorter column names for the console table, same order as csv_header.
table_header = ["config", "runs", "p50_us", "mean_us", "p99_us", "max_us", "inf/s",
                "init_ms", "cpu_%", "rss_kib", "p50 speedup"]
print("| " + " | ".join(table_header) + " |")
print("|" + "---|" * len(table_header))
for row in rows:
    print("| " + " | ".join(row) + " |")
print(f"\nResults: {out}")
print("Median of runs. See summary.csv, per-run .txt reports, and per-call .csv latencies.")
if all(row[1] == "FAILED" for row in rows):
    print("All configurations failed.", file=sys.stderr)
    sys.exit(1)
EOF
