#!/usr/bin/env bash
# Run the policy benchmark on ONNX Runtime CPU and OpenVINO CPU/GPU/NPU, directly and
# through the OpenVINO EP, then print a table of medians and write summary.csv.
#
# A round runs every config once. A run is one benchmark execution of one config. Only runs
# that exit with 0 are summarized; summary.csv has the median of every numeric report field, plus
# the max latency and the total calls over 20 ms over all of those runs.
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
#   BIN         benchmark executable, used as is when set
#               [build/benchmark/onnx_policy_cpu_benchmark, rebuilt before the runs]
#   OVEP_VERSION  OpenVINO EP plugin version installed by scripts/bootstrap_thirdparty_intel.sh [1.8.1]
#   EP_LIB      OpenVINO EP plugin library
#               [thirdparty/onnxruntime-ep-openvino-<OVEP_VERSION>-install/Release/lib/...]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "${ROOT}/benchmark/provenance.sh"
BIN_REBUILT=false
[[ -z "${BIN:-}" ]] && BIN_REBUILT=true
BIN="${BIN:-${ROOT}/build/benchmark/onnx_policy_cpu_benchmark}"
# The plugin path is fixed by version, like ONNX Runtime, so the same plugin is used even when
# several versions are installed.
OVEP_VERSION="${OVEP_VERSION:-1.8.1}"
DEFAULT_EP_LIB="${ROOT}/thirdparty/onnxruntime-ep-openvino-${OVEP_VERSION}-install/Release/lib/libonnxruntime_providers_openvino_plugin.so"
if [[ -n "${EP_LIB:-}" && ! -f "${EP_LIB}" ]]; then
  echo "EP_LIB not found: ${EP_LIB}" >&2
  exit 1
fi
# ONNX Runtime resolves a relative plugin path from its own lib directory, not the working directory.
EP_LIB="${EP_LIB:+$(realpath -s "${EP_LIB}")}"
[[ -z "${EP_LIB}" && -f "${DEFAULT_EP_LIB}" ]] && EP_LIB="${DEFAULT_EP_LIB}"
OUT="${OUT:-${ROOT}/build/benchmark/results/$(date +%Y%m%d-%H%M%S)}"
REPEATS="${REPEATS:-3}"
WARMUP="${WARMUP:-500}"
ITERATIONS="${ITERATIONS:-20000}"
DEFAULT_CONFIGS="onnx:CPU openvino:CPU openvino:GPU openvino:NPU openvino-ep:CPU openvino-ep:GPU openvino-ep:NPU"

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
  echo "OpenVINO EP plugin not found at ${DEFAULT_EP_LIB}. Set EP_LIB or OVEP_VERSION. Skipping openvino-ep runs."
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
if compgen -G "${OUT}/*_round*.txt" > /dev/null || [[ -e "${OUT}/exit_codes.txt" ]]; then
  echo "${OUT} already has benchmark runs. Set OUT to a new directory." >&2
  exit 1
fi

[[ "${BIN_REBUILT}" == true ]] && build_default_bin "${ROOT}" onnx_policy_cpu_benchmark
if [[ ! -x "${BIN}" ]]; then
  echo "Benchmark binary not found: ${BIN}" >&2
  exit 1
fi
mkdir -p "${OUT}"

# CPU power settings and the code, plugin, and driver versions change the results, so record them
# with each result set. Runtime versions and precision are in each run's report, and the model
# SHA-256 is added after the runs.
{
  echo "date=$(date -Iseconds)"
  echo "cpu=$(lscpu | sed -n 's/^Model name: *//p')"
  echo "kernel=$(uname -r)"
  echo "governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown)"
  echo "platform_profile=$(cat /sys/firmware/acpi/platform_profile 2>/dev/null || echo unknown)"
  echo "repeats=${REPEATS} warmup=${WARMUP} iterations=${ITERATIONS} extra_args=$*"
  echo "configs=${configs[*]}"
  echo "bin=${BIN}"
  echo "bin_rebuilt=${BIN_REBUILT}"
  print_provenance "${ROOT}" "${EP_LIB}"
} > "${OUT}/environment.txt"

# Run every config once per round so CPU heat and frequency changes are spread across configs.
for ((round = 1; round <= REPEATS; round++)); do
  for config in "${configs[@]}"; do
    runtime="${config%%:*}"
    device="${config#*:}"
    name="${runtime}_${device}_round${round}"
    args=(--runtime "${runtime}" --device "${device}"
          --warmup "${WARMUP}" --iterations "${ITERATIONS}" --csv "${OUT}/${name}.csv")
    [[ "${runtime}" == "openvino-ep" ]] && args+=(--provider-library "${EP_LIB}")
    echo "[round ${round}/${REPEATS}] ${runtime} ${device}"
    status=0
    # stderr (errors and runtime logs) is kept apart so it cannot be read as a report field.
    "${BIN}" "${args[@]}" "$@" > "${OUT}/${name}.txt" 2> "${OUT}/${name}.err" || status=$?
    echo "${name} ${status}" >> "${OUT}/exit_codes.txt"
    if (( status != 0 )); then
      echo "  Failed (exit ${status}). See ${OUT}/${name}.err"
    fi
  done
done

# Summarize every report field over the runs of each config that exited with 0.
python3 - "${OUT}" "${DEFAULT_CONFIGS}" <<'EOF'
import csv, hashlib, os, re, statistics, sys

out, default_configs = sys.argv[1], sys.argv[2]
FIELD = re.compile(r"^([A-Za-z0-9_.]+)=(.*)$")
# latency_csv is a different path for every run, so it is not a config result.
SKIP = {"latency_csv"}

def parse_report(path):
    report = {}
    with open(path, errors="replace") as f:
        for line in f:
            match = FIELD.match(line.rstrip("\n"))
            if not match or match[1] in SKIP:
                continue
            key, value = match[1], match[2]
            if key == "policy_deadline_misses":  # "N/M (x%)" -> N; M is measured_iterations
                value = value.split("/", 1)[0]
            report[key] = value
    return report

def as_number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def format_number(number):
    return str(int(number)) if number.is_integer() else f"{number:.3f}"

def summarize(values):
    """Median of numeric values; text values are kept if equal, otherwise all are listed."""
    numbers = [as_number(v) for v in values]
    if None not in numbers:
        return format_number(statistics.median(numbers))
    unique = list(dict.fromkeys(values))
    return unique[0] if len(unique) == 1 else " | ".join(unique)

# config name -> parsed reports of runs with exit code 0, and the number of runs with another code
runs_by_config, failed_by_config, failed_runs = {}, {}, []
with open(os.path.join(out, "exit_codes.txt")) as f:
    for line in f:
        name, code = line.split()
        config = re.sub(r"_round\d+$", "", name)
        runs_by_config.setdefault(config, [])
        failed_by_config.setdefault(config, 0)
        if code == "0":
            runs_by_config[config].append(parse_report(os.path.join(out, name + ".txt")))
        else:
            failed_by_config[config] += 1
            failed_runs.append(f"{name} (exit {code})")

# The reports name the model each run loaded, including the compiled default model.
models = dict.fromkeys(r["model"] for runs in runs_by_config.values() for r in runs if "model" in r)
with open(os.path.join(out, "environment.txt"), "a") as f:
    for model in models:
        try:
            with open(model, "rb") as m:
                digest = hashlib.sha256(m.read()).hexdigest()
        except OSError:
            digest = "unknown"
        f.write(f"model_sha256={digest} {model}\n")

# Show configs in DEFAULT_CONFIGS order, e.g. "onnx:CPU" -> "onnx_CPU".
order = [c.replace(":", "_") for c in default_configs.split()]
configs = sorted(runs_by_config, key=lambda c: order.index(c) if c in order else len(order))
# Every field seen in any successful run, in report order.
fields = list(dict.fromkeys(k for c in configs for r in runs_by_config[c] for k in r))
summary = {c: {k: summarize([r[k] for r in runs_by_config[c] if k in r])
               for k in fields if any(k in r for r in runs_by_config[c])}
           for c in configs}

# A median hides the worst call, so the deadline check also uses the max and the total over all
# successful runs of a config.
def over_runs(config, key, reduce):
    numbers = [as_number(r.get(key)) for r in runs_by_config[config]]
    numbers = [n for n in numbers if n is not None]
    return format_number(reduce(numbers)) if numbers else ""

ALL_RUN_FIELDS = [("latency_max_us_all_runs", "latency_max_us", max),
                  ("policy_deadline_misses_total", "policy_deadline_misses", sum),
                  ("measured_iterations_total", "measured_iterations", sum)]
for c in configs:
    for name, key, reduce in ALL_RUN_FIELDS:
        summary[c][name] = over_runs(c, key, reduce)

# Speedup is ONNX Runtime CPU p50 / this config's p50. It is blank if onnx:CPU has no successful run.
def p50(config):
    return as_number(summary[config].get("latency_p50_us"))
baseline = p50("onnx_CPU") if "onnx_CPU" in summary else None
speedup = {c: f"{baseline / p50(c):.2f}x" if baseline and p50(c) else "" for c in configs}

with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    all_run_names = [name for name, _, _ in ALL_RUN_FIELDS]
    writer.writerow(["config", "runs", "failed_runs"] + fields + all_run_names
                    + ["p50_speedup_vs_onnx_cpu"])
    for c in configs:
        writer.writerow([c, len(runs_by_config[c]), failed_by_config[c]]
                        + [summary[c].get(k, "") for k in fields + all_run_names] + [speedup[c]])

# The console table shows the main metrics. summary.csv has every field.
columns = [("p50_us", "latency_p50_us"), ("mean_us", "latency_mean_us"),
           ("p99_us", "latency_p99_us"), ("max_us_all", "latency_max_us_all_runs"),
           (">20ms_all", "policy_deadline_misses_total"),
           ("inf/s", "throughput_inferences_per_sec"), ("init_ms", "initialization_ms"),
           ("cpu_%", "process_cpu_utilization_pct"), ("rss_kib", "rss_after_kib")]

def cell(key, value):
    number = as_number(value)
    if number is None:
        return value or ""
    return str(int(number)) if key == "policy_deadline_misses_total" else f"{number:.1f}"

header = ["config", "ok/total"] + [name for name, _ in columns] + ["p50 speedup"]
print("| " + " | ".join(header) + " |")
print("|" + "---|" * len(header))
for c in configs:
    ok = len(runs_by_config[c])
    runs = f"{ok}/{ok + failed_by_config[c]}" if ok else "FAILED"
    print("| " + " | ".join([c, runs] + [cell(k, summary[c].get(k)) for _, k in columns]
                            + [speedup[c]]) + " |")
print(f"\nResults: {out}")
print("Median of runs that exited with 0; max_us_all is the max and >20ms_all the total over "
      "those runs. See summary.csv for every field, per-run .txt reports and .err logs, and "
      "per-call .csv latencies.")
if failed_runs:
    print("Excluded runs with a non-zero exit code: " + ", ".join(failed_runs), file=sys.stderr)
if not any(runs_by_config.values()):
    print("All configurations failed.", file=sys.stderr)
    sys.exit(1)
EOF
