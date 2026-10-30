# Policy inference benchmark

This standalone C++ harness benchmarks the active ONNX policy using ONNX Runtime, direct OpenVINO Runtime, or the ONNX Runtime OpenVINO Execution Provider (EP) plugin. CPU is the default device. OpenVINO supports explicit CPU, Intel GPU, and Intel NPU selection; native ONNX Runtime also supports NVIDIA GPUs through its CUDA provider. All backends use the same deterministic synthetic input tensors, warm-up count, and metric collection. ONNX Runtime uses the same `ORT_ENABLE_EXTENDED` graph optimization setting as `OrtRunner`.

Building requires the OpenVINO C++ development package with the ONNX frontend and desired device plugins, plus ONNX Runtime. Set `OpenVINO_DIR` to the package's CMake directory if it is not found automatically. The `openvino-ep` mode additionally requires the separately built or packaged `onnxruntime-ep-openvino` shared library. The bundled ONNX Runtime is 1.30.0; the plugin API requires ONNX Runtime 1.23.0 or newer. The current upstream OpenVINO EP plugin build requires ONNX Runtime 1.30.0+ and OpenVINO 2026.4+. Intel GPU/NPU modes also require the corresponding hardware drivers.

On an Intel machine, `scripts/bootstrap_thirdparty_intel.sh` installs ONNX Runtime, OpenVINO, and the OpenVINO EP plugin under `thirdparty/`.

If `--runtime openvino --list-devices` shows only CPU, the GPU and NPU user-mode drivers are probably not installed. Then run the script with `--system-deps`, which needs sudo. `--system-deps` does the following:

- Installs build tools such as CMake and Ninja.
- Installs the newest GPU compute runtime and Level Zero from the `kobuk-team/intel-graphics` PPA.
- Installs NPU driver v1.38.0, unless an NPU driver is already installed.
- Adds you to the `render` group. Log out and back in afterward.

Each measured call includes one input-map copy and the selected backend's tensor setup, inference, and output handling. Each backend keeps its default optimization settings. It also keeps its default thread pool unless `--threads N` is given (see below). So a default run is an out-of-box comparison, not a comparison with the same thread count.

The benchmark discovers model input names and static shapes, then supplies deterministic synthetic values. Inputs and the first output must be float32; the first output must be rank 2 with a static action dimension. Use single-action-row policies (output `[1, action_dimension]`) for comparable output handling: the existing `OrtRunner` copies one action row, whereas other backends copy the complete first output tensor. The ONNX CPU path still uses `OrtRunner::act()` unchanged. Each backend is measured synchronously end to end, including per-call tensor construction, inference, any required host/device transfers, output copy, and return value. Observation generation, reference-clip loading, robot I/O, and the FSM are not included. This isolates policy inference and its wrapper overhead; it is not a full controller or hardware-in-the-loop timing test.

Build and run from the repository root:

```bash
cmake -S benchmark -B build/benchmark
cmake --build build/benchmark -j
./build/benchmark/onnx_policy_cpu_benchmark --runtime onnx
./build/benchmark/onnx_policy_cpu_benchmark --runtime openvino
```

The build type defaults to `Release`. Pass `-DCMAKE_BUILD_TYPE=...` to override it. The executable retains its original name for compatibility, including accelerator runs.

A run fails with an error if any warm-up or measured output is NaN or infinite. The check runs after each call returns, so it is not part of the per-call latency samples. Like the output checksum, it is included in the wall time and throughput (about 10 ns per call).

## Run all configurations

To compare ONNX Runtime CPU against OpenVINO CPU/GPU/NPU (and the OpenVINO EP plugin on each device, when built) in one step:

```bash
benchmark/run_all.sh
```

- The command above runs the karate policy, which is the default model. For another policy, add `--model config/policy/<name>/params/policy.onnx`, where `<name>` is `wbc` or `velocity`.
- Each of the `REPEATS` rounds (default 3) runs every configuration once so CPU heat and frequency changes are spread across configurations. Each round uses a new random order. With a fixed or rotated order, a configuration would run at the same position or after the same configuration in every round, for example `openvino:CPU` right after the 14-core `onnx:CPU` run. The median cannot remove an effect that is the same in every round. A random order turns such effects into noise that the median reduces, but does not remove them. `environment.txt` records the seed as `order_seed` and the order of each round as `round<N>_order`. Set `ORDER_SEED` to repeat the same orders.
- The script prints a Markdown table of per-configuration medians and writes `summary.csv`, the raw per-run reports (`.txt` for stdout, `.err` for stderr), per-call latency CSVs, `exit_codes.txt`, and `environment.txt` under `build/benchmark/results/<timestamp>/`.
- `environment.txt` records the CPU governor and platform profile, the git commit and whether tracked files had uncommitted changes (`git_dirty`), the benchmark binary and whether the script rebuilt it (`bin_rebuilt`), the resolved OpenVINO EP plugin file, the GPU/NPU/Level Zero driver package versions, and the SHA-256 of each model the runs loaded.
- Without `BIN`, the script first runs `cmake --build build/benchmark` for the benchmark target, so the binary matches the source tree that `git_commit` and `git_dirty` describe. Nothing is compiled if the build is up to date. If it compiles something, the CPU is still warm for the first runs, so build first for final results. A `BIN` that is set is used as is (`bin_rebuilt=false`) and may not match `git_commit`.
- Only runs that exit with code 0 are summarized. `summary.csv` has one column per report field: the median for numeric fields (`policy_deadline_misses` is the miss count), the value for text fields that are equal in every run, and all values joined by ` | ` for text fields that differ. Its `runs` and `failed_runs` columns count the runs that were and were not used.
- A median hides the slowest call, so `summary.csv` also has `latency_max_us_all_runs` (the max latency over all successful runs), `policy_deadline_misses_total` (calls over 20 ms, summed over those runs), and `measured_iterations_total`. The console table shows these as `max_us_all` and `>20ms_all`.
- A configuration with no successful run is reported as `FAILED` without stopping the others. The script exits with an error only if every configuration fails.
- Set `CONFIGS` to a space-separated list of `runtime:device` pairs to run a subset (for example `CONFIGS="onnx:CPU openvino:CPU openvino-ep:CPU"` together with `--threads`).
- Override `BIN`, `EP_LIB`, `OUT`, `WARMUP`, `ITERATIONS`, or `ORDER_SEED` through the environment. `OUT` must not already contain runs, so old results are never mixed into a new summary.
- `EP_LIB` defaults to a fixed path, `thirdparty/onnxruntime-ep-openvino-<OVEP_VERSION>-install/Release/lib/libonnxruntime_providers_openvino_plugin.so`, where `OVEP_VERSION` defaults to 1.8.1 as in the bootstrap script. So the same plugin is used even when several versions are installed. If that file does not exist, openvino-ep runs are skipped. An `EP_LIB` that does not exist is an error, and a relative `EP_LIB` is made absolute because ONNX Runtime would resolve it from its own lib directory. `compare_outputs.sh` uses the same rule.
- Extra arguments are passed to every run. The script rejects the options it sets for each configuration: `--runtime`, `--device`, `--warmup`, `--iterations`, `--csv`, `--provider-library`, and `--list-devices`.

## Compare outputs

To check how much accelerator outputs differ from ONNX Runtime CPU (for example because GPU and NPU default to FP16), compare the actions for seeded synthetic inputs:

```bash
benchmark/compare_outputs.sh config/policy/karate/params
```

- The argument is the params directory of the policy to check: `config/policy/<name>/params`, where `<name>` is `karate`, `wbc`, or `velocity`.
- It prints, per configuration, the max absolute action difference over seeds 1 to `SEEDS` (default 200), the max joint target difference in radians, and the `inference_precision` the configuration ran with (see the report fields below).
- The same CSV is saved as `output_diff.csv`, with stderr in `output_diff.err` and the settings (params directory, seeds, configurations, action scale) in `environment.txt`, under `build/benchmark/output_diff/<timestamp>/`. `environment.txt` also has the model SHA-256 and the same git, binary, plugin, and driver fields as `run_all.sh`. Without `BIN`, the script also rebuilds `policy_output_diff` first. Set `OUT` to choose the directory; it must not already hold a comparison.
- The joint target difference is the action difference times the per-joint action scale from `config.yaml` (`action.scale`, or `actions.JointPositionAction.scale`).
- A configuration with a NaN or infinite output is reported as `FAILED` and counts as a failed configuration. A NaN or infinite ONNX Runtime CPU baseline output or action scale stops the comparison with an error (exit code 1).
- `--seeds` must be a plain decimal integer from 1 to 1000000.
- `CONFIGS` defaults to `openvino:CPU openvino:GPU openvino:NPU`. Set it to check other configurations, for example `CONFIGS="openvino-ep:GPU openvino-ep:NPU"`.
- The script wraps `build/benchmark/policy_output_diff`, which can also be run directly with `--action-scale`.
- `compare_outputs.sh` needs Python 3 with PyYAML to read `config.yaml`. Install it with `sudo apt install python3-yaml`, or with `pip install pyyaml` in a virtual environment.

## Device selection

```bash
# Discover devices without opening a model
./build/benchmark/onnx_policy_cpu_benchmark --runtime openvino --list-devices
./build/benchmark/onnx_policy_cpu_benchmark --runtime onnx --list-devices

# Direct OpenVINO on Intel GPU or NPU
./build/benchmark/onnx_policy_cpu_benchmark --runtime openvino --device GPU
./build/benchmark/onnx_policy_cpu_benchmark --runtime openvino --device GPU.1
./build/benchmark/onnx_policy_cpu_benchmark --runtime openvino --device NPU
```

Device names are case-insensitive. OpenVINO accepts `CPU`, `GPU`, `GPU.N`, and `NPU`; `GPU` and `GPU.0` both select the first GPU, including when discovery exposes only one of those aliases. `AUTO`, `MULTI`, and heterogeneous execution are intentionally unsupported. Unavailable or ambiguous devices fail explicitly, never silently switching to CPU.

To build against a different ONNX Runtime package, pass its package root when configuring:

```bash
cmake -S benchmark -B build/benchmark \
  -DONNXRUNTIME_DIR=/path/to/onnxruntime
cmake --build build/benchmark -j
```

Native ONNX Runtime GPU execution requires a **CUDA-enabled** ONNX Runtime package plus compatible CUDA/cuDNN; the bundled CPU package does not provide CUDA:

```bash
./build/benchmark/onnx_policy_cpu_benchmark \
  --runtime onnx --provider cuda --device GPU --device-id 0
./build/benchmark/onnx_policy_cpu_benchmark \
  --runtime onnx --provider cuda --device GPU --list-devices
```

`--provider` is only valid for `onnx` (`cpu` or `cuda`, default `cpu`), and `--device-id` is only valid for CUDA. CUDA listing verifies that the provider is present, not that a particular GPU or driver is usable: ORT does not expose portable CUDA hardware enumeration through this path. Session creation validates the chosen index and dependencies. For Intel GPU/NPU execution through ONNX Runtime, use `openvino-ep` instead.

Build or install the plugin following [onnxruntime-ep-openvino](https://github.com/onnxruntime/onnxruntime-ep-openvino), then pass the plugin shared-library path:

```bash
./build/benchmark/onnx_policy_cpu_benchmark \
  --runtime openvino-ep \
  --device GPU \
  --provider-library /path/to/libonnxruntime_providers_openvino_plugin.so
```

Use the same command with `--list-devices` to enumerate the plugin's devices without loading the model. The EP selects exactly one enumerated device matching `--device`. Accelerator sessions (CUDA and non-CPU OpenVINO EP) disable ORT CPU-node fallback; models that cannot be fully assigned without CPU fallback fail rather than produce a misleading accelerator result. CPU EP mode retains its existing fallback behavior.

The compiled default model is `config/policy/karate/params/policy.onnx`; it does not follow later controller configuration changes automatically. Options:

```bash
./build/benchmark/onnx_policy_cpu_benchmark \
  --runtime openvino-ep \
  --provider-library /path/to/libonnxruntime_providers_openvino_plugin.so \
  --warmup 500 --iterations 20000 \
  --csv benchmark/latency.csv
```

Use `--model PATH` to benchmark a different exported policy. Reported metrics include initialization and warm-up time, min/mean/standard deviation and p50/p90/p95/p99/max inference latency, throughput, process CPU utilization, resident-memory delta, and the percentage of calls exceeding the deployment policy period of 20 ms. Initialization includes creating the selected backend and resolving its device; for direct OpenVINO it also includes ONNX import and device compilation. Model inspection and synthetic-input generation remain outside initialization timing.

Reports include `requested_device`, resolved `device`, `device_name`, and `execution_provider`. For CUDA and the OpenVINO EP, `device_name` is a device identifier rather than a hardware marketing name. CPU runs retain `benchmark=policy_cpu`; accelerator runs use `benchmark=policy_inference`. Existing latency keys and the CSV format (`iteration,latency_us`) are unchanged. CPU utilization and RSS are **host-process** metrics, not accelerator utilization or device memory; reports mark this with `resource_metrics_scope=host_process`. Precision remains at each backend's default. Thread counts also remain at each backend's default unless `--threads N` is given, so default runs are not thread-count-normalized comparisons.

Reports also record the loaded runtime versions: `onnxruntime_version` for ONNX Runtime and the OpenVINO EP, and `openvino_version` for direct OpenVINO and the OpenVINO EP. OpenVINO EP runs also report `openvino_ep_version`, `openvino_ep_library` (the plugin file ONNX Runtime loaded), and `openvino_ep_provider_options`.

Direct OpenVINO and OpenVINO EP runs report `inference_precision` (for example `f32` on CPU and `f16` on GPU and NPU) and where it comes from in `inference_precision_source`. It is queried after the measured calls, so it does not change the timing. ONNX Runtime runs do not report it.

- Direct OpenVINO (`compiled_model`): the precision the compiled model uses, or `unknown` if the device plugin cannot report it.
- OpenVINO EP (`openvino_device_default`): ONNX Runtime cannot query the model the EP compiled. OpenVINO EP 1.8 ignores the `precision` provider option and compiles with the `load_config` settings. This benchmark never sets `INFERENCE_PRECISION_HINT` in `load_config`, so the EP compiles with OpenVINO's default for the device, which the report queries from the same OpenVINO library. The `openvino-ep` inference test checks that this matches the precision direct OpenVINO compiles the model with on the same device.

## CPU threads

`--threads N` (CPU only) sets the CPU inference thread count for a thread-normalized comparison. Reports show it as `threads=N`, or `threads=default` without the option.

- ONNX Runtime sets its intra-op thread count. Because `OrtRunner` has no thread setting, this path uses a plain ONNX Runtime CPU session instead of `OrtRunner::act()`. By default ONNX Runtime pins one thread to each physical core. With an explicit thread count it pins none.
- Direct OpenVINO sets `INFERENCE_NUM_THREADS`.
- The OpenVINO EP sets both the ONNX Runtime intra-op thread count and `INFERENCE_NUM_THREADS` through the EP's `load_config` option, so its process may use more than `N` threads in total.

## Source layout and tests

- `main.cpp`: CLI orchestration and top-level error handling.
- `options.h/.cpp`: defaults, parsing, and runtime/provider/device validation.
- `model.h/.cpp`: model metadata, shapes, and seeded synthetic observations.
- `backends/backend.h/.cpp`: backend interface, selection, and device resolution.
- `backends/backend_onnx.cpp`, `backends/backend_openvino.cpp`, `backends/backend_openvino_ep.cpp`: runtime implementations and discovery.
- `backends/backend_ort.h/.cpp`: shared host-tensor/session inference for CUDA and the OpenVINO EP.
- `benchmark.h/.cpp`: execution timing, host resource sampling, latency statistics, and the finite-output checks and action difference used by `output_diff.cpp`.
- `report.h/.cpp`: console formatting and CSV export.
- `run_all.sh`: runs every runtime/device configuration and writes a summary table.
- `provenance.sh`: rebuilds the default binary and prints the git, plugin, and driver fields for `environment.txt`, used by both scripts.
- `output_diff.cpp`, `compare_outputs.sh`: compare each configuration's actions with ONNX Runtime CPU.
- `tests.cpp`: hardware-independent CLI, device resolution, statistics, input generation, reporting, and output comparison tests.

```bash
cmake --build build/benchmark -j
ctest --test-dir build/benchmark --output-on-failure
```

Tests are enabled by default; configure with `-DBUILD_TESTING=OFF` to omit them. Unit tests do not require accelerator hardware or load a policy. When the default model exists, CTest also registers synthetic CPU inference tests for ONNX Runtime and OpenVINO, checking zero warm-up, one measured iteration, repeatability, action extent, and that `--threads 1` gives the same output. If the OpenVINO EP plugin is installed at `thirdparty/onnxruntime-ep-openvino-<OVEP_VERSION>-install/Release/lib/` (`OVEP_VERSION` defaults to 1.8.1, as in the bootstrap script), or set with `-DOPENVINO_EP_LIBRARY=PATH`, CTest also runs the same checks for `openvino-ep` on CPU. If you build the plugin later, run the CMake configure step again so that CTest finds it. To run the same inference checks on an available OpenVINO accelerator:

```bash
./build/benchmark/policy_benchmark_tests openvino \
  config/policy/karate/params/policy.onnx GPU
./build/benchmark/policy_benchmark_tests openvino-ep \
  config/policy/karate/params/policy.onnx NPU /path/to/libonnxruntime_providers_openvino_plugin.so
```