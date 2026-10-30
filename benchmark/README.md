# Policy inference benchmark

This standalone C++ harness benchmarks the active ONNX policy using ONNX Runtime, direct OpenVINO Runtime, or the ONNX Runtime OpenVINO Execution Provider (EP) plugin. CPU is the default device. OpenVINO supports explicit CPU, Intel GPU, and Intel NPU selection; native ONNX Runtime also supports NVIDIA GPUs through its CUDA provider. All backends use the same deterministic synthetic input tensors, warm-up count, and metric collection. ONNX Runtime uses the same `ORT_ENABLE_EXTENDED` graph optimization setting as `OrtRunner`.

Contents:

- [Key findings](#key-findings)
- [Setup](#setup)
- [Single run](#single-run)
- [What is measured](#what-is-measured)
- [Report fields](#report-fields)
- [Run all configurations](#run-all-configurations)
- [Compare outputs](#compare-outputs)
- [Results: Intel Core Ultra X7 358H](#results-intel-core-ultra-x7-358h)
- [Source layout and tests](#source-layout-and-tests)

## Key findings

Measured on an Intel Core Ultra X7 358H. See [Results: Intel Core Ultra X7 358H](#results-intel-core-ultra-x7-358h) for the test machine, full tables, and all findings.

- All configs meet the 20 ms policy period. None of the 4.8 million measured calls took longer than 20 ms. The slowest took 3.2 ms, and p99 is at most 121 µs.
- CPU is fastest for these small models. GPU and NPU are about 2–5× slower (p50). NPU uses the least host CPU (about 0.17–0.19 core).
- With the same thread count, the faster CPU runtime depends on the model. At 2 or 4 threads, OpenVINO is about 22–25% faster on wbc and ONNX Runtime is about 12–21% faster on velocity. On karate they are within 10% of each other.
- ONNX Runtime uses 12–16 cores by default but is at most about 2.4 µs faster than with 4 threads. It also had 151 of the 156 calls over 2 ms. Consider limiting its thread count on the robot.
- GPU and NPU use FP16 by default. Their actions differ from ONNX Runtime CPU by up to 0.059 (0.026 rad joint target). Test them in closed loop before using them on the robot.

## Setup

### Requirements

Building requires the OpenVINO C++ development package with the ONNX frontend and desired device plugins, plus ONNX Runtime. Set `OpenVINO_DIR` to the package's CMake directory if it is not found automatically. The `openvino-ep` mode additionally requires the separately built or packaged `onnxruntime-ep-openvino` shared library. The bundled ONNX Runtime is 1.30.0; the plugin API requires ONNX Runtime 1.23.0 or newer. The current upstream OpenVINO EP plugin build requires ONNX Runtime 1.30.0+ and OpenVINO 2026.4+. Intel GPU/NPU modes also require the corresponding hardware drivers.

### Install dependencies on an Intel machine

On an Intel machine, `scripts/bootstrap_thirdparty_intel.sh` installs ONNX Runtime, OpenVINO, and the OpenVINO EP plugin under `thirdparty/`.

If `--runtime openvino --list-devices` shows only CPU, the GPU and NPU user-mode drivers are probably not installed. Then run the script with `--system-deps`, which needs sudo. `--system-deps` does the following:

- Installs build tools such as CMake and Ninja.
- Installs the newest GPU compute runtime and Level Zero from the `kobuk-team/intel-graphics` PPA.
- Installs NPU driver v1.38.0, unless an NPU driver is already installed.
- Adds you to the `render` group. Log out and back in afterward.

### Build

Build from the repository root:

```bash
cmake -S benchmark -B build/benchmark
cmake --build build/benchmark -j
```

The build type defaults to `Release`. Pass `-DCMAKE_BUILD_TYPE=...` to override it. The executable retains its original name for compatibility, including accelerator runs.

To build against a different ONNX Runtime package, pass its package root when configuring:

```bash
cmake -S benchmark -B build/benchmark \
  -DONNXRUNTIME_DIR=/path/to/onnxruntime
cmake --build build/benchmark -j
```

## Single run

Run from the repository root:

```bash
./build/benchmark/onnx_policy_cpu_benchmark --runtime onnx
./build/benchmark/onnx_policy_cpu_benchmark --runtime openvino
```

### Devices

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

#### OpenVINO EP plugin

Build or install the plugin following [onnxruntime-ep-openvino](https://github.com/onnxruntime/onnxruntime-ep-openvino), then pass the plugin shared-library path:

```bash
./build/benchmark/onnx_policy_cpu_benchmark \
  --runtime openvino-ep \
  --device GPU \
  --provider-library /path/to/libonnxruntime_providers_openvino_plugin.so
```

Use the same command with `--list-devices` to enumerate the plugin's devices without loading the model. The EP selects exactly one enumerated device matching `--device`. Accelerator sessions (CUDA and non-CPU OpenVINO EP) disable ORT CPU-node fallback; models that cannot be fully assigned without CPU fallback fail rather than produce a misleading accelerator result. CPU EP mode retains its existing fallback behavior.

#### NVIDIA GPU through CUDA

Native ONNX Runtime GPU execution requires a **CUDA-enabled** ONNX Runtime package plus compatible CUDA/cuDNN; the bundled CPU package does not provide CUDA:

```bash
./build/benchmark/onnx_policy_cpu_benchmark \
  --runtime onnx --provider cuda --device GPU --device-id 0
./build/benchmark/onnx_policy_cpu_benchmark \
  --runtime onnx --provider cuda --device GPU --list-devices
```

`--provider` is only valid for `onnx` (`cpu` or `cuda`, default `cpu`), and `--device-id` is only valid for CUDA. CUDA listing verifies that the provider is present, not that a particular GPU or driver is usable: ORT does not expose portable CUDA hardware enumeration through this path. Session creation validates the chosen index and dependencies. For Intel GPU/NPU execution through ONNX Runtime, use `openvino-ep` instead.

### CPU threads

`--threads N` (CPU only) sets the CPU inference thread count for a thread-normalized comparison. Reports show it as `threads=N`, or `threads=default` without the option.

- ONNX Runtime sets its intra-op thread count. Because `OrtRunner` has no thread setting, this path uses a plain ONNX Runtime CPU session instead of `OrtRunner::act()`. By default ONNX Runtime pins one thread to each physical core. With an explicit thread count it pins none.
- Direct OpenVINO sets `INFERENCE_NUM_THREADS`.
- The OpenVINO EP sets both the ONNX Runtime intra-op thread count and `INFERENCE_NUM_THREADS` through the EP's `load_config` option, so its process may use more than `N` threads in total.

### Model and run length

The compiled default model is `config/policy/karate/params/policy.onnx`; it does not follow later controller configuration changes automatically. Options:

```bash
./build/benchmark/onnx_policy_cpu_benchmark \
  --runtime openvino-ep \
  --provider-library /path/to/libonnxruntime_providers_openvino_plugin.so \
  --warmup 500 --iterations 20000 \
  --csv benchmark/latency.csv
```

Use `--model PATH` to benchmark a different exported policy.

## What is measured

Each measured call includes one input-map copy and the selected backend's tensor setup, inference, and output handling. Each backend keeps its default optimization settings. It also keeps its default thread pool unless `--threads N` is given (see [CPU threads](#cpu-threads)). So a default run is an out-of-box comparison, not a comparison with the same thread count.

The benchmark discovers model input names and static shapes, then supplies deterministic synthetic values. Inputs and the first output must be float32; the first output must be rank 2 with a static action dimension. Use single-action-row policies (output `[1, action_dimension]`) for comparable output handling: the existing `OrtRunner` copies one action row, whereas other backends copy the complete first output tensor. The ONNX CPU path still uses `OrtRunner::act()` unchanged. Each backend is measured synchronously end to end, including per-call tensor construction, inference, any required host/device transfers, output copy, and return value. Observation generation, reference-clip loading, robot I/O, and the FSM are not included. This isolates policy inference and its wrapper overhead; it is not a full controller or hardware-in-the-loop timing test.

A run fails with an error if any warm-up or measured output is NaN or infinite. The check runs after each call returns, so it is not part of the per-call latency samples. Like the output checksum, it is included in the wall time and throughput (about 10 ns per call).

## Report fields

Reported metrics include initialization and warm-up time, min/mean/standard deviation and p50/p90/p95/p99/max inference latency, throughput, process CPU utilization, resident-memory delta, and the percentage of calls exceeding the deployment policy period of 20 ms. Initialization includes creating the selected backend and resolving its device; for direct OpenVINO it also includes ONNX import and device compilation. Model inspection and synthetic-input generation remain outside initialization timing.

Reports include `requested_device`, resolved `device`, `device_name`, and `execution_provider`. For CUDA and the OpenVINO EP, `device_name` is a device identifier rather than a hardware marketing name. CPU runs retain `benchmark=policy_cpu`; accelerator runs use `benchmark=policy_inference`. Existing latency keys and the CSV format (`iteration,latency_us`) are unchanged. CPU utilization and RSS are **host-process** metrics, not accelerator utilization or device memory; reports mark this with `resource_metrics_scope=host_process`. Precision remains at each backend's default. Thread counts also remain at each backend's default unless `--threads N` is given, so default runs are not thread-count-normalized comparisons.

Reports also record the loaded runtime versions: `onnxruntime_version` for ONNX Runtime and the OpenVINO EP, and `openvino_version` for direct OpenVINO and the OpenVINO EP. OpenVINO EP runs also report `openvino_ep_version`, `openvino_ep_library` (the plugin file ONNX Runtime loaded), and `openvino_ep_provider_options`.

### Inference precision

Direct OpenVINO and OpenVINO EP runs report `inference_precision` (for example `f32` on CPU and `f16` on GPU and NPU) and where it comes from in `inference_precision_source`. It is queried after the measured calls, so it does not change the timing. ONNX Runtime runs do not report it.

- Direct OpenVINO (`compiled_model`): the precision the compiled model uses, or `unknown` if the device plugin cannot report it.
- OpenVINO EP (`openvino_device_default`): ONNX Runtime cannot query the model the EP compiled. OpenVINO EP 1.8 ignores the `precision` provider option and compiles with the `load_config` settings. This benchmark never sets `INFERENCE_PRECISION_HINT` in `load_config`, so the EP compiles with OpenVINO's default for the device, which the report queries from the same OpenVINO library. The `openvino-ep` inference test checks that this matches the precision direct OpenVINO compiles the model with on the same device.

## Run all configurations

To compare ONNX Runtime CPU against OpenVINO CPU/GPU/NPU (and the OpenVINO EP plugin on each device, when built) in one step:

```bash
benchmark/run_all.sh
```

### Runs

- The command above runs the karate policy, which is the default model. For another policy, add `--model config/policy/<name>/params/policy.onnx`, where `<name>` is `wbc` or `velocity`.
- Each of the `REPEATS` rounds (default 3) runs every configuration once so CPU heat and frequency changes are spread across configurations. Each round uses a new random order. With a fixed or rotated order, a configuration would run at the same position or after the same configuration in every round, for example `openvino:CPU` right after the 14-core `onnx:CPU` run. The median cannot remove an effect that is the same in every round. A random order turns such effects into noise that the median reduces, but does not remove them. `environment.txt` records the seed as `order_seed` and the order of each round as `round<N>_order`. Set `ORDER_SEED` to repeat the same orders.
- Extra arguments are passed to every run. The script rejects the options it sets for each configuration: `--runtime`, `--device`, `--warmup`, `--iterations`, `--csv`, `--provider-library`, and `--list-devices`.

### Output

- The script prints a Markdown table of per-configuration medians and writes `summary.csv`, the raw per-run reports (`.txt` for stdout, `.err` for stderr), per-call latency CSVs, `exit_codes.txt`, and `environment.txt` under `build/benchmark/results/<timestamp>/`. If that directory already exists, for example after another run in the same second, the script adds `-2`, `-3`, and so on to the name.
- `environment.txt` records the CPU governor and platform profile, the git commit and whether tracked files had uncommitted changes (`git_dirty`), the benchmark binary and whether the script rebuilt it (`bin_rebuilt`), the resolved OpenVINO EP plugin file, the GPU/NPU/Level Zero driver package versions, and the SHA-256 of each model the runs loaded.
- Only runs that exit with code 0 are summarized. `summary.csv` has one column per report field: the median for numeric fields (`policy_deadline_misses` is the miss count), the value for text fields that are equal in every run, and all values joined by ` | ` for text fields that differ. Its `runs` and `failed_runs` columns count the runs that were and were not used.
- A median hides the slowest call, so `summary.csv` also has `latency_max_us_all_runs` (the max latency over all successful runs), `policy_deadline_misses_total` (calls over 20 ms, summed over those runs), and `measured_iterations_total`. The console table shows these as `max_us_all` and `>20ms_all`.
- A configuration with no successful run is reported as `FAILED` without stopping the others. The script exits with an error only if every configuration fails.

### Settings

- Set `CONFIGS` to a space-separated list of `runtime:device` pairs to run a subset (for example `CONFIGS="onnx:CPU openvino:CPU openvino-ep:CPU"` together with `--threads`).
- Override `BIN`, `EP_LIB`, `OUT`, `WARMUP`, `ITERATIONS`, or `ORDER_SEED` through the environment. `OUT` must not already contain runs, so old results are never mixed into a new summary.
- Without `BIN`, the script first runs `cmake --build build/benchmark` for the benchmark target, so the binary matches the source tree that `git_commit` and `git_dirty` describe. Nothing is compiled if the build is up to date. If it compiles something, the CPU is still warm for the first runs, so build first for final results. A `BIN` that is set is used as is (`bin_rebuilt=false`) and may not match `git_commit`.
- `EP_LIB` defaults to a fixed path, `thirdparty/onnxruntime-ep-openvino-<OVEP_VERSION>-install/Release/lib/libonnxruntime_providers_openvino_plugin.so`, where `OVEP_VERSION` defaults to 1.8.1 as in the bootstrap script. So the same plugin is used even when several versions are installed. If that file does not exist, openvino-ep runs are skipped. An `EP_LIB` that does not exist is an error, and a relative `EP_LIB` is made absolute because ONNX Runtime would resolve it from its own lib directory. `compare_outputs.sh` uses the same rule.

## Compare outputs

To check how much accelerator outputs differ from ONNX Runtime CPU (for example because GPU and NPU default to FP16), compare the actions for seeded synthetic inputs:

```bash
benchmark/compare_outputs.sh config/policy/karate/params
```

- The argument is the params directory of the policy to check: `config/policy/<name>/params`, where `<name>` is `karate`, `wbc`, or `velocity`.
- `compare_outputs.sh` needs Python 3 with PyYAML to read `config.yaml`. Install it with `sudo apt install python3-yaml`, or with `pip install pyyaml` in a virtual environment.

### Output

- It prints, per configuration, the max absolute action difference over seeds 1 to `SEEDS` (default 200), the max joint target difference in radians, and the `inference_precision` the configuration ran with (see [Report fields](#report-fields)).
- The joint target difference is the action difference times the per-joint action scale from `config.yaml` (`action.scale`, or `actions.JointPositionAction.scale`).
- The same CSV is saved as `output_diff.csv`, with stderr in `output_diff.err` and the settings (params directory, seeds, configurations, action scale) in `environment.txt`, under `build/benchmark/output_diff/<timestamp>/`. As with `run_all.sh`, `-2`, `-3`, and so on are added to the name if that directory already exists. `environment.txt` also has the model SHA-256 and the same git, binary, plugin, and driver fields as `run_all.sh`. Without `BIN`, the script also rebuilds `policy_output_diff` first. Set `OUT` to choose the directory; it must not already hold a comparison.
- A configuration with a NaN or infinite output is reported as `FAILED` and counts as a failed configuration. A NaN or infinite ONNX Runtime CPU baseline output or action scale stops the comparison with an error (exit code 1).

### Settings

- `CONFIGS` defaults to `openvino:CPU openvino:GPU openvino:NPU`. Set it to check other configurations, for example `CONFIGS="openvino-ep:GPU openvino-ep:NPU"`.
- `--seeds` must be a plain decimal integer from 1 to 1000000.
- The script wraps `build/benchmark/policy_output_diff`, which can also be run directly with `--action-scale`.

## Results: Intel Core Ultra X7 358H

ONNX Runtime on CPU compared with OpenVINO on CPU, iGPU, and NPU. OpenVINO is tested directly and through the ONNX Runtime OpenVINO EP plugin. Measured on 2026-10-30 at commit `4dc5895` with no uncommitted changes (`git_dirty=false`). The per-run reports, `summary.csv`, and `environment.txt` are not committed. The [Reproduce](#reproduce) commands write them again under `build/benchmark/`.

### Test machine

| Item | Value |
|---|---|
| CPU | Intel Core Ultra X7 358H (16 cores: 4 P-cores, 8 E-cores, 4 LP E-cores) |
| iGPU / NPU | Intel Arc B390 / Intel AI Boost (NPU5) |
| OS / kernel | Ubuntu 26.04.1, 7.0.0-38-generic |
| CPU power | `performance` governor, platform profile `balanced` |
| ONNX Runtime | 1.30.0 (prebuilt CPU package) |
| OpenVINO | 2026.4.1 (prebuilt archive) |
| OpenVINO EP plugin | 1.8.1 |
| Drivers | GPU 26.35.39758.17, NPU 1.38.0, Level Zero 1.32.0 |
| Runs | 5 rounds, each in a new random order, 500 warm-up + 20000 measured calls per run, median of rounds |

Inputs are synthetic float32 values. Latency is one call: input copy, inference, and output copy. It does not include observation building or robot I/O. The benchmark runs the calls with no wait in between. On the robot, the controller calls the policy every 20 ms instead. So latency and CPU usage on the robot may differ.

### Default settings

Each runtime uses its default threads and precision. p50, p99, init, and CPU are the median of the 5 runs. Max is the slowest single call over all 5 runs (100000 calls). CPU is host process CPU usage, where 100% is one core.

| Model | Runtime | Device | p50 (µs) | p99 (µs) | Max (µs) | Init (ms) | CPU (%) |
|---|---|---|---|---|---|---|---|
| karate | ONNX Runtime | CPU | **20.7** | 22.2 | 348.0 | 3.0 | 1395 |
| | OpenVINO | CPU | 22.7 | 24.6 | 1230.3 | 44.4 | 400 |
| | OpenVINO | GPU | 46.7 | 48.7 | 1214.8 | 83.4 | 100 |
| | OpenVINO | NPU | 72.9 | 120.5 | 462.2 | 137.9 | **16** |
| | OpenVINO EP | CPU | 23.7 | 58.1 | 326.5 | 65.5 | 864 |
| | OpenVINO EP | GPU | 47.4 | 51.8 | 1112.3 | 156.4 | 330 |
| | OpenVINO EP | NPU | 71.3 | 114.5 | 1232.7 | 150.3 | 137 |
| wbc | ONNX Runtime | CPU | 27.1 | 29.6 | 3053.3 | 14.0 | 1254 |
| | OpenVINO | CPU | **22.9** | 24.6 | 126.6 | 39.4 | 400 |
| | OpenVINO | GPU | 46.7 | 55.6 | 1232.3 | 75.3 | 100 |
| | OpenVINO | NPU | 72.9 | 119.0 | 1245.2 | 138.3 | **17** |
| | OpenVINO EP | CPU | 23.5 | 57.7 | 360.0 | 87.9 | 849 |
| | OpenVINO EP | GPU | 47.6 | 51.4 | 1219.7 | 140.8 | 355 |
| | OpenVINO EP | NPU | 71.6 | 112.8 | 2050.7 | 163.7 | 119 |
| velocity | ONNX Runtime | CPU | **10.7** | 16.7 | 3051.8 | 1.6 | 1588 |
| | OpenVINO | CPU | 13.7 | 14.9 | 1184.1 | 43.6 | 400 |
| | OpenVINO | GPU | 34.3 | 36.7 | 1051.8 | 81.0 | 100 |
| | OpenVINO | NPU | 56.6 | 98.0 | 318.3 | 133.0 | **19** |
| | OpenVINO EP | CPU | 18.0 | 19.1 | 118.9 | 60.5 | 1326 |
| | OpenVINO EP | GPU | 35.0 | 41.6 | 1210.1 | 112.4 | 506 |
| | OpenVINO EP | NPU | 55.7 | 99.3 | 3184.4 | 139.1 | 199 |

### Same CPU thread count (`--threads N`)

p50 latency, median of the 5 runs. The OpenVINO EP used more cores than the thread count: 244–290% CPU at `--threads 2` and 566–687% at `--threads 4`. The likely reason is that ONNX Runtime and OpenVINO each run their own thread pool. `--threads` does not choose between P-cores and E-cores.

| Model | Threads | ONNX Runtime p50 (µs) | OpenVINO p50 (µs) | OpenVINO EP p50 (µs) |
|---|---|---|---|---|
| karate | 1 | **46.5** | 51.3 | 54.3 |
| | 2 | 29.4 | **27.9** | 29.4 |
| | 4 | 23.1 | **22.3** | 23.3 |
| wbc | 1 | 52.8 | **51.3** | 54.1 |
| | 2 | 35.8 | **28.0** | 29.3 |
| | 4 | 29.5 | **22.2** | 23.3 |
| velocity | 1 | **13.2** | 14.6 | 15.1 |
| | 2 | **12.0** | 13.7 | 15.3 |
| | 4 | **10.8** | 13.7 | 14.4 |

### Output difference from ONNX Runtime CPU

Max absolute difference of OpenVINO on each device over 200 inputs (seeds 1–200), compared with ONNX Runtime CPU. Measured with `benchmark/compare_outputs.sh`.

| Model | CPU action (unitless) | GPU action (unitless) | NPU action (unitless) | Joint target (rad) |
|---|---|---|---|---|
| karate | 0.000005 | 0.007 | 0.006 | 0.004 |
| wbc | 0.00004 | 0.059 | 0.051 | 0.026 |
| velocity | 0.000005 | 0.009 | 0.009 | 0.004 |

- The action columns are the raw policy output difference, which has no unit.
- Joint target is the action difference multiplied by each joint's action scale in `config.yaml` (`action.scale`, or `actions.JointPositionAction.scale` for velocity). This is how much the joint position target would change, in radians. The value is the larger of GPU and NPU.
- OpenVINO EP gives the same outputs as direct OpenVINO on each device.

GPU and NPU use FP16 by default, so their outputs differ a little. Not yet checked with real observations or in closed loop.

### Findings

- All configs meet the 20 ms policy period. None of the 4.8 million measured calls took longer than 20 ms (`policy_deadline_misses_total` is 0 for every config). The slowest single call was 3.2 ms (OpenVINO EP on NPU, velocity), and 4 configs had a call over 2 ms. The p99 is at most 121 µs for every config.
- Calls over 2 ms were rare (156 of 4.8 million), and 151 of them came from ONNX Runtime with default threads. In 4 of its 15 default runs, up to 50 calls took about 3 ms each. With `--threads N`, ONNX Runtime had no call over 2 ms in its 45 runs. The cause is not confirmed. A likely reason is that by default it pins a thread to every core, so one core that is busy with another process delays the whole call. Single slow calls of about 1.1–1.2 ms occur with every runtime and device, so they likely come from the system rather than the runtime.
- CPU is fastest for these small models. GPU and NPU are about 2–5× slower than the fastest CPU runtime (p50). The cause (for example transfer overhead) was not measured and is unverified.
- With the same thread count, the faster CPU runtime depends on the model. "x% faster" means 1 - fast/slow using p50 latency. For example 22% faster means the faster runtime takes 78% of the time.
  - wbc: OpenVINO is about 22–25% faster at 2 or 4 threads, and about 3% faster at 1 thread.
  - velocity: ONNX Runtime is about 12–21% faster at 2 or 4 threads, and about 10% faster at 1 thread.
  - karate: ONNX Runtime is about 9% faster at 1 thread. OpenVINO is 3–5% faster at 2 or 4 threads.

- karate and wbc have the same network structure. Only the weights differ. Still, ONNX Runtime is 6.3–6.5 µs slower on wbc at every thread setting, while OpenVINO differs by at most 0.2 µs. The cause is not confirmed yet.
- ONNX Runtime uses about 12–16 cores by default. Compared with 4 threads, this saves only about 2.4 µs for karate and wbc, and nothing for velocity (10.7 µs vs 10.8 µs). This gap is not only from the thread count. By default ONNX Runtime pins one thread to each physical core. With an explicit thread count it pins none. The two runs also use different code (`OrtRunner` and a plain session with slightly different per-call handling). The effect of each difference was not measured. On the robot, consider limiting the thread count, which also avoided the 3 ms bursts above. This needs a thread setting in the controller's `OrtRunner`, which would also remove the pinning.
- NPU uses the least host CPU (about 0.17–0.19 core), at about 57–73 µs per call. It was not tested under CPU load so it is unverified whether NPU is better when the CPU is busy.
- Some results change between runs more than the median shows, and the cause is not known. For velocity on direct OpenVINO NPU, p50 was 56–57 µs in 4 runs and 67.6 µs in 1. An earlier measurement on 2026-10-28 had a median of 67.2 µs. For velocity on ONNX Runtime CPU, p99 was about 12 µs in 2 runs and about 17 µs in 3.
- Test GPU and NPU in closed loop before using them on the robot, because of the FP16 output difference.
- OpenVINO EP is close to direct OpenVINO on GPU and NPU, but starts slower and uses more host CPU.
- OpenVINO EP on CPU has a high p99 with default threads for karate and wbc (about 58 µs, 2.5× its p50). This is likely because the ONNX Runtime and OpenVINO thread pools compete for cores. With `--threads 4`, its p99 drops to about 25 µs, and with `--threads 2` to about 32 µs.
- With the `powersave` governor, results are less stable. Use `performance` for repeatable numbers.

### Reproduce

Run from the repository root on the benchmark machine while it is idle. Commit or stash changes first so `environment.txt` records `git_dirty=false`. Each `run_all.sh` call writes its results to a new `build/benchmark/results/<timestamp>/`, and each `compare_outputs.sh` call to a new `build/benchmark/output_diff/<timestamp>/`.

The first two lines save the current CPU governor and switch to `performance`. The last line switches back to the saved governor, for example `powersave`. Before the runs, `cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | sort | uniq -c` should show `performance` for every CPU.

```bash
old_governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
sudo cpupower frequency-set -g performance
for m in karate wbc velocity; do
  REPEATS=5 benchmark/run_all.sh --model config/policy/$m/params/policy.onnx
  for t in 1 2 4; do
    REPEATS=5 CONFIGS="onnx:CPU openvino:CPU openvino-ep:CPU" \
      benchmark/run_all.sh --model config/policy/$m/params/policy.onnx --threads $t
  done
  CONFIGS="openvino:CPU openvino:GPU openvino:NPU openvino-ep:CPU openvino-ep:GPU openvino-ep:NPU" \
    benchmark/compare_outputs.sh config/policy/$m/params
done
sudo cpupower frequency-set -g "${old_governor}"
```

## Source layout and tests

### Source layout

- `main.cpp`: CLI orchestration and top-level error handling.
- `options.h/.cpp`: defaults, parsing, and runtime/provider/device validation.
- `model.h/.cpp`: model metadata, shapes, and seeded synthetic observations.
- `backends/backend.h/.cpp`: backend interface, selection, and device resolution.
- `backends/backend_onnx.cpp`, `backends/backend_openvino.cpp`, `backends/backend_openvino_ep.cpp`: runtime implementations and discovery.
- `backends/backend_ort.h/.cpp`: shared host-tensor/session inference for CUDA and the OpenVINO EP.
- `benchmark.h/.cpp`: execution timing, host resource sampling, latency statistics, and the finite-output checks and action difference used by `output_diff.cpp`.
- `report.h/.cpp`: console formatting and CSV export.
- `run_all.sh`: runs every runtime/device configuration and writes a summary table.
- `provenance.sh`: rebuilds the default binary, creates the default output directory, and prints the git, plugin, and driver fields for `environment.txt`, used by both scripts.
- `output_diff.cpp`, `compare_outputs.sh`: compare each configuration's actions with ONNX Runtime CPU.
- `tests.cpp`: hardware-independent CLI, device resolution, statistics, input generation, reporting, and output comparison tests.

### Tests

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
