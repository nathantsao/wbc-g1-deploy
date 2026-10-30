# Policy inference benchmark: Intel Core Ultra X7 358H

ONNX Runtime on CPU compared with OpenVINO on CPU, iGPU, and NPU. OpenVINO is tested directly and through the ONNX Runtime OpenVINO EP plugin. Each subfolder has the `summary.csv` and `environment.txt` written by `benchmark/run_all.sh`.

## Setup

| Item | Value |
|---|---|
| CPU | Intel Core Ultra X7 358H (16 cores: 4 P-cores, 8 E-cores, 4 LP E-cores) |
| iGPU / NPU | Intel Arc B390 / Intel AI Boost (NPU5) |
| OS / kernel | Ubuntu 26.04, 7.0.0-38-generic |
| CPU power | `performance` governor, platform profile `balanced` |
| ONNX Runtime | 1.30.0 (prebuilt CPU package) |
| OpenVINO | 2026.4.1 (prebuilt archive) |
| OpenVINO EP plugin | 1.8.1 |
| Drivers | GPU 26.35.39758.17, NPU 1.38.0, Level Zero 1.32.0 |
| Runs | 5 rounds, 500 warm-up + 20000 measured calls per run, median of rounds |

Inputs are synthetic float32 values. Latency is one call: input copy, inference, and output copy. It does not include observation building or robot I/O. The benchmark runs the calls with no wait in between. On the robot, the controller calls the policy every 20 ms instead. So latency and CPU usage on the robot may differ.

## Default settings

Each runtime uses its default threads and precision. Latency is in µs. `cpu %` is host CPU usage, where 100% is one core.

| Model | Runtime | Device | p50 | p99 | init ms | cpu % |
|---|---|---|---|---|---|---|
| karate | ONNX Runtime | CPU | **20.6** | 22.5 | 2.8 | 1382 |
| | OpenVINO | CPU | 22.6 | 24.3 | 41.8 | 400 |
| | OpenVINO | GPU | 46.7 | 49.1 | 82.6 | 100 |
| | OpenVINO | NPU | 71.6 | 120.7 | 143.3 | **16** |
| | OpenVINO EP | CPU | 23.6 | 58.1 | 63.7 | 883 |
| | OpenVINO EP | GPU | 47.4 | 52.4 | 139.1 | 356 |
| | OpenVINO EP | NPU | 71.4 | 117.5 | 139.5 | 147 |
| wbc | ONNX Runtime | CPU | 27.1 | 29.3 | 8.7 | 1263 |
| | OpenVINO | CPU | **22.6** | 24.5 | 49.5 | 400 |
| | OpenVINO | GPU | 46.5 | 48.5 | 83.4 | 100 |
| | OpenVINO | NPU | 72.2 | 120.0 | 148.3 | **16** |
| | OpenVINO EP | CPU | 23.7 | 58.2 | 74.3 | 845 |
| | OpenVINO EP | GPU | 47.5 | 51.8 | 131.7 | 335 |
| | OpenVINO EP | NPU | 72.5 | 113.7 | 160.5 | 131 |
| velocity | ONNX Runtime | CPU | **10.7** | 12.0 | 1.2 | 1594 |
| | OpenVINO | CPU | 13.8 | 15.0 | 46.7 | 400 |
| | OpenVINO | GPU | 34.2 | 39.3 | 79.8 | 100 |
| | OpenVINO | NPU | 67.2 | 114.6 | 133.2 | **17** |
| | OpenVINO EP | CPU | 18.0 | 19.2 | 54.2 | 1351 |
| | OpenVINO EP | GPU | 35.1 | 40.9 | 119.3 | 504 |
| | OpenVINO EP | NPU | 55.9 | 101.2 | 138.6 | 198 |

## Same CPU thread count (`--threads N`)

p50 latency in µs. Other metrics are in `<model>-threads-N/summary.csv`. The OpenVINO EP used more cores than the thread count: 244–289% CPU at `--threads 2` and 568–685% at `--threads 4`. The likely reason is that ONNX Runtime and OpenVINO each run their own thread pool. `--threads` does not choose between P-cores and E-cores.

| Model | Threads | ONNX Runtime | OpenVINO | OpenVINO EP |
|---|---|---|---|---|
| karate | 1 | **47.4** | 52.0 | 53.6 |
| | 2 | 29.4 | **28.0** | 29.4 |
| | 4 | 23.1 | **22.4** | 23.2 |
| wbc | 1 | 55.5 | **54.9** | 56.7 |
| | 2 | 35.7 | **27.9** | 29.5 |
| | 4 | 29.6 | **22.4** | 23.3 |
| velocity | 1 | **13.2** | 14.5 | 15.1 |
| | 2 | **12.1** | 14.3 | 15.4 |
| | 4 | **10.6** | 13.6 | 14.4 |

## Output difference from ONNX Runtime CPU

Max difference over 200 inputs (seeds 1–200), compared with ONNX Runtime CPU. Measured with `benchmark/compare_outputs.sh`.

| Model | OpenVINO CPU action | GPU action | NPU action | Joint target (rad) |
|---|---|---|---|---|
| karate | 0.000005 | 0.007 | 0.006 | 0.004 |
| wbc | 0.00004 | 0.059 | 0.051 | 0.026 |
| velocity | 0.000005 | 0.009 | 0.009 | 0.004 |

- The action columns are the raw policy output difference.
- Joint target is the action difference multiplied by each joint's action scale in `config.yaml` (`action.scale`, or `actions.JointPositionAction.scale` for velocity). This is how much the joint position target would change, in radians. The value is the larger of GPU and NPU.
- OpenVINO EP gives the same outputs as direct OpenVINO on each device.

GPU and NPU use FP16 by default, so their outputs differ a little. Not yet checked with real observations or in closed loop.

## Findings

- All configs meet the 20 ms policy period. None of the 4.8 million calls took longer than 20 ms. The slowest single call was 6.5 ms. These come from the per-call latency CSVs, which are not committed; `summary.csv` only keeps the median of each run's max.
- CPU is fastest for these small models. GPU and NPU are 2–6× slower. The cause (for example transfer overhead) was not measured and is unverified.
- With the same thread count, the faster CPU runtime depends on the model. "x% faster" means 1 - fast/slow using p50 latency. For example 22% faster means the faster runtime takes 78% of the time.
  - wbc: OpenVINO is about 22–24% faster at 2 or 4 threads, and about the same at 1 thread.
  - velocity: ONNX Runtime is about 15–22% faster at 2 or 4 threads, and about 9% faster at 1 thread.
  - karate: ONNX Runtime is about 9% faster at 1 thread. OpenVINO is 3–5% faster at 2 or 4 threads.
- karate and wbc have the same network structure. Only the weights differ. Still, ONNX Runtime is 6.3–8.1 µs slower on wbc at every thread setting, while OpenVINO differs by at most 2.9 µs. The cause is not confirmed yet.
- ONNX Runtime uses about 13–16 cores by default. Compared with 4 threads, this saves only about 2.5 µs for karate and wbc, and nothing for velocity (10.7 µs vs 10.6 µs). This gap is not only from the thread count. By default ONNX Runtime pins one thread to each physical core. With an explicit thread count it pins none. The two runs also use different code (`OrtRunner` and a plain session with slightly different per-call handling). The effect of each difference was not measured. On the robot, consider limiting the thread count. This needs a thread setting in the controller's `OrtRunner`, which would also remove the pinning.
- NPU uses the least host CPU (about 0.16 core), at about 70 µs per call. It was not tested under CPU load so it is unverified whether NPU is better when the CPU is busy.
- Test GPU and NPU in closed loop before using them on the robot, because of the FP16 output difference.
- OpenVINO EP is close to direct OpenVINO on GPU and NPU, but starts slower and uses more host CPU.
- OpenVINO EP on CPU has a high p99 with default threads for karate and wbc (about 58 µs, 2.5× its p50). This is likely because the ONNX Runtime and OpenVINO thread pools compete for cores. With `--threads 4`, its p99 drops to about 25 µs, and with `--threads 2` to about 32 µs.
- With the `powersave` governor, results are less stable. Use `performance` for repeatable numbers.

## Reproduce

```bash
sudo cpupower frequency-set -g performance
for m in karate wbc velocity; do
  REPEATS=5 benchmark/run_all.sh --model config/policy/$m/params/policy.onnx
  for t in 1 2 4; do
    REPEATS=5 CONFIGS="onnx:CPU openvino:CPU openvino-ep:CPU" \
      benchmark/run_all.sh --model config/policy/$m/params/policy.onnx --threads $t
  done
  benchmark/compare_outputs.sh config/policy/$m/params
done
```
