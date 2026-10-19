# CPU policy benchmark

This standalone C++ harness benchmarks the active ONNX policy using either the repository's `isaaclab::OrtRunner::act()` implementation or OpenVINO Runtime compiled for `CPU`. Both backends use the same deterministic synthetic input tensors, warm-up count, and metric collection. ONNX Runtime uses the same `ORT_ENABLE_EXTENDED` graph optimization setting as `OrtRunner` and its default CPU provider.

Building both backends requires the OpenVINO C++ development package with the ONNX frontend and CPU plugin, plus the repository's bundled ONNX Runtime. Each measured call includes one input-map copy and the selected backend's tensor setup, inference, and output handling. Backend thread pools and optimizations remain at their respective defaults, so this is an out-of-box CPU comparison, not a thread-count-normalized comparison.

The benchmark discovers model input names and static shapes, then supplies deterministic synthetic values. It measures `OrtRunner::act()` end to end, including its per-call tensor construction, ONNX Runtime `Session::Run`, output copy, and return value. Observation generation, reference-clip loading, robot I/O, and the FSM are not included. This isolates policy inference and its wrapper overhead; it is not a full controller or hardware-in-the-loop timing test.

Build and run from the repository root:

```bash
cmake -S benchmark -B build/benchmark
cmake --build build/benchmark -j
./build/benchmark/onnx_policy_cpu_benchmark --runtime onnx
./build/benchmark/onnx_policy_cpu_benchmark --runtime openvino
```

The default model is `config/policy/karate/params/policy.onnx`, matching the current root `config/config.yaml`. Options:

```bash
./build/benchmark/onnx_policy_cpu_benchmark \
  --runtime openvino --warmup 500 --iterations 20000 \
  --csv benchmark/latency.csv
```

Use `--model PATH` to benchmark a different exported policy. Reported metrics include initialization and warm-up time, min/mean/standard deviation and p50/p90/p95/p99/max inference latency, throughput, process CPU utilization, resident-memory delta, and the percentage of calls exceeding the deployment policy period of 20 ms. Initialization includes creating the selected runtime session; for OpenVINO it also includes ONNX import and CPU compilation. Runtime thread settings are left at each backend's default, so the comparison reflects their out-of-box CPU behavior rather than equalized thread pools.