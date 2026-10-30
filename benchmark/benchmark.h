#pragma once

#include "backends/backend.h"

namespace policy_benchmark {

struct LatencyStats {
  double min = 0.0;
  double mean = 0.0;
  double stddev = 0.0;
  double p50 = 0.0;
  double p90 = 0.0;
  double p95 = 0.0;
  double p99 = 0.0;
  double max = 0.0;
  size_t deadline_misses = 0;
};

struct BenchmarkResult {
  ModelInfo model;
  BackendInfo backend;
  PrecisionInfo precision;
  double initialization_ms = 0.0;
  double warmup_ms = 0.0;
  double wall_seconds = 0.0;
  double cpu_utilization = 0.0;
  long rss_before_kib = -1;
  long rss_after_kib = -1;
  double output_checksum = 0.0;
  std::vector<double> latency_us;
  LatencyStats latency;
};

LatencyStats summarize_latency(const std::vector<double>& latency_us);
BenchmarkResult run_benchmark(const Options& options);

// One action vector per seed.
using SeedOutputs = std::vector<std::vector<float>>;

struct OutputDiff {
  double max_action_diff = 0.0;
  double max_joint_target_diff = 0.0;  // 0 when no action scale is given
};

// Index of the first NaN or Inf in values, or values.size() if all are finite.
size_t first_non_finite(const std::vector<float>& values);

// Throws if the baseline or the action scale has a NaN or Inf, or if a non-empty scale does
// not match the baseline action size.
void check_reference(const SeedOutputs& baseline, const std::vector<float>& scale);

// Max absolute candidate-baseline difference over all seeds and joints. Throws on an invalid
// reference (see check_reference), a non-finite candidate output, or a seed or size mismatch.
OutputDiff compare_outputs(const SeedOutputs& baseline, const SeedOutputs& candidate,
                           const std::vector<float>& scale);

}  // namespace policy_benchmark
