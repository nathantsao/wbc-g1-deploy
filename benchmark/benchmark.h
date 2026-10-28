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

}  // namespace policy_benchmark
