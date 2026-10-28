#include "benchmark.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <sys/resource.h>

namespace policy_benchmark {
namespace {

using Clock = std::chrono::steady_clock;

double timeval_seconds(const timeval& value) {
  return static_cast<double>(value.tv_sec) + static_cast<double>(value.tv_usec) / 1e6;
}

double process_cpu_seconds() {
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) != 0) {
    throw std::runtime_error(std::string("getrusage failed: ") + std::strerror(errno));
  }
  return timeval_seconds(usage.ru_utime) + timeval_seconds(usage.ru_stime);
}

long current_rss_kib() {
  std::ifstream status("/proc/self/status");
  std::string key;
  while (status >> key) {
    if (key == "VmRSS:") {
      long rss;
      std::string unit;
      if (!(status >> rss >> unit) || rss < 0 || unit != "kB") {
        throw std::runtime_error("Invalid VmRSS in /proc/self/status");
      }
      return rss;
    }
    std::string rest;
    std::getline(status, rest);
  }
  throw std::runtime_error("Cannot read VmRSS from /proc/self/status");
}

double percentile(const std::vector<double>& sorted, double fraction) {
  const size_t index = static_cast<size_t>(std::ceil(fraction * sorted.size())) - 1;
  return sorted[std::min(index, sorted.size() - 1)];
}

}  // namespace

LatencyStats summarize_latency(const std::vector<double>& latency_us) {
  if (latency_us.empty()) throw std::runtime_error("Cannot summarize empty latency samples");
  LatencyStats stats;
  stats.mean = std::accumulate(latency_us.begin(), latency_us.end(), 0.0) /
               static_cast<double>(latency_us.size());
  double squared_error_sum = 0.0;
  for (const double latency : latency_us) {
    const double delta = latency - stats.mean;
    squared_error_sum += delta * delta;
    if (latency > 20000.0) ++stats.deadline_misses;
  }
  stats.stddev = std::sqrt(squared_error_sum / static_cast<double>(latency_us.size()));
  auto sorted = latency_us;
  std::sort(sorted.begin(), sorted.end());
  stats.min = sorted.front();
  stats.max = sorted.back();
  stats.p50 = percentile(sorted, 0.50);
  stats.p90 = percentile(sorted, 0.90);
  stats.p95 = percentile(sorted, 0.95);
  stats.p99 = percentile(sorted, 0.99);
  return stats;
}

BenchmarkResult run_benchmark(const Options& options) {
  if (options.measured_iterations == 0) {
    throw std::runtime_error("--iterations must be greater than zero");
  }
  BenchmarkResult result;
  result.model = inspect_model(options.model_path);
  const auto observations = make_observations(result.model, options.seed);

  const auto initialization_start = Clock::now();
  auto runner = make_backend(options, result.model);
  const auto initialization_end = Clock::now();
  result.initialization_ms =
      std::chrono::duration<double, std::milli>(initialization_end - initialization_start).count();
  result.backend = runner->info;

  const auto warmup_start = Clock::now();
  std::vector<float> action;
  for (size_t i = 0; i < options.warmup_iterations; ++i) {
    action = runner->infer(observations);
  }
  result.warmup_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - warmup_start).count();

  result.rss_before_kib = current_rss_kib();
  const double cpu_before = process_cpu_seconds();
  result.latency_us.reserve(options.measured_iterations);
  const auto benchmark_start = Clock::now();
  for (size_t i = 0; i < options.measured_iterations; ++i) {
    const auto call_start = Clock::now();
    action = runner->infer(observations);
    const auto call_end = Clock::now();
    result.latency_us.push_back(
        std::chrono::duration<double, std::micro>(call_end - call_start).count());
    if (!action.empty()) result.output_checksum += action[i % action.size()];
  }
  const auto benchmark_end = Clock::now();
  const double cpu_after = process_cpu_seconds();
  result.rss_after_kib = current_rss_kib();
  result.wall_seconds = std::chrono::duration<double>(benchmark_end - benchmark_start).count();
  result.cpu_utilization = result.wall_seconds > 0.0
      ? (cpu_after - cpu_before) / result.wall_seconds * 100.0 : 0.0;
  result.latency = summarize_latency(result.latency_us);
  return result;
}

}  // namespace policy_benchmark
