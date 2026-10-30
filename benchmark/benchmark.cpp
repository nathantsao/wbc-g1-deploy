#include "benchmark.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
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

void check_finite(const std::vector<float>& values, const std::string& name) {
  const size_t index = first_non_finite(values);
  if (index != values.size()) {
    throw std::runtime_error(name + " is non-finite (" + std::to_string(values[index]) +
                             ") at index " + std::to_string(index));
  }
}

void check_finite(const SeedOutputs& outputs, const std::string& name) {
  for (size_t s = 0; s < outputs.size(); ++s) {
    check_finite(outputs[s], name + " for seed " + std::to_string(s + 1));
  }
}

// Called after each inference returns, so the check is not part of the per-call latency.
void check_output(const std::vector<float>& action, const char* phase, size_t iteration) {
  const size_t index = first_non_finite(action);
  if (index != action.size()) {
    throw std::runtime_error("Non-finite output (" + std::to_string(action[index]) + ") in " +
                             phase + " iteration " + std::to_string(iteration) + ", index " +
                             std::to_string(index));
  }
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
    check_output(action, "warm-up", i);
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
    check_output(action, "measured", i);
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

size_t first_non_finite(const std::vector<float>& values) {
  for (size_t i = 0; i < values.size(); ++i) {
    if (!std::isfinite(values[i])) return i;
  }
  return values.size();
}

void check_reference(const SeedOutputs& baseline, const std::vector<float>& scale) {
  if (baseline.empty()) throw std::runtime_error("Baseline has no outputs");
  check_finite(baseline, "Baseline output");
  check_finite(scale, "--action-scale");
  if (!scale.empty() && scale.size() != baseline.front().size()) {
    throw std::runtime_error("--action-scale has " + std::to_string(scale.size()) +
                             " values but the policy outputs " +
                             std::to_string(baseline.front().size()));
  }
}

OutputDiff compare_outputs(const SeedOutputs& baseline, const SeedOutputs& candidate,
                           const std::vector<float>& scale) {
  check_reference(baseline, scale);
  if (candidate.size() != baseline.size()) {
    throw std::runtime_error("Candidate has " + std::to_string(candidate.size()) +
                             " seeds but the baseline has " + std::to_string(baseline.size()));
  }
  // std::max ignores NaN, so non-finite outputs must be rejected before taking the max.
  check_finite(candidate, "Candidate output");
  OutputDiff diff;
  for (size_t s = 0; s < baseline.size(); ++s) {
    if (candidate[s].size() != baseline[s].size()) {
      throw std::runtime_error("Output size differs for seed " + std::to_string(s + 1));
    }
    for (size_t j = 0; j < baseline[s].size(); ++j) {
      const double action_diff =
          std::abs(static_cast<double>(candidate[s][j]) - static_cast<double>(baseline[s][j]));
      diff.max_action_diff = std::max(diff.max_action_diff, action_diff);
      if (!scale.empty()) {
        diff.max_joint_target_diff =
            std::max(diff.max_joint_target_diff, action_diff * scale[j]);
      }
    }
  }
  return diff;
}

}  // namespace policy_benchmark
