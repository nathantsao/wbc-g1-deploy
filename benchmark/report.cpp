#include "report.h"

#include <fstream>
#include <iomanip>
#include <ostream>
#include <stdexcept>
#include <string>
#include <thread>

namespace policy_benchmark {

void print_report(const Options& options, const BenchmarkResult& result, std::ostream& out) {
  const auto& latency = result.latency;
  out << std::fixed << std::setprecision(3)
      << "benchmark=" << (options.device == "CPU" ? "policy_cpu" : "policy_inference") << '\n'
      << "runtime=" << options.runtime << '\n'
      << "model=" << options.model_path << '\n'
      << "requested_device=" << options.device << '\n'
      << "device=" << result.backend.device << '\n'
      << "device_name=" << result.backend.device_name << '\n'
      << "execution_provider=" << result.backend.execution_provider << '\n';
  for (const auto& [key, value] : result.backend.details) out << key << '=' << value << '\n';
  if (!result.precision.precision.empty()) {
    out << "inference_precision=" << result.precision.precision << '\n'
        << "inference_precision_source=" << result.precision.source << '\n';
  }
  out << "threads=" << (options.threads > 0 ? std::to_string(options.threads) : "default") << '\n';
  if (options.runtime == "onnx" || options.runtime == "openvino-ep") {
    out << "ort_graph_optimization=ORT_ENABLE_EXTENDED\n";
  }
  if (options.runtime == "onnx" && options.provider == "cuda") {
    out << "cuda_device_id=" << options.device_id << '\n';
  }
  if (options.runtime == "openvino-ep") {
    out << "openvino_ep_device=" << result.backend.device << '\n';
  }
  if (options.device != "CPU" && options.runtime != "openvino") {
    out << "ort_cpu_ep_fallback=disabled\n";
  }
  out << "resource_metrics_scope=host_process\n"
      << "hardware_threads=" << std::thread::hardware_concurrency() << '\n'
      << "input_count=" << result.model.inputs.size() << '\n';
  for (const auto& input : result.model.inputs) {
    out << "input." << input.name << ".shape=" << shape_string(input.shape)
        << " elements=" << input.element_count << '\n';
  }
  out << "output.shape=" << shape_string(result.model.output_shape) << '\n'
      << "input_data=deterministic synthetic uniform[-1,1], seed=" << options.seed << '\n'
      << "warmup_iterations=" << options.warmup_iterations << '\n'
      << "measured_iterations=" << options.measured_iterations << '\n'
      << "initialization_ms=" << result.initialization_ms << '\n'
      << "warmup_total_ms=" << result.warmup_ms << '\n'
      << "latency_min_us=" << latency.min << '\n'
      << "latency_mean_us=" << latency.mean << '\n'
      << "latency_stddev_us=" << latency.stddev << '\n'
      << "latency_p50_us=" << latency.p50 << '\n'
      << "latency_p90_us=" << latency.p90 << '\n'
      << "latency_p95_us=" << latency.p95 << '\n'
      << "latency_p99_us=" << latency.p99 << '\n'
      << "latency_max_us=" << latency.max << '\n'
      << "throughput_inferences_per_sec="
      << static_cast<double>(options.measured_iterations) / result.wall_seconds << '\n'
      << "process_cpu_utilization_pct=" << result.cpu_utilization << '\n'
      << "policy_period_ms=20.000\n"
      << "policy_deadline_misses=" << latency.deadline_misses << '/'
      << options.measured_iterations << " ("
      << 100.0 * latency.deadline_misses / options.measured_iterations << "%)\n"
      << "rss_before_kib=" << result.rss_before_kib << '\n'
      << "rss_after_kib=" << result.rss_after_kib << '\n'
      << "rss_delta_kib=" << result.rss_after_kib - result.rss_before_kib << '\n'
      << "output_checksum=" << result.output_checksum << '\n';
}

void write_csv(const std::string& path, const std::vector<double>& latency_us) {
  std::ofstream csv(path);
  if (!csv) throw std::runtime_error("Cannot write CSV: " + path);
  csv << "iteration,latency_us\n" << std::setprecision(9);
  for (size_t i = 0; i < latency_us.size(); ++i) {
    csv << i << ',' << latency_us[i] << '\n';
  }
  csv.close();
  if (!csv) throw std::runtime_error("Cannot write CSV: " + path);
}

}  // namespace policy_benchmark
