#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include <sys/resource.h>

#include <onnxruntime_cxx_api.h>
#include <openvino/openvino.hpp>
#include "isaaclab/algorithms/algorithms.h"

#ifndef WBC_BENCHMARK_DEFAULT_MODEL
#define WBC_BENCHMARK_DEFAULT_MODEL "config/policy/karate/params/policy.onnx"
#endif

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
  std::string runtime = "onnx";
  std::string model_path = WBC_BENCHMARK_DEFAULT_MODEL;
  std::string csv_path;
  size_t warmup_iterations = 200;
  size_t measured_iterations = 10000;
  uint32_t seed = 12345;
};

struct InputSpec {
  std::string name;
  std::vector<int64_t> shape;
  size_t element_count = 1;
};

void print_usage(const char* program) {
  std::cout << "Usage: " << program << " [options]\n"
            << "  --runtime NAME       Runtime backend: onnx or openvino (default: onnx)\n"
            << "  --model PATH         ONNX model (default: active karate policy)\n"
            << "  --warmup N           Warm-up inference calls (default: 200)\n"
            << "  --iterations N       Measured inference calls (default: 10000)\n"
            << "  --seed N             Synthetic input seed (default: 12345)\n"
            << "  --csv PATH           Write per-call latency samples in microseconds\n"
            << "  --help               Show this help\n";
}

size_t parse_count(const std::string& value, const std::string& option) {
  size_t parsed = 0;
  size_t consumed = 0;
  try {
    parsed = std::stoull(value, &consumed);
  } catch (const std::exception&) {
    throw std::runtime_error("Invalid value for " + option + ": " + value);
  }
  if (consumed != value.size()) {
    throw std::runtime_error("Invalid value for " + option + ": " + value);
  }
  return parsed;
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    } else if (arg == "--runtime" || arg == "--model" || arg == "--warmup" ||
           arg == "--iterations" || arg == "--seed" || arg == "--csv") {
      if (i + 1 >= argc) {
        throw std::runtime_error(arg + " requires a value");
      }
      const std::string value = argv[++i];
      if (arg == "--runtime") {
        options.runtime = value;
      } else if (arg == "--model") {
        options.model_path = value;
      } else if (arg == "--warmup") {
        options.warmup_iterations = parse_count(value, arg);
      } else if (arg == "--iterations") {
        options.measured_iterations = parse_count(value, arg);
      } else if (arg == "--seed") {
        const size_t seed = parse_count(value, arg);
        if (seed > std::numeric_limits<uint32_t>::max()) {
          throw std::runtime_error("--seed exceeds uint32 range");
        }
        options.seed = static_cast<uint32_t>(seed);
      } else {
        options.csv_path = value;
      }
    } else {
      throw std::runtime_error("Unknown option: " + arg);
    }
  }
  if (options.measured_iterations == 0) {
    throw std::runtime_error("--iterations must be greater than zero");
  }
  if (options.runtime != "onnx" && options.runtime != "openvino") {
    throw std::runtime_error("--runtime must be 'onnx' or 'openvino'");
  }
  return options;
}

std::string shape_string(const std::vector<int64_t>& shape) {
  std::ostringstream out;
  out << '[';
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) out << ',';
    out << shape[i];
  }
  out << ']';
  return out.str();
}

std::vector<InputSpec> inspect_inputs(const std::string& model_path,
                                      std::vector<int64_t>& output_shape) {
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx_policy_cpu_benchmark_metadata");
  Ort::SessionOptions session_options;
  session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
  Ort::Session session(env, model_path.c_str(), session_options);
  Ort::AllocatorWithDefaultOptions allocator;

  std::vector<InputSpec> inputs;
  for (size_t i = 0; i < session.GetInputCount(); ++i) {
    const auto name = session.GetInputNameAllocated(i, allocator);
    const auto type_info = session.GetInputTypeInfo(i);
    const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    InputSpec spec;
    spec.name = name.get();
    spec.shape = tensor_info.GetShape();
    for (const int64_t dimension : spec.shape) {
      if (dimension <= 0) {
        throw std::runtime_error(
            "OrtRunner requires static positive input dimensions; " + spec.name +
            " has shape " + shape_string(spec.shape));
      }
      const auto dim = static_cast<size_t>(dimension);
      if (spec.element_count > std::numeric_limits<size_t>::max() / dim) {
        throw std::runtime_error("Input tensor is too large: " + spec.name);
      }
      spec.element_count *= dim;
    }
    inputs.push_back(std::move(spec));
  }

  if (session.GetOutputCount() == 0) {
    throw std::runtime_error("ONNX model has no outputs");
  }
  const auto output_type_info = session.GetOutputTypeInfo(0);
  const auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
  output_shape = output_tensor_info.GetShape();
  if (output_shape.size() != 2 || output_shape[1] <= 0) {
    throw std::runtime_error("OrtRunner expects a rank-2 output with a static action dimension");
  }
  return inputs;
}

double timeval_seconds(const timeval& value) {
  return static_cast<double>(value.tv_sec) + static_cast<double>(value.tv_usec) / 1e6;
}

double process_cpu_seconds() {
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) != 0) return 0.0;
  return timeval_seconds(usage.ru_utime) + timeval_seconds(usage.ru_stime);
}

long current_rss_kib() {
  std::ifstream status("/proc/self/status");
  std::string key;
  while (status >> key) {
    if (key == "VmRSS:") {
      long rss = 0;
      std::string unit;
      status >> rss >> unit;
      return rss;
    }
    std::string rest;
    std::getline(status, rest);
  }
  return -1;
}

double percentile(const std::vector<double>& sorted, double fraction) {
  const size_t index = static_cast<size_t>(std::ceil(fraction * sorted.size())) - 1;
  return sorted[std::min(index, sorted.size() - 1)];
}

using ObservationMap = std::unordered_map<std::string, std::vector<float>>;

class InferenceBackend {
public:
  virtual ~InferenceBackend() = default;
  virtual std::vector<float> infer(ObservationMap observations) = 0;
};

class OnnxRuntimeBackend final : public InferenceBackend {
public:
  explicit OnnxRuntimeBackend(const std::string& model_path)
      : runner_(std::make_unique<isaaclab::OrtRunner>(model_path)) {}

  std::vector<float> infer(ObservationMap observations) override {
    return runner_->act(std::move(observations));
  }

private:
  std::unique_ptr<isaaclab::OrtRunner> runner_;
};

class OpenVinoBackend final : public InferenceBackend {
public:
  OpenVinoBackend(const std::string& model_path, const std::vector<InputSpec>& inputs)
      : compiled_model_(std::make_unique<ov::CompiledModel>(
            core_.compile_model(core_.read_model(model_path), "CPU"))),
        request_(std::make_unique<ov::InferRequest>(compiled_model_->create_infer_request())),
        inputs_(inputs) {}

  std::vector<float> infer(ObservationMap observations) override {
    std::vector<ov::Tensor> input_tensors;
    input_tensors.reserve(inputs_.size());
    for (size_t i = 0; i < inputs_.size(); ++i) {
      const auto& input = inputs_[i];
      const auto found = observations.find(input.name);
      if (found == observations.end()) {
        throw std::runtime_error("OpenVINO input not found: " + input.name);
      }
      if (found->second.size() != input.element_count) {
        throw std::runtime_error("OpenVINO input size mismatch: " + input.name);
      }

      ov::Shape shape;
      shape.reserve(input.shape.size());
      for (const int64_t dimension : input.shape) {
        shape.push_back(static_cast<size_t>(dimension));
      }
      input_tensors.emplace_back(
          ov::element::f32, shape, const_cast<float*>(found->second.data()));
      request_->set_input_tensor(i, input_tensors.back());
    }

    request_->infer();
    const ov::Tensor output = request_->get_output_tensor(0);
    const float* output_data = output.data<const float>();
    return std::vector<float>(output_data, output_data + output.get_size());
  }

private:
  ov::Core core_;
  std::unique_ptr<ov::CompiledModel> compiled_model_;
  std::unique_ptr<ov::InferRequest> request_;
  std::vector<InputSpec> inputs_;
};

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    if (!std::ifstream(options.model_path).good()) {
      throw std::runtime_error("Cannot open model: " + options.model_path);
    }

    std::vector<int64_t> output_shape;
    const auto inputs = inspect_inputs(options.model_path, output_shape);
    if (inputs.empty()) {
      throw std::runtime_error("ONNX model has no inputs");
    }

    std::unordered_map<std::string, std::vector<float>> observations;
    std::mt19937 generator(options.seed);
    std::uniform_real_distribution<float> input_distribution(-1.0f, 1.0f);
    for (const auto& input : inputs) {
      auto& values = observations[input.name];
      values.resize(input.element_count);
      for (float& value : values) value = input_distribution(generator);
    }

    const auto initialization_start = Clock::now();
    std::unique_ptr<InferenceBackend> runner;
    if (options.runtime == "onnx") {
      runner = std::make_unique<OnnxRuntimeBackend>(options.model_path);
    } else {
      runner = std::make_unique<OpenVinoBackend>(options.model_path, inputs);
    }
    const auto initialization_end = Clock::now();
    const double initialization_ms =
        std::chrono::duration<double, std::milli>(initialization_end - initialization_start).count();

    const auto warmup_start = Clock::now();
    std::vector<float> action;
    for (size_t i = 0; i < options.warmup_iterations; ++i) {
      action = runner->infer(observations);
    }
    const double warmup_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - warmup_start).count();

    const long rss_before_kib = current_rss_kib();
    const double cpu_before = process_cpu_seconds();
    std::vector<double> latency_us;
    latency_us.reserve(options.measured_iterations);
    double output_checksum = 0.0;

    const auto benchmark_start = Clock::now();
    for (size_t i = 0; i < options.measured_iterations; ++i) {
      const auto call_start = Clock::now();
      action = runner->infer(observations);
      const auto call_end = Clock::now();
      latency_us.push_back(std::chrono::duration<double, std::micro>(call_end - call_start).count());
      if (!action.empty()) output_checksum += action[i % action.size()];
    }
    const auto benchmark_end = Clock::now();
    const double cpu_after = process_cpu_seconds();
    const long rss_after_kib = current_rss_kib();

    const double wall_seconds = std::chrono::duration<double>(benchmark_end - benchmark_start).count();
    const double mean = std::accumulate(latency_us.begin(), latency_us.end(), 0.0) /
                        static_cast<double>(latency_us.size());
    double squared_error_sum = 0.0;
    size_t deadline_misses = 0;
    for (const double latency : latency_us) {
      const double delta = latency - mean;
      squared_error_sum += delta * delta;
      if (latency > 20000.0) ++deadline_misses;
    }
    const double stddev = std::sqrt(squared_error_sum / static_cast<double>(latency_us.size()));
    auto sorted_latency = latency_us;
    std::sort(sorted_latency.begin(), sorted_latency.end());
    const double cpu_utilization = wall_seconds > 0.0
        ? (cpu_after - cpu_before) / wall_seconds * 100.0
        : 0.0;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "benchmark=policy_cpu\n"
              << "runtime=" << options.runtime << '\n'
              << "model=" << options.model_path << '\n'
              << "device=CPU\n";
    if (options.runtime == "onnx") {
      std::cout << "execution_provider=ONNX Runtime default CPU provider\n"
                << "ort_graph_optimization=ORT_ENABLE_EXTENDED\n";
    } else {
      std::cout << "execution_provider=OpenVINO CPU plugin\n";
    }
    std::cout << "hardware_threads=" << std::thread::hardware_concurrency() << '\n'
              << "input_count=" << inputs.size() << '\n';
    for (const auto& input : inputs) {
      std::cout << "input." << input.name << ".shape=" << shape_string(input.shape)
                << " elements=" << input.element_count << '\n';
    }
    std::cout << "output.shape=" << shape_string(output_shape) << '\n'
              << "input_data=deterministic synthetic uniform[-1,1], seed=" << options.seed << '\n'
              << "warmup_iterations=" << options.warmup_iterations << '\n'
              << "measured_iterations=" << options.measured_iterations << '\n'
              << "initialization_ms=" << initialization_ms << '\n'
              << "warmup_total_ms=" << warmup_ms << '\n'
              << "latency_min_us=" << sorted_latency.front() << '\n'
              << "latency_mean_us=" << mean << '\n'
              << "latency_stddev_us=" << stddev << '\n'
              << "latency_p50_us=" << percentile(sorted_latency, 0.50) << '\n'
              << "latency_p90_us=" << percentile(sorted_latency, 0.90) << '\n'
              << "latency_p95_us=" << percentile(sorted_latency, 0.95) << '\n'
              << "latency_p99_us=" << percentile(sorted_latency, 0.99) << '\n'
              << "latency_max_us=" << sorted_latency.back() << '\n'
              << "throughput_inferences_per_sec="
              << static_cast<double>(options.measured_iterations) / wall_seconds << '\n'
              << "process_cpu_utilization_pct=" << cpu_utilization << '\n'
              << "policy_period_ms=20.000\n"
              << "policy_deadline_misses=" << deadline_misses << '/'
              << options.measured_iterations << " ("
              << 100.0 * deadline_misses / options.measured_iterations << "%)\n"
              << "rss_before_kib=" << rss_before_kib << '\n'
              << "rss_after_kib=" << rss_after_kib << '\n'
              << "rss_delta_kib=" << (rss_before_kib >= 0 && rss_after_kib >= 0
                                           ? rss_after_kib - rss_before_kib
                                           : -1)
              << '\n'
              << "output_checksum=" << output_checksum << '\n';

    if (!options.csv_path.empty()) {
      std::ofstream csv(options.csv_path);
      if (!csv) throw std::runtime_error("Cannot write CSV: " + options.csv_path);
      csv << "iteration,latency_us\n" << std::setprecision(9);
      for (size_t i = 0; i < latency_us.size(); ++i) {
        csv << i << ',' << latency_us[i] << '\n';
      }
      std::cout << "latency_csv=" << options.csv_path << '\n';
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "benchmark error: " << error.what() << '\n';
    return 1;
  }
}