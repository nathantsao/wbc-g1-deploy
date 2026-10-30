#include "backends/backend.h"
#include "benchmark.h"
#include "model.h"
#include "options.h"
#include "report.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

void check(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void expect_error(const std::function<void()>& call, const std::string& message) {
  try {
    call();
  } catch (const std::exception& error) {
    check(std::string(error.what()).find(message) != std::string::npos,
          "Unexpected error: " + std::string(error.what()));
    return;
  }
  throw std::runtime_error("Expected error containing: " + message);
}

policy_benchmark::Options parse(std::vector<std::string> args) {
  args.insert(args.begin(), "benchmark");
  std::vector<char*> argv;
  for (auto& arg : args) argv.push_back(arg.data());
  return policy_benchmark::parse_options(static_cast<int>(argv.size()), argv.data());
}

void test_options() {
  const auto defaults = parse({});
  check(defaults.runtime == "onnx" && defaults.device == "CPU" && defaults.provider == "cpu",
        "Default backend changed");
  check(defaults.warmup_iterations == 200 && defaults.measured_iterations == 10000 &&
        defaults.seed == 12345, "Default sampling changed");
  const auto cuda = parse({"--provider", "cuda", "--device", "gpu", "--device-id", "2"});
  check(cuda.device == "GPU" && cuda.device_id == 2, "CUDA options not parsed");
  const auto ov = parse({"--runtime", "openvino", "--device", "gpu.1", "--warmup", "0",
                         "--iterations", "1"});
  check(ov.device == "GPU.1" && ov.warmup_iterations == 0 && ov.measured_iterations == 1,
        "OpenVINO options not parsed");
  const auto ep = parse({"--runtime", "openvino-ep", "--device", "npu",
                         "--provider-library", "/plugin.so", "--list-devices"});
  check(ep.device == "NPU" && ep.list_devices, "EP options not parsed");
  check(parse({"--seed", "4294967295"}).seed == 4294967295U, "Maximum seed rejected");
  check(defaults.threads == 0, "Default threads changed");
  check(parse({"--threads", "2"}).threads == 2, "ONNX threads not parsed");
  check(parse({"--runtime", "openvino", "--threads", "1"}).threads == 1,
        "OpenVINO threads not parsed");
  expect_error([] { parse({"--threads", "0"}); }, "--threads must be greater than zero");
  expect_error([] { parse({"--threads", "2147483648"}); }, "int range");
  expect_error([] { parse({"--runtime", "openvino", "--device", "GPU", "--threads", "2"}); },
               "--threads is only valid");
  expect_error([] { parse({"--iterations", "0"}); }, "greater than zero");
  for (const std::string value : {"-1", "1x", "", " 1", "+1", "18446744073709551616"}) {
    expect_error([&] { parse({"--iterations", value}); }, "Invalid value");
  }
  expect_error([] { parse({"--seed", "4294967296"}); }, "uint32 range");
  expect_error([] { parse({"--model"}); }, "requires a value");
  expect_error([] { parse({"--unknown"}); }, "Unknown option");
  expect_error([] { parse({"--runtime", "bad"}); }, "--runtime must");
  expect_error([] { parse({"--provider", "bad"}); }, "--provider must");
  expect_error([] { parse({"--device", "GPU"}); }, "CUDA requires");
  expect_error([] { parse({"--provider", "cuda"}); }, "CUDA requires");
  expect_error([] { parse({"--device", "NPU"}); }, "CUDA requires");
  expect_error([] { parse({"--device-id", "0"}); }, "--device-id is only valid");
  expect_error([] { parse({"--provider", "cuda", "--device", "GPU", "--device-id", "-1"}); },
               "Invalid value");
  expect_error([] { parse({"--provider", "cuda", "--device", "GPU",
                           "--device-id", "2147483648"}); }, "int range");
  expect_error([] { parse({"--runtime", "openvino", "--provider", "cpu"}); },
               "--provider is only valid");
  expect_error([] { parse({"--runtime", "openvino", "--device-id", "0"}); },
               "--device-id is only valid");
  for (const std::string device : {"AUTO", "MULTI:GPU,CPU", "GPU.", "GPU.-1", "CPU.1", "foo"}) {
    expect_error([&] { parse({"--runtime", "openvino", "--device", device}); }, "--device must");
  }
  expect_error([] { parse({"--runtime", "openvino-ep"}); }, "--provider-library is required");
  expect_error([] { parse({"--provider-library", "/plugin.so"}); }, "only valid");
  check(policy_benchmark::parse_count("3", "--seeds") == 3, "parse_count rejected a digit");
  for (const std::string value : {"3abc", "3.0", " 3", "+3", "-3", "", "0x3"}) {
    expect_error([&] { policy_benchmark::parse_count(value, "--seeds"); },
                 "Invalid value for --seeds");
  }
}

void test_output_compare() {
  using policy_benchmark::compare_outputs;
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  const policy_benchmark::SeedOutputs baseline{{0.0f, 1.0f}, {2.0f, 3.0f}};
  const policy_benchmark::SeedOutputs candidate{{0.5f, 1.0f}, {2.0f, 2.75f}};
  const auto diff = compare_outputs(baseline, candidate, {2.0f, 4.0f});
  check(diff.max_action_diff == 0.5 && diff.max_joint_target_diff == 1.0,
        "Wrong max action or joint target difference");
  check(compare_outputs(baseline, baseline, {}).max_action_diff == 0 &&
        compare_outputs(baseline, candidate, {}).max_joint_target_diff == 0,
        "Identical outputs or missing scale gave a nonzero difference");
  check(policy_benchmark::first_non_finite({1.0f, nan, inf}) == 1 &&
        policy_benchmark::first_non_finite({1.0f, 2.0f}) == 2, "first_non_finite changed");

  for (const float bad : {nan, -nan, inf, -inf}) {
    auto bad_baseline = baseline;
    bad_baseline[1][0] = bad;
    expect_error([&] { policy_benchmark::check_reference(bad_baseline, {}); },
                 "Baseline output for seed 2 is non-finite");
    expect_error([&] { compare_outputs(bad_baseline, candidate, {}); },
                 "Baseline output for seed 2 is non-finite");

    // std::max skips NaN, so without the check a NaN candidate would report a 0.5 difference.
    auto bad_candidate = candidate;
    bad_candidate[1][1] = bad;
    expect_error([&] { compare_outputs(baseline, bad_candidate, {}); },
                 "Candidate output for seed 2 is non-finite");
    expect_error([&] { compare_outputs(baseline, bad_candidate, {1.0f, 1.0f}); },
                 "Candidate output for seed 2 is non-finite");

    expect_error([&] { policy_benchmark::check_reference(baseline, {1.0f, bad}); },
                 "--action-scale is non-finite");
    expect_error([&] { compare_outputs(baseline, candidate, {bad, 1.0f}); },
                 "--action-scale is non-finite");
  }
  expect_error([&] { policy_benchmark::check_reference(baseline, {1.0f}); },
               "--action-scale has 1 values but the policy outputs 2");
  expect_error([&] { compare_outputs(baseline, {{0.0f, 1.0f}}, {}); }, "1 seeds");
  expect_error([&] { compare_outputs(baseline, {{0.0f}, {2.0f, 3.0f}}, {}); },
               "Output size differs for seed 1");
  expect_error([] { policy_benchmark::check_reference({}, {}); }, "no outputs");
}

void test_device_resolution() {
  using policy_benchmark::resolve_device;
  check(resolve_device("CPU", {"CPU", "GPU.0", "GPU.1"}) == "CPU", "CPU resolution failed");
  check(resolve_device("GPU", {"CPU", "GPU.1", "GPU.0"}) == "GPU.0", "GPU alias failed");
  check(resolve_device("GPU", {"GPU"}) == "GPU", "Unindexed GPU failed");
  check(resolve_device("GPU.0", {"GPU"}) == "GPU", "First GPU index alias failed");
  check(resolve_device("GPU.1", {"GPU.0", "GPU.1"}) == "GPU.1", "Indexed GPU failed");
  check(resolve_device("NPU", {"CPU", "NPU"}) == "NPU", "NPU resolution failed");
  expect_error([] { resolve_device("GPU", {"CPU"}); }, "available devices: CPU");
  expect_error([] { resolve_device("GPU.1", {"GPU.0"}); }, "unavailable");
  expect_error([] { resolve_device("GPU.1", {"GPU"}); }, "unavailable");
  expect_error([] { resolve_device("NPU", {}); }, "(none)");
}

void test_statistics() {
  using policy_benchmark::summarize_latency;
  const std::vector<double> values{5, 1, 4, 2, 3};
  const auto stats = summarize_latency(values);
  check(stats.min == 1 && stats.max == 5 && stats.mean == 3, "Min/max/mean changed");
  check(std::abs(stats.stddev - std::sqrt(2.0)) < 1e-12, "Population stddev changed");
  check(stats.p50 == 3 && stats.p90 == 5 && stats.p95 == 5 && stats.p99 == 5,
        "Nearest-rank percentiles changed");
  check(values.front() == 5, "Statistics reordered per-call samples");
  const auto single = summarize_latency({7});
  check(single.p50 == 7 && single.p99 == 7 && single.stddev == 0, "Single sample failed");
  check(summarize_latency({19999.0, 20000.0, 20000.001}).deadline_misses == 1,
        "Deadline must be strictly greater than 20 ms");
  expect_error([] { summarize_latency({}); }, "empty latency");
}

void test_observations() {
  policy_benchmark::ModelInfo model;
  model.inputs = {{"a", {1, 4}, 4}, {"b", {1, 2}, 2}};
  const auto first = policy_benchmark::make_observations(model, 12345);
  check(first == policy_benchmark::make_observations(model, 12345), "Inputs are not deterministic");
  check(first != policy_benchmark::make_observations(model, 12346), "Seed has no effect");
  check(first.at("a").size() == 4 && first.at("b").size() == 2, "Wrong input extents");
  for (const auto& input : first) {
    for (float value : input.second) check(value >= -1 && value <= 1, "Input outside [-1,1]");
  }
  check(policy_benchmark::shape_string({1, 132}) == "[1,132]", "Shape format changed");
}

void test_report_and_csv() {
  policy_benchmark::Options options;
  options.measured_iterations = 3;
  policy_benchmark::BenchmarkResult result;
  result.backend = {"CPU", "test CPU", "test provider"};
  result.model.inputs = {{"obs", {1, 2}, 2}};
  result.model.output_shape = {1, 29};
  result.latency_us = {3, 1, 2};
  result.latency = policy_benchmark::summarize_latency(result.latency_us);
  result.wall_seconds = 1;
  result.rss_before_kib = 10;
  result.rss_after_kib = 12;
  std::ostringstream report;
  policy_benchmark::print_report(options, result, report);
  for (const std::string expected : {
         "benchmark=policy_cpu\n", "requested_device=CPU\n", "device=CPU\n",
         "input.obs.shape=[1,2] elements=2\n", "output.shape=[1,29]\n",
         "latency_mean_us=2.000\n", "throughput_inferences_per_sec=3.000\n",
         "policy_deadline_misses=0/3 (0.000%)\n", "rss_delta_kib=2\n",
         "resource_metrics_scope=host_process\n", "threads=default\n"}) {
    check(report.str().find(expected) != std::string::npos, "Missing report field: " + expected);
  }
  options.runtime = "onnx";
  options.provider = "cuda";
  options.device = "GPU";
  options.device_id = 2;
  result.backend = {"GPU.2", "CUDA device 2", "ONNX Runtime CUDA provider"};
  std::ostringstream gpu_report;
  policy_benchmark::print_report(options, result, gpu_report);
  check(gpu_report.str().find("device=GPU.2\n") != std::string::npos &&
        gpu_report.str().find("cuda_device_id=2\n") != std::string::npos &&
        gpu_report.str().find("ort_cpu_ep_fallback=disabled\n") != std::string::npos,
        "Accelerator report is misleading");

  char path[] = "/tmp/policy-benchmark-csv-XXXXXX";
  const int fd = mkstemp(path);
  if (fd == -1) throw std::runtime_error("Cannot create test CSV");
  close(fd);
  try {
    policy_benchmark::write_csv(path, result.latency_us);
    std::ifstream csv(path);
    std::ostringstream contents;
    contents << csv.rdbuf();
    check(contents.str() == "iteration,latency_us\n0,3\n1,1\n2,2\n",
          "CSV format or sample order changed");
  } catch (...) {
    std::remove(path);
    throw;
  }
  std::remove(path);
  expect_error([&] { policy_benchmark::write_csv("/dev/full", result.latency_us); },
               "Cannot write CSV");
}

void test_inference(const std::string& runtime, const std::string& model_path,
                    const std::string& device, const std::string& provider_library) {
  policy_benchmark::Options options;
  options.runtime = runtime;
  options.device = device;
  options.model_path = model_path;
  options.provider_library_path = provider_library;
  check(runtime != "openvino-ep" || !provider_library.empty(),
        "openvino-ep inference tests require PROVIDER_LIBRARY");
  options.warmup_iterations = 0;
  options.measured_iterations = 1;
  const auto single = policy_benchmark::run_benchmark(options);
  check(single.latency_us.size() == 1 && single.latency.min > 0 &&
        single.latency.min == single.latency.max && single.latency.stddev == 0,
        "One-iteration inference failed");
  check(std::isfinite(single.output_checksum), "Nonfinite inference output");
  options.warmup_iterations = 2;
  options.measured_iterations = 5;
  const auto result = policy_benchmark::run_benchmark(options);
  const auto repeat = policy_benchmark::run_benchmark(options);
  check(result.latency_us.size() == 5 && result.wall_seconds > 0 &&
        result.initialization_ms > 0 && result.warmup_ms > 0,
        "Wrong inference timing or sample count");
  check(result.rss_before_kib > 0 && result.rss_after_kib > 0,
        "Missing host resource metrics");
  check(std::isfinite(result.output_checksum) &&
        std::abs(result.output_checksum - repeat.output_checksum) < 1e-5,
        "Seeded inference output changed between runs");
  const auto model = policy_benchmark::inspect_model(model_path);
  auto backend = policy_benchmark::make_backend(options, model);
  const auto action = backend->infer(policy_benchmark::make_observations(model, options.seed));
  check(action.size() == static_cast<size_t>(model.output_shape[1]), "Wrong action extent");
  for (float value : action) check(std::isfinite(value), "Nonfinite action");
  if (device == "CPU") {
    options.threads = 1;
    const auto single_thread = policy_benchmark::run_benchmark(options);
    check(single_thread.latency_us.size() == 5 &&
          std::abs(single_thread.output_checksum - result.output_checksum) < 1e-3,
          "Explicit thread count changed inference output");
  }
  if (device != "CPU") {
    check(result.backend.device.compare(0, 3, device.substr(0, 3)) == 0,
          "Accelerator request did not execute on the requested device type");
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc >= 3 && argc <= 5) {
      test_inference(argv[1], argv[2], argc >= 4 ? argv[3] : "CPU", argc == 5 ? argv[4] : "");
      std::cout << "Synthetic inference tests passed\n";
      return 0;
    }
    check(argc == 1,
          "Usage: policy_benchmark_tests [RUNTIME MODEL [DEVICE [PROVIDER_LIBRARY]]]");
    test_options();
    test_device_resolution();
    test_statistics();
    test_observations();
    test_report_and_csv();
    test_output_compare();
    std::cout << "All benchmark unit tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Test failure: " << error.what() << '\n';
    return 1;
  }
}
