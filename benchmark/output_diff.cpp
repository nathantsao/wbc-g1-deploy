// Compares each runtime:device policy output with ONNX Runtime CPU on seeded synthetic inputs.
#include "backends/backend.h"
#include "benchmark.h"
#include "model.h"
#include "options.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void print_usage(const char* program) {
  std::cout << "Usage: " << program << " --model PATH [options] RUNTIME:DEVICE...\n"
            << "  --seeds N                Compare seeds 1..N (default: 200)\n"
            << "  --action-scale A,B,...   Per-joint action scale; adds joint target difference\n"
            << "  --provider-library PATH  OpenVINO EP plugin library (for openvino-ep)\n"
            << "Example: " << program << " --model policy.onnx openvino:CPU openvino:GPU\n";
}

std::vector<float> parse_scale(const std::string& text) {
  std::vector<float> scale;
  std::stringstream stream(text);
  std::string item;
  while (std::getline(stream, item, ',')) {
    size_t used = 0;
    scale.push_back(std::stof(item, &used));
    if (used != item.size()) throw std::runtime_error("Invalid --action-scale value: " + item);
  }
  return scale;
}

std::vector<std::vector<float>> run_seeds(const policy_benchmark::Options& options,
                                          const policy_benchmark::ModelInfo& model,
                                          uint32_t seeds,
                                          policy_benchmark::PrecisionInfo* precision = nullptr) {
  auto backend = policy_benchmark::make_backend(options, model);
  std::vector<std::vector<float>> actions;
  actions.reserve(seeds);
  for (uint32_t seed = 1; seed <= seeds; ++seed) {
    actions.push_back(backend->infer(policy_benchmark::make_observations(model, seed)));
  }
  if (precision) *precision = backend->inference_precision();
  return actions;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string model_path;
    std::string provider_library;
    std::vector<float> scale;
    uint32_t seeds = 200;
    std::vector<std::string> configs;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "-h" || arg == "--help") {
        print_usage(argv[0]);
        return 0;
      }
      if (arg == "--model" || arg == "--seeds" || arg == "--action-scale" ||
          arg == "--provider-library") {
        if (i + 1 >= argc) throw std::runtime_error("Missing value for " + arg);
        const std::string value = argv[++i];
        if (arg == "--model") model_path = value;
        if (arg == "--provider-library") provider_library = value;
        if (arg == "--action-scale") scale = parse_scale(value);
        if (arg == "--seeds") {
          const size_t count = policy_benchmark::parse_count(value, arg);
          if (count == 0 || count > 1000000) throw std::runtime_error("--seeds must be 1..1000000");
          seeds = static_cast<uint32_t>(count);
        }
      } else if (arg.find(':') != std::string::npos) {
        configs.push_back(arg);
      } else {
        throw std::runtime_error("Unknown argument: " + arg);
      }
    }
    if (model_path.empty() || configs.empty()) {
      print_usage(argv[0]);
      return 1;
    }

    const auto model = policy_benchmark::inspect_model(model_path);
    policy_benchmark::Options baseline_options;
    baseline_options.model_path = model_path;
    const auto baseline = run_seeds(baseline_options, model, seeds);
    policy_benchmark::check_reference(baseline, scale);

    std::cout << "config,seeds,max_action_diff" << (scale.empty() ? "" : ",max_joint_target_diff_rad")
              << ",inference_precision\n";
    int failures = 0;
    for (const auto& config : configs) {
      policy_benchmark::Options options;
      options.model_path = model_path;
      options.runtime = config.substr(0, config.find(':'));
      options.device = config.substr(config.find(':') + 1);
      std::transform(options.device.begin(), options.device.end(), options.device.begin(),
                     [](unsigned char c) { return std::toupper(c); });
      options.provider_library_path = provider_library;
      try {
        if (options.runtime != "onnx" && options.runtime != "openvino" &&
            options.runtime != "openvino-ep") {
          throw std::runtime_error("Unknown runtime: " + options.runtime);
        }
        if (options.runtime == "openvino-ep" && provider_library.empty()) {
          throw std::runtime_error("openvino-ep requires --provider-library");
        }
        policy_benchmark::PrecisionInfo precision;
        const auto diff = policy_benchmark::compare_outputs(
            baseline, run_seeds(options, model, seeds, &precision), scale);
        std::cout << config << ',' << seeds << ',' << std::setprecision(6) << diff.max_action_diff;
        if (!scale.empty()) std::cout << ',' << diff.max_joint_target_diff;
        std::cout << ',' << precision.precision << '\n';
      } catch (const std::exception& error) {
        ++failures;
        std::cout << config << ",FAILED\n";
        std::cerr << config << ": " << error.what() << '\n';
      }
    }
    return failures == static_cast<int>(configs.size()) ? 1 : 0;
  } catch (const std::exception& error) {
    std::cerr << "output diff error: " << error.what() << '\n';
    return 1;
  }
}
