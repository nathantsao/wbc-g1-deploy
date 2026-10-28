#include "options.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace policy_benchmark {
namespace {

void print_usage(const char* program) {
  std::cout << "Usage: " << program << " [options]\n"
            << "  --runtime NAME       Backend: onnx, openvino, or openvino-ep (default: onnx)\n"
            << "  --device NAME        CPU, GPU, GPU.N (OpenVINO), or NPU (default: CPU)\n"
            << "  --provider NAME      ONNX provider: cpu or cuda (default: cpu)\n"
            << "  --device-id N        CUDA device index (default: 0; requires --provider cuda)\n"
            << "  --list-devices       List devices/providers without loading a model\n"
            << "  --model PATH         ONNX model (default: active karate policy)\n"
            << "  --provider-library PATH  OpenVINO EP plugin library (for openvino-ep)\n"
            << "  --warmup N           Warm-up inference calls (default: 200)\n"
            << "  --iterations N       Measured inference calls (default: 10000)\n"
            << "  --seed N             Synthetic input seed (default: 12345)\n"
            << "  --csv PATH           Write per-call latency samples in microseconds\n"
            << "  --help               Show this help\n";
}

size_t parse_count(const std::string& value, const std::string& option) {
  if (value.empty() || !std::all_of(value.begin(), value.end(),
                                  [](unsigned char c) { return std::isdigit(c); })) {
    throw std::runtime_error("Invalid value for " + option + ": " + value);
  }
  unsigned long long parsed;
  try {
    parsed = std::stoull(value);
  } catch (const std::exception&) {
    throw std::runtime_error("Invalid value for " + option + ": " + value);
  }
  if (parsed > std::numeric_limits<size_t>::max()) {
    throw std::runtime_error("Invalid value for " + option + ": " + value);
  }
  return static_cast<size_t>(parsed);
}

bool is_openvino_device(const std::string& device) {
  if (device == "CPU" || device == "GPU" || device == "NPU") return true;
  return device.size() > 4 && device.compare(0, 4, "GPU.") == 0 &&
         std::all_of(device.begin() + 4, device.end(),
                     [](unsigned char c) { return std::isdigit(c); });
}

}  // namespace

Options parse_options(int argc, char** argv) {
  Options options;
  bool provider_set = false;
  bool device_id_set = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    }
    if (arg == "--list-devices") {
      options.list_devices = true;
      continue;
    }
    if (arg != "--runtime" && arg != "--device" && arg != "--provider" &&
        arg != "--device-id" && arg != "--model" && arg != "--provider-library" &&
        arg != "--warmup" && arg != "--iterations" && arg != "--seed" && arg != "--csv") {
      throw std::runtime_error("Unknown option: " + arg);
    }
    if (i + 1 >= argc) throw std::runtime_error(arg + " requires a value");
    const std::string value = argv[++i];
    if (arg == "--runtime") {
      options.runtime = value;
    } else if (arg == "--device") {
      options.device = value;
      std::transform(options.device.begin(), options.device.end(), options.device.begin(),
                     [](unsigned char c) { return std::toupper(c); });
    } else if (arg == "--provider") {
      provider_set = true;
      options.provider = value;
    } else if (arg == "--device-id") {
      device_id_set = true;
      const size_t id = parse_count(value, arg);
      if (id > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("--device-id exceeds int range");
      }
      options.device_id = static_cast<int>(id);
    } else if (arg == "--model") {
      options.model_path = value;
    } else if (arg == "--provider-library") {
      options.provider_library_path = value;
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
  }
  if (options.measured_iterations == 0) {
    throw std::runtime_error("--iterations must be greater than zero");
  }
  if (options.runtime != "onnx" && options.runtime != "openvino" &&
      options.runtime != "openvino-ep") {
    throw std::runtime_error("--runtime must be 'onnx', 'openvino', or 'openvino-ep'");
  }
  if (options.runtime == "onnx") {
    if (options.provider != "cpu" && options.provider != "cuda") {
      throw std::runtime_error("--provider must be 'cpu' or 'cuda'");
    }
    if ((options.provider == "cpu" && options.device != "CPU") ||
        (options.provider == "cuda" && options.device != "GPU")) {
      throw std::runtime_error(
          "ONNX CPU requires --device CPU; CUDA requires --provider cuda --device GPU");
    }
  } else {
    if (provider_set) throw std::runtime_error("--provider is only valid with --runtime onnx");
    if (!is_openvino_device(options.device)) {
      throw std::runtime_error("--device must be CPU, GPU, GPU.N, or NPU for OpenVINO");
    }
  }
  if (device_id_set && (options.runtime != "onnx" || options.provider != "cuda")) {
    throw std::runtime_error("--device-id is only valid with --runtime onnx --provider cuda");
  }
  if (options.runtime == "openvino-ep" && options.provider_library_path.empty()) {
    throw std::runtime_error("--provider-library is required when --runtime is 'openvino-ep'");
  }
  if (options.runtime != "openvino-ep" && !options.provider_library_path.empty()) {
    throw std::runtime_error("--provider-library is only valid with --runtime openvino-ep");
  }
  return options;
}

}  // namespace policy_benchmark
