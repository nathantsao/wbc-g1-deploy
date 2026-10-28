#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#ifndef WBC_BENCHMARK_DEFAULT_MODEL
#define WBC_BENCHMARK_DEFAULT_MODEL "config/policy/karate/params/policy.onnx"
#endif

namespace policy_benchmark {

struct Options {
  std::string runtime = "onnx";
  std::string device = "CPU";
  std::string provider = "cpu";
  int device_id = 0;
  int threads = 0;  // 0 keeps the runtime default
  std::string model_path = WBC_BENCHMARK_DEFAULT_MODEL;
  std::string provider_library_path;
  std::string csv_path;
  size_t warmup_iterations = 200;
  size_t measured_iterations = 10000;
  uint32_t seed = 12345;
  bool list_devices = false;
};

Options parse_options(int argc, char** argv);

}  // namespace policy_benchmark
