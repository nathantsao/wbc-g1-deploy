#pragma once

#include "isaaclab/algorithms/algorithms.h"

#include <filesystem>
#include <memory>
#include <string>
#include <yaml-cpp/yaml.h>

namespace wbc_deploy {

/// Policy inference settings from the ``inference:`` block of ``config/config.yaml``.
struct InferenceConfig {
  std::string runtime = "onnxruntime";  ///< onnxruntime | openvino | openvino-ep
  std::string device = "CPU";           ///< CPU | GPU | GPU.N | NPU (onnxruntime: CPU only)
  int threads = 0;                      ///< CPU inference threads; 0 = runtime default
  std::string cache_dir;                ///< OpenVINO compiled-model cache; empty = disabled
  std::string openvino_ep_library;      ///< OpenVINO EP plugin (.so) for openvino-ep
  bool npu_turbo = false;               ///< NPU only: keep the NPU at maximum frequency
  double latency_log_interval = 0.0;    ///< Seconds between latency log lines; 0 = off
};

/// Read the root ``inference:`` block; keys in the FSM state's own ``inference:`` block override it.
InferenceConfig load_inference_config(const YAML::Node& root, const YAML::Node& state_cfg);

/// Create the policy runner for ``config``. Loads, compiles, and warms up the model, so call it
/// while constructing the FSM state, never from the control loop. ``label`` prefixes log lines.
std::unique_ptr<isaaclab::Algorithms> make_policy_runner(
  const std::filesystem::path& model_path, const InferenceConfig& config, const std::string& label);

}  // namespace wbc_deploy
