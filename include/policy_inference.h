#pragma once

#include "isaaclab/algorithms/algorithms.h"

#include <filesystem>
#include <memory>
#include <string>
#include <yaml-cpp/yaml.h>

namespace wbc_deploy {

/// Policy inference settings from the ``inference:`` block of ``config/config.yaml``.
struct InferenceConfig {
  std::string runtime = "onnxruntime";  ///< onnxruntime
  std::string device = "CPU";           ///< CPU
  int threads = 0;                      ///< CPU inference threads; 0 = runtime default
};

/// Read the root ``inference:`` block; keys in the FSM state's own ``inference:`` block override it.
InferenceConfig load_inference_config(const YAML::Node& root, const YAML::Node& state_cfg);

/// Create the policy runner for ``config``. Loads and prepares the model, so call it while
/// constructing the FSM state, never from the control loop. ``label`` prefixes log lines.
std::unique_ptr<isaaclab::Algorithms> make_policy_runner(
  const std::filesystem::path& model_path, const InferenceConfig& config, const std::string& label);

}  // namespace wbc_deploy
