#include "policy_inference.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <optional>
#include <stdexcept>
#include <utility>

namespace wbc_deploy {
namespace {

std::optional<YAML::Node> inference_block(const YAML::Node& node, const std::string& where)
{
  if (!node || !node["inference"]) {
    return std::nullopt;
  }
  const YAML::Node block = node["inference"];
  if (!block.IsMap()) {
    throw std::runtime_error(where + " inference: must be a map");
  }
  return block;
}

std::optional<YAML::Node> lookup(
  const std::optional<YAML::Node>& state_block, const std::optional<YAML::Node>& root_block,
  const char* key)
{
  for (const auto* block : {&state_block, &root_block}) {
    if (*block && (**block)[key]) {
      return (**block)[key];
    }
  }
  return std::nullopt;
}

std::string to_lower(std::string text)
{
  std::transform(text.begin(), text.end(), text.begin(),
    [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return text;
}

std::string to_upper(std::string text)
{
  std::transform(text.begin(), text.end(), text.begin(),
    [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return text;
}

std::unique_ptr<isaaclab::Algorithms> create_runner(
  const std::filesystem::path& model_path, const InferenceConfig& config, std::string& description)
{
  if (config.runtime == "onnxruntime") {
    if (config.device != "CPU") {
      throw std::runtime_error("runtime onnxruntime supports only device CPU");
    }
    description = "ONNX Runtime CPU";
    if (config.threads <= 0) {
      return std::make_unique<isaaclab::OrtRunner>(model_path.string());
    }
    isaaclab::OrtRunnerOptions options;
    options.threads = config.threads;
    return std::make_unique<isaaclab::OrtRunner>(model_path.string(), options);
  }

  throw std::runtime_error("unknown runtime " + config.runtime);
}

}  // namespace

InferenceConfig load_inference_config(const YAML::Node& root, const YAML::Node& state_cfg)
{
  const auto root_block = inference_block(root, "config.yaml");
  const auto state_block = inference_block(state_cfg, "FSM state");
  InferenceConfig config;

  if (const auto node = lookup(state_block, root_block, "runtime")) {
    config.runtime = to_lower(node->as<std::string>());
  }
  if (config.runtime != "onnxruntime") {
    throw std::runtime_error("inference.runtime '" + config.runtime
      + "' is invalid; use onnxruntime");
  }

  if (const auto node = lookup(state_block, root_block, "device")) {
    config.device = to_upper(node->as<std::string>());
  }
  if (config.device.empty()) {
    throw std::runtime_error("inference.device must not be empty");
  }

  if (const auto node = lookup(state_block, root_block, "threads")) {
    config.threads = node->as<int>();
  }
  if (config.threads < 0) {
    throw std::runtime_error("inference.threads must be 0 (runtime default) or positive");
  }
  return config;
}

std::unique_ptr<isaaclab::Algorithms> make_policy_runner(
  const std::filesystem::path& model_path, const InferenceConfig& config, const std::string& label)
{
  const auto start = std::chrono::steady_clock::now();
  std::unique_ptr<isaaclab::Algorithms> runner;
  std::string description;
  try {
    runner = create_runner(model_path, config, description);
  } catch (const std::exception& e) {
    const std::string message = label + ": cannot create policy runner (runtime " + config.runtime
      + ", device " + config.device + "): " + e.what();
    spdlog::critical("{}", message);
    throw std::runtime_error(message);
  }
  const auto load_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - start).count();

  spdlog::info("{}: policy inference on {} (threads {}), loaded in {:.0f} ms",
    label, description, config.threads > 0 ? std::to_string(config.threads) : "default", load_ms);
  return runner;
}

}  // namespace wbc_deploy
