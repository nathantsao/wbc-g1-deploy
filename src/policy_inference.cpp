#include "policy_inference.h"

#include "param.h"

#ifdef WBC_WITH_OPENVINO
#include "isaaclab/algorithms/openvino_runner.h"
#endif

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <system_error>
#include <utility>

#ifndef WBC_DEFAULT_OPENVINO_EP_LIBRARY
#define WBC_DEFAULT_OPENVINO_EP_LIBRARY ""
#endif

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

std::string default_cache_dir()
{
  if (const char* xdg = std::getenv("XDG_CACHE_HOME"); xdg && *xdg) {
    return (std::filesystem::path(xdg) / "wbc_g1_deploy" / "openvino").string();
  }
  if (const char* home = std::getenv("HOME"); home && *home) {
    return (std::filesystem::path(home) / ".cache" / "wbc_g1_deploy" / "openvino").string();
  }
  return "";
}

std::string resolve_optional_path(const std::string& path)
{
  return path.empty() ? path : param::resolve_path_under_proj(path).string();
}

std::unique_ptr<isaaclab::Algorithms> create_runner(
  const std::filesystem::path& model_path, const InferenceConfig& config, std::string& description)
{
  if (config.runtime == "onnxruntime") {
    if (config.device != "CPU") {
      throw std::runtime_error("runtime onnxruntime supports only device CPU; use openvino or "
        "openvino-ep for " + config.device);
    }
    description = "ONNX Runtime CPU";
    if (config.threads <= 0) {
      return std::make_unique<isaaclab::OrtRunner>(model_path.string());
    }
    isaaclab::OrtRunnerOptions options;
    options.threads = config.threads;
    return std::make_unique<isaaclab::OrtRunner>(model_path.string(), options);
  }

  if (config.runtime == "openvino") {
#ifdef WBC_WITH_OPENVINO
    isaaclab::OpenVinoRunnerOptions options;
    options.device = config.device;
    options.threads = config.threads;
    options.cache_dir = config.cache_dir;
    auto runner = std::make_unique<isaaclab::OpenVinoRunner>(model_path.string(), options);
    description = "OpenVINO " + runner->device() + " (" + runner->device_full_name() + ", "
      + runner->inference_precision() + ")";
    return runner;
#else
    throw std::runtime_error("this build has no OpenVINO runtime; reconfigure with "
      "-DWBC_ENABLE_OPENVINO=ON");
#endif
  }

  if (config.runtime == "openvino-ep") {
    if (config.openvino_ep_library.empty()) {
      throw std::runtime_error("no OpenVINO EP plugin configured; set inference.openvino_ep_library");
    }
    if (!std::filesystem::exists(config.openvino_ep_library)) {
      throw std::runtime_error("OpenVINO EP plugin not found: " + config.openvino_ep_library
        + " (run scripts/bootstrap_thirdparty_intel.sh or set inference.openvino_ep_library)");
    }
    isaaclab::OrtRunnerOptions options;
    options.threads = config.threads;
    options.openvino_ep_library = config.openvino_ep_library;
    options.openvino_device = config.device;
    options.cache_dir = config.cache_dir;
    auto runner = std::make_unique<isaaclab::OrtRunner>(model_path.string(), options);
    description = "ONNX Runtime + OpenVINO EP " + runner->openvino_device();
    return runner;
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
  if (config.runtime != "onnxruntime" && config.runtime != "openvino"
      && config.runtime != "openvino-ep") {
    throw std::runtime_error("inference.runtime '" + config.runtime
      + "' is invalid; use onnxruntime, openvino, or openvino-ep");
  }

  if (const auto node = lookup(state_block, root_block, "device")) {
    config.device = to_upper(node->as<std::string>());
  }
  if (config.device.empty()) {
    throw std::runtime_error("inference.device must not be empty; use CPU, GPU, or NPU");
  }

  if (const auto node = lookup(state_block, root_block, "threads")) {
    config.threads = node->as<int>();
  }
  if (config.threads < 0) {
    throw std::runtime_error("inference.threads must be 0 (runtime default) or positive");
  }

  // An absent key uses the default cache; an explicit empty string disables caching.
  if (const auto node = lookup(state_block, root_block, "cache_dir")) {
    config.cache_dir = resolve_optional_path(node->IsNull() ? "" : node->as<std::string>());
  } else {
    config.cache_dir = default_cache_dir();
  }

  config.openvino_ep_library = WBC_DEFAULT_OPENVINO_EP_LIBRARY;
  if (const auto node = lookup(state_block, root_block, "openvino_ep_library")) {
    config.openvino_ep_library = resolve_optional_path(node->as<std::string>());
  }
  return config;
}

std::unique_ptr<isaaclab::Algorithms> make_policy_runner(
  const std::filesystem::path& model_path, const InferenceConfig& config, const std::string& label)
{
  const bool uses_openvino = config.runtime != "onnxruntime";
  if (uses_openvino && !config.cache_dir.empty()) {
    std::error_code ignored;
    std::filesystem::create_directories(config.cache_dir, ignored);
  }

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

  spdlog::info("{}: policy inference on {} (threads {}, cache {}), loaded in {:.0f} ms",
    label, description, config.threads > 0 ? std::to_string(config.threads) : "default",
    uses_openvino && !config.cache_dir.empty() ? config.cache_dir : "off", load_ms);
  return runner;
}

}  // namespace wbc_deploy
