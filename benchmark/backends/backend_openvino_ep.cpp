#include "backend_ort.h"

#include <algorithm>
#include <cstring>
#include <ostream>
#include <stdexcept>

namespace policy_benchmark {
#if ORT_API_VERSION >= 23
namespace {

std::vector<Ort::ConstEpDevice> openvino_devices(Ort::Env& env, const Options& options) {
  const auto is_openvino = [](const Ort::ConstEpDevice& device) {
    return std::strcmp(device.EpName(), "OpenVINOExecutionProvider") == 0;
  };
  // ORT shares one environment while any Ort::Env exists. A backend created while another one
  // still exists must not register the plugin again.
  const auto registered = env.GetEpDevices();
  if (std::none_of(registered.begin(), registered.end(), is_openvino)) {
    env.RegisterExecutionProviderLibrary(
        "OpenVINOExecutionProvider", options.provider_library_path.c_str());
  }
  std::vector<Ort::ConstEpDevice> devices;
  for (const auto& device : env.GetEpDevices()) {
    if (is_openvino(device) && device.EpMetadata().GetValue("ov_device") != nullptr) {
      devices.push_back(device);
    }
  }
  if (devices.empty()) {
    throw std::runtime_error(
        "OpenVINO EP did not expose any devices; check the plugin, installation, and drivers");
  }
  return devices;
}

class OpenVinoExecutionProviderBackend final : public OrtSessionBackend {
public:
  OpenVinoExecutionProviderBackend(const Options& options, const ModelInfo& model)
      : OrtSessionBackend(model) {
    const auto devices = openvino_devices(env_, options);
    std::vector<std::string> names;
    for (const auto& device : devices) {
      names.emplace_back(device.EpMetadata().GetValue("ov_device"));
    }
    const auto resolved = resolve_device(options.device, names);
    std::vector<Ort::ConstEpDevice> selected;
    for (const auto& device : devices) {
      if (resolved == device.EpMetadata().GetValue("ov_device")) selected.push_back(device);
    }
    if (selected.size() != 1) {
      throw std::runtime_error("OpenVINO EP device selection is ambiguous: " + resolved);
    }
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
    if (resolved != "CPU") {
      session_options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
    }
    std::unordered_map<std::string, std::string> provider_options;
    if (options.threads > 0) {
      session_options.SetIntraOpNumThreads(options.threads);
      // OpenVINO EP 1.8 ignores num_of_threads, so the thread count goes through load_config.
      provider_options["load_config"] = "{\"CPU\":{\"INFERENCE_NUM_THREADS\":\"" +
                                        std::to_string(options.threads) + "\"}}";
    }
    session_options.AppendExecutionProvider_V2(env_, selected, provider_options);
    session_ = std::make_unique<Ort::Session>(env_, options.model_path.c_str(), session_options);
    info = {resolved, resolved, "ONNX Runtime OpenVINO EP plugin"};
  }
};

}  // namespace

std::unique_ptr<InferenceBackend> make_openvino_ep_backend(const Options& options, const ModelInfo& model) {
  return std::make_unique<OpenVinoExecutionProviderBackend>(options, model);
}

void list_openvino_ep_devices(const Options& options, std::ostream& out) {
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx_policy_benchmark_devices");
  for (const auto& device : openvino_devices(env, options)) {
    out << "provider=" << device.EpName()
        << " device=" << device.EpMetadata().GetValue("ov_device") << '\n';
  }
}
#else
namespace {
[[noreturn]] void unsupported() {
  throw std::runtime_error(
      "The OpenVINO EP plugin requires ONNX Runtime 1.23 or newer; "
      "reconfigure the benchmark with a newer ONNX Runtime");
}
}  // namespace

std::unique_ptr<InferenceBackend> make_openvino_ep_backend(const Options&, const ModelInfo&) {
  unsupported();
}

void list_openvino_ep_devices(const Options&, std::ostream&) {
  unsupported();
}
#endif

}  // namespace policy_benchmark
