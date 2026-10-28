#include "backend.h"

#include <algorithm>
#include <stdexcept>

namespace policy_benchmark {

std::unique_ptr<InferenceBackend> make_backend(const Options& options, const ModelInfo& model) {
  if (options.runtime == "onnx") return make_onnx_backend(options, model);
  if (options.runtime == "openvino") return make_openvino_backend(options, model);
  return make_openvino_ep_backend(options, model);
}

void list_devices(const Options& options, std::ostream& out) {
  if (options.runtime == "onnx") {
    list_onnx_devices(options, out);
  } else if (options.runtime == "openvino") {
    list_openvino_devices(out);
  } else {
    list_openvino_ep_devices(options, out);
  }
}

std::string resolve_device(const std::string& requested, const std::vector<std::string>& available) {
  const auto exact = std::find(available.begin(), available.end(), requested);
  if (exact != available.end()) return *exact;
  if (requested == "GPU") {
    const auto first = std::find_if(available.begin(), available.end(),
                                   [](const std::string& name) { return name == "GPU.0"; });
    if (first != available.end()) return *first;
  }
  if (requested == "GPU.0" &&
      std::find(available.begin(), available.end(), "GPU") != available.end()) {
    return "GPU";
  }
  std::string message = "Requested device " + requested + " is unavailable; available devices:";
  for (const auto& name : available) message += " " + name;
  if (available.empty()) message += " (none)";
  throw std::runtime_error(message);
}

}  // namespace policy_benchmark
