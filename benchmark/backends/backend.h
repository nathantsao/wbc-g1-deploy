#pragma once

#include "model.h"
#include "options.h"

#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace policy_benchmark {

struct BackendInfo {
  std::string device;
  std::string device_name;
  std::string execution_provider;
  // Extra key=value report lines, such as the runtime version and settings that change results.
  std::vector<std::pair<std::string, std::string>> details;
};

struct PrecisionInfo {
  std::string precision;  // e.g. "f16"; empty if the runtime does not report one
  std::string source;     // where the value comes from
};

class InferenceBackend {
public:
  virtual ~InferenceBackend() = default;
  virtual std::vector<float> infer(ObservationMap observations) = 0;
  // The precision the device computes in. It may load a device plugin, so callers query it after
  // the measured calls.
  virtual PrecisionInfo inference_precision() const { return {}; }
  BackendInfo info;
};

std::unique_ptr<InferenceBackend> make_backend(const Options& options, const ModelInfo& model);
void list_devices(const Options& options, std::ostream& out);

std::unique_ptr<InferenceBackend> make_onnx_backend(const Options&, const ModelInfo&);
std::unique_ptr<InferenceBackend> make_openvino_backend(const Options&, const ModelInfo&);
std::unique_ptr<InferenceBackend> make_openvino_ep_backend(const Options&, const ModelInfo&);
void list_onnx_devices(const Options&, std::ostream&);
void list_openvino_devices(std::ostream&);
void list_openvino_ep_devices(const Options&, std::ostream&);

// OpenVINO may enumerate the first GPU as either GPU or GPU.0.
std::string resolve_device(const std::string& requested, const std::vector<std::string>& available);

}  // namespace policy_benchmark
