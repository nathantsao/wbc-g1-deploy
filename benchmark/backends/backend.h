#pragma once

#include "model.h"
#include "options.h"

#include <iosfwd>
#include <memory>

namespace policy_benchmark {

struct BackendInfo {
  std::string device;
  std::string device_name;
  std::string execution_provider;
};

class InferenceBackend {
public:
  virtual ~InferenceBackend() = default;
  virtual std::vector<float> infer(ObservationMap observations) = 0;
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
