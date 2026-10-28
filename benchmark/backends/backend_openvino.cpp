#include "backend.h"

#include <ostream>
#include <stdexcept>
#include <openvino/openvino.hpp>

namespace policy_benchmark {
namespace {

class OpenVinoBackend final : public InferenceBackend {
public:
  OpenVinoBackend(const Options& options, const ModelInfo& model) : inputs_(model.inputs) {
    const std::string device = resolve_device(options.device, core_.get_available_devices());
    ov::AnyMap config;
    if (options.threads > 0) config.insert(ov::inference_num_threads(options.threads));
    compiled_model_ = std::make_unique<ov::CompiledModel>(
        core_.compile_model(core_.read_model(options.model_path), device, config));
    request_ = std::make_unique<ov::InferRequest>(compiled_model_->create_infer_request());
    const auto execution_devices = compiled_model_->get_property(ov::execution_devices);
    if (execution_devices.size() != 1) {
      throw std::runtime_error("OpenVINO benchmark requires exactly one execution device");
    }
    info = {execution_devices.front(), core_.get_property(device, ov::device::full_name),
            "OpenVINO " + options.device + " plugin"};
  }

  std::vector<float> infer(ObservationMap observations) override {
    std::vector<ov::Tensor> input_tensors;
    input_tensors.reserve(inputs_.size());
    for (size_t i = 0; i < inputs_.size(); ++i) {
      const auto& input = inputs_[i];
      const auto found = observations.find(input.name);
      if (found == observations.end()) {
        throw std::runtime_error("OpenVINO input not found: " + input.name);
      }
      if (found->second.size() != input.element_count) {
        throw std::runtime_error("OpenVINO input size mismatch: " + input.name);
      }
      ov::Shape shape;
      shape.reserve(input.shape.size());
      for (const int64_t dimension : input.shape) {
        shape.push_back(static_cast<size_t>(dimension));
      }
      input_tensors.emplace_back(ov::element::f32, shape, found->second.data());
      request_->set_input_tensor(i, input_tensors.back());
    }
    request_->infer();
    const ov::Tensor output = request_->get_output_tensor(0);
    const float* output_data = output.data<const float>();
    return std::vector<float>(output_data, output_data + output.get_size());
  }

private:
  ov::Core core_;
  std::unique_ptr<ov::CompiledModel> compiled_model_;
  std::unique_ptr<ov::InferRequest> request_;
  std::vector<InputSpec> inputs_;
};

}  // namespace

std::unique_ptr<InferenceBackend> make_openvino_backend(const Options& options, const ModelInfo& model) {
  return std::make_unique<OpenVinoBackend>(options, model);
}

void list_openvino_devices(std::ostream& out) {
  ov::Core core;
  const auto devices = core.get_available_devices();
  if (devices.empty()) throw std::runtime_error("OpenVINO did not expose any devices");
  for (const auto& device : devices) {
    out << "device=" << device
        << " name=" << core.get_property(device, ov::device::full_name) << '\n';
  }
}

}  // namespace policy_benchmark
