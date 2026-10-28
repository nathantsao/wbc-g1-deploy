#include "backend_ort.h"

#include <stdexcept>

namespace policy_benchmark {

OrtSessionBackend::OrtSessionBackend(const ModelInfo& model)
    : env_(ORT_LOGGING_LEVEL_WARNING, "onnx_policy_benchmark"), inputs_(model.inputs) {}

std::vector<float> OrtSessionBackend::infer(ObservationMap observations) {
  Ort::MemoryInfo memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<const char*> input_names;
  std::vector<Ort::Value> input_tensors;
  input_names.reserve(inputs_.size());
  input_tensors.reserve(inputs_.size());
  for (const auto& input : inputs_) {
    const auto found = observations.find(input.name);
    if (found == observations.end()) {
      throw std::runtime_error("ONNX Runtime input not found: " + input.name);
    }
    if (found->second.size() != input.element_count) {
      throw std::runtime_error("ONNX Runtime input size mismatch: " + input.name);
    }
    input_names.push_back(input.name.c_str());
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info, found->second.data(), found->second.size(),
        input.shape.data(), input.shape.size()));
  }
  Ort::AllocatorWithDefaultOptions allocator;
  auto output_name = session_->GetOutputNameAllocated(0, allocator);
  const char* output_names[] = {output_name.get()};
  auto outputs = session_->Run(
      Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(),
      input_tensors.size(), output_names, 1);
  const float* output_data = outputs[0].GetTensorData<float>();
  const size_t output_size = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
  return std::vector<float>(output_data, output_data + output_size);
}

}  // namespace policy_benchmark
