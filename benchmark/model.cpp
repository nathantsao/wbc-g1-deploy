#include "model.h"

#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <onnxruntime_cxx_api.h>

namespace policy_benchmark {

std::string shape_string(const std::vector<int64_t>& shape) {
  std::ostringstream out;
  out << '[';
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) out << ',';
    out << shape[i];
  }
  out << ']';
  return out.str();
}

ModelInfo inspect_model(const std::string& model_path) {
  if (!std::ifstream(model_path).good()) {
    throw std::runtime_error("Cannot open model: " + model_path);
  }
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx_policy_benchmark_metadata");
  Ort::SessionOptions session_options;
  session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
  Ort::Session session(env, model_path.c_str(), session_options);
  Ort::AllocatorWithDefaultOptions allocator;
  ModelInfo model;
  for (size_t i = 0; i < session.GetInputCount(); ++i) {
    const auto name = session.GetInputNameAllocated(i, allocator);
    const auto type_info = session.GetInputTypeInfo(i);
    const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    InputSpec spec;
    spec.name = name.get();
    spec.shape = tensor_info.GetShape();
    if (tensor_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      throw std::runtime_error("Benchmark requires float32 input: " + spec.name);
    }
    for (const int64_t dimension : spec.shape) {
      if (dimension <= 0) {
        throw std::runtime_error(
            "OrtRunner requires static positive input dimensions; " + spec.name +
            " has shape " + shape_string(spec.shape));
      }
      const auto dim = static_cast<size_t>(dimension);
      if (spec.element_count > std::numeric_limits<size_t>::max() / dim) {
        throw std::runtime_error("Input tensor is too large: " + spec.name);
      }
      spec.element_count *= dim;
    }
    model.inputs.push_back(std::move(spec));
  }
  if (model.inputs.empty()) throw std::runtime_error("ONNX model has no inputs");
  if (session.GetOutputCount() == 0) throw std::runtime_error("ONNX model has no outputs");
  const auto output_type_info = session.GetOutputTypeInfo(0);
  const auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
  model.output_shape = output_tensor_info.GetShape();
  if (output_tensor_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    throw std::runtime_error("Benchmark requires float32 output");
  }
  if (model.output_shape.size() != 2 || model.output_shape[1] <= 0) {
    throw std::runtime_error("OrtRunner expects a rank-2 output with a static action dimension");
  }
  return model;
}

ObservationMap make_observations(const ModelInfo& model, uint32_t seed) {
  ObservationMap observations;
  std::mt19937 generator(seed);
  std::uniform_real_distribution<float> input_distribution(-1.0f, 1.0f);
  for (const auto& input : model.inputs) {
    auto& values = observations[input.name];
    values.resize(input.element_count);
    for (float& value : values) value = input_distribution(generator);
  }
  return observations;
}

}  // namespace policy_benchmark
