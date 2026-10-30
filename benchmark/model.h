#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace policy_benchmark {

using ObservationMap = std::unordered_map<std::string, std::vector<float>>;

struct InputSpec {
  std::string name;
  std::vector<int64_t> shape;
  size_t element_count = 1;
};

struct ModelInfo {
  std::vector<InputSpec> inputs;
  std::vector<int64_t> output_shape;
};

std::string shape_string(const std::vector<int64_t>& shape);
ModelInfo inspect_model(const std::string& model_path);
ObservationMap make_observations(const ModelInfo& model, uint32_t seed);

}  // namespace policy_benchmark
