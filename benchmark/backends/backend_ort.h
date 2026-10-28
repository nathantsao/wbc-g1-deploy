#pragma once

#include "backend.h"
#include <onnxruntime_cxx_api.h>

namespace policy_benchmark {

// Shared host-tensor handling for CUDA and the OpenVINO EP.
class OrtSessionBackend : public InferenceBackend {
public:
  explicit OrtSessionBackend(const ModelInfo& model);
  std::vector<float> infer(ObservationMap observations) override;

protected:
  Ort::Env env_;
  std::vector<InputSpec> inputs_;
  std::unique_ptr<Ort::Session> session_;
};

}  // namespace policy_benchmark
