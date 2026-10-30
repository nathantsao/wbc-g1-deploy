#include "backend_ort.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <utility>
#include "isaaclab/algorithms/algorithms.h"

namespace policy_benchmark {
namespace {

void require_cuda_provider() {
  const auto providers = Ort::GetAvailableProviders();
  if (std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") == providers.end()) {
    throw std::runtime_error(
        "CUDAExecutionProvider is unavailable; configure ONNXRUNTIME_DIR with a CUDA-enabled "
        "ONNX Runtime package and install compatible CUDA/cuDNN");
  }
}

class OnnxRuntimeBackend final : public InferenceBackend {
public:
  explicit OnnxRuntimeBackend(const std::string& model_path)
      : runner_(std::make_unique<isaaclab::OrtRunner>(model_path)) {
    info = {"CPU", "CPU", "ONNX Runtime default CPU provider",
            {{"onnxruntime_version", Ort::GetVersionString()}}};
  }

  std::vector<float> infer(ObservationMap observations) override {
    return runner_->act(std::move(observations));
  }

private:
  std::unique_ptr<isaaclab::OrtRunner> runner_;
};

class CudaBackend final : public OrtSessionBackend {
public:
  CudaBackend(const Options& options, const ModelInfo& model) : OrtSessionBackend(model) {
    require_cuda_provider();
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
    session_options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
    OrtCUDAProviderOptions cuda_options;
    cuda_options.device_id = options.device_id;
    session_options.AppendExecutionProvider_CUDA(cuda_options);
    session_ = std::make_unique<Ort::Session>(env_, options.model_path.c_str(), session_options);
    info = {"GPU." + std::to_string(options.device_id),
            "CUDA device " + std::to_string(options.device_id), "ONNX Runtime CUDA provider",
            {{"onnxruntime_version", Ort::GetVersionString()}}};
  }
};

// OrtRunner has no thread setting, so explicit thread counts use a plain ORT CPU session.
class OnnxCpuThreadsBackend final : public OrtSessionBackend {
public:
  OnnxCpuThreadsBackend(const Options& options, const ModelInfo& model)
      : OrtSessionBackend(model) {
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
    session_options.SetIntraOpNumThreads(options.threads);
    session_ = std::make_unique<Ort::Session>(env_, options.model_path.c_str(), session_options);
    info = {"CPU", "CPU", "ONNX Runtime CPU provider",
            {{"onnxruntime_version", Ort::GetVersionString()}}};
  }
};

}  // namespace

std::unique_ptr<InferenceBackend> make_onnx_backend(const Options& options, const ModelInfo& model) {
  if (options.provider == "cuda") return std::make_unique<CudaBackend>(options, model);
  if (options.threads > 0) return std::make_unique<OnnxCpuThreadsBackend>(options, model);
  return std::make_unique<OnnxRuntimeBackend>(options.model_path);
}

void list_onnx_devices(const Options& options, std::ostream& out) {
  if (options.provider == "cuda") {
    require_cuda_provider();
    out << "provider=CUDAExecutionProvider\n"
        << "device=GPU\n"
        << "hardware_verified=false\n"
        << "note=ORT does not enumerate CUDA hardware here; --device-id selects an index "
           "and session creation validates it\n";
  } else {
    out << "provider=CPUExecutionProvider\ndevice=CPU\n";
  }
}

}  // namespace policy_benchmark
