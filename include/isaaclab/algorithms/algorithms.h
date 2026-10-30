// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "onnxruntime_cxx_api.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace isaaclab
{

class Algorithms
{
public:
    virtual ~Algorithms() = default;
    virtual std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs) = 0;

    std::vector<float> get_action()
    {
        std::lock_guard<std::mutex> lock(act_mtx_);
        return action;
    }
    
    std::vector<float> action;
protected:
    std::mutex act_mtx_;
};

// OpenVINO may enumerate the first GPU as either GPU or GPU.0.
inline std::string resolve_inference_device(
    const std::string& requested, const std::vector<std::string>& available)
{
    if (std::find(available.begin(), available.end(), requested) != available.end()) {
        return requested;
    }
    if (requested == "GPU"
        && std::find(available.begin(), available.end(), "GPU.0") != available.end()) {
        return "GPU.0";
    }
    if (requested == "GPU.0"
        && std::find(available.begin(), available.end(), "GPU") != available.end()) {
        return "GPU";
    }
    std::string message = "Requested inference device " + requested + " is unavailable; available devices:";
    for (const auto& name : available) {
        message += " " + name;
    }
    if (available.empty()) {
        message += " (none)";
    }
    throw std::runtime_error(message);
}

struct OrtRunnerOptions
{
    int threads = 0;                  // intra-op threads; 0 keeps the ONNX Runtime default
    std::string openvino_ep_library;  // non-empty: run through this OpenVINO EP plugin
    std::string openvino_device = "CPU";
    std::string cache_dir;            // OpenVINO EP model cache; empty disables it
    bool npu_turbo = false;           // NPU only: keep the NPU at maximum frequency
};

class OrtRunner : public Algorithms
{
public:
    OrtRunner(std::string model_path) : OrtRunner(model_path, OrtRunnerOptions{}) {}

    OrtRunner(const std::string& model_path, const OrtRunnerOptions& options)
    {
        // Init Model
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "onnx_model");
        session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
        if (options.threads > 0) {
            session_options.SetIntraOpNumThreads(options.threads);
        }
        if (!options.openvino_ep_library.empty()) {
            append_openvino_ep(options);
        }

        session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);

        for (size_t i = 0; i < session->GetInputCount(); ++i) {
            Ort::TypeInfo input_type = session->GetInputTypeInfo(i);
            input_shapes.push_back(input_type.GetTensorTypeAndShapeInfo().GetShape());
            auto input_name = session->GetInputNameAllocated(i, allocator);
            input_names.push_back(input_name.release());
        }

        for (const auto& shape : input_shapes) {
            size_t size = 1;
            for (const auto& dim : shape) {
                size *= dim;
            }
            input_sizes.push_back(size);
        }

        // Get output shape
        Ort::TypeInfo output_type = session->GetOutputTypeInfo(0);
        output_shape = output_type.GetTensorTypeAndShapeInfo().GetShape();
        auto output_name = session->GetOutputNameAllocated(0, allocator);
        output_names.push_back(output_name.release());

        action.resize(output_shape[1]);

        if (!openvino_device_.empty()) {
            // Device plugins initialize lazily; pay that cost here instead of in the first control step.
            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
            std::vector<std::vector<float>> zeros;
            std::vector<Ort::Value> input_tensors;
            for (size_t i = 0; i < input_names.size(); ++i) {
                zeros.emplace_back(input_sizes[i], 0.0f);
                input_tensors.push_back(Ort::Value::CreateTensor<float>(memory_info, zeros.back().data(),
                    input_sizes[i], input_shapes[i].data(), input_shapes[i].size()));
            }
            session->Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(),
                input_tensors.size(), output_names.data(), 1);
        }
    }

    std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs)
    {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        // make sure all input names are in obs
        for (const auto& name : input_names) {
            if (obs.find(name) == obs.end()) {
                throw std::runtime_error("Input name " + std::string(name) + " not found in observations.");
            }
        }

        // Create input tensors
        std::vector<Ort::Value> input_tensors;
        for(int i(0); i<input_names.size(); ++i)
        {
            const std::string name_str(input_names[i]);
            auto& input_data = obs.at(name_str);
            auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_sizes[i], input_shapes[i].data(), input_shapes[i].size());
            input_tensors.push_back(std::move(input_tensor));
        }

        // Run the model
        auto output_tensor = session->Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_tensors.size(), output_names.data(), 1);

        // Copy output data
        auto floatarr = output_tensor.front().GetTensorMutableData<float>();
        std::lock_guard<std::mutex> lock(act_mtx_);
        std::memcpy(action.data(), floatarr, output_shape[1] * sizeof(float));
        return action;
    }

    // OpenVINO device the EP runs on; empty for the default ONNX Runtime CPU provider.
    const std::string& openvino_device() const { return openvino_device_; }

private:
    static std::string json_escape(const std::string& text)
    {
        std::string escaped;
        for (const char c : text) {
            if (c == '"' || c == '\\') {
                escaped += '\\';
            }
            escaped += c;
        }
        return escaped;
    }

    void append_openvino_ep(const OrtRunnerOptions& options)
    {
#if ORT_API_VERSION >= 23
        const auto is_openvino = [](const Ort::ConstEpDevice& device) {
            return std::strcmp(device.EpName(), "OpenVINOExecutionProvider") == 0;
        };
        // All OrtRunner instances share one ORT environment, so the plugin is registered once.
        auto devices = env.GetEpDevices();
        if (std::none_of(devices.begin(), devices.end(), is_openvino)) {
            env.RegisterExecutionProviderLibrary(
                "OpenVINOExecutionProvider", options.openvino_ep_library);
            devices = env.GetEpDevices();
        }

        std::vector<Ort::ConstEpDevice> openvino_devices;
        std::vector<std::string> names;
        for (const auto& device : devices) {
            const char* ov_device = is_openvino(device) ? device.EpMetadata().GetValue("ov_device") : nullptr;
            if (ov_device) {
                openvino_devices.push_back(device);
                names.emplace_back(ov_device);
            }
        }
        openvino_device_ = resolve_inference_device(options.openvino_device, names);
        std::vector<Ort::ConstEpDevice> selected;
        for (size_t i = 0; i < names.size(); ++i) {
            if (names[i] == openvino_device_) {
                selected.push_back(openvino_devices[i]);
            }
        }

        // OpenVINO EP 1.8 ignores the cache_dir and num_of_threads provider options; the
        // OpenVINO device plugin reads them from load_config instead.
        std::string device_config;
        if (!options.cache_dir.empty()) {
            device_config += "\"CACHE_DIR\":\"" + json_escape(options.cache_dir) + "\"";
        }
        if (options.threads > 0 && openvino_device_ == "CPU") {
            device_config += std::string(device_config.empty() ? "" : ",")
                + "\"INFERENCE_NUM_THREADS\":\"" + std::to_string(options.threads) + "\"";
        }
        if (options.npu_turbo) {
            device_config += std::string(device_config.empty() ? "" : ",") + "\"NPU_TURBO\":\"YES\"";
        }
        std::unordered_map<std::string, std::string> provider_options;
        if (!device_config.empty()) {
            provider_options["load_config"] = "{\"" + openvino_device_ + "\":{" + device_config + "}}";
        }
        if (openvino_device_ != "CPU") {
            // Fail at load time instead of silently running unsupported nodes on the CPU.
            session_options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
        }
        session_options.AppendExecutionProvider_V2(env, selected, provider_options);
#else
        (void)options;
        throw std::runtime_error(
            "The OpenVINO EP plugin requires ONNX Runtime 1.23 or newer");
#endif
    }

    std::string openvino_device_;
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<const char*> input_names;
    std::vector<const char*> output_names;

    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<int64_t> input_sizes;
    std::vector<int64_t> output_shape;
};
};