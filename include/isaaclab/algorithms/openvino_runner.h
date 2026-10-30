#pragma once

#include "isaaclab/algorithms/algorithms.h"

#include <openvino/openvino.hpp>
#include <openvino/runtime/intel_npu/properties.hpp>

#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace isaaclab
{

struct OpenVinoRunnerOptions
{
    std::string device = "CPU";  // CPU, GPU, GPU.N or NPU
    int threads = 0;             // CPU inference threads; 0 keeps the OpenVINO default
    std::string cache_dir;       // compiled-model cache; empty disables it
    bool npu_turbo = false;      // NPU only: keep the NPU at maximum frequency
};

// Runs the policy directly on OpenVINO Runtime. The model is compiled and run once in the
// constructor, so construct it during state initialization, never in the control loop.
class OpenVinoRunner : public Algorithms
{
public:
    OpenVinoRunner(const std::string& model_path, const OpenVinoRunnerOptions& options)
    {
        device_ = resolve_inference_device(options.device, core_.get_available_devices());

        auto model = core_.read_model(model_path);
        // NPU compiles static shapes only, so a dynamic batch dimension is pinned to 1.
        std::map<ov::Output<ov::Node>, ov::PartialShape> static_shapes;
        for (const auto& input : model->inputs()) {
            auto shape = input.get_partial_shape();
            if (shape.is_static()) {
                continue;
            }
            if (shape.rank().is_static() && shape.size() > 0 && shape[0].is_dynamic()) {
                shape[0] = 1;
            }
            if (shape.is_dynamic()) {
                throw std::runtime_error("OpenVINO policy input " + input.get_any_name()
                    + " has dynamic shape " + shape.to_string()
                    + "; only the batch dimension may be dynamic");
            }
            static_shapes[input] = shape;
        }
        if (!static_shapes.empty()) {
            model->reshape(static_shapes);
        }

        ov::AnyMap config{ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)};
        if (!options.cache_dir.empty()) {
            config.emplace(ov::cache_dir(options.cache_dir));
        }
        if (options.threads > 0 && device_ == "CPU") {
            config.emplace(ov::inference_num_threads(options.threads));
        }
        if (options.npu_turbo) {
            config.emplace(ov::intel_npu::turbo(true));
        }
        compiled_model_ = core_.compile_model(model, device_, config);
        request_ = compiled_model_.create_infer_request();

        for (const auto& input : compiled_model_.inputs()) {
            if (input.get_element_type() != ov::element::f32) {
                throw std::runtime_error("OpenVINO policy input " + input.get_any_name()
                    + " must be float32");
            }
            inputs_.push_back({input.get_any_name(), input.get_shape()});
        }
        const auto& output = compiled_model_.output(0);
        const ov::Shape output_shape = output.get_shape();
        if (output.get_element_type() != ov::element::f32 || output_shape.size() != 2) {
            throw std::runtime_error("OpenVINO policy output must be a rank-2 float32 tensor");
        }
        action.resize(output_shape[1]);

        // Device plugins initialize lazily; pay that cost here instead of in the first control step.
        for (size_t i = 0; i < inputs_.size(); ++i) {
            ov::Tensor zeros(ov::element::f32, inputs_[i].shape);
            std::memset(zeros.data(), 0, zeros.get_byte_size());
            request_.set_input_tensor(i, zeros);
        }
        request_.infer();
    }

    std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs) override
    {
        for (size_t i = 0; i < inputs_.size(); ++i) {
            const auto& input = inputs_[i];
            const auto found = obs.find(input.name);
            if (found == obs.end()) {
                throw std::runtime_error("Input name " + input.name + " not found in observations.");
            }
            if (found->second.size() != ov::shape_size(input.shape)) {
                throw std::runtime_error("Input " + input.name + " has "
                    + std::to_string(found->second.size()) + " values; the model expects "
                    + std::to_string(ov::shape_size(input.shape)));
            }
            request_.set_input_tensor(i, ov::Tensor(ov::element::f32, input.shape, found->second.data()));
        }
        request_.infer();

        const ov::Tensor output = request_.get_output_tensor(0);
        std::lock_guard<std::mutex> lock(act_mtx_);
        std::memcpy(action.data(), output.data<const float>(), action.size() * sizeof(float));
        return action;
    }

    const std::string& device() const { return device_; }

    std::string device_full_name() const
    {
        return core_.get_property(device_, ov::device::full_name);
    }

    std::string inference_precision() const
    {
        try {
            return compiled_model_.get_property(ov::hint::inference_precision).get_type_name();
        } catch (const std::exception&) {
        }
        // NPU models imported from the cache do not report it. No precision hint is set, so the
        // device default applies.
        try {
            return core_.get_property(device_, ov::hint::inference_precision).get_type_name();
        } catch (const std::exception&) {
            return "unknown";
        }
    }

private:
    struct Input
    {
        std::string name;
        ov::Shape shape;
    };

    ov::Core core_;
    std::string device_;
    ov::CompiledModel compiled_model_;
    ov::InferRequest request_;
    std::vector<Input> inputs_;
};

}  // namespace isaaclab
