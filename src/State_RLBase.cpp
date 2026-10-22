#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

namespace
{
std::array<float, 3> keyboard_velocity_axes(const std::string& key)
{
    static const std::unordered_map<std::string, std::array<float, 3>> key_commands = {
        {"w", {1.0f, 0.0f, 0.0f}},
        {"s", {-1.0f, 0.0f, 0.0f}},
        {"a", {0.0f, 1.0f, 0.0f}},
        {"d", {0.0f, -1.0f, 0.0f}},
        {"q", {0.0f, 0.0f, 1.0f}},
        {"e", {0.0f, 0.0f, -1.0f}}
    };
    const auto it = key_commands.find(key);
    return it == key_commands.end() ? std::array<float, 3>{} : it->second;
}
}

namespace isaaclab
{
REGISTER_OBSERVATION(velocity_commands)
{
    std::array<float, 3> axes{};
    if (FSMState::use_keyboard)
    {
        if (!FSMState::keyboard) {
            throw std::runtime_error("velocity_commands requires a keyboard in keyboard mode.");
        }
        axes = keyboard_velocity_axes(FSMState::keyboard->key());
    }
    else
    {
        auto* joystick = env->robot->data.joystick;
        if (!joystick) {
            throw std::runtime_error("velocity_commands requires a joystick in joystick mode.");
        }
        // Unitree stick right is negative body-y / negative yaw.
        axes = {joystick->ly(), -joystick->lx(), -joystick->rx()};
    }

    const auto command_name = params["command_name"].as<std::string>("base_velocity");
    const auto ranges = env->cfg["commands"][command_name]["ranges"];
    const std::array<std::string, 3> range_names = {"lin_vel_x", "lin_vel_y", "ang_vel_z"};
    std::vector<float> cmd(3);
    for (size_t i = 0; i < axes.size(); ++i)
    {
        const auto range = ranges[range_names[i]].as<std::vector<float>>();
        if (range.size() != 2 || !std::isfinite(range[0]) || !std::isfinite(range[1])
            || range[0] > 0.0f || range[1] < 0.0f || range[0] > range[1]) {
            throw std::runtime_error("Invalid velocity_commands range: " + range_names[i]);
        }
        if (!std::isfinite(axes[i])) {
            throw std::runtime_error("Non-finite velocity_commands input: " + range_names[i]);
        }
        const float axis = std::clamp(axes[i], -1.0f, 1.0f);
        cmd[i] = axis >= 0.0f ? axis * range[1] : -axis * range[0];
    }
    return cmd;
}

REGISTER_OBSERVATION(gait_phase)
{
    const float period = params["period"].as<float>();
    if (!std::isfinite(period) || period <= 0.0f) {
        throw std::runtime_error("gait_phase period must be finite and positive.");
    }
    env->global_phase = std::fmod(env->global_phase + env->step_dt / period, 1.0f);

    const auto cmd = velocity_commands(env, params);
    const float cmd_norm = std::sqrt(
        cmd[0] * cmd[0] + cmd[1] * cmd[1] + cmd[2] * cmd[2]);
    if (cmd_norm < 0.1f) {
        return {0.0f, 0.0f};
    }

    constexpr float two_pi = 6.2831853071795864769f;
    return {std::sin(two_pi * env->global_phase), std::cos(two_pi * env->global_phase)};
}

}

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::resolve_policy_dir(cfg["policy_dir"].as<std::string>());


    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "config.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "params" / "policy.onnx");

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLBase::run()
{
    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}