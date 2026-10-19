#pragma once

#include "Types.h"
#include "param.h"
#include "FSM/BaseState.h"
#include "isaaclab/devices/keyboard/keyboard.h"
#include "unitree_joystick_dsl.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <string>

class FSMState : public BaseState
{
public:
    FSMState(int state, std::string state_string) 
    : BaseState(state, state_string) 
    {
        spdlog::info("Initializing State_{} ...", state_string);

        registered_checks.emplace_back(
            std::make_pair(
                []()->bool{ return hardwareSafetyFault(); },
                FSMStringMap.right.at("Passive")
            )
        );

        auto transitions = param::config["FSM"][state_string]["transitions"];

        if(transitions)
        {
            auto transition_map = transitions.as<std::map<std::string, std::string>>();

            for(auto it = transition_map.begin(); it != transition_map.end(); ++it)
            {
                std::string target_fsm = it->first;
                if(!FSMStringMap.right.count(target_fsm))
                {
                    spdlog::warn("FSM State_'{}' not found in FSMStringMap!", target_fsm);
                    continue;
                }

                int fsm_id = FSMStringMap.right.at(target_fsm);

                std::string condition = it->second;
                unitree::common::dsl::Parser p(condition);
                auto ast = p.Parse();
                auto func = unitree::common::dsl::Compile(*ast);
                registered_checks.emplace_back(
                    std::make_pair(
                        [func]()->bool{ return func(FSMState::lowstate->joystick); },
                        fsm_id
                    )
                );
            }
        }

        // register for all states
        registered_checks.emplace_back(
            std::make_pair(
                []()->bool{ return lowstate->isTimeout(); },
                FSMStringMap.right.at("Passive")
            )
        );
    }

    void pre_run()
    {
        lowstate->update();
        if(keyboard) keyboard->update();
        if(use_keyboard && keyboard) keyboard_control();
    }

    void post_run()
    {
        lowcmd->unlockAndPublish();
    }

    static std::unique_ptr<LowCmd_t> lowcmd;
    static std::shared_ptr<LowState_t> lowstate;
    static std::shared_ptr<Keyboard> keyboard;

    inline static bool use_keyboard = false;

private:
    static bool hardwareSafetyFault()
    {
        constexpr size_t kCheckedMotors = 29;
        constexpr double kMaxJointVelocity = 35.0;
        constexpr int16_t kHighTemperatureEnter = 90;
        constexpr int16_t kHighTemperatureExit = 85;

        if (!lowstate) {
            return false;
        }

        bool velocity_fault = false;
        bool motor_fault = false;
        bool temperature_fault = false;
        std::string fault_reason;
        {
            std::lock_guard<std::mutex> lock(lowstate->mutex_);
            const auto& motors = lowstate->msg_.motor_state();
            const size_t motor_count = std::min(kCheckedMotors, motors.size());

            for (size_t i = 0; i < motor_count; ++i) {
                const auto& motor = motors[i];
                const double velocity = motor.dq();
                const auto temperatures = motor.temperature();
                const int16_t max_temperature = std::max(temperatures[0], temperatures[1]);
                
                // Check if joint velocity exceeds threshold
                if (std::abs(velocity) > kMaxJointVelocity) {
                    velocity_fault = true;
                    if (fault_reason.empty()) {
                        fault_reason = "motor " + std::to_string(i) + " velocity " + std::to_string(velocity) + " rad/s";
                    }
                }

                // Check motor status for errors
                if (motor.motorstate() != 0) {
                    motor_fault = true;
                    if (fault_reason.empty()) {
                        fault_reason = "motor " + std::to_string(i) + " error code " + std::to_string(motor.motorstate());
                    }
                }
                
                // Check motor temperature
                if (motor_high_temp_[i]) {
                    if (max_temperature < kHighTemperatureExit) {
                        motor_high_temp_[i] = false;
                    }
                } else if (max_temperature >= kHighTemperatureEnter) {
                    motor_high_temp_[i] = true;
                }
                if (motor_high_temp_[i]) {
                    temperature_fault = true;
                    if (fault_reason.empty()) {
                        fault_reason = "motor " + std::to_string(i) + " temperature " + std::to_string(max_temperature);
                    }
                }
            }
        }

        const bool fault = velocity_fault || motor_fault || temperature_fault;
        if (fault && !safety_fault_active_) {
            spdlog::critical("Hardware safety fault; transitioning to Passive: {}", fault_reason);
        }
        safety_fault_active_ = fault;
        return fault;
    }

    inline static std::array<bool, 29> motor_high_temp_{};
    inline static bool safety_fault_active_ = false;

    void keyboard_control()
    {
        auto& joystick = lowstate->joystick;
        joystick.LT.smooth = 1.0f;
        joystick.RT.smooth = 1.0f;

        float lt = 0.0f;
        float rt = 0.0f;
        int a = 0;
        int b = 0;
        int y = 0;
        int up = 0;
        int down = 0;
        int left = 0;
        int right = 0;

        if (keyboard->on_pressed)
        {
            const std::string key = keyboard->key();
            if (key == "1") { lt = 1.0f; b = 1; }
            else if (key == "2") { lt = 1.0f; up = 1; }
            else if (key == "3") { rt = 1.0f; a = 1; }
            else if (key == "4") { lt = 1.0f; down = 1; }
            else if (key == "5") { rt = 1.0f; y = 1; }
            else if (key == "a" || key == "A") a = 1;
            else if (key == "up") up = 1;
            else if (key == "down") down = 1;
            else if (key == "left") { rt = 1.0f; left = 1; }
            else if (key == "right") { rt = 1.0f; right = 1; }
        }

        joystick.LT(lt);
        joystick.RT(rt);
        joystick.A(a);
        joystick.B(b);
        joystick.Y(y);
        joystick.up(up);
        joystick.down(down);
        joystick.left(left);
        joystick.right(right);
    }
};