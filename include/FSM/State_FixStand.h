// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include "FSM/fsm_pose_utils.h"
#include "wbc_entry_mode.h"

#include <algorithm>
#include <cmath>

class State_FixStand : public FSMState
{
public:
    State_FixStand(int state, std::string state_string = "FixStand")
    : FSMState(state, state_string)
    {
        ts_ = param::config["FSM"]["FixStand"]["ts"].as<std::vector<float>>();
        qs_ = param::config["FSM"]["FixStand"]["qs"].as<std::vector<std::vector<float>>>();
        assert(ts_.size() == qs_.size());
        requireUprightForStandingPolicies();
    }

    void enter()
    {
        static auto kp = param::config["FSM"]["FixStand"]["kp"].as<std::vector<float>>();
        static auto kd = param::config["FSM"]["FixStand"]["kd"].as<std::vector<float>>();
        fsm_pose::applyMotorGains(kp, kd);
        qs_[0] = fsm_pose::captureMotorPositions(kp.size());
        t0_ = static_cast<double>(unitree::common::GetCurrentTimeMillisecond()) * 1e-3;
        wbc_deploy::pendingWbcEntryMode() = wbc_deploy::WbcEntryMode::Standing;
    }

    void run()
    {
        fsm_pose::runJointInterpolation(t0_, ts_, qs_);
    }

private:
    static constexpr double kMaxPolicyEntryTiltDeg = 30.0;

    // FixStand does not balance, so the robot falls soon after it is stood up. A standing
    // policy started on a fallen robot whips the joints, so block the WBC tracking and
    // velocity (RLBase) policies until the robot is upright.
    void requireUprightForStandingPolicies()
    {
        const YAML::Node fsms = param::config["FSM"]["_"];
        for (auto& [check, target_id] : registered_checks) {
            const std::string target = FSMStringMap.left.at(target_id);
            const YAML::Node target_cfg = fsms[target];
            if (!target_cfg) {
                continue;
            }
            // Same type lookup as CtrlFSM: the state name when no type is given.
            const std::string type = target_cfg["type"].as<std::string>(target);
            if (type != "WbcTracking" && type != "RLBase") {
                continue;
            }
            check = [requested = std::move(check), target]() {
                if (!requested()) {
                    return false;
                }
                const double tilt_deg = baseTiltDeg();
                if (tilt_deg <= kMaxPolicyEntryTiltDeg) {
                    return true;
                }
                spdlog::error(
                    "Not entering {}: the robot is tilted {:.0f} deg (limit {:.0f} deg) and has "
                    "probably fallen. Stand it up first, or get up from the floor with Passive -> "
                    "FloorReady -> WBC getup (keyboard 1, 4, 5).",
                    target, tilt_deg, kMaxPolicyEntryTiltDeg);
                return false;
            };
        }
    }

    // Angle between the IMU z axis and the world vertical; NaN for an invalid quaternion.
    static double baseTiltDeg()
    {
        std::lock_guard<std::mutex> lock(lowstate->mutex_);
        const auto& q = lowstate->msg_.imu_state().quaternion();  // w, x, y, z
        const double norm2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
        const double cos_tilt = (q[0] * q[0] + q[3] * q[3] - q[1] * q[1] - q[2] * q[2]) / norm2;
        return std::acos(std::clamp(cos_tilt, -1.0, 1.0)) * 180.0 / M_PI;
    }

    double t0_;
    std::vector<float> ts_;
    std::vector<std::vector<float>> qs_;
};

REGISTER_FSM(State_FixStand)
