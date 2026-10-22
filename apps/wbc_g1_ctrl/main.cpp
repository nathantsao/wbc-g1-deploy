#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_FloorReady.h"
#include "FSM/State_WbcReady.h"
#include "FSM/State_RLBase.h"
#include "State_WbcTracking.h"

#include <cmath>

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = std::make_shared<Keyboard>();

void init_fsm_state()
{
    auto lowcmd_sub = std::make_shared<unitree::robot::g1::subscription::LowCmd>();
    usleep(0.2 * 1e6);
    if (!lowcmd_sub->isTimeout()) {
        spdlog::critical(
          "The other process is using the lowcmd channel, please close it first.");
        unitree::robot::go2::shutdown();
    }
    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();
    spdlog::info("Waiting for connection to robot...");
    FSMState::lowstate->wait_for_connection();
    spdlog::info("Connected to robot.");
}

void start_sequence()
{
    constexpr size_t policy_joint_count = 29;
    const auto policy_observations_available = [] {
        if (FSMState::lowstate->isTimeout()) {
            return false;
        }

        std::lock_guard<std::mutex> lock(FSMState::lowstate->mutex_);
        const auto& imu = FSMState::lowstate->msg_.imu_state();
        const auto& motors = FSMState::lowstate->msg_.motor_state();
        if (imu.gyroscope().size() < 3 || imu.quaternion().size() < 4
            || motors.size() < policy_joint_count) {
            return false;
        }

        float quaternion_norm_squared = 0.0f;
        for (size_t i = 0; i < 3; ++i) {
            if (!std::isfinite(imu.gyroscope()[i])) {
                return false;
            }
        }
        for (size_t i = 0; i < 4; ++i) {
            const float component = imu.quaternion()[i];
            if (!std::isfinite(component)) {
                return false;
            }
            quaternion_norm_squared += component * component;
        }
        if (quaternion_norm_squared < 0.25f || quaternion_norm_squared > 2.25f) {
            return false;
        }

        // Check motor encoders
        for (size_t i = 0; i < policy_joint_count; ++i) {
            if (!std::isfinite(motors[i].q()) || !std::isfinite(motors[i].dq())) {
                return false;
            }
        }
        return true;
    };

    spdlog::info("Waiting for valid policy observations from the robot...");
    while (!policy_observations_available()) {
        usleep(10'000);
    }
    spdlog::info("Received IMU and 29-joint policy observations.");

    if (FSMState::use_keyboard) {
        spdlog::info("Skipping joystick startup check in keyboard mode.");
        return;
    }

    std::cout << "Startup safety: release, then press [A] on the joystick to continue.\n";
    bool a_released = false;
    while (true) {
        FSMState::lowstate->update();
        const auto& a = FSMState::lowstate->joystick.A;
        if (!a.pressed) {
            a_released = true;
        } else if (a_released && a.on_pressed) {
            break;
        }
        usleep(10'000);
    }
    std::cout << "Pressed [A] (1/2)\n";

    a_released = false;
    while (true) {
        FSMState::lowstate->update();
        const auto& a = FSMState::lowstate->joystick.A;
        if (!a.pressed) {
            a_released = true;
        } else if (a_released && a.on_pressed) {
            break;
        }
        usleep(10'000);
    }
    std::cout << "Pressed [A] (2/2)\n";
    spdlog::info("Joystick startup check passed.");
}

int main(int argc, char** argv)
{
    auto vm = param::helper(argc, argv);
    FSMState::use_keyboard = vm.count("keyboard") > 0;

    std::cout << " --- wbc_g1_deploy (WBC tracking) --- \n";

    unitree::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());

    init_fsm_state();

    FSMState::lowcmd->msg_.mode_machine() = 5;
    if (!FSMState::lowcmd->check_mode_machine(FSMState::lowstate)) {
        spdlog::critical("Unmatched robot type (expected 29-DoF G1).");
        exit(-1);
    }

    start_sequence();

    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    fsm->start();

    if (FSMState::use_keyboard) {
        std::cout << "Keyboard: 1 Passive, 2 FixStand, 3 WBC, 4 FloorReady, 5 WBC + getup, 6 Velocity\n";
        std::cout << "In WBC: A plays clip, arrows select (left/right), liedown (down), or getup (up).\n";
        std::cout << "In Velocity: w/s forward/back, a/d left/right, q/e yaw left/right.\n";
    } else {
        std::cout << "Standing start: [L2 + D-pad Up] FixStand, then [R2 + A] WBC tracking.\n";
        std::cout << "Floor start: [L2 + D-pad Down] floor pose, then [R2 + Y] getup + WBC.\n";
        std::cout << "In WBC: [R2 + D-pad L/R] select clip, [A] play, [D-pad down/up] liedown/getup.\n";
        std::cout << "Velocity: [R2 + B] to activate.\n";
        std::cout << "In Velocity: left stick forward/strafe, right stick horizontal yaw.\n";
        std::cout << "Press [L2 + B] for Passive.\n";
    }


    while (true) {
        sleep(1);
    }
    return 0;
}
