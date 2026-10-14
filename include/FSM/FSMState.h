#pragma once

#include "Types.h"
#include "param.h"
#include "FSM/BaseState.h"
#include "isaaclab/devices/keyboard/keyboard.h"
#include "unitree_joystick_dsl.hpp"

class FSMState : public BaseState
{
public:
    FSMState(int state, std::string state_string) 
    : BaseState(state, state_string) 
    {
        spdlog::info("Initializing State_{} ...", state_string);

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