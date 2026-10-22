#pragma once

#include "State_WbcTracking.h"

class State_WbcReady : public State_WbcTracking
{
public:
  State_WbcReady(int state, std::string state_string = "WbcReady")
  : State_WbcTracking(state, state_string, true)
  {}
};

REGISTER_FSM(State_WbcReady)
