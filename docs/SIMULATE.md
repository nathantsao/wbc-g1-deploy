# Simulation Deployment

## Dependencies

- unitree_sdk2

```
sudo apt install libglfw3 libglfw3-dev libfmt-dev libeigen3-dev libboost-all-dev libyaml-cpp-dev libspdlog-dev
```

## Build

```
cmake -S simulate -B simulate/build -DCMAKE_BUILD_TYPE=Release
cmake --build simulate/build
```

In Terminal 1:
```
./simulate/build/unitree_mujoco
```

In Terminal 2:
```
cmake -S . -B build
cmake --build build
```
```
cd build
./wbc_g1_ctrl --network=lo --keyboard
```

Run with `--keyboard` to use the terminal instead of a gamepad: `1` Passive, `2` FixStand, `3` WBC from standing, `4` FloorReady, and `5` WBC + get-up from the floor. In WBC, `A` plays the selected clip, left/right arrows browse clips, and up/down arrows get up or lie down. When using MuJoCo without a physical gamepad, set `use_joystick: 0` in `simulate/config.yaml`; the deployment process generates the controls itself.

When starting, press `4` then `5` to get the robot to stand up.