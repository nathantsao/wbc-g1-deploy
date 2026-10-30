# G1 WBC Deploy

Onboard deployment for Unitree G1 whole-body tracking. **One policy, many motion clips** — swap NPZ trajectories at runtime with the joystick.

This stack pairs with [wbc-mjlab](https://github.com/wbc-mjlab/wbc-mjlab), which extends mjlab's original **single-clip** tracking example into a **clip-library** training pipeline (multi-clip datasets, RSI, NPZ conversion). Deploy mirrors that model: one exported ONNX policy plus a `config/clips/` library selected at runtime (`manifest.yaml`).

The controller subscribes to Unitree SDK2 `LowState`, runs the bundled ONNX policy, and publishes motor commands.

## Demo

[![Simulation demo](https://img.youtube.com/vi/CSyczObERIc/maxresdefault.jpg)](https://www.youtube.com/watch?v=CSyczObERIc)

Bundled policy and clip library on loopback (`--network=lo`). [Watch on YouTube](https://www.youtube.com/watch?v=CSyczObERIc).

## Bundled policy

The included `policy.onnx` was trained in [wbc-mjlab](https://github.com/wbc-mjlab/wbc-mjlab) (`Wbc-G1`) mainly on the **LAFAN1 retarget** library, plus a **small subset of BONES-SEED** clips (mostly **back and side flips**). It is a reference controller for the deploy stack — not a universal foundation model.

For motions or skills outside that training mix, **train your own policy** on your clips and drop in the exported artifacts:

```bash
# in wbc-mjlab
uv run wbc-mjlab-train --task Wbc-G1 --dataset lafan
# play exports params/policy.onnx + params/config.yaml → copy into config/policy/
```

See [wbc-mjlab docs](https://wbc-mjlab.github.io/wbc-mjlab/) for datasets, motion conversion, and export. Bundled runtime clips live under `config/clips/` (`manifest.yaml`).

**Standing:** Passive → FixStand (L2 + D-pad Up) → WBC tracking (R2 + A)

**From floor:** Passive → FloorReady (L2 + D-pad Down) → WBC + getup (R2 + Y)

**From Velocity:** Velocity (R2 + A) → WbcReady (select a clip with RT + D-pad left/right) → WBC tracking (R2 + Y, starts the selected clip)

`Wbc_Ready` continuously runs the WBC/karate policy using the final frame of
`pose_clip: getup_01_original` as a frozen reference. Joint positions, torso height,
and gravity direction come from that frame; reference velocities are zero.
It uses the policy's motor gains and PD torque clipping, not the former joint-PD
interpolation. Browsing clips only changes the next clip to play, not the held
reference. Verify balance in simulation before using this on hardware.


## Build and run

### Simulation

```bash
cd simulate
mkdir build && cd build && cmake .. && make -j
./simulate/build/unitree_mujoco
```

### Controller

```bash
scripts/bootstrap_thirdparty.sh
mkdir -p build && cd build && cmake .. && make -j
./wbc_g1_ctrl --network=lo   # or your robot NIC (e.g. eth0)
```

`--network` selects the Unitree DDS interface for `rt/lowstate` / `rt/lowcmd` (domain 0).

On the G1 onboard PC, use `--network=eth0` and set up Ethernet / internet sharing first — see [docs/DEPLOY.md](docs/DEPLOY.md).

## Joystick

| Action | Joystick | Keyboard (Sim-Only) |
|--------|---------|---------------------|
| FixStand (standing) | L2 + D-pad Up | 2 |
| Floor pose (getup frame 0) | L2 + D-pad Down | 4 |
| WBC tracking (standing) | R2 + A (from FixStand) | 3 |
| WBC + getup (from floor) | R2 + Y (from FloorReady) | 5 |
| WbcReady (stable get-up end pose) | R2 + A (from Velocity) | 3 |
| WBC tracking (play selected clip) | R2 + Y (from WbcReady) | 5 |
| Passive | L2 + B | 1 |
| Velocity | R2 + B | 6 |
| Forward / backward (Velocity mode) | Left stick up / down | `w` / `s` |
| Strafe left / right (Velocity mode) | Left stick left / right | `a` / `d` |
| Turn left / right (Velocity mode) | Right stick left / right | `q` / `e` |
| Select clip | RT + D-pad right / left (WbcReady or idle WBC) | Right / Left |
| Play selected clip | A (idle WBC only; WbcReady starts it on transition) | A |
| Liedown | D-pad down (when idle: select mode or after clip finishes; auto-plays) | Down |
| Getup | D-pad up (when down and idle; auto-plays) | Up |

At startup, release and then press the joystick's A button to verify joystick input before the FSM starts. This check is skipped with `--keyboard`.

In Velocity mode, the left stick controls forward/backward and lateral velocity;
the right stick's horizontal axis controls yaw. With `--keyboard`, use W/S for
forward/backward, A/D for left/right, and Q/E for left/right yaw (lowercase keys).
Both input modes use the `velocity_commands` observation and the limits in
`config/policy/velocity/params/config.yaml` under `commands.base_velocity.ranges`.
Neutral sticks or no movement key command zero velocity.

The velocity policy also receives `gait_phase`: sine/cosine of a clock with the
configured `period` (0.6 seconds). Following Unitree's deployment convention,
both values are zero when the velocity-command norm is below 0.1; the clock
continues advancing while stationary and resets when the policy environment resets.

## Dependencies

- [unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2)
- ONNX Runtime 1.30.0 (`scripts/bootstrap_thirdparty.sh`)
- yaml-cpp, Boost, Eigen3, fmt, spdlog, zlib
- Optional, Intel only: OpenVINO 2026.4.1 and the OpenVINO EP plugin 1.8.1
  (`scripts/bootstrap_thirdparty_intel.sh`). See [Inference runtime and device](#inference-runtime-and-device).

Ubuntu packages (typical dev machine):

```bash
sudo apt install -y \
  cmake g++ build-essential \
  libyaml-cpp-dev libboost-all-dev libeigen3-dev \
  libspdlog-dev libfmt-dev zlib1g-dev libglfw3-dev
```

If unitree_sdk2 is installed to a user prefix:

```bash
export UNITREE_SDK_PREFIX=$HOME/.local/unitree_robotics
```

CMake also checks `/opt/unitree_robotics`.

## Config layout

| File | Purpose |
|------|---------|
| `config/config.yaml` | Active `policy_dir`, `clips_dir`, FSM + joystick |
| `config/robot_defaults.yaml` | Real-robot PD gains, `joint_ids_map`, `default_joint_pos`, PD torque clip catalog |
| `config/policy/wbc/params/config.yaml` | Deploy observation/history and action layout |
| `config/policy/wbc/params/policy.onnx` | Policy weights |
| `config/clips/` | Motion NPZ library + `manifest.yaml` |

Set the active policy bundle in `config/config.yaml`:

```yaml
policy_dir: config/policy/wbc
clips_dir: config/clips
clips_manifest: manifest.yaml
```

Per-state FSM keys (`policy_dir`, `clips_dir`, …) override the root values when present.

### PD torque clipping

Before sending motor position targets, WBC can back-solve clipped positions so implied PD torque stays within motor limits:

```yaml
# config/robot_defaults.yaml
pd_torque_clipping:
  enabled: true
  fraction: 0.98          # soft cap = fraction * joint_max_torque
  joint_max_torque: [...] # per policy joint, N·m
```

`Wbc_Tracking.pd_torque_clipping` overrides these defaults when set.

## Inference runtime and device

The `inference:` block in `config/config.yaml` selects how the policy runs. A state can override any key with its own `inference:` block, like `policy_dir`.

```yaml
inference:
  runtime: openvino   # onnxruntime | openvino | openvino-ep
  device: NPU         # CPU | GPU | NPU
```

| Key | Default | Meaning |
|-----|---------|---------|
| `runtime` | `onnxruntime` | `onnxruntime`: ONNX Runtime CPU. `openvino`: OpenVINO Runtime. `openvino-ep`: ONNX Runtime with the OpenVINO Execution Provider (EP) plugin. |
| `device` | `CPU` | `CPU`, `GPU`, or `NPU`. `onnxruntime` supports `CPU` only. |
| `threads` | `0` | CPU inference threads. `0` keeps the runtime default. |
| `cache_dir` | `~/.cache/wbc_g1_deploy/openvino` | Compiled-model cache for `openvino` and `openvino-ep`. `""` turns it off. |
| `openvino_ep_library` | plugin under `thirdparty/` | OpenVINO EP plugin for `openvino-ep`. A relative path starts at the repo root. |
| `latency_log_interval` | `0` | Seconds between inference latency log lines. `0` turns the log off. |

Each state loads its model when the controller starts, never in the control loop, and logs the result:

```
Velocity: policy inference on OpenVINO NPU (Intel(R) AI Boost, f16) (threads default, cache ~/.cache/wbc_g1_deploy/openvino), loaded in 297 ms
```

A wrong setting or a missing device stops the controller at start with an error message. For a missing device, the message lists the available devices.

### Build options

| Runtime | CMake | Install |
|---------|-------|---------|
| `onnxruntime` | default | `scripts/bootstrap_thirdparty.sh` |
| `openvino` | `-DWBC_ENABLE_OPENVINO=ON` | `scripts/bootstrap_thirdparty_intel.sh` |
| `openvino-ep` | default | `scripts/bootstrap_thirdparty_intel.sh` |

```bash
scripts/bootstrap_thirdparty_intel.sh   # add --system-deps for the GPU/NPU drivers (sudo)
cmake -S . -B build -DWBC_ENABLE_OPENVINO=ON
cmake --build build -j
```

- The default build links only ONNX Runtime. Use it on the G1 onboard PC (aarch64).
- `openvino-ep` works without `WBC_ENABLE_OPENVINO`, because the plugin ships its own OpenVINO libraries.
- CMake variables: `OPENVINO_VERSION` (2026.4.1), `OVEP_VERSION` (1.8.1), and `OPENVINO_EP_LIBRARY` (default plugin path).

### Simulator results

Every runtime and device kept the robot standing. Every call finished within the 20 ms policy period.

Test machine: Intel Core Ultra X7 358H, Arc B390 iGPU (driver 26.35.39758.17), NPU driver 1.38.0, `powersave` governor. Ubuntu 26.04, ONNX Runtime 1.30.0, OpenVINO 2026.4.1, OpenVINO EP 1.8.1. Run on 2026-10-30 at commit 2dbb89b.

Test sequence, with `latency_log_interval: 5`:

1. Press `2` (FixStand). The simulator starts with no motor torque, so the robot is already on the floor.
2. Press Backspace in the MuJoCo window to stand the robot up, then press `3` (WBC tracking) right away.
3. Press `a` to play `bow`. Then press Right and `a` to play `karate_move_1`.
4. Press `6` (Velocity). Hold `w` 5 s, `q` 3 s, `s` 3 s, `d` 3 s, then wait 3 s.

The robot counts as fallen if the IMU height drops below 0.45 m or roll or pitch goes above 0.8 rad. The lowest height was 0.52 m, during `karate_move_1`. Each run has about 1,760 Wbc_Tracking calls and 1,000 Velocity calls. p50 is the median and p99 the worst of the 5 s window values.

| Runtime | Device | `threads` | Wbc_Tracking p50 / p99 / max (µs) | Velocity p50 / p99 / max (µs) | Controller CPU (cores) |
|---------|--------|-----------|-----------------------------------|-------------------------------|------------------------|
| `onnxruntime` | CPU | default | 31 / 266 / 610 | 22 / 164 / 252 | 5.49 |
| `onnxruntime` | CPU | 2 | 58 / 320 / 613 | 20 / 127 / 211 | 1.04 |
| `onnxruntime` | CPU | 1 | 192 / 335 / 622 | 71 / 161 / 412 | 0.17 |
| `openvino` | CPU | default | 192 / 608 / 858 | 117 / 460 / 646 | 0.17 |
| `openvino` | CPU | 1 | 245 / 937 / 4,120 | 65 / 249 / 534 | 0.17 |
| `openvino` | GPU | default | 359 / 2,220 / 2,310 | 364 / 9,731 / 12,125 | 0.18 |
| `openvino` | NPU | default | 258 / 655 / 1,252 | 280 / 731 / 868 | 0.16 |
| `openvino-ep` | CPU | default | 212 / 621 / 3,798 | 117 / 254 / 548 | 0.18 |
| `openvino-ep` | GPU | default | 427 / 3,539 / 8,538 | 405 / 2,000 / 2,155 | 0.19 |
| `openvino-ep` | NPU | default | 268 / 684 / 1,598 | 278 / 588 / 856 | 0.17 |

Findings:

- ONNX Runtime CPU with default threads is the fastest, but the controller then uses about 5.5 cores. With `threads: 2` it uses 1 core and Velocity is just as fast. Consider `threads: 2` on the robot.
- All other runs use about 0.17 core.
- GPU is the slowest and has the largest spikes, up to 12.1 ms.
- NPU p50 is 258-280 µs and the slowest call is 1.6 ms. `npu_busy_time_us` grows by about 640 ms per NPU run and does not change in the other runs, so the policy really runs on the NPU.
- Model load takes at most 470 ms per state in these runs.
- Latency here is higher than in [benchmark/README.md](benchmark/README.md), where calls run back to back. In the controller they run every 20 ms.

## Layout

```
wbc_g1_deploy/
  apps/wbc_g1_ctrl/        # main executable
  src/                     # WBC tracking, clips, policy inference, MDP registrations
  include/                 # FSM + Isaac Lab-style obs runtime
  config/
  scripts/
```

## Related repos

| Repo | Role |
|------|------|
| [wbc-mjlab/wbc-mjlab](https://github.com/wbc-mjlab/wbc-mjlab) | Train policies; export `policy.onnx` + `config.yaml` |
| [wbc-mjlab/wbc-g1-deploy](https://github.com/wbc-mjlab/wbc-g1-deploy) | This repo — G1 ONNX runtime |
| [mujocolab/mjlab](https://github.com/mujocolab/mjlab) | Simulation / training stack |
