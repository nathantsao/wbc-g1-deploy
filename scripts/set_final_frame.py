"""
Script to set final frame of a motion clip (.npz) to stay within training distribution.
"""

from pathlib import Path

import numpy as np


FRAME_KEYS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


def _validate_clip(clip, name):
    missing = [key for key in FRAME_KEYS if key not in clip]
    if missing:
        raise ValueError(f"{name} is missing frame arrays: {', '.join(missing)}")

    frame_count = clip["joint_pos"].shape[0]
    if frame_count == 0:
        raise ValueError(f"{name} has no frames")
    for key in FRAME_KEYS:
        array = np.asarray(clip[key])
        if array.ndim == 0 or array.shape[0] != frame_count:
            raise ValueError(f"{name}[{key}] does not have {frame_count} frames")
    return frame_count

def _fixed_frames(start, end, alphas):
    frames = [end for _ in alphas]
    return np.asarray(frames, dtype=start.dtype)

def _linear_frames(start, end, alphas):
    frames = [start + alpha * (end - start) for alpha in alphas]
    return np.asarray(frames, dtype=start.dtype)


def _slerp_frames(start, end, alphas):
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    start_norm = np.linalg.norm(start, axis=-1, keepdims=True)
    end_norm = np.linalg.norm(end, axis=-1, keepdims=True)
    if np.any(start_norm == 0) or np.any(end_norm == 0):
        raise ValueError("body_quat_w contains a zero-length quaternion")
    start = start / start_norm
    end = end / end_norm

    dot = np.sum(start * end, axis=-1, keepdims=True)
    end = np.where(dot < 0, -end, end)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    theta_start = np.arccos(dot)
    sin_theta_start = np.sin(theta_start)
    near = dot > 0.9995

    frames = []
    for alpha in alphas:
        theta = theta_start * alpha
        denominator = np.where(sin_theta_start == 0, 1.0, sin_theta_start)
        spherical = (
            np.sin(theta_start - theta) / denominator * start
            + np.sin(theta) / denominator * end
        )
        linear = start + alpha * (end - start)
        frame = np.where(near, linear, spherical)
        frame /= np.linalg.norm(frame, axis=-1, keepdims=True)
        frames.append(frame)
    return np.asarray(frames, dtype=np.float32)


def set_final_frame(base_clip, target_clip, target_index: int = 0, horizon: int = 80, method: str = "append"):
    if method not in ("linear", "append"):
        raise ValueError(f"Unknown method: {method}")
    if horizon < 0:
        raise ValueError("horizon must be at least 0")

    if horizon == 0:
        return base_clip

    base_frames = _validate_clip(base_clip, "base_clip")
    target_frames = _validate_clip(target_clip, "target_clip")
    if method == "append" and (target_index < 0 or target_index + horizon > target_frames):
        raise ValueError("target_index and horizon exceed the target clip frame range")
    for key in FRAME_KEYS:
        if base_clip[key].shape[1:] != target_clip[key].shape[1:]:
            raise ValueError(f"{key} has incompatible shapes between clips")

    extended_clip = dict(base_clip)
    if method == "linear":
        alphas = np.arange(1, horizon + 1, dtype=np.float64) / horizon
    for key in FRAME_KEYS:
        if method == "append":
            bridge = target_clip[key][target_index:target_index + horizon]
        else:
            start = np.asarray(base_clip[key][-1])
            end = np.asarray(target_clip[key][target_index])
            if key == "body_quat_w":
                # bridge = _slerp_frames(start, end, alphas).astype(base_clip[key].dtype)
                bridge = _fixed_frames(start, end, alphas).astype(base_clip[key].dtype)
            else:
                # bridge = _linear_frames(start, end, alphas)
                bridge = _fixed_frames(start, end, alphas)
        extended_clip[key] = np.concatenate((base_clip[key], bridge), axis=0)

    _validate_clip(extended_clip, "extended_clip")
    if extended_clip["joint_pos"].shape[0] != base_frames + horizon:
        raise ValueError("extended clip has an unexpected frame count")
    return extended_clip


def load_clip(path: Path):
    with np.load(path, allow_pickle=True) as archive:
        return {key: archive[key] for key in archive.files}


if __name__ == "__main__":
    clips_dir = Path(__file__).resolve().parents[1] / "config" / "karate_clips"
    base_clip = load_clip(clips_dir / "getup_01_original.npz")
    target_clip = load_clip(clips_dir / "karate_0005_to_0015_g1.npz")
    extended_clip = set_final_frame(base_clip, target_clip)
    output_path = clips_dir / "getup_01.npz"
    np.savez(output_path, **extended_clip)
    print(f"Saved {output_path} ({extended_clip['joint_pos'].shape[0]} frames)")