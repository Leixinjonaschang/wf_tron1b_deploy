"""WF_TRON1B mjlab RepTS LinVel deploy alignment helpers."""

from __future__ import annotations

import ast

import numpy as np


HISTORY_LENGTH = 5
LIN_PROPRIO_OBS_SIZE = 28
LIN_PROPRIO_HISTORY_SHAPE = (HISTORY_LENGTH, LIN_PROPRIO_OBS_SIZE)
LIN_COMMAND_SIZE = 3
ACTION_CLIP = 2.0
LEG_ACTION_SCALE = 0.5
WHEEL_ACTION_SCALE = 10.0

SDK_JOINT_NAMES = (
    "abad_L_Joint",
    "hip_L_Joint",
    "knee_L_Joint",
    "wheel_L_Joint",
    "abad_R_Joint",
    "hip_R_Joint",
    "knee_R_Joint",
    "wheel_R_Joint",
)

POLICY_ACTION_NAMES = (
    "abad_L_Joint",
    "hip_L_Joint",
    "knee_L_Joint",
    "abad_R_Joint",
    "hip_R_Joint",
    "knee_R_Joint",
    "wheel_L_Joint",
    "wheel_R_Joint",
)

ACTION_TO_SDK_JOINT_INDEX = np.array([0, 1, 2, 4, 5, 6, 3, 7])
LEG_JOINT_INDEXES = np.array([0, 1, 2, 4, 5, 6])
WHEEL_JOINT_INDEXES = np.array([3, 7])
LEG_ACTION_INDEXES = (0, 1, 2, 3, 4, 5)
WHEEL_ACTION_INDEXES = (6, 7)
POLICY_ACTION_SCALES = (
    LEG_ACTION_SCALE,
    LEG_ACTION_SCALE,
    LEG_ACTION_SCALE,
    LEG_ACTION_SCALE,
    LEG_ACTION_SCALE,
    LEG_ACTION_SCALE,
    WHEEL_ACTION_SCALE,
    WHEEL_ACTION_SCALE,
)

PROPRIO_TERM_ORDER = (
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "wheel_vel",
    "actions",
)
TERM_DIMS = {
    "base_ang_vel": 3,
    "projected_gravity": 3,
    "joint_pos": 6,
    "joint_vel": 6,
    "wheel_vel": 2,
    "actions": 8,
    "command": 3,
}
DEFAULT_OBS_NOISE_RANGES = {
    "base_ang_vel": (-0.2, 0.2),
    "projected_gravity": (-0.05, 0.05),
    "joint_pos": (-0.01, 0.01),
    "joint_vel": (-1.5, 1.5),
    "wheel_vel": (-0.5, 0.5),
}


def _vector(name: str, value, size: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    return array


def _normalize_noise_ranges(noise_ranges) -> dict[str, tuple[float, float]]:
    if not noise_ranges:
        return {}

    normalized = {}
    for name, bounds in noise_ranges.items():
        if name not in DEFAULT_OBS_NOISE_RANGES:
            raise ValueError(f"unsupported observation noise term: {name}")
        if len(bounds) != 2:
            raise ValueError(f"noise range for {name} must have two values, got {bounds}")
        low, high = float(bounds[0]), float(bounds[1])
        if high < low:
            raise ValueError(f"noise range for {name} must be [min, max], got {bounds}")
        normalized[name] = (low, high)
    return normalized


def _add_uniform_noise(
    name: str,
    value: np.ndarray,
    noise_ranges: dict[str, tuple[float, float]],
    rng,
) -> np.ndarray:
    bounds = noise_ranges.get(name)
    if bounds is None:
        return value

    low, high = bounds
    noise = rng.uniform(low, high, size=value.shape).astype(np.float32)
    return value + noise


def build_actor_terms(
    base_ang_vel,
    projected_gravity,
    joint_pos,
    joint_vel,
    default_joint_pos,
    default_joint_vel,
    last_action,
    command,
    noise_ranges=None,
    rng=None,
) -> dict[str, np.ndarray]:
    """Build proprioceptive terms and the separate actor command."""

    joint_pos = _vector("joint_pos", joint_pos, 8)
    joint_vel = _vector("joint_vel", joint_vel, 8)
    default_joint_pos = _vector("default_joint_pos", default_joint_pos, 8)
    default_joint_vel = _vector("default_joint_vel", default_joint_vel, 8)
    noise_ranges = _normalize_noise_ranges(noise_ranges)
    if noise_ranges and rng is None:
        rng = np.random.default_rng()

    base_ang_vel = _vector("base_ang_vel", base_ang_vel, 3)
    projected_gravity = _vector("projected_gravity", projected_gravity, 3)
    joint_pos_rel = joint_pos[LEG_JOINT_INDEXES] - default_joint_pos[LEG_JOINT_INDEXES]
    joint_vel_rel = joint_vel[LEG_JOINT_INDEXES] - default_joint_vel[LEG_JOINT_INDEXES]
    wheel_vel = joint_vel[WHEEL_JOINT_INDEXES]

    if noise_ranges:
        base_ang_vel = _add_uniform_noise("base_ang_vel", base_ang_vel, noise_ranges, rng)
        projected_gravity = _add_uniform_noise(
            "projected_gravity", projected_gravity, noise_ranges, rng
        )
        joint_pos_rel = _add_uniform_noise("joint_pos", joint_pos_rel, noise_ranges, rng)
        joint_vel_rel = _add_uniform_noise("joint_vel", joint_vel_rel, noise_ranges, rng)
        wheel_vel = _add_uniform_noise("wheel_vel", wheel_vel, noise_ranges, rng)

    return {
        "base_ang_vel": base_ang_vel,
        "projected_gravity": projected_gravity,
        "joint_pos": joint_pos_rel,
        "joint_vel": joint_vel_rel * np.float32(0.05),
        "wheel_vel": wheel_vel * np.float32(0.5),
        "actions": _vector("last_action", last_action, 8),
        "command": _vector("command", command, 3),
    }


def build_lin_proprio_obs(terms: dict[str, np.ndarray]) -> np.ndarray:
    obs = np.concatenate(
        [_vector(name, terms[name], TERM_DIMS[name]) for name in PROPRIO_TERM_ORDER]
    )
    if obs.shape != (LIN_PROPRIO_OBS_SIZE,):
        raise ValueError(
            f"proprio_obs must have shape ({LIN_PROPRIO_OBS_SIZE},), got {obs.shape}"
        )
    return obs.astype(np.float32, copy=False)


class LinProprioHistory:
    """Maintains oldest-to-newest LinVel proprioceptive history."""

    def __init__(self, history_length: int = HISTORY_LENGTH):
        self.history_length = history_length
        self._frames: list[np.ndarray] | None = None

    def reset(self, terms: dict[str, np.ndarray]) -> None:
        obs = build_lin_proprio_obs(terms)
        self._frames = [obs.copy() for _ in range(self.history_length)]

    def clear(self) -> None:
        self._frames = None

    def update(self, terms: dict[str, np.ndarray]) -> None:
        if self._frames is None:
            self.reset(terms)
            return

        self._frames.pop(0)
        self._frames.append(build_lin_proprio_obs(terms).copy())

    def matrix(self) -> np.ndarray:
        if self._frames is None:
            raise RuntimeError("history has not been initialized")

        obs = np.stack(self._frames, axis=0)
        expected_shape = (self.history_length, LIN_PROPRIO_OBS_SIZE)
        if obs.shape != expected_shape:
            raise ValueError(
                f"proprio_history must have shape {expected_shape}, got {obs.shape}"
            )
        return obs.astype(np.float32, copy=False)

    def update_and_matrix(self, terms: dict[str, np.ndarray]) -> np.ndarray:
        self.update(terms)
        return self.matrix()


def clip_actions(actions, clip: float = ACTION_CLIP) -> np.ndarray:
    return np.clip(_vector("actions", actions, 8), -clip, clip).astype(np.float32, copy=False)


def map_actions_to_sdk_joint_commands(
    actions,
    default_joint_pos,
    leg_action_scale: float = LEG_ACTION_SCALE,
    wheel_action_scale: float = WHEEL_ACTION_SCALE,
    leg_kp: float = 40.0,
    leg_kd: float = 1.8,
    wheel_kp: float = 0.0,
    wheel_kd: float = 0.5,
) -> dict[str, np.ndarray]:
    """Map policy-order actions to SDK joint-order q/dq/Kp/Kd arrays."""

    actions = _vector("actions", actions, 8)
    default_joint_pos = _vector("default_joint_pos", default_joint_pos, 8)

    q = np.zeros(8, dtype=np.float32)
    dq = np.zeros(8, dtype=np.float32)
    tau = np.zeros(8, dtype=np.float32)
    kp = np.zeros(8, dtype=np.float32)
    kd = np.zeros(8, dtype=np.float32)

    for action_idx in LEG_ACTION_INDEXES:
        joint_idx = ACTION_TO_SDK_JOINT_INDEX[action_idx]
        q[joint_idx] = default_joint_pos[joint_idx] + leg_action_scale * actions[action_idx]
        kp[joint_idx] = leg_kp
        kd[joint_idx] = leg_kd

    for action_idx in WHEEL_ACTION_INDEXES:
        joint_idx = ACTION_TO_SDK_JOINT_INDEX[action_idx]
        dq[joint_idx] = wheel_action_scale * actions[action_idx]
        kp[joint_idx] = wheel_kp
        kd[joint_idx] = wheel_kd

    return {"q": q, "dq": dq, "tau": tau, "Kp": kp, "Kd": kd}


def command_from_joystick_axes(axes) -> np.ndarray:
    axes = np.asarray(axes, dtype=np.float32).reshape(-1)
    if axes.shape[0] < 3:
        raise ValueError(f"joystick axes must have at least 3 values, got {axes.shape[0]}")

    clipped = np.clip(axes[:3], -1.0, 1.0)
    forward_command = clipped[1]
    if forward_command > 0.0:
        forward_command *= np.float32(2.0)
    return np.array(
        [forward_command, clipped[0], clipped[2] * np.float32(np.pi / 2.0)],
        dtype=np.float32,
    )


def rate_limit_commands(current, target, rate_limits, dt: float) -> np.ndarray:
    """Limit increases in command magnitude while applying deceleration directly."""
    current = np.asarray(current, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    rate_limits = np.asarray(rate_limits, dtype=np.float32)
    max_delta = rate_limits * np.float32(dt)
    current_magnitude = np.abs(current)
    target_magnitude = np.abs(target)
    limited_magnitude = np.minimum(target_magnitude, current_magnitude + max_delta)
    limited_target = np.copysign(limited_magnitude, target)
    return np.where(target_magnitude > current_magnitude, limited_target, target)


def _metadata_list(metadata: dict[str, str], key: str) -> list[str] | None:
    value = metadata.get(key)
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    if not isinstance(value, str):
        value = str(value)
    stripped = value.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, (list, tuple)):
            return [str(item).strip() for item in parsed if str(item).strip()]
    return [item.strip() for item in value.split(",") if item.strip()]


def _metadata_float_array(metadata: dict[str, str], key: str) -> np.ndarray | None:
    values = _metadata_list(metadata, key)
    if values is None:
        return None
    return np.asarray([float(value) for value in values], dtype=np.float32)


def validate_lin_policy_interface(
    input_names: list[str],
    input_shapes: list[list[int]],
    output_names: list[str],
    output_shapes: list[list[int]],
    metadata: dict[str, str] | None = None,
) -> None:
    expected_input_names = ["proprio_history", "actor_command"]
    expected_input_shapes = [[1, HISTORY_LENGTH, LIN_PROPRIO_OBS_SIZE], [1, LIN_COMMAND_SIZE]]
    expected_output_names = ["actions", "predicted_lin_vel"]
    expected_output_shapes = [[1, 8], [1, 3]]

    if input_names != expected_input_names:
        raise ValueError(f"mjlab_repts_lin ONNX inputs must be {expected_input_names}, got {input_names}")
    if input_shapes != expected_input_shapes:
        raise ValueError(
            f"mjlab_repts_lin ONNX input shapes must be {expected_input_shapes}, got {input_shapes}"
        )
    if output_names != expected_output_names:
        raise ValueError(f"mjlab_repts_lin ONNX outputs must be {expected_output_names}, got {output_names}")
    if output_shapes != expected_output_shapes:
        raise ValueError(
            f"mjlab_repts_lin ONNX output shapes must be {expected_output_shapes}, got {output_shapes}"
        )

    if metadata is None:
        return

    expected_observation_names = list(PROPRIO_TERM_ORDER)
    for metadata_key in (
        "observation_names",
        "student_observation_names",
    ):
        observation_names = _metadata_list(metadata, metadata_key)
        if observation_names is None:
            continue
        if observation_names != expected_observation_names:
            raise ValueError(
                f"mjlab_repts_lin ONNX metadata {metadata_key} must be "
                f"{expected_observation_names}, got {observation_names}"
            )

    command_observation_names = _metadata_list(metadata, "command_observation_names")
    if command_observation_names is not None and command_observation_names != ["command"]:
        raise ValueError(
            "mjlab_repts_lin ONNX metadata command_observation_names must be "
            f"['command'], got {command_observation_names}"
        )

    policy_input_names = _metadata_list(metadata, "policy_input_names")
    if policy_input_names is not None and policy_input_names != expected_input_names:
        raise ValueError(
            f"mjlab_repts_lin ONNX metadata policy_input_names must be "
            f"{expected_input_names}, got {policy_input_names}"
        )

    policy_output_names = _metadata_list(metadata, "policy_output_names")
    if policy_output_names is not None and policy_output_names != expected_output_names:
        raise ValueError(
            f"mjlab_repts_lin ONNX metadata policy_output_names must be "
            f"{expected_output_names}, got {policy_output_names}"
        )

    student_history_length = metadata.get("student_history_length")
    if (
        student_history_length is not None
        and int(student_history_length) != HISTORY_LENGTH
    ):
        raise ValueError(
            f"mjlab_repts_lin ONNX metadata student_history_length must be "
            f"{HISTORY_LENGTH}, got {student_history_length}"
        )

    flatten_history = metadata.get("student_history_flatten_dim")
    if flatten_history is not None and flatten_history.lower() != "false":
        raise ValueError(
            "mjlab_repts_lin ONNX metadata student_history_flatten_dim must be false, "
            f"got {flatten_history}"
        )

    history_order = metadata.get("student_history_order")
    if history_order is not None and history_order != "oldest_to_newest":
        raise ValueError(
            "mjlab_repts_lin ONNX metadata student_history_order must be "
            f"oldest_to_newest, got {history_order}"
        )

    action_target_names = _metadata_list(metadata, "action_target_names")
    if (
        action_target_names is not None
        and action_target_names != list(POLICY_ACTION_NAMES)
    ):
        raise ValueError(
            f"mjlab_repts_lin ONNX metadata action_target_names must be "
            f"{list(POLICY_ACTION_NAMES)}, got {action_target_names}"
        )

    action_scale = _metadata_float_array(metadata, "action_scale")
    if action_scale is not None:
        expected_action_scale = np.asarray(POLICY_ACTION_SCALES, dtype=np.float32)
        if action_scale.shape != expected_action_scale.shape or not np.allclose(
            action_scale, expected_action_scale
        ):
            raise ValueError(
                f"mjlab_repts_lin ONNX metadata action_scale must be "
                f"{list(POLICY_ACTION_SCALES)}, got {action_scale.tolist()}"
            )
