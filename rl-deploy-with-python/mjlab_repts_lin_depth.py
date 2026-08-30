"""WF_TRON1B mjlab RepTS LinVel Depth deploy alignment helpers."""

from __future__ import annotations

from dataclasses import dataclass
import os
import threading
import time
from pathlib import Path

import numpy as np

from mjlab_repts import (
    ACTION_CLIP,
    DEFAULT_OBS_NOISE_RANGES,
    HISTORY_LENGTH,
    LEG_JOINT_INDEXES,
    POLICY_ACTION_NAMES,
    POLICY_ACTION_SCALES,
    SDK_JOINT_NAMES,
    WHEEL_JOINT_INDEXES,
    _add_uniform_noise,
    _metadata_float_array,
    _metadata_list,
    _normalize_noise_ranges,
    _vector,
    clip_actions,
    command_from_joystick_axes,
    map_actions_to_sdk_joint_commands,
)

PROPRIO_OBS_SIZE = 28
PROPRIO_HISTORY_SHAPE = (HISTORY_LENGTH, PROPRIO_OBS_SIZE)
DEPTH_CHANNELS = 1
D435_RAW_DEPTH_HEIGHT = 480
D435_RAW_DEPTH_WIDTH = 848
D435_LEFT_CROP_PX = 128
D435_LEFT_CROP_FRACTION = D435_LEFT_CROP_PX / D435_RAW_DEPTH_WIDTH
DEPTH_HEIGHT = 30
DEPTH_WIDTH = 45
DEPTH_SHAPE = (DEPTH_CHANNELS, DEPTH_HEIGHT, DEPTH_WIDTH)
DEPTH_INPUT_SHAPE = (1, *DEPTH_SHAPE)
DEPTH_MIN_DISTANCE_M = 0.2
DEPTH_MAX_DISTANCE_M = 2.0
GRU_HIDDEN_STATE_SIZE = 128
GRU_HIDDEN_STATE_SHAPE = (1, GRU_HIDDEN_STATE_SIZE)

POLICY_INPUT_NAMES = [
    "proprio_history",
    "actor_command",
    "depth",
    "hidden_state_in",
]
POLICY_OUTPUT_NAMES = ["actions", "predicted_lin_vel", "hidden_state_out"]

PROPRIO_TERM_ORDER = (
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "wheel_vel",
    "actions",
)
PROPRIO_TERM_DIMS = {
    "base_ang_vel": 3,
    "projected_gravity": 3,
    "joint_pos": 6,
    "joint_vel": 6,
    "wheel_vel": 2,
    "actions": 8,
}


def build_proprio_terms(
    base_ang_vel,
    projected_gravity,
    joint_pos,
    joint_vel,
    default_joint_pos,
    default_joint_vel,
    last_action,
    noise_ranges=None,
    rng=None,
) -> dict[str, np.ndarray]:
    """Build proprio_history terms for the LinVel representation policy."""

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
    }


def build_proprio_obs(terms: dict[str, np.ndarray]) -> np.ndarray:
    obs = np.concatenate(
        [_vector(name, terms[name], PROPRIO_TERM_DIMS[name]) for name in PROPRIO_TERM_ORDER]
    )
    if obs.shape != (PROPRIO_OBS_SIZE,):
        raise ValueError(f"proprio_obs must have shape ({PROPRIO_OBS_SIZE},), got {obs.shape}")
    return obs.astype(np.float32, copy=False)


class ProprioHistory:
    """Maintains oldest-to-newest LinVel proprioceptive history."""

    def __init__(self, history_length: int = HISTORY_LENGTH):
        self.history_length = history_length
        self._frames: list[np.ndarray] | None = None

    def reset(self, terms: dict[str, np.ndarray]) -> None:
        obs = build_proprio_obs(terms)
        self._frames = [obs.copy() for _ in range(self.history_length)]

    def update(self, terms: dict[str, np.ndarray]) -> None:
        if self._frames is None:
            self.reset(terms)
            return

        self._frames.pop(0)
        self._frames.append(build_proprio_obs(terms).copy())

    def matrix(self) -> np.ndarray:
        if self._frames is None:
            raise RuntimeError("history has not been initialized")

        obs = np.stack(self._frames, axis=0)
        expected_shape = (self.history_length, PROPRIO_OBS_SIZE)
        if obs.shape != expected_shape:
            raise ValueError(f"proprio_history must have shape {expected_shape}, got {obs.shape}")
        return obs.astype(np.float32, copy=False)

    def update_and_matrix(self, terms: dict[str, np.ndarray]) -> np.ndarray:
        self.update(terms)
        return self.matrix()


def _resize_nearest(image: np.ndarray, height: int, width: int) -> np.ndarray:
    if image.shape == (height, width):
        return image
    row_idx = np.linspace(0, image.shape[0] - 1, height).round().astype(np.int64)
    col_idx = np.linspace(0, image.shape[1] - 1, width).round().astype(np.int64)
    return image[row_idx[:, None], col_idx[None, :]]


def _validate_depth_range(min_depth: float, max_depth: float) -> tuple[float, float]:
    try:
        min_depth = float(min_depth)
        max_depth = float(max_depth)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "depth range must satisfy 0 <= min_depth < max_depth"
        ) from exc
    if (
        not np.isfinite(min_depth)
        or not np.isfinite(max_depth)
        or min_depth < 0.0
        or min_depth >= max_depth
    ):
        raise ValueError(
            "depth range must satisfy 0 <= min_depth < max_depth, "
            f"got ({min_depth}, {max_depth})"
        )
    return min_depth, max_depth


def preprocess_depth_image(
    image,
    *,
    encoding: str | None = None,
    target_shape: tuple[int, int] = (DEPTH_HEIGHT, DEPTH_WIDTH),
    depth_scale: float | None = None,
    min_depth: float = DEPTH_MIN_DISTANCE_M,
    max_depth: float = DEPTH_MAX_DISTANCE_M,
    preprocess_in_onnx: bool = True,
    left_crop_fraction: float = D435_LEFT_CROP_FRACTION,
) -> np.ndarray:
    """Convert full-FOV raw depth to finite metric ONNX input."""

    if not isinstance(preprocess_in_onnx, (bool, np.bool_)):
        raise ValueError(
            "preprocess_in_onnx must be a boolean, "
            f"got {preprocess_in_onnx!r}"
        )
    if not preprocess_in_onnx:
        min_depth, max_depth = _validate_depth_range(min_depth, max_depth)
    depth = np.asarray(image)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"depth image must have shape [H, W] or [H, W, 1], got {depth.shape}")
    try:
        left_crop_fraction = float(left_crop_fraction)
    except (TypeError, ValueError) as exc:
        raise ValueError("left_crop_fraction must be in [0, 1)") from exc
    if not np.isfinite(left_crop_fraction) or not 0.0 <= left_crop_fraction < 1.0:
        raise ValueError("left_crop_fraction must be in [0, 1)")

    if depth_scale is None:
        if encoding in ("16UC1", "mono16"):
            depth_scale = 0.001
        else:
            depth_scale = 1.0

    depth = depth.astype(np.float32) * np.float32(depth_scale)
    if preprocess_in_onnx:
        invalid = ~np.isfinite(depth) | (depth <= np.float32(0.0))
        depth = np.where(invalid, np.float32(0.0), depth)
    else:
        depth = np.nan_to_num(
            depth,
            nan=np.float32(0.0),
            posinf=np.float32(max_depth),
            neginf=np.float32(0.0),
        )
        depth = np.clip(depth, np.float32(min_depth), np.float32(max_depth))
    left_crop = int(round(depth.shape[1] * left_crop_fraction))
    if left_crop >= depth.shape[1]:
        raise ValueError("left_crop_fraction leaves no depth columns")
    if depth.shape == target_shape and left_crop:
        raise ValueError(
            "depth image already has the policy target shape; supply full-FOV raw depth"
        )
    depth = depth[:, left_crop:]
    depth = _resize_nearest(depth, target_shape[0], target_shape[1])
    return depth.reshape(1, 1, target_shape[0], target_shape[1]).astype(np.float32, copy=False)


@dataclass
class DepthSourceConfig:
    source: str = "zero"
    ros_topic: str | None = None
    ros_type: str | None = None
    encoding: str | None = None
    depth_scale: float | None = None
    min_depth: float = DEPTH_MIN_DISTANCE_M
    max_depth: float = DEPTH_MAX_DISTANCE_M
    preprocess_in_onnx: bool = True
    timeout_s: float = 0.2
    max_age_s: float = 0.5
    npy_path: str | None = None


class DepthFrameSource:
    def frame(self) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        pass


class ZeroDepthFrameSource(DepthFrameSource):
    def __init__(self, shape: tuple[int, int, int, int] = DEPTH_INPUT_SHAPE):
        self._frame = np.zeros(shape, dtype=np.float32)

    def frame(self) -> np.ndarray:
        return self._frame


class NpyDepthFrameSource(DepthFrameSource):
    def __init__(self, cfg: DepthSourceConfig):
        if cfg.npy_path is None:
            raise ValueError("depth.npy_path is required when depth.source is 'npy'")
        image = np.load(Path(cfg.npy_path))
        self._frame = preprocess_depth_image(
            image,
            encoding=cfg.encoding,
            depth_scale=cfg.depth_scale,
            min_depth=cfg.min_depth,
            max_depth=cfg.max_depth,
            preprocess_in_onnx=cfg.preprocess_in_onnx,
        )

    def frame(self) -> np.ndarray:
        return self._frame


class NpyLiveDepthFrameSource(DepthFrameSource):
    """Poll a MuJoCo-produced .npy depth frame for sim2sim."""

    def __init__(self, cfg: DepthSourceConfig):
        if cfg.npy_path is None:
            raise ValueError("depth.npy_path is required when depth.source is 'npy_live'")
        self._cfg = cfg
        self._path = Path(cfg.npy_path)
        self._latest_frame: np.ndarray | None = None
        self._latest_mtime_ns: int | None = None

    def frame(self) -> np.ndarray:
        deadline = time.time() + max(self._cfg.timeout_s, 0.0)
        while True:
            if self._path.exists():
                stat = self._path.stat()
                if self._latest_frame is None or stat.st_mtime_ns != self._latest_mtime_ns:
                    image = np.load(self._path)
                    self._latest_frame = preprocess_depth_image(
                        image,
                        encoding=self._cfg.encoding,
                        depth_scale=self._cfg.depth_scale,
                        min_depth=self._cfg.min_depth,
                        max_depth=self._cfg.max_depth,
                        preprocess_in_onnx=self._cfg.preprocess_in_onnx,
                    )
                    self._latest_mtime_ns = stat.st_mtime_ns
                return self._latest_frame
            if time.time() >= deadline:
                raise TimeoutError(f"timed out waiting for depth frame file {self._path}")
            time.sleep(0.001)


def _resolve_ros_type(explicit: str | None = None) -> str:
    value = explicit or os.getenv("ROS_TYPE")
    if value:
        normalized = value.strip().lower()
        if normalized in ("1", "ros1"):
            return "ros1"
        if normalized in ("2", "ros2"):
            raise ValueError("ROS depth is ROS1-only; use ROS_TYPE=ros1")
        raise ValueError("ROS_TYPE must be 'ros1'")

    ros_version = os.getenv("ROS_VERSION")
    if ros_version == "1":
        return "ros1"
    if ros_version == "2":
        raise ValueError("ROS depth is ROS1-only; use a ROS1 environment")

    return "ros1"


class _BufferedRosDepthFrameSource(DepthFrameSource):
    """Shared ROS Image buffering and staleness checks."""

    def __init__(self, cfg: DepthSourceConfig):
        self._cfg = cfg
        self._lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._latest_recv_time_s: float | None = None

    def _callback(self, msg) -> None:
        frame = _ros_image_to_depth_input(
            msg,
            encoding_override=self._cfg.encoding,
            depth_scale=self._cfg.depth_scale,
            min_depth=self._cfg.min_depth,
            max_depth=self._cfg.max_depth,
            preprocess_in_onnx=self._cfg.preprocess_in_onnx,
        )
        recv_time_s = time.time()
        with self._lock:
            self._latest_frame = frame
            self._latest_recv_time_s = recv_time_s

    def frame(self) -> np.ndarray:
        deadline = time.time() + max(self._cfg.timeout_s, 0.0)
        while True:
            with self._lock:
                if self._latest_frame is not None:
                    age_s = time.time() - (self._latest_recv_time_s or 0.0)
                    if self._cfg.max_age_s >= 0.0 and age_s > self._cfg.max_age_s:
                        raise TimeoutError(
                            f"stale ROS depth image on {self._cfg.ros_topic}: "
                            f"age {age_s:.3f}s exceeds {self._cfg.max_age_s:.3f}s"
                        )
                    return self._latest_frame
            if time.time() >= deadline:
                raise TimeoutError(
                    f"timed out waiting for ROS depth image on {self._cfg.ros_topic}"
                )
            time.sleep(0.001)


class _Ros1SubBackend(_BufferedRosDepthFrameSource):
    def __init__(self, cfg: DepthSourceConfig):
        super().__init__(cfg)
        try:
            import rospy
            from sensor_msgs.msg import Image
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "ROS1 depth source requires rospy and sensor_msgs. Source the ROS1 "
                "workspace before running with depth.source=ros."
            ) from exc

        if not rospy.core.is_initialized():
            rospy.init_node(
                "mjlab_repts_gru_lin_depth_source",
                anonymous=True,
                disable_signals=True,
            )

        self._subscriber = rospy.Subscriber(cfg.ros_topic, Image, self._callback, queue_size=1)

    def close(self) -> None:
        self._subscriber.unregister()


class RosDepthFrameSource(DepthFrameSource):
    """ROS sensor_msgs/Image depth source for sim2real deployment."""

    def __init__(self, cfg: DepthSourceConfig):
        if not cfg.ros_topic:
            raise ValueError("depth.ros_topic is required when depth.source is 'ros'")
        _resolve_ros_type(cfg.ros_type)
        self._backend = _Ros1SubBackend(cfg)

    def frame(self) -> np.ndarray:
        return self._backend.frame()

    def close(self) -> None:
        self._backend.close()


def ros_image_to_depth_meters(
    msg,
    *,
    encoding_override: str | None = None,
    depth_scale: float | None = None,
) -> np.ndarray:
    encoding = encoding_override or msg.encoding
    dtype_by_encoding = {
        "16UC1": np.uint16,
        "mono16": np.uint16,
        "32FC1": np.float32,
        "passthrough": np.float32,
    }
    if encoding not in dtype_by_encoding:
        raise ValueError(f"unsupported ROS depth encoding: {encoding}")

    dtype = np.dtype(dtype_by_encoding[encoding])
    if getattr(msg, "is_bigendian", 0):
        dtype = dtype.newbyteorder(">")
    height = int(msg.height)
    width = int(msg.width)
    row_step = int(getattr(msg, "step", 0)) or width * dtype.itemsize
    min_bytes = height * row_step
    if len(msg.data) < min_bytes:
        raise ValueError(
            f"ROS depth image data has {len(msg.data)} bytes, expected at least {min_bytes}"
        )
    if row_step % dtype.itemsize != 0:
        raise ValueError(
            f"ROS depth image step {row_step} is not aligned to dtype {dtype}"
        )
    row_values = row_step // dtype.itemsize
    if row_values < width:
        raise ValueError(
            f"ROS depth image step {row_step} is too small for width {width} and dtype {dtype}"
        )
    image = np.frombuffer(msg.data, dtype=dtype, count=height * row_values)
    image = image.reshape(height, row_values)[:, :width]
    if depth_scale is None:
        if encoding in ("16UC1", "mono16"):
            depth_scale = 0.001
        else:
            depth_scale = 1.0
    depth = image.astype(np.float32) * np.float32(depth_scale)
    return depth


def _ros_image_to_depth_input(
    msg,
    *,
    encoding_override: str | None = None,
    depth_scale: float | None = None,
    min_depth: float = DEPTH_MIN_DISTANCE_M,
    max_depth: float = DEPTH_MAX_DISTANCE_M,
    preprocess_in_onnx: bool = True,
    left_crop_fraction: float = D435_LEFT_CROP_FRACTION,
) -> np.ndarray:
    encoding = encoding_override or msg.encoding
    image = ros_image_to_depth_meters(
        msg,
        encoding_override=encoding_override,
        depth_scale=depth_scale,
    )
    return preprocess_depth_image(
        image,
        encoding=encoding if depth_scale is None else None,
        depth_scale=1.0,
        min_depth=min_depth,
        max_depth=max_depth,
        preprocess_in_onnx=preprocess_in_onnx,
        left_crop_fraction=left_crop_fraction,
    )


def create_depth_frame_source(config: dict | None) -> DepthFrameSource:
    cfg_data = dict(config or {})
    env_map = {
        "source": "MJLAB_DEPTH_SOURCE",
        "ros_topic": "MJLAB_DEPTH_ROS_TOPIC",
        "encoding": "MJLAB_DEPTH_ENCODING",
        "npy_path": "MJLAB_DEPTH_NPY_PATH",
    }
    for key, env_name in env_map.items():
        value = os.getenv(env_name)
        if value:
            cfg_data[key] = value
    for key, env_name in (
        ("depth_scale", "MJLAB_DEPTH_SCALE"),
        ("min_depth", "MJLAB_DEPTH_MIN"),
        ("max_depth", "MJLAB_DEPTH_MAX"),
        ("timeout_s", "MJLAB_DEPTH_TIMEOUT"),
        ("max_age_s", "MJLAB_DEPTH_MAX_AGE"),
    ):
        value = os.getenv(env_name)
        if value:
            cfg_data[key] = float(value)

    cfg = DepthSourceConfig(**cfg_data)
    if not cfg.preprocess_in_onnx:
        _validate_depth_range(cfg.min_depth, cfg.max_depth)
    if cfg.source == "zero":
        return ZeroDepthFrameSource()
    if cfg.source == "npy":
        return NpyDepthFrameSource(cfg)
    if cfg.source == "npy_live":
        return NpyLiveDepthFrameSource(cfg)
    if cfg.source == "ros":
        return RosDepthFrameSource(cfg)
    raise ValueError(f"unsupported depth.source: {cfg.source}")


def _shape_matches(actual: list, expected: list[int]) -> bool:
    if len(actual) != len(expected):
        return False
    for actual_dim, expected_dim in zip(actual, expected):
        if isinstance(actual_dim, str):
            continue
        if int(actual_dim) != expected_dim:
            return False
    return True


def validate_depth_policy_interface(
    input_names: list[str],
    input_shapes: list[list[int]],
    output_names: list[str],
    output_shapes: list[list[int]],
    metadata: dict[str, str] | None = None,
    *,
    hidden_state_shape: tuple[int, int] = GRU_HIDDEN_STATE_SHAPE,
    policy_name: str = "mjlab_repts_gru_lin_depth",
) -> None:
    if (
        len(hidden_state_shape) != 2
        or hidden_state_shape[0] != 1
        or hidden_state_shape[1] <= 0
    ):
        raise ValueError(
            f"{policy_name} hidden_state_shape must be (1, positive_size), "
            f"got {hidden_state_shape}"
        )

    expected_input_shapes = [
        [1, *PROPRIO_HISTORY_SHAPE],
        [1, 3],
        [1, *DEPTH_SHAPE],
        list(hidden_state_shape),
    ]
    expected_output_shapes = [[1, 8], [1, 3], list(hidden_state_shape)]

    if input_names != POLICY_INPUT_NAMES:
        raise ValueError(
            f"{policy_name} ONNX inputs must be {POLICY_INPUT_NAMES}, got {input_names}"
        )
    input_shapes_match = all(
        _shape_matches(actual, expected)
        for actual, expected in zip(input_shapes, expected_input_shapes)
    )
    if not input_shapes_match:
        raise ValueError(
            f"{policy_name} ONNX input shapes must be "
            f"{expected_input_shapes}, got {input_shapes}"
        )
    if output_names != POLICY_OUTPUT_NAMES:
        raise ValueError(
            f"{policy_name} ONNX outputs must be {POLICY_OUTPUT_NAMES}, got {output_names}"
        )
    output_shapes_match = all(
        _shape_matches(actual, expected)
        for actual, expected in zip(output_shapes, expected_output_shapes)
    )
    if not output_shapes_match:
        raise ValueError(
            f"{policy_name} ONNX output shapes must be "
            f"{expected_output_shapes}, got {output_shapes}"
        )

    if metadata is None:
        return

    student_observation_names = _metadata_list(metadata, "student_observation_names")
    if student_observation_names is not None and student_observation_names != list(PROPRIO_TERM_ORDER):
        raise ValueError(
            f"{policy_name} ONNX metadata student_observation_names must be "
            f"{list(PROPRIO_TERM_ORDER)}, got {student_observation_names}"
        )

    command_observation_names = _metadata_list(metadata, "command_observation_names")
    if command_observation_names is not None and command_observation_names != ["command"]:
        raise ValueError(
            f"{policy_name} ONNX metadata command_observation_names must be "
            f"['command'], got {command_observation_names}"
        )

    policy_input_names = _metadata_list(metadata, "policy_input_names")
    legacy_metadata_input_names = POLICY_INPUT_NAMES[:2]
    if (
        policy_input_names is not None
        and policy_input_names != POLICY_INPUT_NAMES
        and policy_input_names != legacy_metadata_input_names
    ):
        raise ValueError(
            f"{policy_name} ONNX metadata policy_input_names must be "
            f"{POLICY_INPUT_NAMES} or legacy {legacy_metadata_input_names}, got {policy_input_names}"
        )

    policy_output_names = _metadata_list(metadata, "policy_output_names")
    if (
        policy_output_names is not None
        and policy_output_names != POLICY_OUTPUT_NAMES[:2]
        and policy_output_names != POLICY_OUTPUT_NAMES
    ):
        raise ValueError(
            f"{policy_name} ONNX metadata policy_output_names must be "
            f"{POLICY_OUTPUT_NAMES[:2]} or {POLICY_OUTPUT_NAMES}, got {policy_output_names}"
        )

    student_history_length = metadata.get("student_history_length")
    if student_history_length is not None and int(student_history_length) != HISTORY_LENGTH:
        raise ValueError(
            f"{policy_name} ONNX metadata student_history_length must be "
            f"{HISTORY_LENGTH}, got {student_history_length}"
        )

    flatten_history = metadata.get("student_history_flatten_dim")
    if flatten_history is not None and flatten_history.lower() != "false":
        raise ValueError(
            f"{policy_name} ONNX metadata student_history_flatten_dim must be false, "
            f"got {flatten_history}"
        )

    history_order = metadata.get("student_history_order")
    if history_order is not None and history_order != "oldest_to_newest":
        raise ValueError(
            f"{policy_name} ONNX metadata student_history_order must be "
            f"oldest_to_newest, got {history_order}"
        )

    action_target_names = _metadata_list(metadata, "action_target_names")
    if action_target_names is not None and action_target_names != list(POLICY_ACTION_NAMES):
        raise ValueError(
            f"{policy_name} ONNX metadata action_target_names must be "
            f"{list(POLICY_ACTION_NAMES)}, got {action_target_names}"
        )

    action_scale = _metadata_float_array(metadata, "action_scale")
    if action_scale is not None:
        expected_action_scale = np.asarray(POLICY_ACTION_SCALES, dtype=np.float32)
        if action_scale.shape != expected_action_scale.shape or not np.allclose(
            action_scale, expected_action_scale
        ):
            raise ValueError(
                f"{policy_name} ONNX metadata action_scale must be "
                f"{list(POLICY_ACTION_SCALES)}, got {action_scale.tolist()}"
            )


__all__ = [
    "ACTION_CLIP",
    "DEFAULT_OBS_NOISE_RANGES",
    "D435_LEFT_CROP_FRACTION",
    "D435_LEFT_CROP_PX",
    "D435_RAW_DEPTH_HEIGHT",
    "D435_RAW_DEPTH_WIDTH",
    "DEPTH_INPUT_SHAPE",
    "GRU_HIDDEN_STATE_SHAPE",
    "POLICY_INPUT_NAMES",
    "POLICY_OUTPUT_NAMES",
    "POLICY_ACTION_NAMES",
    "PROPRIO_HISTORY_SHAPE",
    "PROPRIO_OBS_SIZE",
    "PROPRIO_TERM_DIMS",
    "PROPRIO_TERM_ORDER",
    "SDK_JOINT_NAMES",
    "ProprioHistory",
    "build_proprio_obs",
    "build_proprio_terms",
    "clip_actions",
    "command_from_joystick_axes",
    "create_depth_frame_source",
    "map_actions_to_sdk_joint_commands",
    "preprocess_depth_image",
    "ros_image_to_depth_meters",
    "validate_depth_policy_interface",
    "_resolve_ros_type",
]
