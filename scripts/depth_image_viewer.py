"""Realtime depth image viewer for WF_TRON1B sim2sim."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_ROOT = REPO_ROOT / "rl-deploy-with-python"
sys.path.insert(0, str(DEPLOY_ROOT))

from mjlab_repts_lin_depth import (
    D435_LEFT_CROP_FRACTION,
    D435_RAW_DEPTH_HEIGHT,
    D435_RAW_DEPTH_WIDTH,
    _resolve_ros_type,
    ros_image_to_depth_meters,
)


DEFAULT_TOPIC = "/camera/depth/image_rect_raw"
DEFAULT_MIN_DEPTH = 0.0
DEFAULT_MAX_DEPTH = 10.0
DEFAULT_SCALE = 1
DEFAULT_REFRESH_HZ = 30.0
DEFAULT_GRADCAM_ALPHA = 0.45
DEFAULT_GRADCAM_MAX_AGE_S = 1.0
DEFAULT_WINDOW_HEIGHT = D435_RAW_DEPTH_HEIGHT
DEFAULT_WINDOW_WIDTH = D435_RAW_DEPTH_WIDTH

_TURBO_STOPS = np.array(
    [
        [48, 18, 59],
        [50, 101, 194],
        [43, 180, 233],
        [105, 221, 113],
        [238, 218, 38],
        [230, 107, 29],
        [122, 4, 3],
    ],
    dtype=np.float32,
)

@dataclass
class DepthViewerConfig:
    source: str = "ros"
    ros_topic: str = DEFAULT_TOPIC
    ros_type: str | None = None
    npy_path: str | None = None
    encoding: str | None = None
    depth_scale: float | None = None
    min_depth: float = DEFAULT_MIN_DEPTH
    max_depth: float = DEFAULT_MAX_DEPTH
    scale: int = DEFAULT_SCALE
    refresh_hz: float = DEFAULT_REFRESH_HZ
    colormap: str = "turbo"
    gradcam_path: str | None = None
    gradcam_alpha: float = DEFAULT_GRADCAM_ALPHA


class GradCamFileSource:
    """Best-effort reader for atomically-published asynchronous Grad-CAM files."""

    def __init__(self, path: str):
        self._path = Path(path)
        self._mtime_ns: int | None = None
        self._cam: np.ndarray | None = None
        self._source_time_s: float | None = None
        self._target = "unknown"
        self._frame_count = 0
        self._fps = 0.0
        self._last_update_s: float | None = None

    def latest(self) -> tuple[np.ndarray | None, float | None, str, int, float]:
        try:
            stat = self._path.stat()
        except OSError:
            return self._snapshot()
        if stat.st_mtime_ns == self._mtime_ns:
            return self._snapshot()
        try:
            with np.load(self._path, allow_pickle=False) as payload:
                cam = np.asarray(payload["cam"], dtype=np.float32)
                source_time_s = float(np.asarray(payload["source_time_s"]).item())
                target = str(np.asarray(payload["target"]).item())
            if cam.shape != (30, 45) or not np.isfinite(cam).all() or (cam < 0.0).any():
                return self._snapshot()
        except (KeyError, OSError, ValueError):
            return self._snapshot()
        now = time.time()
        if self._last_update_s is not None:
            instant_fps = 1.0 / max(now - self._last_update_s, 1.0e-6)
            self._fps = instant_fps if self._fps <= 0.0 else 0.9 * self._fps + 0.1 * instant_fps
        self._cam = cam.copy()
        self._source_time_s = source_time_s
        self._target = target
        self._mtime_ns = stat.st_mtime_ns
        self._last_update_s = now
        self._frame_count += 1
        return self._snapshot()

    def _snapshot(self) -> tuple[np.ndarray | None, float | None, str, int, float]:
        cam = None if self._cam is None else self._cam.copy()
        return cam, self._source_time_s, self._target, self._frame_count, self._fps


class _BufferedRosDepthImageSubscriber:
    def __init__(self, cfg: DepthViewerConfig):
        self._cfg = cfg
        self._lock = threading.Lock()
        self._latest_depth_m: np.ndarray | None = None
        self._latest_recv_time_s: float | None = None
        self._frame_count = 0
        self._fps = 0.0

    def _callback(self, msg) -> None:
        depth_m = ros_image_to_depth_meters(
            msg,
            encoding_override=self._cfg.encoding,
            depth_scale=self._cfg.depth_scale,
        )
        recv_time_s = time.time()
        with self._lock:
            if self._latest_recv_time_s is not None:
                dt = max(recv_time_s - self._latest_recv_time_s, 1.0e-6)
                instant_fps = 1.0 / dt
                self._fps = (
                    instant_fps
                    if self._fps <= 0.0
                    else 0.9 * self._fps + 0.1 * instant_fps
                )
            self._latest_depth_m = depth_m
            self._latest_recv_time_s = recv_time_s
            self._frame_count += 1

    def latest(self) -> tuple[np.ndarray | None, float | None, int, float]:
        with self._lock:
            depth = None if self._latest_depth_m is None else self._latest_depth_m.copy()
            return depth, self._latest_recv_time_s, self._frame_count, self._fps


class _Ros1DepthImageSubscriber(_BufferedRosDepthImageSubscriber):
    def __init__(self, cfg: DepthViewerConfig):
        super().__init__(cfg)
        try:
            import rospy
            from sensor_msgs.msg import Image
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "ROS1 depth viewer requires rospy and sensor_msgs. Source the ROS1 "
                "workspace before running the viewer."
            ) from exc

        if not rospy.core.is_initialized():
            rospy.init_node("mjlab_depth_image_viewer", anonymous=True, disable_signals=True)
        self._subscriber = rospy.Subscriber(
            cfg.ros_topic,
            Image,
            self._callback,
            queue_size=1,
        )

    def close(self) -> None:
        self._subscriber.unregister()


class RosDepthImageSubscriber:
    def __init__(self, cfg: DepthViewerConfig):
        _resolve_ros_type(cfg.ros_type)
        self._backend = _Ros1DepthImageSubscriber(cfg)

    def latest(self) -> tuple[np.ndarray | None, float | None, int, float]:
        return self._backend.latest()

    def close(self) -> None:
        self._backend.close()


class NpyLiveDepthImageSource:
    def __init__(self, cfg: DepthViewerConfig):
        if cfg.npy_path is None:
            raise ValueError("MJLAB_DEPTH_NPY_PATH is required for npy_live depth viewer")
        self._path = Path(cfg.npy_path)
        self._latest_mtime_ns: int | None = None
        self._latest_depth_m: np.ndarray | None = None
        self._latest_recv_time_s: float | None = None
        self._frame_count = 0
        self._fps = 0.0

    def latest(self) -> tuple[np.ndarray | None, float | None, int, float]:
        if not self._path.exists():
            return self._snapshot()

        stat = self._path.stat()
        if self._latest_mtime_ns == stat.st_mtime_ns:
            return self._snapshot()

        try:
            depth_m = np.load(self._path).astype(np.float32, copy=False)
        except (OSError, ValueError):
            return self._snapshot()
        if depth_m.ndim == 3 and depth_m.shape[-1] == 1:
            depth_m = depth_m[..., 0]
        if depth_m.ndim != 2:
            return self._snapshot()

        now = time.time()
        if self._latest_recv_time_s is not None:
            dt = max(now - self._latest_recv_time_s, 1.0e-6)
            instant_fps = 1.0 / dt
            self._fps = (
                instant_fps if self._fps <= 0.0 else 0.9 * self._fps + 0.1 * instant_fps
            )
        self._latest_depth_m = depth_m.copy()
        self._latest_recv_time_s = now
        self._latest_mtime_ns = stat.st_mtime_ns
        self._frame_count += 1
        return self._snapshot()

    def _snapshot(self) -> tuple[np.ndarray | None, float | None, int, float]:
        depth = None if self._latest_depth_m is None else self._latest_depth_m.copy()
        return depth, self._latest_recv_time_s, self._frame_count, self._fps

    def close(self) -> None:
        pass


def create_depth_image_source(cfg: DepthViewerConfig):
    if cfg.source == "ros":
        return RosDepthImageSubscriber(cfg)
    if cfg.source == "npy_live":
        return NpyLiveDepthImageSource(cfg)
    raise ValueError("MJLAB_DEPTH_VIEW_SOURCE must be 'ros' or 'npy_live'")


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None or value == "" else float(value)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None or value == "" else int(value)


def config_from_env() -> DepthViewerConfig:
    source = os.getenv("MJLAB_DEPTH_VIEW_SOURCE") or os.getenv("MJLAB_DEPTH_SOURCE", "ros")
    if source == "npy":
        source = "npy_live"
    return DepthViewerConfig(
        source=source,
        ros_topic=os.getenv("MJLAB_DEPTH_ROS_TOPIC", DEFAULT_TOPIC),
        ros_type=os.getenv("ROS_TYPE") or None,
        npy_path=os.getenv("MJLAB_DEPTH_NPY_PATH") or None,
        encoding=os.getenv("MJLAB_DEPTH_ENCODING") or None,
        depth_scale=(
            None
            if not os.getenv("MJLAB_DEPTH_SCALE")
            else float(os.environ["MJLAB_DEPTH_SCALE"])
        ),
        min_depth=_env_float(
            "MJLAB_DEPTH_VIEW_MIN",
            _env_float("MJLAB_DEPTH_MIN", DEFAULT_MIN_DEPTH),
        ),
        max_depth=_env_float(
            "MJLAB_DEPTH_VIEW_MAX",
            _env_float("MJLAB_DEPTH_MAX", DEFAULT_MAX_DEPTH),
        ),
        scale=max(1, _env_int("MJLAB_DEPTH_VIEW_SCALE", DEFAULT_SCALE)),
        refresh_hz=max(1.0, _env_float("MJLAB_DEPTH_VIEW_HZ", DEFAULT_REFRESH_HZ)),
        colormap=os.getenv("MJLAB_DEPTH_VIEW_COLORMAP", "turbo").lower(),
        gradcam_path=os.getenv("MJLAB_DEPTH_GRADCAM_PATH") or None,
        gradcam_alpha=float(np.clip(_env_float("MJLAB_DEPTH_GRADCAM_ALPHA", DEFAULT_GRADCAM_ALPHA), 0.0, 1.0)),
    )


def colorize_depth(
    depth_m: np.ndarray,
    *,
    min_depth: float = DEFAULT_MIN_DEPTH,
    max_depth: float = DEFAULT_MAX_DEPTH,
    colormap: str = "turbo",
) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"depth image must be 2D, got {depth.shape}")
    if max_depth <= min_depth:
        raise ValueError("max_depth must be greater than min_depth")

    finite = np.isfinite(depth)
    valid = finite & (depth > 0.0)
    normalized = np.zeros(depth.shape, dtype=np.float32)
    normalized[valid] = np.clip(
        (depth[valid] - np.float32(min_depth)) / np.float32(max_depth - min_depth),
        0.0,
        1.0,
    )

    if colormap == "gray":
        gray = (normalized * 255.0).astype(np.uint8)
        rgb = np.repeat(gray[..., None], 3, axis=2)
    elif colormap == "turbo":
        rgb = _turbo_colorize(normalized)
    else:
        raise ValueError("MJLAB_DEPTH_VIEW_COLORMAP must be 'turbo' or 'gray'")

    rgb[~valid] = 0
    return rgb


def _turbo_colorize(normalized: np.ndarray) -> np.ndarray:
    positions = normalized * np.float32(len(_TURBO_STOPS) - 1)
    low = np.floor(positions).astype(np.int32)
    high = np.clip(low + 1, 0, len(_TURBO_STOPS) - 1)
    weight = (positions - low.astype(np.float32))[..., None]
    rgb = _TURBO_STOPS[low] * (1.0 - weight) + _TURBO_STOPS[high] * weight
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _resize_bilinear(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Small dependency-free bilinear resize for the 30x45 CAM grid."""

    if image.shape == (height, width):
        return image.astype(np.float32, copy=False)
    rows = np.linspace(0.0, image.shape[0] - 1, height, dtype=np.float32)
    cols = np.linspace(0.0, image.shape[1] - 1, width, dtype=np.float32)
    low_rows = np.floor(rows).astype(np.intp)
    low_cols = np.floor(cols).astype(np.intp)
    high_rows = np.minimum(low_rows + 1, image.shape[0] - 1)
    high_cols = np.minimum(low_cols + 1, image.shape[1] - 1)
    row_weight = (rows - low_rows)[:, None]
    col_weight = (cols - low_cols)[None, :]
    top = image[low_rows[:, None], low_cols[None, :]] * (1.0 - col_weight) + image[
        low_rows[:, None], high_cols[None, :]
    ] * col_weight
    bottom = image[high_rows[:, None], low_cols[None, :]] * (1.0 - col_weight) + image[
        high_rows[:, None], high_cols[None, :]
    ] * col_weight
    return (top * (1.0 - row_weight) + bottom * row_weight).astype(np.float32, copy=False)


def crop_to_policy_fov(depth_m: np.ndarray) -> np.ndarray:
    """Return the camera columns retained by depth-policy preprocessing."""

    depth = np.asarray(depth_m)
    if depth.ndim != 2:
        raise ValueError(f"depth image must be 2D, got {depth.shape}")
    left_crop = int(round(depth.shape[1] * D435_LEFT_CROP_FRACTION))
    if left_crop >= depth.shape[1]:
        raise ValueError("policy crop leaves no depth columns")
    return depth[:, left_crop:]


def overlay_gradcam(depth_rgb: np.ndarray, cam: np.ndarray, alpha: float) -> np.ndarray:
    """Overlay CAM on an RGB image already cropped to the policy FOV."""

    rgb = np.asarray(depth_rgb, dtype=np.uint8).copy()
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"depth_rgb must have shape [H, W, 3], got {rgb.shape}")
    if cam.shape != (30, 45):
        raise ValueError(f"Grad-CAM must have shape (30, 45), got {cam.shape}")
    heat = np.clip(_resize_bilinear(cam, rgb.shape[0], rgb.shape[1]), 0.0, 1.0)
    heat_rgb = np.stack((np.full_like(heat, 255.0), heat * 255.0, np.zeros_like(heat)), axis=-1)
    blend = np.float32(np.clip(alpha, 0.0, 1.0)) * heat[..., None]
    rgb[:] = np.clip(
        rgb.astype(np.float32) * (1.0 - blend) + heat_rgb * blend, 0, 255
    ).astype(np.uint8)
    return rgb


def run_viewer(cfg: DepthViewerConfig) -> None:
    import pygame

    depth_source = create_depth_image_source(cfg)
    gradcam_source = GradCamFileSource(cfg.gradcam_path) if cfg.gradcam_path else None
    pygame.init()
    clock = pygame.time.Clock()
    screen_shape = (
        DEFAULT_WINDOW_WIDTH * cfg.scale,
        DEFAULT_WINDOW_HEIGHT * cfg.scale,
    )
    screen = pygame.display.set_mode(screen_shape)
    screen.fill((0, 0, 0))
    pygame.display.flip()

    try:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            depth_m, recv_time_s, frame_count, fps = depth_source.latest()
            if depth_m is None:
                pygame.display.set_caption(f"Depth viewer waiting: {_source_label(cfg)}")
                clock.tick(cfg.refresh_hz)
                continue

            depth_m = crop_to_policy_fov(depth_m)
            height, width = depth_m.shape
            target_shape = (width * cfg.scale, height * cfg.scale)
            if screen_shape != target_shape:
                screen = pygame.display.set_mode(target_shape)
                screen_shape = target_shape

            rgb = colorize_depth(
                depth_m,
                min_depth=cfg.min_depth,
                max_depth=cfg.max_depth,
                colormap=cfg.colormap,
            )
            cam_status = "off"
            if gradcam_source is not None:
                cam, cam_source_time_s, target, _cam_count, cam_fps = gradcam_source.latest()
                cam_age_s = float("inf") if cam_source_time_s is None else time.time() - cam_source_time_s
                if cam is not None and cam_age_s <= DEFAULT_GRADCAM_MAX_AGE_S:
                    rgb = overlay_gradcam(rgb, cam, cfg.gradcam_alpha)
                    cam_status = f"{target} | {cam_fps:.1f} Hz | age {cam_age_s * 1000.0:.0f} ms"
                elif cam is None:
                    cam_status = "waiting"
                else:
                    cam_status = f"stale | age {cam_age_s * 1000.0:.0f} ms"
            surface = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
            if cfg.scale != 1:
                surface = pygame.transform.scale(surface, target_shape)
            screen.blit(surface, (0, 0))
            pygame.display.flip()

            age_s = 0.0 if recv_time_s is None else time.time() - recv_time_s
            pygame.display.set_caption(
                f"Depth {_source_label(cfg)} | {width}x{height} | "
                f"{fps:.1f} fps | age {age_s * 1000.0:.0f} ms | frame {frame_count} | "
                f"CAM {cam_status}"
            )
            clock.tick(cfg.refresh_hz)
    finally:
        depth_source.close()
        pygame.quit()


def _source_label(cfg: DepthViewerConfig) -> str:
    if cfg.source == "npy_live":
        return cfg.npy_path or "npy_live"
    return cfg.ros_topic


def main() -> None:
    run_viewer(config_from_env())


if __name__ == "__main__":
    main()
