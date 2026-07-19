"""Pure NumPy helpers for rendering depth images in simulator viewers."""

from __future__ import annotations

import numpy as np


_TURBO_STOPS = np.array(
    [[48, 18, 59], [50, 101, 194], [43, 180, 233], [105, 221, 113],
     [238, 218, 38], [230, 107, 29], [122, 4, 3]],
    dtype=np.float32,
)


def colorize_depth(depth_m: np.ndarray, *, min_depth: float = 0.0,
                   max_depth: float = 10.0, colormap: str = "turbo") -> np.ndarray:
    """Convert a two-dimensional depth image in metres to contiguous RGB uint8."""
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"depth image must be 2D, got {depth.shape}")
    if not np.isfinite(min_depth) or not np.isfinite(max_depth) or max_depth <= min_depth:
        raise ValueError("max_depth must be finite and greater than min_depth")
    valid = np.isfinite(depth) & (depth > 0.0)
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
        positions = normalized * np.float32(len(_TURBO_STOPS) - 1)
        low = np.floor(positions).astype(np.int32)
        high = np.clip(low + 1, 0, len(_TURBO_STOPS) - 1)
        weight = (positions - low.astype(np.float32))[..., None]
        rgb = _TURBO_STOPS[low] * (1.0 - weight) + _TURBO_STOPS[high] * weight
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    else:
        raise ValueError("colormap must be 'turbo' or 'gray'")
    rgb[~valid] = 0
    return np.ascontiguousarray(rgb)


def resize_rgb_nearest(rgb: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize RGB without interpolating policy-input pixels."""
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"rgb image must have shape [H, W, 3], got {image.shape}")
    if height <= 0 or width <= 0:
        raise ValueError("resize dimensions must be positive")
    if image.shape[:2] == (height, width):
        return np.ascontiguousarray(image)
    row_idx = np.floor(np.arange(height) * image.shape[0] / height).astype(np.int64)
    col_idx = np.floor(np.arange(width) * image.shape[1] / width).astype(np.int64)
    return np.ascontiguousarray(image[row_idx[:, None], col_idx[None, :]])


def fit_overlay_rect(source_shape: tuple[int, int], viewport_size: tuple[int, int],
                     preferred_size: tuple[int, int], margin: int) -> tuple[int, int, int, int]:
    """Return a bottom-right (x, y, width, height) rectangle inside a viewport."""
    source_height, source_width = source_shape
    viewport_width, viewport_height = viewport_size
    preferred_width, preferred_height = preferred_size
    if min(source_height, source_width, viewport_width, viewport_height) <= 0:
        raise ValueError("source and viewport dimensions must be positive")
    if min(preferred_width, preferred_height) <= 0 or margin < 0:
        raise ValueError("preferred dimensions must be positive and margin non-negative")
    max_width = max(1, viewport_width - 2 * margin)
    max_height = max(1, viewport_height - 2 * margin)
    scale = min(preferred_width / source_width, preferred_height / source_height,
                max_width / source_width, max_height / source_height)
    width = max(1, int(round(source_width * scale)))
    height = max(1, int(round(source_height * scale)))
    return max(0, viewport_width - margin - width), max(0, margin), width, height
