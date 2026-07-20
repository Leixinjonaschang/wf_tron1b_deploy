"""Optional, asynchronous Grad-CAM support for the depth RepTS policy.

This module deliberately does not import torch at module import time.  The
deployment controller imports it unconditionally, but torch is only needed
after ``MJLAB_DEPTH_GRADCAM=1`` has explicitly enabled the feature.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any

import numpy as np


DEFAULT_HZ = 5.0
DEFAULT_ALPHA = 0.45
TARGET_NAME = "actions_l2"


def _enabled_from_env() -> bool:
    value = os.getenv("MJLAB_DEPTH_GRADCAM", "0").strip().lower()
    if value in ("", "0", "false", "no", "off"):
        return False
    if value in ("1", "true", "yes", "on"):
        return True
    raise ValueError("MJLAB_DEPTH_GRADCAM must be 0 or 1")


@dataclass(frozen=True)
class DepthGradCamConfig:
    checkpoint_path: Path
    output_path: Path
    hz: float = DEFAULT_HZ

    @classmethod
    def from_env(cls, policy_path: str | Path) -> "DepthGradCamConfig | None":
        if not _enabled_from_env():
            return None
        policy_path = Path(policy_path)
        checkpoint = Path(
            os.getenv("MJLAB_DEPTH_GRADCAM_CHECKPOINT", policy_path.with_name("policy.pt"))
        )
        default_output = Path("logs/sim2sim/depth_gradcam.npz")
        output = Path(os.getenv("MJLAB_DEPTH_GRADCAM_PATH", default_output))
        hz = float(os.getenv("MJLAB_DEPTH_GRADCAM_HZ", DEFAULT_HZ))
        if not np.isfinite(hz) or hz <= 0.0:
            raise ValueError("MJLAB_DEPTH_GRADCAM_HZ must be a positive finite number")
        return cls(checkpoint_path=checkpoint, output_path=output, hz=hz)


def _torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Grad-CAM requires the optional CPU PyTorch runtime. Install project dependencies "
            "or set MJLAB_DEPTH_GRADCAM=0."
        ) from exc
    return torch


def build_depth_gradcam_policy():
    """Build the inference-only PyTorch counterpart of ``policy.onnx``."""

    torch = _torch()
    nn = torch.nn

    class Normalizer(nn.Module):
        def __init__(self, width: int):
            super().__init__()
            self.register_buffer("_mean", torch.zeros(1, width))
            self.register_buffer("_std", torch.ones(1, width))

        def forward(self, value):
            # Deployment export uses the trainer's empirical-normalization
            # epsilon of 0.01 in addition to the stored standard deviation.
            return (value - self._mean) / (self._std + 1.0e-2)

    class DepthEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=5, stride=2),
                nn.ELU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2),
                nn.ELU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=2),
                nn.ELU(),
            )
            self.projection = nn.Sequential(nn.Flatten(), nn.Linear(256, 64), nn.ELU())

        def forward(self, depth):
            features = self.cnn(depth)
            return self.projection(features), features

    class DepthGradCamPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.proprio_history_obs_normalizer = Normalizer(140)
            self.current_proprio_obs_normalizer = Normalizer(28)
            self.command_obs_normalizer = Normalizer(3)
            self.lin_vel_normalizer = Normalizer(3)
            self.depth_encoder = DepthEncoder()
            self.depth_gru = nn.GRUCell(64, 64)
            self.proprio_encoder = nn.Sequential(
                nn.Linear(204, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(),
                nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 128),
            )
            self.student_latent_head = nn.Linear(128, 64)
            self.lin_vel_head = nn.Linear(128, 3)
            self.actor_head = nn.Sequential(
                nn.Linear(98, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(),
                nn.Linear(256, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(),
                nn.Linear(128, 8),
            )

        def forward(self, proprio_history, actor_command, depth, hidden_state_in):
            depth_embedding, features = self.depth_encoder(depth)
            hidden_state_out = self.depth_gru(depth_embedding, hidden_state_in)
            history = self.proprio_history_obs_normalizer(proprio_history.flatten(1))
            encoded = self.proprio_encoder(torch.cat((history, hidden_state_out), dim=-1))
            latent = self.student_latent_head(encoded)
            predicted_lin_vel = self.lin_vel_head(encoded)
            latent = latent / latent.norm(p=2, dim=-1, keepdim=True).clamp_min(1.0e-6)
            normalized_lin_vel = self.lin_vel_normalizer(predicted_lin_vel)
            current = self.current_proprio_obs_normalizer(proprio_history[:, -1, :])
            command = self.command_obs_normalizer(actor_command)
            actions = self.actor_head(torch.cat((normalized_lin_vel, current, command, latent), dim=-1))
            return actions, predicted_lin_vel, hidden_state_out, features

    return DepthGradCamPolicy()


def load_depth_gradcam_policy(checkpoint_path: str | Path):
    """Load only actor weights with PyTorch's safe weights-only loader."""

    torch = _torch()
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Grad-CAM checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("actor_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Grad-CAM checkpoint must contain actor_state_dict")
    model = build_depth_gradcam_policy()
    needed = set(model.state_dict())
    selected = {key: value for key, value in state_dict.items() if key in needed}
    missing, unexpected = model.load_state_dict(selected, strict=False)
    if missing or unexpected:
        raise ValueError(
            "Grad-CAM checkpoint does not match the deployed depth actor: "
            f"missing={list(missing)}, unexpected={list(unexpected)}"
        )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _as_torch_inputs(inputs: tuple[np.ndarray, ...]):
    torch = _torch()
    return tuple(torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32)) for value in inputs)


def validate_depth_gradcam_policy(model, onnx_session, *, atol: float = 2.0e-4) -> None:
    """Reject an explanation model that is not numerically aligned with ONNX."""

    rng = np.random.default_rng(20260720)
    samples = (
        (
            rng.normal(size=(1, 5, 28)).astype(np.float32),
            rng.normal(size=(1, 3)).astype(np.float32),
            rng.uniform(0.0, 10.0, size=(1, 1, 30, 45)).astype(np.float32),
            np.zeros((1, 64), dtype=np.float32),
        ),
        (
            rng.normal(size=(1, 5, 28)).astype(np.float32),
            rng.normal(size=(1, 3)).astype(np.float32),
            rng.uniform(0.0, 10.0, size=(1, 1, 30, 45)).astype(np.float32),
            rng.normal(size=(1, 64)).astype(np.float32),
        ),
    )
    names = [item.name for item in onnx_session.get_inputs()]
    for sample_index, sample in enumerate(samples):
        expected = onnx_session.run(None, dict(zip(names, sample)))
        with _torch().no_grad():
            actual = model(*_as_torch_inputs(sample))[:3]
        for output_name, torch_value, onnx_value in zip(
            ("actions", "predicted_lin_vel", "hidden_state_out"), actual, expected
        ):
            actual_value = torch_value.detach().cpu().numpy()
            if not np.allclose(actual_value, onnx_value, rtol=atol, atol=atol):
                max_error = float(np.max(np.abs(actual_value - onnx_value)))
                raise ValueError(
                    f"Grad-CAM {output_name} does not match ONNX for sample {sample_index}; "
                    f"maximum absolute error is {max_error:.3e}"
                )


def compute_gradcam(model, inputs: tuple[np.ndarray, ...]) -> np.ndarray:
    """Return non-negative Grad-CAM on the policy's 30x45 depth input grid."""

    torch = _torch()
    proprio, command, depth, hidden = _as_torch_inputs(inputs)
    depth.requires_grad_(True)
    actions, _velocity, _hidden_out, features = model(proprio, command, depth, hidden)
    target = 0.5 * actions.square().sum()
    gradients = torch.autograd.grad(target, features, only_inputs=True)[0]
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * features).sum(dim=1, keepdim=True))
    cam = torch.nn.functional.interpolate(cam, size=(30, 45), mode="bilinear", align_corners=False)
    cam = cam[0, 0]
    maximum = cam.max()
    if torch.isfinite(maximum) and maximum > 0:
        cam = cam / maximum
    else:
        cam = torch.zeros_like(cam)
    result = cam.detach().cpu().numpy().astype(np.float32, copy=False)
    if result.shape != (30, 45) or not np.isfinite(result).all() or (result < 0).any():
        raise ValueError("Grad-CAM output must be finite, non-negative and have shape (30, 45)")
    return result


@dataclass(frozen=True)
class GradCamInput:
    proprio_history: np.ndarray
    actor_command: np.ndarray
    depth: np.ndarray
    hidden_state_in: np.ndarray
    source_time_s: float

    def arrays(self) -> tuple[np.ndarray, ...]:
        return (self.proprio_history, self.actor_command, self.depth, self.hidden_state_in)


class DepthGradCamWorker:
    """A single-slot, latest-frame-wins Grad-CAM worker."""

    def __init__(self, model, output_path: str | Path, hz: float = DEFAULT_HZ):
        self._model = model
        self._output_path = Path(output_path)
        self._period_s = 1.0 / hz
        self._lock = threading.Lock()
        self._pending: GradCamInput | None = None
        self._event = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="depth-gradcam", daemon=True)
        self._thread.start()

    def submit(self, proprio_history, actor_command, depth, hidden_state_in) -> None:
        item = GradCamInput(
            np.asarray(proprio_history, dtype=np.float32).copy(),
            np.asarray(actor_command, dtype=np.float32).copy(),
            np.asarray(depth, dtype=np.float32).copy(),
            np.asarray(hidden_state_in, dtype=np.float32).copy(),
            time.time(),
        )
        with self._lock:
            self._pending = item
        self._event.set()

    def close(self) -> None:
        self._stop.set()
        self._event.set()
        self._thread.join(timeout=1.0)

    def _take_latest(self) -> GradCamInput | None:
        with self._lock:
            item, self._pending = self._pending, None
        return item

    def _run(self) -> None:
        next_allowed = 0.0
        while not self._stop.is_set():
            self._event.wait(timeout=0.1)
            self._event.clear()
            if self._stop.is_set():
                return
            delay = next_allowed - time.monotonic()
            if delay > 0.0:
                if self._stop.wait(delay):
                    return
            item = self._take_latest()
            if item is None:
                continue
            try:
                cam = compute_gradcam(self._model, item.arrays())
                self._publish(cam, item.source_time_s)
            except Exception as exc:  # Explanation failures must never affect control.
                print(f"[mjlab_repts_lin_depth] Grad-CAM worker error: {exc}")
            next_allowed = time.monotonic() + self._period_s

    def _publish(self, cam: np.ndarray, source_time_s: float) -> None:
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=self._output_path.parent,
                prefix=f".{self._output_path.stem}.", suffix=".npz", delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                np.savez(
                    temporary,
                    cam=cam,
                    source_time_s=np.float64(source_time_s),
                    published_time_s=np.float64(time.time()),
                    target=np.array(TARGET_NAME),
                )
            os.replace(temporary_path, self._output_path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink(missing_ok=True)


def create_depth_gradcam_worker(policy_path: str | Path, onnx_session):
    """Return an enabled and ONNX-validated worker, otherwise ``None``.

    Configuration or model problems are reported but intentionally never make
    the real-time ONNX controller unavailable.
    """

    try:
        config = DepthGradCamConfig.from_env(policy_path)
        if config is None:
            return None
        model = load_depth_gradcam_policy(config.checkpoint_path)
        validate_depth_gradcam_policy(model, onnx_session)
        print(
            "[mjlab_repts_lin_depth] Grad-CAM enabled: "
            f"checkpoint={config.checkpoint_path}, path={config.output_path}, hz={config.hz:g}, "
            f"target={TARGET_NAME}"
        )
        return DepthGradCamWorker(model, config.output_path, config.hz)
    except Exception as exc:
        print(f"[mjlab_repts_lin_depth] Grad-CAM disabled: {exc}")
        return None
