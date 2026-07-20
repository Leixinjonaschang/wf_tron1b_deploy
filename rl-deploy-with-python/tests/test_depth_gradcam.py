from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

import numpy as np


DEPLOY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DEPLOY_ROOT.parent
sys.path.insert(0, str(DEPLOY_ROOT))

from depth_gradcam import (  # noqa: E402
    DepthGradCamConfig,
    DepthGradCamWorker,
    compute_gradcam,
    create_depth_gradcam_worker,
    load_depth_gradcam_policy,
    validate_depth_gradcam_policy,
)


POLICY_DIR = DEPLOY_ROOT / "controllers/model/WF_TRON1B/policy/mjlab_repts_lin_depth"
POLICY_ONNX = POLICY_DIR / "policy.onnx"
POLICY_CHECKPOINT = POLICY_DIR / "policy.pt"


def _viewer_module():
    spec = importlib.util.spec_from_file_location(
        "depth_image_viewer_gradcam_test", REPO_ROOT / "scripts/depth_image_viewer.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class DepthGradCamConfigTest(unittest.TestCase):
    def test_disabled_does_not_create_worker_or_import_torch(self):
        with mock.patch.dict(os.environ, {"MJLAB_DEPTH_GRADCAM": "0"}, clear=True):
            self.assertIsNone(DepthGradCamConfig.from_env(POLICY_ONNX))

    def test_gradcam_defaults_to_ten_hz(self):
        with mock.patch.dict(os.environ, {"MJLAB_DEPTH_GRADCAM": "1"}, clear=True):
            self.assertEqual(DepthGradCamConfig.from_env(POLICY_ONNX).hz, 10.0)

    def test_viewer_crop_alignment_and_alpha_blending(self):
        viewer = _viewer_module()
        raw_depth = np.arange(4 * 8, dtype=np.float32).reshape(4, 8)
        cropped_depth = viewer.crop_to_policy_fov(raw_depth)
        self.assertEqual(cropped_depth.shape, (4, 7))
        np.testing.assert_array_equal(cropped_depth, raw_depth[:, 1:])

        depth = np.zeros((4, 7, 3), dtype=np.uint8)
        cam = np.ones((30, 45), dtype=np.float32)
        overlay = viewer.overlay_gradcam(depth, cam, 0.5)
        self.assertTrue(
            np.all(overlay == np.array([127, 127, 0], dtype=np.uint8))
        )

    def test_viewer_crops_d435_raw_frame_to_policy_width(self):
        viewer = _viewer_module()
        raw_depth = np.zeros((480, 848), dtype=np.float32)
        self.assertEqual(viewer.crop_to_policy_fov(raw_depth).shape, (480, 720))

    def test_gradcam_file_reader_ignores_corrupt_and_expired_inputs(self):
        viewer = _viewer_module()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cam.npz"
            path.write_bytes(b"not an npz")
            source = viewer.GradCamFileSource(str(path))
            self.assertIsNone(source.latest()[0])
            np.savez(
                path,
                cam=np.ones((30, 45), dtype=np.float32),
                source_time_s=np.float64(time.time() - 2.0),
                target=np.array("actions_l2"),
            )
            cam, source_time_s, target, _count, _fps = source.latest()
            self.assertEqual(cam.shape, (30, 45))
            self.assertEqual(target, "actions_l2")
            self.assertGreater(time.time() - source_time_s, 1.0)


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires optional torch runtime")
class DepthGradCamPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import onnxruntime as ort

        cls.model = load_depth_gradcam_policy(POLICY_CHECKPOINT)
        cls.session = ort.InferenceSession(str(POLICY_ONNX), providers=["CPUExecutionProvider"])

    def test_checkpoint_rebuild_matches_onnx_and_generates_cam(self):
        validate_depth_gradcam_policy(self.model, self.session)
        rng = np.random.default_rng(14)
        cam = compute_gradcam(
            self.model,
            (
                rng.normal(size=(1, 5, 28)).astype(np.float32),
                rng.normal(size=(1, 3)).astype(np.float32),
                rng.uniform(0.0, 10.0, size=(1, 1, 30, 45)).astype(np.float32),
                rng.normal(size=(1, 64)).astype(np.float32),
            ),
        )
        self.assertEqual(cam.shape, (30, 45))
        self.assertTrue(np.isfinite(cam).all())
        self.assertGreaterEqual(float(cam.min()), 0.0)

    def test_worker_publishes_atomic_npz(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cam.npz"
            worker = DepthGradCamWorker(self.model, path, hz=100.0)
            try:
                worker.submit(
                    np.zeros((1, 5, 28), dtype=np.float32),
                    np.zeros((1, 3), dtype=np.float32),
                    np.ones((1, 1, 30, 45), dtype=np.float32),
                    np.zeros((1, 64), dtype=np.float32),
                )
                deadline = time.monotonic() + 2.0
                while not path.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(path.exists())
                with np.load(path, allow_pickle=False) as payload:
                    self.assertEqual(payload["cam"].shape, (30, 45))
                    self.assertEqual(str(payload["target"].item()), "actions_l2")
            finally:
                worker.close()

    def test_disabled_factory_returns_none(self):
        with mock.patch.dict(os.environ, {"MJLAB_DEPTH_GRADCAM": "0"}, clear=True):
            self.assertIsNone(create_depth_gradcam_worker(POLICY_ONNX, self.session))
