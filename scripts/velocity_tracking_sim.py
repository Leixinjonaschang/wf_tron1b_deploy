"""Synchronous MuJoCo adapter for the production WF_TRON1B controller.

No robot socket is opened. Physics, control, depth and sampling use simulation
time; this evaluates the policy, not ROS/SDK transport latency.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "rl-deploy-with-python"
sys.path.insert(0, str(DEPLOY))

import limxsdk.datatypes as datatypes  # noqa: E402
from controllers.WheelfootController import WheelfootController  # noqa: E402
from mjlab_repts import SDK_JOINT_NAMES  # noqa: E402
from mjlab_repts_lin_depth import preprocess_depth_image  # noqa: E402


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def body_linear_velocity(qvel, rotation):
    """Free-joint translation is world-frame velocity of the base origin."""
    return np.asarray(rotation).reshape(3, 3).T @ np.asarray(qvel)[:3]


class LocalRobot:
    """Only the SDK callback surface used by WheelfootController."""

    def subscribeRobotState(self, callback):
        self.state_callback = callback

    def subscribeImuData(self, callback):
        self.imu_callback = callback

    def subscribeSensorJoy(self, callback):
        pass

    def subscribeDiagnosticValue(self, callback):
        pass

    def publishRobotCmd(self, command):
        self.command = command


class RenderedDepth:
    def __init__(self, model):
        self.renderer = mujoco.Renderer(model, height=480, width=848)
        self.renderer.enable_depth_rendering()
        self.options = mujoco.MjvOption()
        self.options.geomgroup[:] = 0
        self.options.geomgroup[:2] = 1
        self.latest = None

    def capture(self, data):
        self.renderer.update_scene(data, camera="d435", scene_option=self.options)
        raw = self.renderer.render()
        # Match simulator.py's ROS1 16UC1 wire quantization without a ROS clock.
        millimeters = np.clip(
            np.nan_to_num(raw, nan=0.0, posinf=65.535, neginf=0.0) * 1000,
            0, 65535,
        ).astype(np.uint16)
        # Use the deployed GRU policy profile: clip to [0.2, 2.0] m and
        # normalize to [0, 1] before inference, then crop and resize.
        self.latest = preprocess_depth_image(millimeters, encoding="16UC1")

    def frame(self):
        if self.latest is None:
            raise RuntimeError("No depth frame captured for this trial")
        return self.latest

    def close(self):
        self.renderer.close()


class TrackingSimulation:
    def __init__(self, rl_type, min_height=0.35, max_tilt_deg=60.0):
        self.rl_type = rl_type
        self.scene = (
            ROOT / "pointfoot-mujoco-sim/robot-description/pointfoot"
            / "WF_TRON1B/xml/robot.xml"
        )
        self.model = mujoco.MjModel.from_xml_path(str(self.scene))
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep
        self.body_id = self.model.body("base_Link").id
        if self.model.nq != 15 or self.model.nv != 14:
            raise ValueError("Expected WF_TRON1B with one free base and eight joints")
        self.q_indices = [self.model.joint(n).qposadr[0] for n in SDK_JOINT_NAMES]
        self.v_indices = [self.model.joint(n).dofadr[0] for n in SDK_JOINT_NAMES]
        self.actuator_ids = [self.model.actuator(n).id for n in SDK_JOINT_NAMES]
        self.min_height = min_height
        self.min_upright = np.cos(np.deg2rad(max_tilt_deg))
        self.depth = None
        if rl_type.endswith("_depth"):
            self.depth = RenderedDepth(self.model)
        self.depth_steps = max(1, round(1 / (30 * self.dt)))
        self.controller = None

    def reset(self, seed):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        if self.depth:
            self.depth.latest = None
            self.depth.capture(self.data)
        self.robot = LocalRobot()
        self.controller = WheelfootController(
            str(DEPLOY / "controllers/model"), self.robot, "WF_TRON1B",
            self.rl_type, True, depth_source=self.depth,
        )
        c = self.controller
        c.mjlab_repts_obs_noise_rng = np.random.default_rng(seed)
        c.mjlab_repts_diagnostics_printed = True
        # Evaluate from the XML nominal pose with the policy active immediately.
        # The manual deployment STAND interpolation is a separate startup task.
        c.default_joint_angles = np.zeros(c.joint_num)
        c.mode = "WALK"
        self.control_steps = round(1 / (c.loop_frequency * self.dt))
        if not np.isclose(self.control_steps * self.dt * c.loop_frequency, 1):
            raise ValueError("Control period must be an integer number of physics steps")
        self.sample_steps = self.control_steps * c.control_cfg["decimation"]
        self.sample_dt = self.sample_steps * self.dt
        self.steps = 0
        self.robot.command = c.robot_cmd
        self.state = datatypes.RobotState()
        self.imu = datatypes.ImuData()

    def metadata(self):
        c = self.controller
        mesh_dir = self.scene.parent.parent / "meshes"
        return {
            "backend": "synchronous_mujoco_production_controller",
            "startup": "XML qpos0, zero qvel, WALK policy immediately, zero-command warmup",
            "scene": str(self.scene.relative_to(ROOT)),
            "scene_sha256": sha256(self.scene),
            "mesh_sha256": {p.name: sha256(p) for p in sorted(mesh_dir.iterdir())
                            if p.is_file()},
            "policy": str(Path(c.model_policy).relative_to(ROOT)),
            "policy_sha256": sha256(c.model_policy),
            "controller_sha256": sha256(DEPLOY / "controllers/WheelfootController.py"),
            "config_sha256": sha256(c.config_file),
            "config_yaml": Path(c.config_file).read_text(),
            "onnx_metadata": c.policy_metadata,
            "mujoco_version": mujoco.__version__,
            "physics_dt_s": self.dt,
            "control_hz": c.loop_frequency,
            "sample_dt_s": self.sample_dt,
            "policy_hz": 1 / self.sample_dt,
            "observation_noise_enabled": c.mjlab_repts_obs_noise_enabled,
            "observation_noise_ranges": c.mjlab_repts_obs_noise_ranges,
            "depth": None if not self.depth else {
                "raw_shape": [480, 848], "input_shape": [1, 1, 30, 45],
                "capture_period_s": self.depth_steps * self.dt,
                "transport": "in_process_with_ROS1_16UC1_quantization",
            },
        }

    def _publish_state(self):
        self.state.q = self.data.qpos[self.q_indices].tolist()
        self.state.dq = self.data.qvel[self.v_indices].tolist()
        self.state.tau = self.data.ctrl[self.actuator_ids].tolist()
        self.state.stamp = round(self.data.time * 1e9)
        self.robot.state_callback(self.state)
        self.imu.quat = self.data.sensor("quat").data.tolist()
        self.imu.gyro = self.data.sensor("gyro").data.tolist()
        self.imu.acc = self.data.sensor("acc").data.tolist()
        self.imu.stamp = self.state.stamp
        self.robot.imu_callback(self.imu)

    def failure_reason(self):
        if not (np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()):
            return "nonfinite_state"
        if self.data.xpos[self.body_id, 2] < self.min_height:
            return "base_below_min_height"
        if self.data.xmat[self.body_id, 8] < self.min_upright:
            return "base_tilt_exceeded"
        if any(w.number for w in self.data.warning):
            return "mujoco_warning"
        return ""

    def step(self, command):
        """Advance one policy period (or stop earlier on a failed trial)."""
        self.controller.commands = np.asarray(command, dtype=np.float32)
        for _ in range(self.sample_steps):
            if self.depth and self.steps % self.depth_steps == 0:
                self.depth.capture(self.data)
            if self.steps % self.control_steps == 0:
                self._publish_state()
                self.controller.update()
            cmd = self.robot.command
            torque = (
                np.asarray(cmd.Kp) * (np.asarray(cmd.q) - self.data.qpos[self.q_indices])
                + np.asarray(cmd.Kd) * (np.asarray(cmd.dq) - self.data.qvel[self.v_indices])
                + np.asarray(cmd.tau)
            )
            if not np.isfinite(torque).all():
                return "nonfinite_action"
            # Same PD law as simulator.py; motor ctrlrange enforces XML limits.
            self.data.ctrl[self.actuator_ids] = torque
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            self.steps += 1
            reason = self.failure_reason()
            if reason:
                return reason
        return ""

    def observation(self):
        rotation = self.data.xmat[self.body_id].reshape(3, 3)
        velocity = body_linear_velocity(self.data.qvel, rotation)
        pos = self.data.xpos[self.body_id]
        return {
            "sim_time_s": float(self.data.time),
            "vx_body_mps": float(velocity[0]),
            "vy_body_mps": float(velocity[1]),
            "vz_body_mps": float(velocity[2]),
            "vx_world_mps": float(self.data.qvel[0]),
            "vy_world_mps": float(self.data.qvel[1]),
            "x_m": float(pos[0]), "y_m": float(pos[1]), "height_m": float(pos[2]),
            "yaw_rad": float(np.arctan2(rotation[1, 0], rotation[0, 0])),
            "tilt_deg": float(np.rad2deg(np.arccos(np.clip(rotation[2, 2], -1, 1)))),
        }

    def close(self):
        if self.depth:
            self.depth.close()
