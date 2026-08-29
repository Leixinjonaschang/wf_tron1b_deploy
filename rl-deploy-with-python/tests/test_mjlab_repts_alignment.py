from __future__ import annotations

import contextlib
import io
import importlib
import sys
import unittest
from pathlib import Path
from unittest import mock
import types

import numpy as np

try:
    import onnxruntime as ort
except ModuleNotFoundError:
    ort = None

try:
    import onnx
except ModuleNotFoundError:
    onnx = None


DEPLOY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEPLOY_ROOT))

from mjlab_repts import (  # noqa: E402
    ACTION_CLIP,
    DEFAULT_OBS_NOISE_RANGES,
    POLICY_ACTION_NAMES,
    SDK_JOINT_NAMES,
    STUDENT_HISTORY_SHAPE,
    TERM_DIMS,
    TERM_ORDER,
    TermWiseHistory,
    build_actor_obs,
    build_actor_terms,
    clip_actions,
    command_from_joystick_axes,
    map_actions_to_sdk_joint_commands,
    validate_policy_interface,
)


MODEL_DIR = DEPLOY_ROOT / "controllers" / "model"
POLICY_PATH = MODEL_DIR / "WF_TRON1B" / "policy" / "mjlab_repts" / "policy.onnx"
OLD_POLICY_PATHS = (
    MODEL_DIR / "WF_TRON1B" / "policy" / "mjlab_repts" / "policy_1.onnx",
    MODEL_DIR
    / "WF_TRON1B"
    / "policy"
    / "mjlab_repts"
    / "policy_larger_action_rate.onnx",
)
_WHEELFOOT_CLASS = None


def _make_terms(step: int) -> dict[str, np.ndarray]:
    return {
        name: np.full(dim, step * 10 + index, dtype=np.float32)
        for index, (name, dim) in enumerate(
            (name, TERM_DIMS[name]) for name in TERM_ORDER
        )
    }


def _onnx_shape(value_info) -> list[int | str]:
    return [
        dim.dim_value if dim.dim_value else dim.dim_param
        for dim in value_info.type.tensor_type.shape.dim
    ]


def _load_policy_interface(policy_path: Path):
    if ort is not None:
        session = ort.InferenceSession(
            str(policy_path), providers=["CPUExecutionProvider"]
        )
        return (
            [input_info.name for input_info in session.get_inputs()],
            [input_info.shape for input_info in session.get_inputs()],
            [output_info.name for output_info in session.get_outputs()],
            [output_info.shape for output_info in session.get_outputs()],
            session.get_modelmeta().custom_metadata_map,
        )

    if onnx is None:
        raise unittest.SkipTest(
            "onnx or onnxruntime is required to inspect policy.onnx"
        )

    model = onnx.load(policy_path)
    return (
        [input_info.name for input_info in model.graph.input],
        [_onnx_shape(input_info) for input_info in model.graph.input],
        [output_info.name for output_info in model.graph.output],
        [_onnx_shape(output_info) for output_info in model.graph.output],
        {prop.key: prop.value for prop in model.metadata_props},
    )


class FakeRobot:
    def subscribeRobotState(self, callback):
        self.robot_state_callback = callback

    def subscribeImuData(self, callback):
        self.imu_data_callback = callback

    def subscribeSensorJoy(self, callback):
        self.sensor_joy_callback = callback

    def subscribeDiagnosticValue(self, callback):
        self.diagnostic_callback = callback


class FakeRobotCmd:
    def __init__(self):
        self.mode = []
        self.q = []
        self.dq = []
        self.tau = []
        self.Kp = []
        self.Kd = []


class FakeRobotState:
    def __init__(self):
        self.stamp = 0
        self.q = []
        self.dq = []
        self.tau = []


class FakeImuData:
    def __init__(self):
        self.stamp = 0
        self.acc = [0.0, 0.0, 0.0]
        self.gyro = [0.0, 0.0, 0.0]
        self.quat = [0.0, 0.0, 0.0, 1.0]


class FakeSensorJoy:
    pass


class FakeDiagnosticValue:
    pass


def _fake_limxsdk_modules():
    limxsdk = types.ModuleType("limxsdk")
    robot = types.ModuleType("limxsdk.robot")
    rate = types.ModuleType("limxsdk.robot.Rate")
    robot_module = types.ModuleType("limxsdk.robot.Robot")
    robot_type = types.ModuleType("limxsdk.robot.RobotType")
    datatypes = types.ModuleType("limxsdk.datatypes")

    datatypes.RobotCmd = FakeRobotCmd
    datatypes.RobotState = FakeRobotState
    datatypes.ImuData = FakeImuData
    datatypes.SensorJoy = FakeSensorJoy
    datatypes.DiagnosticValue = FakeDiagnosticValue

    return {
        "limxsdk": limxsdk,
        "limxsdk.robot": robot,
        "limxsdk.robot.Rate": rate,
        "limxsdk.robot.Robot": robot_module,
        "limxsdk.robot.RobotType": robot_type,
        "limxsdk.datatypes": datatypes,
    }


def _import_wheelfoot_class():
    global _WHEELFOOT_CLASS
    if _WHEELFOOT_CLASS is not None:
        return _WHEELFOOT_CLASS
    with mock.patch.dict(sys.modules, _fake_limxsdk_modules()):
        module = importlib.import_module("controllers.WheelfootController")
    _WHEELFOOT_CLASS = module.WheelfootController
    return _WHEELFOOT_CLASS


class MaxNoiseRng:
    def uniform(self, low, high, size):
        return np.full(size, high, dtype=np.float32)


class MjlabRepTsAlignmentTest(unittest.TestCase):
    def test_exported_policy_onnx_interface(self):
        input_names, input_shapes, output_names, output_shapes, metadata = (
            _load_policy_interface(POLICY_PATH)
        )

        validate_policy_interface(
            input_names, input_shapes, output_names, output_shapes, metadata
        )

    def test_old_mjlab_repts_policy_interfaces_are_rejected(self):
        for policy_path in OLD_POLICY_PATHS:
            with self.subTest(policy=policy_path.name):
                input_names, input_shapes, output_names, output_shapes, metadata = (
                    _load_policy_interface(policy_path)
                )
                with self.assertRaisesRegex(ValueError, "inputs"):
                    validate_policy_interface(
                        input_names,
                        input_shapes,
                        output_names,
                        output_shapes,
                        metadata,
                    )

    def test_actor_obs_slices_match_mjlab_student_order(self):
        base_ang_vel = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        projected_gravity = np.array([0.1, -0.2, -0.97], dtype=np.float32)
        joint_pos = np.arange(8, dtype=np.float32)
        joint_vel = np.arange(10, 18, dtype=np.float32)
        default_joint_pos = np.full(8, 0.5, dtype=np.float32)
        default_joint_vel = np.arange(1, 9, dtype=np.float32)
        last_action = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
        command = np.array([0.3, -0.4, 1.2], dtype=np.float32)

        terms = build_actor_terms(
            base_ang_vel,
            projected_gravity,
            joint_pos,
            joint_vel,
            default_joint_pos,
            default_joint_vel,
            last_action,
            command,
        )
        actor_obs = build_actor_obs(terms)
        leg_joint_indexes = [0, 1, 2, 4, 5, 6]

        np.testing.assert_allclose(actor_obs[0:3], base_ang_vel)
        np.testing.assert_allclose(actor_obs[3:6], projected_gravity)
        np.testing.assert_allclose(
            actor_obs[6:12],
            joint_pos[leg_joint_indexes] - default_joint_pos[leg_joint_indexes],
        )
        np.testing.assert_allclose(
            actor_obs[12:18],
            (joint_vel[leg_joint_indexes] - default_joint_vel[leg_joint_indexes]) * 0.05,
        )
        np.testing.assert_allclose(actor_obs[18:20], joint_vel[[3, 7]] * 0.5)
        np.testing.assert_allclose(actor_obs[20:28], last_action)
        np.testing.assert_allclose(actor_obs[28:31], command)

    def test_actor_obs_noise_matches_training_ranges_before_scale(self):
        base_ang_vel = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        projected_gravity = np.array([0.1, -0.2, -0.97], dtype=np.float32)
        joint_pos = np.arange(8, dtype=np.float32)
        joint_vel = np.arange(10, 18, dtype=np.float32)
        default_joint_pos = np.full(8, 0.5, dtype=np.float32)
        default_joint_vel = np.arange(1, 9, dtype=np.float32)
        last_action = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
        command = np.array([0.3, -0.4, 1.2], dtype=np.float32)

        terms = build_actor_terms(
            base_ang_vel,
            projected_gravity,
            joint_pos,
            joint_vel,
            default_joint_pos,
            default_joint_vel,
            last_action,
            command,
            noise_ranges=DEFAULT_OBS_NOISE_RANGES,
            rng=MaxNoiseRng(),
        )
        actor_obs = build_actor_obs(terms)
        leg_joint_indexes = [0, 1, 2, 4, 5, 6]

        np.testing.assert_allclose(actor_obs[0:3], base_ang_vel + 0.2)
        np.testing.assert_allclose(actor_obs[3:6], projected_gravity + 0.05)
        np.testing.assert_allclose(
            actor_obs[6:12],
            joint_pos[leg_joint_indexes] - default_joint_pos[leg_joint_indexes] + 0.01,
        )
        np.testing.assert_allclose(
            actor_obs[12:18],
            (joint_vel[leg_joint_indexes] - default_joint_vel[leg_joint_indexes] + 1.5)
            * 0.05,
        )
        np.testing.assert_allclose(actor_obs[18:20], (joint_vel[[3, 7]] + 0.5) * 0.5)
        np.testing.assert_allclose(actor_obs[20:28], last_action)
        np.testing.assert_allclose(actor_obs[28:31], command)

    def test_policy_metadata_observation_layout_includes_wheel_vel(self):
        validate_policy_interface(
            ["student_history"],
            [[1, 5, 31]],
            ["actions"],
            [[1, 8]],
            {
                "student_observation_names": ",".join(TERM_ORDER),
                "student_history_length": "5",
                "student_history_flatten_dim": "false",
                "student_history_order": "oldest_to_newest",
            },
        )

    def test_student_history_is_frame_wise_oldest_to_newest_history(self):
        history = TermWiseHistory()

        for step in range(5):
            history.update(_make_terms(step))

        student_history = history.matrix()
        for step in range(5):
            offset = 0
            for term_index, name in enumerate(TERM_ORDER):
                dim = TERM_DIMS[name]
                expected = np.full(dim, step * 10 + term_index, dtype=np.float32)
                np.testing.assert_allclose(
                    student_history[step, offset : offset + dim], expected
                )
                offset += dim

        self.assertEqual(student_history.shape, STUDENT_HISTORY_SHAPE)

    def test_first_student_history_backfills_current_terms(self):
        history = TermWiseHistory()
        history.update(_make_terms(7))
        student_history = history.matrix()

        for step in range(5):
            offset = 0
            for term_index, name in enumerate(TERM_ORDER):
                dim = TERM_DIMS[name]
                expected = np.full(dim, 70 + term_index, dtype=np.float32)
                np.testing.assert_allclose(
                    student_history[step, offset : offset + dim], expected
                )
                offset += dim

    def test_joystick_command_scales_forward_velocity_to_two_meters_per_second(self):
        command = command_from_joystick_axes(np.array([-2.0, 0.5, 2.0], dtype=np.float32))

        np.testing.assert_allclose(command, np.array([1.0, -1.0, np.pi / 2.0], dtype=np.float32))

        max_forward = command_from_joystick_axes(
            np.array([0.0, 1.0, 0.0], dtype=np.float32)
        )
        np.testing.assert_allclose(max_forward, np.array([2.0, 0.0, 0.0], dtype=np.float32))

    def test_joystick_command_keeps_max_reverse_velocity_at_minus_one(self):
        max_reverse = command_from_joystick_axes(
            np.array([0.0, -1.0, 0.0], dtype=np.float32)
        )

        np.testing.assert_allclose(max_reverse, np.array([-1.0, 0.0, 0.0], dtype=np.float32))

    def test_actions_clip_to_mjlab_student_range(self):
        actions = clip_actions(np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 2.5, 3.0]))

        np.testing.assert_allclose(
            actions,
            np.array([-ACTION_CLIP, -2.0, -1.0, 0.0, 1.0, 2.0, ACTION_CLIP, ACTION_CLIP]),
        )

    def test_policy_actions_map_to_sdk_joint_commands_by_name(self):
        actions = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float32)
        default_q = np.arange(8, dtype=np.float32)

        commands = map_actions_to_sdk_joint_commands(actions, default_q)
        q_by_name = dict(zip(SDK_JOINT_NAMES, commands["q"]))
        dq_by_name = dict(zip(SDK_JOINT_NAMES, commands["dq"]))
        kp_by_name = dict(zip(SDK_JOINT_NAMES, commands["Kp"]))
        kd_by_name = dict(zip(SDK_JOINT_NAMES, commands["Kd"]))

        self.assertEqual(POLICY_ACTION_NAMES[3], "abad_R_Joint")
        np.testing.assert_allclose(q_by_name["abad_L_Joint"], 0.0 + 0.5 * actions[0])
        np.testing.assert_allclose(q_by_name["hip_L_Joint"], 1.0 + 0.5 * actions[1])
        np.testing.assert_allclose(q_by_name["knee_L_Joint"], 2.0 + 0.5 * actions[2])
        np.testing.assert_allclose(q_by_name["abad_R_Joint"], 4.0 + 0.5 * actions[3])
        np.testing.assert_allclose(q_by_name["hip_R_Joint"], 5.0 + 0.5 * actions[4])
        np.testing.assert_allclose(q_by_name["knee_R_Joint"], 6.0 + 0.5 * actions[5])
        np.testing.assert_allclose(dq_by_name["wheel_L_Joint"], 10.0 * actions[6])
        np.testing.assert_allclose(dq_by_name["wheel_R_Joint"], 10.0 * actions[7])
        np.testing.assert_allclose(kp_by_name["abad_L_Joint"], 40.0)
        np.testing.assert_allclose(kd_by_name["knee_R_Joint"], 1.8)
        np.testing.assert_allclose(kp_by_name["wheel_L_Joint"], 0.0)
        np.testing.assert_allclose(kd_by_name["wheel_R_Joint"], 0.5)

    def test_controller_branch_loads_policy_without_encoder(self):
        if ort is None:
            self.skipTest("onnxruntime is required to instantiate WheelfootController")

        WheelfootController = _import_wheelfoot_class()

        controller = WheelfootController(
            str(MODEL_DIR),
            FakeRobot(),
            "WF_TRON1B",
            "mjlab_repts",
            start_controller=False,
        )

        self.assertTrue(controller.config_file.endswith("params_mjlab_repts.yaml"))
        self.assertIsNone(controller.model_encoder)
        self.assertIsNone(controller.encoder_session)
        self.assertEqual(controller.policy_input_names, ["student_history"])

    def test_controller_reads_action_scale_from_policy_metadata(self):
        if ort is None:
            self.skipTest("onnxruntime is required to import WheelfootController")

        WheelfootController = _import_wheelfoot_class()

        controller = WheelfootController.__new__(WheelfootController)
        controller.actions_size = 8
        controller.policy_metadata = {
            "action_target_names": ",".join(POLICY_ACTION_NAMES),
            "action_scale": "0.5,0.5,0.5,0.5,0.5,0.5,10.0,10.0",
        }
        controller.mjlab_repts_leg_action_scale = 1.0
        controller.mjlab_repts_wheel_action_scale = 1.0

        controller.apply_mjlab_repts_policy_metadata()

        self.assertEqual(controller.mjlab_repts_leg_action_scale, 0.5)
        self.assertEqual(controller.mjlab_repts_wheel_action_scale, 10.0)

    def test_controller_first_walk_step_runs_policy_off_decimation_boundary(self):
        if ort is None:
            self.skipTest("onnxruntime is required to instantiate WheelfootController")

        WheelfootController = _import_wheelfoot_class()

        controller = WheelfootController(
            str(MODEL_DIR),
            FakeRobot(),
            "WF_TRON1B",
            "mjlab_repts",
            start_controller=False,
        )

        controller.loop_count = 3
        with contextlib.redirect_stdout(io.StringIO()) as output:
            controller.handle_mjlab_repts_walk_mode()

        self.assertTrue(controller.mjlab_repts_policy_initialized)
        self.assertTrue(controller.mjlab_repts_diagnostics_printed)
        self.assertIn("encoder: disabled", output.getvalue())
        self.assertEqual(controller.observations.shape, (31,))
        self.assertEqual(controller.proprio_history_vector.shape, STUDENT_HISTORY_SHAPE)


if __name__ == "__main__":
    unittest.main()
