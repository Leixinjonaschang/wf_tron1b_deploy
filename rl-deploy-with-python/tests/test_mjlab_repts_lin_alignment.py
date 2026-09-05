from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

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
    LIN_PROPRIO_HISTORY_SHAPE,
    LIN_PROPRIO_OBS_SIZE,
    POLICY_ACTION_NAMES,
    PROPRIO_TERM_ORDER,
    TERM_DIMS,
    LinProprioHistory,
    build_actor_terms,
    build_lin_proprio_obs,
    validate_lin_policy_interface,
)


MODEL_DIR = DEPLOY_ROOT / "controllers" / "model"
POLICY_PATH = MODEL_DIR / "WF_TRON1B" / "policy" / "mjlab_repts_lin" / "policy.onnx"
_WHEELFOOT_MODULE = None


def _make_terms(step: int) -> dict[str, np.ndarray]:
    terms = {
        name: np.full(dim, step * 10 + index, dtype=np.float32)
        for index, (name, dim) in enumerate(
            (name, TERM_DIMS[name]) for name in PROPRIO_TERM_ORDER
        )
    }
    terms["command"] = np.array([101.0, 102.0, 103.0], dtype=np.float32)
    return terms


def _valid_metadata() -> dict[str, str]:
    return {
        "observation_names": ",".join(PROPRIO_TERM_ORDER),
        "student_observation_names": ",".join(PROPRIO_TERM_ORDER),
        "command_observation_names": "command",
        "policy_input_names": "proprio_history,actor_command",
        "policy_output_names": "actions,predicted_lin_vel",
        "student_history_length": "5",
        "student_history_flatten_dim": "false",
        "student_history_order": "oldest_to_newest",
        "action_target_names": ",".join(POLICY_ACTION_NAMES),
        "action_scale": "0.5,0.5,0.5,0.5,0.5,0.5,10.0,10.0",
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


class FakePolicySession:
    def __init__(self, path, *args, **kwargs):
        del args, kwargs
        self.path = str(path)

    def get_inputs(self):
        return [
            SimpleNamespace(name="proprio_history", shape=[1, 5, 28]),
            SimpleNamespace(name="actor_command", shape=[1, 3]),
        ]

    def get_outputs(self):
        return [
            SimpleNamespace(name="actions", shape=[1, 8]),
            SimpleNamespace(name="predicted_lin_vel", shape=[1, 3]),
        ]

    def get_modelmeta(self):
        return SimpleNamespace(custom_metadata_map=_valid_metadata())

    def run(self, output_names, inputs):
        del output_names, inputs
        return [
            np.arange(8, dtype=np.float32).reshape(1, 8),
            np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
        ]


class RecordingSession:
    def __init__(self):
        self.calls = []

    def run(self, output_names, inputs):
        self.calls.append((output_names, inputs))
        return [
            np.arange(8, dtype=np.float32).reshape(1, 8),
            np.array([[0.4, 0.5, 0.6]], dtype=np.float32),
        ]


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


def _import_wheelfoot_module():
    global _WHEELFOOT_MODULE
    if _WHEELFOOT_MODULE is not None:
        return _WHEELFOOT_MODULE
    with mock.patch.dict(sys.modules, _fake_limxsdk_modules()):
        _WHEELFOOT_MODULE = importlib.import_module("controllers.WheelfootController")
    return _WHEELFOOT_MODULE


class MjlabRepTsLinAlignmentTest(unittest.TestCase):
    def test_exported_policy_onnx_interface(self):
        input_names, input_shapes, output_names, output_shapes, metadata = (
            _load_policy_interface(POLICY_PATH)
        )

        validate_lin_policy_interface(
            input_names, input_shapes, output_names, output_shapes, metadata
        )

    def test_lin_proprio_observation_uses_28_dim_term_order_without_command(self):
        terms = _make_terms(2)

        proprio_obs = build_lin_proprio_obs(terms)

        self.assertEqual(proprio_obs.shape, (LIN_PROPRIO_OBS_SIZE,))
        offset = 0
        for term_index, name in enumerate(PROPRIO_TERM_ORDER):
            dim = TERM_DIMS[name]
            expected = np.full(dim, 20 + term_index, dtype=np.float32)
            np.testing.assert_allclose(proprio_obs[offset : offset + dim], expected)
            offset += dim
        self.assertEqual(offset, 28)
        self.assertNotIn(101.0, proprio_obs)

    def test_lin_wheel_velocity_uses_training_observation_scale(self):
        joint_vel = np.arange(8, dtype=np.float32)
        terms = build_actor_terms(
            base_ang_vel=np.zeros(3, dtype=np.float32),
            projected_gravity=np.zeros(3, dtype=np.float32),
            joint_pos=np.zeros(8, dtype=np.float32),
            joint_vel=joint_vel,
            default_joint_pos=np.zeros(8, dtype=np.float32),
            default_joint_vel=np.zeros(8, dtype=np.float32),
            last_action=np.zeros(8, dtype=np.float32),
            command=np.zeros(3, dtype=np.float32),
        )

        np.testing.assert_allclose(terms["wheel_vel"], joint_vel[[3, 7]] * 0.05)

    def test_lin_proprio_history_is_oldest_to_newest_and_backfills_first_frame(self):
        history = LinProprioHistory()

        history.update(_make_terms(7))
        matrix = history.matrix()

        self.assertEqual(matrix.shape, LIN_PROPRIO_HISTORY_SHAPE)
        np.testing.assert_allclose(matrix[0], matrix[-1])

        for step in range(8, 12):
            history.update(_make_terms(step))

        matrix = history.matrix()
        for row, step in enumerate(range(7, 12)):
            np.testing.assert_allclose(matrix[row], build_lin_proprio_obs(_make_terms(step)))

    def test_validate_lin_policy_interface_accepts_export_contract(self):
        validate_lin_policy_interface(
            ["proprio_history", "actor_command"],
            [[1, 5, 28], [1, 3]],
            ["actions", "predicted_lin_vel"],
            [[1, 8], [1, 3]],
            _valid_metadata(),
        )

    def test_validate_lin_policy_interface_rejects_bad_name_or_shape(self):
        with self.assertRaisesRegex(ValueError, "inputs"):
            validate_lin_policy_interface(
                ["student_history", "actor_command"],
                [[1, 5, 28], [1, 3]],
                ["actions", "predicted_lin_vel"],
                [[1, 8], [1, 3]],
                _valid_metadata(),
            )

        with self.assertRaisesRegex(ValueError, "input shapes"):
            validate_lin_policy_interface(
                ["proprio_history", "actor_command"],
                [[1, 5, 31], [1, 3]],
                ["actions", "predicted_lin_vel"],
                [[1, 8], [1, 3]],
                _valid_metadata(),
            )

        metadata = _valid_metadata()
        metadata["student_history_order"] = "newest_to_oldest"
        with self.assertRaisesRegex(ValueError, "student_history_order"):
            validate_lin_policy_interface(
                ["proprio_history", "actor_command"],
                [[1, 5, 28], [1, 3]],
                ["actions", "predicted_lin_vel"],
                [[1, 8], [1, 3]],
                metadata,
            )

    def test_controller_lin_compute_actions_uses_two_inputs_and_saves_prediction(self):
        controller_module = _import_wheelfoot_module()
        controller = controller_module.WheelfootController.__new__(
            controller_module.WheelfootController
        )
        controller.is_mjlab_repts_lin = True
        controller.is_mjlab_repts_depth = False
        controller.policy_input_names = ["proprio_history", "actor_command"]
        controller.policy_output_names = ["actions", "predicted_lin_vel"]
        controller.proprio_history_vector = np.arange(5 * 28, dtype=np.float32).reshape(5, 28)
        controller.commands = np.array([0.2, -0.3, 0.4], dtype=np.float32)
        controller.policy_session = RecordingSession()

        controller.compute_actions()

        output_names, inputs = controller.policy_session.calls[0]
        self.assertEqual(output_names, ["actions", "predicted_lin_vel"])
        self.assertEqual(set(inputs), {"proprio_history", "actor_command"})
        self.assertEqual(inputs["proprio_history"].shape, (1, 5, 28))
        self.assertEqual(inputs["actor_command"].shape, (1, 3))
        np.testing.assert_allclose(controller.actions, np.arange(8, dtype=np.float32))
        np.testing.assert_allclose(
            controller.predicted_lin_vel, np.array([0.4, 0.5, 0.6], dtype=np.float32)
        )

    def test_controller_can_disable_command_acceleration_limit(self):
        controller_module = _import_wheelfoot_module()
        controller = controller_module.WheelfootController.__new__(
            controller_module.WheelfootController
        )
        controller.mode = "WALK"
        controller.command_acceleration_limit_enabled = False
        controller.commands = np.zeros(3, dtype=np.float32)
        controller.target_commands = np.array([1.0, -0.5, 0.8], dtype=np.float32)
        controller.command_rate_limits = np.ones(3, dtype=np.float32)
        controller.loop_frequency = 500
        controller.loop_count = 0
        controller.handle_walk_mode = mock.Mock()
        controller.robot_cmd = object()
        controller.robot = SimpleNamespace(publishRobotCmd=mock.Mock())

        controller.update()

        np.testing.assert_allclose(controller.commands, controller.target_commands)
        controller.handle_walk_mode.assert_called_once_with()

    def test_controller_lin_branch_loads_config_policy_and_no_encoder(self):
        controller_module = _import_wheelfoot_module()

        created_paths = []

        def fake_inference_session(path, *args, **kwargs):
            del args, kwargs
            created_paths.append(str(path))
            self.assertNotIn("encoder.onnx", str(path))
            return FakePolicySession(path)

        with mock.patch.object(controller_module.ort, "InferenceSession", fake_inference_session):
            controller = controller_module.WheelfootController(
                str(MODEL_DIR),
                FakeRobot(),
                "WF_TRON1B",
                "mjlab_repts_lin",
                start_controller=False,
            )

        self.assertTrue(controller.config_file.endswith("params_mjlab_repts_lin.yaml"))
        self.assertTrue(
            controller.model_policy.endswith("policy/mjlab_repts_lin/policy.onnx")
        )
        self.assertIsNone(controller.model_encoder)
        self.assertIsNone(controller.encoder_session)
        self.assertTrue(controller.command_acceleration_limit_enabled)
        self.assertEqual(controller.policy_input_names, ["proprio_history", "actor_command"])
        self.assertEqual(controller.proprio_history_vector.shape, LIN_PROPRIO_HISTORY_SHAPE)
        self.assertEqual(len(created_paths), 1)


if __name__ == "__main__":
    unittest.main()
