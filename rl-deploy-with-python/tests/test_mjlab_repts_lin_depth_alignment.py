from __future__ import annotations

import contextlib
import importlib
import io
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

try:
    import onnxruntime as ort
except ModuleNotFoundError:
    ort = None


DEPLOY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEPLOY_ROOT))

from mjlab_repts_lin_depth import (  # noqa: E402
    D435_RAW_DEPTH_HEIGHT,
    D435_RAW_DEPTH_WIDTH,
    DEPTH_INPUT_SHAPE,
    DEPTH_LEFT_CROP_PX,
    DEPTH_MAX_DISTANCE_M,
    DEPTH_MIN_DISTANCE_M,
    DEPTH_RESIZE_HEIGHT,
    DEPTH_RESIZE_WIDTH,
    GRU_HIDDEN_STATE_SHAPE,
    POLICY_INPUT_NAMES,
    POLICY_OUTPUT_NAMES,
    PROPRIO_HISTORY_SHAPE,
    PROPRIO_TERM_DIMS,
    PROPRIO_TERM_ORDER,
    ProprioHistory,
    build_proprio_obs,
    build_proprio_terms,
    preprocess_depth_image,
    ros_image_to_depth_meters,
    validate_depth_policy_interface,
    _resolve_ros_type,
    _ros_image_to_depth_input,
)
from mjlab_repts import SDK_JOINT_NAMES  # noqa: E402


MODEL_DIR = DEPLOY_ROOT / "controllers" / "model"
POLICY_PATH = (
    MODEL_DIR
    / "WF_TRON1B"
    / "policy"
    / "mjlab_repts_gru_lin_depth"
    / "policy.onnx"
)
_WHEELFOOT_MODULE = None


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


class FakeRosStamp:
    def __init__(self, value=0.0):
        self.value = value

    def to_sec(self):
        return self.value


class FakeRosHeader:
    def __init__(self):
        self.stamp = FakeRosStamp()


class FakeRosImage:
    def __init__(self):
        self.header = FakeRosHeader()


class FakeRotation:
    @staticmethod
    def from_quat(_quat):
        return FakeRotation()

    @staticmethod
    def from_euler(_seq, _angles):
        return FakeRotation()

    def as_euler(self, _seq):
        return np.zeros(3, dtype=np.float32)

    def inv(self):
        return self

    def as_matrix(self):
        return np.eye(3, dtype=np.float32)


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
    scipy = types.ModuleType("scipy")
    spatial = types.ModuleType("scipy.spatial")
    transform = types.ModuleType("scipy.spatial.transform")
    transform.Rotation = FakeRotation

    return {
        "limxsdk": limxsdk,
        "limxsdk.robot": robot,
        "limxsdk.robot.Rate": rate,
        "limxsdk.robot.Robot": robot_module,
        "limxsdk.robot.RobotType": robot_type,
        "limxsdk.datatypes": datatypes,
        "scipy": scipy,
        "scipy.spatial": spatial,
        "scipy.spatial.transform": transform,
    }


def _import_wheelfoot_module():
    global _WHEELFOOT_MODULE
    if _WHEELFOOT_MODULE is not None:
        return _WHEELFOOT_MODULE
    module_path = DEPLOY_ROOT / "controllers" / "WheelfootController.py"
    spec = importlib.util.spec_from_file_location(
        "wheelfoot_controller_lin_depth_test",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, _fake_limxsdk_modules()):
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    _WHEELFOOT_MODULE = module
    return _WHEELFOOT_MODULE


def _fake_ros1_modules(subscriber_callbacks, state=None):
    state = state if state is not None else {"published": []}

    class FakeRospy:
        class core:
            @staticmethod
            def is_initialized():
                return True

        class Time:
            @staticmethod
            def now():
                return None

        @staticmethod
        def Subscriber(_topic, _image_type, callback, queue_size=1):
            del _image_type, queue_size
            subscriber_callbacks.append(callback)

            class FakeSubscriber:
                def unregister(self):
                    pass

            return FakeSubscriber()

        @staticmethod
        def Publisher(topic, image_type, queue_size=1):
            state["publisher"] = (image_type, topic, queue_size)

            class FakePublisher:
                def publish(self, msg):
                    state["published"].append(msg)

            return FakePublisher()

    sensor_msgs = types.ModuleType("sensor_msgs")
    sensor_msgs_msg = types.ModuleType("sensor_msgs.msg")
    sensor_msgs_msg.Image = FakeRosImage
    rospy_module = types.ModuleType("rospy")
    rospy_module.core = FakeRospy.core
    rospy_module.Time = FakeRospy.Time
    rospy_module.Subscriber = FakeRospy.Subscriber
    rospy_module.Publisher = FakeRospy.Publisher
    rospy_module.init_node = lambda *args, **kwargs: None

    return {
        "rospy": rospy_module,
        "sensor_msgs": sensor_msgs,
        "sensor_msgs.msg": sensor_msgs_msg,
    }


def _import_simulator_module(fake_modules):
    module_path = DEPLOY_ROOT.parent / "pointfoot-mujoco-sim" / "simulator.py"
    spec = importlib.util.spec_from_file_location(
        "simulator_ros_depth_test",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    mujoco = types.ModuleType("mujoco")
    viewer = types.ModuleType("mujoco.viewer")
    mujoco.viewer = viewer

    limxsdk = types.ModuleType("limxsdk")
    robot = types.ModuleType("limxsdk.robot")
    rate = types.ModuleType("limxsdk.robot.Rate")
    robot_module = types.ModuleType("limxsdk.robot.Robot")
    robot_type = types.ModuleType("limxsdk.robot.RobotType")
    datatypes = types.ModuleType("limxsdk.datatypes")
    robot.Rate = rate
    robot.Robot = robot_module
    robot.RobotType = robot_type
    limxsdk.robot = robot
    limxsdk.datatypes = datatypes
    datatypes.RobotCmd = FakeRobotCmd
    datatypes.RobotState = FakeRobotState
    datatypes.ImuData = FakeImuData

    import_modules = {
        "mujoco": mujoco,
        "mujoco.viewer": viewer,
        "limxsdk": limxsdk,
        "limxsdk.robot": robot,
        "limxsdk.robot.Rate": rate,
        "limxsdk.robot.Robot": robot_module,
        "limxsdk.robot.RobotType": robot_type,
        "limxsdk.datatypes": datatypes,
    }
    import_modules.update(fake_modules)

    with mock.patch.dict(sys.modules, import_modules):
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return module


class FakeIo:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class FakeMeta:
    custom_metadata_map = {
        "joint_names": ",".join(SDK_JOINT_NAMES),
        "student_observation_names": ",".join(PROPRIO_TERM_ORDER),
        "command_observation_names": "command",
        "policy_input_names": ",".join(POLICY_INPUT_NAMES),
        "policy_output_names": ",".join(POLICY_OUTPUT_NAMES),
        "student_history_length": "5",
        "student_history_flatten_dim": "false",
        "student_history_order": "oldest_to_newest",
        "action_target_names": (
            "abad_L_Joint,hip_L_Joint,knee_L_Joint,"
            "abad_R_Joint,hip_R_Joint,knee_R_Joint,"
            "wheel_L_Joint,wheel_R_Joint"
        ),
        "action_scale": "0.5,0.5,0.5,0.5,0.5,0.5,10.0,10.0",
        "depth_input_dtype": "float32",
        "depth_input_unit": "m",
        "depth_input_shape": "1.000,1.000,30.000,45.000",
        "depth_input_range": "0.15,2.5",
        "depth_invalid_value": "2.5",
        "depth_min_m": "0.15",
        "depth_max_m": "2.5",
        "depth_preprocessing": "external:below_min_to_max,clamp",
    }


class FakePolicySession:
    def __init__(self, *args, **kwargs):
        del args, kwargs

    def get_inputs(self):
        return [
            FakeIo("proprio_history", [1, 5, 28]),
            FakeIo("actor_command", [1, 3]),
            FakeIo("depth", [1, 1, 30, 45]),
            FakeIo("hidden_state_in", list(GRU_HIDDEN_STATE_SHAPE)),
        ]

    def get_outputs(self):
        return [
            FakeIo("actions", [1, 8]),
            FakeIo("predicted_lin_vel", [1, 3]),
            FakeIo("hidden_state_out", list(GRU_HIDDEN_STATE_SHAPE)),
        ]

    def get_modelmeta(self):
        return FakeMeta()

    def run(self, output_names, inputs):
        assert output_names == POLICY_OUTPUT_NAMES
        assert inputs["proprio_history"].shape == (1, *PROPRIO_HISTORY_SHAPE)
        assert inputs["actor_command"].shape == (1, 3)
        assert inputs["depth"].shape == DEPTH_INPUT_SHAPE
        assert inputs["hidden_state_in"].shape == GRU_HIDDEN_STATE_SHAPE
        return [
            np.full((1, 8), 0.25, dtype=np.float32),
            np.array([[0.1, -0.2, 0.3]], dtype=np.float32),
            inputs["hidden_state_in"] + np.float32(1.0),
        ]

class MjlabRepTsLinDepthAlignmentTest(unittest.TestCase):
    def test_depth_policy_matches_expected_interface(self):
        if ort is None:
            self.skipTest("onnxruntime is required to inspect policy.onnx")

        session = ort.InferenceSession(
            str(POLICY_PATH),
            providers=["CPUExecutionProvider"],
        )
        validate_depth_policy_interface(
            [input_info.name for input_info in session.get_inputs()],
            [input_info.shape for input_info in session.get_inputs()],
            [output_info.name for output_info in session.get_outputs()],
            [output_info.shape for output_info in session.get_outputs()],
            session.get_modelmeta().custom_metadata_map,
            hidden_state_shape=GRU_HIDDEN_STATE_SHAPE,
            policy_name="mjlab_repts_gru_lin_depth",
        )

        inputs = {
            input_info.name: np.zeros(input_info.shape, dtype=np.float32)
            for input_info in session.get_inputs()
        }
        inputs["depth"].fill(DEPTH_MAX_DISTANCE_M)
        outputs = session.run(None, inputs)
        self.assertEqual(
            [output.shape for output in outputs],
            [(1, 8), (1, 3), GRU_HIDDEN_STATE_SHAPE],
        )
        self.assertTrue(all(np.isfinite(output).all() for output in outputs))

    def test_controller_uses_external_depth_preprocessing_profile(self):
        wheelfoot_module = _import_wheelfoot_module()
        fake_depth_source = mock.Mock()

        with mock.patch.object(
            wheelfoot_module.ort,
            "InferenceSession",
            FakePolicySession,
        ):
            with mock.patch.object(
                wheelfoot_module,
                "create_depth_frame_source",
                return_value=fake_depth_source,
            ) as create_source:
                with mock.patch.dict(os.environ, {}, clear=True):
                    controller = wheelfoot_module.WheelfootController(
                        str(MODEL_DIR),
                        FakeRobot(),
                        "WF_TRON1B",
                        "mjlab_repts_gru_lin_depth",
                        start_controller=False,
                    )

        self.assertIs(controller.depth_source, fake_depth_source)
        create_source.assert_called_once_with(
            {
                "source": "ros",
                "ros_topic": "/camera0/depth/image_rect_raw",
                "ros_type": "ros1",
                "encoding": None,
                "depth_scale": None,
                "timeout_s": 0.5,
                "max_age_s": 0.5,
                "npy_path": None,
            }
        )

        controller.policy_metadata = {}
        fake_depth_source.frame_stats = None
        fake_depth_source.frame_age_s = None
        fake_depth_source.source_timestamp_s = None
        with contextlib.redirect_stdout(io.StringIO()) as output:
            controller.print_mjlab_repts_diagnostics()
        self.assertIn(
            "ONNX metadata: absent; using locked local deployment contract",
            output.getvalue(),
        )

    def test_depth_source_env_override_remains_available(self):
        lin_depth = importlib.import_module("mjlab_repts_lin_depth")

        with mock.patch.dict(
            os.environ,
            {"MJLAB_DEPTH_SOURCE": "zero"},
            clear=True,
        ):
            source = lin_depth.create_depth_frame_source(
                {
                    "source": "ros",
                    "ros_topic": "/camera0/depth/image_rect_raw",
                    "ros_type": "ros1",
                }
            )

        self.assertIsInstance(source, lin_depth.ZeroDepthFrameSource)
        np.testing.assert_allclose(source.frame(), DEPTH_MAX_DISTANCE_M)

    def test_resolve_ros_type_accepts_only_ros1(self):
        with mock.patch.dict(os.environ, {"ROS_TYPE": "ros1", "ROS_VERSION": "1"}, clear=True):
            self.assertEqual(_resolve_ros_type("ros1"), "ros1")
            self.assertEqual(_resolve_ros_type(None), "ros1")

        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_resolve_ros_type(None), "ros1")

        with mock.patch.dict(os.environ, {"ROS_TYPE": "ros2"}, clear=True):
            with self.assertRaisesRegex(ValueError, "ROS1-only"):
                _resolve_ros_type(None)

        with mock.patch.dict(os.environ, {"ROS_VERSION": "2"}, clear=True):
            with self.assertRaisesRegex(ValueError, "ROS1-only"):
                _resolve_ros_type(None)

    def test_proprio_obs_slices_match_lin_depth_policy_order(self):
        base_ang_vel = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        projected_gravity = np.array([0.1, -0.2, -0.97], dtype=np.float32)
        joint_pos = np.arange(8, dtype=np.float32)
        joint_vel = np.arange(10, 18, dtype=np.float32)
        default_joint_pos = np.full(8, 0.5, dtype=np.float32)
        default_joint_vel = np.arange(1, 9, dtype=np.float32)
        last_action = np.linspace(-1.0, 1.0, 8, dtype=np.float32)

        terms = build_proprio_terms(
            base_ang_vel,
            projected_gravity,
            joint_pos,
            joint_vel,
            default_joint_pos,
            default_joint_vel,
            last_action,
        )
        proprio_obs = build_proprio_obs(terms)
        leg_joint_indexes = [0, 1, 2, 4, 5, 6]

        self.assertEqual(proprio_obs.shape, (28,))
        np.testing.assert_allclose(proprio_obs[0:3], base_ang_vel)
        np.testing.assert_allclose(proprio_obs[3:6], projected_gravity)
        np.testing.assert_allclose(
            proprio_obs[6:12],
            joint_pos[leg_joint_indexes] - default_joint_pos[leg_joint_indexes],
        )
        np.testing.assert_allclose(
            proprio_obs[12:18],
            (joint_vel[leg_joint_indexes] - default_joint_vel[leg_joint_indexes]) * 0.05,
        )
        np.testing.assert_allclose(proprio_obs[18:20], joint_vel[[3, 7]] * 0.05)
        np.testing.assert_allclose(proprio_obs[20:28], last_action)

    def test_proprio_history_is_oldest_to_newest(self):
        history = ProprioHistory()
        for step in range(5):
            terms = {
                name: np.full(dim, step * 10 + index, dtype=np.float32)
                for index, (name, dim) in enumerate(
                    (name, PROPRIO_TERM_DIMS[name]) for name in PROPRIO_TERM_ORDER
                )
            }
            history.update(terms)

        matrix = history.matrix()
        self.assertEqual(matrix.shape, PROPRIO_HISTORY_SHAPE)
        np.testing.assert_allclose(matrix[0, :3], np.full(3, 0, dtype=np.float32))
        np.testing.assert_allclose(matrix[-1, :3], np.full(3, 40, dtype=np.float32))

    def test_depth_preprocess_converts_16uc1_mm_to_meters(self):
        image = np.full(
            (DEPTH_RESIZE_HEIGHT, DEPTH_RESIZE_WIDTH),
            1000,
            dtype=np.uint16,
        )
        image[0, DEPTH_LEFT_CROP_PX:DEPTH_LEFT_CROP_PX + 4] = [
            0,
            1000,
            2000,
            30000,
        ]

        depth = preprocess_depth_image(image, encoding="16UC1")

        self.assertEqual(depth.shape, DEPTH_INPUT_SHAPE)
        np.testing.assert_allclose(
            depth[0, 0, 0, :4],
            np.array([2.5, 1.0, 2.0, 2.5], dtype=np.float32),
        )

    def test_depth_preprocess_applies_fixed_external_range_contract(self):
        samples = np.array(
            [np.nan, np.inf, -np.inf, -1.0, 0.0, 0.149, 0.15, 1.1, 2.5, 2.501],
            dtype=np.float32,
        )
        image = np.ones(
            (DEPTH_RESIZE_HEIGHT, DEPTH_RESIZE_WIDTH), dtype=np.float32
        )
        image[0, DEPTH_LEFT_CROP_PX:DEPTH_LEFT_CROP_PX + samples.size] = samples

        depth = preprocess_depth_image(image)

        self.assertEqual(depth.dtype, np.float32)
        self.assertTrue(depth.flags.c_contiguous)
        np.testing.assert_allclose(
            depth[0, 0, 0, :samples.size],
            np.array([2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 0.15, 1.1, 2.5, 2.5]),
            atol=1.0e-7,
        )
        self.assertTrue(np.all(np.isfinite(depth)))
        self.assertGreaterEqual(float(depth.min()), DEPTH_MIN_DISTANCE_M)
        self.assertLessEqual(float(depth.max()), DEPTH_MAX_DISTANCE_M)

    def test_depth_preprocess_resizes_to_training_width_before_cropping(self):
        raw = np.tile(
            np.linspace(0.15, 2.5, D435_RAW_DEPTH_WIDTH, dtype=np.float32),
            (D435_RAW_DEPTH_HEIGHT, 1),
        )

        depth = preprocess_depth_image(raw)

        row_idx = (
            np.linspace(0, raw.shape[0] - 1, DEPTH_RESIZE_HEIGHT)
            .round()
            .astype(np.int64)
        )
        col_idx = (
            np.linspace(0, raw.shape[1] - 1, DEPTH_RESIZE_WIDTH)
            .round()
            .astype(np.int64)
        )
        resized = raw[row_idx[:, None], col_idx[None, :]]
        expected = resized[:, DEPTH_LEFT_CROP_PX:DEPTH_RESIZE_WIDTH]
        legacy_crop = int(round(raw.shape[1] * (8 / 53)))
        legacy = raw[:, legacy_crop:]
        legacy_col_idx = (
            np.linspace(0, legacy.shape[1] - 1, 45).round().astype(np.int64)
        )
        legacy = legacy[row_idx[:, None], legacy_col_idx[None, :]]
        self.assertEqual(depth.shape, DEPTH_INPUT_SHAPE)
        np.testing.assert_allclose(depth[0, 0], expected)
        self.assertFalse(np.allclose(depth[0, 0], legacy))

    def test_depth_preprocess_crops_training_shape_to_policy_shape(self):
        raw = np.tile(np.linspace(0.15, 2.5, 53, dtype=np.float32), (30, 1))

        depth = preprocess_depth_image(raw)

        self.assertEqual(depth.shape, DEPTH_INPUT_SHAPE)
        expected = raw[:, 8:]
        np.testing.assert_allclose(depth[0, 0], expected)

    def test_depth_preprocess_rejects_invalid_shape_and_depth_scale(self):
        with self.assertRaisesRegex(ValueError, "depth image must have shape"):
            preprocess_depth_image(np.ones((1, 2, 3, 4), dtype=np.float32))
        for depth_scale in (0.0, -1.0, float("nan"), float("inf"), "bad"):
            with self.subTest(depth_scale=depth_scale):
                with self.assertRaisesRegex(ValueError, "depth_scale"):
                    preprocess_depth_image(
                        np.ones((30, 53), dtype=np.float32),
                        depth_scale=depth_scale,
                    )

    def test_depth_preprocess_rejects_already_cropped_policy_frame(self):
        with self.assertRaisesRegex(ValueError, "already has the policy target shape"):
            preprocess_depth_image(np.ones((30, 45), dtype=np.float32))

    def test_ros_depth_image_respects_step_padding(self):
        msg = FakeRosImage()
        msg.height = 2
        msg.width = 2
        msg.encoding = "16UC1"
        msg.is_bigendian = 0
        msg.step = 6
        msg.data = np.array(
            [1000, 2000, 9999, 3000, 4000, 9999],
            dtype=np.uint16,
        ).tobytes()

        depth = _ros_image_to_depth_input(msg)

        self.assertEqual(depth.shape, (1, 1, 30, 45))
        np.testing.assert_allclose(depth[0, 0, 0, 0], 1.0)
        np.testing.assert_allclose(depth[0, 0, 0, -1], 2.0)
        np.testing.assert_allclose(depth[0, 0, -1, 0], 2.5)
        np.testing.assert_allclose(depth[0, 0, -1, -1], 2.5)

    def test_ros_image_to_depth_meters_keeps_raw_shape(self):
        msg = FakeRosImage()
        msg.height = 2
        msg.width = 2
        msg.encoding = "16UC1"
        msg.is_bigendian = 0
        msg.step = 6
        msg.data = np.array(
            [1000, 2000, 9999, 3000, 4000, 9999],
            dtype=np.uint16,
        ).tobytes()

        depth_m = ros_image_to_depth_meters(msg)

        self.assertEqual(depth_m.shape, (2, 2))
        np.testing.assert_allclose(
            depth_m,
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        )

    def test_ros_image_to_depth_meters_keeps_32fc1_meters(self):
        msg = FakeRosImage()
        msg.height = 1
        msg.width = 2
        msg.encoding = "32FC1"
        msg.is_bigendian = 0
        msg.step = 8
        msg.data = np.array([1.25, 2.5], dtype=np.float32).tobytes()

        depth_m = ros_image_to_depth_meters(msg)

        self.assertEqual(depth_m.shape, (1, 2))
        np.testing.assert_allclose(depth_m, np.array([[1.25, 2.5]], dtype=np.float32))

    def test_ros_depth_input_applies_external_range_contract(self):
        msg = FakeRosImage()
        raw = np.ones((30, 53), dtype=np.float32)
        raw[0, 8:16] = [np.nan, np.inf, -np.inf, 0.1, 0.15, 1.1, 2.5, 2.6]
        msg.height, msg.width = raw.shape
        msg.encoding = "32FC1"
        msg.is_bigendian = 0
        msg.step = msg.width * np.dtype(np.float32).itemsize
        msg.data = raw.tobytes()

        depth = _ros_image_to_depth_input(msg)

        self.assertEqual(depth.shape, DEPTH_INPUT_SHAPE)
        self.assertTrue(np.all(np.isfinite(depth)))
        np.testing.assert_allclose(
            depth[0, 0, 0, :8],
            np.array(
                [2.5, 2.5, 2.5, 2.5, 0.15, 1.1, 2.5, 2.5],
                dtype=np.float32,
            ),
            atol=1.0e-7,
        )

    def test_ros_depth_source_rejects_stale_frame(self):
        wheelfoot_module = _import_wheelfoot_module()
        lin_depth = importlib.import_module("mjlab_repts_lin_depth")
        subscriber_callbacks = []
        del wheelfoot_module

        with mock.patch.dict(
            sys.modules,
            _fake_ros1_modules(subscriber_callbacks),
        ):
            source = lin_depth.create_depth_frame_source(
                {
                    "source": "ros",
                    "ros_topic": "/camera/depth/image_rect_raw",
                    "ros_type": "ros1",
                    "timeout_s": 0.01,
                    "max_age_s": 0.01,
                }
            )

        msg = FakeRosImage()
        msg.header = FakeRosHeader()
        msg.height = 1
        msg.width = 1
        msg.encoding = "16UC1"
        msg.is_bigendian = 0
        msg.step = 2
        msg.data = np.array([1000], dtype=np.uint16).tobytes()
        subscriber_callbacks[0](msg)
        source._backend._latest_recv_monotonic_s -= 1.0

        with self.assertRaisesRegex(TimeoutError, "stale ROS depth image"):
            source.frame()

    def test_ros1_depth_source_subscribes_and_updates_frame(self):
        lin_depth = importlib.import_module("mjlab_repts_lin_depth")
        subscriber_callbacks = []

        with mock.patch.dict(sys.modules, _fake_ros1_modules(subscriber_callbacks)):
            source = lin_depth.create_depth_frame_source(
                {
                    "source": "ros",
                    "ros_topic": "/camera/depth/image_rect_raw",
                    "ros_type": "ros1",
                    "timeout_s": 0.01,
                }
            )

        msg = FakeRosImage()
        msg.header = FakeRosHeader()
        msg.header.stamp = FakeRosStamp(123.25)
        msg.height = 1
        msg.width = 1
        msg.encoding = "16UC1"
        msg.is_bigendian = 0
        msg.step = 2
        msg.data = np.array([1500], dtype=np.uint16).tobytes()
        subscriber_callbacks[0](msg)

        first_frame = source.frame()
        second_frame = source.frame()
        self.assertIs(first_frame, second_frame)
        np.testing.assert_allclose(first_frame[0, 0, 0, 0], 1.5)
        self.assertGreaterEqual(source.frame_age_s, 0.0)
        self.assertEqual(source.source_timestamp_s, 123.25)
        stats = source.frame_stats
        self.assertIsNotNone(stats)
        np.testing.assert_allclose(
            [stats["min"], stats["max"], stats["mean"]],
            1.5,
            atol=5.0e-7,
        )

    def test_npy_live_source_reuses_latest_frame_and_rejects_stale_data(self):
        lin_depth = importlib.import_module("mjlab_repts_lin_depth")
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "depth.npy"
            np.save(path, np.full((30, 53), 1.25, dtype=np.float32))
            source = lin_depth.create_depth_frame_source(
                {
                    "source": "npy_live",
                    "npy_path": str(path),
                    "timeout_s": 0.01,
                    "max_age_s": 0.01,
                }
            )

            first_frame = source.frame()
            second_frame = source.frame()
            self.assertIs(first_frame, second_frame)
            np.testing.assert_allclose(first_frame, 1.25)
            self.assertGreaterEqual(source.frame_age_s, 0.0)
            self.assertIsNotNone(source.source_timestamp_s)
            self.assertEqual(
                source.frame_stats,
                {"min": 1.25, "max": 1.25, "mean": 1.25},
            )

            source._latest_recv_monotonic_s -= 1.0
            with self.assertRaisesRegex(TimeoutError, "stale depth frame file"):
                source.frame()

    def test_static_npy_source_uses_fixed_depth_preprocessing(self):
        lin_depth = importlib.import_module("mjlab_repts_lin_depth")
        raw = np.full((30, 53), 1.0, dtype=np.float32)
        raw[:, 8] = 0.0
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "depth.npy"
            np.save(path, raw)
            source = lin_depth.create_depth_frame_source(
                {"source": "npy", "npy_path": str(path)}
            )

        np.testing.assert_allclose(source.frame()[0, 0, :, 0], 2.5)
        np.testing.assert_allclose(source.frame()[0, 0, :, 1:], 1.0)
        self.assertIsNotNone(source.source_timestamp_s)

    def test_ros1_depth_publisher_uses_image_wire_format(self):
        state = {"published": []}
        fake_modules = _fake_ros1_modules([], state)

        with mock.patch.dict(os.environ, {"ROS_TYPE": "ros1"}, clear=False):
            with mock.patch.dict(sys.modules, fake_modules):
                simulator = _import_simulator_module(fake_modules)
                publisher = simulator.RosDepthFramePublisher("/camera/depth/image_rect_raw")
                publisher.publish(np.array([[1.0, 2.0]], dtype=np.float32))

        msg = state["published"][0]
        self.assertEqual(state["publisher"][0], FakeRosImage)
        self.assertEqual(state["publisher"][1], "/camera/depth/image_rect_raw")
        self.assertEqual(msg.header.seq, 0)
        self.assertEqual(msg.encoding, "16UC1")
        self.assertEqual(msg.is_bigendian, 0)
        self.assertEqual(msg.height, 1)
        self.assertEqual(msg.width, 2)
        self.assertEqual(msg.step, 4)
        self.assertEqual(msg.data, np.array([1000, 2000], dtype="<u2").tobytes())

    def test_validate_depth_policy_interface_accepts_expected_metadata(self):
        validate_depth_policy_interface(
            POLICY_INPUT_NAMES,
            [[1, 5, 28], [1, 3], [1, 1, 30, 45], [1, 128]],
            POLICY_OUTPUT_NAMES,
            [[1, 8], [1, 3], [1, 128]],
            FakeMeta.custom_metadata_map,
        )

    def test_validate_depth_policy_interface_rejects_two_input_export_metadata(self):
        metadata = dict(FakeMeta.custom_metadata_map)
        metadata["policy_input_names"] = "proprio_history,actor_command"

        with self.assertRaisesRegex(ValueError, "metadata policy_input_names"):
            validate_depth_policy_interface(
                POLICY_INPUT_NAMES,
                [[1, 5, 28], [1, 3], [1, 1, 30, 45], [1, 128]],
                POLICY_OUTPUT_NAMES,
                [[1, 8], [1, 3], [1, 128]],
                metadata,
            )

    def test_validate_depth_policy_interface_rejects_bad_depth_metadata(self):
        for key, value in (
            ("depth_input_unit", "mm"),
            ("depth_input_range", "0.2,2.0"),
            ("depth_invalid_value", "0.0"),
            ("depth_preprocessing", "onnx:normalize"),
            ("joint_names", "wheel_L_Joint,wheel_R_Joint"),
            ("policy_output_names", "actions,predicted_lin_vel"),
        ):
            with self.subTest(key=key):
                metadata = dict(FakeMeta.custom_metadata_map)
                metadata[key] = value
                with self.assertRaisesRegex(ValueError, key):
                    validate_depth_policy_interface(
                        POLICY_INPUT_NAMES,
                        [[1, 5, 28], [1, 3], [1, 1, 30, 45], [1, 128]],
                        POLICY_OUTPUT_NAMES,
                        [[1, 8], [1, 3], [1, 128]],
                        metadata,
                    )

    def test_validate_depth_policy_interface_rejects_non_depth_interface(self):
        with self.assertRaisesRegex(ValueError, "inputs"):
            validate_depth_policy_interface(
                ["student_history"],
                [[1, 5, 31]],
                ["actions"],
                [[1, 8]],
            )

    def test_validate_depth_policy_interface_rejects_legacy_depth_shape(self):
        with self.assertRaisesRegex(ValueError, "input shapes"):
            validate_depth_policy_interface(
                POLICY_INPUT_NAMES,
                [[1, 5, 28], [1, 3], [1, 1, 28, 48], [1, 128]],
                POLICY_OUTPUT_NAMES,
                [[1, 8], [1, 3], [1, 128]],
            )

    def test_validate_gru_depth_policy_rejects_mismatched_hidden_output(self):
        with self.assertRaisesRegex(ValueError, "output shapes"):
            validate_depth_policy_interface(
                POLICY_INPUT_NAMES,
                [[1, 5, 28], [1, 3], [1, 1, 30, 45], [1, 128]],
                POLICY_OUTPUT_NAMES,
                [[1, 8], [1, 3], [1, 64]],
                hidden_state_shape=GRU_HIDDEN_STATE_SHAPE,
                policy_name="mjlab_repts_gru_lin_depth",
            )

    def test_controller_depth_branch_loads_config_and_feedback_state(self):
        wheelfoot_module = _import_wheelfoot_module()

        with mock.patch.object(
            wheelfoot_module.ort,
            "InferenceSession",
            FakePolicySession,
        ):
            with mock.patch.dict(
                os.environ,
                {"MJLAB_DEPTH_SOURCE": "zero"},
                clear=False,
            ):
                controller = wheelfoot_module.WheelfootController(
                    str(MODEL_DIR),
                    FakeRobot(),
                    "WF_TRON1B",
                    "mjlab_repts_gru_lin_depth",
                    start_controller=False,
                )

        self.assertTrue(controller.is_mjlab_repts_gru_lin_depth)
        self.assertTrue(controller.is_mjlab_repts_depth)
        self.assertTrue(
            controller.config_file.endswith("params_mjlab_repts_gru_lin_depth.yaml")
        )
        self.assertIsNone(controller.model_encoder)
        self.assertIsNone(controller.encoder_session)
        self.assertEqual(controller.depth_hidden_state.shape, GRU_HIDDEN_STATE_SHAPE)
        self.assertEqual(controller.proprio_history_vector.shape, PROPRIO_HISTORY_SHAPE)

        controller.compute_actions()
        controller.compute_actions()
        np.testing.assert_allclose(
            controller.depth_hidden_state,
            np.full(GRU_HIDDEN_STATE_SHAPE, 2.0, dtype=np.float32),
        )

    def test_controller_policy_reset_clears_state_and_refills_history(self):
        wheelfoot_module = _import_wheelfoot_module()

        with mock.patch.object(
            wheelfoot_module.ort,
            "InferenceSession",
            FakePolicySession,
        ):
            with mock.patch.dict(
                os.environ,
                {"MJLAB_DEPTH_SOURCE": "zero"},
                clear=False,
            ):
                controller = wheelfoot_module.WheelfootController(
                    str(MODEL_DIR),
                    FakeRobot(),
                    "WF_TRON1B",
                    "mjlab_repts_gru_lin_depth",
                    start_controller=False,
                )

        controller.compute_observation()
        controller.depth_hidden_state.fill(3.0)
        controller.last_actions.fill(1.0)
        controller.actions.fill(2.0)
        controller.predicted_lin_vel.fill(4.0)
        controller.mjlab_repts_policy_initialized = True

        controller.reset_policy_state()

        self.assertIsNone(controller.mjlab_repts_history._frames)
        np.testing.assert_allclose(controller.proprio_history_vector, 0.0)
        np.testing.assert_allclose(controller.depth_hidden_state, 0.0)
        np.testing.assert_allclose(controller.last_actions, 0.0)
        np.testing.assert_allclose(controller.actions, 0.0)
        np.testing.assert_allclose(controller.predicted_lin_vel, 0.0)
        self.assertFalse(controller.mjlab_repts_policy_initialized)

        controller.compute_observation()
        for frame in controller.proprio_history_vector[1:]:
            np.testing.assert_allclose(frame, controller.proprio_history_vector[0])
        controller.compute_actions()
        np.testing.assert_allclose(controller.depth_hidden_state, 1.0)

    def test_stand_to_walk_transition_resets_policy_state(self):
        controller = object.__new__(_import_wheelfoot_module().WheelfootController)
        controller.stand_percent = 1.0
        controller.mode = "STAND"
        controller.init_state = {}
        controller.reset_policy_state = mock.Mock()

        controller.handle_stand_mode()

        controller.reset_policy_state.assert_called_once_with()
        self.assertEqual(controller.mode, "WALK")

    def test_controller_depth_walk_step_updates_hidden_state(self):
        wheelfoot_module = _import_wheelfoot_module()
        subscriber_callbacks = []

        with mock.patch.dict(sys.modules, _fake_ros1_modules(subscriber_callbacks)):
            with mock.patch.object(wheelfoot_module.ort, "InferenceSession", FakePolicySession):
                env = {"MJLAB_DEPTH_SOURCE": "ros", "ROS_TYPE": "ros1"}
                with mock.patch.dict(os.environ, env, clear=False):
                    controller = wheelfoot_module.WheelfootController(
                        str(MODEL_DIR),
                        FakeRobot(),
                        "WF_TRON1B",
                        "mjlab_repts_gru_lin_depth",
                        start_controller=False,
                    )

        raw_depth = np.full(
            (D435_RAW_DEPTH_HEIGHT, D435_RAW_DEPTH_WIDTH),
            1500,
            dtype=np.uint16,
        )
        msg = FakeRosImage()
        msg.header = FakeRosHeader()
        msg.height, msg.width = raw_depth.shape
        msg.encoding = "16UC1"
        msg.is_bigendian = 0
        msg.step = msg.width * raw_depth.dtype.itemsize
        msg.data = raw_depth.tobytes()
        subscriber_callbacks[0](msg)

        controller.loop_count = 3
        with contextlib.redirect_stdout(io.StringIO()) as output:
            controller.handle_mjlab_repts_walk_mode()

        self.assertTrue(controller.mjlab_repts_policy_initialized)
        self.assertTrue(controller.mjlab_repts_diagnostics_printed)
        self.assertIn("depth source: RosDepthFrameSource", output.getvalue())
        np.testing.assert_allclose(
            controller.depth_hidden_state,
            np.ones(GRU_HIDDEN_STATE_SHAPE, dtype=np.float32),
        )
        np.testing.assert_allclose(
            controller.predicted_lin_vel,
            np.array([0.1, -0.2, 0.3], dtype=np.float32),
        )
        self.assertEqual(controller.depth_source.frame().shape, DEPTH_INPUT_SHAPE)
        self.assertEqual(controller.observations.shape, (28,))


if __name__ == "__main__":
    unittest.main()
