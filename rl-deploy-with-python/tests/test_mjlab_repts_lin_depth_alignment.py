from __future__ import annotations

import contextlib
import importlib
import io
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


DEPLOY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEPLOY_ROOT))

from mjlab_repts_lin_depth import (  # noqa: E402
    DEPTH_INPUT_SHAPE,
    HIDDEN_STATE_SHAPE,
    POLICY_INPUT_NAMES,
    POLICY_OUTPUT_NAMES,
    PROPRIO_HISTORY_SHAPE,
    PROPRIO_TERM_DIMS,
    PROPRIO_TERM_ORDER,
    ProprioHistory,
    build_proprio_obs,
    build_proprio_terms,
    preprocess_depth_image,
    validate_depth_policy_interface,
    _ros_image_to_depth_input,
)


MODEL_DIR = DEPLOY_ROOT / "controllers" / "model"
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
    def to_sec(self):
        return 0.0


class FakeRosHeader:
    def __init__(self):
        self.stamp = FakeRosStamp()


class FakeRosImage:
    pass


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


class FakeIo:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class FakeMeta:
    custom_metadata_map = {
        "student_observation_names": ",".join(PROPRIO_TERM_ORDER),
        "command_observation_names": "command",
        "policy_input_names": ",".join(POLICY_INPUT_NAMES),
        "policy_output_names": "actions,predicted_lin_vel",
        "student_history_length": "5",
        "student_history_flatten_dim": "false",
        "student_history_order": "oldest_to_newest",
        "action_target_names": (
            "abad_L_Joint,hip_L_Joint,knee_L_Joint,"
            "abad_R_Joint,hip_R_Joint,knee_R_Joint,"
            "wheel_L_Joint,wheel_R_Joint"
        ),
        "action_scale": "0.5,0.5,0.5,0.5,0.5,0.5,10.0,10.0",
    }


class FakePolicySession:
    def __init__(self, *args, **kwargs):
        del args, kwargs

    def get_inputs(self):
        return [
            FakeIo("proprio_history", [1, 5, 28]),
            FakeIo("actor_command", [1, 3]),
            FakeIo("depth", [1, 1, 28, 48]),
            FakeIo("hidden_state_in", [1, 64]),
        ]

    def get_outputs(self):
        return [
            FakeIo("actions", [1, 8]),
            FakeIo("predicted_lin_vel", [1, 3]),
            FakeIo("hidden_state_out", [1, 64]),
        ]

    def get_modelmeta(self):
        return FakeMeta()

    def run(self, output_names, inputs):
        assert output_names == POLICY_OUTPUT_NAMES
        assert inputs["proprio_history"].shape == (1, *PROPRIO_HISTORY_SHAPE)
        assert inputs["actor_command"].shape == (1, 3)
        assert inputs["depth"].shape == DEPTH_INPUT_SHAPE
        assert inputs["hidden_state_in"].shape == HIDDEN_STATE_SHAPE
        return [
            np.full((1, 8), 0.25, dtype=np.float32),
            np.array([[0.1, -0.2, 0.3]], dtype=np.float32),
            inputs["hidden_state_in"] + np.float32(1.0),
        ]


class MjlabRepTsLinDepthAlignmentTest(unittest.TestCase):
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
        np.testing.assert_allclose(proprio_obs[18:20], joint_vel[[3, 7]] * 0.5)
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
        image = np.array([[0, 1000], [2000, 30000]], dtype=np.uint16)

        depth = preprocess_depth_image(
            image,
            encoding="16UC1",
            target_shape=(2, 2),
            max_depth=10.0,
        )

        self.assertEqual(depth.shape, (1, 1, 2, 2))
        np.testing.assert_allclose(
            depth[0, 0],
            np.array([[0.0, 1.0], [2.0, 10.0]], dtype=np.float32),
        )

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

        depth = _ros_image_to_depth_input(msg, min_depth=0.0, max_depth=10.0)

        self.assertEqual(depth.shape, (1, 1, 28, 48))
        np.testing.assert_allclose(depth[0, 0, 0, 0], 1.0)
        np.testing.assert_allclose(depth[0, 0, 0, -1], 2.0)
        np.testing.assert_allclose(depth[0, 0, -1, 0], 3.0)
        np.testing.assert_allclose(depth[0, 0, -1, -1], 4.0)

    def test_ros_depth_source_rejects_stale_frame(self):
        wheelfoot_module = _import_wheelfoot_module()
        lin_depth = importlib.import_module("mjlab_repts_lin_depth")
        subscriber_callbacks = []

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

        del wheelfoot_module
        sensor_msgs = types.ModuleType("sensor_msgs")
        sensor_msgs_msg = types.ModuleType("sensor_msgs.msg")
        sensor_msgs_msg.Image = FakeRosImage
        rospy_module = types.ModuleType("rospy")
        rospy_module.core = FakeRospy.core
        rospy_module.Time = FakeRospy.Time
        rospy_module.Subscriber = FakeRospy.Subscriber
        rospy_module.init_node = lambda *args, **kwargs: None

        with mock.patch.dict(
            sys.modules,
            {
                "rospy": rospy_module,
                "sensor_msgs": sensor_msgs,
                "sensor_msgs.msg": sensor_msgs_msg,
            },
        ):
            source = lin_depth.create_depth_frame_source(
                {
                    "source": "ros",
                    "ros_topic": "/camera/depth/image_rect_raw",
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
        source._latest_recv_time_s -= 1.0

        with self.assertRaisesRegex(TimeoutError, "stale ROS depth image"):
            source.frame()

    def test_validate_depth_policy_interface_accepts_expected_metadata(self):
        validate_depth_policy_interface(
            POLICY_INPUT_NAMES,
            [[1, 5, 28], [1, 3], [1, 1, 28, 48], [1, 64]],
            POLICY_OUTPUT_NAMES,
            [[1, 8], [1, 3], [1, 64]],
            FakeMeta.custom_metadata_map,
        )

    def test_validate_depth_policy_interface_accepts_legacy_two_input_metadata(self):
        metadata = dict(FakeMeta.custom_metadata_map)
        metadata["policy_input_names"] = "proprio_history,actor_command"

        validate_depth_policy_interface(
            POLICY_INPUT_NAMES,
            [[1, 5, 28], [1, 3], [1, 1, 28, 48], [1, 64]],
            POLICY_OUTPUT_NAMES,
            [[1, 8], [1, 3], [1, 64]],
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

    def test_controller_branch_loads_depth_policy_without_encoder(self):
        wheelfoot_module = _import_wheelfoot_module()

        with mock.patch.object(wheelfoot_module.ort, "InferenceSession", FakePolicySession):
            env = {"MJLAB_DEPTH_SOURCE": "zero"}
            with mock.patch.dict(os.environ, env, clear=False):
                controller = wheelfoot_module.WheelfootController(
                    str(MODEL_DIR),
                    FakeRobot(),
                    "WF_TRON1B",
                    "mjlab_repts_lin_depth",
                    start_controller=False,
                )

        self.assertTrue(controller.config_file.endswith("params_mjlab_repts_lin_depth.yaml"))
        self.assertIsNone(controller.model_encoder)
        self.assertIsNone(controller.encoder_session)
        self.assertEqual(controller.policy_input_names, POLICY_INPUT_NAMES)
        self.assertEqual(controller.proprio_history_vector.shape, PROPRIO_HISTORY_SHAPE)

    def test_controller_depth_walk_step_updates_hidden_state(self):
        wheelfoot_module = _import_wheelfoot_module()

        with mock.patch.object(wheelfoot_module.ort, "InferenceSession", FakePolicySession):
            env = {"MJLAB_DEPTH_SOURCE": "zero"}
            with mock.patch.dict(os.environ, env, clear=False):
                controller = wheelfoot_module.WheelfootController(
                    str(MODEL_DIR),
                    FakeRobot(),
                    "WF_TRON1B",
                    "mjlab_repts_lin_depth",
                    start_controller=False,
                )

        controller.loop_count = 3
        with contextlib.redirect_stdout(io.StringIO()) as output:
            controller.handle_mjlab_repts_walk_mode()

        self.assertTrue(controller.mjlab_repts_policy_initialized)
        self.assertTrue(controller.mjlab_repts_diagnostics_printed)
        self.assertIn("depth source: ZeroDepthFrameSource", output.getvalue())
        np.testing.assert_allclose(
            controller.depth_hidden_state,
            np.ones(HIDDEN_STATE_SHAPE, dtype=np.float32),
        )
        np.testing.assert_allclose(
            controller.predicted_lin_vel,
            np.array([0.1, -0.2, 0.3], dtype=np.float32),
        )
        self.assertEqual(controller.observations.shape, (28,))


if __name__ == "__main__":
    unittest.main()
