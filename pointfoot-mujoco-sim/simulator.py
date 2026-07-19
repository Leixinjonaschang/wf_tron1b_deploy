# Copyright information
#
# © [2024] LimX Dynamics Technology Co., Ltd. All rights reserved.

import os
import sys
import time
import threading
from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer as viewer
from functools import partial
import limxsdk
import limxsdk.robot.Rate as Rate
import limxsdk.robot.Robot as Robot
import limxsdk.robot.RobotType as RobotType
import limxsdk.datatypes as datatypes
from depth_visualization import colorize_depth, fit_overlay_rect, resize_rgb_nearest


POLICY_DEPTH_HEIGHT = 30
POLICY_DEPTH_WIDTH = 45


def _env_int(name, default, *, minimum=1):
    value = int(os.getenv(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _env_float(name, default):
    value = float(os.getenv(name, str(default)))
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _resolve_ros_type(explicit=None):
    value = explicit or os.getenv("ROS_TYPE")
    if value:
        normalized = value.strip().lower()
        if normalized in ("1", "ros1"):
            return "ros1"
        if normalized in ("2", "ros2"):
            raise ValueError("ROS depth publishing is ROS1-only; use ROS_TYPE=ros1")
        raise ValueError("ROS_TYPE must be 'ros1'")

    ros_version = os.getenv("ROS_VERSION")
    if ros_version == "1":
        return "ros1"
    if ros_version == "2":
        raise ValueError("ROS depth publishing is ROS1-only; use a ROS1 environment")

    return "ros1"


class _RosDepthMessageMixin:
    def _depth_to_msg(self, depth_m):
        import numpy as np

        depth_mm = np.clip(
            np.nan_to_num(depth_m, nan=0.0, posinf=65.535, neginf=0.0) * 1000.0,
            0,
            65535,
        ).astype("<u2")
        depth_mm = np.ascontiguousarray(depth_mm)

        msg = self._Image()
        msg.header.frame_id = self._frame_id
        msg.height, msg.width = depth_mm.shape
        msg.encoding = "16UC1"
        msg.is_bigendian = 0
        msg.step = msg.width * 2
        msg.data = depth_mm.tobytes()
        return msg


class _Ros1DepthFramePublisher(_RosDepthMessageMixin):
    def __init__(self, topic, frame_id="camera_depth_optical_frame"):
        try:
            import rospy
            from sensor_msgs.msg import Image
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "ROS1 depth publishing requires rospy and sensor_msgs. Source the ROS1 "
                "workspace before running with MJLAB_DEPTH_SINK=ros."
            ) from exc

        if not rospy.core.is_initialized():
            rospy.init_node("mujoco_depth_publisher", anonymous=True, disable_signals=True)

        self._rospy = rospy
        self._Image = Image
        self._pub = rospy.Publisher(topic, Image, queue_size=1)
        self._frame_id = frame_id
        self._seq = 0

    def publish(self, depth_m):
        msg = self._depth_to_msg(depth_m)
        msg.header.stamp = self._rospy.Time.now()
        msg.header.seq = self._seq
        self._pub.publish(msg)
        self._seq += 1


class RosDepthFramePublisher:
    def __init__(self, topic, frame_id="camera_depth_optical_frame"):
        _resolve_ros_type()
        self._backend = _Ros1DepthFramePublisher(topic, frame_id=frame_id)

    def publish(self, depth_m):
        self._backend.publish(depth_m)


def decode_policy_depth_image(msg):
    """Decode the controller's exact policy-input debug Image message."""
    if msg.encoding != "32FC1":
        raise ValueError(f"policy depth encoding must be 32FC1, got {msg.encoding}")
    if int(getattr(msg, "is_bigendian", 0)) != 0:
        raise ValueError("policy depth image must be little-endian")
    if (int(msg.height), int(msg.width)) != (POLICY_DEPTH_HEIGHT, POLICY_DEPTH_WIDTH):
        raise ValueError(
            "policy depth image must have shape "
            f"({POLICY_DEPTH_HEIGHT}, {POLICY_DEPTH_WIDTH}), got ({msg.height}, {msg.width})"
        )
    expected_step = POLICY_DEPTH_WIDTH * np.dtype("<f4").itemsize
    if int(msg.step) != expected_step:
        raise ValueError(f"policy depth image step must be {expected_step}, got {msg.step}")
    expected_size = POLICY_DEPTH_HEIGHT * expected_step
    if len(msg.data) != expected_size:
        raise ValueError(
            f"policy depth image payload must have {expected_size} bytes, got {len(msg.data)}"
        )
    depth = np.frombuffer(msg.data, dtype="<f4").reshape(
        POLICY_DEPTH_HEIGHT, POLICY_DEPTH_WIDTH
    )
    return np.ascontiguousarray(depth, dtype=np.float32).copy()


class _Ros1PolicyDepthSubscriber:
    def __init__(self, topic):
        try:
            import rospy
            from sensor_msgs.msg import Image
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Policy depth overlay requires rospy and sensor_msgs. Source the ROS1 "
                "workspace before setting MJLAB_DEPTH_OVERLAY=policy."
            ) from exc
        if not rospy.core.is_initialized():
            rospy.init_node("mujoco_policy_depth_overlay", anonymous=True, disable_signals=True)
        self._lock = threading.Lock()
        self._latest_depth = None
        self._latest_seq = None
        self._latest_recv_time_s = None
        self._last_error_time_s = 0.0
        self._subscriber = rospy.Subscriber(topic, Image, self._callback, queue_size=1)

    def _callback(self, msg):
        try:
            depth = decode_policy_depth_image(msg)
        except ValueError as exc:
            now = time.monotonic()
            if now - self._last_error_time_s >= 5.0:
                print(f"Warning: ignoring invalid policy depth image: {exc}")
                self._last_error_time_s = now
            return
        with self._lock:
            self._latest_depth = depth
            self._latest_seq = int(msg.header.seq)
            self._latest_recv_time_s = time.monotonic()

    def latest(self):
        with self._lock:
            depth = None if self._latest_depth is None else self._latest_depth.copy()
            return depth, self._latest_seq, self._latest_recv_time_s

    def close(self):
        self._subscriber.unregister()


class RosPolicyDepthSubscriber:
    def __init__(self, topic):
        _resolve_ros_type()
        self._backend = _Ros1PolicyDepthSubscriber(topic)

    def latest(self):
        return self._backend.latest()

    def close(self):
        self._backend.close()


class SimulatorMujoco:
    def __init__(self, asset_path, joint_sensor_names, robot): 
        self.robot = robot
        self.joint_sensor_names = joint_sensor_names
        self.joint_num = len(joint_sensor_names)
        
        # Load the MuJoCo model and data from the specified XML asset path
        self.mujoco_model = mujoco.MjModel.from_xml_path(asset_path)
        self.mujoco_data = mujoco.MjData(self.mujoco_model)

        self.depth_export_path = os.getenv("MJLAB_DEPTH_NPY_PATH")
        self.depth_sink = os.getenv("MJLAB_DEPTH_SINK", "npy").lower()
        if self.depth_sink not in ("npy", "ros", "both"):
            raise ValueError("MJLAB_DEPTH_SINK must be one of: npy, ros, both")
        self.depth_renderer = None
        self.depth_scene_option = None
        self.depth_ros_pub = None
        self.depth_ros_topic = os.getenv("MJLAB_DEPTH_ROS_TOPIC", "/camera/depth/image_rect_raw")
        self.depth_ros_frame_id = os.getenv("MJLAB_DEPTH_FRAME_ID", "camera_depth_optical_frame")
        self.depth_camera_name = os.getenv("MJLAB_DEPTH_CAMERA", "d435")
        self.depth_capture_frequency = float(os.getenv("MJLAB_DEPTH_CAPTURE_HZ", "30.0"))
        self.depth_overlay_mode = os.getenv("MJLAB_DEPTH_OVERLAY", "off").strip().lower()
        if self.depth_overlay_mode not in ("off", "policy", "raw"):
            raise ValueError("MJLAB_DEPTH_OVERLAY must be one of: off, policy, raw")
        self.policy_depth_topic = os.getenv(
            "MJLAB_POLICY_DEPTH_TOPIC", "/mjlab/policy/depth_input"
        )
        self.depth_overlay_width = _env_int("MJLAB_DEPTH_OVERLAY_WIDTH", 360)
        self.depth_overlay_height = _env_int("MJLAB_DEPTH_OVERLAY_HEIGHT", 240)
        self.depth_overlay_margin = _env_int("MJLAB_DEPTH_OVERLAY_MARGIN", 12, minimum=0)
        self.depth_overlay_min = _env_float("MJLAB_DEPTH_OVERLAY_MIN", 0.0)
        self.depth_overlay_max = _env_float("MJLAB_DEPTH_OVERLAY_MAX", 10.0)
        if self.depth_overlay_max <= self.depth_overlay_min:
            raise ValueError("MJLAB_DEPTH_OVERLAY_MAX must be greater than MJLAB_DEPTH_OVERLAY_MIN")
        self.depth_overlay_colormap = os.getenv("MJLAB_DEPTH_OVERLAY_COLORMAP", "turbo").lower()
        if self.depth_overlay_colormap not in ("turbo", "gray"):
            raise ValueError("MJLAB_DEPTH_OVERLAY_COLORMAP must be 'turbo' or 'gray'")
        self.depth_overlay_max_age = _env_float("MJLAB_DEPTH_OVERLAY_MAX_AGE", 0.5)
        if self.depth_overlay_max_age < 0.0:
            raise ValueError("MJLAB_DEPTH_OVERLAY_MAX_AGE must be non-negative")
        self.policy_depth_subscriber = None
        self.latest_raw_depth = None
        self.latest_raw_depth_time_s = None

        self.dt = self.mujoco_model.opt.timestep  # Get simulation timestep
        self.fps = 1 / self.dt  # Calculate frames per second (FPS)
        self.depth_capture_period_steps = max(
            1,
            round(self.fps / max(self.depth_capture_frequency, 1.0e-6)),
        )
        depth_enabled = (
            bool(self.depth_export_path)
            or self.depth_sink in ("ros", "both")
            or self.depth_overlay_mode == "raw"
        )
        if depth_enabled:
            self._init_depth_export()
            if self.depth_sink in ("ros", "both"):
                self.depth_ros_pub = RosDepthFramePublisher(
                    self.depth_ros_topic,
                    frame_id=self.depth_ros_frame_id,
                )
        if self.depth_overlay_mode == "policy":
            try:
                self.policy_depth_subscriber = RosPolicyDepthSubscriber(self.policy_depth_topic)
            except Exception as exc:
                print(f"Warning: policy depth overlay subscriber disabled: {exc}")

        # Launch the MuJoCo viewer with the XML's fixed third-person camera.
        self.viewer = viewer.launch_passive(self.mujoco_model, self.mujoco_data, key_callback=self.key_callback, show_left_ui=True, show_right_ui=True)
        track_camera_id = mujoco.mj_name2id(
            self.mujoco_model, mujoco.mjtObj.mjOBJ_CAMERA, "track"
        )
        if track_camera_id < 0:
            raise RuntimeError("MuJoCo follow camera 'track' not found in XML")
        self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.viewer.cam.fixedcamid = track_camera_id
        if self.depth_overlay_mode != "off":
            print(
                "*** Depth overlay enabled: "
                f"mode={self.depth_overlay_mode}, topic={self.policy_depth_topic}, "
                f"size=({self.depth_overlay_width}, {self.depth_overlay_height}), "
                f"range=({self.depth_overlay_min}, {self.depth_overlay_max}) ***"
            )

        # Initialize robot command data with default values
        self.robot_cmd = datatypes.RobotCmd()
        self.robot_cmd.mode = [0. for x in range(0, self.joint_num)]
        self.robot_cmd.q = [0. for x in range(0, self.joint_num)]
        self.robot_cmd.dq = [0. for x in range(0, self.joint_num)]
        self.robot_cmd.tau = [0. for x in range(0, self.joint_num)]
        self.robot_cmd.Kp = [0. for x in range(0, self.joint_num)]
        self.robot_cmd.Kd = [0. for x in range(0, self.joint_num)]

        # Initialize robot state data with default values
        self.robot_state = datatypes.RobotState()
        self.robot_state.tau = [0. for x in range(0, self.joint_num)]
        self.robot_state.q = [0. for x in range(0, self.joint_num)]
        self.robot_state.dq = [0. for x in range(0, self.joint_num)]

        # Initialize IMU data structure
        self.imu_data = datatypes.ImuData()
        self.imu_quat_sensor_adr = self._sensor_address("quat")
        self.imu_gyro_sensor_adr = self._sensor_address("gyro")
        self.imu_acc_sensor_adr = self._sensor_address("acc")

        # Set up callback for receiving robot commands in simulation mode
        self.robotCmdCallbackPartial = partial(self.robotCmdCallback)
        self.robot.subscribeRobotCmdForSim(self.robotCmdCallbackPartial)

    def _sensor_address(self, name):
        sensor_id = mujoco.mj_name2id(
            self.mujoco_model,
            mujoco.mjtObj.mjOBJ_SENSOR,
            name,
        )
        if sensor_id < 0:
            raise RuntimeError(f"MuJoCo sensor '{name}' not found")
        return self.mujoco_model.sensor_adr[sensor_id]

    def _init_depth_export(self):
        camera_id = mujoco.mj_name2id(
            self.mujoco_model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            self.depth_camera_name,
        )
        if camera_id < 0:
            raise RuntimeError(
                f"Depth camera '{self.depth_camera_name}' not found in MuJoCo XML"
            )
        height = int(os.getenv("MJLAB_DEPTH_HEIGHT", "480"))
        width = int(os.getenv("MJLAB_DEPTH_WIDTH", "848"))
        self.depth_renderer = mujoco.Renderer(self.mujoco_model, height=height, width=width)
        self.depth_renderer.enable_depth_rendering()
        self.depth_scene_option = mujoco.MjvOption()
        self.depth_scene_option.geomgroup[:] = 0
        self.depth_scene_option.geomgroup[:2] = 1
        if self.depth_export_path and self.depth_sink in ("npy", "both"):
            Path(self.depth_export_path).parent.mkdir(parents=True, exist_ok=True)
        print(
            f"*** Depth export enabled: camera={self.depth_camera_name}, "
            f"shape=({height}, {width}), sink={self.depth_sink}, "
            f"path={self.depth_export_path}, ros_topic={self.depth_ros_topic} ***"
        )

    def _export_depth_frame(self):
        if self.depth_renderer is None:
            return
        self.depth_renderer.update_scene(
            self.mujoco_data,
            camera=self.depth_camera_name,
            scene_option=self.depth_scene_option,
        )
        depth = self.depth_renderer.render().astype("float32", copy=False)
        if self.depth_overlay_mode == "raw":
            self.latest_raw_depth = depth.copy()
            self.latest_raw_depth_time_s = time.monotonic()
        if self.depth_sink in ("npy", "both") and self.depth_export_path:
            path = Path(self.depth_export_path)
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            with open(tmp_path, "wb") as f:
                import numpy as np

                np.save(f, depth, allow_pickle=False)
            os.replace(tmp_path, path)
        if self.depth_ros_pub is not None:
            self.depth_ros_pub.publish(depth)

    def _set_depth_overlay_status(self, label, detail):
        self.viewer.set_images([])
        self.viewer.set_texts(
            (
                mujoco.mjtFontScale.mjFONTSCALE_100,
                mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                label,
                detail,
            )
        )

    def _update_depth_overlay(self):
        if self.depth_overlay_mode == "off":
            return
        if self.depth_overlay_mode == "policy":
            if self.policy_depth_subscriber is None:
                self._set_depth_overlay_status("POLICY INPUT 30x45", "Subscriber unavailable")
                return
            depth, seq, recv_time_s = self.policy_depth_subscriber.latest()
            label = "POLICY INPUT 30x45"
        else:
            depth = self.latest_raw_depth
            seq = None
            recv_time_s = self.latest_raw_depth_time_s
            label = "RAW CAMERA — NOT POLICY INPUT"

        if depth is None or recv_time_s is None:
            self._set_depth_overlay_status(label, "Waiting for depth")
            return
        age_s = time.monotonic() - recv_time_s
        if age_s > self.depth_overlay_max_age:
            self._set_depth_overlay_status(label, f"Depth stale: {age_s * 1000.0:.0f} ms")
            return

        rgb = colorize_depth(
            depth,
            min_depth=self.depth_overlay_min,
            max_depth=self.depth_overlay_max,
            colormap=self.depth_overlay_colormap,
        )
        viewport = self.viewer.viewport
        x, y, width, height = fit_overlay_rect(
            depth.shape,
            (viewport.width, viewport.height),
            (self.depth_overlay_width, self.depth_overlay_height),
            self.depth_overlay_margin,
        )
        rect = mujoco.MjrRect(x, y, width, height)
        image = resize_rgb_nearest(rgb, height, width)
        valid = np.isfinite(depth) & (depth > 0.0)
        if np.any(valid):
            depth_min = float(np.min(depth[valid]))
            depth_max = float(np.max(depth[valid]))
            range_text = f"{depth_min:.2f}–{depth_max:.2f} m"
        else:
            range_text = "no valid depth"
        seq_text = "" if seq is None else f" | seq {seq}"
        self.viewer.set_images((rect, image))
        self.viewer.set_texts(
            (
                mujoco.mjtFontScale.mjFONTSCALE_100,
                mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                label,
                f"age {age_s * 1000.0:.0f} ms{seq_text} | {range_text}",
            )
        )

    # Callback function for receiving robot command data
    def robotCmdCallback(self, robot_cmd: datatypes.RobotCmd):
        self.robot_cmd = robot_cmd

    # Callback for keypress events in the MuJoCo viewer (currently does nothing)
    def key_callback(self, keycode):
        pass

    def run(self):
        frame_count = 0
        self.rate = Rate(self.fps)  # Set the update rate according to FPS
        while self.viewer.is_running():    
            # Step the MuJoCo physics simulation
            mujoco.mj_step(self.mujoco_model, self.mujoco_data)

            # Update robot state data from simulation
            for i in range(self.joint_num):
                self.robot_state.q[i] = self.mujoco_data.qpos[i + 7]
                self.robot_state.dq[i] = self.mujoco_data.qvel[i + 6]
                self.robot_state.tau[i] = self.mujoco_data.ctrl[i]

                # Apply control commands to the robot based on the received robot command data
                self.mujoco_data.ctrl[i] = (
                    self.robot_cmd.Kp[i] * (self.robot_cmd.q[i] - self.robot_state.q[i]) + 
                    self.robot_cmd.Kd[i] * (self.robot_cmd.dq[i] - self.robot_state.dq[i]) + 
                    self.robot_cmd.tau[i]
                )
        
            # Set the timestamp for the current robot state and publish it
            self.robot_state.stamp = time.time_ns()
            self.robot.publishRobotStateForSim(self.robot_state)

            # Extract IMU data (orientation, gyro, and acceleration) from simulation
            self.imu_data.quat[0] = self.mujoco_data.sensordata[self.imu_quat_sensor_adr + 0]
            self.imu_data.quat[1] = self.mujoco_data.sensordata[self.imu_quat_sensor_adr + 1]
            self.imu_data.quat[2] = self.mujoco_data.sensordata[self.imu_quat_sensor_adr + 2]
            self.imu_data.quat[3] = self.mujoco_data.sensordata[self.imu_quat_sensor_adr + 3]

            self.imu_data.gyro[0] = self.mujoco_data.sensordata[self.imu_gyro_sensor_adr + 0]
            self.imu_data.gyro[1] = self.mujoco_data.sensordata[self.imu_gyro_sensor_adr + 1]
            self.imu_data.gyro[2] = self.mujoco_data.sensordata[self.imu_gyro_sensor_adr + 2]

            self.imu_data.acc[0] = self.mujoco_data.sensordata[self.imu_acc_sensor_adr + 0]
            self.imu_data.acc[1] = self.mujoco_data.sensordata[self.imu_acc_sensor_adr + 1]
            self.imu_data.acc[2] = self.mujoco_data.sensordata[self.imu_acc_sensor_adr + 2]

            # Set the timestamp for the current IMU data and publish it
            self.imu_data.stamp = time.time_ns()
            self.robot.publishImuDataForSim(self.imu_data)

            if frame_count % self.depth_capture_period_steps == 0:
                self._export_depth_frame()

            # Sync the viewer every 20 frames for smoother visualization
            if frame_count % 20 == 0:
                self._update_depth_overlay()
                self.viewer.sync()

            frame_count += 1
            self.rate.sleep()  # Maintain the simulation loop at the correct rate

if __name__ == '__main__': 
    robot_type = os.getenv("ROBOT_TYPE")

    # Check if the ROBOT_TYPE environment variable is set, otherwise exit with an error
    if not robot_type:
        print("Error: Please set the ROBOT_TYPE using 'export ROBOT_TYPE=<robot_type>'.")
        sys.exit(1)

    # Create a Robot instance of the PointFoot type
    robot = Robot(RobotType.PointFoot, True)

    # Default IP address for the robot
    robot_ip = "127.0.0.1"
    
    # Check if command-line argument is provided for robot IP
    if len(sys.argv) > 1:
        robot_ip = sys.argv[1]

    # Initialize the robot with the provided IP address
    if not robot.init(robot_ip):
        sys.exit()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    scene_xml = os.getenv("MJLAB_SCENE", "robot.xml")
    if Path(scene_xml).name != scene_xml:
        print(f"Error: MJLAB_SCENE must be an XML filename, got '{scene_xml}'")
        sys.exit(1)

    # Define the path to the selected robot scene XML file based on robot type.
    model_path = f'{script_dir}/robot-description/pointfoot/{robot_type}/xml/{scene_xml}'

    # Check if the model file exists, otherwise exit with an error
    if not os.path.exists(model_path):
        print(f"Error: scene XML file does not exist: {model_path}")
        sys.exit(1)

    print(f"*** Model File Loaded: robot-description/pointfoot/{robot_type}/xml/{scene_xml} ***")

    # Define the names of the joint sensors used in the robot
    if robot_type.startswith("WF"):
        joint_sensor_names = [
            "abad_L_Joint", "hip_L_Joint", "knee_L_Joint", "wheel_L_Joint", "abad_R_Joint", "hip_R_Joint", "knee_R_Joint", "wheel_R_Joint"
        ]
    elif robot_type.startswith("SF"):
        joint_sensor_names = [
            "abad_L_Joint", "hip_L_Joint", "knee_L_Joint", "ankle_L_Joint", "abad_R_Joint", "hip_R_Joint", "knee_R_Joint", "ankle_R_Joint"
        ]
    else:
        joint_sensor_names = [
            "abad_L_Joint", "hip_L_Joint", "knee_L_Joint", "abad_R_Joint", "hip_R_Joint", "knee_R_Joint"
        ]

    # Create and run the MuJoCo simulator instance
    simulator = SimulatorMujoco(model_path, joint_sensor_names, robot)
    simulator.run()
