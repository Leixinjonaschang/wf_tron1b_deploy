import os
import sys
import copy
import numpy as np
import yaml
import time
import onnxruntime as ort
from scipy.spatial.transform import Rotation as R
from functools import partial
import limxsdk
import limxsdk.robot.Rate as Rate
import limxsdk.robot.Robot as Robot
import limxsdk.robot.RobotType as RobotType
import limxsdk.datatypes as datatypes
from mjlab_repts import (
    ACTION_CLIP,
    DEFAULT_OBS_NOISE_RANGES,
    LIN_PROPRIO_HISTORY_SHAPE,
    POLICY_ACTION_NAMES,
    SDK_JOINT_NAMES,
    STUDENT_HISTORY_SHAPE,
    LinProprioHistory,
    TermWiseHistory,
    build_actor_obs,
    build_actor_terms,
    build_lin_proprio_obs,
    clip_actions,
    command_from_joystick_axes,
    map_actions_to_sdk_joint_commands,
    validate_lin_policy_interface,
    validate_policy_interface,
    _metadata_float_array,
    _metadata_list,
)
from mjlab_repts_lin_depth import (
    GRU_HIDDEN_STATE_SHAPE as GRU_LIN_DEPTH_HIDDEN_STATE_SHAPE,
    HIDDEN_STATE_SHAPE as LIN_DEPTH_HIDDEN_STATE_SHAPE,
    POLICY_ACTION_NAMES as LIN_DEPTH_POLICY_ACTION_NAMES,
    PROPRIO_HISTORY_SHAPE as LIN_DEPTH_PROPRIO_HISTORY_SHAPE,
    ProprioHistory as LinDepthProprioHistory,
    build_proprio_obs as build_lin_depth_proprio_obs,
    build_proprio_terms as build_lin_depth_proprio_terms,
    create_depth_frame_source,
    validate_depth_policy_interface,
)

class WheelfootController:
    def __init__(self, model_dir, robot, robot_type, rl_type, start_controller):
        # Initialize robot and type information
        self.robot = robot
        self.robot_type = robot_type
        self.rl_type = rl_type
        self.is_mjlab_repts = self.rl_type == "mjlab_repts"
        self.is_mjlab_repts_lin = self.rl_type == "mjlab_repts_lin"
        self.is_mjlab_repts_lin_depth = self.rl_type == "mjlab_repts_lin_depth"
        self.is_mjlab_repts_gru_lin_depth = (
            self.rl_type == "mjlab_repts_gru_lin_depth"
        )
        self.is_mjlab_repts_depth = (
            self.is_mjlab_repts_lin_depth
            or self.is_mjlab_repts_gru_lin_depth
        )
        self.is_mjlab_policy = (
            self.is_mjlab_repts
            or self.is_mjlab_repts_lin
            or self.is_mjlab_repts_depth
        )
        self.depth_hidden_state_shape = (
            GRU_LIN_DEPTH_HIDDEN_STATE_SHAPE
            if self.is_mjlab_repts_gru_lin_depth
            else LIN_DEPTH_HIDDEN_STATE_SHAPE
        )
        self.start_controller = start_controller

        # Load configuration and model file paths based on robot type
        if self.is_mjlab_repts_depth:
            config_name = "params_mjlab_repts_lin_depth.yaml"
        elif self.is_mjlab_repts_lin:
            config_name = "params_mjlab_repts_lin.yaml"
        elif self.is_mjlab_repts:
            config_name = "params_mjlab_repts.yaml"
        else:
            config_name = "params.yaml"
        self.config_file = f'{model_dir}/{self.robot_type}/{config_name}'
        self.model_policy = f'{model_dir}/{self.robot_type}/policy/{self.rl_type}/policy.onnx'
        self.model_encoder = None if self.is_mjlab_policy else f'{model_dir}/{self.robot_type}/policy/{self.rl_type}/encoder.onnx'

        # Load configuration settings from the YAML file
        self.load_config(self.config_file)
        
        # Load the ONNX model
        self.initialize_onnx_models()

        # Prepare robot command structure with default values for mode, q, dq, tau, Kp, Kd
        self.robot_cmd = datatypes.RobotCmd()
        self.robot_cmd.mode = [0. for x in range(0, self.joint_num)]
        self.robot_cmd.q = [0. for x in range(0, self.joint_num)]
        self.robot_cmd.dq = [0. for x in range(0, self.joint_num)]
        self.robot_cmd.tau = [0. for x in range(0, self.joint_num)]
        self.robot_cmd.Kp = [self.control_cfg['stiffness'] for x in range(0, self.joint_num)]
        self.robot_cmd.Kd = [self.control_cfg['damping'] for x in range(0, self.joint_num)]

        # Prepare robot state structure
        self.robot_state = datatypes.RobotState()
        self.robot_state.tau = [0. for x in range(0, self.joint_num)]
        self.robot_state.q = [0. for x in range(0, self.joint_num)]
        self.robot_state.dq = [0. for x in range(0, self.joint_num)]
        self.robot_state_tmp = copy.deepcopy(self.robot_state)

        # Initialize IMU (Inertial Measurement Unit) data structure
        self.imu_data = datatypes.ImuData()
        self.imu_data.quat[0] = 0
        self.imu_data.quat[1] = 0
        self.imu_data.quat[2] = 0
        self.imu_data.quat[3] = 1
        self.imu_data_tmp = copy.deepcopy(self.imu_data)

        # Set up a callback to receive updated robot state data
        self.robot_state_callback_partial = partial(self.robot_state_callback)
        self.robot.subscribeRobotState(self.robot_state_callback_partial)

        # Set up a callback to receive updated IMU data
        self.imu_data_callback_partial = partial(self.imu_data_callback)
        self.robot.subscribeImuData(self.imu_data_callback_partial)

        # Set up a callback to receive updated SensorJoy
        self.sensor_joy_callback_partial = partial(self.sensor_joy_callback)
        self.robot.subscribeSensorJoy(self.sensor_joy_callback_partial)

        # Set up a callback to receive diagnostic data
        self.robot_diagnostic_callback_partial = partial(self.robot_diagnostic_callback)
        self.robot.subscribeDiagnosticValue(self.robot_diagnostic_callback_partial)

        # Initialize the calibration state to -1, indicating no calibration has occurred.
        self.calibration_state = -1

        # Flag to start the controller
        self.start_controller = start_controller

        # Gait index
        self.gait_index = 0

        # Flag indicating first received observation
        self.is_first_rec_obs = True

    def initialize_onnx_models(self):
        # Configure ONNX Runtime session options to optimize CPU usage
        session_options = ort.SessionOptions()
        # Limit the number of threads used for parallel computation within individual operators
        session_options.intra_op_num_threads = 1
        # Limit the number of threads used for parallel execution of different operators
        session_options.inter_op_num_threads = 1
        # Enable all possible graph optimizations to improve inference performance
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Disable CPU memory arena to reduce memory fragmentation
        session_options.enable_cpu_mem_arena = False
        # Disable memory pattern optimization to have more control over memory allocation
        session_options.enable_mem_pattern = False

        # Define execution providers to use CPU only, ensuring no GPU inference
        cpu_providers = ['CPUExecutionProvider']
        
        # Load the ONNX model and set up input and output names
        self.policy_session = ort.InferenceSession(self.model_policy, sess_options=session_options, providers=cpu_providers)
        self.policy_input_names = [self.policy_session.get_inputs()[i].name for i in range(self.policy_session.get_inputs().__len__())]
        self.policy_output_names = [self.policy_session.get_outputs()[i].name for i in range(self.policy_session.get_outputs().__len__())]
        self.policy_input_shapes = [self.policy_session.get_inputs()[i].shape for i in range(self.policy_session.get_inputs().__len__())]
        self.policy_output_shapes = [self.policy_session.get_outputs()[i].shape for i in range(self.policy_session.get_outputs().__len__())]
        self.policy_metadata = self.policy_session.get_modelmeta().custom_metadata_map

        if self.is_mjlab_repts:
            validate_policy_interface(
                self.policy_input_names,
                self.policy_input_shapes,
                self.policy_output_names,
                self.policy_output_shapes,
                self.policy_metadata,
            )
            self.apply_mjlab_repts_policy_metadata()
            self.encoder_session = None
            self.encoder_input_names = []
            self.encoder_output_names = []
            self.encoder_input_shapes = []
            self.encoder_output_shapes = []
            return

        if self.is_mjlab_repts_lin:
            validate_lin_policy_interface(
                self.policy_input_names,
                self.policy_input_shapes,
                self.policy_output_names,
                self.policy_output_shapes,
                self.policy_metadata,
            )
            self.apply_mjlab_repts_policy_metadata()
            self.encoder_session = None
            self.encoder_input_names = []
            self.encoder_output_names = []
            self.encoder_input_shapes = []
            self.encoder_output_shapes = []
            return

        if self.is_mjlab_repts_depth:
            validate_depth_policy_interface(
                self.policy_input_names,
                self.policy_input_shapes,
                self.policy_output_names,
                self.policy_output_shapes,
                self.policy_metadata,
                hidden_state_shape=self.depth_hidden_state_shape,
                policy_name=self.rl_type,
            )
            self.apply_mjlab_repts_policy_metadata()
            self.depth_hidden_state = np.zeros(
                self.depth_hidden_state_shape,
                dtype=np.float32,
            )
            self.encoder_session = None
            self.encoder_input_names = []
            self.encoder_output_names = []
            self.encoder_input_shapes = []
            self.encoder_output_shapes = []
            return

        self.encoder_session = ort.InferenceSession(self.model_encoder, sess_options=session_options, providers=cpu_providers)
        self.encoder_input_names = [self.encoder_session.get_inputs()[i].name for i in range(self.encoder_session.get_inputs().__len__())]
        self.encoder_output_names = [self.encoder_session.get_outputs()[i].name for i in range(self.encoder_session.get_outputs().__len__())]
        self.encoder_input_shapes = [self.encoder_session.get_inputs()[i].shape for i in range(self.encoder_session.get_inputs().__len__())]
        self.encoder_output_shapes = [self.encoder_session.get_outputs()[i].shape for i in range(self.encoder_session.get_outputs().__len__())]

    def apply_mjlab_repts_policy_metadata(self):
        action_target_names = _metadata_list(self.policy_metadata, "action_target_names")
        action_scale = _metadata_float_array(self.policy_metadata, "action_scale")
        if action_target_names is None or action_scale is None:
            return

        if action_target_names != list(POLICY_ACTION_NAMES):
            raise ValueError(
                f"{self.rl_type} ONNX metadata action_target_names must be "
                f"{list(POLICY_ACTION_NAMES)}, got {action_target_names}"
            )

        if action_scale.shape != (self.actions_size,):
            raise ValueError(
                f"{self.rl_type} ONNX metadata action_scale must have shape "
                f"({self.actions_size},), got {action_scale.shape}"
            )

        leg_action_scale = action_scale[:6]
        wheel_action_scale = action_scale[6:8]
        if not np.allclose(leg_action_scale, leg_action_scale[0]):
            raise ValueError(f"leg action_scale values must match, got {leg_action_scale}")
        if not np.allclose(wheel_action_scale, wheel_action_scale[0]):
            raise ValueError(f"wheel action_scale values must match, got {wheel_action_scale}")

        self.mjlab_repts_leg_action_scale = float(leg_action_scale[0])
        self.mjlab_repts_wheel_action_scale = float(wheel_action_scale[0])

    # Load the configuration from a YAML file
    def load_config(self, config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        # Assign configuration parameters to controller variables
        self.joint_names = config['PointfootCfg']['joint_names']
        self.init_state = config['PointfootCfg']['init_state']['default_joint_angle']
        self.stand_duration = config['PointfootCfg']['stand_mode']['stand_duration']
        self.control_cfg = config['PointfootCfg']['control']
        self.rl_cfg = config['PointfootCfg']['normalization']
        self.obs_scales = config['PointfootCfg']['normalization']['obs_scales']
        self.actions_size = config['PointfootCfg']['size']['actions_size']
        self.commands_size = config['PointfootCfg']['size']['commands_size']
        self.observations_size = config['PointfootCfg']['size']['observations_size']
        self.obs_history_length = config['PointfootCfg']['size']['obs_history_length']
        self.encoder_output_size = config['PointfootCfg']['size']['encoder_output_size']
        self.imu_orientation_offset = np.array(list(config['PointfootCfg']['imu_orientation_offset'].values()))
        self.user_cmd_cfg = config['PointfootCfg']['user_cmd_scales']
        self.loop_frequency = config['PointfootCfg']['loop_frequency']
        self.encoder_input_size = self.obs_history_length * self.observations_size
        self.mjlab_repts_history = TermWiseHistory(self.obs_history_length)
        if self.is_mjlab_repts_lin:
            self.mjlab_repts_history = LinProprioHistory(self.obs_history_length)
        elif self.is_mjlab_repts_depth:
            self.mjlab_repts_history = LinDepthProprioHistory(self.obs_history_length)
        self.mjlab_repts_diagnostics_printed = False
        self.mjlab_repts_policy_initialized = False
        self.mjlab_repts_action_clip = config['PointfootCfg']['control'].get('action_clip', ACTION_CLIP)
        self.mjlab_repts_leg_action_scale = config['PointfootCfg']['control'].get('leg_action_scale', 0.5)
        self.mjlab_repts_wheel_action_scale = config['PointfootCfg']['control'].get('wheel_action_scale', 10.0)
        self.mjlab_repts_leg_kp = config['PointfootCfg']['control'].get('stiffness', 40.0)
        self.mjlab_repts_leg_kd = config['PointfootCfg']['control'].get('damping', 1.8)
        observation_noise_cfg = config['PointfootCfg'].get('observation_noise', {})
        self.mjlab_repts_obs_noise_enabled = (
            self.is_mjlab_policy
            and self.start_controller
            and bool(observation_noise_cfg.get('enabled', False))
        )
        self.mjlab_repts_obs_noise_ranges = (
            observation_noise_cfg.get('ranges', DEFAULT_OBS_NOISE_RANGES)
            if self.mjlab_repts_obs_noise_enabled
            else None
        )
        self.mjlab_repts_obs_noise_rng = np.random.default_rng(
            observation_noise_cfg.get('seed')
        )
        self.depth_source = None
        self.last_depth_input = None
        self.depth_hidden_state = np.zeros(
            self.depth_hidden_state_shape,
            dtype=np.float32,
        )
        self.predicted_lin_vel = np.zeros(3, dtype=np.float32)
        if self.is_mjlab_repts_depth:
            depth_cfg = dict(config['PointfootCfg'].get('depth', {}))
            if self.is_mjlab_repts_lin_depth:
                depth_cfg.update(config['PointfootCfg'].get('legacy_depth', {}))
            self.depth_source = create_depth_frame_source(depth_cfg)

        # Initialize variables for actions, observations, and commands
        self.proprio_history_vector = np.zeros(self.obs_history_length * self.observations_size)
        self.encoder_out = np.zeros(self.encoder_output_size)
        self.actions = np.zeros(self.actions_size)
        self.observations = np.zeros(self.observations_size)
        self.last_actions = np.zeros(self.actions_size)
        self.commands = np.zeros(self.commands_size)  # command to the robot (e.g., velocity, rotation)
        self.scaled_commands = np.zeros(self.commands_size)
        self.base_lin_vel = np.zeros(3)  # base linear velocity
        self.base_position = np.zeros(3)  # robot base position
        self.loop_count = 0  # loop iteration count
        self.stand_percent = 0  # percentage of time the robot has spent in stand mode
        self.policy_session = None  # ONNX model session for policy inference
        self.joint_num = len(self.joint_names)  # number of joints

        self.joint_pos_idxs = config['PointfootCfg']['size']['jointpos_idxs']
        self.wheel_joint_damping = config['PointfootCfg']['control']['wheel_joint_damping']
        self.wheel_joint_torque_limit = config['PointfootCfg']['control']['wheel_joint_torque_limit']

        # Initialize joint angles based on the initial configuration
        self.init_joint_angles = np.zeros(len(self.joint_names))
        for i in range(len(self.joint_names)):
            self.init_joint_angles[i] = self.init_state[self.joint_names[i]]

        self.default_joint_vel = np.zeros(len(self.joint_names))
        if self.is_mjlab_repts:
            self.proprio_history_vector = np.zeros(STUDENT_HISTORY_SHAPE, dtype=np.float32)
        elif self.is_mjlab_repts_lin:
            self.proprio_history_vector = np.zeros(LIN_PROPRIO_HISTORY_SHAPE, dtype=np.float32)
        elif self.is_mjlab_repts_depth:
            self.proprio_history_vector = np.zeros(LIN_DEPTH_PROPRIO_HISTORY_SHAPE, dtype=np.float32)
        
        # Set initial mode to "STAND"
        self.mode = "STAND"
    
    # Main control loop
    def run(self):
        # Wait until the controller is started
        while not self.start_controller:
          time.sleep(1)

        # Initialize default joint angles for standing
        self.default_joint_angles = np.array([0.0] * len(self.joint_names))
        self.stand_percent += 1 / (self.stand_duration * self.loop_frequency)
        self.mode = "STAND"
        self.loop_count = 0

        # Set the loop rate based on the frequency in the configuration
        rate = Rate(self.loop_frequency)
        while self.start_controller:
            self.update()
            rate.sleep()
        
        # Reset robot command values to ensure a safe stop when exiting the loop
        self.robot_cmd.q = [0. for x in range(0, self.joint_num)]
        self.robot_cmd.dq = [0. for x in range(0, self.joint_num)]
        self.robot_cmd.tau = [0. for x in range(0, self.joint_num)]
        self.robot_cmd.Kp = [0. for x in range(0, self.joint_num)]
        self.robot_cmd.Kd = [1.0 for x in range(0, self.joint_num)]
        self.robot.publishRobotCmd(self.robot_cmd)
        time.sleep(1)

    # Handle the stand mode for smoothly transitioning the robot into standing
    def handle_stand_mode(self):
        self.init_state["hip_L_Joint"] = -0.9
        self.init_state["hip_R_Joint"] = 0.9
        if self.stand_percent < 1:
            for j in range(len(self.joint_names)):
                if (j + 1) % 4 != 0:
                    # Interpolate between initial and default joint angles during stand mode
                    pos_des = self.default_joint_angles[j] * (1 - self.stand_percent) + self.init_state[self.joint_names[j]] * self.stand_percent
                    self.set_joint_command(j, pos_des, 0, 0, self.control_cfg['stiffness'], self.control_cfg['damping'])
                else:
                    self.set_joint_command(0, 0, 0, self.wheel_joint_damping, 0, 0)
            # Increment the stand percentage over time
            self.stand_percent += 3 / (self.stand_duration * self.loop_frequency)
        else:
            # Switch to walk mode after standing
            self.mode = "WALK"

    # Handle the walk mode where the robot moves based on computed actions
    def handle_walk_mode(self):
        if self.is_mjlab_policy:
            self.handle_mjlab_repts_walk_mode()
            return

        self.init_joint_angles[1] = 0.0
        self.init_joint_angles[5] = 0.0

        # Update the temporary robot state and IMU data
        self.robot_state_tmp = copy.deepcopy(self.robot_state)
        self.imu_data_tmp = copy.deepcopy(self.imu_data)

        # Execute actions every 'decimation' iterations
        if self.loop_count % self.control_cfg['decimation'] == 0:
            self.compute_observation()
            self.compute_encoder()
            self.compute_actions()
            # Clip the actions within predefined limits
            action_min = -self.rl_cfg['clip_scales']['clip_actions']
            action_max = self.rl_cfg['clip_scales']['clip_actions']
            self.actions = np.clip(self.actions, action_min, action_max)

            # swap actions positions back to deep first, only when action updated
            if self.rl_type == "isaaclab":
                self.actions = self.swap_positions(self.actions, reverse=True)

        # Iterate over the joints and set commands based on actions
        joint_pos = np.array(self.robot_state_tmp.q)
        joint_vel = np.array(self.robot_state_tmp.dq)

        for i in range(len(joint_pos)):
            if (i + 1) % 4 != 0:
                # Compute the limits for the action based on joint position and velocity
                action_min = (joint_pos[i] - self.init_joint_angles[i] +
                              (self.control_cfg['damping'] * joint_vel[i] - self.control_cfg['user_torque_limit']) /
                              self.control_cfg['stiffness'])
                action_max = (joint_pos[i] - self.init_joint_angles[i] +
                              (self.control_cfg['damping'] * joint_vel[i] + self.control_cfg['user_torque_limit']) /
                              self.control_cfg['stiffness'])

                # Clip action within limits
                self.actions[i] = max(action_min / self.control_cfg['action_scale_pos'],
                                      min(action_max / self.control_cfg['action_scale_pos'], self.actions[i]))

                # Compute the desired joint position and set it
                pos_des = self.actions[i] * self.control_cfg['action_scale_pos'] + self.init_joint_angles[i]
                self.set_joint_command(i, pos_des, 0, 0, self.control_cfg['stiffness'], self.control_cfg['damping'])

                # Save the last action for reference
                self.last_actions[i] = self.actions[i]
            else:
                action_min = joint_vel[i] - self.wheel_joint_torque_limit / self.wheel_joint_damping
                action_max = joint_vel[i] + self.wheel_joint_torque_limit / self.wheel_joint_damping
                self.last_actions[i] = self.actions[i]
                self.actions[i] = max(action_min / self.wheel_joint_damping,
                                      min(action_max / self.wheel_joint_damping, self.actions[i]))
                velocity_des = self.actions[i] * self.wheel_joint_damping
                self.set_joint_command(i, 0, velocity_des, 0, 0, self.wheel_joint_damping)

    def handle_mjlab_repts_walk_mode(self):
        # Update the temporary robot state and IMU data
        self.robot_state_tmp = copy.deepcopy(self.robot_state)
        self.imu_data_tmp = copy.deepcopy(self.imu_data)

        if (not self.mjlab_repts_policy_initialized) or self.loop_count % self.control_cfg['decimation'] == 0:
            self.compute_observation()
            self.compute_actions()
            self.actions = clip_actions(self.actions, self.mjlab_repts_action_clip)
            self.last_actions = self.actions.copy()
            self.mjlab_repts_policy_initialized = True
            self.print_mjlab_repts_diagnostics()

        joint_commands = map_actions_to_sdk_joint_commands(
            self.actions,
            self.init_joint_angles,
            leg_action_scale=self.mjlab_repts_leg_action_scale,
            wheel_action_scale=self.mjlab_repts_wheel_action_scale,
            leg_kp=self.mjlab_repts_leg_kp,
            leg_kd=self.mjlab_repts_leg_kd,
            wheel_kp=0.0,
            wheel_kd=self.wheel_joint_damping,
        )
        for i in range(self.joint_num):
            self.set_joint_command(
                i,
                joint_commands["q"][i],
                joint_commands["dq"][i],
                joint_commands["tau"][i],
                joint_commands["Kp"][i],
                joint_commands["Kd"][i],
            )

    def print_mjlab_repts_diagnostics(self):
        if self.mjlab_repts_diagnostics_printed:
            return

        tag = f"[{self.rl_type}]"
        action_names = (
            LIN_DEPTH_POLICY_ACTION_NAMES
            if self.is_mjlab_repts_depth
            else POLICY_ACTION_NAMES
        )
        print(tag, "policy:", self.model_policy)
        print(tag, "inputs:", list(zip(self.policy_input_names, self.policy_input_shapes)))
        print(tag, "outputs:", list(zip(self.policy_output_names, self.policy_output_shapes)))
        print(tag, "action order:", list(action_names))
        print(tag, "SDK joint order:", list(SDK_JOINT_NAMES))
        print(tag, "command range: vx_body [-1, 1], vy_body [-1, 1], yaw_rate [-pi/2, pi/2]")
        print(tag, "encoder: disabled")
        if self.is_mjlab_repts_depth:
            print(tag, "depth source:", type(self.depth_source).__name__)
            profile = (
                "metric meters (ONNX preprocessing)"
                if self.is_mjlab_repts_gru_lin_depth
                else "legacy metric"
            )
            print(tag, "depth processing:", profile)
            if self.last_depth_input is not None:
                print(
                    tag,
                    "depth input min/max/mean:",
                    float(np.min(self.last_depth_input)),
                    float(np.max(self.last_depth_input)),
                    float(np.mean(self.last_depth_input)),
                )
        if self.is_mjlab_repts_lin:
            print(tag, "predicted lin vel:", self.predicted_lin_vel)
        print(tag, "sim2sim obs noise enabled:", self.mjlab_repts_obs_noise_enabled)
        self.mjlab_repts_diagnostics_printed = True

    def swap_positions(self, initial_array, reverse=False, exclude_wheel=False):
        if not exclude_wheel:
            joint_idx_lab = [0, 4, 1, 5, 2, 6, 3, 7]
        else:
            joint_idx_lab = [0, 3, 1, 4, 2, 5]
        new_array = np.zeros(initial_array.shape)
        for i in range(len(joint_idx_lab)):
            if not reverse:
                new_array[i] = initial_array[joint_idx_lab[i]]
            else:
                new_array[joint_idx_lab[i]] = initial_array[i]
        return new_array

    def compute_mjlab_repts_observation(self):
        imu_orientation = np.array(self.imu_data_tmp.quat)
        q_wi = R.from_quat(imu_orientation).as_euler('zyx')
        inverse_rot = R.from_euler('zyx', q_wi).inv().as_matrix()

        gravity_vector = np.array([0, 0, -1])
        projected_gravity = np.dot(inverse_rot, gravity_vector)

        base_ang_vel = np.array(self.imu_data_tmp.gyro)
        rot = R.from_euler('zyx', self.imu_orientation_offset).as_matrix()
        base_ang_vel = np.dot(rot, base_ang_vel)
        projected_gravity = np.dot(rot, projected_gravity)

        if self.is_mjlab_repts_depth:
            terms = build_lin_depth_proprio_terms(
                base_ang_vel,
                projected_gravity,
                np.array(self.robot_state_tmp.q),
                np.array(self.robot_state_tmp.dq),
                self.init_joint_angles,
                self.default_joint_vel,
                self.last_actions,
                noise_ranges=self.mjlab_repts_obs_noise_ranges,
                rng=self.mjlab_repts_obs_noise_rng,
            )
            self.observations = build_lin_depth_proprio_obs(terms)
            self.proprio_history_vector = self.mjlab_repts_history.update_and_matrix(terms)
            return

        terms = build_actor_terms(
            base_ang_vel,
            projected_gravity,
            np.array(self.robot_state_tmp.q),
            np.array(self.robot_state_tmp.dq),
            self.init_joint_angles,
            self.default_joint_vel,
            self.last_actions,
            self.commands,
            noise_ranges=self.mjlab_repts_obs_noise_ranges,
            rng=self.mjlab_repts_obs_noise_rng,
        )
        self.observations = (
            build_lin_proprio_obs(terms)
            if self.is_mjlab_repts_lin
            else build_actor_obs(terms)
        )
        self.proprio_history_vector = self.mjlab_repts_history.update_and_matrix(terms)
    
    def compute_observation(self):
        if self.is_mjlab_policy:
            self.compute_mjlab_repts_observation()
            return

        # Convert IMU orientation from quaternion to Euler angles (ZYX convention)
        imu_orientation = np.array(self.imu_data_tmp.quat)
        q_wi = R.from_quat(imu_orientation).as_euler('zyx')  # Quaternion to Euler ZYX conversion
        inverse_rot = R.from_euler('zyx', q_wi).inv().as_matrix()  # Get the inverse rotation matrix

        # Project the gravity vector (pointing downwards) into the body frame
        gravity_vector = np.array([0, 0, -1])  # Gravity in world frame (z-axis down)
        projected_gravity = np.dot(inverse_rot, gravity_vector)  # Transform gravity into body frame

        # Retrieve base angular velocity from the IMU data
        base_ang_vel = np.array(self.imu_data_tmp.gyro)
        # Apply IMU orientation offset correction (using Euler angles)
        rot = R.from_euler('zyx', self.imu_orientation_offset).as_matrix()  # Rotation matrix for offset correction
        base_ang_vel = np.dot(rot, base_ang_vel)  # Apply correction to angular velocity
        projected_gravity = np.dot(rot, projected_gravity)  # Apply correction to projected gravity

        # Retrieve joint positions and velocities from the robot state
        joint_positions = np.array(self.robot_state_tmp.q)
        joint_velocities = np.array(self.robot_state_tmp.dq)

        # Retrieve the last actions that were applied to the robot
        actions = np.array(self.last_actions)

        # Create a command scaler matrix for linear and angular velocities
        command_scaler = np.diag([
            self.user_cmd_cfg['lin_vel_x'],  # Scale factor for linear velocity in x direction
            self.user_cmd_cfg['lin_vel_y'],  # Scale factor for linear velocity in y direction
            self.user_cmd_cfg['ang_vel_yaw']  # Scale factor for yaw (angular velocity)
        ])

        # Apply scaling to the command inputs (velocity commands)
        self.scaled_commands = np.dot(command_scaler, self.commands)

        # Populate observation vector
        joint_pos_value = (joint_positions - self.init_joint_angles) * self.obs_scales['dof_pos']

        # In WF, joint pos does not include wheel speed, index(3, 7) needs to be removed
        joint_pos_input = np.array([joint_pos_value[idx] for idx in self.joint_pos_idxs])
        # swap positions in joint_pos, joint_vel and actions if mode is isaaclab
        if self.rl_type == "isaaclab":
            joint_pos_input = self.swap_positions(joint_pos_input, exclude_wheel=True)
            joint_velocities = self.swap_positions(joint_velocities)
            actions = self.swap_positions(actions)

        # Create the observation vector by concatenating various state variables:
        # - Base angular velocity (scaled)
        # - Projected gravity vector
        # - Joint positions (difference from initial angles, scaled)
        # - Joint velocities (scaled)
        # - Last actions applied to the robot
        # - Scaled command inputs
        obs = np.concatenate([
            base_ang_vel * self.obs_scales['ang_vel'],  # Scaled base angular velocity
            projected_gravity,  # Projected gravity vector in body frame
            joint_pos_input,  # Scaled joint positions
            joint_velocities * self.obs_scales['dof_vel'],  # Scaled joint velocities
            actions  # Last actions taken by the robot
        ])

        # Check if this is the first recorded observation
        if self.is_first_rec_obs:
            # Calculate the total size of the encoder input
            input_size = np.prod(self.encoder_input_shapes[0])
            
            # Initialize the proprioceptive history buffer with zeros
            self.proprio_history_buffer = np.zeros(input_size)

            # Fill the proprioceptive history buffer with the current observation for the entire history length
            for i in range(self.obs_history_length):
                self.proprio_history_buffer[i * self.observations_size:(i + 1) * self.observations_size] = obs

            # Update the flag to indicate that the first observation has been processed
            self.is_first_rec_obs = False
        
        # Shift the existing proprioceptive history buffer to the left
        self.proprio_history_buffer[:-self.observations_size] = self.proprio_history_buffer[self.observations_size:]

        # Add the current observation to the end of the proprioceptive history buffer
        self.proprio_history_buffer[-self.observations_size:] = obs

        # Convert the proprioceptive history buffer to a numpy array
        self.proprio_history_vector = np.array(self.proprio_history_buffer)

        # Clip the observation values to within the specified limits for stability
        self.observations = np.clip(
            obs, 
            -self.rl_cfg['clip_scales']['clip_observations'],  # Lower limit for clipping
            self.rl_cfg['clip_scales']['clip_observations']  # Upper limit for clipping
        )

    def compute_actions(self):
        """
        Computes the actions based on the current observations using the policy session.
        """
        if self.is_mjlab_repts:
            student_history = self.proprio_history_vector.astype(np.float32).reshape(
                1, *STUDENT_HISTORY_SHAPE
            )
            inputs = {
                self.policy_input_names[0]: student_history,
            }
            output = self.policy_session.run(self.policy_output_names, inputs)
            self.actions = np.array(output).flatten()
            return

        if self.is_mjlab_repts_lin:
            proprio_history = self.proprio_history_vector.astype(np.float32).reshape(
                1, *LIN_PROPRIO_HISTORY_SHAPE
            )
            actor_command = self.commands.astype(np.float32).reshape(1, 3)
            inputs = {
                self.policy_input_names[0]: proprio_history,
                self.policy_input_names[1]: actor_command,
            }
            output = self.policy_session.run(self.policy_output_names, inputs)
            self.actions = np.asarray(output[0], dtype=np.float32).reshape(-1)
            self.predicted_lin_vel = np.asarray(output[1], dtype=np.float32).reshape(-1)
            return

        if self.is_mjlab_repts_depth:
            if self.depth_source is None:
                raise RuntimeError("depth source is not initialized")
            proprio_history = self.proprio_history_vector.astype(np.float32).reshape(
                1, *LIN_DEPTH_PROPRIO_HISTORY_SHAPE
            )
            actor_command = self.commands.astype(np.float32).reshape(1, 3)
            depth = self.depth_source.frame().astype(np.float32)
            self.last_depth_input = depth
            hidden_state = self.depth_hidden_state.astype(np.float32)
            inputs = {
                self.policy_input_names[0]: proprio_history,
                self.policy_input_names[1]: actor_command,
                self.policy_input_names[2]: depth,
                self.policy_input_names[3]: hidden_state,
            }
            output = self.policy_session.run(self.policy_output_names, inputs)
            self.actions = np.asarray(output[0], dtype=np.float32).reshape(-1)
            self.predicted_lin_vel = np.asarray(output[1], dtype=np.float32).reshape(-1)
            self.depth_hidden_state = np.asarray(output[2], dtype=np.float32).reshape(
                self.depth_hidden_state_shape
            )
            return

        # Concatenate observations into a single tensor and convert to float32
        input_tensor = np.concatenate([self.encoder_out, self.observations, self.scaled_commands], axis=0)
        input_tensor = input_tensor.astype(np.float32)
        
        # Create a dictionary of inputs for the policy session
        inputs = {self.policy_input_names[0]: input_tensor}
        
        # Run the policy session and get the output
        output = self.policy_session.run(self.policy_output_names, inputs)
        
        # Flatten the output and store it as actions
        self.actions = np.array(output).flatten()

    def compute_encoder(self):
        """
        Computes the encoder output based on the proprioceptive history buffer.

        This method first concatenates the proprioceptive history buffer into a single input tensor.
        Then it converts the input tensor to the float32 data type. After that, it creates a dictionary
        of inputs for the encoder session and runs the encoder session to get the output. Finally,
        it flattens the output and stores it as the encoder output.
        """
        if self.is_mjlab_policy:
            return

        # Concatenate the proprioceptive history buffer into a single tensor and convert to float32
        input_tensor = np.concatenate([self.proprio_history_buffer], axis=0)
        input_tensor = input_tensor.astype(np.float32)

        # Create a dictionary of inputs for the encoder session
        inputs = {self.encoder_input_names[0]: input_tensor}

        # Run the encoder session and get the output
        output = self.encoder_session.run(self.encoder_output_names, inputs)

        # Flatten the output and store it as the encoder output
        self.encoder_out = np.array(output).flatten()
 
    def set_joint_command(self, joint_index, q, dq, tau, kp, kd):
        """
        Sends a command to configure the state of a specific joint.
        This method updates the joint's desired position, velocity, torque, and control gains.
        Replace this implementation with the actual communication logic for your hardware.

        Parameters:
        joint_index (int): The index of the joint to be controlled.
        q (float): The desired joint position, typically in radians or degrees.
        dq (float): The desired joint velocity, typically in radians/second or degrees/second.
        tau (float): The desired joint torque, typically in Newton-meters (Nm).
        kp (float): The proportional gain for position control.
        kd (float): The derivative gain for velocity control.
        """
        self.robot_cmd.q[joint_index] = q
        self.robot_cmd.dq[joint_index] = dq
        self.robot_cmd.tau[joint_index] = tau
        self.robot_cmd.Kp[joint_index] = kp
        self.robot_cmd.Kd[joint_index] = kd

    def update(self):
        """
        Updates the robot's state based on the current mode and publishes the robot command.
        """
        if self.mode == "STAND":
            self.handle_stand_mode()
        elif self.mode == "WALK":
            self.handle_walk_mode()
        
        # Increment the loop count
        self.loop_count += 1

        # Publish the robot command
        self.robot.publishRobotCmd(self.robot_cmd)
        
    # Callback function for receiving robot command data
    def robot_state_callback(self, robot_state: datatypes.RobotState):
        """
        Callback function to update the robot state from incoming data.
        
        Parameters:
        robot_state (datatypes.RobotState): The current state of the robot.
        """
        self.robot_state = robot_state

    # Callback function for receiving imu data
    def imu_data_callback(self, imu_data: datatypes.ImuData):
        """
        Callback function to update IMU data from incoming data.
        
        Parameters:
        imu_data (datatypes.ImuData): The IMU data containing stamp, acceleration, gyro, and quaternion.
        """
        self.imu_data.stamp = imu_data.stamp
        self.imu_data.acc = imu_data.acc
        self.imu_data.gyro = imu_data.gyro
        
        # Rotate quaternion values
        self.imu_data.quat[0] = imu_data.quat[1]
        self.imu_data.quat[1] = imu_data.quat[2]
        self.imu_data.quat[2] = imu_data.quat[3]
        self.imu_data.quat[3] = imu_data.quat[0]

    # Callback function for receiving sensor joy data
    def sensor_joy_callback(self, sensor_joy: datatypes.SensorJoy):
        # Check if the robot is in the calibration state and both L1 (button index 4) and Y (button index 3) buttons are pressed.
        if not self.start_controller and self.calibration_state == 0 and sensor_joy.buttons[4] == 1 and sensor_joy.buttons[3] == 1:
          print(f"L1 + Y: start_controller...")
          self.start_controller = True

        # Check if both L1 (button index 4) and X (button index 2) are pressed to stop the controller
        if self.start_controller and sensor_joy.buttons[4] == 1 and sensor_joy.buttons[2] == 1:
          print(f"L1 + X: stop_controller...")
          self.start_controller = False

        linear_x  = sensor_joy.axes[1]
        linear_y  = sensor_joy.axes[0]
        angular_z = sensor_joy.axes[2]

        linear_x  = 1.0 if linear_x > 1.0 else (-1.0 if linear_x < -1.0 else linear_x)
        linear_y  = 1.0 if linear_y > 1.0 else (-1.0 if linear_y < -1.0 else linear_y)
        angular_z = 1.0 if angular_z > 1.0 else (-1.0 if angular_z < -1.0 else angular_z)

        if self.is_mjlab_policy:
            self.commands = command_from_joystick_axes(sensor_joy.axes)
            return

        self.commands[0] = linear_x * 0.5
        self.commands[1] = linear_y * 0.5
        self.commands[2] = angular_z * 0.5

    # Callback function for receiving diagnostic data
    def robot_diagnostic_callback(self, diagnostic_value: datatypes.DiagnosticValue):
      # Check if the received diagnostic data is related to calibration.
      if diagnostic_value.name == "calibration":
        print(f"Calibration state: {diagnostic_value.code}")
        self.calibration_state = diagnostic_value.code
