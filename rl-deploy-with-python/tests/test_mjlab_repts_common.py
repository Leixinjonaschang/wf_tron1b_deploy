from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


DEPLOY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEPLOY_ROOT))

from mjlab_repts import (  # noqa: E402
    ACTION_CLIP,
    POLICY_ACTION_NAMES,
    SDK_JOINT_NAMES,
    clip_actions,
    command_from_joystick_axes,
    map_actions_to_sdk_joint_commands,
    rate_limit_commands,
)


class MjlabRepTsCommonTest(unittest.TestCase):
    def test_joystick_command_scales_velocity(self):
        command = command_from_joystick_axes(
            np.array([-2.0, 0.5, 2.0], dtype=np.float32)
        )
        np.testing.assert_allclose(
            command,
            np.array([1.0, -1.0, np.pi / 2.0], dtype=np.float32),
        )

        max_forward = command_from_joystick_axes(
            np.array([0.0, 1.0, 0.0], dtype=np.float32)
        )
        np.testing.assert_allclose(
            max_forward,
            np.array([2.0, 0.0, 0.0], dtype=np.float32),
        )

        max_reverse = command_from_joystick_axes(
            np.array([0.0, -1.0, 0.0], dtype=np.float32)
        )
        np.testing.assert_allclose(
            max_reverse,
            np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        )

    def test_velocity_command_acceleration_is_rate_limited(self):
        rate_limits = np.array([1.0, 2.0, 4.0], dtype=np.float32)
        command = rate_limit_commands(
            current=np.zeros(3, dtype=np.float32),
            target=np.array([1.0, -1.0, 0.1], dtype=np.float32),
            rate_limits=rate_limits,
            dt=0.1,
        )
        np.testing.assert_allclose(command, [0.1, -0.2, 0.1])

    def test_velocity_command_deceleration_is_immediate(self):
        command = rate_limit_commands(
            current=np.array([1.0, -1.0, 0.5], dtype=np.float32),
            target=np.array([0.2, -0.1, 0.0], dtype=np.float32),
            rate_limits=np.array([1.0, 2.0, 4.0], dtype=np.float32),
            dt=0.1,
        )
        np.testing.assert_allclose(command, [0.2, -0.1, 0.0])

    def test_velocity_command_reversal_limits_only_magnitude_increase(self):
        reversed_command = rate_limit_commands(
            current=np.array([0.4, -0.4, 0.4], dtype=np.float32),
            target=np.array([-1.0, 1.0, -0.4], dtype=np.float32),
            rate_limits=np.array([1.0, 2.0, 4.0], dtype=np.float32),
            dt=0.1,
        )
        np.testing.assert_allclose(reversed_command, [-0.5, 0.6, -0.4])

    def test_actions_clip_to_policy_range(self):
        actions = clip_actions(
            np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 2.5, 3.0])
        )
        np.testing.assert_allclose(
            actions,
            np.array(
                [
                    -ACTION_CLIP,
                    -2.0,
                    -1.0,
                    0.0,
                    1.0,
                    2.0,
                    ACTION_CLIP,
                    ACTION_CLIP,
                ]
            ),
        )

    def test_policy_actions_map_to_sdk_joint_commands_by_name(self):
        actions = np.array(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            dtype=np.float32,
        )
        default_q = np.arange(8, dtype=np.float32)

        commands = map_actions_to_sdk_joint_commands(actions, default_q)
        q_by_name = dict(zip(SDK_JOINT_NAMES, commands["q"]))
        dq_by_name = dict(zip(SDK_JOINT_NAMES, commands["dq"]))

        self.assertEqual(POLICY_ACTION_NAMES[3], "abad_R_Joint")
        np.testing.assert_allclose(
            q_by_name["abad_L_Joint"],
            0.0 + 0.5 * actions[0],
        )
        np.testing.assert_allclose(
            q_by_name["knee_R_Joint"],
            6.0 + 0.5 * actions[5],
        )
        np.testing.assert_allclose(dq_by_name["wheel_L_Joint"], 10.0 * actions[6])
        np.testing.assert_allclose(dq_by_name["wheel_R_Joint"], 10.0 * actions[7])


if __name__ == "__main__":
    unittest.main()
