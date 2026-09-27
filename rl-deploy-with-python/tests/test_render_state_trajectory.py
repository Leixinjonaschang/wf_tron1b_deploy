"""Check time windows, world-space trajectories and spline fitting without OpenGL."""

import importlib.util
from pathlib import Path
import unittest

import mujoco
import numpy as np


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "render_state_trajectory.py"
SPEC = importlib.util.spec_from_file_location("render_state_trajectory", SCRIPT)
renderer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(renderer)

ROBOT_XML = """
<mujoco>
  <worldbody>
    <body name="base_Link" pos="0 0 1">
      <freejoint/>
      <inertial pos="0.2 0 -0.1" mass="1" diaginertia="0.01 0.01 0.01"/>
      <geom type="sphere" size="0.1"/>
      <body pos="0 1 0">
        <joint type="hinge" axis="0 0 1"/>
        <geom type="sphere" size="0.1"/>
        <body name="wheel_L_Link" pos="1 0 0">
          <geom type="sphere" size="0.1"/>
        </body>
      </body>
      <body name="wheel_R_Link" pos="1 -1 0">
        <geom type="sphere" size="0.1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


class WheelTrajectoryTests(unittest.TestCase):
    def test_centers_follow_root_translation_rotation_and_leg_articulation(self):
        model = mujoco.MjModel.from_xml_string(ROBOT_XML)
        poses = np.tile(model.qpos0, (3, 1))
        poses[1:, :3] = [3, 4, 2]
        poses[1:, 3:7] = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
        poses[2, 7] = np.pi / 2
        positions = renderer.body_trajectories(
            mujoco, model, mujoco.MjData(model), poses, np.arange(3),
            renderer.WHEEL_BODY_NAMES,
        )
        np.testing.assert_allclose(
            positions,
            [[[1, 1, 1], [2, 5, 2], [1, 4, 2]],
             [[1, -1, 1], [4, 5, 2], [4, 5, 2]]],
            atol=1e-12,
        )

    def test_full_trail_preserves_turns_when_only_one_pose_is_displayed(self):
        times = np.arange(6, dtype=float)
        model = mujoco.MjModel.from_xml_string(ROBOT_XML)
        poses = np.tile(model.qpos0, (6, 1))
        poses[:, 0] = [100, 0, 2, -1, 1, 100]
        selected = renderer.select_frames(times, 1, 4, count=1)
        np.testing.assert_array_equal(selected, [1])
        indices = renderer.interval_frames(times, 1, 4)
        positions = renderer.body_trajectories(
            mujoco, model, mujoco.MjData(model), poses, indices,
            renderer.WHEEL_BODY_NAMES,
        )
        np.testing.assert_array_equal(times[indices], [1, 2, 3, 4])
        np.testing.assert_allclose(positions[0, :, 0], [1, 3, 0, 2])

    def test_interval_does_not_extrapolate_or_pick_rows_outside_bounds(self):
        times = np.array([0.001, 0.101, 0.201, 0.301])
        np.testing.assert_array_equal(renderer.interval_frames(times, 0.1, 0.3), [1, 2])
        np.testing.assert_array_equal(renderer.interval_frames(times, 0.201, 0.201), [2])
        with self.assertRaisesRegex(ValueError, "No recorded states"):
            renderer.interval_frames(times, 0.11, 0.19)


class BaseTrajectoryTests(unittest.TestCase):
    def test_base_com_includes_rotated_inertial_offset_and_ignores_leg_motion(self):
        model = mujoco.MjModel.from_xml_string(ROBOT_XML)
        poses = np.tile(model.qpos0, (3, 1))
        poses[1:, :3] = [3, 4, 2]
        poses[1:, 3:7] = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
        poses[2, 7] = np.pi / 2
        positions = renderer.body_trajectories(
            mujoco, model, mujoco.MjData(model), poses, np.arange(3),
            (renderer.BASE_BODY_NAME,), center_of_mass=True,
        )
        np.testing.assert_allclose(
            positions[0], [[0.2, 0, 0.9], [3, 4.2, 1.9], [3, 4.2, 1.9]],
            atol=1e-12,
        )


class ContactPointTests(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string('''
        <mujoco>
          <worldbody>
            <geom name="floor" type="plane" size="5 5 0.1"/>
            <geom name="step" type="box" pos="2 0 0.2" size="0.5 1 0.2"/>
            <body name="base_Link" pos="0 0 0.099">
              <freejoint/>
              <geom name="base_collision" type="sphere" size="0.05" pos="0 0 0.5"/>
              <body name="wheel_L_Link" pos="0 -0.4 0">
                <geom name="wheel_L_collision" type="sphere" size="0.1"/>
              </body>
              <body name="wheel_R_Link" pos="0 0.4 0">
                <geom name="wheel_R_collision" type="sphere" size="0.1"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        ''')
        self.data = mujoco.MjData(self.model)

    def contacts(self):
        mujoco.mj_forward(self.model, self.data)
        return renderer.wheel_terrain_contacts(mujoco, self.model, self.data)

    def test_flat_ground_contacts_match_collision_positions(self):
        contacts = self.contacts()
        self.assertEqual({c['wheel_body'] for c in contacts}, set(renderer.WHEEL_BODY_NAMES))
        self.assertEqual(len(contacts), 2)
        for contact in contacts:
            self.assertIn('floor', contact['geom_names'])
            side_y = -0.4 if contact['wheel_body'] == 'wheel_L_Link' else 0.4
            np.testing.assert_allclose(
                contact['position_world_m'], [0, side_y, -0.0005], atol=1e-12,
            )

    def test_raised_terrain_contacts_keep_the_surface_height(self):
        self.data.qpos[:3] = [2, 0, 0.499]
        contacts = self.contacts()
        self.assertEqual(len(contacts), 2)
        for contact in contacts:
            self.assertIn('step', contact['geom_names'])
            self.assertAlmostEqual(contact['position_world_m'][2], 0.3995)

    def test_airborne_and_inactive_gap_contacts_are_not_drawn(self):
        self.data.qpos[2] = 1.0
        self.assertEqual(self.contacts(), [])
        self.model.geom_margin[:] = 0.02
        self.model.geom_gap[:] = 0.015
        self.data.qpos[2] = 0.15
        self.assertEqual(self.contacts(), [])
        self.assertGreater(self.data.ncon, 0)
        # Contacts within the combined 0.04 m margin are active even before touching.
        self.model.geom_gap[:] = 0.0
        self.data.qpos[2] = 0.11
        contacts = self.contacts()
        self.assertEqual(len(contacts), 2)
        self.assertTrue(all(c['distance_m'] > 0 for c in contacts))

    def test_nonwheel_robot_contacts_are_filtered_out(self):
        # Turn the base upside down so its own geom also collides with the floor.
        self.data.qpos[3:7] = [0, 1, 0, 0]
        contacts = self.contacts()
        self.assertGreater(self.data.ncon, len(contacts))
        self.assertEqual(len(contacts), 2)
        self.assertTrue(all('base_collision' not in c['geom_names'] for c in contacts))


class SplineTrajectoryTests(unittest.TestCase):
    def test_smoothing_reduces_noise_and_keeps_endpoints(self):
        times = np.linspace(60, 67, 71)
        truth = np.column_stack([times - 60, np.zeros(71), np.ones(71)])
        noisy = truth.copy()
        noisy[1:-1, 2] += np.random.default_rng(7).normal(0, 0.02, 69)
        render_times, fitted = renderer.smooth_trajectories(
            times, noisy[None], 'bspline', 0.03, 8,
        )
        np.testing.assert_array_equal(fitted[0, [0, -1]], noisy[[0, -1]])
        np.testing.assert_allclose(render_times[::8], times)
        self.assertLess(
            np.mean((fitted[0, ::8] - truth) ** 2),
            np.mean((noisy - truth) ** 2) * 0.2,
        )
        np.testing.assert_allclose(noisy[0], truth[0])

    def test_zero_tolerance_interpolates_irregular_times(self):
        times = np.array([1.0, 1.1, 1.7, 3.0, 3.2])
        positions = np.array([[[0, 0, 1], [1, 2, 1], [1, 3, 2],
                               [-1, 4, 1], [0, 0, 1]]], dtype=float)
        render_times, fitted = renderer.smooth_trajectories(
            times, positions, 'bspline', 0.0, 4,
        )
        np.testing.assert_allclose(fitted[:, ::4], positions, atol=1e-12)
        self.assertEqual(render_times[0], times[0])
        self.assertEqual(render_times[-1], times[-1])

    def test_short_and_stationary_intervals_remain_finite(self):
        for length in [1, 2, 3, 10]:
            with self.subTest(length=length):
                times = np.arange(length, dtype=float)
                positions = np.ones((1, length, 3))
                _, fitted = renderer.smooth_trajectories(
                    times, positions, 'bspline', 0.02, 8,
                )
                np.testing.assert_allclose(fitted, 1.0, atol=1e-12)
        times = np.array([0.0, 1.0])
        positions = np.array([[[0, 0, 0], [1, 2, 3]]], dtype=float)
        _, fitted = renderer.smooth_trajectories(times, positions, 'bspline', 0.02, 8)
        np.testing.assert_allclose(fitted[0], np.linspace(positions[0, 0], positions[0, 1], 9))

    def test_raw_mode_preserves_every_recorded_point(self):
        times = np.arange(4, dtype=float)
        positions = np.random.default_rng(4).normal(size=(3, 4, 3))
        render_times, fitted = renderer.smooth_trajectories(
            times, positions, 'raw', 0.2, 8,
        )
        np.testing.assert_array_equal(render_times, times)
        np.testing.assert_array_equal(fitted, positions)


if __name__ == "__main__":
    unittest.main()
