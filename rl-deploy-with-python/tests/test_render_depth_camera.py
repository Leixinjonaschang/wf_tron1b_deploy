"""Check camera orientation, aspect ratio and terrain clipping without OpenGL."""

import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np


SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts'
with patch.object(sys, 'path', [str(SCRIPTS), *sys.path]):
    import render_depth_camera as renderer


CAMERA_XML = """
<mujoco>
  <worldbody>
    <geom type="plane" size="5 5 0.1"/>
    <body pos="2 3 1">
      <freejoint/>
      <geom type="sphere" size="0.1" rgba="1 1 1 0"/>
      <camera name="depth" fovy="90"/>
    </body>
  </worldbody>
</mujoco>
"""


class DepthCameraTests(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string(CAMERA_XML)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

    def wireframe(self, clip=True):
        return renderer.build_fov_wireframe(
            mujoco, self.model, self.data, 0, 200, 100,
            length=2.0, clip=clip, radius=0.005, samples_per_edge=4,
        )

    def test_image_axes_and_aspect_ratio(self):
        rays = renderer.camera_rays(90, 200, 100, [[-1, 1], [1, -1], [0, 0]])
        np.testing.assert_allclose(rays, [[-2, 1, -1], [2, -1, -1], [0, 0, -1]])

    def test_wireframe_follows_camera_world_translation_and_yaw(self):
        self.data.qpos[3:7] = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
        mujoco.mj_forward(self.model, self.data)
        fov = self.wireframe(clip=False)
        np.testing.assert_allclose(fov['origin'], [2, 3, 1])
        # Top-left at axial depth 2: local (-4, 2, -2), then world yaw +90°.
        np.testing.assert_allclose(fov['points'][0], [0, -1, -1], atol=1e-12)
        self.assertTrue(np.all(fov['geom_ids'] == -1))

    def test_plane_clipping_uses_depth_not_euclidean_ray_length(self):
        fov = self.wireframe()
        np.testing.assert_allclose(fov['points'][::4],
                                   [[0, 4, 0], [4, 4, 0], [4, 2, 0], [0, 2, 0]],
                                   atol=1e-12)
        self.assertTrue(np.all(fov['geom_ids'] == 0))
        # Display offsets must lift the wire just above the surface.
        self.assertTrue(np.all(fov['segments'][:, :, 2] > 0))

    def test_clipping_stops_at_raised_terrain_instead_of_floor(self):
        xml = CAMERA_XML.replace('<body pos=',
                                 '<geom type="box" pos="2 3 0.25" size="3 3 0.25"/>'
                                 '<body pos=')
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        fov = self.wireframe()
        np.testing.assert_allclose(fov['points'][:, 2], 0.5)
        self.assertTrue(np.all(fov['geom_ids'] == 1))

    def test_render_decorations_do_not_change_physics_or_cast_shadows(self):
        fov = self.wireframe()
        scene = mujoco.MjvScene(self.model, maxgeom=100)
        qpos = self.data.qpos.copy()
        ngeom = self.model.ngeom
        renderer.add_fov_wireframe(mujoco, scene, fov['segments'], 0.005, [1, 0, 0], 0.7)
        self.assertGreater(scene.ngeom, 0)
        self.assertTrue(all(g.category == mujoco.mjtCatBit.mjCAT_DECOR
                            for g in scene.geoms[:scene.ngeom]))
        self.assertEqual(self.model.ngeom, ngeom)
        np.testing.assert_array_equal(self.data.qpos, qpos)

    def test_nearest_timestamp_and_out_of_range(self):
        times = np.array([60.001, 60.101, 60.201])
        self.assertEqual(renderer.nearest_state_index(times, 60.1), 1)
        self.assertEqual(renderer.nearest_state_index(times, 60.001), 0)
        for time in (59, 61, np.nan, np.inf):
            with self.assertRaises(ValueError):
                renderer.nearest_state_index(times, time)

    def test_grayscale_clipping_preserves_raw_depth(self):
        depth = np.array([[0.1, 0.2, 1.1, 2, 3]], dtype=np.float32)
        original = depth.copy()
        preview = renderer.depth_preview(depth, 0.2, 2)
        np.testing.assert_array_equal(preview[0, :, 0], [0, 0, 128, 255, 255])
        np.testing.assert_array_equal(depth, original)
        self.assertEqual(preview.dtype, np.uint8)


class TerrainScanTests(unittest.TestCase):
    def scene(self, xml=CAMERA_XML):
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return model, data

    def scan(self, model, data, columns=4, rows=2, max_depth=2):
        return renderer.terrain_scan_points(
            mujoco, model, data, 0, 200, 100, columns, rows, max_depth,
        )

    def test_image_plane_is_half_meter_along_rotated_optical_axis(self):
        model, data = self.scene()
        data.qpos[3:7] = [np.cos(0.3), 0, np.sin(0.3), 0]
        mujoco.mj_forward(model, data)
        fov = renderer.build_fov_wireframe(
            mujoco, model, data, 0, 200, 100, 0.5, False, 0.005,
        )
        normal = -data.cam_xmat[0].reshape(3, 3)[:, 2]
        np.testing.assert_allclose((fov['points'] - fov['origin']) @ normal, 0.5)
        np.testing.assert_allclose(fov['points'].mean(axis=0), fov['origin'] + 0.5 * normal)
        # Four planar edges and four corner rays, no projected ground boundary.
        self.assertEqual(fov['segments'].shape, (8, 2, 3))

    def test_regular_image_grid_intersects_ground_beyond_display_plane(self):
        model, data = self.scene()
        scan = self.scan(model, data)
        expected = [[x, y, 0] for y in (3.5, 2.5) for x in (0.5, 1.5, 2.5, 3.5)]
        np.testing.assert_allclose(scan['points'], expected, atol=1e-12)
        np.testing.assert_allclose(scan['normals'], np.tile([0, 0, 1], (8, 1)))

    def test_first_hit_on_robot_excludes_hidden_ground(self):
        xml = CAMERA_XML.replace(
            '<geom type="sphere" size="0.1" rgba="1 1 1 0"/>',
            '<geom type="box" pos="0 0 -0.4" size="0.1 0.1 0.1"/>',
        )
        model, data = self.scene(xml)
        scan = self.scan(model, data, columns=3, rows=3)
        self.assertEqual(len(scan['points']), 8)
        self.assertFalse(np.any(np.all(scan['image_xy'] == 0, axis=1)))
        np.testing.assert_allclose(scan['points'][:, 2], 0, atol=1e-12)

    def test_static_child_terrain_occludes_floor(self):
        xml = CAMERA_XML.replace(
            '<body pos="2 3 1">',
            '<body pos="2 3 0.25"><geom type="box" size="3 3 0.25"/></body>'
            '<body pos="2 3 1">',
        )
        model, data = self.scene(xml)
        scan = self.scan(model, data)
        self.assertEqual(len(scan['points']), 8)
        np.testing.assert_allclose(scan['points'][:, 2], 0.5)
        self.assertTrue(np.all(scan['geom_ids'] == 1))

    def test_scan_range_and_missed_rays_do_not_create_markers(self):
        model, data = self.scene()
        self.assertEqual(self.scan(model, data, max_depth=0.8)['points'].shape, (0, 3))
        data.qpos[3:7] = [0, 1, 0, 0]  # Camera points into the sky.
        mujoco.mj_forward(model, data)
        self.assertEqual(self.scan(model, data)['points'].shape, (0, 3))

    def test_scan_markers_are_decorations_above_the_surface(self):
        model, data = self.scene()
        scan = self.scan(model, data)
        scene = mujoco.MjvScene(model, maxgeom=100)
        renderer.add_scan_dots(mujoco, scene, scan, 0.01, [1, 1, 0], 0.8)
        self.assertEqual(scene.ngeom, len(scan['points']))
        for geom in scene.geoms[:scene.ngeom]:
            self.assertEqual(geom.category, mujoco.mjtCatBit.mjCAT_DECOR)
            self.assertAlmostEqual(geom.pos[2], 0.005)


class DepthImagePlaneTests(unittest.TestCase):
    def test_texture_assets_preserve_physical_scene(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scene.xml'
            path.write_text(CAMERA_XML)
            original = mujoco.MjModel.from_xml_path(str(path))
            model, texture_id, material_id = renderer.load_scene_with_depth_texture(
                mujoco, path, 12, 8,
            )
        self.assertEqual(model.ngeom, original.ngeom)
        self.assertEqual(model.nq, original.nq)
        np.testing.assert_array_equal(model.qpos0, original.qpos0)
        np.testing.assert_array_equal(model.geom_pos, original.geom_pos)
        self.assertEqual(model.mat_texid[material_id, mujoco.mjtTextureRole.mjTEXROLE_RGB],
                         texture_id)
        image = np.arange(8 * 12 * 3, dtype=np.uint8).reshape(8, 12, 3)
        renderer.set_depth_texture(model, texture_id, image)
        start = model.tex_adr[texture_id]
        np.testing.assert_array_equal(model.tex_data[start:start + image.size], image.ravel())

    def test_panel_corners_match_the_rotated_fov_image_rectangle(self):
        model = mujoco.MjModel.from_xml_string(CAMERA_XML)
        data = mujoco.MjData(model)
        data.qpos[3:7] = [np.cos(0.3), 0, np.sin(0.3), 0]
        mujoco.mj_forward(model, data)
        fov = renderer.build_fov_wireframe(
            mujoco, model, data, 0, 200, 100, 0.5, False, 0.0075,
        )
        scene = mujoco.MjvScene(model, maxgeom=10)
        renderer.add_depth_image_plane(mujoco, scene, fov, 0.5, 200, 100, 0, 1)
        geom = scene.geoms[0]
        local_corners = np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]])
        world_corners = (local_corners * geom.size) @ geom.mat.reshape(3, 3).T + geom.pos
        np.testing.assert_allclose(world_corners, fov['points'], atol=1e-6)
        self.assertEqual(geom.category, mujoco.mjtCatBit.mjCAT_DECOR)
        self.assertEqual(geom.rgba[3], 1)
        self.assertEqual(model.ngeom, 2)


if __name__ == '__main__':
    unittest.main()
