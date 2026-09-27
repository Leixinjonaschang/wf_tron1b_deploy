#!/usr/bin/env python3
"""Render recorded MuJoCo poses together, preserving their terrain coordinates."""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = REPO / 'pointfoot-mujoco-sim/robot-description/pointfoot/WF_TRON1B/xml/scene_rough_ground.xml'


def load_states(path, nq):
    """Read qpos in native MuJoCo order (root quaternion is wxyz)."""
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        expected = [f'qpos_{i}' for i in range(nq)]
        if [name for name in fields if name.startswith('qpos_')] != expected:
            raise ValueError(f'CSV qpos columns must match the scene: {expected}')
        if 'sim_time' not in fields:
            raise ValueError('CSV is missing sim_time')
        rows = list(reader)
    if not rows:
        raise ValueError('CSV contains no states')
    times = np.array([float(row['sim_time']) for row in rows])
    poses = np.array([[float(row[name]) for name in expected] for row in rows])
    if not np.isfinite(times).all() or not np.isfinite(poses).all():
        raise ValueError('CSV contains non-finite timestamps or qpos')
    if np.any(np.diff(times) <= 0):
        raise ValueError('sim_time must increase strictly; use one recording per CSV')
    if np.any(np.abs(np.linalg.norm(poses[:, 3:7], axis=1) - 1) > 0.01):
        raise ValueError('Root quaternions must be normalized MuJoCo wxyz values')
    return times, poses


def select_frames(times, start, end, count):
    """Select unique rows nearest evenly spaced simulation times."""
    start = times[0] if start is None else start
    end = times[-1] if end is None else end
    if not np.isfinite([start, end]).all() or start > end:
        raise ValueError('Start/end must be finite and start <= end')
    candidates = np.flatnonzero((times >= start) & (times <= end))
    if not len(candidates):
        raise ValueError('No recorded states in the requested time interval')
    targets = np.linspace(times[candidates[0]], times[candidates[-1]], count)
    indices = np.unique([candidates[np.argmin(abs(times[candidates] - t))] for t in targets])
    return indices


def configure_floor(model, mujoco, color):
    """Use a flat gray floor; grid lines are added as render-only geometry."""
    material_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "matplane")
    if material_id >= 0:
        model.mat_texid[material_id] = -1
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id < 0:
        return
    model.geom_rgba[floor_id, :3] = np.asarray(color, dtype=np.float64)
    model.geom_rgba[floor_id, 3] = 1.0


def add_floor_grid(mujoco, scene, low, high, spacing, width, color):
    """Add subtle render-only grid lines over the finite visible area."""
    if spacing <= 0 or width <= 0:
        return
    half_width = width / 2.0
    # The scene floor is at z=0; keep the grid just above it to avoid z-fighting.
    z = 0.003
    identity = np.eye(3, dtype=np.float64).reshape(-1)
    rgba = np.array([*color, 1.0], dtype=np.float32)
    x_values = np.arange(np.floor(low[0] / spacing) * spacing, high[0] + spacing, spacing)
    y_values = np.arange(np.floor(low[1] / spacing) * spacing, high[1] + spacing, spacing)
    for x in x_values:
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom, mujoco.mjtGeom.mjGEOM_BOX,
            np.array([half_width, (high[1] - low[1]) / 2, half_width], dtype=np.float64),
            np.array([x, (low[1] + high[1]) / 2, z], dtype=np.float64),
            identity, rgba,
        )
        scene.ngeom += 1
    for y in y_values:
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom, mujoco.mjtGeom.mjGEOM_BOX,
            np.array([(high[0] - low[0]) / 2, half_width, half_width], dtype=np.float64),
            np.array([(low[0] + high[0]) / 2, y, z], dtype=np.float64),
            identity, rgba,
        )
        scene.ngeom += 1


def render(args):
    # Must be configured before importing MuJoCo's OpenGL backend.
    if args.gl:
        os.environ['MUJOCO_GL'] = args.gl
    import mujoco
    import pygame

    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    configure_floor(model, mujoco, args.floor_color)
    if model.njnt == 0 or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError('Expected a WF_TRON1B scene with the root free joint first')
    times, poses = load_states(args.csv, model.nq)
    indices = select_frames(times, args.start, args.end, args.count)
    data = mujoco.MjData(model)

    # Hide robot collision proxies; retain every terrain geom, including group 0.
    robot = model.body_rootid[model.geom_bodyid] == model.jnt_bodyid[0]
    model.geom_rgba[robot & (model.geom_group != 1), 3] = 0
    option = mujoco.MjvOption()
    option.geomgroup[:] = 1
    option.sitegroup[:] = 0
    model.vis.global_.offwidth = max(args.width, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(args.height, model.vis.global_.offheight)

    # Fit the poses and finite terrain geoms; the infinite floor is excluded.
    points = [poses[indices, :3] - 0.8, poses[indices, :3] + 0.8]
    mujoco.mj_forward(model, data)
    terrain = (model.geom_bodyid == 0) & (model.geom_type != mujoco.mjtGeom.mjGEOM_PLANE)
    if terrain.any():
        radii = model.geom_rbound[terrain, None]
        points.extend([data.geom_xpos[terrain] - radii, data.geom_xpos[terrain] + radii])
    points = np.concatenate(points)
    low, high = points.min(axis=0), points.max(axis=0)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = args.lookat if args.lookat else (low + high) / 2
    fovy = np.deg2rad(model.vis.global_.fovy)
    half_angle = min(fovy / 2, np.arctan(np.tan(fovy / 2) * args.width / args.height))
    camera.distance = args.distance or max(2.0, np.linalg.norm(high - low) / (2 * np.sin(half_angle)))
    camera.azimuth = args.azimuth
    camera.elevation = args.elevation

    capacity = max(10000, model.ngeom * (len(indices) + 1) + 100)
    with mujoco.Renderer(model, height=args.height, width=args.width, max_geom=capacity) as renderer:
        # Establish lights/camera once. Rebuild geometry with the static scene once
        # and dynamic robot geometry once per pose. MuJoCo preserves mesh IDs.
        renderer.update_scene(data, camera=camera, scene_option=option)
        scene = renderer.scene
        scene.ngeom = 0
        perturb = mujoco.MjvPerturb()
        mujoco.mjv_addGeoms(model, data, option, perturb, mujoco.mjtCatBit.mjCAT_STATIC, scene)
        add_floor_grid(
            mujoco, scene, low, high, args.floor_grid_size,
            args.floor_grid_width, args.floor_grid_color,
        )
        for i, index in enumerate(indices):
            data.qpos[:] = poses[index]
            mujoco.mj_forward(model, data)
            first = scene.ngeom
            mujoco.mjv_addGeoms(model, data, option, perturb, mujoco.mjtCatBit.mjCAT_DYNAMIC, scene)
            alpha = 1.0 if i == len(indices) - 1 else args.ghost_alpha
            for g in range(first, scene.ngeom):
                scene.geoms[g].rgba[3] *= alpha
        scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = args.shadows
        pixels = renderer.render()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    surface = pygame.image.frombuffer(pixels.tobytes(), (args.width, args.height), 'RGB')
    pygame.image.save(surface, str(args.output))
    # Keep an exact record of which source poses appear in the image.
    metadata = {
        'source_csv': str(args.csv.resolve()), 'scene': str(args.scene.resolve()),
        'mujoco_version': mujoco.__version__, 'source_row_indices_zero_based': indices.tolist(),
        'sim_times': times[indices].tolist(), 'qpos': poses[indices].tolist(),
        'camera': {'lookat': camera.lookat.tolist(), 'distance': camera.distance,
                   'azimuth': camera.azimuth, 'elevation': camera.elevation},
        'ghost_alpha': args.ghost_alpha, 'shadows': args.shadows,
        'floor_color': list(args.floor_color), 'floor_grid_color': list(args.floor_grid_color),
        'floor_grid_size': args.floor_grid_size, 'floor_grid_width': args.floor_grid_width,
        'width': args.width, 'height': args.height,
    }
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Rendered {len(indices)} poses: {np.round(times[indices], 3).tolist()} s')
    print(f'Image: {args.output.resolve()}')
    print(f'Pose/camera record: {args.output.with_suffix(".json").resolve()}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, required=True, help='Simulator state CSV')
    parser.add_argument('--scene', type=Path, default=DEFAULT_SCENE, help='Original recording terrain XML')
    parser.add_argument('--output', type=Path, default=Path('logs/sim2sim/trajectory.png'))
    parser.add_argument('--start', type=float, help='First simulation time in seconds (inclusive)')
    parser.add_argument('--end', type=float, help='Last simulation time in seconds (inclusive)')
    parser.add_argument('--count', type=int, default=8, help='Number of evenly spaced poses')
    parser.add_argument('--ghost-alpha', type=float, default=0.65, help='Earlier pose opacity; final pose is opaque')
    parser.add_argument('--azimuth', type=float, default=90)
    parser.add_argument('--elevation', type=float, default=-25)
    parser.add_argument('--distance', type=float, help='Camera distance in meters; default auto fit')
    parser.add_argument('--lookat', type=float, nargs=3, metavar=('X', 'Y', 'Z'))
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--shadows', action='store_true')
    parser.add_argument('--floor-color', type=float, nargs=3, default=(0.55, 0.57, 0.60), metavar=('R', 'G', 'B'), help='Gray floor RGB color in [0, 1]')
    parser.add_argument('--floor-grid-color', type=float, nargs=3, default=(0.36, 0.38, 0.41), metavar=('R', 'G', 'B'), help='Subtle grid RGB color in [0, 1]')
    parser.add_argument('--floor-grid-size', type=float, default=1.0, help='Grid spacing in meters')
    parser.add_argument('--floor-grid-width', type=float, default=0.012, help='Grid line width in meters')
    parser.add_argument('--gl', choices=['egl', 'osmesa', 'glfw'], help='OpenGL backend (egl for headless GPU)')
    args = parser.parse_args()
    if args.count < 1 or min(args.width, args.height) < 1:
        parser.error('count, width and height must be positive')
    if not 0 <= args.ghost_alpha <= 1:
        parser.error('ghost-alpha must be between 0 and 1')
    values = [args.azimuth, args.elevation, *args.floor_color, *args.floor_grid_color, *(args.lookat or [])]
    if not np.isfinite(values).all() or (args.distance is not None and (not np.isfinite(args.distance) or args.distance <= 0)):
        parser.error('Camera values must be finite and distance must be positive')
    if args.output.suffix.lower() != '.png':
        parser.error('output must have a .png extension')
    colors = np.concatenate([args.floor_color, args.floor_grid_color])
    if not np.isfinite(colors).all() or not (0 <= colors).all() or not (colors <= 1).all():
        parser.error('floor and grid colors must be in [0, 1]')
    if not np.isfinite([args.floor_grid_size, args.floor_grid_width]).all() or args.floor_grid_size <= 0 or args.floor_grid_width <= 0:
        parser.error('floor grid size and width must be positive')
    render(args)


if __name__ == '__main__':
    main()
