#!/usr/bin/env python3
"""Render recorded MuJoCo poses together, preserving their terrain coordinates."""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO / 'data/state_run01.csv'
DEFAULT_SCENE = REPO / 'pointfoot-mujoco-sim/robot-description/pointfoot/WF_TRON1B/xml/scene_rough_ground.xml'
WHEEL_BODY_NAMES = ('wheel_L_Link', 'wheel_R_Link')
BASE_BODY_NAME = 'base_Link'


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


def interval_frames(times, start, end):
    """Return every recorded row within the inclusive simulation-time interval."""
    start = times[0] if start is None else start
    end = times[-1] if end is None else end
    if not np.isfinite([start, end]).all() or start > end:
        raise ValueError('Start/end must be finite and start <= end')
    candidates = np.flatnonzero((times >= start) & (times <= end))
    if not len(candidates):
        raise ValueError('No recorded states in the requested time interval')
    return candidates


def select_frames(times, start, end, count):
    """Select unique rows nearest evenly spaced simulation times."""
    candidates = interval_frames(times, start, end)
    targets = np.linspace(times[candidates[0]], times[candidates[-1]], count)
    indices = np.unique([candidates[np.argmin(abs(times[candidates] - t))] for t in targets])
    return indices


def body_trajectories(mujoco, model, data, poses, indices, body_names,
                      *, center_of_mass=False):
    """Recover body origins or inertial centers of mass in world coordinates."""
    body_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in body_names
    ]
    missing = [name for name, body_id in zip(body_names, body_ids) if body_id < 0]
    if missing:
        raise ValueError(f'Trajectory bodies not found: {", ".join(missing)}')
    positions = np.empty((len(body_ids), len(indices), 3), dtype=np.float64)
    for sample, index in enumerate(indices):
        data.qpos[:] = poses[index]
        mujoco.mj_forward(model, data)
        world_positions = data.xipos if center_of_mass else data.xpos
        positions[:, sample] = world_positions[body_ids]
    return positions


def smooth_trajectories(times, trajectories, method, smoothing_m, samples_per_segment):
    """Fit time-parametric 3D B-splines, keeping the measured interval endpoints."""
    if method == 'raw' or len(times) == 1:
        return times.copy(), trajectories.copy()

    from scipy.interpolate import make_splprep

    # Include each recorded timestamp, even when the recording is irregular.
    render_times = np.concatenate([
        *[np.linspace(start, end, samples_per_segment, endpoint=False)
          for start, end in zip(times[:-1], times[1:])],
        times[-1:],
    ])
    duration = times[-1] - times[0]
    u = (times - times[0]) / duration
    render_u = (render_times - times[0]) / duration
    weights = np.ones(len(times))
    weights[[0, -1]] = 1.0e6
    smoothed = []
    for positions in trajectories:
        spline, _ = make_splprep(
            positions.T, u=u, w=weights if smoothing_m > 0 else None,
            k=min(3, len(times) - 1),
            s=len(times) * smoothing_m ** 2,
        )
        points = spline(render_u).T
        # Endpoint weights constrain the fit; remove floating-point drift exactly.
        points[[0, -1]] = positions[[0, -1]]
        smoothed.append(points)
    return render_times, np.asarray(smoothed)


def add_trajectory_trails(mujoco, scene, trajectories, radius, colors, alpha):
    """Draw world-space polylines as render-only capsules, including single points."""
    identity = np.eye(3, dtype=np.float64).reshape(-1)
    for positions, color in zip(trajectories, colors):
        rgba = np.array([*color, alpha], dtype=np.float32)
        # Retain a point for a one-frame interval or a stationary body.
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom, mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([radius, 0.0, 0.0]), positions[0], identity, rgba,
        )
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        scene.ngeom += 1
        for start, end in zip(positions[:-1], positions[1:]):
            if np.linalg.norm(end - start) <= 1.0e-9:
                continue
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
                np.zeros(3), np.zeros(3), identity, rgba,
            )
            mujoco.mjv_connector(
                geom, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, start, end,
            )
            geom.category = mujoco.mjtCatBit.mjCAT_DECOR
            scene.ngeom += 1


def configure_floor(model, mujoco, color):
    """Use a plain, non-emissive floor that receives lighting and shadows."""
    material_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "matplane")
    if material_id >= 0:
        model.mat_texid[material_id] = -1
        model.mat_specular[material_id] = 0
        model.mat_reflectance[material_id] = 0
        model.mat_emission[material_id] = 0.0
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id < 0:
        return
    model.geom_rgba[floor_id, :3] = np.asarray(color, dtype=np.float64)
    model.geom_rgba[floor_id, 3] = 1.0


def add_floor_grid(mujoco, scene, low, high, spacing, width, color):
    """Add thin render-only grid lines just above the white floor."""
    center = (low + high) / 2
    half_extent = (high - low) / 2
    identity = np.eye(3, dtype=np.float64).reshape(-1)
    rgba = np.array([*color, 1.0], dtype=np.float32)
    for axis in (0, 1):
        coordinates = np.arange(low[axis], high[axis] + spacing * 0.5, spacing)
        for coordinate in coordinates:
            size = np.array([*half_extent, 0.0005], dtype=np.float64)
            size[axis] = width / 2
            position = np.array([*center, 0.003], dtype=np.float64)
            position[axis] = coordinate
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom, mujoco.mjtGeom.mjGEOM_BOX, size, position, identity, rgba,
            )
            geom.category = mujoco.mjtCatBit.mjCAT_DECOR
            scene.ngeom += 1


def wheel_terrain_contacts(mujoco, model, data):
    """Read active wheel/environment contacts after mj_forward on an original pose."""
    wheel_bodies = {}
    for name in WHEEL_BODY_NAMES:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f'Contact wheel body not found: {name}')
        wheel_bodies[body_id] = name
    contacts = []
    for contact in data.contact:
        # Include contacts within the model's margin, but exclude inactive gaps.
        if contact.efc_address < 0:
            continue
        geom_ids = [int(geom_id) for geom_id in contact.geom]
        if min(geom_ids) < 0:
            continue
        body_ids = model.geom_bodyid[geom_ids]
        for side in (0, 1):
            wheel_body, other_body = body_ids[side], body_ids[1 - side]
            if wheel_body not in wheel_bodies:
                continue
            if model.body_rootid[wheel_body] == model.body_rootid[other_body]:
                continue
            contacts.append({
                'wheel_body': wheel_bodies[wheel_body],
                'geom_ids': geom_ids,
                'geom_names': [
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                    for geom_id in geom_ids
                ],
                'position_world_m': contact.pos.tolist(),
                'distance_m': float(contact.dist),
                'inclusion_margin_m': float(contact.includemargin),
            })
            break
    return contacts


def add_contact_points(mujoco, scene, frames, radius, color, alpha):
    """Draw contact markers at their reconstructed world positions."""
    rgba = np.array([*color, alpha], dtype=np.float32)
    identity = np.eye(3, dtype=np.float64).reshape(-1)
    for frame in frames:
        for contact in frame['contacts']:
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([radius, 0.0, 0.0]),
                np.asarray(contact['position_world_m']), identity, rgba,
            )
            geom.category = mujoco.mjtCatBit.mjCAT_DECOR
            scene.ngeom += 1


def add_top_light(mujoco, scene, position, intensity, shadows=True):
    """Add a white, diffuse-only overhead light to the offline render."""
    if scene.nlight >= len(scene.lights):
        raise ValueError('No free scene light slot for the top light')
    light = scene.lights[scene.nlight]
    light.pos[:] = position
    light.dir[:] = (0.0, 0.0, -1.0)
    light.ambient[:] = 0.0
    light.diffuse[:] = intensity
    light.specular[:] = 0.0
    light.headlight = False
    light.castshadow = shadows
    # MuJoCo 3.10 replaced the directional boolean with a light-type enum.
    if hasattr(light, 'type'):
        light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        light.texid = -1
    else:
        light.directional = True
    scene.nlight += 1


def render(args):
    # Must be configured before importing MuJoCo's OpenGL backend.
    os.environ.setdefault('MESA_SHADER_CACHE_DISABLE', 'true')
    if args.gl:
        os.environ['MUJOCO_GL'] = args.gl
    import mujoco
    import pygame

    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    configure_floor(model, mujoco, args.floor_color)
    model.light_castshadow[:] = args.shadows
    if model.njnt == 0 or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError('Expected a WF_TRON1B scene with the root free joint first')
    times, poses = load_states(args.csv, model.nq)
    indices = select_frames(times, args.start, args.end, args.count)
    data = mujoco.MjData(model)
    trail_indices = (
        interval_frames(times, args.start, args.end)
        if args.wheel_trails or args.base_trail else []
    )
    trails = {}
    trail_settings = (
        ('wheel_trails', args.wheel_trails, WHEEL_BODY_NAMES, False,
         args.wheel_trail_radius, (args.wheel_left_color, args.wheel_right_color)),
        ('base_trail', args.base_trail, (BASE_BODY_NAME,), True,
         args.base_trail_radius, (args.base_trail_color,)),
    )
    for key, enabled, names, use_com, radius, colors in trail_settings:
        if not enabled:
            continue
        raw_positions = body_trajectories(
            mujoco, model, data, poses, trail_indices, names,
            center_of_mass=use_com,
        )
        render_times, render_positions = smooth_trajectories(
            times[trail_indices], raw_positions, args.traj_method,
            args.traj_smoothing, args.traj_samples_per_segment,
        )
        trails[key] = {
            'names': names, 'radius': radius, 'colors': colors,
            'raw_positions': raw_positions, 'render_positions': render_positions,
            'render_times': render_times,
            'point': 'body center of mass' if use_com else 'wheel body origin',
        }

    contact_frames = []
    if args.contact_points:
        for index in indices:
            data.qpos[:] = poses[index]
            mujoco.mj_forward(model, data)
            contact_frames.append({
                'source_row_index_zero_based': int(index),
                'sim_time': float(times[index]),
                'contacts': wheel_terrain_contacts(mujoco, model, data),
            })
    contact_positions = np.array([
        contact['position_world_m']
        for frame in contact_frames for contact in frame['contacts']
    ]).reshape(-1, 3)

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
    if len(contact_positions):
        points.extend([
            contact_positions - args.contact_point_radius,
            contact_positions + args.contact_point_radius,
        ])
    for trail in trails.values():
        trail_points = trail['render_positions'].reshape(-1, 3)
        points.extend([
            trail_points - trail['radius'],
            trail_points + trail['radius'],
        ])
    mujoco.mj_forward(model, data)
    terrain = (model.geom_bodyid == 0) & (model.geom_type != mujoco.mjtGeom.mjGEOM_PLANE)
    if terrain.any():
        model.geom_rgba[terrain, :3] = args.terrain_color
        radii = model.geom_rbound[terrain, None]
        points.extend([data.geom_xpos[terrain] - radii, data.geom_xpos[terrain] + radii])
    points = np.concatenate(points)
    low, high = points.min(axis=0), points.max(axis=0)
    shadow_center = (low + high) / 2
    shadow_radius = max(1.0, np.linalg.norm(high - low) / 2 + 0.5)
    if args.shadows:
        # Fit shadow maps to every displayed pose, not just the XML's initial pose.
        model.vis.map.shadowclip = shadow_radius / model.stat.extent
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = args.lookat if args.lookat else (low + high) / 2
    fovy = np.deg2rad(model.vis.global_.fovy)
    half_angle = min(fovy / 2, np.arctan(np.tan(fovy / 2) * args.width / args.height))
    camera.distance = args.distance or max(2.0, np.linalg.norm(high - low) / (2 * np.sin(half_angle)))
    camera.azimuth = args.azimuth
    camera.elevation = args.elevation

    # Extend the grid beyond the robot poses without changing the camera fit.
    grid_low = np.minimum(low[:2], camera.lookat[:2] - 2 * camera.distance)
    grid_high = np.maximum(high[:2], camera.lookat[:2] + 2 * camera.distance)
    grid_low = np.floor(grid_low / args.floor_grid_size) * args.floor_grid_size
    grid_high = np.ceil(grid_high / args.floor_grid_size) * args.floor_grid_size
    grid_capacity = (
        int(np.ceil((grid_high - grid_low) / args.floor_grid_size).sum()) + 4
        if args.floor_grid else 0
    )
    trail_capacity = sum(
        len(trail['names']) * len(trail['render_times'])
        for trail in trails.values()
    )
    capacity = max(
        10000, model.ngeom * (len(indices) + 1) + trail_capacity
        + grid_capacity + len(contact_positions) + 100,
    )
    with mujoco.Renderer(model, height=args.height, width=args.width, max_geom=capacity) as renderer:
        # Establish lights/camera once. Rebuild geometry with the static scene once
        # and dynamic robot geometry once per pose. MuJoCo preserves mesh IDs.
        renderer.update_scene(data, camera=camera, scene_option=option)
        scene = renderer.scene
        if args.top_light:
            add_top_light(
                mujoco, scene, (camera.lookat[0], camera.lookat[1], high[2] + 5.0),
                args.top_light_intensity, shadows=args.shadows,
            )
        if args.shadows:
            for light in scene.lights[:scene.nlight]:
                directional = (
                    light.type == mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
                    if hasattr(light, 'type') else light.directional
                )
                if light.castshadow and directional:
                    light.pos[:] = shadow_center - light.dir * (shadow_radius + 1.0)
        scene.ngeom = 0
        perturb = mujoco.MjvPerturb()
        mujoco.mjv_addGeoms(model, data, option, perturb, mujoco.mjtCatBit.mjCAT_STATIC, scene)
        if args.floor_grid:
            add_floor_grid(
                mujoco, scene, grid_low, grid_high, args.floor_grid_size,
                args.floor_grid_width, args.floor_grid_color,
            )
        for trail in trails.values():
            add_trajectory_trails(
                mujoco, scene, trail['render_positions'],
                trail['radius'], trail['colors'], args.traj_alpha,
            )
        for i, index in enumerate(indices):
            data.qpos[:] = poses[index]
            mujoco.mj_forward(model, data)
            first = scene.ngeom
            mujoco.mjv_addGeoms(model, data, option, perturb, mujoco.mjtCatBit.mjCAT_DYNAMIC, scene)
            alpha = 1.0 if i == len(indices) - 1 else args.ghost_alpha
            for g in range(first, scene.ngeom):
                geom = scene.geoms[g]
                geom.rgba[3] *= alpha
                if (args.robot_matte and geom.objtype == mujoco.mjtObj.mjOBJ_GEOM
                        and robot[geom.objid]):
                    # Visual meshes have no named material; set their render properties.
                    geom.specular = 0.0
                    geom.shininess = 0.0
                    geom.reflectance = 0.0
        if args.contact_points:
            add_contact_points(
                mujoco, scene, contact_frames,
                args.contact_point_radius, args.contact_point_color,
                args.contact_point_alpha,
            )
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
        'shadow_map': {
            'center_world_m': shadow_center.tolist(),
            'half_extent_m': shadow_radius,
        } if args.shadows else None,
        'top_light': {
            'enabled': args.top_light, 'intensity': args.top_light_intensity,
            'direction_world': [0.0, 0.0, -1.0],
            'diffuse': [args.top_light_intensity] * 3,
            'specular': [0.0, 0.0, 0.0],
            'castshadow': args.top_light and args.shadows,
        },
        'terrain_color': list(args.terrain_color), 'robot_matte': args.robot_matte,
        'floor_color': list(args.floor_color), 'floor_grid': args.floor_grid,
        'floor_emission': 0.0,
        'floor_grid_color': list(args.floor_grid_color),
        'floor_grid_size': args.floor_grid_size,
        'floor_grid_width': args.floor_grid_width,
        'width': args.width, 'height': args.height,
        'wheel_trails': {'enabled': args.wheel_trails},
        'base_trail': {'enabled': args.base_trail},
        'contact_points': {'enabled': args.contact_points},
        'trajectory_smoothing': {
            'method': args.traj_method, 'parameter': 'simulation_time',
            'smoothing_m': args.traj_smoothing,
            'samples_per_segment': args.traj_samples_per_segment,
            'endpoint_policy': 'fixed',
        },
    }
    if args.contact_points:
        metadata['contact_points'].update({
            'source': 'reconstructed from original qpos and scene via mj_forward',
            'scope': 'selected_poses',
            'filter': 'active wheel/environment constraints, including contact margin',
            'radius_m': args.contact_point_radius,
            'color': list(args.contact_point_color),
            'alpha': args.contact_point_alpha,
            'count': len(contact_positions),
            'frames': contact_frames,
        })
    for key, trail in trails.items():
        metadata[key].update({
            'point': trail['point'],
            'radius_m': trail['radius'],
            'alpha': args.traj_alpha,
            'source_row_indices_zero_based': trail_indices.tolist(),
            'sim_times': times[trail_indices].tolist(),
            'render_sim_times': trail['render_times'].tolist(),
            'bodies': {
                name: {
                    'color': list(color),
                    'positions_world_m': raw.tolist(),
                    'render_positions_world_m': smoothed.tolist(),
                }
                for name, color, raw, smoothed in zip(
                    trail['names'], trail['colors'], trail['raw_positions'],
                    trail['render_positions'],
                )
            },
        })
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Rendered {len(indices)} poses: {np.round(times[indices], 3).tolist()} s')
    if args.contact_points:
        print(f'Reconstructed {len(contact_positions)} wheel/environment contact points '
              f'across {len(indices)} poses')
    for key, trail in trails.items():
        print(f'{key}: {len(trail_indices)} recorded samples -> '
              f'{len(trail["render_times"])} rendered points per body '
              f'({args.traj_method})')
    print(f'Image: {args.output.resolve()}')
    print(f'Pose/camera record: {args.output.with_suffix(".json").resolve()}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=DEFAULT_CSV,
                        help='Simulator state CSV (default: data/state_run01.csv relative to repository root)')
    parser.add_argument('--scene', type=Path, default=DEFAULT_SCENE, help='Original recording terrain XML')
    parser.add_argument('--output', type=Path,
                        default=Path('logs/sim2sim/flat_rough_flat_smooth_traj.png'))
    parser.add_argument('--start', type=float, default=60.0, help='First simulation time in seconds (inclusive)')
    parser.add_argument('--end', type=float, default=67.0, help='Last simulation time in seconds (inclusive)')
    parser.add_argument('--count', type=int, default=9, help='Number of evenly spaced poses')
    parser.add_argument('--ghost-alpha', type=float, default=1.0, help='Earlier pose opacity; final pose is opaque')
    parser.add_argument('--robot-matte', action=argparse.BooleanOptionalAction,
                        default=True, help='Disable specular highlights and reflections on the robot')
    parser.add_argument('--wheel-trails', action=argparse.BooleanOptionalAction, default=True,
                        help='Show both wheel-center trajectories using every recorded frame in the interval (default: enabled)')
    parser.add_argument('--wheel-trail-radius', type=float, default=0.015, help='Wheel trajectory tube radius in meters')
    parser.add_argument('--wheel-left-color', type=float, nargs=3, default=(0.95, 0.25, 0.10), metavar=('R', 'G', 'B'))
    parser.add_argument('--wheel-right-color', type=float, nargs=3, default=(0.10, 0.55, 1.00), metavar=('R', 'G', 'B'))
    parser.add_argument('--base-trail', action=argparse.BooleanOptionalAction,
                        default=True, help='Show the base_Link center-of-mass trajectory')
    parser.add_argument('--base-trail-radius', type=float, default=0.015,
                        help='Base center-of-mass trajectory tube radius in meters')
    parser.add_argument('--base-trail-color', type=float, nargs=3,
                        default=(0.10, 0.65, 0.25), metavar=('R', 'G', 'B'))
    parser.add_argument('--contact-points', action=argparse.BooleanOptionalAction,
                        default=True, help='Show reconstructed wheel/terrain contacts at the displayed poses')
    parser.add_argument('--contact-point-radius', type=float, default=0.05,
                        help='Contact marker radius in meters')
    parser.add_argument('--contact-point-color', type=float, nargs=3,
                        default=(1.0, 0.94, 0.60), metavar=('R', 'G', 'B'))
    parser.add_argument('--contact-point-alpha', type=float, default=0.65,
                        help='Contact marker opacity in [0, 1]')
    parser.add_argument('--traj-alpha', type=float, default=0.3,
                        help='Opacity of all wheel and base trajectories in [0, 1]')
    parser.add_argument('--traj-method', choices=['bspline', 'raw'], default='bspline',
                        help='Time-parametric B-spline smoothing or raw recorded polylines')
    parser.add_argument('--traj-smoothing', type=float, default=0.015,
                        help='B-spline RMS fitting tolerance in meters; larger is smoother; 0 interpolates the samples')
    parser.add_argument('--traj-samples-per-segment', type=int, default=8,
                        help='Number of spline segments per recorded time interval')
    parser.add_argument('--azimuth', type=float, default=270)
    parser.add_argument('--elevation', type=float, default=-17)
    parser.add_argument('--distance', type=float, default=4.5, help='Camera distance in meters')
    parser.add_argument('--lookat', type=float, nargs=3, default=(5.8, 0.9, 0.7),
                        metavar=('X', 'Y', 'Z'))
    parser.add_argument('--width', type=int, default=3840)
    parser.add_argument('--height', type=int, default=2160)
    parser.add_argument('--shadows', action=argparse.BooleanOptionalAction,
                        default=True, help='Render shadows from scene lights and the top light')
    parser.add_argument('--top-light', action=argparse.BooleanOptionalAction,
                        default=False, help='Add a diffuse-only white overhead light')
    parser.add_argument('--top-light-intensity', type=float, default=0.0,
                        help='Additional overhead diffuse light strength (default: 0.0)')
    parser.add_argument('--terrain-color', type=float, nargs=3,
                        default=(0.30, 0.58, 0.32), metavar=('R', 'G', 'B'),
                        help='Terrain obstacle RGB color in [0, 1] (default: green)')
    parser.add_argument('--floor-color', type=float, nargs=3, default=(1.0, 1.0, 1.0),
                        metavar=('R', 'G', 'B'), help='Plain floor RGB color in [0, 1] (default: white)')
    parser.add_argument('--floor-grid', action=argparse.BooleanOptionalAction,
                        default=False, help='Show gray grid lines on the floor')
    parser.add_argument('--floor-grid-color', type=float, nargs=3,
                        default=(0.70, 0.70, 0.70), metavar=('R', 'G', 'B'))
    parser.add_argument('--floor-grid-size', type=float, default=1.0,
                        help='Floor grid spacing in meters')
    parser.add_argument('--floor-grid-width', type=float, default=0.006,
                        help='Floor grid line width in meters')
    parser.add_argument('--gl', choices=['egl', 'osmesa', 'glfw'], default='egl',
                        help='OpenGL backend (default: egl for headless rendering)')
    args = parser.parse_args()
    if args.count < 1 or min(args.width, args.height) < 1:
        parser.error('count, width and height must be positive')
    alphas = np.array([args.ghost_alpha, args.traj_alpha, args.contact_point_alpha])
    if not np.isfinite(alphas).all() or (alphas < 0).any() or (alphas > 1).any():
        parser.error('ghost-alpha, traj-alpha and contact-point-alpha must be in [0, 1]')
    if not np.isfinite(args.top_light_intensity) or args.top_light_intensity < 0:
        parser.error('top-light-intensity must be finite and nonnegative')
    values = [args.azimuth, args.elevation, *(args.lookat or [])]
    if not np.isfinite(values).all() or (args.distance is not None and (not np.isfinite(args.distance) or args.distance <= 0)):
        parser.error('Camera values must be finite and distance must be positive')
    if args.output.suffix.lower() != '.png':
        parser.error('output must have a .png extension')
    colors = np.concatenate([
        args.floor_color, args.floor_grid_color, args.wheel_left_color, args.wheel_right_color,
        args.base_trail_color, args.contact_point_color, args.terrain_color,
    ])
    if not np.isfinite(colors).all() or not (0 <= colors).all() or not (colors <= 1).all():
        parser.error('floor, grid, terrain, trajectory and contact colors must be in [0, 1]')
    grid_dimensions = np.array([args.floor_grid_size, args.floor_grid_width])
    if not np.isfinite(grid_dimensions).all() or (grid_dimensions <= 0).any():
        parser.error('floor grid spacing and width must be finite and positive')
    radii = np.array([
        args.wheel_trail_radius, args.base_trail_radius, args.contact_point_radius,
    ])
    if not np.isfinite(radii).all() or (radii <= 0).any():
        parser.error('trajectory and contact radii must be finite and positive')
    if not np.isfinite(args.traj_smoothing) or args.traj_smoothing < 0:
        parser.error('traj-smoothing must be finite and nonnegative')
    if args.traj_samples_per_segment < 1:
        parser.error('traj-samples-per-segment must be positive')
    render(args)


if __name__ == '__main__':
    main()
