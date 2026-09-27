#!/usr/bin/env python3
"""Render one recorded robot pose, its camera FOV, and a separate depth image."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from render_state_trajectory import (
    DEFAULT_CSV,
    DEFAULT_SCENE,
    add_top_light,
    configure_floor,
    load_states,
)


def nearest_state_index(times, requested_time):
    """Select an original row, rejecting requests outside the recording."""
    if not np.isfinite(requested_time) or not times[0] <= requested_time <= times[-1]:
        raise ValueError(
            f'--time must be within the recording: {times[0]:.6f}–{times[-1]:.6f} s'
        )
    return int(np.argmin(np.abs(times - requested_time)))


def camera_rays(fovy, width, height, image_xy):
    """Rays at normalized image coordinates: x right, y up; axial depth is one."""
    half_height = np.tan(np.deg2rad(fovy) / 2)
    xy = np.asarray(image_xy, dtype=np.float64)
    return np.column_stack((
        xy[:, 0] * half_height * width / height,
        xy[:, 1] * half_height,
        -np.ones(len(xy)),
    ))


def build_fov_wireframe(mujoco, model, data, camera_id, width, height,
                        length, clip, radius, samples_per_edge=64):
    """Trace the image border, optionally stopping its rays at visible surfaces.

    Points follow the full image border, starting at top left. The four corner
    rays form the pyramid sides. Offsetting hit points slightly toward the camera
    keeps the outline visible on terrain without changing recorded hit positions.
    """
    if model.cam_projection[camera_id] != 0 or np.any(model.cam_sensorsize[camera_id]):
        raise ValueError('FOV drawing requires a perspective camera configured with fovy')
    fovy = float(model.cam_fovy[camera_id])
    corners = np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]])
    border = np.concatenate([
        np.linspace(a, b, samples_per_edge, endpoint=False)
        for a, b in zip(corners, np.roll(corners, -1, axis=0))
    ]) if clip else corners
    origin = data.cam_xpos[camera_id].copy()
    rotation = data.cam_xmat[camera_id].reshape(3, 3).copy()
    rays = camera_rays(fovy, width, height, border) @ rotation.T
    ray_norms = np.linalg.norm(rays, axis=1)
    directions = rays / ray_norms[:, None]
    near = float(model.vis.map.znear * model.stat.extent)
    far = float(model.vis.map.zfar * model.stat.extent)
    if not near < length < far:
        raise ValueError(f'--fov-length must lie between camera clips {near:.4f} and {far:.4f} m')
    depths = np.full(len(rays), length)
    geom_ids = np.full(len(rays), -1, dtype=np.int32)
    if clip:
        groups = np.array([1, 1, 0, 0, 0, 0], dtype=np.uint8)
        for i, direction in enumerate(directions):
            geom_id = np.array([-1], dtype=np.int32)
            # Match the depth renderer's near plane and geom groups, including
            # visible robot parts if the camera is partially self-occluded.
            distance = mujoco.mj_ray(
                model, data, origin + near * rays[i], direction,
                groups, True, -1, geom_id,
            )
            hit_depth = near + distance / ray_norms[i]
            if distance >= 0 and hit_depth <= length:
                depths[i] = hit_depth
                geom_ids[i] = geom_id[0]
    points = origin + rays * depths[:, None]
    display_points = points.copy()
    hit = geom_ids >= 0
    display_points[hit] -= directions[hit] * radius * 1.5
    segments = []
    # Avoid joining an occlusion edge across a large depth discontinuity.
    for i in range(len(points)):
        j = (i + 1) % len(points)
        if not clip or abs(depths[i] - depths[j]) <= 0.15:
            segments.append([display_points[i], display_points[j]])
    for i in range(0, len(points), samples_per_edge if clip else 1):
        segments.append([origin, display_points[i]])
    return {
        'origin': origin,
        'rotation': rotation,
        'fovy': fovy,
        'fovx': float(np.rad2deg(2 * np.arctan(
            np.tan(np.deg2rad(fovy) / 2) * width / height,
        ))),
        'points': points,
        'geom_ids': geom_ids,
        'segments': np.asarray(segments),
        'clip_near': near,
        'clip_far': far,
    }


def terrain_scan_points(mujoco, model, data, camera_id, width, height,
                        columns, rows, max_depth):
    """Sample image rays; keep only first hits on static terrain, including floor.

    Cast against both robot and terrain so self-occluded ground is not marked.
    The image-plane distance used to illustrate the FOV does not limit these rays.
    """
    x = 2 * (np.arange(columns) + 0.5) / columns - 1
    y = 1 - 2 * (np.arange(rows) + 0.5) / rows
    image_xy = np.stack(np.meshgrid(x, y), axis=-1).reshape(-1, 2)
    rotation = data.cam_xmat[camera_id].reshape(3, 3)
    rays = camera_rays(model.cam_fovy[camera_id], width, height, image_xy) @ rotation.T
    origin = data.cam_xpos[camera_id]
    near = float(model.vis.map.znear * model.stat.extent)
    far = min(max_depth, float(model.vis.map.zfar * model.stat.extent))
    groups = np.array([1, 1, 0, 0, 0, 0], dtype=np.uint8)
    points, normals, geom_ids, selected_xy = [], [], [], []
    for xy, ray in zip(image_xy, rays):
        ray_norm = np.linalg.norm(ray)
        direction = ray / ray_norm
        geom_id = np.array([-1], dtype=np.int32)
        normal = np.zeros(3)
        distance = mujoco.mj_ray(
            model, data, origin + near * ray, direction,
            groups, True, -1, geom_id, normal,
        )
        if distance < 0 or near + distance / ray_norm > far:
            continue
        body_id = model.geom_bodyid[geom_id[0]]
        if model.body_weldid[body_id] != 0:
            continue
        # Orient the surface normal toward the camera for the marker offset.
        if np.dot(normal, direction) > 0:
            normal *= -1
        points.append(origin + near * ray + distance * direction)
        normals.append(normal)
        geom_ids.append(int(geom_id[0]))
        selected_xy.append(xy)
    return {
        'points': np.asarray(points).reshape(-1, 3),
        'normals': np.asarray(normals).reshape(-1, 3),
        'geom_ids': np.asarray(geom_ids, dtype=np.int32),
        'image_xy': np.asarray(selected_xy).reshape(-1, 2),
    }


def add_scan_dots(mujoco, scene, scan, radius, color, alpha):
    """Draw non-shadow-casting markers just outside the hit surfaces."""
    if scene.ngeom + len(scan['points']) > scene.maxgeom:
        raise ValueError('Not enough scene geometry capacity for scan dots')
    positions = scan['points'] + scan['normals'] * radius * 0.5
    rgba = np.array([*color, alpha], dtype=np.float32)
    for position in positions:
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom, mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([radius, 0, 0]), position, np.eye(3).reshape(-1), rgba,
        )
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        geom.specular = 0
        geom.shininess = 0
        geom.reflectance = 0
        scene.ngeom += 1


def add_fov_wireframe(mujoco, scene, segments, radius, color, alpha):
    """Add matte, non-shadow-casting lines only to the overview scene."""
    identity = np.eye(3).reshape(-1)
    rgba = np.array([*color, alpha], dtype=np.float32)
    if scene.ngeom + len(segments) > scene.maxgeom:
        raise ValueError('Not enough scene geometry capacity for the FOV wireframe')
    for start, end in segments:
        if np.linalg.norm(end - start) < 1e-9:
            continue
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3), np.zeros(3), identity, rgba,
        )
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, start, end)
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        geom.specular = 0
        geom.shininess = 0
        geom.reflectance = 0
        scene.ngeom += 1


def overlay_fov_wireframe(mujoco, renderer, overview, fov, radius, color, alpha):
    """Render a coverage mask from the same viewpoint, then overlay the wire.

    An image plane can intersect the terrain for a downward camera.
    This separate schematic pass preserves its geometry while making all edges
    visible; it does not alter terrain occlusion, shadows, or the sensor image.
    """
    scene = renderer.scene
    scene.ngeom = 0
    add_fov_wireframe(mujoco, scene, fov['segments'], radius, (1, 1, 1), 1)
    for geom in scene.geoms[:scene.ngeom]:
        geom.emission = 1
    for flag in (mujoco.mjtRndFlag.mjRND_SKYBOX, mujoco.mjtRndFlag.mjRND_FOG,
                 mujoco.mjtRndFlag.mjRND_HAZE, mujoco.mjtRndFlag.mjRND_SHADOW,
                 mujoco.mjtRndFlag.mjRND_REFLECTION):
        scene.flags[flag] = False
    mask = renderer.render().max(axis=2).astype(np.float32)[..., None] / 255
    weight = mask * alpha
    pixels = overview * (1 - weight) + np.asarray(color) * 255 * weight
    return np.rint(pixels).clip(0, 255).astype(np.uint8)


def depth_preview(depth, minimum, maximum):
    """8-bit grayscale: near black, far white; only the preview is clipped."""
    values = np.nan_to_num(depth, nan=maximum, posinf=maximum, neginf=minimum)
    gray = np.rint(np.clip((values - minimum) / (maximum - minimum), 0, 1) * 255)
    return np.repeat(gray.astype(np.uint8)[..., None], 3, axis=2)


def load_scene_with_depth_texture(mujoco, path, width, height):
    """Reserve texture/material assets without adding any physical geometry."""
    spec = mujoco.MjSpec.from_file(str(path.resolve()))
    texture = spec.add_texture(
        name='__depth_fov_texture', type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT, width=width, height=height,
        rgb1=[1, 1, 1], rgb2=[1, 1, 1],
    )
    material = spec.add_material(
        name='__depth_fov_material', texuniform=False, texrepeat=[1, 1],
        emission=1, specular=0, shininess=0, reflectance=0, rgba=[1, 1, 1, 1],
    )
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = texture.name
    model = spec.compile()
    texture_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, texture.name)
    material_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, material.name)
    return model, texture_id, material_id


def set_depth_texture(model, texture_id, pixels):
    """Set the top-to-bottom RGB texture before creating the overview renderer."""
    shape = (model.tex_height[texture_id], model.tex_width[texture_id],
             model.tex_nchannel[texture_id])
    if pixels.shape != shape or pixels.dtype != np.uint8:
        raise ValueError(f'Depth texture must have shape {shape} and dtype uint8')
    start = model.tex_adr[texture_id]
    # MuJoCo's primitive 2D texture mapping already maps +Y to the image top.
    model.tex_data[start:start + pixels.size] = pixels.reshape(-1)


def add_depth_image_plane(mujoco, scene, fov, distance, width, height,
                          material_id, alpha):
    """Add a finite, two-sided textured panel within the FOV image rectangle."""
    if scene.ngeom == scene.maxgeom:
        raise ValueError('Not enough scene geometry capacity for the depth image plane')
    half_height = distance * np.tan(np.deg2rad(fov['fovy']) / 2)
    center = fov['origin'] - distance * fov['rotation'][:, 2]
    geom = scene.geoms[scene.ngeom]
    # A 0.2 mm box has two opaque faces; MuJoCo makes the back of mjGEOM_PLANE
    # automatically translucent. Both box faces use the same camera X/Y mapping.
    mujoco.mjv_initGeom(
        geom, mujoco.mjtGeom.mjGEOM_BOX,
        np.array([half_height * width / height, half_height, 0.0001]),
        center, fov['rotation'].reshape(-1), np.array([1, 1, 1, alpha], dtype=np.float32),
    )
    geom.matid = material_id
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
    geom.emission = 1
    geom.specular = 0
    geom.shininess = 0
    geom.reflectance = 0
    scene.ngeom += 1


def save_png(path, pixels):
    import pygame

    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = pixels.shape[:2]
    surface = pygame.image.frombuffer(pixels.tobytes(), (width, height), 'RGB')
    pygame.image.save(surface, str(path))


def render(args):
    os.environ.setdefault('MESA_SHADER_CACHE_DISABLE', 'true')
    os.environ['MUJOCO_GL'] = args.gl
    import mujoco

    texture_id = material_id = None
    if args.depth_plane:
        model, texture_id, material_id = load_scene_with_depth_texture(
            mujoco, args.scene, args.depth_width, args.depth_height,
        )
    else:
        model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    if model.njnt == 0 or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError('Expected a WF_TRON1B scene with the root free joint first')
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera)
    if camera_id < 0:
        raise ValueError(f'Camera {args.camera!r} does not exist in {args.scene}')
    times, poses = load_states(args.csv, model.nq)
    index = nearest_state_index(times, args.time)
    data = mujoco.MjData(model)
    data.qpos[:] = poses[index]
    data.time = times[index]
    mujoco.mj_forward(model, data)
    model.vis.global_.offwidth = max(args.width, args.depth_width, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(args.height, args.depth_height, model.vis.global_.offheight)

    # Match SimulatorMujoco._init_depth_export before applying overview styling.
    # FOV markers never enter the sensor scene or modify the physical model.
    depth_option = mujoco.MjvOption()
    depth_option.geomgroup[:] = 0
    depth_option.geomgroup[:2] = 1
    fov = build_fov_wireframe(
        mujoco, model, data, camera_id, args.depth_width, args.depth_height,
        args.fov_length, args.fov_clip, args.fov_radius,
    )
    scan = terrain_scan_points(
        mujoco, model, data, camera_id, args.depth_width, args.depth_height,
        args.scan_columns, args.scan_rows, args.scan_max_depth,
    ) if args.scan_dots else {
        'points': np.empty((0, 3)), 'normals': np.empty((0, 3)),
        'geom_ids': np.empty(0, dtype=np.int32), 'image_xy': np.empty((0, 2)),
    }
    with mujoco.Renderer(model, height=args.depth_height, width=args.depth_width) as renderer:
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=args.camera, scene_option=depth_option)
        depth = renderer.render().copy()
    preview = depth_preview(depth, args.depth_min, args.depth_max)
    if args.depth_plane:
        set_depth_texture(model, texture_id, preview)

    configure_floor(model, mujoco, args.floor_color)
    model.light_castshadow[:] = args.shadows
    model.light_diffuse[:] *= args.light_scale
    model.vis.headlight.diffuse[:] *= args.light_scale
    model.vis.headlight.ambient[:] *= args.light_scale
    robot = model.body_rootid[model.geom_bodyid] == model.jnt_bodyid[0]
    model.geom_rgba[robot & (model.geom_group != 1), 3] = 0
    terrain = (model.body_rootid[model.geom_bodyid] == 0) & (
        model.geom_type != mujoco.mjtGeom.mjGEOM_PLANE
    )
    model.geom_rgba[terrain, :3] = args.terrain_color
    option = mujoco.MjvOption()
    option.geomgroup[:] = 1
    option.sitegroup[:] = 0

    # Frame this robot and its FOV rather than the full terrain strip.
    robot_points = data.geom_xpos[robot & (model.geom_group == 1)]
    points = np.concatenate((robot_points - 0.2, robot_points + 0.2,
                             fov['points'], fov['origin'][None], scan['points']))
    low, high = points.min(axis=0), points.max(axis=0)
    center = (low + high) / 2
    radius = max(0.5, np.linalg.norm(high - low) / 2)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = center if args.lookat is None else args.lookat
    half_fovy = np.deg2rad(model.vis.global_.fovy) / 2
    half_angle = min(half_fovy, np.arctan(np.tan(half_fovy) * args.width / args.height))
    camera.distance = args.distance or 1.15 * radius / np.sin(half_angle)
    camera.azimuth = args.azimuth
    camera.elevation = args.elevation
    shadow_radius = radius + 1.0
    model.vis.map.shadowclip = shadow_radius / model.stat.extent
    capacity = max(10000, model.ngeom + len(fov['segments']) + len(scan['points']) + 100)
    with mujoco.Renderer(model, height=args.height, width=args.width,
                         max_geom=capacity) as renderer:
        renderer.update_scene(data, camera=camera, scene_option=option)
        scene = renderer.scene
        if args.top_light:
            add_top_light(mujoco, scene, center + [0, 0, 5],
                          args.top_light_intensity, shadows=args.shadows)
        for light in scene.lights[:scene.nlight]:
            if light.castshadow and light.type == mujoco.mjtLightType.mjLIGHT_DIRECTIONAL:
                light.pos[:] = center - light.dir * (shadow_radius + 1)
        for geom in scene.geoms[:scene.ngeom]:
            if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and robot[geom.objid]:
                geom.rgba[3] *= args.robot_alpha
                if geom.rgba[3] == 0:
                    # MuJoCo's shadow pass ignores alpha; hidden robots must
                    # also be excluded from casting shadows in the overview.
                    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
                if args.robot_matte:
                    geom.specular = 0
                    geom.shininess = 0
                    geom.reflectance = 0
        add_scan_dots(mujoco, scene, scan, args.scan_radius, args.scan_color, args.scan_alpha)
        if args.depth_plane:
            add_depth_image_plane(
                mujoco, scene, fov, args.fov_length, args.depth_width,
                args.depth_height, material_id, args.depth_plane_alpha,
            )
        if not args.fov_overlay:
            add_fov_wireframe(mujoco, scene, fov['segments'], args.fov_radius,
                              args.fov_color, args.fov_alpha)
        scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = args.shadows
        overview = renderer.render().copy()
        if args.fov_overlay:
            overview = overlay_fov_wireframe(
                mujoco, renderer, overview, fov,
                args.fov_radius, args.fov_color, args.fov_alpha,
            )

    save_png(args.output, overview)
    save_png(args.depth_output, preview)
    depth_path = args.depth_output.with_suffix('.npy')
    np.save(depth_path, depth.astype(np.float32), allow_pickle=False)
    metadata = {
        'source_csv': str(args.csv.resolve()),
        'scene': str(args.scene.resolve()),
        'mujoco_version': mujoco.__version__,
        'requested_time_s': args.time,
        'sim_time': float(times[index]),
        'time_error_s': float(times[index] - args.time),
        'source_row_index_zero_based': index,
        'qpos': poses[index].tolist(),
        'camera': {
            'name': args.camera,
            'position_world_m': fov['origin'].tolist(),
            'rotation_camera_to_world': fov['rotation'].tolist(),
            'forward_world': (-fov['rotation'][:, 2]).tolist(),
            'fovy_deg': fov['fovy'], 'fovx_deg': fov['fovx'],
            'clip_near_m': fov['clip_near'], 'clip_far_m': fov['clip_far'],
        },
        'fov': {
            'max_axial_depth_m': args.fov_length,
            'image_plane_distance_from_camera_m': args.fov_length,
            'image_plane_center_world_m': (
                fov['origin'] - args.fov_length * fov['rotation'][:, 2]
            ).tolist(),
            'image_plane_normal_world': (-fov['rotation'][:, 2]).tolist(),
            'clip_to_visible_surfaces': args.fov_clip,
            'overlay_without_scene_occlusion': args.fov_overlay,
            'color': args.fov_color, 'alpha': args.fov_alpha, 'radius_m': args.fov_radius,
            'border_points_world_m': fov['points'].tolist(),
            'border_hit_geom_ids': fov['geom_ids'].tolist(),
            'surface_display_offset_toward_camera_m': (
                1.5 * args.fov_radius if args.fov_clip else 0
            ),
        },
        'scan_dots': {
            'enabled': args.scan_dots,
            'sampling': 'uniform image grid cell centers, first visible static surface only',
            'columns': args.scan_columns, 'rows': args.scan_rows,
            'max_axial_depth_m': args.scan_max_depth,
            'count': len(scan['points']),
            'radius_m': args.scan_radius, 'color': args.scan_color, 'alpha': args.scan_alpha,
            'positions_world_m': scan['points'].tolist(),
            'surface_normals_world': scan['normals'].tolist(),
            'geom_ids': scan['geom_ids'].tolist(),
            'image_xy_normalized': scan['image_xy'].tolist(),
            'display_offset_along_surface_normal_m': 0.5 * args.scan_radius,
        },
        'depth': {
            'png': str(args.depth_output.resolve()), 'npy': str(depth_path.resolve()),
            'width': args.depth_width, 'height': args.depth_height,
            'units': 'm', 'quantity': 'axial depth along camera -Z, not Euclidean range',
            'source': 're-rendered from recorded qpos and scene, not recorded sensor data',
            'geom_groups': [0, 1], 'policy_crop': False,
            'preview': 'linear grayscale; near black, far white',
            'preview_range_m': [args.depth_min, args.depth_max],
            'raw_npy_clipped': False,
            'background': 'MuJoCo far clipping depth',
        },
        'depth_plane': {
            'enabled': args.depth_plane, 'alpha': args.depth_plane_alpha,
            'axial_distance_from_camera_m': args.fov_length,
            'texture_source': str(args.depth_output.resolve()),
            'preview_range_m': [args.depth_min, args.depth_max],
            'two_sided': True, 'panel_thickness_m': 0.0002,
            'scene_occlusion': True, 'casts_shadows': False,
        },
        'overview': {
            'png': str(args.output.resolve()), 'width': args.width, 'height': args.height,
            'lookat': camera.lookat.tolist(), 'distance': camera.distance,
            'azimuth': camera.azimuth, 'elevation': camera.elevation,
            'terrain_color': args.terrain_color, 'floor_color': args.floor_color,
            'robot_matte': args.robot_matte, 'robot_alpha': args.robot_alpha,
            'shadows': args.shadows,
            'light_scale': args.light_scale, 'top_light': args.top_light,
            'top_light_intensity': args.top_light_intensity,
        },
    }
    metadata_path = args.output.with_suffix('.json')
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Requested {args.time:.6f} s; selected row {index}, {times[index]:.6f} s')
    print(f'{args.camera} FOV: horizontal {fov["fovx"]:.2f}°, vertical {fov["fovy"]:.2f}°')
    print(f'Terrain scan dots: {len(scan["points"])}')
    for path in (args.output, args.depth_output, depth_path, metadata_path):
        print(path.resolve())


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=DEFAULT_CSV)
    parser.add_argument('--scene', type=Path, default=DEFAULT_SCENE)
    parser.add_argument('--time', type=float, default=63.5, help='Simulation seconds; nearest recorded row')
    parser.add_argument('--camera', default='d435', help='Camera name in scene XML')
    parser.add_argument('--output', type=Path, default=Path('logs/sim2sim/robot_depth_fov.png'))
    parser.add_argument('--depth-output', type=Path, help='Default: <output stem>_depth.png, plus raw .npy')
    parser.add_argument('--width', type=int, default=3840)
    parser.add_argument('--height', type=int, default=2160)
    parser.add_argument('--depth-width', type=int, default=848)
    parser.add_argument('--depth-height', type=int, default=480)
    parser.add_argument('--depth-min', type=float, default=0.2, help='Preview black level in meters')
    parser.add_argument('--depth-max', type=float, default=2.0, help='Preview white level in meters')
    parser.add_argument('--depth-plane', action=argparse.BooleanOptionalAction, default=True,
                        help='Display the depth image on the FOV image plane')
    parser.add_argument('--depth-plane-alpha', type=float, default=1.0,
                        help='Depth image plane opacity, independent of FOV line opacity')
    parser.add_argument('--fov-length', type=float, default=0.5,
                        help='Image plane normal distance from camera center in meters')
    parser.add_argument('--fov-clip', action=argparse.BooleanOptionalAction, default=False,
                        help='Optional legacy mode: stop FOV border rays at visible surfaces')
    parser.add_argument('--fov-overlay', action=argparse.BooleanOptionalAction, default=False,
                        help='Show the complete wireframe even behind terrain')
    parser.add_argument('--fov-color', type=float, nargs=3, default=(0.5, 0.5, 0.5))
    parser.add_argument('--fov-alpha', type=float, default=1.0)
    parser.add_argument('--fov-radius', type=float, default=0.0075, help='Wire radius in meters')
    parser.add_argument('--scan-dots', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--scan-columns', type=int, default=32, help='Image-grid sampling columns')
    parser.add_argument('--scan-rows', type=int, default=20, help='Image-grid sampling rows')
    parser.add_argument('--scan-max-depth', type=float, default=2.0,
                        help='Maximum axial depth of scan hits in meters; independent of FOV length')
    parser.add_argument('--scan-radius', type=float, default=0.008, help='Scan marker radius in meters')
    parser.add_argument('--scan-color', type=float, nargs=3, default=(1.0, 0.85, 0.2))
    parser.add_argument('--scan-alpha', type=float, default=0.85)
    parser.add_argument('--azimuth', type=float, default=240)
    parser.add_argument('--elevation', type=float, default=-25)
    parser.add_argument('--distance', type=float, help='Overview distance; default auto-fit')
    parser.add_argument('--lookat', type=float, nargs=3, help='Overview target; default robot/FOV center')
    parser.add_argument('--terrain-color', type=float, nargs=3, default=(0.30, 0.58, 0.32))
    parser.add_argument('--floor-color', type=float, nargs=3, default=(1.0, 1.0, 1.0))
    parser.add_argument('--robot-matte', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--robot-alpha', type=float, default=1.0,
                        help='Overview robot opacity: 0 invisible, 1 opaque; sensor depth unchanged')
    parser.add_argument('--shadows', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--light-scale', type=float, default=1.0, help='Overview scene/headlight multiplier')
    parser.add_argument('--top-light', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--top-light-intensity', type=float, default=0.35)
    parser.add_argument('--gl', choices=('egl', 'osmesa', 'glfw'), default='egl')
    args = parser.parse_args(argv)
    if args.depth_output is None:
        args.depth_output = args.output.with_name(args.output.stem + '_depth.png')
    if min(args.width, args.height, args.depth_width, args.depth_height) <= 0:
        parser.error('Image dimensions must be positive')
    if min(args.scan_columns, args.scan_rows) <= 0:
        parser.error('Scan rows and columns must be positive')
    values = [args.time, args.depth_min, args.depth_max, args.fov_length,
              args.fov_radius, args.fov_alpha, args.azimuth, args.elevation,
              args.light_scale, args.top_light_intensity,
              args.scan_radius, args.scan_alpha, args.scan_max_depth,
              args.depth_plane_alpha, args.robot_alpha]
    if args.lookat is not None:
        values.extend(args.lookat)
    if args.distance is not None:
        values.append(args.distance)
    if not np.isfinite(values).all():
        parser.error('Numeric parameters must be finite')
    if not 0 <= args.depth_min < args.depth_max:
        parser.error('Require 0 <= --depth-min < --depth-max')
    if min(args.fov_radius, args.fov_length, args.scan_radius, args.scan_max_depth) <= 0 or (
        args.distance is not None and args.distance <= 0
    ):
        parser.error('FOV/scan radii and distances must be positive')
    if not (0 <= args.fov_alpha <= 1 and 0 <= args.scan_alpha <= 1):
        parser.error('FOV and scan alpha must be in [0, 1]')
    if not 0 <= args.depth_plane_alpha <= 1:
        parser.error('Depth plane alpha must be in [0, 1]')
    if not 0 <= args.robot_alpha <= 1:
        parser.error('Robot alpha must be in [0, 1]')
    if min(args.light_scale, args.top_light_intensity) < 0:
        parser.error('Light intensities must be nonnegative')
    for color in (args.fov_color, args.floor_color, args.terrain_color, args.scan_color):
        if not np.isfinite(color).all() or np.any(np.array(color) < 0) or np.any(np.array(color) > 1):
            parser.error('Colors must be finite RGB values in [0, 1]')
    if any(path.suffix.lower() != '.png' for path in (args.output, args.depth_output)):
        parser.error('Overview and depth output paths must end in .png')
    outputs = [args.output, args.depth_output, args.depth_output.with_suffix('.npy'),
               args.output.with_suffix('.json')]
    if len({path.resolve() for path in outputs}) != len(outputs):
        parser.error('Output paths must be distinct')
    return args


def main():
    render(parse_args())


if __name__ == '__main__':
    main()
