# ROS1 Depth Sim2sim Guide

This guide covers WF_TRON1B sim2sim with ROS1 depth transport in Docker. The host does not need ROS. Before the first run, complete the [NVIDIA GPU setup](../README.md#1-one-time-gpu-setup) in the root README.

## Quick Start

Run these commands from the repository root.

### 1. Build the image

```bash
sudo scripts/docker_build_ros1.sh
```

Rebuild after changing the Dockerfile or ROS/conda dependencies.

### 2. Create or start the container

```bash
xhost +local:docker
sudo -E scripts/docker_start_ros1.sh
```

This creates `tron_deploy` on the first run and starts it on later runs. It does not start sim2sim.

### 3. Enter the container and run sim2sim

```bash
sudo docker exec -it tron_deploy bash
```

```bash
/work/scripts/docker_run_sim2sim_ros1.sh
```

The launcher starts MuJoCo, the Python RL controller, ROS1 depth transport, the depth viewer, and the joystick. Depth is published on `/camera/depth/image_rect_raw` and logs are written to `/work/logs/sim2sim/`.

## Runtime Options

Disable only the depth viewer:

```bash
MJLAB_DEPTH_VIEW=0 /work/scripts/docker_run_sim2sim_ros1.sh
```

Enable the Grad-CAM overlay:

```bash
MJLAB_DEPTH_GRADCAM=1 MJLAB_DEPTH_VIEW=1 \
  /work/scripts/docker_run_sim2sim_ros1.sh
```

Generate terrain scenes after changing their definitions:

```bash
uv run python utils/terrain_tool/terrain_generator.py
```

Select a terrain scene; the default is the flat `robot.xml`:

```bash
MJLAB_SCENE=scene_stairs.xml /work/scripts/docker_run_sim2sim_ros1.sh
```

Available scenes are `scene_stairs.xml`, `scene_slope.xml`, `scene_rough_ground.xml`, `scene_obstacle.xml`, and the combined `scene_terrain.xml`.

## Manual Step-by-Step Start

Open separate terminals, enter the same container in each, and run:

```bash
sudo docker exec -it tron_deploy bash
cd /work
```

Terminal 1, ROS master:

```bash
roscore
```

If `roscore` fails because of `roslaunch`, use `rosmaster --core`.

Terminal 2, simulator:

```bash
python pointfoot-mujoco-sim/simulator.py 127.0.0.1
```

Terminal 3, controller:

```bash
ROBOT_TYPE=WF_TRON1B RL_TYPE=mjlab_repts_lin_depth python rl-deploy-with-python/main.py 127.0.0.1
```

Terminal 4, optional depth viewer:

```bash
python scripts/depth_image_viewer.py
```

Terminal 5, joystick:

```bash
pointfoot-mujoco-sim/robot-joystick/robot-joystick
```

## Checks and Troubleshooting

Inside the container, verify ROS depth transport with:

```bash
rostopic list
rostopic hz /camera/depth/image_rect_raw
python scripts/ros1_depth_smoke.py
```

Expected topics include `/camera/depth/image_rect_raw` and `/rosout`.

- `This script is intended to run inside the container...`: enter `tron_deploy` before running `/work/scripts/docker_run_sim2sim_ros1.sh`.
- `Unable to register with master node`: start `roscore` or use the combined launcher.
- `permission denied while trying to connect to the docker API`: use `sudo docker ...` or add your user to the Docker group.
- `glx: failed to create dri3 screen`: this is usually harmless if the MuJoCo window opens and the simulator runs.

## Container Lifecycle

Stop the persistent container:

```bash
sudo docker stop tron_deploy
```

Start it again later:

```bash
sudo -E scripts/docker_start_ros1.sh
```

Remove it only when you intentionally need to recreate it:

```bash
sudo docker rm tron_deploy
```
