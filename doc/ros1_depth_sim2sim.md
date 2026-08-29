# ROS1 Depth Sim2sim README

This is the recommended workflow for running WF_TRON1B sim2sim with ROS1 depth
transport in Docker. The host does not need ROS installed.

## What Runs

- MuJoCo simulator
- Python RL controller
- ROS1 depth topic: `/camera/depth/image_rect_raw`
- Optional Python depth viewer: `scripts/depth_image_viewer.py`
- Virtual joystick in the foreground

The repository is mounted into the container at `/work`.

## 1. Build Image

Run on the host:

```bash
cd /media/phi/641A24011A23CF3C/ubuntu_data/CLX/project/wheeled_legged_proj/wf_tron1b_deploy
sudo -E IMAGE_NAME=tron_sim2sim:latest scripts/docker_build_ros1.sh
```

Rebuild after changing the Dockerfile or ROS/conda dependencies.

## 2. Start Container

Run on the host:

```bash
xhost +local:docker
sudo -E IMAGE_NAME=tron_sim2sim:latest CONTAINER_NAME=tron_deploy scripts/docker_start_ros1.sh
```

This creates or starts a persistent container named `tron_deploy`. It does not
start sim2sim.

Enter the container:

```bash
sudo docker exec -it tron_deploy bash
```

## 3. Run Sim2sim

Run inside the container:

```bash
/work/scripts/docker_run_sim2sim_ros1.sh
```

The launcher defaults to `mjlab_repts_gru_lin_depth`. The deployment side
supplies finite metric depth with invalid samples encoded as `0 m`; the ONNX
graph performs the `[0.2, 2.0] m` range mapping and `[0, 1]` normalization.

Stop sim2sim with `Ctrl-C` in the terminal running the joystick.

### Run terrain scenes

Generate the committed WF_TRON1B terrain scenes from the repository root when
you change their definitions:

```bash
uv run python utils/terrain_tool/terrain_generator.py
```

Select one scene for sim2sim (the default remains the flat `robot.xml` scene):

```bash
MJLAB_SCENE=scene_stairs.xml /work/scripts/docker_run_sim2sim_ros1.sh
```

Available scene files are `scene_stairs.xml`, `scene_slope.xml`,
`scene_rough_ground.xml`, and `scene_obstacle.xml`; each contains only its
named terrain. `scene_terrain.xml` remains available as the combined course.
All terrain is placed in front of the D435 camera along `+x` and uses MuJoCo
group 0, so it is included in the depth renderer. Perlin/image height-field
helpers are optional and require `pip install noise opencv-python` when used.

Logs are written to:

```bash
/work/logs/sim2sim/
```

## Disable Depth Viewer

Run inside the container:

```bash
MJLAB_DEPTH_VIEW=0 /work/scripts/docker_run_sim2sim_ros1.sh
```

This disables only the Python depth viewer. The simulator and controller still
use ROS1 depth transport.

## Manual Step-by-Step Start

Open several terminals and enter the same container in each one:

```bash
sudo docker exec -it tron_deploy bash
cd /work
```

Terminal 1, start ROS master:

```bash
roscore
```

If `roscore` fails because of `roslaunch`, use:

```bash
rosmaster --core
```

Terminal 2, start simulator:

```bash
python pointfoot-mujoco-sim/simulator.py 127.0.0.1
```

Terminal 3, start controller:

```bash
ROBOT_TYPE=WF_TRON1B RL_TYPE=mjlab_repts_gru_lin_depth python rl-deploy-with-python/main.py 127.0.0.1
```

simulator、controller 和一键启动脚本应统一使用
`RL_TYPE=mjlab_repts_gru_lin_depth`。

Terminal 4, optional depth viewer:

```bash
python scripts/depth_image_viewer.py
```

Terminal 5, start joystick:

```bash
pointfoot-mujoco-sim/robot-joystick/robot-joystick
```

## Useful Checks

Inside the container:

```bash
rostopic list
rostopic hz /camera/depth/image_rect_raw
uv --version
python scripts/ros1_depth_smoke.py
```

Expected topic list includes:

```text
/camera/depth/image_rect_raw
/rosout
```

## Script Roles

- `scripts/docker_build_ros1.sh`: host-side image build.
- `scripts/docker_start_ros1.sh`: host-side persistent container start.
- `scripts/docker_run_sim2sim_ros1.sh`: container-side sim2sim launcher.
- `scripts/start_sim2sim.sh`: starts ROS master if needed, simulator, controller, depth
  viewer, and joystick.

## Stop Or Remove Container

Stop the persistent container:

```bash
sudo docker stop tron_deploy
```

Start it again later:

```bash
sudo -E scripts/docker_start_ros1.sh
```

Remove it only when you want to recreate it:

```bash
sudo docker rm tron_deploy
```

## Common Problems

`This script is intended to run inside the container...`

You ran `scripts/docker_run_sim2sim_ros1.sh` on the host. First enter Docker:

```bash
sudo docker exec -it tron_deploy bash
```

`Unable to register with master node`

ROS master is not running. Start `roscore` or `rosmaster --core` first, or use
`/work/scripts/docker_run_sim2sim_ros1.sh`.

`permission denied while trying to connect to the docker API`

Use `sudo docker ...`, or add your user to the Docker group.

`glx: failed to create dri3 screen`

This can appear with NVIDIA/GLX in Docker. If the MuJoCo window opens and the
simulator runs, it is usually harmless.
