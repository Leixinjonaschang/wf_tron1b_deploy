# WF_TRON1B Sim2sim

[中文版 README](README.zh-CN.md)

This repository supports two sim2sim workflows:

- Perceptive depth policy: Docker + ROS1 depth topic.
- Non-perceptive REPTS policy: runs directly in the host `uv` environment without ROS.

## Depth-Based Perceptive Sim-to-Sim Test

For the depth-based policy: `mjlab_repts_lin_depth`. Run the following steps in order.

### 1. One-time GPU setup

On an Ubuntu/Debian host, Docker must have access to a working NVIDIA driver.
Confirm that `nvidia-smi` succeeds, then install and configure the NVIDIA Container Toolkit:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 2. Build the image

From the repository root:

```bash
sudo scripts/docker_build_ros1.sh
```

### 3. Create or start the container

```bash
xhost +local:docker
sudo -E scripts/docker_start_ros1.sh
```

The helper creates `tron_deploy` on its first run and starts it on later runs.

### 4. Enter the container

```bash
sudo docker exec -it tron_deploy bash
```

### 5. Run sim2sim

```bash
/work/scripts/docker_run_sim2sim_ros1.sh
```

This starts MuJoCo, the RL controller, ROS1 depth transport, the depth viewer, and the joystick. The depth topic is `/camera/depth/image_rect_raw`.

Disable only the depth viewer:

```bash
MJLAB_DEPTH_VIEW=0 /work/scripts/docker_run_sim2sim_ros1.sh
```

### Grad-CAM Overlay

Enable the depth-policy Grad-CAM overlay with `MJLAB_DEPTH_GRADCAM=1`:

```bash
MJLAB_DEPTH_GRADCAM=1 MJLAB_DEPTH_VIEW=1 \
  /work/scripts/docker_run_sim2sim_ros1.sh
```

Logs are written to `/work/logs/sim2sim/`. See [the ROS1 depth guide](doc/ros1_depth_sim2sim.md) for manual startup and troubleshooting.

## Terrain Scenes

Generate terrain scenes after changing their definitions:

```bash
uv run python utils/terrain_tool/terrain_generator.py
```

Available scenes are `scene_stairs.xml`, `scene_slope.xml`, `scene_rough_ground.xml`, `scene_obstacle.xml`, and the combined `scene_terrain.xml`. Use `MJLAB_SCENE` to select one; the default is the flat `robot.xml`.

```bash
MJLAB_SCENE=scene_rough_ground.xml /work/scripts/docker_run_sim2sim_ros1.sh
```

## Non-Perceptive Sim-to-Sim Test

For `mjlab_repts`, prepare the host Python environment:

```bash
uv sync
```

Run the simulator, controller, and joystick in three host terminals.

Terminal 1:

```bash
ROBOT_TYPE=WF_TRON1B RL_TYPE=mjlab_repts uv run python pointfoot-mujoco-sim/simulator.py
```

Terminal 2:

```bash
ROBOT_TYPE=WF_TRON1B RL_TYPE=mjlab_repts uv run python rl-deploy-with-python/main.py
```

Terminal 3:

```bash
pointfoot-mujoco-sim/robot-joystick/robot-joystick
```

For the LinVel variant, replace `mjlab_repts` with `mjlab_repts_lin` in terminals 1 and 2.

Logs are written to `logs/sim2sim/`.
