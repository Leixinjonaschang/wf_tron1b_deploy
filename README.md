# WF_TRON1B Sim2sim

本仓库常用两种 sim2sim 流程：

- Perceptive depth policy：Docker + ROS1 depth topic。
- Non-perceptive REPTS policy：宿主机 `uv` 环境直接运行，不需要 ROS。

## Depth-Based Perceptive Sim-to-Sim Test

用于 depth-based policy：`mjlab_repts_lin_depth`。

在宿主机执行, 根据 Dockerfile 构建 docker 镜像：

```bash
cd wf_tron1b_deploy
sudo -E IMAGE_NAME=tron_sim2sim:latest scripts/docker_build_ros1.sh
```

`IMAGE_NAME=tron_sim2sim:latest` sets the Docker image tag used by the helper
script. It is optional because that is already the default; use a different
tag only when keeping multiple image versions. `sudo` runs the Docker command
with administrator privileges and `-E` preserves the caller's environment
(notably `DISPLAY`, proxy settings, and explicitly supplied variables). The
same command can therefore be written more simply as `sudo scripts/docker_build_ros1.sh`
when the defaults are sufficient.

### NVIDIA GPU prerequisite (Ubuntu/Debian host)

The ROS1 Docker workflow uses the host GPU for MuJoCo rendering. Before creating
`tron_deploy`, Docker must be able to access a working NVIDIA driver (`nvidia-smi`
must succeed on the host) and the NVIDIA Container Toolkit must be installed and
configured. Docker itself and an NVIDIA GPU driver are host prerequisites; a CUDA
toolkit installation is not required.

Install and configure the toolkit using NVIDIA's production repository:

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

If `tron_deploy` was created before this setup, remove and recreate that persistent
container so it receives the GPU runtime configuration:

```bash
sudo docker rm -f tron_deploy
```

After starting the container below, verify GPU access with:

```bash
sudo docker exec tron_deploy nvidia-smi
```

If Docker reports `failed to discover GPU vendor from CDI`, re-check that host
`nvidia-smi` works, rerun `nvidia-ctk runtime configure --runtime=docker`, restart
Docker, then recreate `tron_deploy`.


启动持久 Docker 容器：

```bash
xhost +local:docker
sudo -E IMAGE_NAME=tron_sim2sim:latest CONTAINER_NAME=tron_deploy scripts/docker_start_ros1.sh
```

进入容器：

```bash
sudo docker exec -it tron_deploy bash
```

在容器内运行 sim2sim：

```bash
/work/scripts/docker_run_sim2sim_ros1.sh
```

这会启动 MuJoCo、RL controller、ROS1 depth 传输、可选 depth viewer 和 joystick。
depth topic 是：

```text
/camera/depth/image_rect_raw
```

只关闭 depth viewer：

```bash
MJLAB_DEPTH_VIEW=0 /work/scripts/docker_run_sim2sim_ros1.sh
```

### Optional Grad-CAM Overlay

Grad-CAM is disabled by default, so existing images and launch commands keep
their original ONNX-only control behavior. To enable the visual overlay, first
rebuild the image after pulling these changes: it now includes the CPU PyTorch
runtime used only by the background explanation worker. A persistent container
created from an old image must be recreated after the rebuild:

```bash
# Host
sudo scripts/docker_build_ros1.sh
sudo docker rm -f tron_deploy
xhost +local:docker
sudo -E scripts/docker_start_ros1.sh
```

Then, inside the recreated container, start sim2sim with the feature enabled:

```bash
MJLAB_DEPTH_GRADCAM=1 MJLAB_DEPTH_GRADCAM_HZ=5.0 MJLAB_DEPTH_VIEW=1 \
  /work/scripts/docker_run_sim2sim_ros1.sh
```

The worker writes `depth_gradcam.npz` to `/work/logs/sim2sim/` by default. The
viewer displays only CAM frames no older than one second, over the policy's
retained D435 field of view. `MJLAB_DEPTH_GRADCAM_ALPHA` controls opacity
(default `0.45`), and `MJLAB_DEPTH_GRADCAM_PATH` overrides the output path.

日志：

```text
/work/logs/sim2sim/
```

更多细节见：`doc/ros1_depth_sim2sim.md`。

## Terrain Scenes

离线地形工具位于 `utils/terrain_tool/`。修改地形定义后，在仓库根目录生成全部场景：

```bash
uv run python utils/terrain_tool/terrain_generator.py
```

生成的场景位于
`pointfoot-mujoco-sim/robot-description/pointfoot/WF_TRON1B/xml/`：

- `scene_stairs.xml`：低台阶。
- `scene_slope.xml`：缓坡。
- `scene_rough_ground.xml`：碎石/不平地。
- `scene_obstacle.xml`：偏置圆柱障碍物。
- `scene_terrain.xml`：以上四类地形的组合课程。

用 `MJLAB_SCENE` 选择场景；未设置时仍使用平地 `robot.xml`。例如在 ROS1 Docker 容器内运行碎石场景：

```bash
MJLAB_SCENE=scene_rough_ground.xml /work/scripts/docker_run_sim2sim_ros1.sh
```

`scene_rough_ground.xml` 的难度定义在
`utils/terrain_tool/terrain_generator.py` 的 `add_rough_ground_course()`：

- `init_pos[2]` 与 `box_size[2]` 控制露出高度；当前约为 8–12 cm。
- `box_size_rand[2]` 控制高度起伏。
- `box_euler_rand` 控制随机倾角（单位为弧度）。
- `separation` 控制块间缝隙，`nums` 控制地形覆盖范围。

每次调整后重新运行生成器；它会同步更新独立碎石场景和组合场景。

## Non-Perceptive Sim-to-Sim Test

用于 non-perceptive policy sim2sim：`mjlab_repts` 或 `mjlab_repts_lin`。
这个流程直接在宿主机 `uv` 环境运行，不需要 ROS。

准备宿主机 Python 环境：

```bash
uv sync
```
### Representation TS for Blind Locomotion

两个终端分别启动：

```bash
export ROBOT_TYPE=WF_TRON1B && export RL_TYPE=mjlab_repts && uv run python pointfoot-mujoco-sim/simulator.py 
```

```bash
export ROBOT_TYPE=WF_TRON1B && export RL_TYPE=mjlab_repts && uv run python rl-deploy-with-python/main.py 
```

```bash
pointfoot-mujoco-sim/robot-joystick/robot-joystick
```

### Representation TS with Linear Velocity Prediction for Blind Locomotion

运行 LinVel variant：
```bash
export ROBOT_TYPE=WF_TRON1B && export RL_TYPE=mjlab_repts_lin && uv run python pointfoot-mujoco-sim/simulator.py 
```

```bash
export ROBOT_TYPE=WF_TRON1B && export RL_TYPE=mjlab_repts_lin && uv run python rl-deploy-with-python/main.py 
```

```bash
pointfoot-mujoco-sim/robot-joystick/robot-joystick
```

日志：

```text
logs/sim2sim/
```
