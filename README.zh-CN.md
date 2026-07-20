# WF_TRON1B Sim2sim

[English README](README.md)

本仓库支持两种 sim2sim 流程：

- 感知型深度策略：Docker + ROS1 深度图话题。
- 非感知 REPTS 策略：直接在宿主机 `uv` 环境中运行，不需要 ROS。

## 基于深度的感知型 Sim-to-Sim 测试

用于深度策略：`mjlab_repts_lin_depth`。请按以下顺序执行。

### 1. 一次性 GPU 配置

在 Ubuntu/Debian 宿主机上，Docker 必须能访问可用的 NVIDIA 驱动。确认 `nvidia-smi` 能成功运行后，安装并配置 NVIDIA Container Toolkit：

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

### 2. 构建镜像

在仓库根目录执行：

```bash
sudo scripts/docker_build_ros1.sh
```

### 3. 创建或启动容器

```bash
xhost +local:docker
sudo -E scripts/docker_start_ros1.sh
```

辅助脚本首次运行时创建 `tron_deploy`，后续运行时会直接启动它。

### 4. 进入容器

```bash
sudo docker exec -it tron_deploy bash
```

### 5. 运行 sim2sim

```bash
/work/scripts/docker_run_sim2sim_ros1.sh
```

该命令会启动 MuJoCo、RL 控制器、ROS1 深度图传输、深度图查看器和手柄程序。深度图话题为 `/camera/depth/image_rect_raw`。

仅关闭深度图查看器：

```bash
MJLAB_DEPTH_VIEW=0 /work/scripts/docker_run_sim2sim_ros1.sh
```

### Grad-CAM 叠加层

通过 `MJLAB_DEPTH_GRADCAM=1` 启用深度策略的 Grad-CAM 叠加层：

```bash
MJLAB_DEPTH_GRADCAM=1 MJLAB_DEPTH_VIEW=1 \
  /work/scripts/docker_run_sim2sim_ros1.sh
```

日志写入 `/work/logs/sim2sim/`。手动启动和故障排查见 [ROS1 深度图指南](doc/ros1_depth_sim2sim.md)。

## 地形场景

修改地形定义后，生成场景：

```bash
uv run python utils/terrain_tool/terrain_generator.py
```

可用场景为 `scene_stairs.xml`、`scene_slope.xml`、`scene_rough_ground.xml`、`scene_obstacle.xml`，以及组合场景 `scene_terrain.xml`。通过 `MJLAB_SCENE` 选择场景；默认使用平地 `robot.xml`。

```bash
MJLAB_SCENE=scene_rough_ground.xml /work/scripts/docker_run_sim2sim_ros1.sh
```

## 非感知型 Sim-to-Sim 测试

用于 `mjlab_repts`，先准备宿主机 Python 环境：

```bash
uv sync
```

在宿主机的三个终端中分别运行模拟器、控制器和手柄程序。

终端 1：

```bash
ROBOT_TYPE=WF_TRON1B RL_TYPE=mjlab_repts uv run python pointfoot-mujoco-sim/simulator.py
```

终端 2：

```bash
ROBOT_TYPE=WF_TRON1B RL_TYPE=mjlab_repts uv run python rl-deploy-with-python/main.py
```

终端 3：

```bash
pointfoot-mujoco-sim/robot-joystick/robot-joystick
```

如需运行 LinVel 变体，将终端 1 和终端 2 中的 `mjlab_repts` 替换为 `mjlab_repts_lin`。

日志写入 `logs/sim2sim/`。
