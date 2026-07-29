# WF_TRON1B Sim2sim

本仓库常用两种 sim2sim 流程：

- Perceptive depth policy：Docker + ROS1 depth topic。
- Non-perceptive REPTS policy：宿主机 `uv` 环境直接运行，不需要 ROS。

## Depth-Based Perceptive Sim-to-Sim Test

支持两种 depth-based policy：

- `mjlab_repts_lin_depth`：原 student encoder，GRU hidden state 为 `[1, 64]`。
- `mjlab_repts_gru_lin_depth`：新 student encoder，GRU hidden state 为 `[1, 128]`。

下列命令默认展示原策略；运行新策略时，将 simulator 和 controller 命令中的
`RL_TYPE` 同时改为 `mjlab_repts_gru_lin_depth`。两种策略共用相同的 depth
topic、裁剪、缩放和 `MJLAB_DEPTH_*` 配置。

### 宿主机本地 ROS1 sim2sim（手动启动）

以下流程在宿主机上直接运行 MuJoCo、controller 和 ROS1 depth
transport。首先在仓库根目录同步 Python 环境：

```bash
uv sync
```

#### 检查和选择宿主机 GPU renderer

MuJoCo 的物理仿真仍在 CPU 上运行，窗口和 depth image 使用当前桌面会话的
OpenGL renderer。renderer 由启动终端的环境变量选择，不需要修改 Python
代码。

安装检查工具并查看当前 renderer：

```bash
sudo apt install mesa-utils
glxinfo -B | grep -E 'OpenGL vendor|OpenGL renderer'
```

常见结果：

- `AMD RENOIR`、`Mesa Intel` 或 `AMD Radeon`：使用对应的 AMD/Intel GPU。
- `NVIDIA GeForce ...`：使用 NVIDIA GPU。
- `llvmpipe` 或 `softpipe`：使用 CPU 软件渲染。

本机显示 `AMD RENOIR` 时已经是 AMD 核显 GPU 渲染，并非 CPU 渲染。如果
AMD 核显性能足够，可以直接按后续命令运行。

对于 AMD/Intel 核显加 NVIDIA 独显的笔记本，先确认 NVIDIA 驱动正常：

```bash
nvidia-smi
```

需要临时让整个一键 sim2sim 流程使用 NVIDIA PRIME Render Offload 时，在
同一条启动命令前加环境变量：

```bash
__NV_PRIME_RENDER_OFFLOAD=1 \
__GLX_VENDOR_LIBRARY_NAME=nvidia \
ROBOT_TYPE=WF_TRON1B \
RL_TYPE=mjlab_repts_gru_lin_depth \
scripts/start_sim2sim.sh
```

如果手动启动各进程，只需给 MuJoCo simulator 命令添加相同前缀：

```bash
__NV_PRIME_RENDER_OFFLOAD=1 \
__GLX_VENDOR_LIBRARY_NAME=nvidia \
ROBOT_TYPE=WF_TRON1B \
RL_TYPE=mjlab_repts_gru_lin_depth \
MJLAB_DEPTH_SINK=ros \
uv run python pointfoot-mujoco-sim/simulator.py
```

启动前可验证 PRIME Offload 是否会选择 NVIDIA：

```bash
__NV_PRIME_RENDER_OFFLOAD=1 \
__GLX_VENDOR_LIBRARY_NAME=nvidia \
glxinfo -B | grep -E 'OpenGL vendor|OpenGL renderer'
```

预期 vendor 为 `NVIDIA Corporation`，renderer 为具体的 NVIDIA GPU。
上述变量仅对当前命令生效，不会永久修改系统；不添加前缀即可恢复默认的
AMD renderer。若 `nvidia-smi` 本身报错，应先修复 NVIDIA 驱动，再尝试
PRIME Offload。

下面每一步均在独立 Bash 终端中执行，并保持前面启动的进程持续运行。
所有 ROS 节点必须使用同一个本地 master、本机地址和 depth topic。

1. 启动本地 ROS master：

```bash
cd ~/CLX/wf_tron1b_deploy
source /opt/ros/noetic/setup.bash

env -u ROS_HOSTNAME \
  ROS_MASTER_URI=http://127.0.0.1:11311 \
  ROS_IP=127.0.0.1 \
  roscore
```

2. 启动 MuJoCo simulator 并发布 depth image：

```bash
cd ~/CLX/wf_tron1b_deploy
source /opt/ros/noetic/setup.bash

env -u ROS_HOSTNAME \
  ROS_MASTER_URI=http://127.0.0.1:11311 \
  ROS_IP=127.0.0.1 \
  ROBOT_TYPE=WF_TRON1B \
  RL_TYPE=mjlab_repts_lin_depth \
  MJLAB_DEPTH_SINK=ros \
  MJLAB_DEPTH_ROS_TOPIC=/camera/depth/image_rect_raw \
  uv run python pointfoot-mujoco-sim/simulator.py
```

simulator 终端应显示 `sink=ros` 和
`ros_topic=/camera/depth/image_rect_raw`。

3. 确认 depth publisher 地址和帧率：

```bash
cd ~/CLX/wf_tron1b_deploy
source /opt/ros/noetic/setup.bash
export ROS_MASTER_URI=http://127.0.0.1:11311
export ROS_IP=127.0.0.1
unset ROS_HOSTNAME

rostopic info /camera/depth/image_rect_raw
rostopic hz /camera/depth/image_rect_raw
```

`rostopic info` 中的 publisher URI 应为 `http://127.0.0.1:<port>/`，
帧率应接近 `30 Hz`。如果 publisher URI 仍为 `10.192.1.200`，说明
simulator 启动时继承了真机调试用的 `ROS_IP`，需要停止后按第 2 步重启。

4. depth topic 正常后启动 RL controller：

```bash
cd ~/CLX/wf_tron1b_deploy
source /opt/ros/noetic/setup.bash

env -u ROS_HOSTNAME \
  ROS_MASTER_URI=http://127.0.0.1:11311 \
  ROS_IP=127.0.0.1 \
  ROBOT_TYPE=WF_TRON1B \
  RL_TYPE=mjlab_repts_lin_depth \
  MJLAB_DEPTH_SOURCE=ros \
  MJLAB_DEPTH_ROS_TOPIC=/camera/depth/image_rect_raw \
  uv run python rl-deploy-with-python/main.py
```

5. 启动虚拟遥控器：

```bash
cd ~/CLX/wf_tron1b_deploy
pointfoot-mujoco-sim/robot-joystick/robot-joystick
```

如果 controller 报 `timed out waiting for ROS depth image`，先重新执行第 3 步，
确认 simulator 仍在运行、publisher URI 可访问，且 simulator 与 controller
使用完全相同的 topic。

### Docker ROS1 sim2sim

在宿主机执行, 根据 Dockerfile 构建 docker 镜像：

```bash
cd wf_tron1b_deploy
sudo -E IMAGE_NAME=tron_sim2sim:latest scripts/docker_build_ros1.sh
```

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
