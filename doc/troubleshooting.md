# Troubleshooting

本文只处理当前推荐组合：`WF_TRON1B + mjlab_repts_gru_lin_depth`。历史策略问题见
[Legacy Compatibility](legacy/README.md)。

## Docker 与 GPU

### Docker API permission denied

使用 `sudo docker ...`，或按系统规范把当前用户加入 Docker group。确认 daemon 正常：

```bash
sudo docker info
```

### `failed to discover GPU vendor from CDI`

先确认宿主机 `nvidia-smi` 正常，再重新配置 NVIDIA Container Toolkit：

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
sudo docker rm -f tron_deploy
sudo -E scripts/docker_start_ros1.sh
```

删除容器不会删除仓库；重新创建容器前应先停止正在运行的 Sim-to-Sim。

### `glx: failed to create dri3 screen` 或没有 GUI

检查宿主机 `DISPLAY`、X11 socket 和 renderer：

```bash
echo "$DISPLAY"
glxinfo -B | grep -E 'OpenGL vendor|OpenGL renderer'
xhost +local:docker
```

若只是不需要 depth viewer，可在容器内使用：

```bash
MJLAB_DEPTH_VIEW=0 /work/scripts/docker_run_sim2sim_ros1.sh
```

这不会关闭 simulator 到 controller 的 ROS depth 传输。

混合显卡宿主机需要 NVIDIA PRIME Render Offload 时，可在启动容器前确认
`nvidia-smi`，并根据宿主机驱动配置 `__NV_PRIME_RENDER_OFFLOAD=1` 与
`__GLX_VENDOR_LIBRARY_NAME=nvidia`。若 `nvidia-smi` 本身失败，应先修复驱动。

## Ubuntu 20.04 本地 renderer

本地 Sim-to-Sim 直接使用宿主机 OpenGL。检查当前 renderer：

```bash
sudo apt install mesa-utils
glxinfo -B | grep -E 'OpenGL vendor|OpenGL renderer'
```

`AMD Radeon`、`Mesa Intel` 或具体 NVIDIA 型号表示正在使用对应 GPU；`llvmpipe` 或
`softpipe` 表示 CPU 软件渲染。混合显卡笔记本需要临时使用 NVIDIA PRIME 时：

```bash
__NV_PRIME_RENDER_OFFLOAD=1 \
__GLX_VENDOR_LIBRARY_NAME=nvidia \
ROS_TYPE=ros1 \
ROBOT_TYPE=WF_TRON1B \
RL_TYPE=mjlab_repts_gru_lin_depth \
scripts/start_sim2sim.sh
```

## ROS1

### `Unable to register with master node`

确认所有终端使用同一个 ROS master：

```bash
echo "$ROS_MASTER_URI"
rostopic list
```

Sim-to-Sim 默认由启动器创建本地 master。真机默认使用：

```bash
export ROS_MASTER_URI=http://10.192.1.2:11311
export ROS_IP=10.192.1.200
unset ROS_HOSTNAME
```

不要让本地 Sim-to-Sim 进程继承真机的 `ROS_IP=10.192.1.200`。

### Python 无法导入 `rospy` 或 `sensor_msgs`

先 source ROS1，并确认启动 controller 的同一个 Python 可以导入模块：

```bash
source /opt/ros/noetic/setup.bash
python3 -c 'import rospy; import sensor_msgs.msg; print("ROS1 Python OK")'
```

不要在一个 Python 环境中检查 ROS，却在另一个环境中启动 controller。

## Depth

### `timed out waiting for ROS depth image`

依次检查 topic 名称、类型、publisher 和帧率：

```bash
rostopic list | grep depth
rostopic type /camera/depth/image_rect_raw
rostopic info /camera/depth/image_rect_raw
rostopic hz /camera/depth/image_rect_raw
```

Sim-to-Sim 默认 topic 为 `/camera/depth/image_rect_raw`；真机默认 topic 为
`/camera0/depth/image_rect_raw`。simulator、controller 和 viewer 必须使用同一个 topic。

### stale depth frame

当前最大允许帧龄为 `0.5 s`。检查相机帧率、ROS 网络延迟、publisher 是否卡死，以及
系统时钟是否异常。不要简单增大 `MJLAB_DEPTH_MAX_AGE` 来掩盖持续丢帧。

### depth shape 或二次裁剪错误

controller 需要完整 D435 raw frame `480×848`。部署端会先以最近邻 resize 到
`30×53`，再删除底部 10 行和左侧 8 列得到 `20×45`；不要发布已裁剪的
policy-sized depth。接口必须最终得到 float32 `[1,1,20,45]`。

### depth 全零、全远或尺度异常

- `16UC1`/`mono16` 按毫米处理，必须乘 `0.001` 转米；
- `32FC1` 应已经是米；
- NaN、±Inf、负数、0 和低于 `0.15 m` 的值会变为 `2.5 m`；
- 部署端裁剪到 `[0.15, 2.5] m`，不做归一化，ONNX 直接消费米制 depth。

先用 `rqt_image_view` 或仓库 depth viewer 检查原始图像，再检查 controller 日志中的
depth min/max/mean。

## ONNX 与控制

### ONNX interface mismatch

当前策略名称和 shape 必须与根目录 README 的 Policy Contract 完全一致。确认：

```text
inputs:  proprio_history, actor_command, depth, hidden_state_in
outputs: actions, predicted_lin_vel, hidden_state_out
```

不要重命名 tensor，也不要将旧 `[1,64]` hidden state policy 放入当前策略目录。

### 机器人动作方向或左右侧不正确

立即停止并重新核对 action metadata。Policy action 顺序与 LimX SDK 的交错关节顺序不同，
必须通过 controller 的按名称映射处理，不得直接按下标复制。

## 日志与最小诊断

Sim-to-Sim 日志位于：

```text
logs/sim2sim/
```

优先查看：

```bash
tail -n 100 logs/sim2sim/simulator.log
tail -n 100 logs/sim2sim/controller.log
tail -n 100 logs/sim2sim/roscore.log
```

仍无法定位时，记录使用的 commit、完整启动命令、ROS topic/type/hz、controller 的 ONNX
input/output 日志以及首次异常堆栈。
