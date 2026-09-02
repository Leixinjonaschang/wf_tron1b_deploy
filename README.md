# WF-TRON1B Subject Locomotion Deployment

## 1. 项目介绍

本项目部署面向 WF-TRON1B 的轮足 Subject Locomotion 方法。仓库包含策略模型、
RealSense depth 接入、ONNX Runtime 推理、LimX SDK 控制、MuJoCo 仿真以及部署验证工具。

当前唯一推荐策略为：

```text
ROBOT_TYPE=WF_TRON1B
RL_TYPE=mjlab_repts_gru_lin_depth
```

新策略必须先通过 Sim-to-Sim，确认 observation、depth、action mapping 和 GRU hidden
state 均符合训练时 contract，之后才能进入 Sim-to-Real。首次真机测试必须吊装，现场必须有
可立即停止机器人的操作者。

## 2. 部署流程

```text
Training Export
    → Sim-to-Sim
    → 验证 observation / depth / action / hidden state
    → Sim-to-Real
```

Sim-to-Sim 不是展示步骤，而是部署门禁：它用于尽早发现 ONNX 接口、关节顺序、depth
单位、图像裁剪和 recurrent state 不一致。第 3 节全部检查通过前，不要连接真机执行策略。

## 3. Sim-to-Sim

### 3.1 根据宿主机选择运行方式

两种方式运行相同的 simulator、controller、ROS1 depth transport 和 ONNX policy；区别
只在于运行环境：

| 宿主机 | Sim-to-Sim 方式 | 原因 |
| --- | --- | --- |
| Ubuntu 20.04 | 本地原生运行 | 系统原生支持 ROS1 Noetic |
| Ubuntu 22.04 | Docker 运行 | 使用容器中的 RoboStack Noetic + Python 3.11 解决环境兼容问题 |

Ubuntu 22.04 上不建议把 ROS1 Noetic 强行安装到宿主机。Docker 正是为该场景提供的兼容
环境；宿主机不需要安装 ROS。两种方式最终必须通过第 3.4 节的同一套检查。

### 3.2 Ubuntu 20.04：本地 Sim-to-Sim

本地方式让 MuJoCo、ROS1、controller、depth viewer 和 joystick 全部直接运行在宿主机。
宿主机需要 ROS1 Noetic、`uv`、图形桌面和可用的 OpenGL renderer。

首次运行，在仓库根目录准备 Python 与 ROS 环境：

```bash
uv sync
source /opt/ros/noetic/setup.bash

export ROS_MASTER_URI=http://127.0.0.1:11311
export ROS_IP=127.0.0.1
unset ROS_HOSTNAME

uv run python -c 'import rospy, sensor_msgs.msg, mujoco, onnxruntime, limxsdk; print("local runtime OK")'
```

最后一条命令必须成功；否则说明当前 `uv` Python 看不到 ROS1 packages，应先修复环境。
不要让本地 Sim-to-Sim 继承真机使用的 `ROS_MASTER_URI` 或 `ROS_IP=10.192.1.200`。

在同一终端执行本地一键启动命令：

```bash
ROS_TYPE=ros1 \
ROBOT_TYPE=WF_TRON1B \
RL_TYPE=mjlab_repts_gru_lin_depth \
scripts/start_sim2sim.sh
```

启动器会在需要时创建本地 ROS master，然后启动 simulator、controller、可选 depth
viewer 和虚拟遥控器。按虚拟遥控器终端中的 `Ctrl-C` 会停止全部子进程。renderer 与
NVIDIA PRIME 排查见 [Troubleshooting](doc/troubleshooting.md)。

### 3.3 Ubuntu 22.04：Docker Sim-to-Sim

Docker 镜像使用 RoboStack Noetic + Python 3.11，让 ROS1 与 MuJoCo、ONNX Runtime 和
当前 Python 代码处于同一个兼容环境。宿主机需要 Docker 和 X11；启动脚本默认请求
NVIDIA GPU，因此 NVIDIA 主机还需要 NVIDIA Container Toolkit。首次使用时在仓库根目录执行：

```bash
sudo -E scripts/docker_build_ros1.sh
xhost +local:docker
sudo -E scripts/docker_start_ros1.sh
sudo docker exec -it tron_deploy bash
```

进入 `tron_deploy` 容器后，一键启动 Sim-to-Sim：

```bash
/work/scripts/docker_run_sim2sim_ros1.sh
```

按运行虚拟遥控器的终端中的 `Ctrl-C`，启动器会停止本次 Sim-to-Sim 的全部子进程。
Docker、GPU、DISPLAY 或 ROS 问题见 [Troubleshooting](doc/troubleshooting.md)。

### 3.4 两种方式共同的检查项

在另一个终端检查同一个 Sim-to-Sim topic。Docker 方式应进入同一个容器；本地方式应先
source ROS1，并保持第 3.2 节的本地 ROS 网络变量：

```bash
rostopic type /camera/depth/image_rect_raw
rostopic hz /camera/depth/image_rect_raw
```

验证 ROS1 publish/subscribe 和 policy-side depth preprocessing：

```bash
# Ubuntu 20.04 本地方式
MJLAB_DEPTH_ROS_TOPIC=/wf_depth_smoke uv run python scripts/ros1_depth_smoke.py

# Ubuntu 22.04 Docker 方式（在容器内）
MJLAB_DEPTH_ROS_TOPIC=/wf_depth_smoke python scripts/ros1_depth_smoke.py
```

进入 Sim-to-Real 前，必须同时满足：

- depth topic 类型为 `sensor_msgs/Image`，帧率稳定在约 `30 Hz`；
- smoke test 输出 `ROS1 depth smoke passed`，预处理结果 shape 为 `[1, 1, 30, 45]`；
- depth viewer 中图像连续更新，不冻结、不长期全零；
- controller 启动日志显示正确的 input/output shape、action order 和 ROS depth source；
- 日志中没有 missing/stale depth、NaN/Inf、ONNX interface mismatch 或进程提前退出；
- 平地站立和运动稳定，指令方向、左右关节和轮速方向正确；
- 下列回归测试全部通过：

```bash
uv run --with pytest pytest rl-deploy-with-python/tests
```

详细 Docker/ROS 操作见 [ROS1 Depth Sim-to-Sim](doc/ros1_depth_sim2sim.md)。

## 4. Sim-to-Real

### 4.1 安全检查

- 首次部署、策略更新或 contract 变化后，机器人必须保持吊装；
- 确认现场安全区域、急停方式和停止按键，安排专人操作遥控器；
- 确认机器人处于开发者模式并已完成校零；
- 确认所用 LimX SDK 与机器人本体软件版本匹配；
- 确认第 3.4 节所有 Sim-to-Sim 检查已通过；
- 未确认 depth、关节顺序和 action scale 前，不得启动行走。

### 4.2 RealSense 与 ROS topic 检查

开发电脑默认网络配置为 `10.192.1.200`，机器人默认地址为 `10.192.1.2`：

```bash
ping 10.192.1.2
source /opt/ros/noetic/setup.bash
export ROS_MASTER_URI=http://10.192.1.2:11311
export ROS_IP=10.192.1.200
unset ROS_HOSTNAME

rostopic type /camera0/depth/image_rect_raw
rostopic hz /camera0/depth/image_rect_raw
```

必须确认真实 topic 的消息类型为 `sensor_msgs/Image`、帧率稳定、画面有效。现场 topic
如有不同，以 `rostopic list` 为准，并通过 `MJLAB_DEPTH_ROS_TOPIC` 显式覆盖。完整相机检查见
[RealSense Depth](doc/realsense_depth.md)。

### 4.3 唯一推荐启动命令

在已经能够导入 `rospy`、`sensor_msgs`、`onnxruntime` 和 `limxsdk` 的 ROS1 Python
环境中，从仓库根目录运行：

```bash
source /opt/ros/noetic/setup.bash
export ROS_MASTER_URI=http://10.192.1.2:11311
export ROS_IP=10.192.1.200
unset ROS_HOSTNAME

ROBOT_TYPE=WF_TRON1B \
RL_TYPE=mjlab_repts_gru_lin_depth \
MJLAB_DEPTH_SOURCE=ros \
MJLAB_DEPTH_ROS_TOPIC=/camera0/depth/image_rect_raw \
python3 rl-deploy-with-python/main.py 10.192.1.2
```

controller 连接真机后不会自动开始行走。确认吊装和校零状态后，使用遥控器
`L1 + △`（代码中的 `L1 + Y`）启动策略。

### 4.4 停止和恢复

- 正常停止：先按 `L1 + □`（代码中的 `L1 + X`），等待 controller 下发安全停止命令；
- 结束进程：机器人停止后再按 `Ctrl-C`；
- 异常动作、depth 超时或通信异常：立即使用现场急停/停止方式，不要依赖终端操作；
- 恢复：排除故障、重新吊装并重新完成 topic 与校零检查，然后重新启动 controller；
- 不要用循环自启动掩盖持续崩溃或反复 depth timeout。

真机安装、网络、SDK、自启动和安全细节见
[Sim-to-Real Deployment](doc/real_robot_deployment.md)。

## 5. Policy Contract

当前 contract 对应：
[policy.onnx](rl-deploy-with-python/controllers/model/WF_TRON1B/policy/mjlab_repts_gru_lin_depth/policy.onnx)。
controller 会在启动时校验名称、shape 和 metadata；不兼容的 ONNX 会直接被拒绝。

### 5.1 ONNX 输入输出

| 方向 | 名称 | Shape | 含义 |
| --- | --- | --- | --- |
| Input | `proprio_history` | `[1, 5, 28]` | 5 帧本体感知历史，oldest-to-newest |
| Input | `actor_command` | `[1, 3]` | 机体速度指令 |
| Input | `depth` | `[1, 1, 30, 45]` | float32 米制 depth |
| Input | `hidden_state_in` | `[1, 128]` | GRU 上一时刻状态 |
| Output | `actions` | `[1, 8]` | 6 个腿关节位置 action + 2 个轮速 action |
| Output | `predicted_lin_vel` | `[1, 3]` | 预测机体线速度 |
| Output | `hidden_state_out` | `[1, 128]` | 传给下一 policy step 的 GRU 状态 |

`proprio_history` 每帧 28 维，顺序为：`base_ang_vel(3)`、
`projected_gravity(3)`、`joint_pos(6)`、`joint_vel(6)`、`wheel_vel(2)`、
`actions(8)`。

### 5.2 Depth contract

```text
ROS sensor_msgs/Image
    → 16UC1/mono16 × 0.001 转为 meters（32FC1 保持 meters）
    → 完整 FOV nearest-neighbor resize 到 30×53
    → 左裁 8 列，得到 30×45
    → finite 且 >= 0.15 m 的值有效，其余映射为 2.5 m
    → clamp [0.15, 2.5] m（不归一化）
    → float32 [1, 1, 30, 45]
    → ONNX 直接消费外部预处理结果
```

必须传入完整 FOV raw depth。不要在相机端预先裁成 `30×45`，否则会造成重复裁剪。
ROS source 使用接收时的 monotonic clock 判断帧龄，超过 `0.5 s` 的旧帧会被拒绝；
30 Hz 的最新 depth 可由多个 50 Hz policy step 复用。NPY live fallback 同样拒绝长期未更新的文件。

### 5.3 Action 与 recurrent state

Policy action 顺序固定为：

```text
abad_L, hip_L, knee_L, abad_R, hip_R, knee_R, wheel_L, wheel_R
```

- 前 6 维：腿关节位置 action，scale 为 `0.5`；
- 后 2 维：轮关节速度 action，scale 为 `10.0`；
- action clip 为 `[-2.0, 2.0]`；
- controller 按名称映射到 LimX SDK 的交错关节顺序，禁止按数组位置自行重排；
- `hidden_state_in` 首帧初始化为零，此后必须把 `hidden_state_out` 原样传入下一步；
- controller 启动、`STAND → WALK` 首次推理前以及 controller 停止后，会统一清空
  hidden state、5 帧 history、last action、当前 action 和预测速度；
- controller loop 为 `500 Hz`、decimation 为 `10`，policy 更新频率为 `50 Hz`。

若 ONNX metadata 存在，controller 会校验米制 depth、`[0.15, 2.5] m`、external
preprocessing、关节/action 顺序与 scale；metadata 为空时使用上述本地锁定契约并打印提示。

## 6. Repository Structure

```text
wf_tron1b_deploy/
├── rl-deploy-with-python/       # ONNX controller、policy contract 与测试
├── pointfoot-mujoco-sim/        # MuJoCo simulator、WF_TRON1B 模型与虚拟遥控器
├── scripts/                     # Docker/ROS1 启动、depth viewer 与 smoke check
├── utils/terrain_tool/          # 离线地形场景生成器
├── doc/                         # Sim-to-Sim、RealSense、真机和故障排查文档
├── Dockerfile                   # 推荐 ROS1 Sim-to-Sim 环境
└── pyproject.toml               # 宿主机 Python 依赖
```

## 7. Detailed Documentation

- [Ubuntu 22.04 ROS1 Depth Sim-to-Sim / Docker](doc/ros1_depth_sim2sim.md)
- [RealSense D435i Depth 与 ROS topic](doc/realsense_depth.md)
- [Sim-to-Real 真机部署](doc/real_robot_deployment.md)
- [Troubleshooting](doc/troubleshooting.md)
- [Legacy Compatibility](doc/legacy/README.md)

## 8. Legacy Compatibility

以下策略只用于历史复现或兼容性验证，不属于当前推荐部署主线：

| `RL_TYPE` | 感知输入 | 当前状态 | 说明 |
| --- | --- | --- | --- |
| `mjlab_repts_lin` | proprioception | 兼容保留 | 无 depth 的 LinVel policy |
| `isaacgym` | proprioception | 兼容保留 | 历史 policy + encoder 接口 |
| `isaaclab` | proprioception | 兼容保留 | 历史 policy + encoder 接口 |
| `mjlab_repts` | proprioception | 已退出当前入口 | 仅通过 Git 历史追溯 |
| `mjlab_repts_lin_depth` | depth + proprioception | 已退出当前入口 | 旧 `[1,64]` recurrent policy |

历史命令、差异和使用边界统一见 [doc/legacy/README.md](doc/legacy/README.md)。不要把
legacy policy 的参数、预处理或启动命令与 `mjlab_repts_gru_lin_depth` 混用。
