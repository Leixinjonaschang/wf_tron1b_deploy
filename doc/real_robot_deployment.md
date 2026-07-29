# TRON1 真机部署整理

来源：LimX 官方网页《TRON1 SDK开发指南》，访问日期 2026-07-16。

原文链接：https://www.limxdynamics.com/zh/documents/799585387524788224

本文只整理真机部署相关流程，并按本仓库 `WF_TRON1B` Python 部署场景补充注意事项。官方示例常用 `PF_TRON1A`，在本仓库中应按实际机器人替换为 `WF_TRON1B`。

## 适用范围

- 真机低层运动控制调试：LimX lowlevel SDK 示例或本仓库 Python controller。
- RL 策略部署：将训练得到的 ONNX policy 和配置放到机器人可访问路径后运行。
- 自启动部署：把调试通过的控制程序拷贝到机器人主控电脑，并配置 `/home/guest/autolaunch/autolaunch.sh`。

## 安全前提

- 首次真机测试和部署测试完成前，机器人必须保持吊装。
- 开发者模式、校零和启动行走都应由现场操作者确认安全区域后执行。
- 首次部署可能出现预期外动作，操作者和机器人都应处于安全状态。
- 如果 `limxsdk` 与机器人本体软件版本不匹配，需要在机器人端卸载旧包后重新安装匹配版本。

## 机器人模式和遥控器按键

| 按键 | 作用 |
| --- | --- |
| `R1 + Left` | 切换到开发者模式。切换后本体主机会重启，模式掉电后仍保持。 |
| `R1 + Right` | 切换到遥控模式，运行预安装控制算法。 |
| `L1 + R1` | 校零，使关节回到初始位置。 |
| `L1 + △` | 启动行走功能。 |
| `L1 + □` | 关闭行走功能。 |

左摇杆通常控制前进、后退、左转、右转；右摇杆通常控制横向运动。

## 网络和登录

- 开发电脑通过外置网口连接机器人。
- 开发电脑 IP 设置为 `10.192.1.200`。
- 机器人默认 IP 为 `10.192.1.2`。
- 用 `ping 10.192.1.2` 确认连通。
- SSH/SCP 官方默认用户为 `guest`，默认密码为 `123456`。
- 机器人管理页面通常在 `http://10.192.1.2:8080`。

不要把现场修改后的私有账号、密码、密钥或机器人特定凭据写入仓库。

## 本仓库真机运行方式

调试时可以先在开发电脑上通过机器人 IP 运行 controller：

```bash
ROBOT_TYPE=WF_TRON1B RL_TYPE=mjlab_repts \
  uv run python rl-deploy-with-python/main.py 10.192.1.2
```

如果使用 depth policy，需要先确认相机 depth topic 可用，再运行：

```bash
source /opt/ros/noetic/setup.bash
ROBOT_TYPE=WF_TRON1B \
RL_TYPE=mjlab_repts_lin_depth \
python3 rl-deploy-with-python/main.py 10.192.1.2
```

`mjlab_repts_lin_depth` 默认从对应参数 YAML 读取 ROS1 depth 配置，真机
topic 为 `/camera0/depth/image_rect_raw`。实际 topic 以 `rostopic list`
为准；临时调试仍可通过 `MJLAB_DEPTH_*` 环境变量覆盖 YAML。相机 depth
获取流程见 [realsense_depth.md](./realsense_depth.md)。

## Python 部署到机器人

调试通过后，可以把仓库拷贝到机器人主控电脑：

```bash
scp -r wf_tron1b_deploy guest@10.192.1.2:/home/guest/
```

在机器人端安装或更新 lowlevel SDK wheel。路径按实际拷贝位置和 CPU 架构选择：

```bash
ssh guest@10.192.1.2
pip install /home/guest/wf_tron1b_deploy/pointfoot-mujoco-sim/limxsdk-lowlevel/python3/amd64/limxsdk-*-py3-none-any.whl
```

在机器人端手动运行时：

```bash
cd /home/guest/wf_tron1b_deploy/rl-deploy-with-python
export ROBOT_TYPE=WF_TRON1B
export RL_TYPE=mjlab_repts
python3 main.py 10.192.1.2
```

如果使用 `mjlab_repts_lin_depth`，需要先 source ROS1 环境，再把
`RL_TYPE` 改为 `mjlab_repts_lin_depth`。depth source、topic、量程和超时
默认由 `params_mjlab_repts_lin_depth.yaml` 提供，不需要重复导出对应环境变量。

## 自启动配置

登录机器人并编辑自启动脚本：

```bash
ssh guest@10.192.1.2
busybox vi /home/guest/autolaunch/autolaunch.sh
```

Python controller 自启动模板：

```bash
#!/bin/bash

export ROBOT_TYPE=WF_TRON1B
export RL_TYPE=mjlab_repts

while true; do
  cd /home/guest/wf_tron1b_deploy/rl-deploy-with-python
  python3 main.py 10.192.1.2
  sleep 3
done
```

Depth policy 自启动模板：

```bash
#!/bin/bash

source /opt/ros/noetic/setup.bash

export ROBOT_TYPE=WF_TRON1B
export RL_TYPE=mjlab_repts_lin_depth

while true; do
  cd /home/guest/wf_tron1b_deploy/rl-deploy-with-python
  python3 main.py 10.192.1.2
  sleep 3
done
```

保存后重启机器人：

```bash
reboot
```

系统启动后，用遥控器 `L1 + △` 开启行走，`L1 + □` 关闭行走。

## ROS C++ 部署要点

官方 ROS C++ 部署路径适合使用 `tron1-rl-deploy-ros`：

- 开发环境推荐 Ubuntu 20.04 + ROS Noetic。
- 依赖包括 ROS control/Gazebo/PlotJuggler/joy、CMake、Eigen、OpenCV、Boost、TBB、URDFDOM、Orocos KDL 等。
- ONNX Runtime 官方示例使用 `v1.10.0`，需要把 include 和 lib 安装到系统路径。
- 工作空间通常包含 `limxsdk-lowlevel`、`tron1-gazebo-ros`、`robot-description`、`robot-visualization` 和 `tron1-rl-deploy-ros`。
- 模型和配置替换位置为 `tron1-rl-deploy-ros/robot_controllers/config/pointfoot/<ROBOT_TYPE>/`。
- 真机调试前修改 `pointfoot_hw.launch` 中的机器人 IP 为 `10.192.1.2`，编译后运行 `roslaunch robot_hw pointfoot_hw.launch`。

本仓库主要维护 Python 部署路径；ROS C++ 路径只作为官方部署参考。
