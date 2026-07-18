# RealSense D435i Depth 获取

来源：LimX 官方网页《TRON1 SDK开发指南》第 9 章 RealSense 相机，访问日期 2026-07-16。

原文链接：https://www.limxdynamics.com/zh/documents/799585387524788224

TRON1 机身安装 RealSense D435i，相机可提供 RGB、depth 和点云数据，官方说明主要面向 ROS Noetic 访问机器人端相机数据。本文单独记录 depth 获取流程，并补充本仓库 `mjlab_repts_lin_depth` 的对接方式。

## 前提

- 开发电脑安装 ROS Noetic，建议 `ros-noetic-desktop-full`。
- 开发电脑通过外置网口连接机器人。
- 开发电脑 IP 为 `10.192.1.200`。
- 机器人 IP 为 `10.192.1.2`。
- `ping 10.192.1.2` 可连通。

## 打开机器人相机

如果相机未启动：

1. 打开机器人管理页面 `http://10.192.1.2:8080`。
2. 进入机器人信息页。
3. 点击打开 D435i 相机。

## 配置 ROS 网络

在开发电脑终端中配置 ROS master 和本机 ROS IP：

```bash
source /opt/ros/noetic/setup.bash
export ROS_MASTER_URI=http://10.192.1.2:11311
export ROS_IP=10.192.1.200
```

可将上述变量写入开发电脑 `.bashrc`，但要确认只在连接机器人网络时启用，避免影响其他 ROS 环境。

## 查找 depth topic

列出机器人发布的相机 topic：

```bash
rostopic list
```

常见 depth topic 可能类似：

```text
/camera/depth/image_rect_raw
/camera/depth/color/points
```

以现场 `rostopic list` 输出为准。本仓库 sim2sim 默认 depth image topic 是 `/camera/depth/image_rect_raw`，真机 topic 如果不同，需要设置 `MJLAB_DEPTH_ROS_TOPIC`。

查看 depth image 消息类型：

```bash
rostopic type /camera/depth/image_rect_raw
```

期望用于 policy 的输入是 `sensor_msgs/Image`。本仓库 `mjlab_repts_lin_depth` 支持 `16UC1`、`mono16`、`32FC1` 和 `passthrough`，其中 `16UC1`/`mono16` 默认按毫米转米，scale 为 `0.001`。

## 可视化检查

安装并打开 `rqt_image_view`：

```bash
sudo apt install ros-noetic-rqt-image-view
rqt_image_view
```

在界面中选择 depth image topic 检查图像是否连续更新。

也可以使用 RViz：

```bash
rviz
```

在 RViz 中添加 `Image` 或 `PointCloud2` display，并选择相机相关 topic。

## 对接本仓库 depth policy

真机使用相机 depth 时，controller 侧只需要订阅相机 topic，不需要 simulator 的 `MJLAB_DEPTH_SINK`。

示例：

```bash
source /opt/ros/noetic/setup.bash
export ROS_MASTER_URI=http://10.192.1.2:11311
export ROS_IP=10.192.1.200

export ROS_TYPE=ros1
export ROBOT_TYPE=WF_TRON1B
export RL_TYPE=mjlab_repts_lin_depth
export MJLAB_DEPTH_SOURCE=ros
export MJLAB_DEPTH_ROS_TOPIC=/camera/depth/image_rect_raw
export MJLAB_DEPTH_MAX_AGE=0.5

python3 rl-deploy-with-python/main.py 10.192.1.2
```

`mjlab_repts_lin_depth` 的处理链路：

1. `RosDepthFrameSource` 订阅 `MJLAB_DEPTH_ROS_TOPIC`。
2. ROS callback 将 `sensor_msgs/Image` 转为 numpy depth。
3. `16UC1`/`mono16` 从毫米转米，`32FC1` 保持米单位。
4. depth 被裁剪到 `MJLAB_DEPTH_MIN` 和 `MJLAB_DEPTH_MAX`，默认 `0.0` 到 `10.0`。
5. 完整 D435 raw frame `480x848` 左裁 128 列，得到 `480x720`。
6. 左裁后的图像以最近邻 resize 到 ONNX 输入 `[1, 1, 30, 45]`。
7. `WheelfootController.compute_actions()` 在 policy loop 中读取最新帧并输入 ONNX。

sim2sim 的 `d435` producer 默认发布完整 `480x848` 米深度、频率 `30 Hz`。真机和 sim2sim 都必须将完整 FOV raw frame 交给 source；不要预先传入已经裁成 `30x45` 的图像，避免重复左裁。部署前必须导出输入 depth 为 `[1, 1, 30, 45]` 的新 ONNX；旧 `[1, 1, 28, 48]` ONNX 会被接口校验拒绝。

如果 ROS topic 间隔超过 `MJLAB_DEPTH_MAX_AGE`，controller 会抛出 stale frame `TimeoutError`，这是为了避免策略使用过期 depth。

## 排查清单

- `ping 10.192.1.2` 是否通。
- `ROS_MASTER_URI` 是否指向 `http://10.192.1.2:11311`。
- `ROS_IP` 是否为开发电脑在机器人网络中的 IP。
- 管理页面中 D435i 是否已经打开。
- `rostopic list` 是否能看到相机 topic。
- `rostopic hz <depth_topic>` 是否有稳定频率。
- `rqt_image_view` 或 RViz 是否能显示 depth。
- `MJLAB_DEPTH_ROS_TOPIC` 是否与真实 depth image topic 一致。
- Python 环境是否能 import `rospy` 和 `sensor_msgs.msg`。
