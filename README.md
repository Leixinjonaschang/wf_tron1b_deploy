# WF_TRON1B Sim2sim

本仓库常用两种 sim2sim 流程：

- Perceptive depth policy：Docker + ROS1 depth topic。
- Non-perceptive REPTS policy：宿主机 `uv` 环境直接运行，不需要 ROS。

## Depth-Based Perceptive Sim-to-Sim Test

用于 depth-based policy：`mjlab_repts_lin_depth`。

在宿主机执行, 根据 Dockerfile 构建 docker 镜像：

```bash
cd /media/phi/641A24011A23CF3C/ubuntu_data/CLX/project/wheeled_legged_proj/wf_tron1b_deploy
sudo -E IMAGE_NAME=tron_sim2sim:latest scripts/docker_build_ros1.sh
```


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

## Non-Perceptive Sim-to-Sim Test

用于 non-depth REPTS policy：`mjlab_repts`。
这个流程直接在宿主机 `uv` 环境运行，不需要 ROS。

准备宿主机 Python 环境：

```bash
uv sync
```


手动分终端启动：

```bash
ROBOT_TYPE=WF_TRON1B uv run python pointfoot-mujoco-sim/simulator.py 127.0.0.1
```

```bash
ROBOT_TYPE=WF_TRON1B RL_TYPE=mjlab_repts uv run python rl-deploy-with-python/main.py 127.0.0.1
```

```bash
pointfoot-mujoco-sim/robot-joystick/robot-joystick
```

日志：

```text
logs/sim2sim/
```
