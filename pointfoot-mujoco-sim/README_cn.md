# 中文 | [English](README.md)
# tron1-mujoco-sim

## 1. 运行仿真

- 打开一个 Bash 终端。

- 下载 MuJoCo 仿真器代码：

  ```
  git clone --recurse https://github.com/limxdynamics/tron1-mujoco-sim.git
  ```

- 安装运动控制开发库：

  - Linux x86_64 环境

    ```
    pip install tron1-mujoco-sim/limxsdk-lowlevel/python3/amd64/limxsdk-*-py3-none-any.whl
    ```

  - Linux aarch64 环境

    ```
    pip install tron1-mujoco-sim/limxsdk-lowlevel/python3/aarch64/limxsdk-*-py3-none-any.whl
    ```

- 设置机器人类型

  - 通过 Shell 命令 `tree -L 1 tron1-mujoco-sim/robot-description/pointfoot` 列出可用的机器人类型：

    ```
    limx@limx:~$ tree -L 1 tron1-mujoco-sim/robot-description/pointfoot
    tron1-mujoco-sim/robot-description/pointfoot
    └── WF_TRON1B

    ```

  - 设置本项目支持的机器人型号：

    ```
    echo 'export ROBOT_TYPE=WF_TRON1B' >> ~/.bashrc && source ~/.bashrc
    ```

- 运行 MuJoCo 仿真器：

  ```
  python tron1-mujoco-sim/simulator.py
  ```

## 2. 编译 SDK

- 打开一个 Bash 终端。

- 安装编译所需环境

  ```
  sudo apt update
  sudo apt install -y cmake build-essential
  ```

- 编译 SDK：

  ```
  cd tron1-mujoco-sim/limxsdk-lowlevel
  mkdir -p build
  cd build
  cmake ..
  make
  ```

## 3. 仿真展示

![](doc/simulator.gif)
