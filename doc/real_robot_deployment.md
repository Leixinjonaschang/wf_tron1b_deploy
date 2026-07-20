# TRON1 / WF_TRON1B 真机部署

来源：LimX 官方《TRON1 SDK 开发指南》；访问和复核日期：2026-07-20。

原文：[TRON1 SDK 开发指南](https://www.limxdynamics.com/zh/documents/799585387524788224)。本文重点整理其第 3、4、8、9 章中与 Python 低层控制、RL 真机调试和部署有关的内容，并核对本仓库的 `WF_TRON1B` 实现。官方例子多为 `PF_TRON1A` / `isaacgym`；机器人型号、策略类型和模型必须按本项目实际值替换。

## 结论：只复制目录再给 IP，不能视为“现在就能跑”

**有条件地可以部署，但不能仅凭这两件事确认可运行。**

- `rl-deploy-with-python` 自身包含 `WF_TRON1B` 的配置和 `mjlab_repts`、`mjlab_repts_lin`、`mjlab_repts_lin_depth` 的 ONNX 模型；`main.py` 也会为 `WF_TRON1B` 选择 `WheelfootController`。因此，控制代码和模型路径本身具备独立拷到机器人 `/home/guest/...` 的条件。
- 但该目录不是一个自包含的 Python 运行环境：没有独立 `pyproject.toml` 或 requirements 文件，机器人端仍必须具备正确架构、Python/SDK ABI 和 Python 依赖。仅复制文件夹通常会在 `import limxsdk`、`numpy`、`scipy`、`onnxruntime` 或 YAML 读取前后失败。
- 即使进程和 SDK 已经通过 IP 建连，**真机 IP 不会自动启动控制器或让机器人行走**。当前代码在 IP 不是 `127.0.0.1` 时，等待机器人诊断状态为已校零，并等待遥控器 `L1 + △`（代码中的按键名为 `L1 + Y`）才进入控制循环。
- 首次运行前还必须获得开发者模式授权、切入开发者模式、完成校零，并在吊装状态下测试。SDK/本体软件不匹配、错误的 CPU 架构或未就绪的遥控器/诊断数据，都不能由“给 IP”解决。

换言之：**“直接运行”目前不可下结论；“完成下面的目标机预检和吊装调试后运行”是可行路径。**本仓库的本地接口测试全部通过，但这不验证机器人端 SDK、固件、网络或运动安全性。

## 官方真机流程（Python RL）

### 安全和模式前提

1. 首次真机调试和部署测试结束前，机器人必须保持吊装；清空运动范围，现场人员持有遥控器/急停。
2. 开机后按 `R1 + Left` 进入**开发者模式（需授权）**。本体主机会重启，模式会在断电后保持。`R1 + Right` 返回遥控模式并运行预装控制算法。
3. 在任何自定义运控程序前，按 `L1 + R1` 校零，使关节回到初始位置。
4. 控制器进程就绪后，按 `L1 + △` 启动行走，`L1 + □` 关闭行走；左摇杆控制前后和转向，右摇杆控制横移。以现场遥控器标识和机器人软件版本为准。

不要把未验证的自定义策略用于地面、载人、靠近人员或复杂地形。官方文档也要求真机部署期间保持吊装。

### 网络、登录和自启动

- 调试时，开发电脑用外置以太网连接机器人。官方默认示例为电脑 `10.192.1.200`、机器人 `10.192.1.2`，先用 `ping` 确认连通；实际地址以机器人信息页和现场网络配置为准。
- 官方示例将算法目录以 `scp` 拷入机器人 `/home/guest/`，然后在机器人上安装 lowlevel SDK。部署完成后，开发电脑的外接网线不再是运行程序的前提。
- 可在机器人上配置 `/home/guest/autolaunch/autolaunch.sh`，以循环方式启动已验证的 `main.py`。自启动前必须先完成手动吊装调试；否则每次重启都可能自动接管低层控制。
- 不要将机器人账号密码、私钥或现场 IP 写入仓库。使用设备交付凭据并按现场安全要求管理。

## 本仓库的静态核对结果

| 项目 | 结论 | 依据 |
| --- | --- | --- |
| 入口和 IP | 支持 | `main.py` 读取第一个位置参数为 robot IP，调用 `robot.init(robot_ip)`；不读取 `ROBOT_IP` 环境变量。 |
| WF 策略选择 | 支持 | `ROBOT_TYPE=WF_TRON1B` 时选择 `WheelfootController`；三种 `mjlab_repts*` 类型被明确限制为这个机器人型号。 |
| 模型与配置 | 已随目录提供 | 根据 `RL_TYPE` 选取相应 `params_mjlab_repts*.yaml` 和 `controllers/model/WF_TRON1B/policy/<RL_TYPE>/policy.onnx`。 |
| 真机启控 | 需要遥控器事件 | 非 `127.0.0.1` 时 `start_controller=False`；回调只有在诊断码为 `0`（校零完成）且收到 `L1 + △/Y` 时置为真。 |
| 停止行为 | 支持 | 收到 `L1 + □/X` 会退出控制循环，并发布零位置/速度/力矩、`Kp=0`、`Kd=1` 的命令。仍应优先使用实体急停和现场安全流程。 |
| 控制频率 | 500 Hz 命令循环 | `mjlab_repts` 配置为 `loop_frequency: 500`、`decimation: 10`；机器人端 CPU 必须实际能稳定满足。 |

`main.py` 按自身位置解析模型目录，所以从机器人端执行 `python3 /home/guest/<部署目录>/main.py <机器人IP>` 时，不依赖仓库顶层目录。它仍依赖该目录中完整的 `controllers/`、`controllers/model/WF_TRON1B/`、`mjlab_repts.py` 和 `mjlab_repts_lin_depth.py`。

## 机器人端运行环境：必须先确认

官方 Python 部署章节指定 Conda 的 Python 3.8 环境。仓库内可供机器人使用的 SDK 是：

```text
rl-deploy-with-python/limxsdk-lowlevel/python3/aarch64/limxsdk-3.4.0-py3-none-any.whl
```

其二进制扩展是 `ARM aarch64`，并动态依赖 `libpython3.8.so.1.0`。**它不能安装到 x86_64 机器人，也不应假定能在没有 Python 3.8 ABI 的解释器中工作。**仓库顶层的 `pyproject.toml` 则用于开发机，固定的是另一份 `amd64` SDK `3.4.2`；不要把这个 x86_64 wheel 复制到 ARM 本体上。

在真正安装前，登录机器人执行以下只读预检：

```bash
uname -m
python3 --version
getconf LONG_BIT
```

预期 CPU 为 `aarch64`。再确认目标解释器能实际导入依赖：

```bash
python3 -c 'import limxsdk, numpy, yaml, scipy, onnxruntime; print("runtime imports OK")'
python3 -c 'import onnxruntime as ort; print(ort.get_available_providers())'
```

需要的直接运行时依赖至少包括 `limxsdk`、`numpy`、`PyYAML`、`scipy` 和 `onnxruntime`；随附 SDK wheel 的元数据还声明 `mujoco`、`pandas`、`pygame`。如果机器人不能访问包索引，应先在开发机准备与目标 `aarch64` / Python 3.8 匹配的离线 wheel 集，而不是在机器人上临时联网安装。SDK 版本还必须与机器人本体软件兼容；官方文档明确要求不匹配时先替换旧 SDK。

## 推荐的最小部署顺序（不改控制代码）

以下将 `<robot-ip>`、`<deploy-dir>` 和 `<python3.8>` 替换为现场已确认值；不要在未吊装状态下直接执行最后一步。

```bash
# 开发机：在已连通的以太网下，复制完整控制目录。
scp -r rl-deploy-with-python guest@<robot-ip>:/home/guest/<deploy-dir>

# 机器人端：仅当 uname -m 为 aarch64 且 Python 3.8 ABI 已确认时安装匹配的 SDK。
<python3.8> -m pip install \
  /home/guest/<deploy-dir>/limxsdk-lowlevel/python3/aarch64/limxsdk-3.4.0-py3-none-any.whl

# 机器人端：先检查模型文件和运行时导入。
test -f /home/guest/<deploy-dir>/controllers/model/WF_TRON1B/policy/mjlab_repts/policy.onnx
<python3.8> -c 'import limxsdk, numpy, yaml, scipy, onnxruntime; print("OK")'

# 在开发者模式、校零、吊装状态下手动调试；真机自己的 IP 仍应作为参数传入。
cd /home/guest/<deploy-dir>
ROBOT_TYPE=WF_TRON1B RL_TYPE=mjlab_repts \
  <python3.8> main.py <robot-ip>
```

上面最后一条只会使进程订阅 RobotState、IMU、遥控器和诊断数据；它会等待校零完成后的 `L1 + △/Y`，随后先做约 1 秒的站立过渡才进入策略循环。先观察这一过程、确认遥控器停止有效和诊断无异常，才考虑写入 `autolaunch.sh`。

## 自启动（只在手动验证通过后）

官方路径是编辑 `/home/guest/autolaunch/autolaunch.sh` 后重启机器人。建议脚本明确写出解释器、目录和环境变量，避免系统默认 `python3` 指向错误版本：

```bash
#!/bin/bash
while true; do
  export ROBOT_TYPE=WF_TRON1B
  export RL_TYPE=mjlab_repts
  cd /home/guest/<deploy-dir> || exit 1
  /path/to/python3.8 main.py <robot-ip>
  sleep 3
done
```

先保留一条可靠的进入遥控模式/急停路径，再启用该脚本。脚本退出或接到 `L1 + □/X` 后，代码会发出低增益停止命令；这不是实体急停的替代品。

## Depth 策略的附加条件

`RL_TYPE=mjlab_repts_lin_depth` 不能按普通策略“只给 IP”运行。除上述条件外还要求：

- 可用的 ROS 运行环境和 `rospy`、`sensor_msgs`；启动时需要 source 对应 ROS 环境。
- 深度 topic 默认是 `/camera/depth/image_rect_raw`，输入为完整 `480x848` 的 `16UC1` 毫米深度图；控制器裁掉左 128 列并缩放为 ONNX 所需的 `[1, 1, 30, 45]`。
- 深度帧必须持续新鲜。默认缺帧/超时或帧龄超过 `0.5 s` 会抛出 `TimeoutError`，策略不能安全接管。

相机和 ROS 话题验证见 [realsense_depth.md](./realsense_depth.md)。普通的 `mjlab_repts` 不依赖这条 ROS 深度链路。

## 验证范围

本仓库于 2026-07-20 执行：

```bash
uv run --with pytest pytest rl-deploy-with-python/tests
```

结果：`43 passed`。这验证了策略观察量、ONNX 接口和深度预处理的离线一致性；不验证实际机器人本体的 SDK ABI、固件授权、网络、传感器数据率、控制频率或机械安全。

## 官方 ROS C++ 路径（仅供参考）

官方还提供 ROS C++ 部署路线，推荐 Ubuntu 20.04 + ROS Noetic，采用对应的 `tron1-rl-deploy-ros`、`limxsdk-lowlevel` 和硬件 launch 文件。该路径与本仓库维护的 Python / ONNX 入口不同；不要混用其 `pointfoot_hw.launch`、模型目录或环境变量来启动这里的 `mjlab_repts` 策略。
