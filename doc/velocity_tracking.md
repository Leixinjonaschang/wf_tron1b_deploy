# 平地多方向速度跟踪实验

分三步运行：**采集 → 校验与汇总 → 作图**。`collect` 不生成图；`plot`
要求完整采集和有效汇总，且会核对所有原始文件的 SHA256。

## 实验定义

- 固定 `WF_TRON1B/xml/robot.xml` 无限平地，默认每隔 5°，方向为
  `0, 5, …, 355°`，共 72 个方向，不重复采集 360°。
- 使用当前机身坐标系：0° 前、90° 左、180° 后、270° 右。
  `cmd = [speed*cos(theta), speed*sin(theta), 0]`，不施加转向命令。
  机器人实际航向可能漂移；这里测的是机身速度指令跟踪，不是世界坐标直线路径跟踪。
- 默认速度模长 0.5 m/s，每方向 10 次，共 720 次；每次从 XML 默认姿态、零速度独立重置，
  直接启动 WALK policy，零命令预热 2 s，目标命令稳定 3 s，再测量 10 s。
  不包含交互式启动时的 STAND 插值。所有时长都使用仿真时间。
- 每次新建控制器，清空 observation history、last action、GRU hidden state。
  保留 YAML 的观测噪声配置，每次重复设置独立、可复现的随机种子；每组内随机化方向顺序。
  默认不额外随机化物理参数或初始姿态。关闭观测噪声后，确定性重复可能完全相同。
- 物理频率沿用 XML（1000 Hz），控制器沿用 YAML（500 Hz），策略和采样为 50 Hz。
  速度真值取浮动基座原点的 MuJoCo 速度，经当前姿态旋转到机身系。
  policy 的 `predicted_lin_vel` 不参与误差计算。

瞬时误差（m/s）：

```text
e(t) = sqrt((vx_body(t) - cmd_vx)^2 + (vy_body(t) - cmd_vy)^2)
E(direction, repeat) = mean_t(e(t))   # 只使用测量段
mean(direction)   = mean_repeat(E)
median(direction) = median_repeat(E)
```

因此 mean / median 都描述**多次实验的单次平均误差**，不是把时间点当作独立重复。
另存每次实验的 RMS 误差和 x/y 速度偏差。

## 运行

仓库的 LimX SDK 锁定 NumPy 1.26.3，建议使用 Python 3.12：

```bash
uv sync --frozen --python 3.12

# 第一步：只采集。输出目录必须尚不存在，以免覆盖实验数据。
MUJOCO_GL=egl uv run --frozen python scripts/velocity_tracking.py collect \
  --rl-type mjlab_repts_gru_lin_depth \
  --speed 0.5 --direction-step 5 --repeats 10 --workers 6 \
  --warmup 2 --settle 3 --duration 10 \
  --seed 20260926 --output logs/velocity_tracking/flat_05_deg5_r10

# 第二步：读取完整原始数据、校验采样和指令、生成两级统计。
uv run --frozen python scripts/velocity_tracking.py summarize logs/velocity_tracking/flat_05_deg5_r10

# 检查 direction_summary.csv 和 summary.json 后，再单独作图。
uv run --frozen --with 'matplotlib>=3.8,<3.11' python scripts/velocity_tracking.py \
  plot logs/velocity_tracking/flat_05_deg5_r10
```

也支持无深度策略 `mjlab_repts_lin`。
参数未给出时，策略采用环境变量 `RL_TYPE`，否则使用当前 Docker 默认的
`mjlab_repts_gru_lin_depth`。若需要不同速度，请为每档速度使用独立输出目录；
全方向测试速度限制为 `(0, 1] m/s`，避免超出侧向和倒车指令范围。
时长必须为策略周期 0.02 s 的正整数倍。

`--direction-step` 必须为 360 的正整数约数；例如设为 30 可以复现旧的
12 方向网格。`--workers` 默认 1；CPU/内存允许时可设为 6 并行采集。
每个进程独立持有 MuJoCo、ONNX 和渲染上下文，种子仅由实验计划决定，
不会因为任务完成顺序改变。主进程按计划顺序保存索引并校验各进程配置一致。
使用多进程时，默认设置 `LP_NUM_THREADS=1`，避免软件渲染线程过度竞争。

这一路径直接驱动现有 `WheelfootController.update()`、ONNX 和动作映射，
使用与 `simulator.py` 相同的模型、PD 控制律和 XML 力矩限幅。
不启动机器人网络连接或手柄。物理、控制和深度按确定的仿真步调度，
所以实验不包含 ROS/SDK 通信延迟、异步进程调度和实时性能的影响。
不要将结果直接解释为完整 ROS 通信链路的性能。

深度策略实际渲染 480×848 D435 图像，沿用模拟器的约 30 Hz 采集间隔，
模拟 ROS1 `16UC1` 毫米量化，再调用现有裁剪/缩放函数生成 `[1,1,30,45]`。
GRU policy 使用与当前部署路径相同的深度预处理：部署端将深度裁剪到
`[0.2, 2.0] m` 并归一化到 `[0, 1]`，之后裁剪和缩放，再送入 ONNX。
可在已有 Docker 环境内直接用 `python` 运行同样命令；需要可用的 EGL 渲染环境。
如使用其他 MuJoCo GL 后端，在进程启动前设置 `MUJOCO_GL`。

## 数据与图

| 文件 | 内容 |
| --- | --- |
| `manifest.json` | 参数、种子、顺序、策略/场景/配置哈希、配置原文、运行状态 |
| `schedule.csv` | 计划的每个方向和重复序号 |
| `raw/rNNN_dDDD.csv` | 每次的启动/稳定/测量时间序列，命令、机身/世界速度、姿态和误差 |
| `trials.csv` | 每次是否完成、失败原因和原始文件路径 |
| `trial_summary.csv` | 每次测量段的平均/RMS误差、x/y偏差 |
| `direction_summary.csv` | 每方向跨重复 mean、median、标准差、成功/失败次数 |
| `summary.json` | 校验后的总次数和汇总文件哈希 |
| `velocity_tracking_polar.png/.pdf/.svg` | 参考样式的彩色散点与 mean / median 极坐标曲线 |

彩色圆点表示各次实验的 E，采用蓝—黄—红色图；黑色实线表示各方向跨重复 mean，
黑色虚线表示各方向跨重复 median。两条曲线绘制在同一个极坐标图中，
直接连接相邻实测方向的统计量并闭合 0°/360° 接缝，不做平滑拟合。
图例数值是全部有效 trial 的 E 的总体 mean / median；曲线使用各方向的统计值。
角度是命令方向，半径和颜色都表示误差大小，单位 m/s；不进行角度 jitter。
外圈半径固定为 **1.0 m/s**，颜色也固定按 **0–1.0 m/s** 映射，不随本轮最大误差缩放。
若数据超过 1.0 m/s，绘图会报错，避免静默截掉超界点。
5° 网格的全部 72 个方向都参与绘图，文字刻度每 45° 标注一次，避免标签拥挤。
某方向存在失败时，在该方向标注有效次数；完整计数始终保存在汇总 CSV 中。

默认基座高度低于 0.35 m、倾斜超过 60°、MuJoCo warning 或非有限状态/动作
都会终止当前 trial，保存失败前的数据，然后重置并继续其余 trial。
失败 trial 不计入误差统计，避免不完整测量段产生虚假的低误差；所有失败次数
都会显示，某方向全失败则无统计点并标注 failed。解读时应同时查看有效次数，
有效实验的误差不能代替失败率。

运行中断时保留已落盘的数据，但不会允许作为完整实验作图。重新运行时使用新目录。
输出默认放在已被 git 忽略的 `logs/velocity_tracking/`。
