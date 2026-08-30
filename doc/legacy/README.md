# Legacy Compatibility

本目录记录历史策略和兼容路径。当前推荐部署只使用
`WF_TRON1B + mjlab_repts_gru_lin_depth`；其 Sim-to-Sim 和 Sim-to-Real 命令见仓库根目录
[README](../../README.md)。

## 策略状态

| `RL_TYPE` | 状态 | 输入 | 备注 |
| --- | --- | --- | --- |
| `mjlab_repts_lin` | 当前入口兼容 | proprioception | 无 depth 的 LinVel policy |
| `isaacgym` | 当前入口兼容 | proprioception + encoder | 历史导出接口 |
| `isaaclab` | 当前入口兼容 | proprioception + encoder | 历史导出接口 |
| `mjlab_repts` | 当前入口不支持 | proprioception | 仅通过 Git 历史追溯 |
| `mjlab_repts_lin_depth` | 当前入口不支持 | depth + proprioception | 旧 `[1,64]` recurrent policy |

“当前入口兼容”只表示代码和模型仍可加载，不表示它属于推荐部署验收流程。

## Non-Perceptive LinVel Sim-to-Sim

`mjlab_repts_lin` 不使用 ROS depth。需要历史复现时，在三个终端分别运行：

```bash
ROBOT_TYPE=WF_TRON1B RL_TYPE=mjlab_repts_lin \
  uv run python pointfoot-mujoco-sim/simulator.py 127.0.0.1
```

```bash
ROBOT_TYPE=WF_TRON1B RL_TYPE=mjlab_repts_lin \
  uv run python rl-deploy-with-python/main.py 127.0.0.1
```

```bash
pointfoot-mujoco-sim/robot-joystick/robot-joystick
```

该流程不能用于证明当前 perceptive policy 的 depth、crop/resize、invalid sentinel 或 GRU
state 正确。

## IsaacGym / IsaacLab

对应 policy 与 encoder 位于：

```text
rl-deploy-with-python/controllers/model/WF_TRON1B/policy/isaacgym/
rl-deploy-with-python/controllers/model/WF_TRON1B/policy/isaaclab/
```

它们保留上游 encoder + policy 接口，使用通用 `params.yaml`。如需复现，应单独记录所用
training export、observation order 和 action mapping，不要沿用当前 GRU depth contract。

## 已退出当前入口的策略

`mjlab_repts` 和 `mjlab_repts_lin_depth` 已不在 `main.py` 的支持列表中，当前工作树也不再
提供对应推荐模型或配置。需要研究旧行为时应检出明确的历史 commit，在隔离分支中运行；
不要只复制旧 ONNX 到当前策略目录。

旧 `mjlab_repts_lin_depth` 使用 `[1,64]` recurrent state，其 depth 处理边界与当前
`mjlab_repts_gru_lin_depth` 不同，二者不可混用。

## 兼容原则

- legacy policy 不得出现在根 README 的推荐命令中；
- 每次历史复现都应固定 commit、policy hash、YAML 和 SDK 版本；
- 旧模型不通过当前 Policy Contract 时，应在历史分支修复，不应放宽当前校验；
- legacy 结果不能替代当前 Sim-to-Sim → Sim-to-Real 验收。
