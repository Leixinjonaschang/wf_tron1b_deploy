# Repository Guidelines

## Project Structure & Module Organization

This repository packages WF_TRON1B MuJoCo simulation and Python RL deployment assets.

- `pointfoot-mujoco-sim/`: MuJoCo simulator, robot descriptions, joystick binary, SDK wheel, and simulator docs.
- `rl-deploy-with-python/`: controller entry point, controller implementations, ONNX policies, depth sources, and tests.
- `rl-deploy-with-python/controllers/model/WF_TRON1B/`: WF_TRON1B model configuration and policy files.
- `rl-deploy-with-python/tests/`: alignment and policy-interface tests.
- `logs/sim2sim/`: runtime logs from `start_sim2sim.sh`; do not commit generated logs.
- `start_sim2sim.sh`: convenience launcher for simulator, controller, and joystick.

## Build, Test, and Development Commands

- `uv sync`: create/update the local Python 3.13 environment from `pyproject.toml` and `uv.lock`.
- `ROBOT_TYPE=WF_TRON1B RL_TYPE=mjlab_repts ./start_sim2sim.sh`: launch MuJoCo sim2sim with the default WF_TRON1B policy.
- `ROBOT_TYPE=WF_TRON1B RL_TYPE=mjlab_repts_lin_depth ./start_sim2sim.sh`: launch depth-enabled sim2sim over ROS `/camera/depth/image_rect_raw`.
- `uv run python pointfoot-mujoco-sim/simulator.py 127.0.0.1`: run only the simulator.
- `ROBOT_TYPE=WF_TRON1B RL_TYPE=mjlab_repts uv run python rl-deploy-with-python/main.py 127.0.0.1`: run only the controller.
- `uv run --with pytest pytest rl-deploy-with-python/tests`: run the test suite.
- `bash -n start_sim2sim.sh`: validate launcher syntax after shell edits.

## Coding Style & Naming Conventions

Use Python 3.13-compatible code. Follow PEP 8: four-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, and uppercase constants such as `ROBOT_TYPE` or `POLICY_PATH`. Keep scripts executable only when they are intended to be run directly. Prefer `pathlib.Path` for filesystem paths in Python tests and helper code.

## Testing Guidelines

Tests live in `rl-deploy-with-python/tests` and should be named `test_*.py`. Add focused tests when changing observation construction, action mapping, policy metadata, ROS depth parsing, or controller compatibility. Hardware, simulator, LimX SDK, and ROS behavior should be isolated behind mocks where possible.

## ROS Depth Workflow

`mjlab_repts_lin_depth` uses ROS1 `sensor_msgs/Image` by default. The simulator publishes depth with `MJLAB_DEPTH_SINK=ros`; the controller consumes it with `MJLAB_DEPTH_SOURCE=ros`. Keep the shared topic default `/camera/depth/image_rect_raw` unless matching a real camera launch. The legacy file path remains available with `MJLAB_DEPTH_SOURCE=npy_live MJLAB_DEPTH_SINK=npy`.

## Commit & Pull Request Guidelines

The current history uses concise imperative commits, for example `Initialize WF_TRON1B robot perceptive deploy repo.` Keep future commits similarly direct and scoped. Pull requests should include a short summary, test results, affected robot/RL types, and any required environment variables. Include screenshots or log excerpts only for simulator or runtime behavior changes.

## Security & Configuration Tips

Do not commit private credentials, generated logs, local virtual environments, or robot-specific secrets. Set runtime configuration through environment variables such as `ROBOT_TYPE`, `RL_TYPE`, `ROBOT_IP`, `MJLAB_DEPTH_ROS_TOPIC`, and `MJLAB_DEPTH_MAX_AGE`.
