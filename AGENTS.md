# Repository Guidelines

## Project Structure & Module Organization

This repository packages WF_TRON1B simulation and Python RL deployment assets.

- `pointfoot-mujoco-sim/`: MuJoCo simulator, robot descriptions, joystick binary, SDK wheel, and simulator docs.
- `rl-deploy-with-python/`: controller entry point, controller implementations, ONNX policies, deployment helpers, and tests.
- `rl-deploy-with-python/controllers/model/WF_TRON1B/`: WF_TRON1B model configuration and policy files.
- `rl-deploy-with-python/tests/`: alignment and policy-interface tests.
- `logs/sim2sim/`: runtime logs produced by `start_sim2sim.sh`; do not commit generated logs.
- `start_sim2sim.sh`: convenience launcher for simulator, controller, and joystick.

## Build, Test, and Development Commands

- `uv sync`: create/update the local Python 3.13 environment from `pyproject.toml` and `uv.lock`.
- `ROBOT_TYPE=WF_TRON1B RL_TYPE=mjlab_repts ./start_sim2sim.sh`: launch MuJoCo sim2sim with the default WF_TRON1B policy.
- `ROBOT_TYPE=WF_TRON1B RL_TYPE=mjlab_repts_lin_depth ./start_sim2sim.sh`: launch the depth-enabled deployment path.
- `uv run python pointfoot-mujoco-sim/simulator.py 127.0.0.1`: run only the simulator.
- `ROBOT_TYPE=WF_TRON1B RL_TYPE=mjlab_repts uv run python rl-deploy-with-python/main.py 127.0.0.1`: run only the controller.
- `uv run --with pytest pytest rl-deploy-with-python/tests`: run the test suite.

## Coding Style & Naming Conventions

Use Python 3.13-compatible code. Follow PEP 8: four-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, and uppercase constants such as `ROBOT_TYPE` or `POLICY_PATH`. Keep scripts executable only when they are intended to be run directly. Prefer `pathlib.Path` for filesystem paths in Python tests and helper code.

## Testing Guidelines

Tests live in `rl-deploy-with-python/tests` and should be named `test_*.py`. Add focused tests when changing observation construction, action mapping, policy metadata, or controller compatibility. Hardware or simulator-dependent behavior should be isolated behind mocks where possible, as existing tests do for LimX SDK modules.

## Commit & Pull Request Guidelines

The current history uses concise imperative commits, for example `Initialize WF_TRON1B robot perceptive deploy repo.` Keep future commits similarly direct and scoped. Pull requests should include a short summary, test results, affected robot/RL types, and any required environment variables. Include screenshots or log excerpts only for simulator or runtime behavior changes.

## Security & Configuration Tips

Do not commit private credentials, generated logs, local virtual environments, or robot-specific secrets. Set runtime configuration through environment variables such as `ROBOT_TYPE`, `RL_TYPE`, `ROBOT_IP`, and `MJLAB_DEPTH_NPY_PATH`.
