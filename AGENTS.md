# Repository Guidelines

## Project Structure & Module Organization

This repository packages WF_TRON1B MuJoCo simulation and Python RL deployment assets.

- `pointfoot-mujoco-sim/`: MuJoCo simulator, robot descriptions, joystick binary, SDK wheel, and simulator docs.
- `rl-deploy-with-python/`: controller entry point, controller implementations, ONNX policies, depth sources, and tests.
- `rl-deploy-with-python/controllers/model/WF_TRON1B/`: WF_TRON1B model configuration and policy files.
- `rl-deploy-with-python/tests/`: alignment and policy-interface tests.
- `doc/`: deployment notes distilled from vendor documentation, including true-robot deployment and RealSense depth acquisition.
- `logs/sim2sim/`: runtime logs from `start_sim2sim.sh`; do not commit generated logs.
- `start_sim2sim.sh`: convenience launcher for simulator, controller, and joystick.

## Build, Test, and Development Commands

- `uv sync`: create/update the local Python environment from `pyproject.toml` and `uv.lock`.
- ROS1 depth sim2sim should run in the Docker image described by `Dockerfile`; build it with `scripts/docker_build_ros1.sh`.
  Smoke-check ROS1 depth transport with
  `docker run --rm --network host -v "$PWD:/work:rw" wf-tron1b-deploy:ros1 bash -lc 'python scripts/ros1_depth_smoke.py'`.
- `ROBOT_TYPE=WF_TRON1B RL_TYPE=mjlab_repts ./start_sim2sim.sh`: launch MuJoCo sim2sim with the default WF_TRON1B policy.
- `scripts/docker_run_sim2sim_ros1.sh`: launch depth-enabled ROS1 Docker sim2sim over `/camera/depth/image_rect_raw`.
- `uv run python pointfoot-mujoco-sim/simulator.py 127.0.0.1`: run only the simulator.
- `ROBOT_TYPE=WF_TRON1B RL_TYPE=mjlab_repts uv run python rl-deploy-with-python/main.py 127.0.0.1`: run only the controller.
- `uv run --with pytest pytest rl-deploy-with-python/tests`: run the test suite.
- `bash -n start_sim2sim.sh`: validate launcher syntax after shell edits.

## Coding Style & Naming Conventions

Use Python 3.10-compatible code. Follow PEP 8: four-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, and uppercase constants such as `ROBOT_TYPE` or `POLICY_PATH`. Keep scripts executable only when they are intended to be run directly. Prefer `pathlib.Path` for filesystem paths in Python tests and helper code.

## Testing Guidelines

Tests live in `rl-deploy-with-python/tests` and should be named `test_*.py`. Add focused tests when changing observation construction, action mapping, policy metadata, ROS depth parsing, or controller compatibility. Hardware, simulator, LimX SDK, and ROS behavior should be isolated behind mocks where possible.

## ROS Depth Workflow

`mjlab_repts_lin_depth` passes depth out-of-band from the LimX robot state/control path. The supported transport is ROS1 `sensor_msgs/Image`: `start_sim2sim.sh` sets `ROS_TYPE=ros1`, `MJLAB_DEPTH_SINK=ros`, `MJLAB_DEPTH_SOURCE=ros`, `MJLAB_DEPTH_ROS_TOPIC=/camera/depth/image_rect_raw`, `MJLAB_DEPTH_CAPTURE_HZ=25.0`, `MJLAB_DEPTH_HEIGHT=28`, `MJLAB_DEPTH_WIDTH=48`, and `MJLAB_DEPTH_MAX_AGE=0.5`.

In `pointfoot-mujoco-sim/simulator.py`, `SimulatorMujoco._export_depth_frame()` renders the MuJoCo `d435` camera every `depth_capture_period_steps`, converts meters to `16UC1` millimeters for ROS1, and publishes on `MJLAB_DEPTH_ROS_TOPIC`. In `rl-deploy-with-python/mjlab_repts_lin_depth.py`, `RosDepthFrameSource` subscribes to the same topic, converts `16UC1` back to meters with scale `0.001`, clips/resizes to `[1, 1, 28, 48]`, and stores only the latest frame behind a lock. `WheelfootController.compute_actions()` reads `depth_source.frame()` in the policy loop and feeds ONNX inputs `proprio_history`, `actor_command`, `depth`, and `hidden_state_in`; stale or missing frames raise `TimeoutError` according to `MJLAB_DEPTH_TIMEOUT` and `MJLAB_DEPTH_MAX_AGE`.

The legacy file transport remains available with `MJLAB_DEPTH_SOURCE=npy_live MJLAB_DEPTH_SINK=npy`. The simulator writes raw float32 meter depth to `MJLAB_DEPTH_NPY_PATH` through a temp file plus `os.replace()`, and the controller polls file mtime before preprocessing. Use this only as a fallback when ROS packages are unavailable or when explicitly testing file-based transfer.

## Deployment Notes

- [doc/real_robot_deployment.md](doc/real_robot_deployment.md): true-robot deployment checklist, safety steps, network setup, Python deployment, autolaunch, and ROS C++ reference path.
- [doc/realsense_depth.md](doc/realsense_depth.md): RealSense D435i startup, ROS Noetic network setup, depth topic discovery, visualization, and `mjlab_repts_lin_depth` integration.
- [doc/ros1_depth_sim2sim.md](doc/ros1_depth_sim2sim.md): Docker build/run commands for ROS1 depth sim2sim and the ROS1 depth smoke check.

## Commit & Pull Request Guidelines

The current history uses concise imperative commits, for example `Initialize WF_TRON1B robot perceptive deploy repo.` Keep future commits similarly direct and scoped. Pull requests should include a short summary, test results, affected robot/RL types, and any required environment variables. Include screenshots or log excerpts only for simulator or runtime behavior changes.

## Security & Configuration Tips

Do not commit private credentials, generated logs, local virtual environments, or robot-specific secrets. Set runtime configuration through environment variables such as `ROBOT_TYPE`, `RL_TYPE`, `ROBOT_IP`, `MJLAB_DEPTH_ROS_TOPIC`, and `MJLAB_DEPTH_MAX_AGE`.
