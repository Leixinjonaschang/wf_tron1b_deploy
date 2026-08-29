# Repository Guidelines

## Project Structure & Module Organization

This repository packages WF_TRON1B MuJoCo simulation and Python RL deployment assets.

- `pointfoot-mujoco-sim/`: MuJoCo simulator, robot descriptions, joystick binary, SDK wheel, and simulator docs.
- `rl-deploy-with-python/`: controller entry point, controller implementations, ONNX policies, depth sources, and tests.
- `rl-deploy-with-python/controllers/model/WF_TRON1B/`: WF_TRON1B model configuration and policy files.
- `rl-deploy-with-python/tests/`: alignment and policy-interface tests.
- `scripts/`: Docker helpers, ROS1 smoke checks, and the Python depth viewer.
- `doc/`: deployment notes distilled from vendor documentation, including true-robot deployment and RealSense depth acquisition.
- `logs/sim2sim/`: runtime logs from `scripts/start_sim2sim.sh`; do not commit generated logs.
- `scripts/start_sim2sim.sh`: in-container sim2sim process launcher used by `scripts/docker_run_sim2sim_ros1.sh`; keep it unless the Docker launcher is replaced.

## Build, Test, and Development Commands

- `uv sync`: create/update the local Python environment from `pyproject.toml` and `uv.lock`.
- ROS1 depth sim2sim should run in the Docker image described by `Dockerfile`; build it with `scripts/docker_build_ros1.sh`.
  Smoke-check ROS1 depth transport with
  `sudo docker exec -w /work tron_deploy python scripts/ros1_depth_smoke.py` after starting the container.
- `sudo -E scripts/docker_start_ros1.sh`: create or start the persistent ROS1 Docker container named `tron_deploy`.
- `sudo docker exec -it tron_deploy bash`: enter the persistent container.
- `/work/scripts/docker_run_sim2sim_ros1.sh`: run depth-enabled ROS1 sim2sim inside the container over `/camera/depth/image_rect_raw`.
- `ROBOT_TYPE=WF_TRON1B RL_TYPE=mjlab_repts_lin scripts/start_sim2sim.sh`: launch MuJoCo sim2sim with the default WF_TRON1B policy.
- `uv run python pointfoot-mujoco-sim/simulator.py 127.0.0.1`: run only the simulator.
- `ROBOT_TYPE=WF_TRON1B RL_TYPE=mjlab_repts_lin uv run python rl-deploy-with-python/main.py 127.0.0.1`: run only the controller.
- `ROBOT_TYPE=WF_TRON1B RL_TYPE=mjlab_repts_lin uv run python rl-deploy-with-python/main.py 127.0.0.1`: run the LinVel REPTS controller variant.
- `python scripts/depth_image_viewer.py`: run only the ROS/npy depth viewer.
- `uv run --with pytest pytest rl-deploy-with-python/tests`: run the test suite.
- `bash -n scripts/start_sim2sim.sh scripts/docker_build_ros1.sh scripts/docker_start_ros1.sh scripts/docker_run_sim2sim_ros1.sh`: validate launcher syntax after shell edits.

## Coding Style & Naming Conventions

Use Python 3.10-compatible code. Follow PEP 8: four-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, and uppercase constants such as `ROBOT_TYPE` or `POLICY_PATH`. Keep scripts executable only when they are intended to be run directly. Prefer `pathlib.Path` for filesystem paths in Python tests and helper code.

## Testing Guidelines

Tests live in `rl-deploy-with-python/tests` and should be named `test_*.py`. Add focused tests when changing observation construction, action mapping, policy metadata, ROS depth parsing, or controller compatibility. Hardware, simulator, LimX SDK, and ROS behavior should be isolated behind mocks where possible.

## ROS Depth Workflow

`mjlab_repts_gru_lin_depth` passes depth out-of-band from the LimX robot state/control path. The supported transport is ROS1 `sensor_msgs/Image`: `scripts/start_sim2sim.sh` sets `ROS_TYPE=ros1`, `MJLAB_DEPTH_SINK=ros`, `MJLAB_DEPTH_SOURCE=ros`, `MJLAB_DEPTH_ROS_TOPIC=/camera/depth/image_rect_raw`, `MJLAB_DEPTH_CAPTURE_HZ=30.0`, `MJLAB_DEPTH_HEIGHT=480`, `MJLAB_DEPTH_WIDTH=848`, and `MJLAB_DEPTH_MAX_AGE=0.5`.

In `pointfoot-mujoco-sim/simulator.py`, `SimulatorMujoco._export_depth_frame()` renders the full `480x848` MuJoCo `d435` frame every `depth_capture_period_steps`, converts meters to `16UC1` millimeters for ROS1, and publishes on `MJLAB_DEPTH_ROS_TOPIC`. In `rl-deploy-with-python/mjlab_repts_lin_depth.py`, `RosDepthFrameSource` subscribes to the same topic, converts `16UC1` back to meters with scale `0.001`, sanitizes invalid values, removes the leftmost 128 columns (`480x848` to `480x720`), then uses nearest-neighbor resize to produce float32 ONNX input `[1, 1, 30, 45]`; clipping and normalization happen inside ONNX. It stores only the latest frame behind a lock; pre-cropped policy-sized input is rejected to prevent a second crop. `WheelfootController.compute_actions()` reads `depth_source.frame()` in the policy loop and feeds ONNX inputs `proprio_history`, `actor_command`, `depth`, and `hidden_state_in`; stale or missing frames raise `TimeoutError` according to `MJLAB_DEPTH_TIMEOUT` and `MJLAB_DEPTH_MAX_AGE`. The depth ONNX interface requires `[1, 1, 30, 45]`; legacy `[1, 1, 28, 48]` policies are rejected.

The legacy file transport remains available with `MJLAB_DEPTH_SOURCE=npy_live MJLAB_DEPTH_SINK=npy`. The simulator writes raw float32 meter depth to `MJLAB_DEPTH_NPY_PATH` through a temp file plus `os.replace()`, and the controller polls file mtime before preprocessing. Use this only as a fallback when ROS packages are unavailable or when explicitly testing file-based transfer.

## Deployment Notes

- [doc/real_robot_deployment.md](doc/real_robot_deployment.md): true-robot deployment checklist, safety steps, network setup, Python deployment, autolaunch, and ROS C++ reference path.
- [doc/realsense_depth.md](doc/realsense_depth.md): RealSense D435i startup, ROS Noetic network setup, depth topic discovery, visualization, and `mjlab_repts_gru_lin_depth` integration.
- [doc/ros1_depth_sim2sim.md](doc/ros1_depth_sim2sim.md): concise Docker workflow for building the image, starting `tron_deploy`, running sim2sim, manual step-by-step process startup, and common fixes.

## Commit & Pull Request Guidelines

The current history uses concise imperative commits, for example `Initialize WF_TRON1B robot perceptive deploy repo.` Keep future commits similarly direct and scoped. Pull requests should include a short summary, test results, affected robot/RL types, and any required environment variables. Include screenshots or log excerpts only for simulator or runtime behavior changes.

## Security & Configuration Tips

Do not commit private credentials, generated logs, local virtual environments, or robot-specific secrets. Set runtime configuration through environment variables such as `ROBOT_TYPE`, `RL_TYPE`, `ROBOT_IP`, `MJLAB_DEPTH_ROS_TOPIC`, and `MJLAB_DEPTH_MAX_AGE`.
