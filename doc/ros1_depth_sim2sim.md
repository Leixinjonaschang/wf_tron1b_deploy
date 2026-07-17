# ROS1 Depth Sim2sim in Docker

This repository uses ROS1 for perceptive depth transport. The host machine does not need ROS1 installed; build and run the provided Docker image when the host only has ROS2 Humble or no ROS installation.

## Build the image

```bash
scripts/docker_build_ros1.sh
```

The image is tagged `wf-tron1b-deploy:ros1` by default. Override with `IMAGE_NAME=...` when needed.

## Run the ROS1 smoke check

```bash
docker run --rm --network host -v "$PWD:/work:rw" wf-tron1b-deploy:ros1 \
  bash -lc 'python scripts/ros1_depth_smoke.py'
```

The smoke check starts `roscore` if needed, publishes one synthetic `16UC1` depth image on `/camera/depth/image_rect_raw`, and verifies that `RosDepthFrameSource` receives a `[1, 1, 28, 48]` meter-scaled policy tensor.

## Run interactive sim2sim

Allow local X11 clients if your desktop requires it:

```bash
xhost +local:docker
```

Then run:

```bash
scripts/docker_run_sim2sim_ros1.sh
```

The helper mounts the repository into `/work`, uses host networking, forwards X11, sets `ROS_TYPE=ros1`, and launches:

- MuJoCo simulator with `MJLAB_DEPTH_SINK=ros`
- Python controller with `MJLAB_DEPTH_SOURCE=ros`
- ROS1 depth viewer (`scripts/depth_image_viewer.py`), unless `MJLAB_DEPTH_VIEW=0`
- virtual joystick in the foreground

Useful overrides:

```bash
MJLAB_DEPTH_VIEW=0 scripts/docker_run_sim2sim_ros1.sh
MJLAB_DEPTH_CAPTURE_HZ=15 scripts/docker_run_sim2sim_ros1.sh
MJLAB_DEPTH_ROS_TOPIC=/camera/depth/image_rect_raw scripts/docker_run_sim2sim_ros1.sh
```

Logs are written under `logs/sim2sim/`.
