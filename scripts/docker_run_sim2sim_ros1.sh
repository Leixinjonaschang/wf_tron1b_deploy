#!/usr/bin/env bash
set -euo pipefail

# Run this script inside the already-started ROS1 Docker container.
# It does not create, rebuild, stop, or remove Docker containers.
#
# Host-side container startup is handled by:
#   scripts/docker_start_ros1.sh
#
# Typical container usage:
#   docker exec -it tron_deploy bash
#   /work/scripts/docker_run_sim2sim_ros1.sh

if [[ ! -d /work || ! -f /work/scripts/start_sim2sim.sh ]]; then
  echo "This script is intended to run inside the container with the repo mounted at /work." >&2
  exit 1
fi

cd /work

# The image uses a conda env named "sim". Activate it when available so ROS1,
# uv, MuJoCo, ONNX Runtime, and LimX SDK all come from the same environment.
if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /opt/conda/etc/profile.d/conda.sh
  set +u
  conda activate sim
  set -u
fi

# Default robot/policy pair for ROS depth sim2sim. Callers can override any of
# these with environment variables before invoking this script.
export ROBOT_TYPE="${ROBOT_TYPE:-WF_TRON1B}"
export RL_TYPE="${RL_TYPE:-mjlab_repts_gru_lin_depth}"
export ROBOT_IP="${ROBOT_IP:-127.0.0.1}"
export PYTHON="${PYTHON:-python}"

DEPTH_VIEW_MIN_DEFAULT=0.2
DEPTH_VIEW_MAX_DEFAULT=2.0

# ROS1 depth transport defaults. The simulator publishes sensor_msgs/Image on
# this topic, and the controller/depth viewer subscribe to the same topic.
export ROS_TYPE="${ROS_TYPE:-ros1}"
export MJLAB_DEPTH_SOURCE="${MJLAB_DEPTH_SOURCE:-ros}"
export MJLAB_DEPTH_SINK="${MJLAB_DEPTH_SINK:-ros}"
export MJLAB_DEPTH_ROS_TOPIC="${MJLAB_DEPTH_ROS_TOPIC:-/camera/depth/image_rect_raw}"
export MJLAB_DEPTH_CAPTURE_HZ="${MJLAB_DEPTH_CAPTURE_HZ:-30.0}"
export MJLAB_DEPTH_HEIGHT="${MJLAB_DEPTH_HEIGHT:-480}"
export MJLAB_DEPTH_WIDTH="${MJLAB_DEPTH_WIDTH:-848}"
export MJLAB_DEPTH_MAX_AGE="${MJLAB_DEPTH_MAX_AGE:-0.5}"
export MJLAB_DEPTH_TIMEOUT="${MJLAB_DEPTH_TIMEOUT:-2.0}"

# Keep the legacy file path configured for fallback/debug modes that use npy.
export MJLAB_DEPTH_NPY_PATH="${MJLAB_DEPTH_NPY_PATH:-/work/logs/sim2sim/depth_frame.npy}"

# Enable the Python depth viewer by default when DISPLAY is available. Set
# MJLAB_DEPTH_VIEW=0 to run simulator/controller/joystick without the viewer.
if [[ -z "${MJLAB_DEPTH_VIEW:-}" ]]; then
  if [[ -n "${DISPLAY:-}" ]]; then
    export MJLAB_DEPTH_VIEW=1
  else
    export MJLAB_DEPTH_VIEW=0
  fi
fi
export MJLAB_DEPTH_VIEW_SOURCE="${MJLAB_DEPTH_VIEW_SOURCE:-${MJLAB_DEPTH_SOURCE}}"
export MJLAB_DEPTH_VIEW_SCALE="${MJLAB_DEPTH_VIEW_SCALE:-1}"
export MJLAB_DEPTH_VIEW_MIN="${MJLAB_DEPTH_VIEW_MIN:-${DEPTH_VIEW_MIN_DEFAULT}}"
export MJLAB_DEPTH_VIEW_MAX="${MJLAB_DEPTH_VIEW_MAX:-${DEPTH_VIEW_MAX_DEFAULT}}"
export MJLAB_DEPTH_VIEW_HZ="${MJLAB_DEPTH_VIEW_HZ:-30.0}"
export MJLAB_DEPTH_VIEW_COLORMAP="${MJLAB_DEPTH_VIEW_COLORMAP:-turbo}"

exec /work/scripts/start_sim2sim.sh "$@"
