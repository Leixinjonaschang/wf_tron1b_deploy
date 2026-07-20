#!/usr/bin/env bash
# set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SIM_DIR="${REPO_DIR}/pointfoot-mujoco-sim"
CTRL_DIR="${REPO_DIR}/rl-deploy-with-python"
LOG_DIR="${REPO_DIR}/logs/sim2sim"

ROBOT_TYPE="${ROBOT_TYPE:-WF_TRON1B}"
RL_TYPE="${RL_TYPE:-mjlab_repts}"
MJLAB_SCENE="${MJLAB_SCENE:-robot.xml}"
ROBOT_IP="${ROBOT_IP:-127.0.0.1}"
SIM_START_DELAY="${SIM_START_DELAY:-2}"
CTRL_START_DELAY="${CTRL_START_DELAY:-1}"
LIMXSDK_WHL="${LIMXSDK_WHL:-${SIM_DIR}/limxsdk-lowlevel/python3/amd64/limxsdk-3.4.2-py3-none-any.whl}"

SIMULATOR="${SIM_DIR}/simulator.py"
CONTROLLER="${CTRL_DIR}/main.py"
DEPTH_VIEWER="${SCRIPT_DIR}/depth_image_viewer.py"
JOYSTICK="${SIM_DIR}/robot-joystick/robot-joystick"
MODEL_XML="${SIM_DIR}/robot-description/pointfoot/${ROBOT_TYPE}/xml/${MJLAB_SCENE}"
POLICY="${CTRL_DIR}/controllers/model/${ROBOT_TYPE}/policy/${RL_TYPE}/policy.onnx"
DEPTH_FRAME_PATH="${MJLAB_DEPTH_NPY_PATH:-${LOG_DIR}/depth_frame.npy}"
DEPTH_GRADCAM_PATH="${MJLAB_DEPTH_GRADCAM_PATH:-${LOG_DIR}/depth_gradcam.npz}"

PIDS=()
PYTHON_CMD=()

usage() {
  cat <<EOF
Usage:
  ROBOT_TYPE=WF_TRON1B RL_TYPE=mjlab_repts scripts/start_sim2sim.sh
  ROS_TYPE=ros1 ROBOT_TYPE=WF_TRON1B RL_TYPE=mjlab_repts_lin_depth scripts/start_sim2sim.sh

Defaults:
  ROBOT_TYPE=${ROBOT_TYPE}
  RL_TYPE=${RL_TYPE}
  MJLAB_SCENE=${MJLAB_SCENE}
  ROBOT_IP=${ROBOT_IP}
  SIM_START_DELAY=${SIM_START_DELAY}
  CTRL_START_DELAY=${CTRL_START_DELAY}
  LIMXSDK_WHL=${LIMXSDK_WHL}

Processes started:
  1. Python pointfoot-mujoco-sim/simulator.py
  2. Python rl-deploy-with-python/main.py
  3. Optional Python scripts/depth_image_viewer.py
  4. pointfoot-mujoco-sim/robot-joystick/robot-joystick
EOF
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -e "${path}" ]]; then
    echo "Missing ${label}: ${path}" >&2
    exit 1
  fi
}

setup_python_cmd() {
  if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_CMD=("${PYTHON}")
    return
  fi

  PYTHON_CMD=(
    uv run
    --with onnxruntime
    --with scipy
    --with pyyaml
    --with mujoco
    --with pygame
    --with 'torch==2.7.1+cpu' --index https://download.pytorch.org/whl/cpu
    --with "${LIMXSDK_WHL}"
    python
  )
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if ((${#PIDS[@]} > 0)); then
    echo
    echo "Stopping sim2sim processes..."
    kill "${PIDS[@]}" 2>/dev/null || true
    wait "${PIDS[@]}" 2>/dev/null || true
  fi
  exit "${status}"
}

start_bg() {
  local name="$1"
  local log_file="$2"
  shift 2

  echo "Starting ${name}; log: ${log_file}"
  "$@" >"${log_file}" 2>&1 &
  local pid=$!
  PIDS+=("${pid}")
  sleep 1
  if ! kill -0 "${pid}" 2>/dev/null; then
    echo "${name} exited early. Last log lines:" >&2
    tail -n 40 "${log_file}" >&2 || true
    exit 1
  fi
}

try_start_ros_master() {
  local name="$1"
  local log_file="$2"
  shift 2

  echo "Starting ${name}; log: ${log_file}"
  "$@" >"${log_file}" 2>&1 &
  local pid=$!
  for _ in {1..20}; do
    if rostopic list >/dev/null 2>&1; then
      PIDS+=("${pid}")
      return 0
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "${name} exited before becoming ready. Last log lines:" >&2
      tail -n 40 "${log_file}" >&2 || true
      wait "${pid}" 2>/dev/null || true
      return 1
    fi
    sleep 0.5
  done

  echo "${name} did not become ready. Last log lines:" >&2
  tail -n 40 "${log_file}" >&2 || true
  kill "${pid}" 2>/dev/null || true
  wait "${pid}" 2>/dev/null || true
  return 1
}

uses_ros_depth() {
  [[ "${RL_TYPE}" == "mjlab_repts_lin_depth" ]] || return 1
  [[ "${MJLAB_DEPTH_SOURCE:-}" == "ros" || "${MJLAB_DEPTH_SINK:-}" == "ros" || "${MJLAB_DEPTH_SINK:-}" == "both" ]]
}

uses_npy_depth() {
  [[ "${RL_TYPE}" == "mjlab_repts_lin_depth" ]] || return 1
  [[ "${MJLAB_DEPTH_SOURCE:-}" == "npy" || "${MJLAB_DEPTH_SOURCE:-}" == "npy_live" || "${MJLAB_DEPTH_SINK:-}" == "npy" || "${MJLAB_DEPTH_SINK:-}" == "both" ]]
}

uses_depth_viewer() {
  [[ "${RL_TYPE}" == "mjlab_repts_lin_depth" ]] || return 1
  [[ "${MJLAB_DEPTH_VIEW:-}" != "0" ]] || return 1
  [[ "${MJLAB_DEPTH_VIEW:-}" == "1" || -n "${DISPLAY:-}" ]]
}

resolve_ros_type() {
  local value="${ROS_TYPE:-}"
  if [[ -n "${value}" ]]; then
    case "${value}" in
      1|ros1) echo "ros1"; return ;;
      *)
        echo "ROS_TYPE must be ros1 for depth transport, got '${value}'." >&2
        exit 1
        ;;
    esac
  fi

  case "${ROS_VERSION:-}" in
    1) echo "ros1" ;;
    2)
      echo "ROS depth transport is ROS1-only. Use the provided ROS1 Docker image for sim2sim." >&2
      exit 1
      ;;
    *)
      echo "ros1"
      ;;
  esac
}

ensure_ros_master() {
  if ! command -v rostopic >/dev/null 2>&1; then
    echo "Missing rostopic. Source a ROS1 environment before running ROS depth." >&2
    exit 1
  fi
  if rostopic list >/dev/null 2>&1; then
    return
  fi

  if command -v roscore >/dev/null 2>&1; then
    if try_start_ros_master "roscore" "${LOG_DIR}/roscore.log" roscore; then
      return
    fi
    echo "Falling back to rosmaster --core." >&2
  else
    echo "Missing roscore; trying rosmaster --core." >&2
  fi

  if ! command -v rosmaster >/dev/null 2>&1; then
    echo "Missing rosmaster and no ROS master is reachable." >&2
    exit 1
  fi

  if try_start_ros_master "rosmaster" "${LOG_DIR}/rosmaster.log" rosmaster --core; then
    return
  fi
  exit 1
}

ensure_python_ros_modules() {
  if ! "${PYTHON_CMD[@]}" -c 'import rospy; import sensor_msgs.msg' >/dev/null 2>&1; then
    cat >&2 <<EOF
The selected Python cannot import rospy and sensor_msgs.
Source ROS1 and run with a Python that can see ROS packages, for example:
  source /opt/ros/noetic/setup.bash
  PYTHON=python3 ROS_TYPE=ros1 ROBOT_TYPE=WF_TRON1B RL_TYPE=mjlab_repts_lin_depth scripts/start_sim2sim.sh

For the legacy file-based fallback, set:
  MJLAB_DEPTH_SOURCE=npy_live MJLAB_DEPTH_SINK=npy
EOF
    exit 1
  fi
}

ensure_ros_runtime() {
  ensure_ros_master
}

configure_ros_depth() {
  if uses_ros_depth; then
    local resolved_ros_type
    resolved_ros_type="$(resolve_ros_type)" || exit 1
    export ROS_TYPE="${resolved_ros_type}"
  fi
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

require_file "${SIMULATOR}" "simulator"
require_file "${CONTROLLER}" "controller"
require_file "${JOYSTICK}" "virtual joystick"
require_file "${MODEL_XML}" "robot model XML"
require_file "${POLICY}" "policy"
require_file "${LIMXSDK_WHL}" "LimX SDK wheel"
if uses_depth_viewer; then
  require_file "${DEPTH_VIEWER}" "depth viewer"
fi

mkdir -p "${LOG_DIR}"
setup_python_cmd

export ROBOT_TYPE
export RL_TYPE
export MJLAB_SCENE
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
if [[ "${RL_TYPE}" == "mjlab_repts_lin_depth" ]]; then
  export ROS_TYPE="${ROS_TYPE:-ros1}"
  export MJLAB_DEPTH_SOURCE="${MJLAB_DEPTH_SOURCE:-ros}"
  export MJLAB_DEPTH_SINK="${MJLAB_DEPTH_SINK:-ros}"
  export MJLAB_DEPTH_ROS_TOPIC="${MJLAB_DEPTH_ROS_TOPIC:-/camera/depth/image_rect_raw}"
  export MJLAB_DEPTH_CAPTURE_HZ="${MJLAB_DEPTH_CAPTURE_HZ:-30.0}"
  export MJLAB_DEPTH_HEIGHT="${MJLAB_DEPTH_HEIGHT:-480}"
  export MJLAB_DEPTH_WIDTH="${MJLAB_DEPTH_WIDTH:-848}"
  export MJLAB_DEPTH_MAX_AGE="${MJLAB_DEPTH_MAX_AGE:-0.5}"
  export MJLAB_DEPTH_GRADCAM="${MJLAB_DEPTH_GRADCAM:-0}"
  export MJLAB_DEPTH_GRADCAM_HZ="${MJLAB_DEPTH_GRADCAM_HZ:-30.0}"
  export MJLAB_DEPTH_GRADCAM_PATH="${DEPTH_GRADCAM_PATH}"
  export MJLAB_DEPTH_GRADCAM_ALPHA="${MJLAB_DEPTH_GRADCAM_ALPHA:-0.45}"
  if uses_npy_depth; then
    export MJLAB_DEPTH_NPY_PATH="${MJLAB_DEPTH_NPY_PATH:-${DEPTH_FRAME_PATH}}"
  fi
  configure_ros_depth
fi

trap cleanup EXIT INT TERM

if uses_ros_depth; then
  ensure_python_ros_modules
  ensure_ros_runtime
fi

echo "ROBOT_TYPE=${ROBOT_TYPE}"
echo "RL_TYPE=${RL_TYPE}"
echo "MJLAB_SCENE=${MJLAB_SCENE}"
echo "ROBOT_IP=${ROBOT_IP}"
echo "SIM_START_DELAY=${SIM_START_DELAY}"
echo "CTRL_START_DELAY=${CTRL_START_DELAY}"
echo "PYTHON_CMD=${PYTHON_CMD[*]}"
if [[ "${RL_TYPE}" == "mjlab_repts_lin_depth" ]]; then
  echo "ROS_TYPE=${ROS_TYPE:-}"
  echo "ROS_VERSION=${ROS_VERSION:-}"
  echo "ROS_DISTRO=${ROS_DISTRO:-}"
  echo "MJLAB_DEPTH_SOURCE=${MJLAB_DEPTH_SOURCE}"
  echo "MJLAB_DEPTH_SINK=${MJLAB_DEPTH_SINK}"
  echo "MJLAB_DEPTH_ROS_TOPIC=${MJLAB_DEPTH_ROS_TOPIC}"
  echo "MJLAB_DEPTH_MAX_AGE=${MJLAB_DEPTH_MAX_AGE}"
  echo "MJLAB_DEPTH_VIEW=${MJLAB_DEPTH_VIEW:-auto}"
  echo "MJLAB_DEPTH_GRADCAM=${MJLAB_DEPTH_GRADCAM}"
  echo "MJLAB_DEPTH_GRADCAM_HZ=${MJLAB_DEPTH_GRADCAM_HZ}"
  echo "MJLAB_DEPTH_GRADCAM_PATH=${MJLAB_DEPTH_GRADCAM_PATH}"
  if uses_npy_depth; then
    echo "MJLAB_DEPTH_NPY_PATH=${MJLAB_DEPTH_NPY_PATH}"
  fi
fi

start_bg "MuJoCo simulator" "${LOG_DIR}/simulator.log" \
  "${PYTHON_CMD[@]}" "${SIMULATOR}" "${ROBOT_IP}"
sleep "${SIM_START_DELAY}"

start_bg "RL controller" "${LOG_DIR}/controller.log" \
  "${PYTHON_CMD[@]}" "${CONTROLLER}" "${ROBOT_IP}"
sleep "${CTRL_START_DELAY}"

if uses_depth_viewer; then
  start_bg "depth viewer" "${LOG_DIR}/depth_viewer.log" \
    "${PYTHON_CMD[@]}" "${DEPTH_VIEWER}"
fi

echo "Starting virtual joystick in foreground. Press Ctrl-C here to stop everything."
"${JOYSTICK}"
