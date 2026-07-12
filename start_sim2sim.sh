#!/usr/bin/env bash
# set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_DIR="${SCRIPT_DIR}/pointfoot-mujoco-sim"
CTRL_DIR="${SCRIPT_DIR}/rl-deploy-with-python"
LOG_DIR="${SCRIPT_DIR}/logs/sim2sim"

ROBOT_TYPE="${ROBOT_TYPE:-WF_TRON1B}"
RL_TYPE="${RL_TYPE:-mjlab_repts}"
ROBOT_IP="${ROBOT_IP:-127.0.0.1}"
SIM_START_DELAY="${SIM_START_DELAY:-2}"
CTRL_START_DELAY="${CTRL_START_DELAY:-1}"
LIMXSDK_WHL="${LIMXSDK_WHL:-${SIM_DIR}/limxsdk-lowlevel/python3/amd64/limxsdk-3.4.2-py3-none-any.whl}"

SIMULATOR="${SIM_DIR}/simulator.py"
CONTROLLER="${CTRL_DIR}/main.py"
JOYSTICK="${SIM_DIR}/robot-joystick/robot-joystick"
MODEL_XML="${SIM_DIR}/robot-description/pointfoot/${ROBOT_TYPE}/xml/robot.xml"
POLICY="${CTRL_DIR}/controllers/model/${ROBOT_TYPE}/policy/${RL_TYPE}/policy.onnx"
DEPTH_FRAME_PATH="${MJLAB_DEPTH_NPY_PATH:-${LOG_DIR}/depth_frame.npy}"

PIDS=()
PYTHON_CMD=()

usage() {
  cat <<EOF
Usage:
  ROBOT_TYPE=WF_TRON1B RL_TYPE=mjlab_repts ./start_sim2sim.sh
  ROBOT_TYPE=WF_TRON1B RL_TYPE=mjlab_repts_lin_depth ./start_sim2sim.sh

Defaults:
  ROBOT_TYPE=${ROBOT_TYPE}
  RL_TYPE=${RL_TYPE}
  ROBOT_IP=${ROBOT_IP}
  SIM_START_DELAY=${SIM_START_DELAY}
  CTRL_START_DELAY=${CTRL_START_DELAY}
  LIMXSDK_WHL=${LIMXSDK_WHL}

Processes started:
  1. uv run ... python pointfoot-mujoco-sim/simulator.py
  2. uv run ... python rl-deploy-with-python/main.py
  3. pointfoot-mujoco-sim/robot-joystick/robot-joystick
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

uses_ros_depth() {
  [[ "${RL_TYPE}" == "mjlab_repts_lin_depth" ]] || return 1
  [[ "${MJLAB_DEPTH_SOURCE:-}" == "ros" || "${MJLAB_DEPTH_SINK:-}" == "ros" || "${MJLAB_DEPTH_SINK:-}" == "both" ]]
}

uses_npy_depth() {
  [[ "${RL_TYPE}" == "mjlab_repts_lin_depth" ]] || return 1
  [[ "${MJLAB_DEPTH_SOURCE:-}" == "npy" || "${MJLAB_DEPTH_SOURCE:-}" == "npy_live" || "${MJLAB_DEPTH_SINK:-}" == "npy" || "${MJLAB_DEPTH_SINK:-}" == "both" ]]
}

ensure_ros_master() {
  if ! command -v rostopic >/dev/null 2>&1; then
    echo "Missing rostopic. Source a ROS1 environment before running ROS depth." >&2
    exit 1
  fi
  if rostopic list >/dev/null 2>&1; then
    return
  fi
  if ! command -v roscore >/dev/null 2>&1; then
    echo "Missing roscore and no ROS master is reachable." >&2
    exit 1
  fi

  start_bg "roscore" "${LOG_DIR}/roscore.log" roscore
  for _ in {1..20}; do
    if rostopic list >/dev/null 2>&1; then
      return
    fi
    sleep 0.5
  done

  echo "roscore did not become ready. Last log lines:" >&2
  tail -n 40 "${LOG_DIR}/roscore.log" >&2 || true
  exit 1
}

ensure_python_ros_modules() {
  if ! "${PYTHON_CMD[@]}" -c 'import rospy; import sensor_msgs.msg' >/dev/null 2>&1; then
    cat >&2 <<EOF
The selected Python cannot import rospy and sensor_msgs.
Source ROS1 and run with a Python that can see ROS packages, for example:
  source /opt/ros/noetic/setup.bash
  PYTHON=python3 ROBOT_TYPE=WF_TRON1B RL_TYPE=mjlab_repts_lin_depth ./start_sim2sim.sh

For the legacy file-based fallback, set:
  MJLAB_DEPTH_SOURCE=npy_live MJLAB_DEPTH_SINK=npy
EOF
    exit 1
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

mkdir -p "${LOG_DIR}"
setup_python_cmd

export ROBOT_TYPE
export RL_TYPE
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
if [[ "${RL_TYPE}" == "mjlab_repts_lin_depth" ]]; then
  export MJLAB_DEPTH_SOURCE="${MJLAB_DEPTH_SOURCE:-ros}"
  export MJLAB_DEPTH_SINK="${MJLAB_DEPTH_SINK:-ros}"
  export MJLAB_DEPTH_ROS_TOPIC="${MJLAB_DEPTH_ROS_TOPIC:-/camera/depth/image_rect_raw}"
  export MJLAB_DEPTH_CAPTURE_HZ="${MJLAB_DEPTH_CAPTURE_HZ:-25.0}"
  export MJLAB_DEPTH_HEIGHT="${MJLAB_DEPTH_HEIGHT:-28}"
  export MJLAB_DEPTH_WIDTH="${MJLAB_DEPTH_WIDTH:-48}"
  export MJLAB_DEPTH_MAX_AGE="${MJLAB_DEPTH_MAX_AGE:-0.5}"
  if uses_npy_depth; then
    export MJLAB_DEPTH_NPY_PATH="${MJLAB_DEPTH_NPY_PATH:-${DEPTH_FRAME_PATH}}"
  fi
fi

trap cleanup EXIT INT TERM

if uses_ros_depth; then
  ensure_python_ros_modules
  ensure_ros_master
fi

echo "ROBOT_TYPE=${ROBOT_TYPE}"
echo "RL_TYPE=${RL_TYPE}"
echo "ROBOT_IP=${ROBOT_IP}"
echo "SIM_START_DELAY=${SIM_START_DELAY}"
echo "CTRL_START_DELAY=${CTRL_START_DELAY}"
echo "PYTHON_CMD=${PYTHON_CMD[*]}"
if [[ "${RL_TYPE}" == "mjlab_repts_lin_depth" ]]; then
  echo "MJLAB_DEPTH_SOURCE=${MJLAB_DEPTH_SOURCE}"
  echo "MJLAB_DEPTH_SINK=${MJLAB_DEPTH_SINK}"
  echo "MJLAB_DEPTH_ROS_TOPIC=${MJLAB_DEPTH_ROS_TOPIC}"
  echo "MJLAB_DEPTH_MAX_AGE=${MJLAB_DEPTH_MAX_AGE}"
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

echo "Starting virtual joystick in foreground. Press Ctrl-C here to stop everything."
"${JOYSTICK}"
