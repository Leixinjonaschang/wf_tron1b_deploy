#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-tron_sim2sim:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-tron_deploy}"

if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  if [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}")" == "true" ]]; then
    echo "Container already running: ${CONTAINER_NAME}"
  else
    echo "Starting existing container: ${CONTAINER_NAME}"
    docker start "${CONTAINER_NAME}" >/dev/null
  fi
  echo "Enter with: docker exec -it ${CONTAINER_NAME} bash"
  exit 0
fi

if [[ -z "${DISPLAY:-}" ]]; then
  echo "DISPLAY is not set; GUI windows will not be available inside the container." >&2
fi

XAUTH_FILE="${XAUTH_FILE:-/tmp/wf_tron1b_docker.xauth}"
touch "${XAUTH_FILE}"
if [[ -n "${DISPLAY:-}" ]] && command -v xauth >/dev/null 2>&1; then
  xauth nlist "${DISPLAY}" 2>/dev/null | sed -e 's/^..../ffff/' | xauth -f "${XAUTH_FILE}" nmerge - 2>/dev/null || true
fi

DOCKER_DEVICES=()
if [[ -e /dev/dri ]]; then
  DOCKER_DEVICES=(--device /dev/dri)
fi

docker run -d \
  --name "${CONTAINER_NAME}" \
  --network host \
  -e DISPLAY="${DISPLAY:-}" \
  -e XAUTHORITY=/tmp/.docker.xauth \
  -e ROBOT_TYPE="${ROBOT_TYPE:-WF_TRON1B}" \
  -e RL_TYPE="${RL_TYPE:-mjlab_repts_lin_depth}" \
  -e PYTHON=python \
  -e ROS_TYPE=ros1 \
  -e MJLAB_DEPTH_SOURCE="${MJLAB_DEPTH_SOURCE:-ros}" \
  -e MJLAB_DEPTH_SINK="${MJLAB_DEPTH_SINK:-ros}" \
  -e MJLAB_DEPTH_ROS_TOPIC="${MJLAB_DEPTH_ROS_TOPIC:-/camera/depth/image_rect_raw}" \
  -e MJLAB_DEPTH_TIMEOUT="${MJLAB_DEPTH_TIMEOUT:-2.0}" \
  -e MJLAB_DEPTH_CAPTURE_HZ="${MJLAB_DEPTH_CAPTURE_HZ:-25.0}" \
  -e MJLAB_DEPTH_HEIGHT="${MJLAB_DEPTH_HEIGHT:-28}" \
  -e MJLAB_DEPTH_WIDTH="${MJLAB_DEPTH_WIDTH:-48}" \
  -e MJLAB_DEPTH_MAX_AGE="${MJLAB_DEPTH_MAX_AGE:-0.5}" \
  -e MJLAB_DEPTH_NPY_PATH="${MJLAB_DEPTH_NPY_PATH:-/work/logs/sim2sim/depth_frame.npy}" \
  -e MJLAB_DEPTH_VIEW="${MJLAB_DEPTH_VIEW:-1}" \
  -e MJLAB_DEPTH_VIEW_SOURCE="${MJLAB_DEPTH_VIEW_SOURCE:-${MJLAB_DEPTH_SOURCE:-ros}}" \
  -e MJLAB_DEPTH_VIEW_SCALE="${MJLAB_DEPTH_VIEW_SCALE:-10}" \
  -e MJLAB_DEPTH_VIEW_MIN="${MJLAB_DEPTH_VIEW_MIN:-0.0}" \
  -e MJLAB_DEPTH_VIEW_MAX="${MJLAB_DEPTH_VIEW_MAX:-10.0}" \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "${XAUTH_FILE}:/tmp/.docker.xauth:ro" \
  -v "${REPO_DIR}:/work:rw" \
  "${DOCKER_DEVICES[@]}" \
  "${IMAGE_NAME}" \
  bash -lc 'source /opt/conda/etc/profile.d/conda.sh && set +u && conda activate sim && set -u && cd /work && sleep infinity'

echo "Started container: ${CONTAINER_NAME}"
echo "Enter with: docker exec -it ${CONTAINER_NAME} bash"
