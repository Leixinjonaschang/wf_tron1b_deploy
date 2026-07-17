# ROS1 Noetic for sim2sim — via RoboStack on Python 3.11
# Why RoboStack (not the official ros:noetic image):
#   Noetic's own Python is 3.8, but your deps (mujoco/onnxruntime/scipy) and your
#   code's `X | None` typing need Python >=3.11. RoboStack ships ROS Noetic as conda
#   packages that run on 3.11, so rospy + your modern deps live in ONE env.
#
# Build context = your repo ROOT (the folder that contains start_sim2sim.sh).

FROM condaforge/miniforge3:latest

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# System GL/X libraries the MuJoCo viewer + offscreen depth renderer need
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglx-mesa0 libgl1-mesa-dri libegl1 libglib2.0-0 \
      libx11-6 libxext6 libxrender1 libxi6 libxrandr2 libxcursor1 libxinerama1 libxfixes3 \
      fontconfig libdbus-1-3 libxkbcommon-x11-0 libxcb-xinerama0 libxcb-icccm4 \
      libxcb-image0 libxcb-keysyms1 libxcb-render-util0 libxcb-cursor0 \
    && rm -rf /var/lib/apt/lists/*

# ROS Noetic (RoboStack) + desktop GUI tools on Python 3.11.
# The strict channel priority keeps RoboStack/conda-forge solves reproducible.
RUN mamba config set channel_priority strict && \
    mamba create -y -n sim -c conda-forge -c robostack-noetic \
      python=3.11 \
      ros-noetic-desktop \
      libopencv=4.10.* && \
    mamba clean -ya

ENV CONDA_DEFAULT_ENV=sim
ENV PATH=/opt/conda/envs/sim/bin:$PATH
ENV ROS_VERSION=1
ENV ROS_DISTRO=noetic
ENV ROS_PYTHON_VERSION=3

# LimX SDK wheel (pure-python, installs on 3.11). Path is relative to repo root.
COPY pointfoot-mujoco-sim/limxsdk-lowlevel/python3/amd64/limxsdk-3.4.2-py3-none-any.whl /tmp/

# Runtime Python deps. Keep these pins aligned with pyproject.toml/uv.lock and
# LimX's numpy constraint (<1.26.4).
RUN python -m pip install --no-cache-dir \
      uv==0.9.7 \
      mujoco==3.10.0 \
      onnxruntime==1.27.0 \
      numpy==1.26.3 \
      scipy==1.15.3 \
      pyyaml==6.0.3 \
      pygame==2.6.1 \
      pandas==3.0.3 \
      /tmp/limxsdk-3.4.2-py3-none-any.whl && \
    rm -f /tmp/limxsdk-3.4.2-py3-none-any.whl

# Build-time smoke test for the exact imports and ROS1 tools needed by depth sim2sim.
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate sim && \
    uv --version && \
    command -v roscore && command -v rostopic && \
    command -v roslaunch && command -v rosmaster && \
    command -v rqt_image_view && \
    ldd "${CONDA_PREFIX}/lib/librqt_image_view.so" >/tmp/rqt-image-view-ldd.log && \
    ! grep -q "not found" /tmp/rqt-image-view-ldd.log && \
    roscore --help >/dev/null && \
    ( \
      rosmaster --core >/tmp/rosmaster-smoke.log 2>&1 & \
      ros_pid=$!; \
      for _ in {1..40}; do \
        rostopic list >/dev/null 2>&1 && break; \
        if ! kill -0 "${ros_pid}" 2>/dev/null; then \
          cat /tmp/rosmaster-smoke.log >&2; \
          exit 1; \
        fi; \
        sleep 0.25; \
      done; \
      rostopic list >/dev/null; \
      status=$?; \
      kill "${ros_pid}" 2>/dev/null || true; \
      wait "${ros_pid}" 2>/dev/null || true; \
      exit "${status}"; \
    ) && \
    python -c "import rospy, sensor_msgs.msg, cv_bridge, mujoco, onnxruntime, scipy, limxsdk; print('ok')"

# Auto-activate the env in interactive shells too.
RUN echo "source /opt/conda/etc/profile.d/conda.sh && conda activate sim" >> /root/.bashrc

WORKDIR /work
CMD ["bash"]
