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
    && rm -rf /var/lib/apt/lists/*

# ROS Noetic (RoboStack) + rospy + sensor_msgs on Python 3.11.
# The strict channel priority keeps RoboStack/conda-forge solves reproducible.
RUN mamba config set channel_priority strict && \
    mamba create -y -n sim python=3.11 && \
    mamba install -y -n sim -c robostack-staging -c conda-forge \
      ros-noetic-ros-base ros-noetic-rospy ros-noetic-sensor-msgs && \
    mamba clean -ya

ENV CONDA_DEFAULT_ENV=sim
ENV PATH=/opt/conda/envs/sim/bin:$PATH

# LimX SDK wheel (pure-python, installs on 3.11). Path is relative to repo root.
COPY pointfoot-mujoco-sim/limxsdk-lowlevel/python3/amd64/limxsdk-3.4.2-py3-none-any.whl /tmp/

# Runtime Python deps. Keep these pins aligned with pyproject.toml/uv.lock and
# LimX's numpy constraint (<1.26.4).
RUN python -m pip install --no-cache-dir \
      mujoco==3.10.0 \
      onnxruntime==1.27.0 \
      numpy==1.26.3 \
      scipy==1.15.3 \
      pyyaml==6.0.3 \
      pygame==2.6.1 \
      pandas==3.0.3 \
      /tmp/limxsdk-3.4.2-py3-none-any.whl && \
    rm -f /tmp/limxsdk-3.4.2-py3-none-any.whl

# Build-time smoke test for the exact imports needed by ROS depth sim2sim.
RUN python -c "import rospy, sensor_msgs.msg, mujoco, onnxruntime, scipy, limxsdk; print('ok')"

# Auto-activate the env in interactive shells too.
RUN echo "source /opt/conda/etc/profile.d/conda.sh && conda activate sim" >> /root/.bashrc

WORKDIR /work
CMD ["bash"]
