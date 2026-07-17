#!/usr/bin/env python3
"""Smoke-check ROS1 depth publish/subscribe and policy-side preprocessing."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_ROOT = REPO_ROOT / "rl-deploy-with-python"
sys.path.insert(0, str(DEPLOY_ROOT))

from mjlab_repts_lin_depth import DEPTH_INPUT_SHAPE, create_depth_frame_source  # noqa: E402


TOPIC = os.getenv("MJLAB_DEPTH_ROS_TOPIC", "/camera/depth/image_rect_raw")


def _rostopic_list_ready() -> bool:
    try:
        subprocess.run(
            ["rostopic", "list"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=2.0,
        )
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def _wait_for_ros_master(process: subprocess.Popen, label: str) -> subprocess.Popen:
    deadline = time.time() + 10.0
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"{label} exited early with code {process.returncode}")
        if _rostopic_list_ready():
            return process
        time.sleep(0.2)
    process.terminate()
    raise RuntimeError(f"{label} did not become ready")


def _start_ros_master_if_needed() -> subprocess.Popen | None:
    if _rostopic_list_ready():
        return None

    if shutil.which("roscore") is not None:
        roscore = subprocess.Popen(
            ["roscore"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            return _wait_for_ros_master(roscore, "roscore")
        except RuntimeError:
            roscore.terminate()
            try:
                roscore.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                roscore.kill()

    if shutil.which("rosmaster") is None:
        raise RuntimeError("no ROS master is reachable, and rosmaster is not on PATH")

    rosmaster = subprocess.Popen(
        ["rosmaster", "--core"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return _wait_for_ros_master(rosmaster, "rosmaster --core")


def _publish_one_depth_frame(topic: str) -> None:
    import rospy
    from sensor_msgs.msg import Image

    pub = rospy.Publisher(topic, Image, queue_size=1)
    deadline = time.time() + 5.0
    while pub.get_num_connections() < 1 and time.time() < deadline:
        time.sleep(0.02)

    msg = Image()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = "smoke_depth_frame"
    msg.height = 2
    msg.width = 2
    msg.encoding = "16UC1"
    msg.is_bigendian = 0
    msg.step = msg.width * 2
    msg.data = np.array([1000, 2000, 3000, 4000], dtype="<u2").tobytes()
    pub.publish(msg)


def main() -> int:
    os.environ["ROS_TYPE"] = "ros1"
    ros_master = _start_ros_master_if_needed()
    source = None
    try:
        import rospy

        if not rospy.core.is_initialized():
            rospy.init_node("wf_tron1b_depth_smoke", anonymous=True, disable_signals=True)
        source = create_depth_frame_source(
            {
                "source": "ros",
                "ros_topic": TOPIC,
                "ros_type": "ros1",
                "timeout_s": 2.0,
                "max_age_s": 2.0,
            }
        )
        _publish_one_depth_frame(TOPIC)
        frame = source.frame()
        assert frame.shape == DEPTH_INPUT_SHAPE, frame.shape
        np.testing.assert_allclose(frame[0, 0, 0, 0], 1.0, rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(frame[0, 0, -1, -1], 4.0, rtol=0.0, atol=1.0e-6)
        print(f"ROS1 depth smoke passed on {TOPIC}: shape={frame.shape}")
        return 0
    finally:
        if source is not None:
            source.close()
        if ros_master is not None:
            ros_master.terminate()
            try:
                ros_master.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                ros_master.kill()


if __name__ == "__main__":
    raise SystemExit(main())
