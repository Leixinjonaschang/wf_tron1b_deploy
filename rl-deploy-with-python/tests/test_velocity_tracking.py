from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import velocity_tracking as tracking


@pytest.mark.parametrize("step,repeats", [(5, 10), (30, 5), (10, 2)])
def test_commands_cover_circle_without_duplicate_endpoint(step, repeats):
    schedule = tracking.make_schedule(repeats, 123, step)
    assert schedule == tracking.make_schedule(repeats, 123, step)
    assert schedule != tracking.make_schedule(repeats, 124, step)
    count = repeats * (360 // step)
    assert len(schedule) == count
    assert len({t["seed"] for t in schedule}) == count
    assert len({t["trial_id"] for t in schedule}) == count
    for repeat in range(1, repeats + 1):
        directions = [t["direction_deg"] for t in schedule if t["repeat"] == repeat]
        assert sorted(directions) == list(range(0, 360, step))
    for direction in range(0, 360, step):
        vx, vy, yaw = tracking.command_for_direction(direction, 0.5)
        assert math.hypot(vx, vy) == pytest.approx(0.5)
        assert yaw == 0
    assert tracking.command_for_direction(90, 0.5)[1] == pytest.approx(0.5)


@pytest.mark.parametrize("step", [0, -5, 7, 361, 2.5])
def test_invalid_direction_step_is_rejected(step):
    with pytest.raises(ValueError, match="divisor"):
        tracking.direction_grid(step)


def test_collection_handles_worker_metadata_arriving_after_a_result(tmp_path, monkeypatch):
    # A worker's later trial can be yielded first if several futures are ready.
    args = SimpleNamespace(
        output=tmp_path / "run", rl_type="mjlab_repts_lin", speed=0.5,
        direction_step=180, repeats=1, seed=123, workers=1,
        warmup=0.02, settle=0.02, duration=0.02, min_height=0.35, max_tilt=60,
    )
    metadata_results = iter((None, {"backend": "test_fixture"}))

    def fake_trial(output, trial):
        path = f"raw/{trial['trial_id']}.csv"
        tracking.write_csv(output / path, [{"fixture": "not_experiment_data"}])
        return ({**trial, "status": "ok", "failure_reason": "", "raw_file": path},
                next(metadata_results))

    monkeypatch.setattr(tracking, "initialize_worker", lambda config: None)
    monkeypatch.setattr(tracking, "close_worker", lambda: None)
    monkeypatch.setattr(tracking, "collect_trial", fake_trial)
    tracking.collect(args)
    manifest = json.loads((args.output / "manifest.json").read_text())
    assert manifest["status"] == "collected"
    assert manifest["simulation"]["backend"] == "test_fixture"
    assert len(tracking.read_csv(args.output / "trials.csv")) == 2


def test_error_is_vector_difference_not_speed_difference():
    # Equal speed in opposite directions must NOT have zero tracking error.
    assert tracking.tracking_error(-0.5, 0, 0.5, 0) == 1
    assert tracking.tracking_error(0.2, 0.4, 0.5, 0) == pytest.approx(0.5)


def test_world_velocity_is_rotated_into_current_body_frame():
    from velocity_tracking_sim import body_linear_velocity
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    np.testing.assert_allclose(body_linear_velocity([0, 0.5, 0, 0, 0, 0], rotation),
                               [0.5, 0, 0])


def test_supplied_depth_source_bypasses_ros_creation():
    from velocity_tracking_sim import DEPLOY, LocalRobot, WheelfootController
    source = object()
    module = sys.modules[WheelfootController.__module__]
    with mock.patch.object(module, "create_depth_frame_source") as factory:
        controller = WheelfootController(
            str(DEPLOY / "controllers/model"), LocalRobot(), "WF_TRON1B",
            "mjlab_repts_gru_lin_depth", True, depth_source=source,
        )
    factory.assert_not_called()
    assert controller.depth_source is source


def test_rendered_depth_matches_ros_policy_input(monkeypatch):
    from mjlab_repts_lin_depth import _ros_image_to_depth_input
    import velocity_tracking_sim as simulation

    # Distinct columns expose crop/resize errors as well as unit/range errors.
    columns = np.linspace(0.0, 3.0, 848, dtype=np.float32)
    raw = np.tile(columns, (480, 1))
    renderer = mock.Mock()
    renderer.render.return_value = raw
    monkeypatch.setattr(simulation, "mujoco", SimpleNamespace(
        Renderer=mock.Mock(return_value=renderer),
        MjvOption=lambda: SimpleNamespace(geomgroup=np.zeros(6, dtype=int)),
    ))
    source = simulation.RenderedDepth(object())
    source.capture(object())

    wire_depth = (raw * 1000).astype(np.uint16)
    msg = SimpleNamespace(
        height=480, width=848, encoding="16UC1", is_bigendian=0,
        step=848 * 2, data=wire_depth.tobytes(),
    )
    np.testing.assert_array_equal(source.frame(), _ros_image_to_depth_input(msg))
    assert source.frame().shape == (1, 1, 30, 45)
    assert source.frame().dtype == np.float32
    assert 0 <= source.frame().min() <= source.frame().max() <= 1

    renderer.render.return_value = np.full((480, 848), 1.0, dtype=np.float32)
    source.capture(object())
    np.testing.assert_allclose(source.frame(), (1.0 - 0.2) / (2.0 - 0.2))
    source.close()


def test_injected_depth_preserves_policy_outputs_and_joint_commands(tmp_path, monkeypatch):
    from mjlab_repts_lin_depth import preprocess_depth_image
    from velocity_tracking_sim import DEPLOY, LocalRobot, WheelfootController

    raw = np.full((480, 848), 1.0, dtype=np.float32)
    depth_path = tmp_path / "depth.npy"
    np.save(depth_path, raw)
    # Exercise the existing factory and environment overrides without ROS.
    for name in tuple(os.environ):
        if name.startswith("MJLAB_DEPTH_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("MJLAB_DEPTH_SOURCE", "npy")
    monkeypatch.setenv("MJLAB_DEPTH_NPY_PATH", str(depth_path))
    source = SimpleNamespace(frame=lambda: preprocess_depth_image(raw))
    controllers = [
        WheelfootController(
            str(DEPLOY / "controllers/model"), LocalRobot(), "WF_TRON1B",
            "mjlab_repts_gru_lin_depth", False, **kwargs,
        )
        for kwargs in ({}, {"depth_source": source})
    ]
    for controller in controllers:
        controller.mjlab_repts_diagnostics_printed = True
        controller.robot_state.q = controller.init_joint_angles.tolist()

    # Use the real ONNX policy over multiple commands and recurrent updates.
    for step, command in enumerate(((0, 0, 0), (0.5, 0, 0), (0, 0.3, 0.2))):
        for controller in controllers:
            controller.commands = np.asarray(command, dtype=np.float32)
            controller.loop_count = step * controller.control_cfg["decimation"]
            controller.handle_mjlab_repts_walk_mode()
        for name in ("last_depth_input", "proprio_history_vector", "actions",
                     "predicted_lin_vel", "depth_hidden_state"):
            actual = getattr(controllers[1], name)
            assert np.isfinite(actual).all()
            np.testing.assert_array_equal(actual, getattr(controllers[0], name))
        for name in ("q", "dq", "tau", "Kp", "Kd"):
            np.testing.assert_array_equal(
                getattr(controllers[0].robot_cmd, name),
                getattr(controllers[1].robot_cmd, name),
            )


def create_dataset(run, step=30):
    (run / "raw").mkdir()
    schedule = tracking.make_schedule(3, 100, step)
    manifest = {
        "status": "collected", "schedule": schedule,
        "directions_deg": list(range(0, 360, step)),
        "config": {"speed_mps": 0.5, "repeats": 3, "warmup_s": 0.02,
                   "settle_s": 0.02, "measure_s": 0.04},
        "simulation": {"sample_dt_s": 0.02}, "files_sha256": {},
    }
    trials = []
    # Trial means are 1, 2, 9: across-repeat mean=4, median=2.
    # Within-trial medians and pooled time medians are different concepts.
    for trial in schedule:
        value = (1, 2, 9)[trial["repeat"] - 1]
        failed = trial["direction_deg"] == 90 and trial["repeat"] == 3
        cmd = tracking.command_for_direction(trial["direction_deg"], 0.5)
        rows = []
        for index, phase in enumerate(("warmup", "settle", "measure", "measure")):
            error = value if phase == "measure" else 999
            rows.append({
                "trial_id": trial["trial_id"], "phase": phase,
                "sim_time_s": (index + 1) * 0.02,
                "cmd_vx_mps": cmd[0], "cmd_vy_mps": cmd[1], "cmd_yaw_radps": 0,
                "vx_body_mps": cmd[0] + error, "vy_body_mps": cmd[1],
                "error_l2_mps": error,
                "failure_reason": "base_tilt_exceeded" if failed and index == 3 else "",
            })
        path = f"raw/{trial['trial_id']}.csv"
        tracking.write_csv(run / path, rows)
        manifest["files_sha256"][path] = tracking.digest(run / path)
        trials.append({**trial, "status": "failed" if failed else "ok",
                       "failure_reason": "base_tilt_exceeded" if failed else "",
                       "raw_file": path})
    tracking.write_csv(run / "trials.csv", trials)
    tracking.write_csv(run / "schedule.csv", schedule)
    for name in ("trials.csv", "schedule.csv"):
        manifest["files_sha256"][name] = tracking.digest(run / name)
    tracking.write_json(run / "manifest.json", manifest)
    return manifest


def test_summary_aggregates_trial_means_and_excludes_warmup(tmp_path):
    create_dataset(tmp_path)
    directions = tracking.summarize(tmp_path)
    assert directions[0]["mean_error_mps"] == pytest.approx(4)
    assert directions[0]["median_error_mps"] == pytest.approx(2)
    assert directions[3]["n_failed"] == 1
    assert directions[3]["n_success"] == 2
    assert directions[3]["mean_error_mps"] == pytest.approx(1.5)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["n_failed"] == 1
    assert summary["n_success"] == 35
    assert not list(tmp_path.glob("*.png"))


def test_five_degree_summary_keeps_all_72_directions(tmp_path):
    create_dataset(tmp_path, step=5)
    directions = tracking.summarize(tmp_path)
    assert [r["direction_deg"] for r in directions] == list(range(0, 360, 5))
    assert all(r["n_total"] == 3 for r in directions)
    assert directions[18]["n_success"] == 2
    assert directions[1]["mean_error_mps"] == pytest.approx(4)
    assert directions[71]["median_error_mps"] == pytest.approx(2)


def test_incomplete_or_modified_data_cannot_be_plotted(tmp_path):
    manifest = create_dataset(tmp_path)
    manifest["status"] = "interrupted"
    tracking.write_json(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="incomplete"):
        tracking.plot(tmp_path)
    manifest["status"] = "collected"
    tracking.write_json(tmp_path / "manifest.json", manifest)
    tracking.summarize(tmp_path)
    raw = tmp_path / f"raw/{manifest['schedule'][0]['trial_id']}.csv"
    raw.write_text(raw.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum"):
        tracking.plot(tmp_path)


def test_successful_trial_with_missing_samples_is_rejected(tmp_path):
    manifest = create_dataset(tmp_path)
    name = f"raw/{manifest['schedule'][0]['trial_id']}.csv"
    rows = tracking.read_csv(tmp_path / name)
    tracking.write_csv(tmp_path / name, rows[:-1])
    manifest["files_sha256"][name] = tracking.digest(tmp_path / name)
    tracking.write_json(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="Incomplete measure"):
        tracking.summarize(tmp_path)


def test_stale_summary_cannot_be_plotted(tmp_path):
    manifest = create_dataset(tmp_path)
    tracking.summarize(tmp_path)
    manifest["config"]["speed_mps"] = 0.6
    tracking.write_json(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="stale"):
        tracking.plot(tmp_path)


def test_nonintegral_measurement_period_is_rejected():
    with pytest.raises(ValueError, match="multiple"):
        tracking.period_count(1.001, 0.02)
