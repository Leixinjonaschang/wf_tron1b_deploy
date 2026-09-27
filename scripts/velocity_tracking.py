#!/usr/bin/env python3
"""Collect first, summarize second, plot last: directional velocity tracking."""

from __future__ import annotations

import argparse
import atexit
import csv
import hashlib
import json
import math
import multiprocessing
import os
import random
import statistics
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RL_TYPES = (
    "mjlab_repts_lin",
    "mjlab_repts_gru_lin_depth",
)
METRIC = "time_mean_of_planar_velocity_error_l2_mps"


def command_for_direction(direction_deg, speed):
    theta = math.radians(direction_deg)
    return (speed * math.cos(theta), speed * math.sin(theta), 0.0)


def tracking_error(vx, vy, cmd_vx, cmd_vy):
    return math.hypot(vx - cmd_vx, vy - cmd_vy)


def direction_grid(step_deg):
    if not isinstance(step_deg, int) or step_deg <= 0 or 360 % step_deg:
        raise ValueError("direction step must be a positive integer divisor of 360")
    return list(range(0, 360, step_deg))


def make_schedule(repeats, seed, direction_step_deg=5):
    grid = direction_grid(direction_step_deg)
    rng = random.Random(seed)
    trials = []
    for repeat in range(repeats):
        directions = list(range(len(grid)))
        rng.shuffle(directions)
        for direction in directions:
            trials.append({
                "trial_id": f"r{repeat + 1:03d}_d{grid[direction]:03d}",
                "repeat": repeat + 1, "direction_deg": grid[direction],
                "seed": seed + repeat * len(grid) + direction,
            })
    return trials


def write_json(path, data):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def period_count(duration, dt):
    count = round(duration / dt)
    if count < 1 or not math.isclose(count * dt, duration, abs_tol=1e-8):
        raise ValueError(f"Duration {duration} must be a positive multiple of {dt} s")
    return count


_WORKER_SIM = None
_WORKER_CONFIG = None
_WORKER_METADATA_SENT = False


def initialize_worker(config):
    """Each spawned process owns its physics state, ONNX session and GL context."""
    from velocity_tracking_sim import TrackingSimulation
    global _WORKER_SIM, _WORKER_CONFIG, _WORKER_METADATA_SENT
    _WORKER_CONFIG = config
    _WORKER_METADATA_SENT = False
    _WORKER_SIM = TrackingSimulation(
        config["rl_type"], config["min_height_m"], config["max_tilt_deg"],
    )
    atexit.register(close_worker)


def close_worker():
    global _WORKER_SIM
    if _WORKER_SIM is not None:
        _WORKER_SIM.close()
        _WORKER_SIM = None


def collect_trial(output, trial):
    global _WORKER_METADATA_SENT
    sim, config = _WORKER_SIM, _WORKER_CONFIG
    sim.reset(trial["seed"])
    metadata = None
    if not _WORKER_METADATA_SENT:
        metadata = sim.metadata()
        _WORKER_METADATA_SENT = True
    command = command_for_direction(trial["direction_deg"], config["speed_mps"])
    periods = [
        ("warmup", period_count(config["warmup_s"], sim.sample_dt), (0.0, 0.0, 0.0)),
        ("settle", period_count(config["settle_s"], sim.sample_dt), command),
        ("measure", period_count(config["measure_s"], sim.sample_dt), command),
    ]
    relative_path = f"raw/{trial['trial_id']}.csv"
    reason = ""
    with (output / relative_path).open("w", newline="") as stream:
        writer = None
        for phase, count, command in periods:
            for sample in range(count):
                reason = sim.step(command)
                row = {
                    **trial, "phase": phase, "phase_sample": sample,
                    "cmd_vx_mps": command[0], "cmd_vy_mps": command[1],
                    "cmd_yaw_radps": command[2], **sim.observation(),
                    "failure_reason": reason,
                }
                row["error_l2_mps"] = tracking_error(
                    row["vx_body_mps"], row["vy_body_mps"], *command[:2],
                )
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                if reason:
                    break
            stream.flush()
            if reason:
                break
    result = {
        **trial, "status": "failed" if reason else "ok",
        "failure_reason": reason, "raw_file": relative_path,
    }
    return result, metadata


def collect(args):
    # Set before importing MuJoCo/GL. The caller can select another backend.
    os.environ.setdefault("MUJOCO_GL", "egl")
    # Avoid multiplying Mesa's software-renderer threads in parallel workers.
    if args.workers > 1:
        os.environ.setdefault("LP_NUM_THREADS", "1")

    output = args.output or ROOT / "logs/velocity_tracking" / datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%S_%fZ")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "raw").mkdir()
    schedule = make_schedule(args.repeats, args.seed, args.direction_step)
    config = {
        "rl_type": args.rl_type, "speed_mps": args.speed,
        "repeats": args.repeats, "seed": args.seed,
        "direction_step_deg": args.direction_step, "workers": args.workers,
        "warmup_s": args.warmup, "settle_s": args.settle,
        "measure_s": args.duration,
        "min_height_m": args.min_height, "max_tilt_deg": args.max_tilt,
    }
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True,
    )
    manifest = {
        "schema_version": 1, "status": "collecting", "config": config,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git.stdout.strip(), "metric": METRIC,
        "frame": "instantaneous_body_xy", "yaw_rate_cmd_radps": 0,
        "directions_deg": direction_grid(args.direction_step), "schedule": schedule,
        "files_sha256": {},
    }
    write_json(output / "manifest.json", manifest)
    write_csv(output / "schedule.csv", schedule)
    pool = None
    results = {}
    print(f"Collecting into {output}", flush=True)
    try:
        if args.workers == 1:
            initialize_worker(config)
            completed = (collect_trial(output, trial) for trial in schedule)
        else:
            pool = ProcessPoolExecutor(
                max_workers=args.workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=initialize_worker, initargs=(config,),
            )
            futures = [pool.submit(collect_trial, output, trial) for trial in schedule]
            completed = (future.result() for future in as_completed(futures))
        for result, metadata in completed:
            if "simulation" not in manifest and metadata is not None:
                manifest["simulation"] = metadata
                manifest["simulation"]["python_version"] = os.sys.version
                manifest["simulation"]["collector_sha256"] = digest(Path(__file__))
                manifest["simulation"]["adapter_sha256"] = digest(
                    Path(__file__).with_name("velocity_tracking_sim.py")
                )
                write_json(output / "manifest.json", manifest)
            if metadata is not None:
                for key, value in metadata.items():
                    if manifest["simulation"][key] != value:
                        raise ValueError(f"Worker simulation metadata mismatch: {key}")
            results[result["trial_id"]] = result
            # Completion order may differ, but the saved trial list is canonical.
            write_csv(output / "trials.csv", [
                results[t["trial_id"]] for t in schedule if t["trial_id"] in results
            ])
            relative_path = result["raw_file"]
            manifest["files_sha256"][relative_path] = digest(output / relative_path)
            write_json(output / "manifest.json", manifest)
            print(f"[{len(results)}/{len(schedule)}] {result['trial_id']}: "
                  f"{result['failure_reason'] or 'ok'}", flush=True)
        for name in ("trials.csv", "schedule.csv"):
            manifest["files_sha256"][name] = digest(output / name)
        manifest["status"] = "collected"
        manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "manifest.json", manifest)
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        write_json(output / "manifest.json", manifest)
        raise
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        else:
            close_worker()
    print(f"Data collection complete. Next: python scripts/velocity_tracking.py "
          f"summarize {output}")
    return output


def verify_collection(run):
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest["status"] != "collected":
        raise ValueError("Collection is incomplete; finish a new run before summarizing")
    expected_paths = {"trials.csv", "schedule.csv"} | {
        f"raw/{t['trial_id']}.csv" for t in manifest["schedule"]
    }
    if set(manifest["files_sha256"]) != expected_paths:
        raise ValueError("Missing or unexpected data files in manifest")
    for name, checksum in manifest["files_sha256"].items():
        if digest(run / name) != checksum:
            raise ValueError(f"Data checksum mismatch: {name}")
    return manifest


def summarize(run):
    manifest = verify_collection(run)
    config = manifest["config"]
    dt = manifest["simulation"]["sample_dt_s"]
    expected = period_count(config["measure_s"], dt)
    trials = read_csv(run / "trials.csv")
    if [t["trial_id"] for t in trials] != [t["trial_id"] for t in manifest["schedule"]]:
        raise ValueError("Trial list does not match the scheduled experiment")
    summaries = []
    for trial, planned in zip(trials, manifest["schedule"]):
        for key in ("trial_id", "repeat", "seed", "direction_deg"):
            if str(planned[key]) != trial[key]:
                raise ValueError(f"Incorrect trial metadata: {trial['trial_id']} {key}")
        if trial["status"] not in ("ok", "failed"):
            raise ValueError("Unknown trial status")
        if trial["raw_file"] != f"raw/{trial['trial_id']}.csv":
            raise ValueError("Incorrect raw file path")
        rows = read_csv(run / trial["raw_file"])
        measured = [r for r in rows if r["phase"] == "measure"]
        errors, ex, ey = [], [], []
        cmd = command_for_direction(int(trial["direction_deg"]), config["speed_mps"])
        previous_time = 0.0
        for row in rows:
            if row["trial_id"] != trial["trial_id"]:
                raise ValueError("Raw sample belongs to another trial")
            if float(row["sim_time_s"]) <= previous_time:
                raise ValueError("Non-monotonic simulation timestamps")
            previous_time = float(row["sim_time_s"])
        if trial["status"] == "ok":
            for phase, duration in (("warmup", config["warmup_s"]),
                                    ("settle", config["settle_s"]),
                                    ("measure", config["measure_s"])):
                phase_rows = [r for r in rows if r["phase"] == phase]
                if len(phase_rows) != period_count(duration, dt):
                    raise ValueError(f"Incomplete {phase}: {trial['trial_id']}")
            for index, row in enumerate(rows):
                if not math.isclose(float(row["sim_time_s"]), (index + 1) * dt,
                                    abs_tol=1e-7):
                    raise ValueError("Unexpected sampling interval")
                if row["failure_reason"]:
                    raise ValueError("Failed sample marked as successful trial")
            if len(measured) != expected:
                raise ValueError("Incorrect measurement count")
            for row in measured:
                values = [float(row[k]) for k in (
                    "vx_body_mps", "vy_body_mps", "cmd_vx_mps", "cmd_vy_mps",
                    "cmd_yaw_radps", "error_l2_mps",
                )]
                if not all(math.isfinite(x) for x in values):
                    raise ValueError("Nonfinite measurement in successful trial")
                if any(not math.isclose(a, b, abs_tol=1e-7)
                       for a, b in zip(values[2:5], cmd)):
                    raise ValueError("Measured command differs from scheduled command")
                error = tracking_error(*values[:4])
                if not math.isclose(error, values[5], abs_tol=1e-7):
                    raise ValueError("Stored error differs from ground truth")
                errors.append(error)
                ex.append(values[0] - values[2])
                ey.append(values[1] - values[3])
        elif not trial["failure_reason"] or not any(r["failure_reason"] for r in rows):
            raise ValueError("Failed trial lacks a failure reason")
        summaries.append({
            **trial, "measurement_samples": len(measured),
            "mean_error_mps": statistics.fmean(errors) if errors else "",
            "rms_error_mps": math.sqrt(statistics.fmean(e * e for e in errors)) if errors else "",
            "bias_vx_mps": statistics.fmean(ex) if errors else "",
            "bias_vy_mps": statistics.fmean(ey) if errors else "",
        })
    directions = []
    for direction in manifest["directions_deg"]:
        group = [t for t in summaries if int(t["direction_deg"]) == direction]
        values = [t["mean_error_mps"] for t in group if t["status"] == "ok"]
        if len(group) != config["repeats"]:
            raise ValueError(f"Incomplete repeats for {direction} degrees")
        directions.append({
            "direction_deg": direction, "speed_mps": config["speed_mps"],
            "n_total": len(group), "n_success": len(values),
            "n_failed": len(group) - len(values),
            "mean_error_mps": statistics.fmean(values) if values else "",
            "median_error_mps": statistics.median(values) if values else "",
            "std_error_mps": statistics.stdev(values) if len(values) > 1 else "",
        })
    write_csv(run / "trial_summary.csv", summaries)
    write_csv(run / "direction_summary.csv", directions)
    write_json(run / "summary.json", {
        "metric": METRIC, "aggregation": "mean_and_median_across_trial_means",
        "failure_handling": "excluded_from_error_statistics; counts_always_reported",
        "manifest_sha256": digest(run / "manifest.json"),
        "files_sha256": {name: digest(run / name) for name in (
            "trial_summary.csv", "direction_summary.csv",
        )},
        "n_trials": len(summaries),
        "n_success": sum(t["status"] == "ok" for t in summaries),
        "n_failed": sum(t["status"] == "failed" for t in summaries),
    })
    print(f"Validated {len(summaries)} trials. Summaries written to {run}")
    return directions


def plot(run):
    # Validate all saved data BEFORE importing any plotting package.
    manifest = verify_collection(run)
    summary = json.loads((run / "summary.json").read_text())
    if summary["manifest_sha256"] != digest(run / "manifest.json"):
        raise ValueError("Summary is stale; run summarize again")
    for name, checksum in summary["files_sha256"].items():
        if digest(run / name) != checksum:
            raise ValueError(f"Summary checksum mismatch: {name}")
    directions = read_csv(run / "direction_summary.csv")
    trials = read_csv(run / "trial_summary.csv")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    good = [t for t in trials if t["status"] == "ok"]
    values = [float(t["mean_error_mps"]) for t in good]
    limit = 1.0  # Fixed physical scale: the outer circle represents 1 m/s.
    if any(value > limit for value in values):
        raise ValueError("Trial error exceeds the fixed 1 m/s radius; refusing to clip data")
    norm = Normalize(0, limit)
    config = manifest["config"]
    fig = plt.figure(figsize=(8.2, 8.5), facecolor="white")
    ax = fig.add_axes([0.12, 0.14, 0.76, 0.66], projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(1)
    ax.set_ylim(0, limit)
    ax.set_thetagrids(range(0, 360, 45))
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_rlabel_position(45)
    ax.tick_params(axis="x", pad=10, labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(color="#a9a9a9", linewidth=0.7, alpha=0.8)
    ax.spines["polar"].set_linewidth(0.9)
    ax.scatter(
        [math.radians(float(t["direction_deg"])) for t in good], values,
        c=values, cmap="RdYlBu_r", norm=norm, s=20, alpha=0.95,
        linewidths=0, zorder=2,
    )

    # Connect only measured direction statistics; NaNs break failed directions.
    # Close the circular seam explicitly, without smoothing or angular jitter.
    angles = [math.radians(float(row["direction_deg"])) for row in directions]
    angles.append(angles[0] + 2 * math.pi)
    overall = (
        statistics.fmean(values) if values else None,
        statistics.median(values) if values else None,
    )
    for key, label, linestyle, value in zip(
        ("mean_error_mps", "median_error_mps"), ("Mean", "Median"),
        ("-", "--"), overall,
    ):
        radial = [float(row[key]) if row[key] != "" else math.nan for row in directions]
        radial.append(radial[0])
        legend = f"{label}: {value:.3f} m/s" if value is not None else f"{label}: n/a"
        ax.plot(angles, radial, color="black", linestyle=linestyle,
                linewidth=1.6, label=legend, zorder=3)
    for row in directions:
        if int(row["n_failed"]):
            ax.text(math.radians(float(row["direction_deg"])), limit * 0.95,
                    f"{row['n_success']}/{row['n_total']}", color="firebrick",
                    ha="center", fontsize=8)
    fig.text(0.5, 0.88, config["rl_type"], ha="center", va="center",
             fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.16, 1.12), fontsize=10,
              frameon=True, facecolor="white", edgecolor="#cccccc", framealpha=1)
    fig.suptitle("Velocity Tracking Error", fontsize=17, fontweight="bold", y=0.99)
    step = config.get("direction_step_deg", 360 / len(directions))
    fig.text(
        0.5, 0.938,
        f"{step:g}° spacing · {config['repeats']} trials/direction · "
        f"Command speed: {config['speed_mps']:g} m/s",
        ha="center", va="top", fontsize=12, fontweight="bold",
    )
    fig.text(0.032, 0.49, "Error (m/s)", rotation="vertical", va="center", fontsize=11)
    fig.text(
        0.5, 0.052,
        "Points: per-trial mean errors. Curves: mean / median at each direction.\n"
        f"Legend: overall statistics of {len(good)} trials. "
        "Radius & color scale: 0–1 m/s.",
        ha="center", va="center", fontsize=8.5, color="#444444", linespacing=1.6,
    )
    if summary["n_failed"]:
        fig.text(0.5, 0.015, f"Failed trials excluded: {summary['n_failed']}",
                 ha="center", color="firebrick", fontsize=9)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(run / f"velocity_tracking_polar.{extension}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"Saved polar PNG, PDF and SVG to {run}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    collect_parser = sub.add_parser("collect", help="Save raw data; never plots")
    collect_parser.add_argument("--rl-type", choices=RL_TYPES,
                                default=os.getenv("RL_TYPE", "mjlab_repts_gru_lin_depth"))
    collect_parser.add_argument("--speed", type=float, default=0.5)
    collect_parser.add_argument("--repeats", type=int, default=10)
    collect_parser.add_argument("--direction-step", type=int, default=5,
                                help="Direction spacing in degrees; must divide 360")
    collect_parser.add_argument("--workers", type=int, default=1,
                                help="Independent simulation processes")
    collect_parser.add_argument("--seed", type=int, default=20260926)
    collect_parser.add_argument("--warmup", type=float, default=2.0,
                                help="Zero-command policy warmup from XML pose [s]")
    collect_parser.add_argument("--settle", type=float, default=3.0)
    collect_parser.add_argument("--duration", type=float, default=10.0)
    collect_parser.add_argument("--min-height", type=float, default=0.35)
    collect_parser.add_argument("--max-tilt", type=float, default=60.0)
    collect_parser.add_argument("--output", type=Path)
    for action in ("summarize", "plot"):
        sub.add_parser(action).add_argument("run", type=Path)
    args = parser.parse_args()
    if args.action == "collect":
        if args.repeats < 1 or args.seed < 0:
            parser.error("repeats must be positive and seed nonnegative")
        if args.workers < 1:
            parser.error("workers must be positive")
        try:
            direction_grid(args.direction_step)
        except ValueError as exc:
            parser.error(str(exc))
        for name in ("speed", "warmup", "settle", "duration", "min_height", "max_tilt"):
            if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
                parser.error(f"{name} must be finite and positive")
        if args.speed > 1:
            parser.error("Full-circle speed must be <= 1 m/s (lateral/reverse command range)")
        if args.max_tilt >= 90:
            parser.error("max-tilt must be below 90 degrees")
        collect(args)
    elif args.action == "summarize":
        summarize(args.run.resolve())
    else:
        plot(args.run.resolve())


if __name__ == "__main__":
    main()
