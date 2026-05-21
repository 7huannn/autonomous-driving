#!/usr/bin/env python3
"""Analyze Stage 14 live rollout directories and summarize metric quality."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Stage 14 live rollout metrics")
    parser.add_argument("--planner-dir", type=Path, required=True, help="Planner rollout directory")
    parser.add_argument("--random-dir", type=Path, required=True, help="Random baseline rollout directory")
    parser.add_argument("--autopilot-noise-dir", type=Path, required=True, help="Autopilot-noise baseline rollout directory")
    parser.add_argument("--output-json", type=Path, required=True, help="Output comparison JSON path")
    parser.add_argument("--output-md", type=Path, default=None, help="Optional output comparison markdown path")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def to_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def sorted_metadata_files(meta_dir: Path) -> list[Path]:
    return sorted(meta_dir.glob("*.json"))


def parse_location(meta: dict[str, Any]) -> tuple[float, float, float] | None:
    ego = meta.get("ego_vehicle")
    if not isinstance(ego, dict):
        return None
    loc = ego.get("location")
    if not isinstance(loc, dict):
        return None
    x = to_float(loc.get("x"))
    y = to_float(loc.get("y"))
    z = to_float(loc.get("z"))
    if x is None or y is None or z is None:
        return None
    return (x, y, z)


def metric_from_series(series: list[float], *, reduce: str, default: float = 0.0) -> float:
    if not series:
        return default
    if reduce == "mean":
        return float(sum(series) / len(series))
    if reduce == "max":
        return float(max(series))
    if reduce == "sum":
        return float(sum(series))
    raise ValueError(f"Unknown reducer: {reduce}")


def analyze_run(run_name: str, rollout_dir: Path) -> dict[str, Any]:
    rollout_dir = rollout_dir.resolve()
    summary_path = rollout_dir / "recording_summary.json"
    planning_path = rollout_dir / "planning_metrics.json"
    metadata_dir = rollout_dir / "metadata"

    summary = load_json(summary_path)
    planning = load_json(planning_path)
    meta_files = sorted_metadata_files(metadata_dir)

    speeds: list[float] = []
    progress_values: list[float] = []
    rewards: list[float] = []
    collisions: list[float] = []
    lane_invasion_frames = 0
    offroad_frames = 0
    stuck_frames = 0
    distance_traveled_m = 0.0
    previous_loc: tuple[float, float, float] | None = None
    last_done_reason: str | None = None

    for path in meta_files:
        meta = load_json(path)
        telemetry = meta.get("telemetry", {}) if isinstance(meta.get("telemetry"), dict) else {}
        wm = meta.get("world_model", {}) if isinstance(meta.get("world_model"), dict) else {}

        speed = to_float(telemetry.get("speed_kmh"))
        if speed is not None:
            speeds.append(speed)

        progress = to_float(telemetry.get("progress_m"))
        if progress is not None:
            progress_values.append(progress)

        reward = to_float(wm.get("reward"))
        if reward is not None:
            rewards.append(reward)

        collision = to_float(telemetry.get("collision_intensity"))
        if collision is None:
            collision = 0.0
        collisions.append(collision)

        if to_bool(telemetry.get("lane_invasion")):
            lane_invasion_frames += 1
        if to_bool(telemetry.get("offroad")):
            offroad_frames += 1
        if to_bool(telemetry.get("stuck")):
            stuck_frames += 1

        loc = parse_location(meta)
        if loc is not None and previous_loc is not None:
            distance_traveled_m += math.dist(previous_loc, loc)
        if loc is not None:
            previous_loc = loc

        done_reason = wm.get("done_reason")
        if isinstance(done_reason, str) and done_reason.strip():
            last_done_reason = done_reason.strip()

    frames_from_summary = summary.get("frames_recorded")
    frames_recorded = int(frames_from_summary) if isinstance(frames_from_summary, int) else len(meta_files)

    episode_reward = None
    wm_summary = summary.get("world_model", {}) if isinstance(summary.get("world_model"), dict) else {}
    if to_float(wm_summary.get("episode_reward")) is not None:
        episode_reward = float(wm_summary["episode_reward"])
    elif rewards:
        episode_reward = float(sum(rewards))

    done_reason = wm_summary.get("done_reason") if isinstance(wm_summary.get("done_reason"), str) else None
    if (done_reason is None or not done_reason.strip()) and last_done_reason:
        done_reason = last_done_reason

    progress_final = progress_values[-1] if progress_values else 0.0
    progress_total = 0.0
    for prev, curr in zip(progress_values[:-1], progress_values[1:]):
        progress_total += max(0.0, curr - prev)
    if not progress_values:
        progress_total = 0.0

    collision_sum = metric_from_series(collisions, reduce="sum", default=0.0)
    collision_max = metric_from_series(collisions, reduce="max", default=0.0)
    collision_frames = sum(1 for value in collisions if value > 0.0)
    stuck_ratio = (float(stuck_frames) / float(frames_recorded)) if frames_recorded > 0 else 0.0

    policy = planning.get("policy")
    if not isinstance(policy, str) or not policy.strip():
        policy = wm_summary.get("policy") if isinstance(wm_summary.get("policy"), str) else run_name

    planner_mean = to_float(planning.get("mean_planner_pred_reward"))
    random_mean = to_float(planning.get("mean_random_pred_reward"))
    planner_beats_random_pred = bool(planning.get("planner_beats_random"))

    return {
        "run": run_name,
        "rollout_dir": str(rollout_dir),
        "policy": str(policy),
        "frames_recorded": int(frames_recorded),
        "episode_reward": episode_reward,
        "done_reason": done_reason,
        "progress_final_m": float(progress_final),
        "progress_total_m": float(progress_total),
        "distance_traveled_m": float(distance_traveled_m),
        "avg_speed_kmh": metric_from_series(speeds, reduce="mean", default=0.0),
        "max_speed_kmh": metric_from_series(speeds, reduce="max", default=0.0),
        "collision_max_intensity": float(collision_max),
        "collision_sum_intensity": float(collision_sum),
        "collision_frames": int(collision_frames),
        "lane_invasion_frames": int(lane_invasion_frames),
        "offroad_frames": int(offroad_frames),
        "stuck_frames": int(stuck_frames),
        "stuck_ratio": float(stuck_ratio),
        "planner_mean_pred_reward": planner_mean,
        "planner_mean_random_pred_reward": random_mean,
        "planner_beats_random_predicted": bool(planner_beats_random_pred),
    }


def beats_by_safety(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return (
        float(a["collision_sum_intensity"]) <= float(b["collision_sum_intensity"])
        and int(a["collision_frames"]) <= int(b["collision_frames"])
        and int(a["lane_invasion_frames"]) <= int(b["lane_invasion_frames"])
        and int(a["offroad_frames"]) <= int(b["offroad_frames"])
        and float(a["stuck_ratio"]) <= float(b["stuck_ratio"])
    )


def build_aggregate_decisions(
    planner: dict[str, Any],
    random_run: dict[str, Any],
    autopilot_noise: dict[str, Any],
) -> dict[str, Any]:
    planner_reward = to_float(planner.get("episode_reward"))
    random_reward = to_float(random_run.get("episode_reward"))
    autopilot_reward = to_float(autopilot_noise.get("episode_reward"))

    planner_progress = float(planner.get("progress_total_m", 0.0))
    random_progress = float(random_run.get("progress_total_m", 0.0))
    autopilot_progress = float(autopilot_noise.get("progress_total_m", 0.0))

    beats_random_by_reward = (
        planner_reward is not None and random_reward is not None and planner_reward > random_reward
    )
    beats_random_by_progress = planner_progress > random_progress
    beats_random_by_safety = beats_by_safety(planner, random_run)

    beats_autopilot_noise_by_reward = (
        planner_reward is not None and autopilot_reward is not None and planner_reward > autopilot_reward
    )
    beats_autopilot_noise_by_progress = planner_progress > autopilot_progress

    planner_stuck = int(planner.get("stuck_frames", 0)) > 0 or str(planner.get("done_reason") or "") == "stuck"
    planner_beats_random_only_by_reward = (
        beats_random_by_reward and not beats_random_by_progress and not beats_random_by_safety
    )

    summary_line = "Planner does not beat random on reward."
    if beats_random_by_reward and planner_stuck:
        summary_line = "Planner beats random by reward but still gets stuck."
    elif beats_random_by_reward and not planner_stuck:
        summary_line = "Planner beats random by reward in this smoke run."

    return {
        "beats_random_by_reward": bool(beats_random_by_reward),
        "beats_random_by_progress": bool(beats_random_by_progress),
        "beats_random_by_safety": bool(beats_random_by_safety),
        "beats_autopilot_noise_by_reward": bool(beats_autopilot_noise_by_reward),
        "beats_autopilot_noise_by_progress": bool(beats_autopilot_noise_by_progress),
        "planner_stuck": bool(planner_stuck),
        "planner_beats_random_only_by_reward": bool(planner_beats_random_only_by_reward),
        "summary": summary_line,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    runs = payload["runs"]
    decisions = payload["aggregate_decisions"]
    lines: list[str] = []
    lines.append("# Stage 14 Live Comparison")
    lines.append("")
    lines.append("| run | policy | frames | episode_reward | done_reason | progress_total_m | distance_traveled_m | avg_speed_kmh | collision_sum | stuck_frames | stuck_ratio |")
    lines.append("|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|")
    for run in runs:
        lines.append(
            "| {run} | {policy} | {frames} | {reward:.6f} | {done} | {progress:.6f} | {distance:.6f} | {avg_speed:.6f} | {collision:.6f} | {stuck_frames} | {stuck_ratio:.6f} |".format(
                run=run["run"],
                policy=run["policy"],
                frames=run["frames_recorded"],
                reward=float(run["episode_reward"]) if run["episode_reward"] is not None else float("nan"),
                done=str(run["done_reason"]),
                progress=float(run["progress_total_m"]),
                distance=float(run["distance_traveled_m"]),
                avg_speed=float(run["avg_speed_kmh"]),
                collision=float(run["collision_sum_intensity"]),
                stuck_frames=int(run["stuck_frames"]),
                stuck_ratio=float(run["stuck_ratio"]),
            )
        )
    lines.append("")
    lines.append("## Aggregate Decisions")
    lines.append("")
    for key in [
        "beats_random_by_reward",
        "beats_random_by_progress",
        "beats_random_by_safety",
        "beats_autopilot_noise_by_reward",
        "beats_autopilot_noise_by_progress",
        "planner_stuck",
        "planner_beats_random_only_by_reward",
    ]:
        lines.append(f"- `{key}`: `{decisions[key]}`")
    lines.append(f"- summary: {decisions['summary']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    planner = analyze_run("stage14_live_planner_100", args.planner_dir)
    random_run = analyze_run("stage14_live_random_100", args.random_dir)
    autopilot_noise = analyze_run("stage14_live_autopilot_noise_100", args.autopilot_noise_dir)

    decisions = build_aggregate_decisions(planner, random_run, autopilot_noise)
    payload = {
        "runs": [planner, random_run, autopilot_noise],
        "aggregate_decisions": decisions,
    }

    args.output_json = args.output_json.resolve()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[PASS] wrote {args.output_json}")

    if args.output_md is not None:
        args.output_md = args.output_md.resolve()
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(args.output_md, payload)
        print(f"[PASS] wrote {args.output_md}")

    print(json.dumps(decisions, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
