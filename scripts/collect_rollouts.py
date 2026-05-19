#!/usr/bin/env python3
"""Stage 13 rollout collector with retry/resume and manifest writing."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from world_model import ensure_stage13_scope, load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Stage 13 CARLA rollouts with retry/resume")
    parser.add_argument("--config", type=Path, default=Path("configs/sensor_config.yaml"), help="Sensor config")
    parser.add_argument(
        "--world-model-config",
        type=Path,
        default=Path("configs/world_model.yaml"),
        help="World model config containing stage13.mode and reward/planner settings",
    )
    parser.add_argument("--num-rollouts", type=int, required=True, help="Number of rollout directories to produce")
    parser.add_argument("--frames-per-rollout", type=int, required=True, help="Frames per rollout")
    parser.add_argument("--policy", default="autopilot_noise", help="Policy label for manifest")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output root for rollout_NNNNN directories")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    parser.add_argument("--max-retries", type=int, default=2, help="Retries per rollout on recorder failure")
    parser.add_argument("--allow-stage13-control", action="store_true", help="Required for live_planning/iterative mode")
    parser.add_argument("--overwrite", action="store_true", help="Delete existing rollout subdirs before reruns")
    return parser.parse_args()


def load_sensor_config(path: Path) -> dict[str, Any]:
    return load_yaml_config(path)


def is_rollout_complete(rollout_dir: Path) -> bool:
    complete_path = rollout_dir / "dataset_complete.json"
    if not complete_path.exists():
        return False
    try:
        payload = json.loads(complete_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(payload.get("complete", False))


def load_summary(rollout_dir: Path) -> dict[str, Any]:
    summary_path = rollout_dir / "recording_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing recording summary: {summary_path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid summary format: {summary_path}")
    return payload


def write_rollout_manifest(
    rollout_dir: Path,
    rollout_id: str,
    policy: str,
    sensor_config: dict[str, Any],
    world_model_config: dict[str, Any],
    frames: int,
) -> dict[str, Any]:
    planner_cfg = world_model_config.get("planner", {}) if isinstance(world_model_config.get("planner"), dict) else {}
    rollout_cfg = world_model_config.get("rollout", {}) if isinstance(world_model_config.get("rollout"), dict) else {}
    manifest = {
        "rollout_id": rollout_id,
        "policy": str(policy),
        "map": sensor_config.get("carla", {}).get("map", "unknown"),
        "weather": sensor_config.get("carla", {}).get("weather", "unknown"),
        "frames": int(frames),
        "action_order": list(planner_cfg.get("action_order", rollout_cfg.get("action_order", ["steer", "throttle", "brake"]))),
        "reward_version": str(rollout_cfg.get("reward_version", "carla_progress_v1")),
        "terminal_version": str(rollout_cfg.get("terminal_version", "carla_terminal_v1")),
    }
    path = rollout_dir / "rollout_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def run_one_rollout(
    args: argparse.Namespace,
    rollout_dir: Path,
    rollout_seed: int,
) -> None:
    cmd = [
        sys.executable,
        "scripts/carla_recorder.py",
        "--config",
        str(args.config),
        "--world-model-config",
        str(args.world_model_config),
        "--output-dir",
        str(rollout_dir),
        "--num-frames",
        str(args.frames_per_rollout),
        "--policy",
        str(args.policy),
        "--seed",
        str(rollout_seed),
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    if args.allow_stage13_control:
        cmd.append("--allow-stage13-control")

    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"carla_recorder failed: {proc.stderr.strip() or proc.stdout.strip()}")


def main() -> int:
    args = parse_args()
    if args.num_rollouts <= 0:
        raise ValueError("--num-rollouts must be > 0")
    if args.frames_per_rollout <= 0:
        raise ValueError("--frames-per-rollout must be > 0")

    sensor_config = load_sensor_config(args.config.resolve())
    world_model_config = load_yaml_config(args.world_model_config.resolve())
    ensure_stage13_scope(world_model_config, allow_stage13_control=bool(args.allow_stage13_control))

    out_root = args.output_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    manifests: list[dict[str, Any]] = []
    for idx in range(args.num_rollouts):
        rollout_id = f"rollout_{idx:05d}"
        rollout_dir = out_root / rollout_id
        if is_rollout_complete(rollout_dir):
            summary = load_summary(rollout_dir)
            manifests.append(
                write_rollout_manifest(
                    rollout_dir=rollout_dir,
                    rollout_id=rollout_id,
                    policy=args.policy,
                    sensor_config=sensor_config,
                    world_model_config=world_model_config,
                    frames=int(summary.get("frames_recorded", 0)),
                )
            )
            print(f"[SKIP] {rollout_id} already complete")
            continue

        retries = max(0, int(args.max_retries))
        attempt = 0
        success = False
        while attempt <= retries and not success:
            attempt += 1
            try:
                run_one_rollout(args=args, rollout_dir=rollout_dir, rollout_seed=args.seed + idx * 100 + attempt)
                if not is_rollout_complete(rollout_dir):
                    raise RuntimeError("dataset_complete.json is missing or marks rollout incomplete")
                summary = load_summary(rollout_dir)
                manifests.append(
                    write_rollout_manifest(
                        rollout_dir=rollout_dir,
                        rollout_id=rollout_id,
                        policy=args.policy,
                        sensor_config=sensor_config,
                        world_model_config=world_model_config,
                        frames=int(summary.get("frames_recorded", 0)),
                    )
                )
                success = True
                print(f"[PASS] {rollout_id} (attempt {attempt})")
            except Exception as exc:
                print(f"[WARN] {rollout_id} failed attempt {attempt}: {exc}")
                if attempt > retries:
                    raise

    batch_manifest = {
        "num_rollouts_requested": int(args.num_rollouts),
        "num_rollouts_written": len(manifests),
        "frames_per_rollout_requested": int(args.frames_per_rollout),
        "policy": str(args.policy),
        "rollouts": manifests,
    }
    (out_root / "rollouts_manifest.json").write_text(json.dumps(batch_manifest, indent=2), encoding="utf-8")
    print(f"[PASS] rollout collection complete: {len(manifests)}/{args.num_rollouts}")
    print(f"output_dir: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
