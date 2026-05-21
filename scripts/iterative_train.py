#!/usr/bin/env python3
"""Stage 13B iterative collect/train/evaluate loop orchestrator."""

from __future__ import annotations

import argparse
import json
import shutil
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
    parser = argparse.ArgumentParser(description="Run Stage 13B iterative collect/train/evaluate")
    parser.add_argument("--world-model-config", type=Path, default=Path("configs/world_model.yaml"))
    parser.add_argument("--sensor-config", type=Path, default=Path("configs/sensor_config.yaml"))
    parser.add_argument("--initial-vae-checkpoint", type=Path, required=True)
    parser.add_argument("--initial-mdrnn-checkpoint", type=Path, required=True)
    parser.add_argument("--planner", choices=("rmhc", "rhea"), default="rmhc")
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--rollouts-per-cycle", type=int, default=1)
    parser.add_argument("--frames-per-rollout", type=int, default=100)
    parser.add_argument("--baseline-policy", choices=("off", "autopilot_noise", "autopilot", "random"), default="autopilot_noise")
    parser.add_argument("--baseline-rollouts", type=int, default=1)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/stage13_iterative"))
    parser.add_argument("--output-root", type=Path, default=Path("output/world_model/iterative"))
    parser.add_argument(
        "--python",
        type=str,
        default=None,
        help="Compatibility alias used for both CARLA and training commands",
    )
    parser.add_argument(
        "--carla-python",
        type=str,
        default=None,
        help="Python executable for live CARLA rollout collection",
    )
    parser.add_argument(
        "--train-python",
        type=str,
        default=None,
        help="Python executable for VAE/MDRNN/offline planner commands",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-stage13-control", action="store_true", help="Required for iterative mode")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_cmd(cmd: list[str], dry_run: bool) -> tuple[int, str, str]:
    if dry_run:
        print("[DRY-RUN]", " ".join(cmd))
        return 0, "", ""
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, capture_output=True, check=False)
    return proc.returncode, proc.stdout, proc.stderr


def ensure_success(cmd: list[str], dry_run: bool) -> None:
    code, stdout, stderr = run_cmd(cmd, dry_run=dry_run)
    if code != 0:
        raise RuntimeError(
            "Command failed with exit code {}\nCMD: {}\nSTDOUT:\n{}\nSTDERR:\n{}".format(
                code,
                " ".join(cmd),
                stdout.strip(),
                stderr.strip(),
            )
        )


def find_next_rollout_id(data_root: Path) -> str:
    existing = [p.name for p in data_root.glob("rollout_*") if p.is_dir()]
    if not existing:
        return "rollout_00000"
    indices = []
    for name in existing:
        try:
            indices.append(int(name.split("_", 1)[1]))
        except Exception:
            continue
    nxt = (max(indices) + 1) if indices else 0
    return f"rollout_{nxt:05d}"


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid JSON object: {path}")
    return data


def check_rollout_complete(rollout_dir: Path, dry_run: bool) -> None:
    if dry_run:
        return
    marker = rollout_dir / "dataset_complete.json"
    if not marker.is_file():
        raise RuntimeError(f"Missing dataset_complete.json: {marker}")
    payload = load_json(marker)
    if not bool(payload.get("complete", False)):
        raise RuntimeError(f"Rollout incomplete: {marker}")


def stage13_flags(allow_stage13_control: bool) -> list[str]:
    return ["--allow-stage13-control"] if allow_stage13_control else []


def resolve_python_executables(args: argparse.Namespace) -> tuple[str, str]:
    default_python = args.python or sys.executable
    carla_python = args.carla_python or default_python
    train_python = args.train_python or default_python
    return carla_python, train_python


def assert_within_path(path: Path, parent: Path, *, label: str) -> None:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError as exc:
        raise ValueError(f"Refusing overwrite cleanup for {label}: {path.resolve()} is outside {parent.resolve()}") from exc


def assert_within_repo(path: Path, *, label: str) -> None:
    assert_within_path(path, REPO_ROOT, label=label)


def cleanup_overwrite_targets(data_root: Path, output_root: Path) -> None:
    assert_within_repo(data_root, label="--data-root")
    assert_within_repo(output_root, label="--output-root")

    for child in sorted(data_root.glob("rollout_*")):
        if not child.is_dir():
            continue
        assert_within_path(child, data_root, label="data-root child")
        shutil.rmtree(child)

    for child in sorted(output_root.glob("cycle_*")):
        if not child.is_dir():
            continue
        assert_within_path(child, output_root, label="output-root child")
        shutil.rmtree(child)

    summary_path = output_root / "iterative_summary.json"
    if summary_path.is_file():
        assert_within_path(summary_path, output_root, label="iterative_summary.json")
        summary_path.unlink()


def collect_rollouts(
    args: argparse.Namespace,
    data_root: Path,
    cycle_index: int,
    policy: str,
    num_rollouts: int,
    vae_ckpt: Path,
    mdrnn_ckpt: Path,
    dry_run: bool,
    python_executable: str,
) -> list[Path]:
    rollouts: list[Path] = []
    for rollout_local_idx in range(num_rollouts):
        rollout_id = find_next_rollout_id(data_root)
        rollout_dir = data_root / rollout_id
        seed = args.seed + cycle_index * 1000 + rollout_local_idx
        cmd = [
            python_executable,
            "scripts/run_planning_agent.py",
            "--config",
            str(args.sensor_config),
            "--world-model-config",
            str(args.world_model_config),
            "--vae-checkpoint",
            str(vae_ckpt),
            "--mdrnn-checkpoint",
            str(mdrnn_ckpt),
            "--planner",
            str(args.planner),
            "--policy",
            str(policy),
            "--output-dir",
            str(rollout_dir),
            "--max-frames",
            str(args.frames_per_rollout),
            "--seed",
            str(seed),
            "--device",
            str(args.device),
            *stage13_flags(args.allow_stage13_control),
        ]
        if args.overwrite:
            cmd.append("--overwrite")
        ensure_success(cmd, dry_run=dry_run)
        if dry_run:
            rollout_dir.mkdir(parents=True, exist_ok=True)
        check_rollout_complete(rollout_dir, dry_run=dry_run)
        rollouts.append(rollout_dir)
    return rollouts


def run_training_cycle(
    args: argparse.Namespace,
    cycle_index: int,
    data_root: Path,
    output_root: Path,
    vae_ckpt_in: Path,
    mdrnn_ckpt_in: Path,
) -> dict[str, Any]:
    cycle_dir = output_root / f"cycle_{cycle_index:03d}"
    cycle_dir.mkdir(parents=True, exist_ok=True)
    carla_python, train_python = resolve_python_executables(args)

    planner_rollouts = collect_rollouts(
        args=args,
        data_root=data_root,
        cycle_index=cycle_index,
        policy="planner",
        num_rollouts=args.rollouts_per_cycle,
        vae_ckpt=vae_ckpt_in,
        mdrnn_ckpt=mdrnn_ckpt_in,
        dry_run=bool(args.dry_run),
        python_executable=carla_python,
    )

    baseline_rollouts: list[Path] = []
    if args.baseline_policy != "off" and args.baseline_rollouts > 0:
        baseline_rollouts = collect_rollouts(
            args=args,
            data_root=data_root,
            cycle_index=cycle_index + 100,
            policy=str(args.baseline_policy),
            num_rollouts=args.baseline_rollouts,
            vae_ckpt=vae_ckpt_in,
            mdrnn_ckpt=mdrnn_ckpt_in,
            dry_run=bool(args.dry_run),
            python_executable=carla_python,
        )

    vae_out = cycle_dir / "vae"
    latents_out = cycle_dir / "latents"
    mdrnn_out = cycle_dir / "mdrnn"
    plan_eval_out = cycle_dir / "planning_eval"
    wm_cfg = load_yaml_config(args.world_model_config.resolve())
    mdrnn_cfg = wm_cfg.get("mdrnn", {}) if isinstance(wm_cfg.get("mdrnn"), dict) else {}
    requested_sequence_length = int(mdrnn_cfg.get("sequence_length_smoke", 100))
    max_sequence_length = max(1, int(args.frames_per_rollout) - 1)
    mdrnn_sequence_length = max(1, min(requested_sequence_length, max_sequence_length))

    vae_cmd = [
        train_python,
        "scripts/train_vae.py",
        "--config",
        str(args.world_model_config),
        "--data-dir",
        str(data_root),
        "--output-dir",
        str(vae_out),
        "--device",
        str(args.device),
        *stage13_flags(args.allow_stage13_control),
    ]
    if args.dry_run:
        vae_cmd.extend(["--epochs", "1"])
    ensure_success(vae_cmd, dry_run=bool(args.dry_run))

    vae_ckpt_out = vae_out / "best.pth"
    encode_cmd = [
        train_python,
        "scripts/encode_rollouts.py",
        "--config",
        str(args.world_model_config),
        "--vae-checkpoint",
        str(vae_ckpt_out),
        "--data-dir",
        str(data_root),
        "--output-dir",
        str(latents_out),
        "--write-reconstructions",
        "--device",
        str(args.device),
        *stage13_flags(args.allow_stage13_control),
    ]
    ensure_success(encode_cmd, dry_run=bool(args.dry_run))

    mdrnn_cmd = [
        train_python,
        "scripts/train_mdrnn.py",
        "--config",
        str(args.world_model_config),
        "--latent-dir",
        str(latents_out),
        "--data-dir",
        str(data_root),
        "--output-dir",
        str(mdrnn_out),
        "--device",
        str(args.device),
        "--sequence-length",
        str(mdrnn_sequence_length),
        *stage13_flags(args.allow_stage13_control),
    ]
    if args.dry_run:
        mdrnn_cmd.extend(["--epochs", "1", "--sequence-length", "10", "--hidden-size", "64"])
    ensure_success(mdrnn_cmd, dry_run=bool(args.dry_run))

    mdrnn_ckpt_out = mdrnn_out / "best.pth"
    eval_cmd = [
        train_python,
        "scripts/run_planner.py",
        "--config",
        str(args.world_model_config),
        "--vae-checkpoint",
        str(vae_ckpt_out),
        "--mdrnn-checkpoint",
        str(mdrnn_ckpt_out),
        "--latent-dir",
        str(latents_out),
        "--planner",
        str(args.planner),
        "--episodes",
        "20",
        "--output-dir",
        str(plan_eval_out),
        "--device",
        str(args.device),
        *stage13_flags(args.allow_stage13_control),
    ]
    ensure_success(eval_cmd, dry_run=bool(args.dry_run))

    cycle_summary = {
        "cycle": cycle_index,
        "carla_python": str(carla_python),
        "train_python": str(train_python),
        "planner_rollouts": [str(p) for p in planner_rollouts],
        "baseline_rollouts": [str(p) for p in baseline_rollouts],
        "vae_checkpoint": str(vae_ckpt_out),
        "mdrnn_checkpoint": str(mdrnn_ckpt_out),
        "latents_dir": str(latents_out),
        "planning_eval_dir": str(plan_eval_out),
    }
    (cycle_dir / "cycle_summary.json").write_text(json.dumps(cycle_summary, indent=2), encoding="utf-8")
    return cycle_summary


def main() -> int:
    args = parse_args()
    if args.cycles <= 0:
        raise ValueError("--cycles must be > 0")
    if args.rollouts_per_cycle <= 0:
        raise ValueError("--rollouts-per-cycle must be > 0")
    if args.frames_per_rollout <= 0:
        raise ValueError("--frames-per-rollout must be > 0")

    wm_cfg = load_yaml_config(args.world_model_config.resolve())
    mode = ensure_stage13_scope(wm_cfg, allow_stage13_control=bool(args.allow_stage13_control))
    if mode != "iterative":
        raise ValueError(f"iterative_train.py requires stage13.mode=iterative, got '{mode}'")

    if not args.initial_vae_checkpoint.resolve().is_file():
        raise FileNotFoundError(f"Initial VAE checkpoint not found: {args.initial_vae_checkpoint}")
    if not args.initial_mdrnn_checkpoint.resolve().is_file():
        raise FileNotFoundError(f"Initial MDRNN checkpoint not found: {args.initial_mdrnn_checkpoint}")

    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    if args.overwrite:
        cleanup_overwrite_targets(data_root=data_root, output_root=output_root)
    data_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    vae_ckpt = args.initial_vae_checkpoint.resolve()
    mdrnn_ckpt = args.initial_mdrnn_checkpoint.resolve()

    all_cycles = []
    for cycle_idx in range(args.cycles):
        summary = run_training_cycle(
            args=args,
            cycle_index=cycle_idx,
            data_root=data_root,
            output_root=output_root,
            vae_ckpt_in=vae_ckpt,
            mdrnn_ckpt_in=mdrnn_ckpt,
        )
        vae_ckpt = Path(summary["vae_checkpoint"])
        mdrnn_ckpt = Path(summary["mdrnn_checkpoint"])
        all_cycles.append(summary)

    report = {
        "mode": mode,
        "dry_run": bool(args.dry_run),
        "carla_python": str(resolve_python_executables(args)[0]),
        "train_python": str(resolve_python_executables(args)[1]),
        "cycles": int(args.cycles),
        "rollouts_per_cycle": int(args.rollouts_per_cycle),
        "frames_per_rollout": int(args.frames_per_rollout),
        "planner": args.planner,
        "baseline_policy": args.baseline_policy,
        "baseline_rollouts": int(args.baseline_rollouts),
        "data_root": str(data_root),
        "output_root": str(output_root),
        "last_vae_checkpoint": str(vae_ckpt),
        "last_mdrnn_checkpoint": str(mdrnn_ckpt),
        "cycle_summaries": all_cycles,
    }
    (output_root / "iterative_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("[PASS] iterative_train complete")
    print(f"summary: {output_root / 'iterative_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
