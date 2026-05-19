#!/usr/bin/env python3
"""Offline Stage 13 planner evaluation in latent space."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from world_model import ensure_stage13_scope, load_yaml_config
from world_model.mdrnn import MDRNN
from world_model.mdrnn_dataset import discover_latent_rollouts
from world_model.planners import PlannerSpec, RHEAPlanner, RMHCPlanner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run latent-space planner evaluation")
    parser.add_argument("--config", type=Path, default=Path("configs/world_model.yaml"))
    parser.add_argument("--vae-checkpoint", type=Path, required=True, help="Required for plan compatibility (not used directly)")
    parser.add_argument("--mdrnn-checkpoint", type=Path, required=True)
    parser.add_argument("--latent-dir", type=Path, default=Path("output/world_model/latents_smoke"))
    parser.add_argument("--planner", choices=("rmhc", "rhea"), default="rmhc")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--allow-stage13-control", action="store_true", help="Required for live_planning/iterative mode")
    return parser.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_mdrnn(cfg: dict[str, Any], ckpt: Path, device: torch.device) -> MDRNN:
    mdrnn_cfg = cfg.get("mdrnn", {}) if isinstance(cfg.get("mdrnn"), dict) else {}
    model = MDRNN(
        latent_size=int(mdrnn_cfg.get("latent_size", 64)),
        action_size=int(mdrnn_cfg.get("action_size", 3)),
        hidden_size=int(mdrnn_cfg.get("hidden_size_smoke", 256)),
        num_gaussians=int(mdrnn_cfg.get("num_gaussians", 5)),
    ).to(device)
    payload = torch.load(ckpt, map_location=device)
    state = payload.get("model", payload)
    model.load_state_dict(state)
    model.eval()
    return model


def load_start_latents(latent_dir: Path, episodes: int) -> list[np.ndarray]:
    starts: list[np.ndarray] = []
    rollout_dirs = discover_latent_rollouts(latent_dir)
    for rollout in rollout_dirs:
        z = np.load(rollout / "z.npy").astype(np.float32)
        for idx in range(min(len(z), 10)):
            starts.append(z[idx])
            if len(starts) >= episodes:
                return starts
    if not starts:
        raise RuntimeError(f"No start latents found in {latent_dir}")
    return starts[:episodes]


def latent_rollout_score(
    model: MDRNN,
    start_z: np.ndarray,
    sequence: np.ndarray,
    device: torch.device,
) -> float:
    with torch.no_grad():
        z = torch.from_numpy(start_z).to(device)
        hidden = None
        total_reward = 0.0
        for action in sequence:
            z_t = z.view(1, 1, -1)
            a_t = torch.from_numpy(action.astype(np.float32)).to(device).view(1, 1, -1)
            preds, hidden = model(z_t, a_t, hidden)

            pi = torch.softmax(preds["pi_logits"][0, 0], dim=-1)
            k = int(torch.argmax(pi).item())
            z = preds["mu"][0, 0, k]
            reward = float(preds["reward"][0, 0].item())
            total_reward += reward
            done_prob = float(torch.sigmoid(preds["done_logits"][0, 0]).item())
            if done_prob > 0.5:
                break
    return float(total_reward)


def render_trace(trace: list[float], output_path: Path) -> None:
    h, w = 480, 860
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    cv2.rectangle(canvas, (60, 40), (w - 30, h - 60), (20, 20, 20), 2)
    if len(trace) >= 2:
        y_min = float(min(trace))
        y_max = float(max(trace))
        if abs(y_max - y_min) < 1e-6:
            y_max = y_min + 1.0
        points = []
        for i, val in enumerate(trace):
            x = int(60 + i * (w - 100) / max(len(trace) - 1, 1))
            y = int((h - 60) - (val - y_min) * (h - 120) / (y_max - y_min))
            points.append((x, y))
        for p0, p1 in zip(points[:-1], points[1:]):
            cv2.line(canvas, p0, p1, (0, 100, 220), 2)
    cv2.putText(canvas, "Stage 13 Planner Predicted Reward Trace", (70, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.imwrite(str(output_path), canvas)


def main() -> int:
    args = parse_args()
    cfg = load_yaml_config(args.config.resolve())
    ensure_stage13_scope(cfg, allow_stage13_control=bool(args.allow_stage13_control))

    if not args.vae_checkpoint.resolve().is_file():
        raise FileNotFoundError(f"VAE checkpoint not found: {args.vae_checkpoint}")

    planner_cfg = cfg.get("planner", {}) if isinstance(cfg.get("planner"), dict) else {}
    action_order = tuple(planner_cfg.get("action_order", ["steer", "throttle", "brake"]))
    mut = planner_cfg.get("mutation_std", {})
    bounds = planner_cfg.get("action_bounds", {})

    spec = PlannerSpec(
        action_order=action_order,
        horizon=int(planner_cfg.get("horizon", 20)),
        generations=int(planner_cfg.get("generations", 10)),
        mutation_std=np.array([
            float(mut.get("steer", 0.15)),
            float(mut.get("throttle", 0.10)),
            float(mut.get("brake", 0.10)),
        ], dtype=np.float32),
        action_low=np.array([
            float(bounds.get("steer", [-1.0, 1.0])[0]),
            float(bounds.get("throttle", [0.0, 1.0])[0]),
            float(bounds.get("brake", [0.0, 1.0])[0]),
        ], dtype=np.float32),
        action_high=np.array([
            float(bounds.get("steer", [-1.0, 1.0])[1]),
            float(bounds.get("throttle", [0.0, 1.0])[1]),
            float(bounds.get("brake", [0.0, 1.0])[1]),
        ], dtype=np.float32),
    )

    rng = np.random.default_rng(args.seed)
    planner = RMHCPlanner(spec, rng=rng) if args.planner == "rmhc" else RHEAPlanner(spec, rng=rng)

    device = resolve_device(args.device)
    model = load_mdrnn(cfg, args.mdrnn_checkpoint.resolve(), device)
    starts = load_start_latents(args.latent_dir.resolve(), args.episodes)

    planner_scores: list[float] = []
    random_scores: list[float] = []
    rewards_trace: list[float] = []

    for i, start_z in enumerate(starts):
        eval_fn = lambda seq, z=start_z: latent_rollout_score(model, z, seq, device)
        result = planner.plan(evaluate_fn=eval_fn)
        planner_scores.append(float(result.score))

        random_trials = [float(eval_fn(planner.sample_sequence())) for _ in range(max(10, spec.generations))]
        random_scores.append(float(np.mean(random_trials)))

        if i == 0:
            running = []
            acc = 0.0
            for action in result.action_sequence:
                acc += latent_rollout_score(model, start_z, np.expand_dims(action, axis=0), device)
                running.append(acc)
            rewards_trace = running

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "planner": args.planner,
        "episodes": len(starts),
        "horizon": spec.horizon,
        "generations": spec.generations,
        "mean_planner_pred_reward": float(np.mean(planner_scores)),
        "mean_random_pred_reward": float(np.mean(random_scores)),
        "planner_beats_random": bool(np.mean(planner_scores) > np.mean(random_scores)),
        "planner_scores": planner_scores,
        "random_scores": random_scores,
    }
    (out_dir / "planner_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    render_trace(rewards_trace if rewards_trace else [0.0], out_dir / "rollout_visualization.png")

    sweep = {
        "generation_sweep": planner_cfg.get("generation_sweep", [5, 10, 15]),
        "horizon_sweep": planner_cfg.get("horizon_sweep", [5, 10, 20]),
    }
    (out_dir / "planner_sweeps.json").write_text(json.dumps(sweep, indent=2), encoding="utf-8")

    print("[PASS] planner evaluation complete")
    print(f"planner_mean: {metrics['mean_planner_pred_reward']:.6f}")
    print(f"random_mean: {metrics['mean_random_pred_reward']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
