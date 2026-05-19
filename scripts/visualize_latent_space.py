#!/usr/bin/env python3
"""Visualize latent cache with PCA projection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from world_model import ensure_stage13_scope, load_yaml_config
from world_model.mdrnn_dataset import discover_latent_rollouts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 13 latent space visualization")
    parser.add_argument("--config", type=Path, default=Path("configs/world_model.yaml"))
    parser.add_argument("--latent-dir", type=Path, required=True)
    parser.add_argument("--max-points", type=int, default=5000)
    parser.add_argument("--output", type=Path, default=Path("output/world_model/latent_space_pca.png"))
    parser.add_argument("--report-json", type=Path, default=Path("output/world_model/latent_space_report.json"))
    parser.add_argument("--allow-stage13-control", action="store_true", help="Required for live_planning/iterative mode")
    return parser.parse_args()


def pca_2d(x: np.ndarray) -> np.ndarray:
    x = x - np.mean(x, axis=0, keepdims=True)
    u, s, _vt = np.linalg.svd(x, full_matrices=False)
    return u[:, :2] * s[:2]


def render(points: np.ndarray, colors: np.ndarray, output: Path) -> None:
    h, w = 900, 900
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    x = points[:, 0]
    y = points[:, 1]
    x = (x - x.min()) / max(x.max() - x.min(), 1e-6)
    y = (y - y.min()) / max(y.max() - y.min(), 1e-6)

    for i in range(points.shape[0]):
        px = int(40 + x[i] * (w - 80))
        py = int(40 + y[i] * (h - 80))
        c = tuple(int(v) for v in colors[i])
        cv2.circle(canvas, (px, py), 2, c, -1)

    cv2.putText(canvas, "Stage 13 Latent Space (PCA)", (40, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas)


def main() -> int:
    args = parse_args()
    cfg = load_yaml_config(args.config.resolve())
    ensure_stage13_scope(cfg, allow_stage13_control=bool(args.allow_stage13_control))

    rollouts = discover_latent_rollouts(args.latent_dir.resolve())
    all_points = []
    all_colors = []
    palette = np.array(
        [
            [66, 135, 245],
            [245, 166, 35],
            [80, 200, 120],
            [240, 90, 90],
            [156, 95, 230],
        ],
        dtype=np.uint8,
    )

    for ridx, rollout in enumerate(rollouts):
        z = np.load(rollout / "z.npy").astype(np.float32)
        all_points.append(z)
        all_colors.append(np.repeat(palette[ridx % len(palette)][None, :], z.shape[0], axis=0))

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)

    if points.shape[0] > args.max_points:
        idx = np.random.default_rng(42).choice(points.shape[0], size=args.max_points, replace=False)
        points = points[idx]
        colors = colors[idx]

    proj = pca_2d(points)
    render(proj, colors, args.output.resolve())

    report = {
        "num_rollouts": len(rollouts),
        "num_points": int(points.shape[0]),
        "latent_dim": int(points.shape[1]),
        "output": str(args.output.resolve()),
    }
    args.report_json.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.report_json.resolve().write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("[PASS] latent visualization generated")
    print(f"output: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
