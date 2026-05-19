#!/usr/bin/env python3
"""Decode a short dreamed rollout using VAE + MDRNN."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from world_model import ensure_stage13_scope, load_yaml_config
from world_model.mdrnn import MDRNN
from world_model.vae import ConvVAE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Stage 13 dream rollout")
    parser.add_argument("--config", type=Path, default=Path("configs/world_model.yaml"))
    parser.add_argument("--vae-checkpoint", type=Path, required=True)
    parser.add_argument("--mdrnn-checkpoint", type=Path, required=True)
    parser.add_argument("--latent-dir", type=Path, required=True, help="Latent cache root or single rollout latent dir")
    parser.add_argument("--rollout-id", default="", help="Optional rollout directory name to seed from")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--allow-stage13-control", action="store_true", help="Required for live_planning/iterative mode")
    return parser.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def discover_rollout_latent_dir(latent_root: Path, rollout_id: str) -> Path:
    latent_root = latent_root.resolve()
    if (latent_root / "z.npy").is_file():
        return latent_root
    if rollout_id:
        candidate = latent_root / rollout_id
        if (candidate / "z.npy").is_file():
            return candidate
        raise FileNotFoundError(f"Rollout latent dir not found: {candidate}")
    for p in sorted(latent_root.iterdir()):
        if p.is_dir() and (p / "z.npy").is_file():
            return p
    raise FileNotFoundError(f"No latent rollout dirs in {latent_root}")


def load_vae(ckpt: Path, image_size: int, channels: int, latent_size: int, device: torch.device) -> ConvVAE:
    model = ConvVAE(image_size=image_size, channels=channels, latent_size=latent_size).to(device)
    payload = torch.load(ckpt, map_location=device)
    state = payload.get("model", payload)
    model.load_state_dict(state)
    model.eval()
    return model


def load_mdrnn(
    ckpt: Path,
    latent_size: int,
    action_size: int,
    hidden_size: int,
    num_gaussians: int,
    device: torch.device,
) -> MDRNN:
    model = MDRNN(latent_size=latent_size, action_size=action_size, hidden_size=hidden_size, num_gaussians=num_gaussians).to(device)
    payload = torch.load(ckpt, map_location=device)
    state = payload.get("model", payload)
    model.load_state_dict(state)
    model.eval()
    return model


def render_frame(vae: ConvVAE, latent: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        rec = vae.decode(latent.unsqueeze(0)).squeeze(0).detach().cpu().numpy()
    rgb = np.transpose(rec, (1, 2, 0))
    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def main() -> int:
    args = parse_args()
    cfg = load_yaml_config(args.config.resolve())
    ensure_stage13_scope(cfg, allow_stage13_control=bool(args.allow_stage13_control))

    vae_cfg = cfg.get("vae", {}) if isinstance(cfg.get("vae"), dict) else {}
    mdrnn_cfg = cfg.get("mdrnn", {}) if isinstance(cfg.get("mdrnn"), dict) else {}

    image_size = int(vae_cfg.get("image_size", 64))
    channels = int(vae_cfg.get("channels", 3))
    latent_size = int(vae_cfg.get("latent_size", 64))
    action_size = int(mdrnn_cfg.get("action_size", 3))
    hidden_size = int(mdrnn_cfg.get("hidden_size_smoke", 256))
    num_gaussians = int(mdrnn_cfg.get("num_gaussians", 5))

    device = resolve_device(args.device)
    vae = load_vae(args.vae_checkpoint.resolve(), image_size, channels, latent_size, device)
    mdrnn = load_mdrnn(args.mdrnn_checkpoint.resolve(), latent_size, action_size, hidden_size, num_gaussians, device)

    latent_rollout_dir = discover_rollout_latent_dir(args.latent_dir, args.rollout_id)
    z = np.load(latent_rollout_dir / "z.npy").astype(np.float32)
    actions = np.load(latent_rollout_dir / "actions.npy").astype(np.float32)

    steps = min(int(args.steps), len(actions))
    if steps <= 0:
        raise ValueError("No actions available for dream rollout")

    out_dir = args.output_dir.resolve()
    frames_dir = out_dir / "dream_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "dream_rollout.mp4"

    latent = torch.from_numpy(z[0]).to(device)
    hidden = None

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.fps),
        (image_size, image_size),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {video_path}")

    rewards = []
    dones = []
    try:
        for t in range(steps):
            frame = render_frame(vae, latent)
            frame_path = frames_dir / f"{t:06d}.png"
            cv2.imwrite(str(frame_path), frame)
            writer.write(frame)

            z_t = latent.view(1, 1, -1)
            a_t = torch.from_numpy(actions[t]).to(device).view(1, 1, -1)
            preds, hidden = mdrnn(z_t, a_t, hidden)

            pi = torch.softmax(preds["pi_logits"][0, 0], dim=-1)
            best_idx = int(torch.argmax(pi).item())
            latent = preds["mu"][0, 0, best_idx]
            rewards.append(float(preds["reward"][0, 0].item()))
            done_prob = float(torch.sigmoid(preds["done_logits"][0, 0]).item())
            dones.append(done_prob)
    finally:
        writer.release()

    summary = {
        "seed_rollout": latent_rollout_dir.name,
        "steps": int(steps),
        "avg_pred_reward": float(np.mean(rewards)) if rewards else 0.0,
        "avg_pred_done_prob": float(np.mean(dones)) if dones else 0.0,
        "video": str(video_path),
        "frames_dir": str(frames_dir),
    }
    (out_dir / "dream_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[PASS] dream rollout generated")
    print(f"video: {video_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
