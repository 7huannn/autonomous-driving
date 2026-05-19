#!/usr/bin/env python3
"""Encode rollout RGB frames into latent vectors using trained ConvVAE."""

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
from world_model.vae import ConvVAE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode Stage 13 rollouts into latent cache")
    parser.add_argument("--config", type=Path, default=Path("configs/world_model.yaml"))
    parser.add_argument("--vae-checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True, help="Single rollout dir or root containing rollout_* dirs")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--write-reconstructions", action="store_true", help="Write per-frame VAE reconstructions")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--allow-stage13-control", action="store_true", help="Required for live_planning/iterative mode")
    return parser.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def discover_rollout_dirs(root: Path) -> list[Path]:
    root = root.resolve()
    if (root / "rgb").is_dir() and (root / "metadata").is_dir():
        return [root]
    out = [p for p in sorted(root.iterdir()) if p.is_dir() and (p / "rgb").is_dir() and (p / "metadata").is_dir()]
    if not out:
        raise FileNotFoundError(f"No rollout directories found under {root}")
    return out


def load_model(checkpoint: Path, image_size: int, channels: int, latent_size: int, device: torch.device) -> ConvVAE:
    model = ConvVAE(image_size=image_size, channels=channels, latent_size=latent_size).to(device)
    payload = torch.load(checkpoint, map_location=device)
    state = payload.get("model", payload)
    model.load_state_dict(state)
    model.eval()
    return model


def load_frame_tensor(path: Path, image_size: int) -> torch.Tensor:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    resized = cv2.resize(bgr, (image_size, image_size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    arr = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)
    return torch.from_numpy(arr)


def read_frame_metadata(metadata_path: Path) -> tuple[np.ndarray, float, float]:
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    control = meta.get("control", {}) if isinstance(meta.get("control"), dict) else {}
    world_model = meta.get("world_model", {}) if isinstance(meta.get("world_model"), dict) else {}
    action = np.array(
        [
            float(control.get("steer", 0.0)),
            float(control.get("throttle", 0.0)),
            float(control.get("brake", 0.0)),
        ],
        dtype=np.float32,
    )
    reward = float(world_model.get("reward", 0.0))
    done = float(bool(world_model.get("done", False)))
    return action, reward, done


def encode_rollout(
    model: ConvVAE,
    rollout_dir: Path,
    output_dir: Path,
    image_size: int,
    device: torch.device,
    write_reconstructions: bool,
) -> dict[str, Any]:
    rgb_dir = rollout_dir / "rgb"
    metadata_dir = rollout_dir / "metadata"
    stems = sorted(p.stem for p in rgb_dir.glob("*.png"))
    if not stems:
        raise RuntimeError(f"No frames in {rgb_dir}")

    latents = []
    actions = []
    rewards = []
    dones = []
    out_dir = output_dir / rollout_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    recon_dir = out_dir / "reconstructions" if write_reconstructions else None
    if recon_dir is not None:
        recon_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for stem in stems:
            x = load_frame_tensor(rgb_dir / f"{stem}.png", image_size=image_size).unsqueeze(0).to(device)
            recon, mu, _logvar = model(x)
            z = mu.squeeze(0).cpu().numpy().astype(np.float32)
            action, reward, done = read_frame_metadata(metadata_dir / f"{stem}.json")

            latents.append(z)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            if recon_dir is not None:
                rec = recon.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
                rec = np.clip(rec * 255.0, 0, 255).astype(np.uint8)
                rec_bgr = cv2.cvtColor(rec, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(recon_dir / f"{stem}.png"), rec_bgr)
    np.save(out_dir / "z.npy", np.stack(latents, axis=0))
    np.save(out_dir / "actions.npy", np.stack(actions, axis=0))
    np.save(out_dir / "rewards.npy", np.array(rewards, dtype=np.float32))
    np.save(out_dir / "dones.npy", np.array(dones, dtype=np.float32))
    (out_dir / "frame_stems.json").write_text(json.dumps(stems, indent=2), encoding="utf-8")

    return {
        "rollout_id": rollout_dir.name,
        "frames": len(stems),
        "output_dir": str(out_dir),
        "reconstructions_dir": str(recon_dir) if recon_dir is not None else None,
    }


def main() -> int:
    args = parse_args()
    cfg = load_yaml_config(args.config.resolve())
    ensure_stage13_scope(cfg, allow_stage13_control=bool(args.allow_stage13_control))
    vae_cfg = cfg.get("vae", {}) if isinstance(cfg.get("vae"), dict) else {}

    image_size = int(vae_cfg.get("image_size", 64))
    channels = int(vae_cfg.get("channels", 3))
    latent_size = int(vae_cfg.get("latent_size", 64))

    device = resolve_device(args.device)
    model = load_model(args.vae_checkpoint.resolve(), image_size, channels, latent_size, device)

    rollout_dirs = discover_rollout_dirs(args.data_dir.resolve())
    out_root = args.output_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = []
    for rollout_dir in rollout_dirs:
        entry = encode_rollout(
            model,
            rollout_dir,
            out_root,
            image_size=image_size,
            device=device,
            write_reconstructions=bool(args.write_reconstructions),
        )
        manifest.append(entry)
        print(f"[PASS] encoded {entry['rollout_id']} ({entry['frames']} frames)")

    report = {
        "num_rollouts": len(manifest),
        "total_frames": int(sum(int(m["frames"]) for m in manifest)),
        "latent_size": latent_size,
        "rollouts": manifest,
    }
    (out_root / "latent_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[PASS] latent cache written: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
