#!/usr/bin/env python3
"""Train ConvVAE for Stage 13 world model."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from world_model import ensure_stage13_scope, load_yaml_config
from world_model.vae import ConvVAE
from world_model.vae_dataset import VAEFrameDataset, discover_rgb_frames
from world_model.vae_trainer import train_vae


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage 13 ConvVAE")
    parser.add_argument("--config", type=Path, default=Path("configs/world_model.yaml"), help="World model config")
    parser.add_argument("--data-dir", type=Path, required=True, help="Rollout directory or root containing rollouts")
    parser.add_argument("--output-dir", type=Path, required=True, help="VAE output dir")
    parser.add_argument("--epochs", type=int, default=0, help="Override epochs (0 uses config epochs_smoke)")
    parser.add_argument("--batch-size", type=int, default=0, help="Override batch size")
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


def main() -> int:
    args = parse_args()
    cfg = load_yaml_config(args.config.resolve())
    ensure_stage13_scope(cfg, allow_stage13_control=bool(args.allow_stage13_control))

    vae_cfg = cfg.get("vae", {}) if isinstance(cfg.get("vae"), dict) else {}
    image_size = int(vae_cfg.get("image_size", 64))
    channels = int(vae_cfg.get("channels", 3))
    latent_size = int(vae_cfg.get("latent_size", 64))
    batch_size = int(args.batch_size) if args.batch_size > 0 else int(vae_cfg.get("batch_size", 32))
    epochs = int(args.epochs) if args.epochs > 0 else int(vae_cfg.get("epochs_smoke", 3))
    lr = float(vae_cfg.get("learning_rate", 1e-4))
    kl_weight = float(vae_cfg.get("kl_weight", 1.0))

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    frames = discover_rgb_frames(args.data_dir.resolve())
    dataset = VAEFrameDataset(frames, image_size=image_size)
    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    device = resolve_device(args.device)
    model = ConvVAE(image_size=image_size, channels=channels, latent_size=latent_size)

    metrics = train_vae(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        learning_rate=lr,
        kl_weight=kl_weight,
        device=device,
        output_dir=args.output_dir.resolve(),
    )
    print("[PASS] VAE training complete")
    print(f"frames: {len(dataset)}")
    print(f"best_val_total: {metrics['best_val_total']:.6f}")
    print(f"checkpoint: {metrics['best_checkpoint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
