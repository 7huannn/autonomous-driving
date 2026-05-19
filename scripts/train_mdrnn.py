#!/usr/bin/env python3
"""Train Stage 13 MDRNN on latent cache outputs."""

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
from world_model.mdrnn import MDRNN
from world_model.mdrnn_dataset import MDRNNDataset, discover_latent_rollouts
from world_model.mdrnn_trainer import train_mdrnn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage 13 MDRNN")
    parser.add_argument("--config", type=Path, default=Path("configs/world_model.yaml"))
    parser.add_argument("--latent-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None, help="Optional raw rollout dir, stored for traceability")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=0)
    parser.add_argument("--hidden-size", type=int, default=0)
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

    mdrnn_cfg = cfg.get("mdrnn", {}) if isinstance(cfg.get("mdrnn"), dict) else {}
    latent_size = int(mdrnn_cfg.get("latent_size", 64))
    action_size = int(mdrnn_cfg.get("action_size", 3))
    hidden_size = int(args.hidden_size) if args.hidden_size > 0 else int(mdrnn_cfg.get("hidden_size_smoke", 256))
    num_gaussians = int(mdrnn_cfg.get("num_gaussians", 5))
    sequence_length = int(args.sequence_length) if args.sequence_length > 0 else int(mdrnn_cfg.get("sequence_length_smoke", 100))
    batch_size = int(args.batch_size) if args.batch_size > 0 else int(mdrnn_cfg.get("batch_size", 1))
    epochs = int(args.epochs) if args.epochs > 0 else int(mdrnn_cfg.get("epochs_smoke", 3))
    learning_rate = float(mdrnn_cfg.get("learning_rate", 1e-3))

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rollout_dirs = discover_latent_rollouts(args.latent_dir.resolve())
    dataset = MDRNNDataset(rollout_dirs, sequence_length=sequence_length, stride=1)
    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    device = resolve_device(args.device)
    model = MDRNN(
        latent_size=latent_size,
        action_size=action_size,
        hidden_size=hidden_size,
        num_gaussians=num_gaussians,
    )

    metrics = train_mdrnn(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
        output_dir=args.output_dir.resolve(),
    )
    print("[PASS] MDRNN training complete")
    print(f"sequences: {len(dataset)}")
    print(f"best_val_total: {metrics['best_val_total']:.6f}")
    print(f"checkpoint: {metrics['best_checkpoint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
