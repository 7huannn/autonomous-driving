"""Training loop utilities for ConvVAE."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from .vae import ConvVAE, vae_loss


@torch.no_grad()
def save_reconstruction_grid(model: ConvVAE, batch: torch.Tensor, output_path: Path) -> None:
    model.eval()
    recon, _, _ = model(batch)
    src = batch.detach().cpu().numpy()
    rec = recon.detach().cpu().numpy()

    n = min(8, src.shape[0])
    rows = []
    for i in range(n):
        top = np.transpose(src[i], (1, 2, 0))
        bottom = np.transpose(rec[i], (1, 2, 0))
        rows.append(np.concatenate([top, bottom], axis=0))
    grid = np.concatenate(rows, axis=1)
    grid = np.clip(grid * 255.0, 0, 255).astype(np.uint8)
    grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), grid_bgr)


def train_vae(
    model: ConvVAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    kl_weight: float,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = output_dir / "best.pth"
    history: list[dict[str, float]] = []

    best_val = float("inf")
    recon_batch = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_total = 0.0
        train_recon = 0.0
        train_kl = 0.0
        train_count = 0

        for x in train_loader:
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon, mu, logvar = model(x)
            loss, metrics = vae_loss(recon, x, mu, logvar, kl_weight=kl_weight)
            loss.backward()
            optimizer.step()

            bsz = x.size(0)
            train_count += bsz
            train_total += metrics["total"] * bsz
            train_recon += metrics["reconstruction_mse"] * bsz
            train_kl += metrics["kl"] * bsz
            if recon_batch is None:
                recon_batch = x[: min(8, bsz)].detach()

        model.eval()
        val_total = 0.0
        val_recon = 0.0
        val_kl = 0.0
        val_count = 0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                recon, mu, logvar = model(x)
                _loss, metrics = vae_loss(recon, x, mu, logvar, kl_weight=kl_weight)
                bsz = x.size(0)
                val_count += bsz
                val_total += metrics["total"] * bsz
                val_recon += metrics["reconstruction_mse"] * bsz
                val_kl += metrics["kl"] * bsz

        epoch_metrics = {
            "epoch": float(epoch),
            "train_total": train_total / max(train_count, 1),
            "train_reconstruction_mse": train_recon / max(train_count, 1),
            "train_kl": train_kl / max(train_count, 1),
            "val_total": val_total / max(val_count, 1),
            "val_reconstruction_mse": val_recon / max(val_count, 1),
            "val_kl": val_kl / max(val_count, 1),
        }
        history.append(epoch_metrics)

        if epoch_metrics["val_total"] < best_val:
            best_val = epoch_metrics["val_total"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_total": best_val}, best_ckpt)

        print(
            f"epoch={epoch:03d} train={epoch_metrics['train_total']:.6f} "
            f"val={epoch_metrics['val_total']:.6f} recon={epoch_metrics['val_reconstruction_mse']:.6f} kl={epoch_metrics['val_kl']:.6f}"
        )

    if recon_batch is not None:
        save_reconstruction_grid(model, recon_batch.to(device), output_dir / "recon_grid.png")

    out = {
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "kl_weight": float(kl_weight),
        "best_val_total": float(best_val),
        "history": history,
        "best_checkpoint": str(best_ckpt),
    }
    (output_dir / "training_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
