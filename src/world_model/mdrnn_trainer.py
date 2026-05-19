"""Trainer for MDRNN world-model dynamics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .mdrnn import MDRNN
from .mdrnn_loss import mdrnn_loss


def train_mdrnn(
    model: MDRNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = output_dir / "best.pth"
    history: list[dict[str, float]] = []

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        train_totals = {"total": 0.0, "gmm_nll": 0.0, "reward_mse": 0.0, "done_bce": 0.0}
        train_count = 0

        for batch in train_loader:
            z_t = batch["z_t"].to(device)
            a_t = batch["a_t"].to(device)
            z_next = batch["z_next"].to(device)
            reward = batch["reward"].to(device)
            done = batch["done"].to(device)

            optimizer.zero_grad(set_to_none=True)
            preds, _hidden = model(z_t, a_t)
            loss, metrics = mdrnn_loss(preds, z_next, reward, done)
            loss.backward()
            optimizer.step()

            bsz = z_t.size(0)
            train_count += bsz
            for key in train_totals:
                train_totals[key] += metrics[key] * bsz

        model.eval()
        val_totals = {"total": 0.0, "gmm_nll": 0.0, "reward_mse": 0.0, "done_bce": 0.0}
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                z_t = batch["z_t"].to(device)
                a_t = batch["a_t"].to(device)
                z_next = batch["z_next"].to(device)
                reward = batch["reward"].to(device)
                done = batch["done"].to(device)

                preds, _hidden = model(z_t, a_t)
                _loss, metrics = mdrnn_loss(preds, z_next, reward, done)
                bsz = z_t.size(0)
                val_count += bsz
                for key in val_totals:
                    val_totals[key] += metrics[key] * bsz

        epoch_metrics = {
            "epoch": float(epoch),
            "train_total": train_totals["total"] / max(train_count, 1),
            "train_gmm_nll": train_totals["gmm_nll"] / max(train_count, 1),
            "train_reward_mse": train_totals["reward_mse"] / max(train_count, 1),
            "train_done_bce": train_totals["done_bce"] / max(train_count, 1),
            "val_total": val_totals["total"] / max(val_count, 1),
            "val_gmm_nll": val_totals["gmm_nll"] / max(val_count, 1),
            "val_reward_mse": val_totals["reward_mse"] / max(val_count, 1),
            "val_done_bce": val_totals["done_bce"] / max(val_count, 1),
        }
        history.append(epoch_metrics)

        if epoch_metrics["val_total"] < best_val:
            best_val = epoch_metrics["val_total"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_total": best_val}, best_ckpt)

        print(
            f"epoch={epoch:03d} train={epoch_metrics['train_total']:.6f} "
            f"val={epoch_metrics['val_total']:.6f} gmm={epoch_metrics['val_gmm_nll']:.6f} "
            f"reward={epoch_metrics['val_reward_mse']:.6f} done={epoch_metrics['val_done_bce']:.6f}"
        )

    out = {
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "best_val_total": float(best_val),
        "history": history,
        "best_checkpoint": str(best_ckpt),
    }
    (output_dir / "training_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
