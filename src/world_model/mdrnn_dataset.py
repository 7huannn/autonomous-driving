"""Sequence dataset backed by latent cache outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SequenceIndex:
    rollout_dir: Path
    start: int


def discover_latent_rollouts(latent_dir: Path) -> list[Path]:
    latent_dir = latent_dir.resolve()
    if (latent_dir / "z.npy").is_file():
        return [latent_dir]
    rollouts = [p for p in sorted(latent_dir.iterdir()) if p.is_dir() and (p / "z.npy").is_file()]
    if not rollouts:
        raise FileNotFoundError(f"No latent rollouts found under {latent_dir}")
    return rollouts


class MDRNNDataset(Dataset[dict[str, Tensor]]):
    def __init__(self, latent_rollout_dirs: list[Path], sequence_length: int = 100, stride: int = 1) -> None:
        self.sequence_length = int(sequence_length)
        self.stride = int(stride)
        self.entries: list[SequenceIndex] = []
        self.buffers: dict[Path, dict[str, np.ndarray]] = {}

        for rollout_dir in latent_rollout_dirs:
            z = np.load(rollout_dir / "z.npy")
            actions = np.load(rollout_dir / "actions.npy")
            rewards = np.load(rollout_dir / "rewards.npy")
            dones = np.load(rollout_dir / "dones.npy")
            n = min(len(z), len(actions), len(rewards), len(dones))
            if n < self.sequence_length + 1:
                continue
            self.buffers[rollout_dir] = {
                "z": z[:n].astype(np.float32),
                "actions": actions[:n].astype(np.float32),
                "rewards": rewards[:n].astype(np.float32),
                "dones": dones[:n].astype(np.float32),
            }
            for start in range(0, n - self.sequence_length - 1 + 1, self.stride):
                self.entries.append(SequenceIndex(rollout_dir=rollout_dir, start=start))

        if not self.entries:
            raise ValueError("No sequences available for MDRNN dataset")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        entry = self.entries[idx]
        buf = self.buffers[entry.rollout_dir]
        s = entry.start
        e = s + self.sequence_length

        z_t = buf["z"][s:e]
        a_t = buf["actions"][s:e]
        z_next = buf["z"][s + 1 : e + 1]
        reward = buf["rewards"][s:e]
        done = buf["dones"][s:e]

        return {
            "z_t": torch.from_numpy(z_t),
            "a_t": torch.from_numpy(a_t),
            "z_next": torch.from_numpy(z_next),
            "reward": torch.from_numpy(reward),
            "done": torch.from_numpy(done),
        }
