"""Dataset helpers for Stage 13 VAE training."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


def discover_rgb_frames(data_dir: Path) -> list[Path]:
    data_dir = data_dir.resolve()
    if (data_dir / "rgb").is_dir():
        frames = sorted((data_dir / "rgb").glob("*.png"))
        if frames:
            return frames

    frames = sorted(data_dir.glob("**/rgb/*.png"))
    if not frames:
        raise FileNotFoundError(f"No RGB frames found under {data_dir}")
    return frames


class VAEFrameDataset(Dataset[Tensor]):
    def __init__(self, frame_paths: list[Path], image_size: int = 64) -> None:
        if not frame_paths:
            raise ValueError("frame_paths must be non-empty")
        self.frame_paths = frame_paths
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.frame_paths)

    def __getitem__(self, idx: int) -> Tensor:
        path = self.frame_paths[idx]
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read image: {path}")
        resized = cv2.resize(bgr, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        arr = rgb.astype(np.float32) / 255.0
        chw = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(chw)
