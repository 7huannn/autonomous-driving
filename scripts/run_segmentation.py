#!/usr/bin/env python3
"""Stage 07 segmentation integration runner for CARLA PAD data."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


CITYSCAPES_COLORS = np.array(
    [
        [128, 64, 128],   # road
        [244, 35, 232],   # sidewalk
        [70, 70, 70],     # building
        [102, 102, 156],  # wall
        [190, 153, 153],  # fence
        [153, 153, 153],  # pole
        [250, 170, 30],   # traffic light
        [220, 220, 0],    # traffic sign
        [107, 142, 35],   # vegetation
        [152, 251, 152],  # terrain
        [70, 130, 180],   # sky
        [220, 20, 60],    # person
        [255, 0, 0],      # rider
        [0, 0, 142],      # car
        [0, 0, 70],       # truck
        [0, 60, 100],     # bus
        [0, 80, 100],     # train
        [0, 0, 230],      # motorcycle
        [119, 11, 32],    # bicycle
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PAD ERFNet inference and evaluation on CARLA frames")
    parser.add_argument(
        "--mode",
        choices=["infer", "evaluate", "infer_and_evaluate"],
        default="infer_and_evaluate",
        help="Pipeline mode",
    )

    parser.add_argument("--pad-repo", type=Path, default=Path("../repos/pytorch-auto-drive"), help="Path to PAD repo")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/semantic_segmentation/erfnet/cityscapes_512x1024.py"),
        help="PAD config path relative to PAD repo",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data/checkpoints/erfnet_cityscapes_512x1024_20200918.pt"),
        help="ERFNet checkpoint path",
    )
    parser.add_argument(
        "--encoder-checkpoint",
        type=Path,
        default=Path("data/checkpoints/erfnet_encoder_pretrained.pth.tar"),
        help="ERFNet encoder pretrain path required by PAD model config",
    )

    parser.add_argument("--image-dir", type=Path, default=Path("data/processed/pad_format/images"), help="Input RGB directory")
    parser.add_argument("--gt-dir", type=Path, default=Path("data/processed/pad_format/masks"), help="Ground-truth mask directory")
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=Path("output/segmentation/predictions"),
        help="Predicted mask output directory",
    )
    parser.add_argument(
        "--overlay-dir",
        type=Path,
        default=Path("output/segmentation/overlays"),
        help="Overlay output directory",
    )
    parser.add_argument(
        "--eval-output",
        type=Path,
        default=Path("output/segmentation/eval_results.json"),
        help="Evaluation JSON output path",
    )

    parser.add_argument("--num-classes", type=int, default=19, help="Number of segmentation classes")
    parser.add_argument("--ignore-index", type=int, default=255, help="Ignore index in GT masks")
    parser.add_argument("--num-frames", type=int, default=0, help="Limit number of frames (0 = all)")
    parser.add_argument("--batch-size", type=int, default=1, help="Inference batch size")
    parser.add_argument("--mixed-precision", action="store_true", help="Enable autocast for inference")
    parser.add_argument("--device", default="cuda:0", help="Torch device")
    parser.add_argument("--save-overlays", action="store_true", help="Save overlay images")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite prediction/overlay directories")
    return parser.parse_args()


def load_pad_model(args: argparse.Namespace, repo_root: Path):
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from utils.args import read_config  # type: ignore
    from utils.models import MODELS  # type: ignore
    from utils.common import load_checkpoint  # type: ignore

    config_path = args.config
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    cfg = read_config(str(config_path))

    model_cfg = dict(cfg["model"])
    model_cfg["pretrained_weights"] = str(args.encoder_checkpoint.resolve())
    model = MODELS.from_dict(model_cfg)

    checkpoint = args.checkpoint.resolve() if not args.checkpoint.is_absolute() else args.checkpoint
    load_checkpoint(model, optimizer=None, lr_scheduler=None, filename=str(checkpoint), strict=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return model, cfg, device


def extract_input_size(cfg: dict[str, Any]) -> tuple[int, int]:
    aug = cfg.get("test_augmentation", {})
    transforms = aug.get("transforms", []) if isinstance(aug, dict) else []
    for t in transforms:
        if isinstance(t, dict) and t.get("name") == "Resize":
            size = t.get("size_image")
            if isinstance(size, (list, tuple)) and len(size) == 2:
                return int(size[0]), int(size[1])
    return (512, 1024)


def list_frame_pairs(image_dir: Path, gt_dir: Path | None, num_frames: int) -> list[tuple[Path, Path | None]]:
    images = sorted(image_dir.glob("*.png"))
    if num_frames > 0:
        images = images[:num_frames]
    if not images:
        raise RuntimeError(f"No input images found in {image_dir}")

    pairs: list[tuple[Path, Path | None]] = []
    for img in images:
        gt = None
        if gt_dir is not None:
            cand = gt_dir / img.name
            if cand.exists():
                gt = cand
        pairs.append((img, gt))
    return pairs


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    valid = (mask >= 0) & (mask < CITYSCAPES_COLORS.shape[0])
    out[valid] = CITYSCAPES_COLORS[mask[valid]]
    return out


def run_inference(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.pad_repo.resolve()
    image_dir = args.image_dir.resolve()

    pred_dir = args.pred_dir.resolve()
    overlay_dir = args.overlay_dir.resolve()
    if args.overwrite:
        if pred_dir.exists():
            for p in pred_dir.glob("*.png"):
                p.unlink()
        if overlay_dir.exists():
            for p in overlay_dir.glob("*.png"):
                p.unlink()

    pred_dir.mkdir(parents=True, exist_ok=True)
    if args.save_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    model, cfg, device = load_pad_model(args, repo_root=repo_root)
    in_h, in_w = extract_input_size(cfg)

    pairs = list_frame_pairs(image_dir=image_dir, gt_dir=None, num_frames=args.num_frames)

    total_time = 0.0
    processed = 0

    autocast_enabled = bool(args.mixed_precision and device.type == "cuda" and torch.__version__ >= "1.6.0")

    with torch.no_grad():
        for start in range(0, len(pairs), args.batch_size):
            batch_pairs = pairs[start : start + args.batch_size]
            batch_tensors = []
            batch_rgbs: list[np.ndarray] = []
            batch_hw: list[tuple[int, int]] = []
            batch_names: list[str] = []

            for img_path, _ in batch_pairs:
                bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise RuntimeError(f"Failed to read image: {img_path}")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                h0, w0 = rgb.shape[:2]

                tensor = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                tensor = torch.nn.functional.interpolate(
                    tensor, size=(in_h, in_w), mode="bilinear", align_corners=True
                )
                batch_tensors.append(tensor)
                batch_rgbs.append(rgb)
                batch_hw.append((h0, w0))
                batch_names.append(img_path.name)

            input_batch = torch.cat(batch_tensors, dim=0).to(device)

            t0 = time.time()
            if autocast_enabled:
                with torch.cuda.amp.autocast():
                    logits_batch = model(input_batch)["out"]
            else:
                logits_batch = model(input_batch)["out"]
            total_time += time.time() - t0

            for i, (h0, w0) in enumerate(batch_hw):
                logits = torch.nn.functional.interpolate(
                    logits_batch[i : i + 1], size=(h0, w0), mode="bilinear", align_corners=True
                )
                pred = logits.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
                processed += 1

                cv2.imwrite(str(pred_dir / batch_names[i]), pred)
                if args.save_overlays:
                    color = colorize_mask(pred)
                    overlay = cv2.addWeighted(batch_rgbs[i], 0.65, color, 0.35, 0.0)
                    cv2.imwrite(
                        str(overlay_dir / batch_names[i]),
                        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
                    )

    avg_ms = (total_time / processed * 1000.0) if processed > 0 else None
    fps = (processed / total_time) if total_time > 0 else None

    return {
        "num_frames": processed,
        "input_size_model": [in_h, in_w],
        "avg_forward_ms": avg_ms,
        "forward_fps": fps,
        "pred_dir": str(pred_dir),
        "overlay_dir": str(overlay_dir) if args.save_overlays else None,
        "mixed_precision": bool(autocast_enabled),
        "device": str(device),
    }


def compute_confusion(pred: np.ndarray, gt: np.ndarray, num_classes: int, ignore_index: int) -> np.ndarray:
    mask = gt != ignore_index
    gt_valid = gt[mask]
    pred_valid = pred[mask]
    valid = (gt_valid >= 0) & (gt_valid < num_classes) & (pred_valid >= 0) & (pred_valid < num_classes)
    gt_valid = gt_valid[valid]
    pred_valid = pred_valid[valid]
    hist = np.bincount(num_classes * gt_valid.astype(np.int64) + pred_valid.astype(np.int64), minlength=num_classes**2)
    return hist.reshape(num_classes, num_classes)


def evaluate_predictions(args: argparse.Namespace) -> dict[str, Any]:
    pred_dir = args.pred_dir.resolve()
    gt_dir = args.gt_dir.resolve()

    pred_files = sorted(pred_dir.glob("*.png"))
    if args.num_frames > 0:
        pred_files = pred_files[: args.num_frames]
    if not pred_files:
        raise RuntimeError(f"No predicted masks found in {pred_dir}")

    conf = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
    used = 0

    for pf in pred_files:
        gf = gt_dir / pf.name
        if not gf.exists():
            continue

        pred = cv2.imread(str(pf), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(str(gf), cv2.IMREAD_GRAYSCALE)
        if pred is None or gt is None:
            continue

        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

        conf += compute_confusion(pred=pred, gt=gt, num_classes=args.num_classes, ignore_index=args.ignore_index)
        used += 1

    if used == 0:
        raise RuntimeError("No matched prediction/GT pairs for evaluation")

    ious = []
    per_class = {}
    for cls in range(args.num_classes):
        tp = conf[cls, cls]
        fn = conf[cls, :].sum() - tp
        fp = conf[:, cls].sum() - tp
        denom = tp + fp + fn
        iou = float(tp / denom) if denom > 0 else float("nan")
        ious.append(iou)
        per_class[str(cls)] = {
            "iou": None if np.isnan(iou) else iou,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
        }

    valid_ious = [x for x in ious if not np.isnan(x)]
    miou = float(np.mean(valid_ious)) if valid_ious else None

    pixel_acc = float(np.trace(conf) / conf.sum()) if conf.sum() > 0 else None

    result = {
        "num_pairs_evaluated": used,
        "num_classes": args.num_classes,
        "ignore_index": args.ignore_index,
        "mIoU": miou,
        "pixel_accuracy": pixel_acc,
        "per_class": per_class,
    }
    return result


def main() -> int:
    args = parse_args()
    outputs: dict[str, Any] = {"mode": args.mode}

    if args.mode in ("infer", "infer_and_evaluate"):
        outputs["inference"] = run_inference(args)

    if args.mode in ("evaluate", "infer_and_evaluate"):
        eval_result = evaluate_predictions(args)
        outputs["evaluation"] = eval_result
        args.eval_output.parent.mkdir(parents=True, exist_ok=True)
        args.eval_output.write_text(json.dumps(eval_result, indent=2), encoding="utf-8")

    print("[PASS] run_segmentation complete")
    print(json.dumps(outputs, indent=2))
    if "evaluation" in outputs:
        print(f"[PASS] evaluation saved: {args.eval_output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
