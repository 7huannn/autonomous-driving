#!/usr/bin/env python3
"""Convert CARLA recorder outputs to PAD semantic-segmentation dataset format."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# CARLA 0.10+ semantic ID -> Cityscapes trainId (0..18, 255 ignore)
CARLA_010_TO_CITYSCAPES = {
    0: 255,  # NONE / unlabeled
    1: 0,    # Roads
    2: 1,    # Sidewalks
    3: 2,    # Buildings
    4: 3,    # Walls
    5: 4,    # Fences
    6: 5,    # Poles
    7: 6,    # TrafficLight
    8: 7,    # TrafficSigns
    9: 8,    # Vegetation
    10: 9,   # Terrain
    11: 10,  # Sky
    12: 11,  # Pedestrians
    13: 12,  # Rider
    14: 13,  # Car
    15: 14,  # Truck
    16: 15,  # Bus
    17: 16,  # Train
    18: 17,  # Motorcycle
    19: 18,  # Bicycle
    20: 255,  # Static
    21: 255,  # Dynamic
    22: 255,  # Other
    23: 255,  # Water
    24: 0,    # RoadLines -> road
    25: 9,    # Ground -> terrain
    26: 2,    # Bridge -> building-like structure
    27: 16,   # RailTrack -> train class proxy
    28: 4,    # GuardRail -> fence
}

# Legacy CARLA (<0.9.14) semantic ID -> Cityscapes trainId
CARLA_LEGACY_TO_CITYSCAPES = {
    0: 255,
    1: 2,
    2: 4,
    3: 255,
    4: 11,
    5: 5,
    6: 7,
    7: 0,
    8: 1,
    9: 8,
    10: 13,
    11: 3,
    12: 7,
    13: 10,
    14: 6,
    15: 255,
    16: 255,
    17: 255,
    18: 6,
    19: 255,
    20: 255,
    21: 255,
    22: 9,
}


@dataclass(frozen=True)
class SourceSelection:
    input_dir: Path
    num_common_frames: int
    stems: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert CARLA rgb/semantic data to PAD format (images, masks, train/val splits). "
            "Supports merging multiple raw datasets in source order."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Single CARLA raw recording directory (legacy option)",
    )
    parser.add_argument(
        "--input-dirs",
        type=Path,
        nargs="+",
        default=None,
        help="Multiple CARLA raw recording directories to merge in-order",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/pad_format"), help="Output dataset root")
    parser.add_argument(
        "--num-frames",
        type=int,
        default=0,
        help="Maximum number of frames per input dataset (0 = all)",
    )
    parser.add_argument("--start-index", type=int, default=0, help="Start index in each sorted source frame list")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    parser.add_argument("--shuffle-split", action="store_true", help="Shuffle frame IDs before split")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if exists (required when output-dir already exists)",
    )
    parser.add_argument(
        "--mapping-profile",
        choices=("carla_010", "legacy_023"),
        default="carla_010",
        help="Semantic ID mapping profile (default: carla_010)",
    )
    parser.add_argument(
        "--unknown-class-id",
        type=int,
        default=255,
        help="Fallback mapped class ID for unknown CARLA semantic tags",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path to save conversion summary JSON (default: <output-dir>/conversion_summary.json)",
    )
    parser.add_argument(
        "--manifest-json",
        type=Path,
        default=None,
        help="Optional path to save source manifest JSON (default: <output-dir>/conversion_manifest.json)",
    )
    return parser.parse_args()


def resolve_input_dirs(args: argparse.Namespace) -> list[Path]:
    if args.input_dirs and args.input_dir:
        raise ValueError("Use either --input-dir or --input-dirs, not both")
    if not args.input_dirs and not args.input_dir:
        raise ValueError("Either --input-dir or --input-dirs is required")
    if args.input_dirs:
        return [p.resolve() for p in args.input_dirs]
    return [args.input_dir.resolve()]


def mapping_table(profile: str) -> dict[int, int]:
    if profile == "carla_010":
        return CARLA_010_TO_CITYSCAPES
    return CARLA_LEGACY_TO_CITYSCAPES


def build_lut(profile: str, unknown_class_id: int) -> tuple[np.ndarray, np.ndarray]:
    table = mapping_table(profile)
    lut = np.full((256,), np.uint8(unknown_class_id), dtype=np.uint8)
    known = np.zeros((256,), dtype=bool)
    for src, dst in table.items():
        if 0 <= src < 256:
            lut[src] = np.uint8(dst)
            known[src] = True
    return lut, known


def collect_common_stems(input_dir: Path) -> list[str]:
    rgb_dir = input_dir / "rgb"
    sem_dir = input_dir / "semantic"
    if not rgb_dir.is_dir() or not sem_dir.is_dir():
        raise FileNotFoundError(f"Expected subdirectories 'rgb' and 'semantic' in {input_dir}")

    rgb_stems = {p.stem for p in rgb_dir.glob("*.png")}
    sem_stems = {p.stem for p in sem_dir.glob("*.png")}
    common = sorted(rgb_stems & sem_stems)
    if not common:
        raise RuntimeError(f"No common PNG stems between {rgb_dir} and {sem_dir}")
    return common


def select_sources(input_dirs: list[Path], start_index: int, num_frames: int) -> list[SourceSelection]:
    selected: list[SourceSelection] = []
    for input_dir in input_dirs:
        common = collect_common_stems(input_dir)
        if start_index >= len(common):
            raise RuntimeError(
                f"start-index={start_index} is out of range for {input_dir} (num_common_frames={len(common)})"
            )
        if num_frames > 0:
            stems = common[start_index : start_index + num_frames]
        else:
            stems = common[start_index:]
        if not stems:
            raise RuntimeError(f"No frames selected from {input_dir}")
        selected.append(SourceSelection(input_dir=input_dir, num_common_frames=len(common), stems=stems))
    return selected


def prepare_output_dirs(output_dir: Path, overwrite: bool) -> tuple[Path, Path, Path]:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=False)
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    splits_dir = output_dir / "splits"
    images_dir.mkdir(parents=True, exist_ok=False)
    masks_dir.mkdir(parents=True, exist_ok=False)
    splits_dir.mkdir(parents=True, exist_ok=False)
    return images_dir, masks_dir, splits_dir


def write_split_file(path: Path, frame_ids: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for frame_id in frame_ids:
            f.write(f"{frame_id}\n")


def compact_histogram(arr: np.ndarray) -> dict[str, int]:
    out: dict[str, int] = {}
    for idx, val in enumerate(arr.tolist()):
        if int(val) > 0:
            out[str(idx)] = int(val)
    return out


def main() -> int:
    args = parse_args()
    input_dirs = resolve_input_dirs(args)
    output_dir = args.output_dir.resolve()

    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train-ratio must be in (0, 1)")
    if args.start_index < 0:
        raise ValueError("--start-index must be >= 0")
    if args.num_frames < 0:
        raise ValueError("--num-frames must be >= 0")

    sources = select_sources(input_dirs=input_dirs, start_index=args.start_index, num_frames=args.num_frames)
    images_dir, masks_dir, splits_dir = prepare_output_dirs(output_dir, args.overwrite)

    lut, known_table = build_lut(profile=args.mapping_profile, unknown_class_id=args.unknown_class_id)

    raw_hist = np.zeros((256,), dtype=np.int64)
    mapped_hist = np.zeros((256,), dtype=np.int64)
    unknown_hist = np.zeros((256,), dtype=np.int64)

    converted = 0
    manifest_sources = []
    for source in sources:
        manifest_sources.append(
            {
                "input_dir": str(source.input_dir),
                "num_source_common_frames": int(source.num_common_frames),
                "num_selected_frames": int(len(source.stems)),
                "selected_first_stem": source.stems[0],
                "selected_last_stem": source.stems[-1],
            }
        )

        for stem in source.stems:
            src_rgb = source.input_dir / "rgb" / f"{stem}.png"
            src_sem = source.input_dir / "semantic" / f"{stem}.png"
            out_name = f"{converted:06d}.png"

            shutil.copy2(src_rgb, images_dir / out_name)

            sem_bgr = cv2.imread(str(src_sem), cv2.IMREAD_COLOR)
            if sem_bgr is None:
                raise RuntimeError(f"Failed to read semantic image: {src_sem}")

            carla_ids = sem_bgr[:, :, 2]
            raw_counts = np.bincount(carla_ids.reshape(-1), minlength=256).astype(np.int64)
            raw_hist += raw_counts

            unknown_mask = ~known_table[carla_ids]
            if np.any(unknown_mask):
                unknown_ids, unknown_counts = np.unique(carla_ids[unknown_mask], return_counts=True)
                for uid, cnt in zip(unknown_ids.tolist(), unknown_counts.tolist()):
                    unknown_hist[int(uid)] += int(cnt)

            mapped_mask = lut[carla_ids]
            mapped_counts = np.bincount(mapped_mask.reshape(-1), minlength=256).astype(np.int64)
            mapped_hist += mapped_counts

            if not cv2.imwrite(str(masks_dir / out_name), mapped_mask):
                raise RuntimeError(f"Failed to write mapped mask: {masks_dir / out_name}")

            converted += 1

    frame_ids = [f"{i:06d}" for i in range(converted)]
    split_ids = frame_ids.copy()
    if args.shuffle_split:
        rng = random.Random(args.seed)
        rng.shuffle(split_ids)

    train_count = int(len(split_ids) * args.train_ratio)
    train_ids = split_ids[:train_count]
    val_ids = split_ids[train_count:]

    write_split_file(splits_dir / "train.txt", train_ids)
    write_split_file(splits_dir / "val.txt", val_ids)

    sample_img = cv2.imread(str(images_dir / "000000.png"), cv2.IMREAD_COLOR)
    sample_mask = cv2.imread(str(masks_dir / "000000.png"), cv2.IMREAD_GRAYSCALE)
    if sample_img is None or sample_mask is None:
        raise RuntimeError("Failed to read converted sample for validation")
    if sample_img.shape[:2] != sample_mask.shape[:2]:
        raise RuntimeError(
            f"RGB/mask shape mismatch: {sample_img.shape[:2]} vs {sample_mask.shape[:2]}"
        )

    total_pixels = int(raw_hist.sum())
    unknown_pixels = int(unknown_hist.sum())
    ignore_pixels = int(mapped_hist[255])

    unknown_ids = [int(i) for i, c in enumerate(unknown_hist.tolist()) if c > 0]
    mapped_seen = [int(i) for i, c in enumerate(mapped_hist.tolist()) if c > 0]

    summary = {
        "input_dir": str(input_dirs[0]) if len(input_dirs) == 1 else None,
        "input_dirs": [str(p) for p in input_dirs],
        "output_dir": str(output_dir),
        "mapping_profile": args.mapping_profile,
        "num_source_datasets": len(sources),
        "num_source_common_frames_total": int(sum(s.num_common_frames for s in sources)),
        "num_converted_frames": converted,
        "start_index": args.start_index,
        "num_frames_requested_per_source": args.num_frames,
        "train_ratio": args.train_ratio,
        "train_frames": len(train_ids),
        "val_frames": len(val_ids),
        "shuffle_split": bool(args.shuffle_split),
        "seed": args.seed,
        "raw_semantic_ids_seen": [int(i) for i, c in enumerate(raw_hist.tolist()) if c > 0],
        "mapped_cityscapes_classes_seen": mapped_seen,
        "raw_semantic_histogram": compact_histogram(raw_hist),
        "mapped_class_histogram": compact_histogram(mapped_hist),
        "unknown_semantic_ids": unknown_ids,
        "unknown_semantic_histogram": compact_histogram(unknown_hist),
        "unknown_pixel_ratio": (unknown_pixels / total_pixels) if total_pixels > 0 else None,
        "ignore_pixel_ratio": (ignore_pixels / total_pixels) if total_pixels > 0 else None,
        "sample_image_shape_hwc": list(sample_img.shape),
        "sample_mask_shape_hw": list(sample_mask.shape),
    }

    summary_path = args.summary_json.resolve() if args.summary_json else output_dir / "conversion_summary.json"
    manifest_path = args.manifest_json.resolve() if args.manifest_json else output_dir / "conversion_manifest.json"

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    manifest = {
        "output_dir": str(output_dir),
        "mapping_profile": args.mapping_profile,
        "sources": manifest_sources,
        "frame_id_policy": "Output frame IDs are regenerated sequentially from 000000 in merged source order.",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("[PASS] PAD conversion complete")
    print(json.dumps(summary, indent=2))
    print(f"[PASS] Summary saved: {summary_path}")
    print(f"[PASS] Manifest saved: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
