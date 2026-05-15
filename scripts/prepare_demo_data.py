#!/usr/bin/env python3
"""Prepare a compact aligned CARLA raw subset for offline demo."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


MODAL_EXT = {
    "rgb": ".png",
    "semantic": ".png",
    "lidar": ".npy",
    "metadata": ".json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare aligned demo subset from a CARLA raw recording")
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw/recording_001"), help="Source raw recording directory")
    parser.add_argument("--output-dir", type=Path, default=Path("deploy/demo_data"), help="Output subset directory")
    parser.add_argument("--num-frames", type=int, default=100, help="Number of aligned frames to copy")
    parser.add_argument("--start-index", type=int, default=0, help="Start index in sorted aligned stems")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output directory if it exists")
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("deploy/demo_data_report.json"),
        help="Path to write preparation report JSON",
    )
    return parser.parse_args()


def collect_stems(modal_dir: Path, ext: str) -> set[str]:
    if not modal_dir.is_dir():
        raise FileNotFoundError(f"Missing directory: {modal_dir}")
    return {p.stem for p in modal_dir.glob(f"*{ext}")}


def main() -> int:
    args = parse_args()
    src = args.input_dir.resolve()
    dst = args.output_dir.resolve()

    if args.num_frames <= 0:
        raise ValueError("--num-frames must be > 0")

    stems_by_modal: dict[str, set[str]] = {}
    for modal, ext in MODAL_EXT.items():
        stems_by_modal[modal] = collect_stems(src / modal, ext)

    aligned = sorted(set.intersection(*stems_by_modal.values()))
    if not aligned:
        raise RuntimeError("No aligned stems found across rgb/semantic/lidar/metadata")

    if args.start_index < 0 or args.start_index >= len(aligned):
        raise RuntimeError(f"start-index {args.start_index} out of range (aligned={len(aligned)})")

    selected = aligned[args.start_index : args.start_index + args.num_frames]
    if not selected:
        raise RuntimeError("Selection is empty after applying start-index/num-frames")

    if dst.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory exists: {dst}. Use --overwrite")
        shutil.rmtree(dst)

    for modal in MODAL_EXT:
        (dst / modal).mkdir(parents=True, exist_ok=False)

    calib_src = src / "calib"
    if calib_src.is_dir():
        shutil.copytree(calib_src, dst / "calib")

    mapping: list[dict[str, str]] = []
    for i, stem in enumerate(selected, start=1):
        out_stem = f"{i-1:06d}"
        for modal, ext in MODAL_EXT.items():
            src_file = src / modal / f"{stem}{ext}"
            dst_file = dst / modal / f"{out_stem}{ext}"
            shutil.copy2(src_file, dst_file)
        mapping.append({"source_stem": stem, "demo_stem": out_stem})
        if i % 20 == 0 or i == len(selected):
            print(f"[prepare_demo_data] copied {i}/{len(selected)} frames")

    report = {
        "input_dir": str(src),
        "output_dir": str(dst),
        "aligned_frame_count": len(aligned),
        "selected_frame_count": len(selected),
        "start_index": args.start_index,
        "num_frames": args.num_frames,
        "first_source_stem": selected[0],
        "last_source_stem": selected[-1],
        "first_demo_stem": mapping[0]["demo_stem"],
        "last_demo_stem": mapping[-1]["demo_stem"],
        "has_calib": calib_src.is_dir(),
        "id_policy": "Demo stems are regenerated sequentially from 000000 in source order.",
        "source_to_demo_mapping": mapping,
    }

    report_path = args.report_json.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[prepare_demo_data] report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
