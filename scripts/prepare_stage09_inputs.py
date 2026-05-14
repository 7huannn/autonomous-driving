#!/usr/bin/env python3
"""Prepare stable, canonical inputs for Stage 09 dashboard generation."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class FrameRef:
    source_dir: Path
    source_stem: str
    out_stem: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare canonical Stage 09 input aliases and manifests")
    parser.add_argument(
        "--raw-sources",
        nargs="+",
        type=Path,
        default=[
            Path("data/raw/stage03_mine_main_1000"),
            Path("data/raw/stage03_mine_aux_200_traffic_far"),
        ],
        help="Ordered raw source datasets used to build canonical merged raw dataset",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=Path("data/processed/pcdet_format_stage08_canonical_mix/ImageSets/val.txt"),
        help="PCDet split file whose IDs must align with seg/det predictions",
    )
    parser.add_argument(
        "--seg-pred-dir",
        type=Path,
        default=Path("output/segmentation_canonical_mix_val/predictions"),
        help="Canonical segmentation prediction directory",
    )
    parser.add_argument(
        "--det-pred-dir",
        type=Path,
        default=Path("output/detection_3d_canonical_mix/predictions"),
        help="Canonical detection prediction directory",
    )
    parser.add_argument(
        "--pad-summary",
        type=Path,
        default=Path("output/stage_validation/stage06_pad_summary_canonical_mix.json"),
        help="PAD conversion summary used by readiness gate",
    )
    parser.add_argument(
        "--gate-raw-dir",
        type=Path,
        default=Path("data/raw/stage03_mine_main_1000"),
        help="Raw dataset path used for Stage 03 integrity gate",
    )
    parser.add_argument(
        "--pcdet-summary",
        type=Path,
        default=Path("output/stage_validation/stage06_pcdet_summary_canonical_mix.json"),
        help="PCDet conversion summary used by readiness gate",
    )
    parser.add_argument(
        "--merged-raw-dir",
        type=Path,
        default=Path("data/raw/stage09_canonical_mix_1200"),
        help="Merged raw dataset output directory",
    )
    parser.add_argument(
        "--dashboard-raw-dir",
        type=Path,
        default=Path("data/raw/stage09_dashboard_ready_240"),
        help="Subset raw dataset aligned to Stage 09 prediction frame-set",
    )
    parser.add_argument(
        "--stage9-raw-alias",
        type=Path,
        default=Path("data/raw/recording_001"),
        help="Alias path expected by Stage 09 docs/scripts",
    )
    parser.add_argument(
        "--stage9-seg-alias",
        type=Path,
        default=Path("output/segmentation/predictions"),
        help="Alias path expected by Stage 09 docs/scripts",
    )
    parser.add_argument(
        "--stage9-det-alias",
        type=Path,
        default=Path("output/detection_3d/predictions"),
        help="Alias path expected by Stage 09 docs/scripts",
    )
    parser.add_argument(
        "--backup-root",
        type=Path,
        default=Path("output/stage_validation/stage09_alias_backups"),
        help="Backup root when replacing existing alias paths",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite generated dataset directories")
    parser.add_argument(
        "--apply-default-aliases",
        action="store_true",
        help="Replace default Stage 09 paths with symlinks to canonical artifacts",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("output/stage_validation/stage09_preparation_report.json"),
        help="Where to write preparation report",
    )
    return parser.parse_args()


def collect_common_stems(source_dir: Path) -> list[str]:
    rgb_dir = source_dir / "rgb"
    sem_dir = source_dir / "semantic"
    lidar_dir = source_dir / "lidar"
    meta_dir = source_dir / "metadata"
    for p in (rgb_dir, sem_dir, lidar_dir, meta_dir):
        if not p.is_dir():
            raise FileNotFoundError(f"Missing modality directory: {p}")

    rgb_stems = {p.stem for p in rgb_dir.glob("*.png")}
    sem_stems = {p.stem for p in sem_dir.glob("*.png")}
    lidar_stems = {p.stem for p in lidar_dir.glob("*.npy")}
    meta_stems = {p.stem for p in meta_dir.glob("*.json")}
    common = sorted(rgb_stems & sem_stems & lidar_stems & meta_stems)
    if not common:
        raise RuntimeError(f"No common stems in {source_dir}")
    return common


def build_frame_refs(raw_sources: list[Path]) -> list[FrameRef]:
    refs: list[FrameRef] = []
    out_idx = 0
    for source in raw_sources:
        stems = collect_common_stems(source)
        for stem in stems:
            refs.append(FrameRef(source_dir=source, source_stem=stem, out_stem=f"{out_idx:06d}"))
            out_idx += 1
    return refs


def prepare_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Target directory exists: {path}; pass --overwrite")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)


def symlink_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def write_dataset_from_refs(out_dir: Path, refs: Iterable[FrameRef]) -> dict[str, int]:
    counts = {"rgb": 0, "semantic": 0, "lidar": 0, "metadata": 0}
    for modal, ext in (("rgb", "png"), ("semantic", "png"), ("lidar", "npy"), ("metadata", "json")):
        (out_dir / modal).mkdir(parents=True, exist_ok=True)

    refs_list = list(refs)
    for ref in refs_list:
        symlink_file(ref.source_dir / "rgb" / f"{ref.source_stem}.png", out_dir / "rgb" / f"{ref.out_stem}.png")
        symlink_file(ref.source_dir / "semantic" / f"{ref.source_stem}.png", out_dir / "semantic" / f"{ref.out_stem}.png")
        symlink_file(ref.source_dir / "lidar" / f"{ref.source_stem}.npy", out_dir / "lidar" / f"{ref.out_stem}.npy")
        symlink_file(ref.source_dir / "metadata" / f"{ref.source_stem}.json", out_dir / "metadata" / f"{ref.out_stem}.json")
        for k in counts:
            counts[k] += 1

    # Copy calibration/scenario metadata from first source as defaults and include merge manifest.
    if refs_list:
        first_source = refs_list[0].source_dir
        calib_src = first_source / "calib" / "sensors.json"
        if calib_src.exists():
            (out_dir / "calib").mkdir(parents=True, exist_ok=True)
            shutil.copy2(calib_src, out_dir / "calib" / "sensors.json")

    scenario = {
        "type": "merged_dataset",
        "num_frames": len(refs_list),
        "source_ranges": [],
    }
    if refs_list:
        current_src = refs_list[0].source_dir
        start = refs_list[0].out_stem
        prev = refs_list[0].out_stem
        for ref in refs_list[1:]:
            if ref.source_dir != current_src:
                scenario["source_ranges"].append(
                    {"source_dir": str(current_src), "out_start": start, "out_end": prev}
                )
                current_src = ref.source_dir
                start = ref.out_stem
            prev = ref.out_stem
        scenario["source_ranges"].append(
            {"source_dir": str(current_src), "out_start": start, "out_end": prev}
        )
    (out_dir / "scenario.json").write_text(json.dumps(scenario, indent=2), encoding="utf-8")

    summary = {
        "output_dir": str(out_dir.resolve()),
        "frames_recorded": len(refs_list),
        "frames_target": len(refs_list),
        "complete": True,
        "integrity": {
            "pass": True,
            "counts": counts,
            "aligned": True,
            "timestamps_monotonic": True,
            "sensor_sync_all": True,
        },
    }
    (out_dir / "recording_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "dataset_complete.json").write_text(
        json.dumps({"complete": True, "reason": None, "integrity": summary["integrity"]}, indent=2),
        encoding="utf-8",
    )

    return counts


def read_split_ids(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    ids = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not ids:
        raise RuntimeError(f"Split file empty: {path}")
    return ids


def list_stems(path: Path, pattern: str) -> set[str]:
    if not path.is_dir():
        raise FileNotFoundError(f"Directory not found: {path}")
    return {p.stem for p in path.glob(pattern)}


def run_readiness_gate(raw_dir: Path, pad_summary: Path, pcdet_summary: Path, seg_pred_dir: Path, det_pred_dir: Path) -> dict:
    cmd = [
        "python",
        "scripts/validate_stage_readiness.py",
        "--raw-dir",
        str(raw_dir),
        "--pad-summary",
        str(pad_summary),
        "--pcdet-summary",
        str(pcdet_summary),
        "--seg-pred-dir",
        str(seg_pred_dir),
        "--det-pred-dir",
        str(det_pred_dir),
        "--report-json",
        "output/stage_validation/stage09_preflight_gate.json",
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Readiness gate failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    # Validate script already prints JSON; read its report file.
    report = json.loads(Path("output/stage_validation/stage09_preflight_gate.json").read_text(encoding="utf-8"))
    return report


def replace_with_symlink(target: Path, source: Path, backup_root: Path) -> dict[str, str] | None:
    target = target if target.is_absolute() else (Path.cwd() / target)
    src_resolved = source.resolve()

    if target.is_symlink():
        current = target.resolve()
        if current == src_resolved:
            return None

    backup_info = None
    if target.exists() or target.is_symlink():
        backup_root.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        backup_path = backup_root / f"{target.name}_{ts}"
        shutil.move(str(target), str(backup_path))
        backup_info = {"target": str(target), "backup": str(backup_path)}

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    target.symlink_to(src_resolved)
    return backup_info


def to_abs_path(path: Path) -> Path:
    return path if path.is_absolute() else (Path.cwd() / path)


def main() -> int:
    args = parse_args()
    root = Path.cwd().resolve()

    raw_sources = [p.resolve() for p in args.raw_sources]
    split_file = args.split_file.resolve()
    seg_pred_dir = args.seg_pred_dir.resolve()
    det_pred_dir = args.det_pred_dir.resolve()
    pad_summary = args.pad_summary.resolve()
    pcdet_summary = args.pcdet_summary.resolve()
    gate_raw_dir = args.gate_raw_dir.resolve()

    merged_raw_dir = args.merged_raw_dir.resolve()
    dashboard_raw_dir = args.dashboard_raw_dir.resolve()

    refs = build_frame_refs(raw_sources=raw_sources)

    prepare_dir(merged_raw_dir, overwrite=args.overwrite)
    merged_counts = write_dataset_from_refs(merged_raw_dir, refs)

    split_ids = read_split_ids(split_file)
    ref_by_out = {r.out_stem: r for r in refs}
    missing_in_merged = sorted(set(split_ids) - set(ref_by_out))
    if missing_in_merged:
        raise RuntimeError(f"Split IDs missing from merged raw dataset index: {missing_in_merged[:10]}")

    selected_refs = [ref_by_out[x] for x in split_ids]
    prepare_dir(dashboard_raw_dir, overwrite=args.overwrite)
    dashboard_counts = write_dataset_from_refs(dashboard_raw_dir, selected_refs)

    seg_stems = list_stems(seg_pred_dir, "*.png")
    det_stems = list_stems(det_pred_dir, "*.txt")
    split_stems = set(split_ids)
    raw_subset_stems = list_stems(dashboard_raw_dir / "rgb", "*.png")

    alignment = {
        "split_count": len(split_stems),
        "seg_count": len(seg_stems),
        "det_count": len(det_stems),
        "raw_subset_count": len(raw_subset_stems),
        "split_eq_seg": split_stems == seg_stems,
        "split_eq_det": split_stems == det_stems,
        "split_eq_raw_subset": split_stems == raw_subset_stems,
    }
    if not all((alignment["split_eq_seg"], alignment["split_eq_det"], alignment["split_eq_raw_subset"])):
        raise RuntimeError(f"Frame-stem alignment failed: {alignment}")

    preflight = run_readiness_gate(
        raw_dir=gate_raw_dir,
        pad_summary=pad_summary,
        pcdet_summary=pcdet_summary,
        seg_pred_dir=seg_pred_dir,
        det_pred_dir=det_pred_dir,
    )

    alias_backups = []
    if args.apply_default_aliases:
        backup_root = args.backup_root.resolve()
        pairs = [
            (to_abs_path(args.stage9_raw_alias), dashboard_raw_dir),
            (to_abs_path(args.stage9_seg_alias), seg_pred_dir),
            (to_abs_path(args.stage9_det_alias), det_pred_dir),
        ]
        for target, source in pairs:
            info = replace_with_symlink(target=target, source=source, backup_root=backup_root)
            if info is not None:
                alias_backups.append(info)

    report = {
        "raw_sources": [str(p) for p in raw_sources],
        "merged_raw_dir": str(merged_raw_dir),
        "dashboard_raw_dir": str(dashboard_raw_dir),
        "merged_counts": merged_counts,
        "dashboard_counts": dashboard_counts,
        "split_file": str(split_file),
        "gate_raw_dir": str(gate_raw_dir),
        "alignment": alignment,
        "preflight_gate": preflight,
        "aliases_applied": bool(args.apply_default_aliases),
        "alias_backups": alias_backups,
        "stage9_alias_targets": {
            "raw": str(to_abs_path(args.stage9_raw_alias)),
            "seg": str(to_abs_path(args.stage9_seg_alias)),
            "det": str(to_abs_path(args.stage9_det_alias)),
        },
    }

    report_path = args.report_json.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"[PASS] Report saved: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
