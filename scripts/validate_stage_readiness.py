#!/usr/bin/env python3
"""Hard-gate validator for Stage 03-08 readiness before Stage 09."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Stage 03-08 artifacts with hard-fail gates")
    parser.add_argument("--raw-dir", type=Path, required=True, help="Canonical raw dataset directory (rgb/semantic/lidar/metadata)")
    parser.add_argument("--pad-summary", type=Path, default=None, help="PAD conversion summary JSON")
    parser.add_argument("--pcdet-summary", type=Path, default=None, help="PCDet conversion summary JSON")
    parser.add_argument("--pcdet-label-dir", type=Path, default=None, help="PCDet labels directory (fallback if summary missing)")
    parser.add_argument("--seg-pred-dir", type=Path, default=None, help="Segmentation prediction directory (*.png)")
    parser.add_argument("--det-pred-dir", type=Path, default=None, help="LiDAR prediction directory (*.txt)")
    parser.add_argument(
        "--min-pad-non-ignore-classes",
        type=int,
        default=2,
        help="Minimum number of non-ignore mapped classes required in PAD summary",
    )
    parser.add_argument(
        "--no-require-secondary-gt",
        dest="require_secondary_gt",
        action="store_false",
        help="Disable secondary GT requirement (debug only)",
    )
    parser.set_defaults(require_secondary_gt=True)
    parser.add_argument("--allow-empty-det-preds", action="store_true", help="Allow empty det prediction directory")
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("output/stage_validation/stage_readiness_report.json"),
        help="Path to write validation report JSON",
    )
    return parser.parse_args()


def list_stems(path: Path, pattern: str) -> list[str]:
    return sorted(p.stem for p in path.glob(pattern))


def check_raw_integrity(raw_dir: Path) -> dict[str, Any]:
    rgb_dir = raw_dir / "rgb"
    sem_dir = raw_dir / "semantic"
    lidar_dir = raw_dir / "lidar"
    meta_dir = raw_dir / "metadata"

    missing_dirs = [str(p) for p in (rgb_dir, sem_dir, lidar_dir, meta_dir) if not p.is_dir()]
    if missing_dirs:
        return {
            "pass": False,
            "reason": f"Missing required subdirs: {missing_dirs}",
            "missing_dirs": missing_dirs,
        }

    rgb_stems = set(list_stems(rgb_dir, "*.png"))
    sem_stems = set(list_stems(sem_dir, "*.png"))
    lidar_stems = set(list_stems(lidar_dir, "*.npy"))
    meta_stems = set(list_stems(meta_dir, "*.json"))

    counts = {
        "rgb": len(rgb_stems),
        "semantic": len(sem_stems),
        "lidar": len(lidar_stems),
        "metadata": len(meta_stems),
    }

    common = rgb_stems & sem_stems & lidar_stems & meta_stems
    aligned = (
        len(rgb_stems) == len(sem_stems) == len(lidar_stems) == len(meta_stems) == len(common)
        and len(common) > 0
    )

    monotonic = True
    sensor_sync_all = True
    missing_sensor_keys = 0
    invalid_meta = 0
    prev_ts = -1.0

    for stem in sorted(common):
        meta_path = meta_dir / f"{stem}.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            timestamp = float(meta.get("timestamp", -1.0))
            if timestamp <= prev_ts:
                monotonic = False
            prev_ts = timestamp

            sf = meta.get("sensor_frames", {})
            if not isinstance(sf, dict) or not all(k in sf for k in ("rgb_camera", "semantic_camera", "lidar")):
                missing_sensor_keys += 1
                sensor_sync_all = False
            else:
                values = [int(sf["rgb_camera"]), int(sf["semantic_camera"]), int(sf["lidar"])]
                if not (values[0] == values[1] == values[2]):
                    sensor_sync_all = False
        except Exception:
            invalid_meta += 1
            monotonic = False
            sensor_sync_all = False

    ok = aligned and monotonic and sensor_sync_all and missing_sensor_keys == 0 and invalid_meta == 0
    return {
        "pass": ok,
        "counts": counts,
        "num_common_stems": len(common),
        "aligned": aligned,
        "timestamps_monotonic": monotonic,
        "sensor_sync_all": sensor_sync_all,
        "missing_sensor_keys": missing_sensor_keys,
        "invalid_meta": invalid_meta,
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def check_pad_coverage(pad_summary: Path, min_non_ignore_classes: int) -> dict[str, Any]:
    if not pad_summary.exists():
        return {"pass": False, "reason": f"PAD summary not found: {pad_summary}"}

    data = load_json(pad_summary)
    mapped_hist = data.get("mapped_class_histogram", {})
    if not isinstance(mapped_hist, dict):
        return {"pass": False, "reason": "mapped_class_histogram missing in PAD summary"}

    non_ignore_ids = [int(k) for k, v in mapped_hist.items() if int(k) != 255 and int(v) > 0]
    unknown_ids = data.get("unknown_semantic_ids", [])

    ok = len(non_ignore_ids) >= min_non_ignore_classes
    return {
        "pass": ok,
        "non_ignore_classes_seen": sorted(non_ignore_ids),
        "num_non_ignore_classes": len(non_ignore_ids),
        "unknown_semantic_ids": unknown_ids,
        "threshold": min_non_ignore_classes,
    }


def parse_label_line(line: str) -> str | None:
    parts = line.strip().split()
    if len(parts) != 8:
        return None
    return parts[-1]


def check_pcdet_gt(pcdet_summary: Path | None, pcdet_label_dir: Path | None, require_secondary: bool) -> dict[str, Any]:
    label_hist = {"Vehicle": 0, "Pedestrian": 0, "Cyclist": 0}
    frames_with_empty_labels = None
    num_converted = None

    if pcdet_summary is not None and pcdet_summary.exists():
        data = load_json(pcdet_summary)
        src_hist = data.get("label_class_histogram", {})
        if isinstance(src_hist, dict):
            for cls in label_hist:
                label_hist[cls] = int(src_hist.get(cls, 0))
        frames_with_empty_labels = data.get("frames_with_empty_labels")
        num_converted = data.get("num_converted_frames")

    elif pcdet_label_dir is not None and pcdet_label_dir.is_dir():
        label_files = sorted(pcdet_label_dir.glob("*.txt"))
        num_converted = len(label_files)
        empty = 0
        for lf in label_files:
            lines = [ln for ln in lf.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if not lines:
                empty += 1
            for line in lines:
                cls = parse_label_line(line)
                if cls in label_hist:
                    label_hist[cls] += 1
        frames_with_empty_labels = empty
    else:
        return {
            "pass": False,
            "reason": "Need --pcdet-summary or --pcdet-label-dir for GT validation",
        }

    has_vehicle = label_hist["Vehicle"] > 0
    has_secondary = (label_hist["Pedestrian"] > 0) or (label_hist["Cyclist"] > 0)

    if require_secondary:
        ok = has_vehicle and has_secondary
    else:
        ok = has_vehicle

    if num_converted is not None and num_converted > 0 and frames_with_empty_labels is not None:
        all_empty = int(frames_with_empty_labels) >= int(num_converted)
        ok = ok and not all_empty

    return {
        "pass": ok,
        "label_class_histogram": label_hist,
        "has_vehicle": has_vehicle,
        "has_secondary": has_secondary,
        "require_secondary": bool(require_secondary),
        "frames_with_empty_labels": frames_with_empty_labels,
        "num_converted_frames": num_converted,
    }


def check_output_alignment(seg_pred_dir: Path, det_pred_dir: Path, allow_empty_det_preds: bool) -> dict[str, Any]:
    if not seg_pred_dir.is_dir():
        return {"pass": False, "reason": f"Seg prediction dir not found: {seg_pred_dir}"}
    if not det_pred_dir.is_dir():
        return {"pass": False, "reason": f"Det prediction dir not found: {det_pred_dir}"}

    seg_stems = set(list_stems(seg_pred_dir, "*.png"))
    det_stems = set(list_stems(det_pred_dir, "*.txt"))

    if not seg_stems:
        return {"pass": False, "reason": "No segmentation prediction files found"}
    if not det_stems and not allow_empty_det_preds:
        return {"pass": False, "reason": "No detection prediction files found"}

    same = seg_stems == det_stems
    return {
        "pass": same,
        "num_seg_preds": len(seg_stems),
        "num_det_preds": len(det_stems),
        "num_only_seg": len(seg_stems - det_stems),
        "num_only_det": len(det_stems - seg_stems),
    }


def main() -> int:
    args = parse_args()

    report: dict[str, Any] = {
        "raw_integrity": check_raw_integrity(args.raw_dir.resolve()),
    }

    if args.pad_summary is not None:
        report["pad_coverage"] = check_pad_coverage(
            pad_summary=args.pad_summary.resolve(),
            min_non_ignore_classes=int(args.min_pad_non_ignore_classes),
        )

    report["pcdet_gt"] = check_pcdet_gt(
        pcdet_summary=args.pcdet_summary.resolve() if args.pcdet_summary is not None else None,
        pcdet_label_dir=args.pcdet_label_dir.resolve() if args.pcdet_label_dir is not None else None,
        require_secondary=bool(args.require_secondary_gt),
    )

    if args.seg_pred_dir is not None and args.det_pred_dir is not None:
        report["output_alignment"] = check_output_alignment(
            seg_pred_dir=args.seg_pred_dir.resolve(),
            det_pred_dir=args.det_pred_dir.resolve(),
            allow_empty_det_preds=bool(args.allow_empty_det_preds),
        )

    failed = [name for name, result in report.items() if not bool(result.get("pass", False))]
    report["overall_pass"] = len(failed) == 0
    report["failed_checks"] = failed

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"[PASS] Report saved: {args.report_json.resolve()}")

    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
