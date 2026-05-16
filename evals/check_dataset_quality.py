#!/usr/bin/env python3
"""
Dataset Quality Check for CARLA Perception Lab Experiment Harness.

Inspects a PCDet-format dataset path and validates structural integrity:
- Required info/dbinfo pickle files
- Train/val split files in ImageSets/
- Sample count (fail on 0)
- Label file readability and class distribution
- Point cloud file existence
- Optional point-support statistics

Usage:
    python evals/check_dataset_quality.py \
        --experiment-id EXP001 \
        --dataset data/processed/pcdet_format_ped_strict20_easy4_v1
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def check_path_exists(path):
    """Check if a path exists."""
    return Path(path).exists()


def count_files(directory, extension):
    """Count files with a given extension in a directory."""
    d = Path(directory)
    if not d.exists():
        return 0
    return len(list(d.glob(f"*.{extension}")))


def read_split_file(split_path):
    """Read a split file and return list of sample IDs."""
    if not Path(split_path).exists():
        return None
    with open(split_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def parse_label_file(label_path):
    """
    Parse a KITTI-style label file.
    Expected format per line: x y z dx dy dz heading class_name
    Returns list of class names found.
    """
    classes = []
    try:
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 8:
                    # Class name is the last token
                    classes.append(parts[-1])
                elif len(parts) >= 1:
                    # Try last token anyway
                    classes.append(parts[-1])
    except Exception:
        return None
    return classes


def compute_point_stats(points_dir, sample_ids=None, max_samples=50):
    """
    Compute basic point cloud statistics.
    Returns dict with min/max/mean point counts.
    """
    points_path = Path(points_dir)
    if not points_path.exists():
        return None

    try:
        import numpy as np
    except ImportError:
        return {"error": "numpy not available for point stats"}

    point_counts = []
    files = sorted(points_path.glob("*.npy"))
    if sample_ids is not None:
        files = [points_path / f"{sid}.npy" for sid in sample_ids if (points_path / f"{sid}.npy").exists()]

    for f in files[:max_samples]:
        try:
            pts = np.load(str(f))
            point_counts.append(pts.shape[0])
        except Exception:
            point_counts.append(0)

    if not point_counts:
        return None

    return {
        "num_files_checked": len(point_counts),
        "min_points": int(min(point_counts)),
        "max_points": int(max(point_counts)),
        "mean_points": float(sum(point_counts) / len(point_counts)),
        "zero_point_files": sum(1 for pc in point_counts if pc == 0),
    }


def run_dataset_quality_check(dataset_path, experiment_id):
    """
    Run full dataset quality check.

    Args:
        dataset_path: Path to PCDet-format dataset.
        experiment_id: Experiment identifier for output naming.

    Returns:
        dict: Full quality check results.
    """
    ds = Path(dataset_path)
    results = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "overall_pass": True,
        "failures": [],
        "warnings": [],
        "checks": {},
    }

    # 1) Dataset path existence
    if not ds.exists():
        results["overall_pass"] = False
        results["failures"].append(f"Dataset path does not exist: {dataset_path}")
        return results

    if not ds.is_dir():
        results["overall_pass"] = False
        results["failures"].append(f"Dataset path is not a directory: {dataset_path}")
        return results

    results["checks"]["path_exists"] = True

    # 2) Info files
    info_files = {
        "custom_infos_train.pkl": ds / "custom_infos_train.pkl",
        "custom_infos_val.pkl": ds / "custom_infos_val.pkl",
        "custom_dbinfos_train.pkl": ds / "custom_dbinfos_train.pkl",
    }
    results["checks"]["info_files"] = {}
    for name, path in info_files.items():
        exists = path.exists()
        results["checks"]["info_files"][name] = {
            "exists": exists,
            "size_bytes": path.stat().st_size if exists else 0,
        }
        if not exists and name == "custom_infos_val.pkl":
            results["overall_pass"] = False
            results["failures"].append(f"Required info file missing: {name}")
        elif not exists:
            results["warnings"].append(f"Info file missing (may be optional): {name}")

    # 3) ImageSets (train/val splits)
    imagesets_dir = ds / "ImageSets"
    results["checks"]["imagesets"] = {"dir_exists": imagesets_dir.exists()}

    train_split = read_split_file(imagesets_dir / "train.txt")
    val_split = read_split_file(imagesets_dir / "val.txt")

    results["checks"]["imagesets"]["train_split"] = {
        "exists": train_split is not None,
        "count": len(train_split) if train_split else 0,
    }
    results["checks"]["imagesets"]["val_split"] = {
        "exists": val_split is not None,
        "count": len(val_split) if val_split else 0,
    }

    if val_split is None:
        results["overall_pass"] = False
        results["failures"].append("Val split file missing: ImageSets/val.txt")
    elif len(val_split) == 0:
        results["overall_pass"] = False
        results["failures"].append("Val split is empty (0 samples)")

    if train_split is not None and len(train_split) == 0:
        results["warnings"].append("Train split is empty (0 samples)")

    # 4) Sample counts
    labels_dir = ds / "labels"
    points_dir = ds / "points"

    num_labels = count_files(labels_dir, "txt")
    num_points = count_files(points_dir, "npy")

    results["checks"]["sample_counts"] = {
        "label_files": num_labels,
        "point_cloud_files": num_points,
        "labels_dir_exists": labels_dir.exists(),
        "points_dir_exists": points_dir.exists(),
    }

    total_samples = max(num_labels, num_points)
    if total_samples == 0:
        results["overall_pass"] = False
        results["failures"].append("Dataset has 0 samples (no label or point cloud files)")
    elif num_labels == 0:
        results["overall_pass"] = False
        results["failures"].append("No label files found")
    elif num_points == 0:
        results["overall_pass"] = False
        results["failures"].append("No point cloud files found")

    if num_labels != num_points:
        results["warnings"].append(
            f"Label/point mismatch: {num_labels} labels vs {num_points} point clouds"
        )

    # 5) Class distribution from labels
    if labels_dir.exists():
        all_classes = Counter()
        readable_labels = 0
        unreadable_labels = 0

        for lf in sorted(labels_dir.glob("*.txt")):
            classes = parse_label_file(lf)
            if classes is not None:
                readable_labels += 1
                for c in classes:
                    all_classes[c] += 1
            else:
                unreadable_labels += 1

        results["checks"]["class_distribution"] = {
            "readable_label_files": readable_labels,
            "unreadable_label_files": unreadable_labels,
            "classes": dict(all_classes),
            "total_annotations": sum(all_classes.values()),
        }

        if unreadable_labels > 0:
            results["warnings"].append(f"{unreadable_labels} label files could not be parsed")

        if sum(all_classes.values()) == 0 and readable_labels > 0:
            results["warnings"].append("All label files are readable but contain 0 annotations")
    else:
        results["checks"]["class_distribution"] = None

    # 6) Point support statistics (optional)
    if points_dir.exists():
        try:
            all_sample_ids = val_split if val_split else None
            stats = compute_point_stats(points_dir, sample_ids=all_sample_ids)
            results["checks"]["point_stats"] = stats
        except Exception as e:
            results["checks"]["point_stats"] = {"error": str(e)}
    else:
        results["checks"]["point_stats"] = None

    # 7) Conversion metadata (if available)
    manifest = ds / "conversion_manifest.json"
    summary = ds / "conversion_summary.json"
    results["checks"]["conversion_metadata"] = {
        "manifest_exists": manifest.exists(),
        "summary_exists": summary.exists(),
    }
    if summary.exists():
        try:
            with open(summary, "r") as f:
                conv_summary = json.load(f)
            results["checks"]["conversion_metadata"]["summary"] = conv_summary
        except Exception:
            results["checks"]["conversion_metadata"]["summary"] = None

    return results


def write_results(results, output_path):
    """Write results to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[check_dataset_quality] Results written to: {output_path}")


def print_summary(results):
    """Print human-readable summary."""
    print("=" * 60)
    print("  CARLA Perception Lab — Dataset Quality Check")
    print("=" * 60)
    print(f"  Experiment ID : {results['experiment_id']}")
    print(f"  Dataset path  : {results['dataset_path']}")
    print()

    checks = results.get("checks", {})

    # Sample counts
    sc = checks.get("sample_counts", {})
    print(f"  Label files      : {sc.get('label_files', 'N/A')}")
    print(f"  Point cloud files: {sc.get('point_cloud_files', 'N/A')}")

    # Splits
    img = checks.get("imagesets", {})
    ts = img.get("train_split", {})
    vs = img.get("val_split", {})
    print(f"  Train split      : {ts.get('count', 'N/A')} samples")
    print(f"  Val split        : {vs.get('count', 'N/A')} samples")
    print()

    # Class distribution
    cd = checks.get("class_distribution")
    if cd:
        print("  Class distribution:")
        for cls, count in cd.get("classes", {}).items():
            print(f"    {cls:15s}: {count}")
        print(f"    Total annotations: {cd.get('total_annotations', 0)}")
    print()

    # Point stats
    ps = checks.get("point_stats")
    if ps and "error" not in ps:
        print("  Point cloud stats (sampled):")
        print(f"    Files checked : {ps.get('num_files_checked', 'N/A')}")
        print(f"    Min points    : {ps.get('min_points', 'N/A')}")
        print(f"    Max points    : {ps.get('max_points', 'N/A')}")
        print(f"    Mean points   : {ps.get('mean_points', 'N/A'):.1f}")
    print()

    if results["overall_pass"]:
        print("  ✓ OVERALL: PASS")
    else:
        print("  ✗ OVERALL: FAIL")
    for f in results.get("failures", []):
        print(f"    ✗ {f}")
    for w in results.get("warnings", []):
        print(f"    ⚠ {w}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Dataset quality check for CARLA Perception Lab experiment harness."
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        required=True,
        help="Experiment identifier.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to PCDet-format dataset directory.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path. Default: outputs/benchmarks/dataset_quality_<experiment_id>.json",
    )
    args = parser.parse_args()

    # Resolve dataset path relative to project root
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path

    output_path = args.output or str(
        PROJECT_ROOT / "outputs" / "benchmarks" / f"dataset_quality_{args.experiment_id}.json"
    )

    results = run_dataset_quality_check(str(dataset_path), args.experiment_id)
    print_summary(results)
    write_results(results, output_path)

    if not results["overall_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
