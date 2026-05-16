#!/usr/bin/env python3
"""
Fusion/Rescore stage for CARLA Perception Lab experiment harness.

Combines class AP from two existing evaluation JSON files:
- Vehicle AP from --vehicle-result
- Pedestrian AP from --pedestrian-result

Supports conservative rescore modes over class AP values without modifying
source detections, labels, splits, or metrics definitions.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

TARGET_CLASSES = ["Vehicle", "Pedestrian"]


def resolve_path(path_str):
    p = Path(path_str)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def load_json_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_class_entry(class_ap, class_name):
    if not isinstance(class_ap, dict):
        return None
    for key, value in class_ap.items():
        if str(key).lower() == class_name.lower():
            return value
    return None


def extract_metrics(data, source_name):
    class_ap = data.get("class_ap", {}) if isinstance(data, dict) else {}

    metrics = {
        "mAP": data.get("mAP", data.get("best_mAP")) if isinstance(data, dict) else None,
        "Vehicle_AP": None,
        "Pedestrian_AP": None,
        "class_ap": class_ap if isinstance(class_ap, dict) else {},
        "num_frames_evaluated": data.get("num_frames_evaluated") if isinstance(data, dict) else None,
        "total_gt_boxes": data.get("total_gt_boxes") if isinstance(data, dict) else None,
    }

    if isinstance(data, dict):
        metrics["Vehicle_AP"] = data.get("Vehicle_AP")
        metrics["Pedestrian_AP"] = data.get("Pedestrian_AP")

    if metrics["Vehicle_AP"] is None:
        vehicle_entry = get_class_entry(metrics["class_ap"], "Vehicle")
        if isinstance(vehicle_entry, dict):
            metrics["Vehicle_AP"] = vehicle_entry.get("ap")

    if metrics["Pedestrian_AP"] is None:
        ped_entry = get_class_entry(metrics["class_ap"], "Pedestrian")
        if isinstance(ped_entry, dict):
            metrics["Pedestrian_AP"] = ped_entry.get("ap")

    missing = []
    if metrics["mAP"] is None:
        missing.append("mAP")
    if metrics["Vehicle_AP"] is None:
        missing.append("Vehicle_AP")
    if metrics["Pedestrian_AP"] is None:
        missing.append("Pedestrian_AP")

    return metrics, missing


def get_git_info():
    info = {"git_commit": None, "git_status_short": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if commit.returncode == 0:
            info["git_commit"] = commit.stdout.strip()
    except Exception:
        pass

    try:
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if status.returncode == 0:
            lines = [ln for ln in status.stdout.splitlines() if ln.strip()]
            info["git_status_short"] = {
                "num_changed": len(lines),
                "sample": lines[:20],
            }
    except Exception:
        pass

    return info


def compute_fused_map(vehicle_ap, pedestrian_ap, rescore_mode, vehicle_weight, pedestrian_weight):
    if rescore_mode == "none":
        return (vehicle_ap + pedestrian_ap) / 2.0

    if rescore_mode == "class_weighted_mean":
        wsum = vehicle_weight + pedestrian_weight
        if wsum <= 0:
            raise ValueError("vehicle_weight + pedestrian_weight must be > 0 for class_weighted_mean")
        return (vehicle_ap * vehicle_weight + pedestrian_ap * pedestrian_weight) / wsum

    raise ValueError(f"Unsupported rescore_mode: {rescore_mode}")


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def build_failure_result(experiment_id, reason, vehicle_result_path, pedestrian_result_path):
    return {
        "experiment_id": experiment_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "error_missing_metrics",
        "overall_pass": False,
        "reason": reason,
        "result_file": None,
        "vehicle_result": str(vehicle_result_path),
        "pedestrian_result": str(pedestrian_result_path),
    }


def parse_args():
    ap = argparse.ArgumentParser(description="Fuse and optionally rescore evaluation results.")
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--vehicle-result", required=True)
    ap.add_argument("--pedestrian-result", required=True)
    ap.add_argument("--output-dir", default="outputs/benchmarks")
    ap.add_argument("--output", default=None)
    ap.add_argument("--manifest-output", default=None)
    ap.add_argument("--rescore-mode", default="none", choices=["none", "class_weighted_mean"])
    ap.add_argument("--vehicle-weight", type=float, default=1.0)
    ap.add_argument("--pedestrian-weight", type=float, default=1.0)
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()

    vehicle_path = resolve_path(args.vehicle_result)
    pedestrian_path = resolve_path(args.pedestrian_result)

    output_dir = resolve_path(args.output_dir)
    output_path = resolve_path(args.output) if args.output else output_dir / f"fusion_rescore_{args.experiment_id}.json"
    manifest_path = resolve_path(args.manifest_output) if args.manifest_output else output_dir / f"fusion_rescore_{args.experiment_id}_manifest.json"

    if not vehicle_path.exists() or not pedestrian_path.exists():
        reason = f"Missing source file(s): vehicle_exists={vehicle_path.exists()}, pedestrian_exists={pedestrian_path.exists()}"
        failure = build_failure_result(args.experiment_id, reason, vehicle_path, pedestrian_path)
        write_json(output_path, failure)
        print(f"[fusion_rescore] FAIL: {reason}")
        print(f"[fusion_rescore] Written failure output: {output_path}")
        sys.exit(1)

    vehicle_data = load_json_file(vehicle_path)
    pedestrian_data = load_json_file(pedestrian_path)

    vehicle_metrics, vehicle_missing = extract_metrics(vehicle_data, "vehicle_result")
    pedestrian_metrics, pedestrian_missing = extract_metrics(pedestrian_data, "pedestrian_result")

    required_errors = []
    if vehicle_metrics.get("Vehicle_AP") is None:
        required_errors.append("vehicle_result missing Vehicle_AP")
    if pedestrian_metrics.get("Pedestrian_AP") is None:
        required_errors.append("pedestrian_result missing Pedestrian_AP")

    if required_errors:
        reason = "; ".join(required_errors)
        failure = build_failure_result(args.experiment_id, reason, vehicle_path, pedestrian_path)
        failure["source_metric_missing"] = {
            "vehicle_missing": vehicle_missing,
            "pedestrian_missing": pedestrian_missing,
        }
        write_json(output_path, failure)
        print(f"[fusion_rescore] FAIL: {reason}")
        print(f"[fusion_rescore] Written failure output: {output_path}")
        sys.exit(1)

    fused_vehicle_ap = float(vehicle_metrics["Vehicle_AP"])
    fused_pedestrian_ap = float(pedestrian_metrics["Pedestrian_AP"])
    fused_map = float(
        compute_fused_map(
            fused_vehicle_ap,
            fused_pedestrian_ap,
            args.rescore_mode,
            args.vehicle_weight,
            args.pedestrian_weight,
        )
    )

    vehicle_class_entry = get_class_entry(vehicle_metrics.get("class_ap", {}), "Vehicle") or {}
    ped_class_entry = get_class_entry(pedestrian_metrics.get("class_ap", {}), "Pedestrian") or {}
    cyclist_entry = get_class_entry(vehicle_metrics.get("class_ap", {}), "Cyclist")
    if cyclist_entry is None:
        cyclist_entry = get_class_entry(pedestrian_metrics.get("class_ap", {}), "Cyclist")

    class_ap = {
        "Vehicle": {
            "ap": fused_vehicle_ap,
            "num_gt": vehicle_class_entry.get("num_gt"),
            "num_pred": vehicle_class_entry.get("num_pred"),
            "precision": vehicle_class_entry.get("precision"),
            "recall": vehicle_class_entry.get("recall"),
            "iou_threshold": vehicle_class_entry.get("iou_threshold"),
        },
        "Pedestrian": {
            "ap": fused_pedestrian_ap,
            "num_gt": ped_class_entry.get("num_gt"),
            "num_pred": ped_class_entry.get("num_pred"),
            "precision": ped_class_entry.get("precision"),
            "recall": ped_class_entry.get("recall"),
            "iou_threshold": ped_class_entry.get("iou_threshold"),
        },
    }
    if isinstance(cyclist_entry, dict):
        class_ap["Cyclist"] = cyclist_entry

    num_frames = vehicle_data.get("num_frames_evaluated")
    if num_frames is None:
        num_frames = pedestrian_data.get("num_frames_evaluated")

    total_gt = vehicle_data.get("total_gt_boxes")
    if total_gt is None:
        total_gt = pedestrian_data.get("total_gt_boxes")

    result = {
        "experiment_id": args.experiment_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "fusion_rescore_complete",
        "overall_pass": True,
        "result_file": str(output_path),
        "method": "vehicle_from_vehicle_result__pedestrian_from_pedestrian_result",
        "rescore_mode": args.rescore_mode,
        "vehicle_weight": args.vehicle_weight,
        "pedestrian_weight": args.pedestrian_weight,
        "min_score": args.min_score,
        "num_frames_evaluated": num_frames,
        "total_gt_boxes": total_gt,
        "metric": vehicle_data.get("metric", pedestrian_data.get("metric", "BEV_AP_custom")),
        "class_ap": class_ap,
        "mAP": fused_map,
        "Vehicle_AP": fused_vehicle_ap,
        "Pedestrian_AP": fused_pedestrian_ap,
        "best_mAP": fused_map,
        "provenance": {
            "vehicle_result": str(vehicle_path),
            "pedestrian_result": str(pedestrian_path),
            "vehicle_source_metrics": vehicle_metrics,
            "pedestrian_source_metrics": pedestrian_metrics,
            "fusion_method": "vehicle_from_vehicle_result__pedestrian_from_pedestrian_result",
            "rescore": {
                "mode": args.rescore_mode,
                "vehicle_weight": args.vehicle_weight,
                "pedestrian_weight": args.pedestrian_weight,
                "min_score": args.min_score,
            },
            "source_metric_missing": {
                "vehicle_missing": vehicle_missing,
                "pedestrian_missing": pedestrian_missing,
            },
        },
    }

    manifest = {
        "experiment_id": args.experiment_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "vehicle_result": str(vehicle_path),
        "pedestrian_result": str(pedestrian_path),
        "vehicle_result_sha256": sha256_file(vehicle_path),
        "pedestrian_result_sha256": sha256_file(pedestrian_path),
        "rescore_mode": args.rescore_mode,
        "vehicle_weight": args.vehicle_weight,
        "pedestrian_weight": args.pedestrian_weight,
        "min_score": args.min_score,
        "output": str(output_path),
        "manifest_output": str(manifest_path),
        "fused_vehicle_ap": fused_vehicle_ap,
        "fused_pedestrian_ap": fused_pedestrian_ap,
        "fused_mAP": fused_map,
        "command_args": vars(args),
    }
    manifest.update(get_git_info())

    write_json(output_path, result)
    write_json(manifest_path, manifest)

    if args.verbose:
        print(json.dumps(result, indent=2))
        print(json.dumps(manifest, indent=2))

    print("[fusion_rescore] Completed")
    print(f"[fusion_rescore] Output   : {output_path}")
    print(f"[fusion_rescore] Manifest : {manifest_path}")
    print(f"[fusion_rescore] mAP      : {fused_map:.6f}")
    print(f"[fusion_rescore] Vehicle  : {fused_vehicle_ap:.6f}")
    print(f"[fusion_rescore] Pedestrian: {fused_pedestrian_ap:.6f}")


if __name__ == "__main__":
    main()
