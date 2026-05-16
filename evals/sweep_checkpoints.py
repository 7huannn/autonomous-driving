#!/usr/bin/env python3
"""
Checkpoint Sweep for CARLA Perception Lab Experiment Harness.

Evaluates checkpoints in a given epoch range, parses resulting eval JSON files,
and aggregates mAP / Vehicle_AP / Pedestrian_AP for each checkpoint.

Usage:
    python evals/sweep_checkpoints.py \
        --experiment-id EXP001 \
        --checkpoint-dir repos/OpenPCDet/output/.../ckpt \
        --config configs/carla_lidar_ped_strict20_easy4_v1.yaml \
        --start-epoch 51 \
        --end-epoch 80 \
        --conda-env pcdet

Note: This script can operate in two modes:
  1. --run-eval: Actually run evaluation (requires correct conda env and pcdet)
  2. --scan-existing: Scan for existing eval result files in an output directory
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def resolve_dataset_dir_from_config(config_path):
    """
    Resolve dataset directory from model config DATA_CONFIG.DATA_PATH.
    """
    if not config_path:
        return None
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    if not cfg_path.exists():
        return None
    try:
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
    except Exception:
        return None

    data_cfg = cfg.get("DATA_CONFIG", {}) if isinstance(cfg, dict) else {}
    data_path = data_cfg.get("DATA_PATH")
    if not data_path:
        return None
    ds = Path(data_path)
    if not ds.is_absolute():
        ds = PROJECT_ROOT / ds
    return str(ds.resolve())


def find_existing_eval_results(output_dir, start_epoch, end_epoch):
    """
    Scan for existing eval_results_eN.json files in an output directory.

    Args:
        output_dir: Directory containing eval result JSON files.
        start_epoch: Start epoch (inclusive).
        end_epoch: End epoch (inclusive).

    Returns:
        dict: epoch -> file path mapping for found files.
    """
    found = {}
    output_path = Path(output_dir)
    if not output_path.exists():
        return found

    for epoch in range(start_epoch, end_epoch + 1):
        # Try common naming patterns
        for pattern in [
            f"eval_results_e{epoch}.json",
            f"eval_results_epoch_{epoch}.json",
        ]:
            candidate = output_path / pattern
            if candidate.exists():
                found[epoch] = str(candidate)
                break

    return found


def parse_eval_result(result_path):
    """
    Parse an eval result JSON file and extract metrics.

    Returns:
        dict with mAP, Vehicle_AP, Pedestrian_AP, or error info.
    """
    try:
        with open(result_path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return {
            "success": False,
            "error": f"Failed to parse {result_path}: {e}",
        }

    mAP = data.get("mAP", None)
    class_ap = data.get("class_ap", {})

    vehicle_ap = None
    pedestrian_ap = None
    cyclist_ap = None

    if "Vehicle" in class_ap:
        vehicle_ap = class_ap["Vehicle"].get("ap", None)
    if "Pedestrian" in class_ap:
        pedestrian_ap = class_ap["Pedestrian"].get("ap", None)
    if "Cyclist" in class_ap:
        cyclist_ap = class_ap["Cyclist"].get("ap", None)

    if mAP is None:
        return {
            "success": False,
            "error": f"No 'mAP' field in {result_path}",
        }

    return {
        "success": True,
        "mAP": mAP,
        "Vehicle_AP": vehicle_ap,
        "Pedestrian_AP": pedestrian_ap,
        "Cyclist_AP": cyclist_ap,
        "num_frames_evaluated": data.get("num_frames_evaluated", None),
        "total_gt_boxes": data.get("total_gt_boxes", None),
        "result_file": str(result_path),
    }


def run_eval_command(epoch, checkpoint_dir, config, conda_env, eval_script, output_dir, dataset_dir):
    """
    Run evaluation for a single checkpoint epoch using conda run.

    Returns:
        dict with success status and output/error info.
    """
    # Construct checkpoint path
    ckpt_path = Path(checkpoint_dir) / f"checkpoint_epoch_{epoch}.pth"
    if not ckpt_path.exists():
        return {
            "success": False,
            "error": f"Checkpoint not found: {ckpt_path}",
        }

    # Build eval command
    eval_cmd = eval_script or str(PROJECT_ROOT / "scripts" / "run_lidar_det.py")
    cmd = [
        "conda", "run", "-n", conda_env, "--no-capture-output",
        "python", eval_cmd,
        "--mode", "evaluate",
        "--cfg-file", str(config),
        "--checkpoint", str(ckpt_path),
        "--dataset-dir", str(dataset_dir),
        "--eval-output", str(Path(output_dir) / f"eval_results_e{epoch}.json"),
        "--overwrite",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout per checkpoint
            cwd=str(PROJECT_ROOT),
        )
        return {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout[-500:] if result.stdout else "",
            "stderr": result.stderr[-500:] if result.stderr else "",
            "error": None if result.returncode == 0 else (
                f"Exit code {result.returncode}: "
                f"{(result.stderr or result.stdout or '').strip().splitlines()[-1] if (result.stderr or result.stdout) else 'no stderr'}"
            ),
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"Evaluation timed out for epoch {epoch}",
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to run eval for epoch {epoch}: {e}",
        }


def run_sweep(args):
    """
    Run the full checkpoint sweep.

    Returns:
        dict: Full sweep results.
    """
    results = {
        "experiment_id": args.experiment_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint_dir": args.checkpoint_dir,
        "config": args.config,
        "dataset_dir": args.dataset_dir,
        "start_epoch": args.start_epoch,
        "end_epoch": args.end_epoch,
        "conda_env": args.conda_env,
        "mode": "scan_existing" if args.scan_existing else "run_eval",
        "epochs": [],
        "summary": {
            "total_epochs": 0,
            "successful_evals": 0,
            "failed_evals": 0,
            "best_mAP": None,
            "best_epoch": None,
        },
    }

    if args.scan_existing:
        # Scan for existing result files
        scan_dir = args.scan_existing
        found = find_existing_eval_results(scan_dir, args.start_epoch, args.end_epoch)

        for epoch in range(args.start_epoch, args.end_epoch + 1):
            entry = {"epoch": epoch}
            if epoch in found:
                parsed = parse_eval_result(found[epoch])
                entry.update(parsed)
            else:
                entry["success"] = False
                entry["error"] = f"No eval result file found for epoch {epoch}"
            results["epochs"].append(entry)
    else:
        # Run evaluations
        dataset_dir = args.dataset_dir or resolve_dataset_dir_from_config(args.config)
        if not dataset_dir:
            raise ValueError(
                "Could not resolve dataset dir. Provide --dataset-dir or set DATA_CONFIG.DATA_PATH in config."
            )
        if not Path(dataset_dir).exists():
            raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")
        results["dataset_dir"] = dataset_dir

        output_dir = args.output_dir or str(
            PROJECT_ROOT / "output" / f"detection_3d_{args.experiment_id}"
        )
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        for epoch in range(args.start_epoch, args.end_epoch + 1):
            print(f"[sweep] Evaluating epoch {epoch}...")
            entry = {"epoch": epoch}

            run_result = run_eval_command(
                epoch=epoch,
                checkpoint_dir=args.checkpoint_dir,
                config=args.config,
                conda_env=args.conda_env,
                eval_script=args.eval_script,
                output_dir=output_dir,
                dataset_dir=dataset_dir,
            )

            if run_result["success"]:
                result_file = Path(output_dir) / f"eval_results_e{epoch}.json"
                if result_file.exists():
                    parsed = parse_eval_result(str(result_file))
                    entry.update(parsed)
                else:
                    entry["success"] = False
                    entry["error"] = "Eval ran but result file not found"
            else:
                entry["success"] = False
                entry["error"] = run_result.get("error", "Unknown error")

            results["epochs"].append(entry)

    # Compute summary
    successful = [e for e in results["epochs"] if e.get("success")]
    failed = [e for e in results["epochs"] if not e.get("success")]

    results["summary"]["total_epochs"] = len(results["epochs"])
    results["summary"]["successful_evals"] = len(successful)
    results["summary"]["failed_evals"] = len(failed)

    if successful:
        best = max(successful, key=lambda x: x.get("mAP", -1))
        results["summary"]["best_mAP"] = best.get("mAP")
        results["summary"]["best_epoch"] = best.get("epoch")
        results["summary"]["best_Vehicle_AP"] = best.get("Vehicle_AP")
        results["summary"]["best_Pedestrian_AP"] = best.get("Pedestrian_AP")

    # Record failed epochs explicitly
    results["failed_epochs"] = [
        {"epoch": e["epoch"], "error": e.get("error", "unknown")}
        for e in failed
    ]

    return results


def write_results(results, output_base):
    """Write results to JSON and CSV."""
    output_base = Path(output_base)
    json_path = output_base.with_suffix(".json")
    csv_path = output_base.with_suffix(".csv")

    json_path.parent.mkdir(parents=True, exist_ok=True)

    # Write JSON
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[sweep] JSON results written to: {json_path}")

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "success", "mAP", "Vehicle_AP", "Pedestrian_AP",
            "Cyclist_AP", "num_frames", "total_gt", "error"
        ])
        for e in results["epochs"]:
            writer.writerow([
                e.get("epoch", ""),
                e.get("success", False),
                e.get("mAP", ""),
                e.get("Vehicle_AP", ""),
                e.get("Pedestrian_AP", ""),
                e.get("Cyclist_AP", ""),
                e.get("num_frames_evaluated", ""),
                e.get("total_gt_boxes", ""),
                e.get("error", ""),
            ])
    print(f"[sweep] CSV results written to: {csv_path}")


def print_summary(results):
    """Print human-readable summary."""
    print("=" * 60)
    print("  CARLA Perception Lab — Checkpoint Sweep")
    print("=" * 60)
    print(f"  Experiment ID    : {results['experiment_id']}")
    print(f"  Mode             : {results['mode']}")
    if results.get("dataset_dir"):
        print(f"  Dataset dir      : {results['dataset_dir']}")
    print(f"  Epoch range      : {results['start_epoch']} - {results['end_epoch']}")
    s = results["summary"]
    print(f"  Total epochs     : {s['total_epochs']}")
    print(f"  Successful evals : {s['successful_evals']}")
    print(f"  Failed evals     : {s['failed_evals']}")
    print()
    if s["best_epoch"] is not None:
        print(f"  Best epoch       : {s['best_epoch']}")
        print(f"  Best mAP         : {s['best_mAP']:.6f}")
        print(f"  Vehicle AP       : {s.get('best_Vehicle_AP', 'N/A')}")
        print(f"  Pedestrian AP    : {s.get('best_Pedestrian_AP', 'N/A')}")
    else:
        print("  No successful evaluations found.")

    if results["failed_epochs"]:
        print()
        print("  Failed epochs:")
        for fe in results["failed_epochs"][:10]:
            print(f"    Epoch {fe['epoch']}: {fe['error']}")
        if len(results["failed_epochs"]) > 10:
            print(f"    ... and {len(results['failed_epochs']) - 10} more")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Checkpoint sweep for CARLA Perception Lab experiment harness."
    )
    parser.add_argument("--experiment-id", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Directory containing checkpoint_epoch_N.pth files.")
    parser.add_argument("--config", type=str, default=None,
                        help="Model config YAML file path.")
    parser.add_argument("--dataset-dir", type=str, default=None,
                        help="Dataset root path. Default: from config DATA_CONFIG.DATA_PATH.")
    parser.add_argument("--start-epoch", type=int, required=True)
    parser.add_argument("--end-epoch", type=int, required=True)
    parser.add_argument("--conda-env", type=str, default="pcdet",
                        help="Conda environment to use for evaluation.")
    parser.add_argument("--eval-script", type=str, default=None,
                        help="Evaluation script path. Default: scripts/run_lidar_det.py")
    parser.add_argument("--output", type=str, default=None,
                        help="Output base path (without extension). "
                             "Default: outputs/benchmarks/checkpoint_sweep_<experiment_id>")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to write eval result JSONs (when running evals).")
    parser.add_argument("--scan-existing", type=str, default=None,
                        help="Instead of running evals, scan this directory for existing "
                             "eval_results_eN.json files.")

    args = parser.parse_args()

    output_base = args.output or str(
        PROJECT_ROOT / "outputs" / "benchmarks" / f"checkpoint_sweep_{args.experiment_id}"
    )

    results = run_sweep(args)
    print_summary(results)
    write_results(results, output_base)

    if results["summary"]["successful_evals"] == 0:
        print("\n[sweep] WARNING: No successful evaluations. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
