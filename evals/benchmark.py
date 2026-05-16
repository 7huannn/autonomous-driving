#!/usr/bin/env python3
"""
Benchmark utilities for CARLA Perception Lab Experiment Harness.

Provides functions to load and aggregate benchmark results from the
experiment harness outputs. Can be used standalone or imported.
"""
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BENCHMARKS_DIR = PROJECT_ROOT / "outputs" / "benchmarks"


def list_experiments(registry_path=None):
    """List all experiments from the registry."""
    reg = Path(registry_path or SCRIPT_DIR / "experiment_registry.jsonl")
    if not reg.exists():
        return []
    experiments = []
    with open(reg) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    experiments.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return experiments


def get_experiment_summary(experiment_id):
    """Get a summary of an experiment's harness outputs."""
    summary = {"experiment_id": experiment_id, "files": {}}
    patterns = [
        ("env_check", "env_check.json"),
        ("dataset_quality", f"dataset_quality_{experiment_id}.json"),
        ("checkpoint_sweep", f"checkpoint_sweep_{experiment_id}.json"),
        ("best_checkpoint", f"best_checkpoint_{experiment_id}.json"),
        ("comparison", f"experiment_comparison_{experiment_id}.json"),
        ("gate_provisional", f"gate_provisional_{experiment_id}.json"),
        ("gate_target", f"gate_target_{experiment_id}.json"),
    ]
    for name, fname in patterns:
        p = BENCHMARKS_DIR / fname
        summary["files"][name] = {"exists": p.exists(), "path": str(p)}
        if p.exists():
            try:
                with open(p) as f:
                    data = json.load(f)
                if name == "best_checkpoint":
                    summary["best_mAP"] = data.get("best_mAP")
                    summary["status"] = data.get("status")
                elif name.startswith("gate_"):
                    summary[f"{name}_pass"] = data.get("overall_pass")
            except Exception:
                pass
    return summary


def print_registry():
    """Print a formatted table of all registered experiments."""
    exps = list_experiments()
    if not exps:
        print("No experiments registered.")
        return
    print(f"{'ID':<20s} {'mAP':>8s} {'V-AP':>8s} {'P-AP':>8s} {'Status':<15s}")
    print("-" * 65)
    for e in exps:
        print(f"{e.get('experiment_id','?'):<20s} "
              f"{e.get('mAP',0):>8.4f} "
              f"{e.get('Vehicle_AP',0):>8.4f} "
              f"{e.get('Pedestrian_AP',0):>8.4f} "
              f"{e.get('status','?'):<15s}")


if __name__ == "__main__":
    print("=== Registered Experiments ===")
    print_registry()
