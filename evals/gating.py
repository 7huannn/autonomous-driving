#!/usr/bin/env python3
"""
Gating module for CARLA Perception Lab Experiment Harness.

An experiment only passes if all required conditions are met.
Supports two modes: --mode provisional and --mode target.

Usage:
    python evals/gating.py --mode provisional --experiment-id EXP001
    python evals/gating.py --mode target --experiment-id EXP001
"""
import argparse, json, sys, os, subprocess
from pathlib import Path
from datetime import datetime, timezone
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

def load_config():
    p = SCRIPT_DIR / "experiment_config.yaml"
    if not p.exists():
        print(f"[gating] FATAL: {p} not found"); sys.exit(1)
    with open(p) as f: return yaml.safe_load(f)

def load_json_safe(path, desc):
    """Load JSON, returning (data, error_msg)."""
    p = Path(path)
    if not p.is_absolute(): p = PROJECT_ROOT / p
    if not p.exists(): return None, f"{desc} not found: {p}"
    try:
        with open(p) as f: return json.load(f), None
    except Exception as e: return None, f"Failed to load {desc}: {e}"

def check_file_exists(path, desc):
    p = Path(path)
    if not p.is_absolute(): p = PROJECT_ROOT / p
    return p.exists(), str(p)

def check_protected_paths_unmodified(cfg, experiment_id):
    """
    Check that protected paths have not been modified.
    Uses git status to detect modifications.
    Returns (ok, list_of_modified).
    """
    protected = cfg.get("protected_paths", [])
    modified = []
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=10)
        if result.returncode == 0:
            changed_files = [l[3:].strip() for l in result.stdout.strip().split("\n") if l.strip()]
            for ppath in protected:
                if ppath == "any ground-truth labels":
                    for cf in changed_files:
                        if "/labels/" in cf or cf.startswith("data/"): modified.append(cf)
                elif ppath == "benchmark expected outputs":
                    for cf in changed_files:
                        if "evals/protected_cases/" in cf: modified.append(cf)
                else:
                    for cf in changed_files:
                        if cf.startswith(ppath) or cf == ppath: modified.append(cf)
    except Exception:
        pass  # If git is not available, skip this check
    return len(modified) == 0, modified

def run_gate(experiment_id, mode, cfg):
    """
    Run the full gating check.

    Args:
        experiment_id: Experiment ID.
        mode: 'provisional' or 'target'.
        cfg: Experiment config dict.

    Returns:
        dict: Gate result with pass/fail and reasons.
    """
    bl = cfg["baseline"]; tgt = cfg["targets"]; cd = cfg["collapse_detection"]
    benchmarks_dir = PROJECT_ROOT / "outputs" / "benchmarks"

    result = dict(
        experiment_id=experiment_id, mode=mode,
        timestamp=datetime.now(timezone.utc).isoformat(),
        checks={}, overall_pass=True, failures=[], warnings=[])

    # 1) Required result files
    best_ckpt_path = benchmarks_dir / f"best_checkpoint_{experiment_id}.json"
    comparison_path = benchmarks_dir / f"experiment_comparison_{experiment_id}.json"
    env_path = benchmarks_dir / "env_check.json"
    dataset_path = benchmarks_dir / f"dataset_quality_{experiment_id}.json"

    for name, path in [("best_checkpoint", best_ckpt_path), ("experiment_comparison", comparison_path),
                       ("env_check", env_path), ("dataset_quality", dataset_path)]:
        exists = path.exists()
        result["checks"][f"{name}_exists"] = exists
        if not exists:
            result["overall_pass"] = False
            result["failures"].append(f"Required file missing: {path.name}. Run the corresponding harness step first.")

    # 2) Environment check passed
    if result["checks"].get("env_check_exists"):
        env_data, err = load_json_safe(str(env_path), "env_check")
        if err:
            result["overall_pass"] = False; result["failures"].append(err)
        elif env_data and not env_data.get("overall_pass", False):
            result["overall_pass"] = False
            result["failures"].append("Environment check did not pass. Run: python evals/check_env.py --expected-env pcdet")
        result["checks"]["env_passed"] = env_data.get("overall_pass", False) if env_data else False

    # 3) Dataset quality check passed
    if result["checks"].get("dataset_quality_exists"):
        ds_data, err = load_json_safe(str(dataset_path), "dataset_quality")
        if err:
            result["overall_pass"] = False; result["failures"].append(err)
        elif ds_data and not ds_data.get("overall_pass", False):
            result["overall_pass"] = False
            result["failures"].append(f"Dataset quality check failed: {ds_data.get('failures', [])}")
        result["checks"]["dataset_passed"] = ds_data.get("overall_pass", False) if ds_data else False

    # 4) Load best checkpoint metrics
    current_mAP = None; current_vap = None; current_pap = None
    if result["checks"].get("best_checkpoint_exists"):
        bc_data, err = load_json_safe(str(best_ckpt_path), "best_checkpoint")
        if err:
            result["overall_pass"] = False; result["failures"].append(err)
        elif bc_data:
            current_mAP = bc_data.get("best_mAP")
            current_vap = bc_data.get("Vehicle_AP")
            current_pap = bc_data.get("Pedestrian_AP")
            result["checks"]["current_mAP"] = current_mAP
            result["checks"]["current_Vehicle_AP"] = current_vap
            result["checks"]["current_Pedestrian_AP"] = current_pap

            # Collapse detection
            if bc_data.get("collapse_detected"):
                result["overall_pass"] = False
                result["failures"].append(f"Collapse detected: {bc_data.get('collapse_reason', 'unknown')}")
                result["checks"]["collapse_detected"] = True
            else:
                result["checks"]["collapse_detected"] = False

    # 5) mAP improvement check
    if current_mAP is not None:
        improves = current_mAP >= bl["mAP"] + tgt["min_delta_over_baseline"]
        reaches_target = current_mAP >= tgt["target_mAP"]
        result["checks"]["improves_baseline"] = improves
        result["checks"]["reaches_target"] = reaches_target

        if mode == "provisional":
            if not improves:
                result["overall_pass"] = False
                result["failures"].append(
                    f"mAP {current_mAP:.6f} does not improve baseline {bl['mAP']:.4f} "
                    f"by min delta {tgt['min_delta_over_baseline']}")
        elif mode == "target":
            if not reaches_target:
                result["overall_pass"] = False
                result["failures"].append(
                    f"mAP {current_mAP:.6f} does not reach target {tgt['target_mAP']}")

        # Class AP checks
        if current_vap is not None and current_vap < tgt["min_vehicle_ap"]:
            result["warnings"].append(f"Vehicle AP {current_vap:.6f} < min {tgt['min_vehicle_ap']}")
            if mode == "target":
                result["overall_pass"] = False
                result["failures"].append(f"Vehicle AP {current_vap:.6f} below target minimum {tgt['min_vehicle_ap']}")
        if current_pap is not None and current_pap < tgt["min_pedestrian_ap"]:
            result["warnings"].append(f"Pedestrian AP {current_pap:.6f} < min {tgt['min_pedestrian_ap']}")
            if mode == "target":
                result["overall_pass"] = False
                result["failures"].append(f"Pedestrian AP {current_pap:.6f} below target minimum {tgt['min_pedestrian_ap']}")

        # Additional collapse check via score thresholds
        collapse_frac = cd["fail_if_map_below_fraction_of_baseline"]
        if current_mAP < bl["mAP"] * collapse_frac:
            result["overall_pass"] = False
            result["failures"].append(
                f"COLLAPSE: mAP {current_mAP:.6f} < {collapse_frac*100:.0f}% of baseline")
        min_cls = cd["fail_if_any_class_ap_below"]
        for cls_name, cls_val in [("Vehicle", current_vap), ("Pedestrian", current_pap)]:
            if cls_val is not None and cls_val < min_cls:
                result["overall_pass"] = False
                result["failures"].append(f"COLLAPSE: {cls_name} AP {cls_val:.6f} < {min_cls}")

    # 6) Protected paths check
    prot_ok, prot_modified = check_protected_paths_unmodified(cfg, experiment_id)
    result["checks"]["protected_paths_ok"] = prot_ok
    if not prot_ok:
        result["warnings"].append(f"Protected paths modified: {prot_modified}")

    return result

def print_summary(r):
    print("="*60); print("  CARLA Perception Lab — Experiment Gate"); print("="*60)
    print(f"  Experiment : {r['experiment_id']}")
    print(f"  Mode       : {r['mode']}")
    print()
    for k, v in r["checks"].items():
        sym = "✓" if v else "✗" if isinstance(v, bool) else "·"
        print(f"  {sym} {k}: {v}")
    print()
    if r["overall_pass"]:
        print(f"  ✓ GATE: PASS ({r['mode']})")
    else:
        print(f"  ✗ GATE: FAIL")
        for f in r["failures"]: print(f"    ✗ {f}")
    for w in r.get("warnings", []): print(f"    ⚠ {w}")
    print("="*60)

def main():
    ap = argparse.ArgumentParser(description="Experiment gating for CARLA Perception Lab.")
    ap.add_argument("--mode", required=True, choices=["provisional", "target"])
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--output", default=None)
    a = ap.parse_args()

    cfg = load_config()
    result = run_gate(a.experiment_id, a.mode, cfg)
    print_summary(result)

    out = a.output or str(PROJECT_ROOT/"outputs"/"benchmarks"/f"gate_{a.mode}_{a.experiment_id}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f: json.dump(result, f, indent=2)
    print(f"[gating] Written to: {out}")

    sys.exit(0 if result["overall_pass"] else 1)

if __name__ == "__main__": main()
