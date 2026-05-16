#!/usr/bin/env python3
"""
Experiment Comparison for CARLA Perception Lab.

Compares a current experiment's best result against the global baseline
and optionally a previous experiment. Outputs absolute/relative deltas,
class-level deltas, and pass/fail decision.

Usage:
    python evals/compare_experiments.py \
        --experiment-id EXP001 \
        --current outputs/benchmarks/best_checkpoint_EXP001.json \
        --baseline output/detection_3d_canonical_mix_v3/eval_results_fusion_rescore_best.json
"""
import argparse, json, sys
from pathlib import Path
from datetime import datetime, timezone
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

def load_config():
    p = SCRIPT_DIR / "experiment_config.yaml"
    if not p.exists():
        print(f"ERROR: {p} not found"); sys.exit(1)
    with open(p) as f: return yaml.safe_load(f)

def load_json(path, label):
    p = Path(path)
    if not p.is_absolute(): p = PROJECT_ROOT / p
    if not p.exists():
        print(f"ERROR: {label} not found: {p}")
        print(f"  This file is required for comparison.")
        print(f"  If this is a best_checkpoint file, run select_best_checkpoint.py first.")
        print(f"  If this is a baseline file, check experiment_config.yaml.")
        sys.exit(1)
    with open(p) as f: return json.load(f)

def extract_metrics(data):
    """Extract mAP, Vehicle_AP, Pedestrian_AP from either a best_checkpoint or eval_results JSON."""
    if "best_mAP" in data:
        return {"mAP": data["best_mAP"], "Vehicle_AP": data.get("Vehicle_AP"), "Pedestrian_AP": data.get("Pedestrian_AP")}
    if "mAP" in data:
        vap = data.get("Vehicle_AP")
        pap = data.get("Pedestrian_AP")
        ca = data.get("class_ap", {})
        if vap is None and "Vehicle" in ca: vap = ca["Vehicle"].get("ap")
        if pap is None and "Pedestrian" in ca: pap = ca["Pedestrian"].get("ap")
        return {"mAP": data["mAP"], "Vehicle_AP": vap, "Pedestrian_AP": pap}
    return None

def write_failure_output(experiment_id, output_path, current_metrics, baseline_metrics, status, reason):
    result = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "current": current_metrics,
        "baseline": baseline_metrics,
        "previous": None,
        "deltas": {},
        "pass_fail": "fail",
        "overall_pass": False,
        "status": status,
        "reason": reason,
    }
    out = output_path or str(PROJECT_ROOT / "outputs" / "benchmarks" / f"experiment_comparison_{experiment_id}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print("=" * 60)
    print("  CARLA Perception Lab — Experiment Comparison")
    print("=" * 60)
    print(f"  Experiment : {experiment_id}")
    print("  Decision   : FAIL")
    print(f"    -> {reason}")
    print("=" * 60)
    print(f"[compare] Written to: {out}")
    return result

def compare(current_metrics, baseline_metrics, cfg, experiment_id, previous_metrics=None):
    tgt = cfg["targets"]; cd = cfg["collapse_detection"]; bl = cfg["baseline"]
    r = dict(experiment_id=experiment_id, timestamp=datetime.now(timezone.utc).isoformat(),
        current=current_metrics, baseline=baseline_metrics, previous=previous_metrics,
        deltas={}, pass_fail="fail", reason=[])

    # Absolute and relative deltas
    d_map = current_metrics["mAP"] - baseline_metrics["mAP"]
    r["deltas"]["mAP_absolute"] = d_map
    r["deltas"]["mAP_relative"] = d_map / baseline_metrics["mAP"] if baseline_metrics["mAP"] > 0 else None
    for cls in ["Vehicle_AP", "Pedestrian_AP"]:
        cv = current_metrics.get(cls); bv = baseline_metrics.get(cls)
        if cv is not None and bv is not None:
            r["deltas"][f"{cls}_absolute"] = cv - bv
            r["deltas"][f"{cls}_relative"] = (cv - bv) / bv if bv > 0 else None
    if previous_metrics:
        pd = current_metrics["mAP"] - previous_metrics["mAP"]
        r["deltas"]["mAP_vs_previous_absolute"] = pd
        r["deltas"]["mAP_vs_previous_relative"] = pd / previous_metrics["mAP"] if previous_metrics["mAP"] > 0 else None

    # Check collapse
    ct = bl["mAP"] * cd["fail_if_map_below_fraction_of_baseline"]
    if current_metrics["mAP"] < ct:
        r["reason"].append(f"COLLAPSE: mAP {current_metrics['mAP']:.6f} < {ct:.6f}")
    mc = cd["fail_if_any_class_ap_below"]
    for cls, val in [("Vehicle_AP", current_metrics.get("Vehicle_AP")), ("Pedestrian_AP", current_metrics.get("Pedestrian_AP"))]:
        if val is not None and val < mc:
            r["reason"].append(f"COLLAPSE: {cls} {val:.6f} < {mc}")

    # Check improvement
    improves = current_metrics["mAP"] >= baseline_metrics["mAP"] + tgt["min_delta_over_baseline"]
    reaches_target = current_metrics["mAP"] >= tgt["target_mAP"]
    meets_vehicle = current_metrics.get("Vehicle_AP") is None or current_metrics["Vehicle_AP"] >= tgt["min_vehicle_ap"]
    meets_ped = current_metrics.get("Pedestrian_AP") is None or current_metrics["Pedestrian_AP"] >= tgt["min_pedestrian_ap"]

    if reaches_target and meets_vehicle and meets_ped and not r["reason"]:
        r["pass_fail"] = "target_pass"
        r["reason"].append("Target mAP >= 0.60 reached with class AP thresholds met")
    elif improves and not r["reason"]:
        r["pass_fail"] = "provisional_pass"
        r["reason"].append(f"Improves baseline by {d_map:+.6f} (min delta: {tgt['min_delta_over_baseline']})")
    else:
        r["pass_fail"] = "fail"
        if not r["reason"]:
            r["reason"].append(f"mAP {current_metrics['mAP']:.6f} does not improve baseline {baseline_metrics['mAP']:.4f} by min delta {tgt['min_delta_over_baseline']}")

    return r

def print_summary(r):
    print("="*60); print("  CARLA Perception Lab — Experiment Comparison"); print("="*60)
    print(f"  Experiment : {r['experiment_id']}")
    c = r["current"]; b = r["baseline"]
    print(f"\n  {'Metric':<20s} {'Current':>12s} {'Baseline':>12s} {'Delta':>12s}")
    print(f"  {'-'*56}")
    print(f"  {'mAP':<20s} {c['mAP']:>12.6f} {b['mAP']:>12.6f} {r['deltas']['mAP_absolute']:>+12.6f}")
    for cls in ["Vehicle_AP", "Pedestrian_AP"]:
        cv = c.get(cls); bv = b.get(cls); d = r["deltas"].get(f"{cls}_absolute")
        print(f"  {cls:<20s} {cv if cv is not None else 'N/A':>12} {bv if bv is not None else 'N/A':>12} {d if d is not None else 'N/A':>12}")
    print(f"\n  Decision: {r['pass_fail'].upper()}")
    for reason in r["reason"]: print(f"    → {reason}")
    print("="*60)

def main():
    ap = argparse.ArgumentParser(description="Compare experiment against baseline.")
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--current", required=True, help="Current experiment best checkpoint JSON.")
    ap.add_argument("--baseline", default=None, help="Baseline eval JSON. Default: from experiment_config.yaml")
    ap.add_argument("--previous", default=None, help="Optional previous experiment JSON for comparison.")
    ap.add_argument("--output", default=None)
    a = ap.parse_args()

    cfg = load_config()
    cur_data = load_json(a.current, "current experiment")
    cur_m = extract_metrics(cur_data)
    if cur_m is None:
        write_failure_output(
            experiment_id=a.experiment_id,
            output_path=a.output,
            current_metrics=None,
            baseline_metrics=None,
            status="no_valid_current_metrics",
            reason="current experiment has no valid mAP",
        )
        sys.exit(1)

    bl_path = a.baseline or str(PROJECT_ROOT / cfg["baseline"]["result_file"])
    bl_data = load_json(bl_path, "baseline")
    bl_m = extract_metrics(bl_data)
    if bl_m is None:
        print("ERROR: Cannot extract metrics from baseline file"); sys.exit(1)
    if cur_m.get("mAP") is None:
        write_failure_output(
            experiment_id=a.experiment_id,
            output_path=a.output,
            current_metrics=cur_m,
            baseline_metrics=bl_m,
            status="no_valid_current_metrics",
            reason="current experiment has no valid mAP",
        )
        sys.exit(1)

    prev_m = None
    if a.previous:
        prev_data = load_json(a.previous, "previous experiment")
        prev_m = extract_metrics(prev_data)

    result = compare(cur_m, bl_m, cfg, a.experiment_id, prev_m)
    print_summary(result)

    out = a.output or str(PROJECT_ROOT/"outputs"/"benchmarks"/f"experiment_comparison_{a.experiment_id}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f: json.dump(result, f, indent=2)
    print(f"[compare] Written to: {out}")

    if result["pass_fail"] == "fail": sys.exit(1)

if __name__ == "__main__": main()
