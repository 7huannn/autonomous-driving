#!/usr/bin/env python3
"""
Best Checkpoint Selection for CARLA Perception Lab.

Reads a checkpoint sweep JSON, selects the best checkpoint by mAP
(tie-break: Pedestrian AP, then Vehicle AP), detects collapse, and
compares against the fixed baseline.

Usage:
    python evals/select_best_checkpoint.py \
        --experiment-id EXP001 \
        --sweep outputs/benchmarks/checkpoint_sweep_EXP001.json
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
        print(f"[select_best] ERROR: {p} not found"); sys.exit(1)
    with open(p) as f:
        return yaml.safe_load(f)

def load_sweep(path):
    p = Path(path)
    if not p.exists():
        print(f"[select_best] ERROR: Sweep file not found: {path}")
        print("Run sweep_checkpoints.py first."); sys.exit(1)
    with open(p) as f:
        return json.load(f)

def select_best(sweep, cfg):
    bl = cfg["baseline"]; cd = cfg["collapse_detection"]; tgt = cfg["targets"]
    ok = [e for e in sweep.get("epochs", []) if e.get("success") and e.get("mAP") is not None]
    r = dict(experiment_id=sweep.get("experiment_id","?"), timestamp=datetime.now(timezone.utc).isoformat(),
        sweep_file=None, total_epochs_evaluated=len(sweep.get("epochs",[])), successful_epochs=len(ok),
        best_epoch=None, best_mAP=None, Vehicle_AP=None, Pedestrian_AP=None, best_result_file=None,
        baseline_mAP=bl["mAP"], baseline_Vehicle_AP=bl["Vehicle_AP"], baseline_Pedestrian_AP=bl["Pedestrian_AP"],
        target_mAP=tgt["target_mAP"], delta_mAP=None, relative_delta_mAP=None,
        target_reached=False, improves_baseline=False, collapse_detected=False, collapse_reason=None, status="no_data")
    if not ok:
        r["status"]="no_successful_evals"; r["collapse_detected"]=True
        r["collapse_reason"]="No successful evaluations"; return r
    ok.sort(key=lambda x:(x.get("mAP",-1), x.get("Pedestrian_AP",-1) or -1, x.get("Vehicle_AP",-1) or -1), reverse=True)
    b = ok[0]; bm = b["mAP"]; bv = b.get("Vehicle_AP"); bp = b.get("Pedestrian_AP")
    r.update(best_epoch=b.get("epoch"), best_mAP=bm, Vehicle_AP=bv, Pedestrian_AP=bp, best_result_file=b.get("result_file"))
    r["delta_mAP"] = bm - bl["mAP"]
    r["relative_delta_mAP"] = (bm - bl["mAP"])/bl["mAP"] if bl["mAP"]>0 else None
    r["target_reached"] = bm >= tgt["target_mAP"]
    r["improves_baseline"] = bm >= bl["mAP"] + tgt["min_delta_over_baseline"]
    ct = bl["mAP"] * cd["fail_if_map_below_fraction_of_baseline"]
    reasons = []
    if bm < ct:
        reasons.append(f"mAP {bm:.6f} < {cd['fail_if_map_below_fraction_of_baseline']*100:.0f}% of baseline ({ct:.6f})")
    mc = cd["fail_if_any_class_ap_below"]
    if bv is not None and bv < mc: reasons.append(f"Vehicle AP {bv:.6f} < {mc}")
    if bp is not None and bp < mc: reasons.append(f"Pedestrian AP {bp:.6f} < {mc}")
    if reasons:
        r["collapse_detected"]=True; r["collapse_reason"]="; ".join(reasons)
    if r["collapse_detected"]: r["status"]="collapsed"
    elif r["target_reached"]: r["status"]="target_reached"
    elif r["improves_baseline"]: r["status"]="improved"
    else: r["status"]="no_improvement"
    return r

def print_summary(r):
    print("="*60); print("  CARLA Perception Lab — Best Checkpoint Selection"); print("="*60)
    print(f"  Experiment       : {r['experiment_id']}")
    print(f"  Epochs evaluated : {r['total_epochs_evaluated']}  (successful: {r['successful_epochs']})")
    if r["best_epoch"] is not None:
        print(f"  Best epoch       : {r['best_epoch']}")
        print(f"  Best mAP         : {r['best_mAP']:.6f}")
        print(f"  Vehicle AP       : {r['Vehicle_AP']}"); print(f"  Pedestrian AP    : {r['Pedestrian_AP']}")
        print(f"  Baseline mAP     : {r['baseline_mAP']:.4f}")
        print(f"  Delta mAP        : {r['delta_mAP']:+.6f}")
        print(f"  Target >= 0.60   : {'✓' if r['target_reached'] else '✗'}")
        print(f"  Improves baseline: {'✓' if r['improves_baseline'] else '✗'}")
        print(f"  Collapse         : {'YES' if r['collapse_detected'] else 'no'}")
        if r['collapse_reason']: print(f"    Reason: {r['collapse_reason']}")
    else:
        print("  No successful evaluations.")
    sym = {"target_reached":"✓✓","improved":"✓","no_improvement":"—","collapsed":"✗✗","no_data":"?","no_successful_evals":"✗"}
    print(f"\n  Status: {sym.get(r['status'],'?')} {r['status'].upper()}"); print("="*60)

def main():
    ap = argparse.ArgumentParser(description="Select best checkpoint from sweep results.")
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--sweep", required=True, help="Path to sweep JSON.")
    ap.add_argument("--output", default=None)
    a = ap.parse_args()
    out = a.output or str(PROJECT_ROOT/"outputs"/"benchmarks"/f"best_checkpoint_{a.experiment_id}.json")
    cfg = load_config(); sw = load_sweep(a.sweep)
    r = select_best(sw, cfg); r["sweep_file"] = a.sweep
    print_summary(r)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out,"w") as f: json.dump(r, f, indent=2)
    print(f"[select_best] Written to: {out}")
    if r["collapse_detected"]: sys.exit(2)
    elif not r["improves_baseline"]: sys.exit(1)

if __name__=="__main__": main()
