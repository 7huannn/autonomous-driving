#!/usr/bin/env python3
"""
Score module for CARLA Perception Lab Experiment Harness.

Provides scoring utilities to parse eval result JSON files and compute
aggregate metrics. Used by gating.py and other harness scripts.
"""
import json
from pathlib import Path


def load_eval_result(path):
    """Load and validate an eval result JSON file."""
    p = Path(path)
    if not p.exists():
        return None, f"File not found: {path}"
    try:
        with open(p) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return None, f"Failed to parse {path}: {e}"
    return data, None


def extract_scores(data):
    """
    Extract standardized scores from an eval result dict.

    Supports both raw eval_results format and best_checkpoint format.

    Returns:
        dict with mAP, Vehicle_AP, Pedestrian_AP, or None on failure.
    """
    if data is None:
        return None

    scores = {}

    # best_checkpoint format
    if "best_mAP" in data:
        scores["mAP"] = data["best_mAP"]
        scores["Vehicle_AP"] = data.get("Vehicle_AP")
        scores["Pedestrian_AP"] = data.get("Pedestrian_AP")
        return scores

    # Raw eval_results format
    if "mAP" in data:
        scores["mAP"] = data["mAP"]
        ca = data.get("class_ap", {})
        scores["Vehicle_AP"] = ca.get("Vehicle", {}).get("ap") if "Vehicle" in ca else data.get("Vehicle_AP")
        scores["Pedestrian_AP"] = ca.get("Pedestrian", {}).get("ap") if "Pedestrian" in ca else data.get("Pedestrian_AP")
        return scores

    return None


def score_improves_baseline(scores, baseline_mAP, min_delta):
    """Check if scores improve over baseline by min_delta."""
    if scores is None or scores.get("mAP") is None:
        return False
    return scores["mAP"] >= baseline_mAP + min_delta


def score_reaches_target(scores, target_mAP, min_vehicle_ap, min_pedestrian_ap):
    """Check if scores reach the final target."""
    if scores is None or scores.get("mAP") is None:
        return False
    if scores["mAP"] < target_mAP:
        return False
    if scores.get("Vehicle_AP") is not None and scores["Vehicle_AP"] < min_vehicle_ap:
        return False
    if scores.get("Pedestrian_AP") is not None and scores["Pedestrian_AP"] < min_pedestrian_ap:
        return False
    return True


def detect_collapse(scores, baseline_mAP, fraction_threshold, min_class_ap):
    """
    Detect if scores indicate a collapsed experiment.

    Returns:
        (collapsed: bool, reasons: list[str])
    """
    if scores is None or scores.get("mAP") is None:
        return True, ["No scores available"]

    reasons = []
    threshold = baseline_mAP * fraction_threshold
    if scores["mAP"] < threshold:
        reasons.append(f"mAP {scores['mAP']:.6f} < {fraction_threshold*100:.0f}% of baseline ({threshold:.6f})")

    if scores.get("Vehicle_AP") is not None and scores["Vehicle_AP"] < min_class_ap:
        reasons.append(f"Vehicle AP {scores['Vehicle_AP']:.6f} < {min_class_ap}")
    if scores.get("Pedestrian_AP") is not None and scores["Pedestrian_AP"] < min_class_ap:
        reasons.append(f"Pedestrian AP {scores['Pedestrian_AP']:.6f} < {min_class_ap}")

    return len(reasons) > 0, reasons
