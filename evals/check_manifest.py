#!/usr/bin/env python3
"""Manifest validation for baseline/provisional eligibility."""

import argparse
import hashlib
import json
import socket
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

BASE_REQUIRED = [
    "manifest_version",
    "experiment_id",
    "result_file",
    "result_sha256",
    "result_type",
    "metric_name",
    "mAP",
    "class_ap",
    "dataset",
    "config",
    "source_files",
    "command",
    "git",
    "runtime",
    "validation",
    "eligibility",
]

ALLOWED_RESULT_TYPES = {"raw_checkpoint_eval", "fusion_rescore", "postprocess_eval"}


def resolve_path(path_str):
    p = Path(path_str)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_class_ap_value(class_ap_obj, cls_name):
    v = class_ap_obj.get(cls_name)
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):
        ap = v.get("ap")
        if isinstance(ap, (int, float)):
            return float(ap)
    return None


def add_error(report, code, msg):
    report["errors"].append({"code": code, "message": msg})


def add_warning(report, code, msg):
    report["warnings"].append({"code": code, "message": msg})


def validate_manifest(manifest, mode):
    report = {
        "experiment_id": manifest.get("experiment_id", "unknown"),
        "mode": mode,
        "errors": [],
        "warnings": [],
        "checks": {},
    }

    # Required top-level fields
    for key in BASE_REQUIRED:
        ok = key in manifest
        report["checks"][f"has_{key}"] = ok
        if not ok:
            add_error(report, "missing_field", f"Missing top-level field: {key}")

    if report["errors"]:
        report["overall_pass"] = False
        return report

    # Basic value checks
    result_type = manifest.get("result_type")
    if result_type not in ALLOWED_RESULT_TYPES:
        add_error(report, "invalid_result_type", f"result_type must be one of {sorted(ALLOWED_RESULT_TYPES)}")

    class_ap = manifest.get("class_ap", {})
    if not isinstance(class_ap, dict):
        add_error(report, "invalid_class_ap", "class_ap must be an object")
    else:
        for cls in ["Vehicle", "Pedestrian"]:
            if cls not in class_ap:
                add_error(report, "missing_class_ap", f"class_ap missing required key: {cls}")

    # Path existence checks
    def check_path_field(path_str, field_name, required=True):
        if path_str is None:
            if required:
                add_error(report, "missing_path", f"{field_name} is null")
            return None
        p = resolve_path(path_str)
        exists = p.exists()
        report["checks"][f"exists_{field_name}"] = exists
        if not exists:
            add_error(report, "missing_file", f"{field_name} does not exist: {p}")
            return None
        return p

    result_path = check_path_field(manifest.get("result_file"), "result_file", required=True)
    dataset = manifest.get("dataset", {})
    config = manifest.get("config", {})
    command = manifest.get("command", {})
    git = manifest.get("git", {})
    runtime = manifest.get("runtime", {})
    validation = manifest.get("validation", {})

    config_path = check_path_field(config.get("path") if isinstance(config, dict) else None, "config.path", required=True)
    split_path = check_path_field(dataset.get("split_file") if isinstance(dataset, dict) else None, "dataset.split_file", required=True)

    source_files = manifest.get("source_files", [])
    if not isinstance(source_files, list) or len(source_files) == 0:
        add_error(report, "invalid_source_files", "source_files must be a non-empty list")
        source_files = []

    src_paths = []
    for i, src in enumerate(source_files):
        if not isinstance(src, dict):
            add_error(report, "invalid_source_entry", f"source_files[{i}] must be object")
            continue
        if "path" not in src or "role" not in src:
            add_error(report, "missing_source_fields", f"source_files[{i}] missing path/role")
            continue
        p = check_path_field(src.get("path"), f"source_files[{i}].path", required=True)
        if p is not None:
            src_paths.append((i, p, src))

    # Validation file references exist if provided
    for vf in ["env_check_file", "dataset_quality_file", "compare_file", "gate_file"]:
        val = validation.get(vf) if isinstance(validation, dict) else None
        if val not in [None, "", "unknown"]:
            check_path_field(val, f"validation.{vf}", required=True)

    # Hash checks
    if result_path is not None:
        actual = sha256_file(result_path)
        expected = manifest.get("result_sha256")
        report["checks"]["result_sha256_match"] = (expected == actual)
        if expected != actual:
            add_error(report, "hash_mismatch", f"result_sha256 mismatch for {result_path}")
        report["checks"]["result_sha256_actual"] = actual

    if config_path is not None:
        actual = sha256_file(config_path)
        expected = config.get("sha256") if isinstance(config, dict) else None
        report["checks"]["config_sha256_match"] = (expected == actual)
        if expected != actual:
            add_error(report, "hash_mismatch", f"config.sha256 mismatch for {config_path}")
        report["checks"]["config_sha256_actual"] = actual

    if split_path is not None:
        actual = sha256_file(split_path)
        expected = dataset.get("split_sha256") if isinstance(dataset, dict) else None
        report["checks"]["split_sha256_match"] = (expected == actual)
        if expected != actual:
            add_error(report, "hash_mismatch", f"dataset.split_sha256 mismatch for {split_path}")
        report["checks"]["split_sha256_actual"] = actual

    for i, p, src in src_paths:
        actual = sha256_file(p)
        expected = src.get("sha256")
        report["checks"][f"source_{i}_sha256_match"] = (expected == actual)
        if expected != actual:
            add_error(report, "hash_mismatch", f"source_files[{i}].sha256 mismatch for {p}")

    # Metric consistency check against result json
    if result_path is not None:
        try:
            result_data = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception as e:
            add_error(report, "result_parse_error", f"Could not parse result_file JSON: {e}")
            result_data = None

        if isinstance(result_data, dict):
            result_map = result_data.get("mAP", result_data.get("best_mAP"))
            manifest_map = manifest.get("mAP")
            report["checks"]["mAP_match"] = (result_map == manifest_map)
            if result_map != manifest_map:
                add_error(report, "metric_mismatch", f"manifest mAP ({manifest_map}) != result mAP ({result_map})")

            result_class_ap = result_data.get("class_ap", {})
            for cls in ["Vehicle", "Pedestrian"]:
                m_val = get_class_ap_value(class_ap, cls)
                r_val = get_class_ap_value(result_class_ap, cls)
                report["checks"][f"class_ap_{cls}_match"] = (m_val == r_val)
                if m_val != r_val:
                    add_error(report, "metric_mismatch", f"manifest class_ap[{cls}] ({m_val}) != result class_ap[{cls}] ({r_val})")

    # Baseline-grade strict checks
    if mode == "baseline-grade":
        # eligibility must explicitly claim baseline-grade
        elig = manifest.get("eligibility", {})
        if not isinstance(elig, dict) or elig.get("baseline_grade") is not True:
            add_error(report, "baseline_grade_required", "eligibility.baseline_grade must be true for baseline-grade mode")

        commit = git.get("commit") if isinstance(git, dict) else None
        status_short = git.get("status_short") if isinstance(git, dict) else None
        if commit in [None, "", "unknown"]:
            add_error(report, "missing_git_commit", "git.commit is required in baseline-grade mode")
        if status_short in [None, "", "unknown"]:
            add_error(report, "missing_git_status", "git.status_short must be explicitly recorded in baseline-grade mode")

        executable = command.get("executable") if isinstance(command, dict) else None
        args = command.get("args") if isinstance(command, dict) else None
        cwd = command.get("cwd") if isinstance(command, dict) else None
        conda_env = command.get("conda_env") if isinstance(command, dict) else None
        if executable in [None, "", "unknown"]:
            add_error(report, "missing_command", "command.executable is required in baseline-grade mode")
        if not isinstance(args, list) or len(args) == 0:
            add_error(report, "missing_command_args", "command.args must be a non-empty list in baseline-grade mode")
        if cwd in [None, "", "unknown"]:
            add_error(report, "missing_cwd", "command.cwd is required in baseline-grade mode")
        if conda_env in [None, "", "unknown"]:
            add_error(report, "missing_conda_env", "command.conda_env is required in baseline-grade mode")

        split_sha = dataset.get("split_sha256") if isinstance(dataset, dict) else None
        if split_sha in [None, "", "unknown"]:
            add_error(report, "missing_split_hash", "dataset.split_sha256 is required in baseline-grade mode")

        for i, _, src in src_paths:
            ssha = src.get("sha256")
            if ssha in [None, "", "unknown"]:
                add_error(report, "missing_source_hash", f"source_files[{i}].sha256 is required in baseline-grade mode")

        if result_type in {"fusion_rescore", "postprocess_eval"}:
            rparams = manifest.get("rescore_params")
            if not isinstance(rparams, dict) or len(rparams) == 0:
                add_error(report, "missing_rescore_params", "rescore_params must be provided for fusion_rescore/postprocess_eval")
            else:
                # forbid unknown placeholders in strict mode
                rtxt = json.dumps(rparams).lower()
                if "unknown" in rtxt or "missing" in rtxt:
                    add_error(report, "unknown_rescore_params", "rescore_params contains unknown/missing placeholders")

    # Runtime sanity warnings
    host = runtime.get("hostname") if isinstance(runtime, dict) else None
    if host in [None, "", "unknown"]:
        add_warning(report, "runtime_hostname_missing", "runtime.hostname is not set")

    report["overall_pass"] = len(report["errors"]) == 0
    return report


def main():
    ap = argparse.ArgumentParser(description="Validate experiment manifest file.")
    ap.add_argument("--manifest", required=True, help="Path to manifest JSON.")
    ap.add_argument("--mode", required=True, choices=["baseline-grade", "provisional"])
    args = ap.parse_args()

    manifest_path = resolve_path(args.manifest)
    if not manifest_path.exists():
        print(f"[check_manifest] ERROR: manifest not found: {manifest_path}")
        sys.exit(1)

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[check_manifest] ERROR: failed to parse manifest JSON: {e}")
        sys.exit(1)

    report = validate_manifest(manifest, args.mode)
    exp_id = manifest.get("experiment_id", "unknown")
    out_path = PROJECT_ROOT / "outputs" / "benchmarks" / f"manifest_check_{exp_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 60)
    print("  CARLA Perception Lab — Manifest Check")
    print("=" * 60)
    print(f"  Experiment : {exp_id}")
    print(f"  Mode       : {args.mode}")
    print(f"  Manifest   : {manifest_path}")
    print(f"  Report     : {out_path}")
    print(f"  Errors     : {len(report['errors'])}")
    print(f"  Warnings   : {len(report['warnings'])}")
    if report["overall_pass"]:
        print("\n  PASS")
    else:
        print("\n  FAIL")
        for e in report["errors"][:20]:
            print(f"    - {e['code']}: {e['message']}")
        if len(report["errors"]) > 20:
            print(f"    - ... and {len(report['errors']) - 20} more")
    print("=" * 60)

    sys.exit(0 if report["overall_pass"] else 1)


if __name__ == "__main__":
    main()
