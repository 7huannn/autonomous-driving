#!/usr/bin/env python3
"""
Environment Guard for CARLA Perception Lab Experiment Harness.

Verifies that the current Python environment matches the expected conda
environment and that all required packages for OpenPCDet evaluation are
importable. Writes results to outputs/benchmarks/env_check.json.

Usage:
    python evals/check_env.py --expected-env pcdet
"""

import argparse
import json
import os
import sys
import importlib
from pathlib import Path
from datetime import datetime, timezone


# Resolve project root relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def get_conda_env_name():
    """Detect the current conda environment name, if any."""
    return os.environ.get("CONDA_DEFAULT_ENV", None)


def check_package(name):
    """Try to import a package and return (ok, version_or_error)."""
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "unknown")
        return True, version
    except ImportError as e:
        return False, str(e)


def run_env_check(expected_env=None):
    """
    Run all environment checks.

    Returns:
        dict: Full check results including pass/fail status.
    """
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "conda_env_detected": get_conda_env_name(),
        "expected_conda_env": expected_env,
        "conda_env_match": None,
        "packages": {},
        "pcdet_import_ok": False,
        "overall_pass": True,
        "failures": [],
    }

    # --- Conda environment check ---
    detected_env = results["conda_env_detected"]
    if expected_env:
        if detected_env is None:
            results["conda_env_match"] = False
            results["failures"].append(
                f"Expected conda env '{expected_env}' but no conda env is active. "
                f"Python executable: {sys.executable}"
            )
        elif detected_env != expected_env:
            results["conda_env_match"] = False
            results["failures"].append(
                f"Expected conda env '{expected_env}' but found '{detected_env}'. "
                f"Python executable: {sys.executable}"
            )
        else:
            results["conda_env_match"] = True
    else:
        results["conda_env_match"] = None  # No expectation set

    # --- Required packages ---
    required_packages = ["easydict", "numpy", "torch"]
    for pkg_name in required_packages:
        ok, version_or_err = check_package(pkg_name)
        results["packages"][pkg_name] = {
            "importable": ok,
            "version": version_or_err if ok else None,
            "error": version_or_err if not ok else None,
        }
        if not ok:
            results["failures"].append(
                f"Required package '{pkg_name}' not importable: {version_or_err}"
            )

    # --- Optional: pcdet ---
    pcdet_ok, pcdet_ver = check_package("pcdet")
    results["pcdet_import_ok"] = pcdet_ok
    results["packages"]["pcdet"] = {
        "importable": pcdet_ok,
        "version": pcdet_ver if pcdet_ok else None,
        "error": pcdet_ver if not pcdet_ok else None,
    }
    if not pcdet_ok:
        # pcdet is optional but we report it
        results["failures"].append(
            f"Optional package 'pcdet' not importable: {pcdet_ver}. "
            f"This is required for checkpoint evaluation."
        )

    # --- Numpy version (special attention) ---
    np_info = results["packages"].get("numpy", {})
    if np_info.get("importable"):
        results["numpy_version"] = np_info["version"]
    else:
        results["numpy_version"] = None

    # --- Torch version ---
    torch_info = results["packages"].get("torch", {})
    if torch_info.get("importable"):
        results["torch_version"] = torch_info["version"]
        # Also check CUDA availability
        try:
            import torch
            results["torch_cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                results["torch_cuda_device"] = torch.cuda.get_device_name(0)
            else:
                results["torch_cuda_device"] = None
        except Exception:
            results["torch_cuda_available"] = False
            results["torch_cuda_device"] = None
    else:
        results["torch_version"] = None
        results["torch_cuda_available"] = False
        results["torch_cuda_device"] = None

    # --- Overall pass/fail ---
    # Fail if conda env doesn't match (when expected) or required packages missing
    if results["conda_env_match"] is False:
        results["overall_pass"] = False
    for pkg_name in required_packages:
        if not results["packages"][pkg_name]["importable"]:
            results["overall_pass"] = False

    return results


def write_results(results, output_path):
    """Write results JSON to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[check_env] Results written to: {output_path}")


def print_summary(results):
    """Print a human-readable summary."""
    print("=" * 60)
    print("  CARLA Perception Lab — Environment Check")
    print("=" * 60)
    print(f"  Python executable : {results['python_executable']}")
    print(f"  Python version    : {results['python_version'].split()[0]}")
    print(f"  Conda env detected: {results['conda_env_detected'] or '(none)'}")
    print(f"  Expected conda env: {results['expected_conda_env'] or '(any)'}")
    print(f"  Conda env match   : {results['conda_env_match']}")
    print()

    print("  Packages:")
    for pkg, info in results["packages"].items():
        status = "✓" if info["importable"] else "✗"
        ver = info["version"] or info["error"]
        print(f"    {status} {pkg:12s} : {ver}")
    print()

    if results.get("torch_cuda_available"):
        print(f"  CUDA available    : Yes ({results.get('torch_cuda_device', 'unknown')})")
    else:
        print(f"  CUDA available    : No")
    print()

    if results["overall_pass"]:
        print("  ✓ OVERALL: PASS")
    else:
        print("  ✗ OVERALL: FAIL")
        for f in results["failures"]:
            print(f"    - {f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Environment guard for CARLA Perception Lab experiment harness."
    )
    parser.add_argument(
        "--expected-env",
        type=str,
        default=None,
        help="Expected conda environment name (e.g., 'pcdet').",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path. Default: outputs/benchmarks/env_check.json",
    )
    args = parser.parse_args()

    output_path = args.output or str(
        PROJECT_ROOT / "outputs" / "benchmarks" / "env_check.json"
    )

    results = run_env_check(expected_env=args.expected_env)
    print_summary(results)
    write_results(results, output_path)

    if not results["overall_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
