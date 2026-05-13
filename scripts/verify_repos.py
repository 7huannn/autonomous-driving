#!/usr/bin/env python3
"""
Verify upstream repo structure for CARLA Perception Lab.

Checks that all three upstream repos exist, have the expected key files,
and reports git metadata (branch, commit hash) for reproducibility.

Usage:
    python verify_repos.py [--repos-dir REPOS_DIR]
    python verify_repos.py --help
"""

import argparse
import os
import subprocess
import sys
import json
from pathlib import Path


# Define expected structure per repo
EXPECTED_STRUCTURE = {
    "carla": {
        "description": "CARLA Simulator (UE5)",
        "key_files": [
            "README.md",
            "Docs/start_quickstart.md",
            "Docs/core_sensors.md",
            "Docs/ref_sensors.md",
            "Docs/adv_recorder.md",
            "Docs/build_docker.md",
            "PythonAPI/examples/manual_control.py",
            "PythonAPI/examples/generate_traffic.py",
            "PythonAPI/examples/sensor_synchronization.py",
            "PythonAPI/examples/open3d_lidar.py",
            "PythonAPI/examples/visualize_multiple_sensors.py",
            "PythonAPI/examples/requirements.txt",
        ],
        "key_dirs": [
            "Docs",
            "PythonAPI",
            "PythonAPI/examples",
        ],
    },
    "pytorch-auto-drive": {
        "description": "PytorchAutoDrive — Segmentation + Lane Detection",
        "key_files": [
            "README.md",
            "requirements.txt",
            "main_semseg.py",
            "main_landet.py",
            "docs/INSTALL.md",
            "docs/SEGMENTATION.md",
            "docs/VISUALIZATION.md",
            "docs/BENCHMARK.md",
            "docs/DEPLOY.md",
            "docs/MODEL_ZOO.md",
            "docs/IMAGENET_MODELS.md",
            "configs/semantic_segmentation/erfnet/cityscapes_512x1024.py",
        ],
        "key_dirs": [
            "configs/semantic_segmentation/erfnet",
            "configs/semantic_segmentation/enet",
            "configs/lane_detection",
            "tools/vis",
            "docs",
        ],
    },
    "OpenPCDet": {
        "description": "OpenPCDet — LiDAR 3D Object Detection",
        "key_files": [
            "README.md",
            "requirements.txt",
            "setup.py",
            "docs/INSTALL.md",
            "docs/DEMO.md",
            "docs/GETTING_STARTED.md",
            "docs/CUSTOM_DATASET_TUTORIAL.md",
            "tools/demo.py",
            "tools/train.py",
            "tools/test.py",
            "tools/cfgs/kitti_models/pointpillar.yaml",
            "tools/cfgs/kitti_models/second.yaml",
            "tools/cfgs/kitti_models/pv_rcnn.yaml",
            "tools/cfgs/dataset_configs/custom_dataset.yaml",
        ],
        "key_dirs": [
            "pcdet",
            "tools",
            "tools/cfgs/kitti_models",
            "tools/cfgs/dataset_configs",
            "docker",
        ],
    },
}


def get_git_info(repo_path: Path) -> dict:
    """Get git branch, commit hash, and date for a repo."""
    info = {"branch": "unknown", "commit": "unknown", "date": "unknown"}
    try:
        info["branch"] = subprocess.check_output(
            ["git", "branch", "--show-current"],
            cwd=repo_path, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        pass
    try:
        info["commit"] = subprocess.check_output(
            ["git", "log", "-1", "--format=%H"],
            cwd=repo_path, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        pass
    try:
        info["date"] = subprocess.check_output(
            ["git", "log", "-1", "--format=%ci"],
            cwd=repo_path, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        pass
    return info


def get_dir_size(path: Path) -> str:
    """Get human-readable directory size."""
    try:
        result = subprocess.check_output(
            ["du", "-sh", str(path)],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return result.split("\t")[0]
    except Exception:
        return "unknown"


def verify_repo(repos_dir: Path, repo_name: str, spec: dict) -> dict:
    """Verify a single repo and return results."""
    repo_path = repos_dir / repo_name
    result = {
        "name": repo_name,
        "description": spec["description"],
        "exists": repo_path.is_dir(),
        "path": str(repo_path),
        "git_info": {},
        "size": "unknown",
        "missing_files": [],
        "missing_dirs": [],
        "found_files": [],
        "found_dirs": [],
        "status": "FAIL",
    }

    if not result["exists"]:
        result["missing_files"] = spec["key_files"]
        result["missing_dirs"] = spec["key_dirs"]
        return result

    result["git_info"] = get_git_info(repo_path)
    result["size"] = get_dir_size(repo_path)

    for f in spec["key_files"]:
        if (repo_path / f).is_file():
            result["found_files"].append(f)
        else:
            result["missing_files"].append(f)

    for d in spec["key_dirs"]:
        if (repo_path / d).is_dir():
            result["found_dirs"].append(d)
        else:
            result["missing_dirs"].append(d)

    if not result["missing_files"] and not result["missing_dirs"]:
        result["status"] = "PASS"
    elif len(result["missing_files"]) <= 2:
        result["status"] = "WARN"
    else:
        result["status"] = "FAIL"

    return result


def print_report(results: list[dict]) -> bool:
    """Print verification report. Returns True if all passed."""
    all_passed = True
    print("=" * 70)
    print("CARLA Perception Lab — Repo Audit Verification")
    print("=" * 70)

    for r in results:
        status_icon = {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌"}.get(r["status"], "?")
        print(f"\n{status_icon} {r['name']} — {r['description']}")
        print(f"   Path: {r['path']}")
        print(f"   Exists: {r['exists']}")

        if r["exists"]:
            gi = r["git_info"]
            print(f"   Branch: {gi.get('branch', '?')}")
            print(f"   Commit: {gi.get('commit', '?')[:12]}")
            print(f"   Date:   {gi.get('date', '?')}")
            print(f"   Size:   {r['size']}")
            print(f"   Files:  {len(r['found_files'])}/{len(r['found_files']) + len(r['missing_files'])} found")
            print(f"   Dirs:   {len(r['found_dirs'])}/{len(r['found_dirs']) + len(r['missing_dirs'])} found")

            if r["missing_files"]:
                print(f"   ⚠️  Missing files:")
                for f in r["missing_files"]:
                    print(f"       - {f}")
            if r["missing_dirs"]:
                print(f"   ⚠️  Missing dirs:")
                for d in r["missing_dirs"]:
                    print(f"       - {d}")

        if r["status"] != "PASS":
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL REPOS VERIFIED SUCCESSFULLY")
    else:
        print("⚠️  SOME REPOS HAVE ISSUES — see above")
    print("=" * 70)

    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Verify upstream repos for CARLA Perception Lab.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python verify_repos.py --repos-dir ../repos
    python verify_repos.py --repos-dir ../repos --json
        """,
    )
    parser.add_argument(
        "--repos-dir",
        type=str,
        default="../repos",
        help="Path to the directory containing upstream repos (default: ../repos)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON instead of human-readable report",
    )
    args = parser.parse_args()

    repos_dir = Path(args.repos_dir).resolve()

    if not repos_dir.is_dir():
        print(f"ERROR: repos directory not found: {repos_dir}")
        sys.exit(1)

    results = []
    for repo_name, spec in EXPECTED_STRUCTURE.items():
        results.append(verify_repo(repos_dir, repo_name, spec))

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        all_passed = print_report(results)
        sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
