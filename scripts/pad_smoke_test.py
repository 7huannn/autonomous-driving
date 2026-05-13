#!/usr/bin/env python3
"""Stage 04 PAD smoke test runner.

Run ERFNet semantic segmentation smoke test on Cityscapes sample images,
with optional downloads and profiling.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_CKPT_URL = "https://drive.google.com/file/d/1uzBSboKD-Xt0K6VHd2aF561Cy13q9xRe/view?usp=sharing"
DEFAULT_ENCODER_URL = "https://github.com/Eromera/erfnet_pytorch/raw/master/trained_models/erfnet_encoder_pretrained.pth.tar"
DEFAULT_TEST_IMAGES_URL = "https://drive.google.com/file/d/1XQvBS1uoHeIgUv7oDQ4Vp1tWYi0oAGhU/view"


def parse_resolutions(text: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for token in text.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if "x" not in token:
            raise argparse.ArgumentTypeError(f"invalid resolution token: {token}")
        h_str, w_str = token.split("x", 1)
        try:
            h = int(h_str)
            w = int(w_str)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid resolution token: {token}") from exc
        out.append((h, w))
    if not out:
        raise argparse.ArgumentTypeError("at least one resolution is required")
    return out


def run_cmd(cmd: List[str], cwd: Path, env: Dict[str, str] | None = None, dry_run: bool = False) -> subprocess.CompletedProcess:
    cmd_str = " ".join(cmd)
    print(f"[CMD] (cwd={cwd}) {cmd_str}")
    if dry_run:
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )


def read_gpu_mem_mib() -> int | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        ).strip().splitlines()
        return int(out[0])
    except Exception:
        return None


def run_monitored_cmd(cmd: List[str], cwd: Path, env: Dict[str, str] | None = None, dry_run: bool = False) -> Tuple[subprocess.CompletedProcess, Dict[str, float | int | None]]:
    cmd_str = " ".join(cmd)
    print(f"[CMD] (cwd={cwd}) {cmd_str}")
    if dry_run:
        return (
            subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr=""),
            {
                "elapsed_seconds": 0.0,
                "baseline_mem_mib": None,
                "peak_mem_mib": None,
                "delta_mem_mib": None,
            },
        )

    baseline = read_gpu_mem_mib()
    peak = baseline
    start = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    while proc.poll() is None:
        mem = read_gpu_mem_mib()
        if mem is not None:
            if peak is None:
                peak = mem
            else:
                peak = max(peak, mem)
        time.sleep(0.2)

    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=stdout, stderr=stderr)

    elapsed = round(time.time() - start, 3)
    delta = None
    if baseline is not None and peak is not None:
        delta = peak - baseline

    return (
        subprocess.CompletedProcess(args=cmd, returncode=proc.returncode, stdout=stdout, stderr=stderr),
        {
            "elapsed_seconds": elapsed,
            "baseline_mem_mib": baseline,
            "peak_mem_mib": peak,
            "delta_mem_mib": delta,
        },
    )


def download_assets(args: argparse.Namespace, dry_run: bool) -> None:
    if args.skip_download:
        return

    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    args.encoder_path.parent.mkdir(parents=True, exist_ok=True)
    args.test_images_zip.parent.mkdir(parents=True, exist_ok=True)

    if not args.checkpoint_path.exists():
        run_cmd([
            sys.executable,
            "-m",
            "gdown",
            "--fuzzy",
            args.checkpoint_url,
            "-O",
            str(args.checkpoint_path),
        ], cwd=args.repo_root, dry_run=dry_run)

    if not args.encoder_path.exists():
        run_cmd([
            "wget",
            "-O",
            str(args.encoder_path),
            args.encoder_url,
        ], cwd=args.repo_root, dry_run=dry_run)

    if not args.test_images_zip.exists():
        run_cmd([
            sys.executable,
            "-m",
            "gdown",
            "--fuzzy",
            args.test_images_url,
            "-O",
            str(args.test_images_zip),
        ], cwd=args.repo_root, dry_run=dry_run)

    if not args.test_images_dir.exists():
        args.test_images_dir.mkdir(parents=True, exist_ok=True)
        run_cmd([
            "unzip",
            "-o",
            str(args.test_images_zip),
            "-d",
            str(args.test_images_dir),
        ], cwd=args.repo_root, dry_run=dry_run)


def run_inference(args: argparse.Namespace, dry_run: bool) -> Dict[str, object]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.test_images_dir / "seg_test_images" / "munster"
    cfg_options = f"model.pretrained_weights='{args.encoder_path}'"

    cmd = [
        sys.executable,
        "tools/vis/seg_img_dir.py",
        "--image-path",
        str(image_dir),
        "--image-suffix",
        args.image_suffix,
        "--save-path",
        str(args.output_dir),
        "--pred",
        "--config",
        str(args.pad_config),
        "--checkpoint",
        str(args.checkpoint_path),
        "--cfg-options",
        cfg_options,
    ]
    if args.mixed_precision:
        cmd.append("--mixed-precision")

    proc, mem_info = run_monitored_cmd(cmd, cwd=args.pad_repo, dry_run=dry_run)

    outputs = sorted(args.output_dir.glob("*.png"))
    if not dry_run:
        if not outputs:
            raise RuntimeError(f"no output PNG generated in {args.output_dir}")
        args.doc_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(outputs[0], args.doc_image)

    return {
        "stdout_tail": proc.stdout.splitlines()[-20:],
        "stderr_tail": proc.stderr.splitlines()[-20:],
        "num_output_images": len(outputs),
        "doc_image": str(args.doc_image),
        "metrics": mem_info,
    }


def extract_profile_stats(text: str) -> Dict[str, float | None]:
    fps = None
    flops = None
    params = None
    for line in text.splitlines():
        if "GPU FPS:" in line:
            try:
                fps = float(line.split(":")[-1].strip())
            except ValueError:
                pass
        if "FLOPs(G):" in line:
            try:
                flops = float(line.split(":")[-1].strip())
            except ValueError:
                pass
        if "Number of parameters:" in line:
            try:
                params = float(line.split(":")[-1].strip())
            except ValueError:
                pass
    return {
        "gpu_fps": fps,
        "flops_g": flops,
        "params_m": params,
    }


def run_profiling(args: argparse.Namespace, dry_run: bool) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    cfg_options = f"model.pretrained_weights='{args.encoder_path}'"

    for h, w in args.profile_resolutions:
        cmd = [
            sys.executable,
            "tools/profiling.py",
            "--mode=simple",
            "--config",
            str(args.pad_config),
            "--times",
            str(args.profile_times),
            "--height",
            str(h),
            "--width",
            str(w),
            "--cfg-options",
            cfg_options,
        ]

        proc, mem_info = run_monitored_cmd(cmd, cwd=args.pad_repo, dry_run=dry_run)
        stats = extract_profile_stats(proc.stdout + "\n" + proc.stderr)
        results.append(
            {
                "resolution": f"{h}x{w}",
                "profile_times": args.profile_times,
                "metrics": {**stats, **mem_info},
            }
        )

    return results


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Stage 04 PAD smoke test runner (ERFNet Cityscapes)"
    )
    parser.add_argument("--pad-repo", type=Path, default=(repo_root / "../repos/pytorch-auto-drive").resolve())
    parser.add_argument("--checkpoint-url", default=DEFAULT_CKPT_URL)
    parser.add_argument("--checkpoint-path", type=Path, default=repo_root / "data/checkpoints/erfnet_cityscapes_512x1024_20200918.pt")
    parser.add_argument("--encoder-url", default=DEFAULT_ENCODER_URL)
    parser.add_argument("--encoder-path", type=Path, default=repo_root / "data/checkpoints/erfnet_encoder_pretrained.pth.tar")
    parser.add_argument("--test-images-url", default=DEFAULT_TEST_IMAGES_URL)
    parser.add_argument("--test-images-zip", type=Path, default=repo_root / "data/processed/PAD_test_images.zip")
    parser.add_argument("--test-images-dir", type=Path, default=repo_root / "data/processed/pad_test_images")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "output/pad_smoke_test")
    parser.add_argument("--doc-image", type=Path, default=repo_root / "docs/pad_smoke_test.png")
    parser.add_argument("--pad-config", type=Path, default=Path("configs/semantic_segmentation/erfnet/cityscapes_512x1024.py"))
    parser.add_argument("--image-suffix", default="_leftImg8bit.png")
    parser.add_argument("--profile-resolutions", type=parse_resolutions, default=parse_resolutions("512x1024,720x1280"))
    parser.add_argument("--profile-times", type=int, default=3)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-profiling", action="store_true")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--summary-json", type=Path, default=repo_root / "output/pad_smoke_test/stage04_summary.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.repo_root = repo_root
    args.pad_repo = args.pad_repo.resolve()
    args.checkpoint_path = args.checkpoint_path.resolve()
    args.encoder_path = args.encoder_path.resolve()
    args.test_images_zip = args.test_images_zip.resolve()
    args.test_images_dir = args.test_images_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.doc_image = args.doc_image.resolve()
    args.summary_json = args.summary_json.resolve()

    if not args.pad_repo.exists():
        raise FileNotFoundError(f"PAD repo not found: {args.pad_repo}")

    return args


def main() -> int:
    args = parse_args()

    summary: Dict[str, object] = {
        "stage": "STAGE_04_PAD_SMOKE_TEST",
        "pad_repo": str(args.pad_repo),
        "checkpoint_path": str(args.checkpoint_path),
        "encoder_path": str(args.encoder_path),
        "test_images_dir": str(args.test_images_dir),
        "output_dir": str(args.output_dir),
        "doc_image": str(args.doc_image),
        "mixed_precision": bool(args.mixed_precision),
        "dry_run": bool(args.dry_run),
    }

    download_assets(args, dry_run=args.dry_run)
    summary["inference"] = run_inference(args, dry_run=args.dry_run)

    if not args.skip_profiling:
        summary["profiling"] = run_profiling(args, dry_run=args.dry_run)
    else:
        summary["profiling"] = []

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[PASS] summary written: {args.summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
