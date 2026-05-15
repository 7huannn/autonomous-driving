#!/usr/bin/env python3
"""Stage 10 benchmark runner for segmentation/detection pipelines."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str
    wall_seconds: float
    peak_vram_mib: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 10 benchmark/evaluation utility")
    parser.add_argument("--mode", choices=("benchmark", "report", "vram"), default="benchmark")
    parser.add_argument("--model", choices=("segmentation", "detection"), help="Required for benchmark/vram")

    parser.add_argument("--output", type=Path, default=Path("output/benchmark/benchmark_results.json"), help="Output file path")
    parser.add_argument("--report-output", type=Path, default=Path("output/benchmark/benchmark_report.md"), help="Markdown report output")
    parser.add_argument("--seg-results", type=Path, default=Path("output/benchmark/seg_benchmark.json"), help="Seg benchmark JSON")
    parser.add_argument("--det-results", type=Path, default=Path("output/benchmark/det_benchmark.json"), help="Det benchmark JSON")

    parser.add_argument("--num-frames", type=int, default=100)
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--vram-poll-seconds", type=float, default=0.2)
    parser.add_argument("--no-vram", action="store_true", help="Disable VRAM polling")

    parser.add_argument("--seg-conda-env", default="pad")
    parser.add_argument("--det-conda-env", default="pcdet")

    parser.add_argument("--seg-image-dir", type=Path, default=Path("data/processed/pad_format_stage08_canonical_mix_val/images"))
    parser.add_argument("--seg-gt-dir", type=Path, default=Path("data/processed/pad_format_stage08_canonical_mix_val/masks"))
    parser.add_argument("--seg-pad-repo", type=Path, default=Path("../repos/pytorch-auto-drive"))
    parser.add_argument(
        "--seg-config",
        type=Path,
        default=Path("configs/semantic_segmentation/erfnet/cityscapes_512x1024.py"),
    )
    parser.add_argument(
        "--seg-checkpoint",
        type=Path,
        default=Path("data/checkpoints/erfnet_cityscapes_512x1024_20200918.pt"),
    )
    parser.add_argument(
        "--seg-encoder-checkpoint",
        type=Path,
        default=Path("data/checkpoints/erfnet_encoder_pretrained.pth.tar"),
    )
    parser.add_argument("--seg-eval-json", type=Path, default=Path("output/segmentation_canonical_mix_val/eval_results.json"))

    parser.add_argument("--det-openpcdet-repo", type=Path, default=Path("../repos/OpenPCDet"))
    parser.add_argument("--det-dataset-dir", type=Path, default=Path("data/processed/pcdet_format_stage08_canonical_mix"))
    parser.add_argument("--det-cfg-file", type=Path, default=Path("configs/carla_lidar_probe.yaml"))
    parser.add_argument(
        "--det-checkpoint",
        type=Path,
        default=None,
        help=(
            "Detection checkpoint path. If omitted, auto-discover a fine-tuned "
            "carla_probe_finetune checkpoint under --det-openpcdet-repo/output, "
            "then fallback to data/checkpoints/pointpillar_7728.pth."
        ),
    )
    parser.add_argument("--det-split", choices=("train", "val"), default="val")
    parser.add_argument("--det-score-thresh", type=float, default=0.1)
    parser.add_argument("--det-eval-json", type=Path, default=Path("output/detection_3d_canonical_mix/eval_results.json"))

    parser.add_argument("--tmp-root", type=Path, default=Path("output/benchmark/tmp"))
    return parser.parse_args()


def resolve_args(args: argparse.Namespace) -> None:
    for key in (
        "output",
        "report_output",
        "seg_results",
        "det_results",
        "seg_image_dir",
        "seg_gt_dir",
        "seg_pad_repo",
        "seg_checkpoint",
        "seg_encoder_checkpoint",
        "seg_eval_json",
        "det_openpcdet_repo",
        "det_dataset_dir",
        "det_cfg_file",
        "det_checkpoint",
        "det_eval_json",
        "tmp_root",
    ):
        val = getattr(args, key)
        if isinstance(val, Path):
            setattr(args, key, val.resolve())

    if isinstance(args.det_checkpoint, Path):
        args.det_checkpoint = args.det_checkpoint.resolve()


def resolve_det_checkpoint(args: argparse.Namespace) -> Path:
    if isinstance(args.det_checkpoint, Path):
        return args.det_checkpoint

    output_root = args.det_openpcdet_repo / "output"
    if output_root.is_dir():
        matches = sorted(
            output_root.glob(
                "**/configs/carla_lidar_probe/carla_probe_finetune/ckpt/checkpoint_epoch_5.pth"
            )
        )
        if matches:
            return matches[-1].resolve()

    return Path("data/checkpoints/pointpillar_7728.pth").resolve()


def query_gpu_memory_mib() -> int | None:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        return None

    if proc.returncode != 0:
        return None

    values = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(int(line))
        except ValueError:
            continue
    if not values:
        return None
    return max(values)


def extract_json_block(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError(f"Could not parse JSON block from stdout:\n{text[-1200:]}")
    payload = text[start : end + 1]
    return json.loads(payload)


def run_with_optional_vram(cmd: list[str], poll_seconds: float, measure_vram: bool) -> CommandResult:
    t0 = time.time()
    if not measure_vram:
        proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
        return CommandResult(
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            wall_seconds=time.time() - t0,
            peak_vram_mib=None,
        )

    p = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    peak = None
    while p.poll() is None:
        mem = query_gpu_memory_mib()
        if mem is not None:
            peak = mem if peak is None else max(peak, mem)
        time.sleep(max(0.05, poll_seconds))

    stdout, stderr = p.communicate()
    return CommandResult(
        returncode=p.returncode,
        stdout=stdout,
        stderr=stderr,
        wall_seconds=time.time() - t0,
        peak_vram_mib=peak,
    )


def checkpoint_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"checkpoint_path": str(path), "exists": False, "size_bytes": None, "params": None, "params_million": None}

    size_bytes = int(path.stat().st_size)
    params = None
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict):
            state = None
            for key in ("model_state", "model", "state_dict", "model_state_dict"):
                if key in obj and hasattr(obj[key], "values"):
                    state = obj[key]
                    break
            if state is None and hasattr(obj, "values"):
                state = obj
            if state is not None:
                total = 0
                for value in state.values():
                    if hasattr(value, "numel"):
                        total += int(value.numel())
                params = int(total) if total > 0 else None
    except Exception:
        params = None

    return {
        "checkpoint_path": str(path),
        "exists": True,
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 3),
        "params": params,
        "params_million": (params / 1_000_000.0) if params is not None else None,
    }


def benchmark_segmentation(args: argparse.Namespace) -> dict[str, Any]:
    if not args.seg_image_dir.is_dir():
        raise FileNotFoundError(f"Seg image dir missing: {args.seg_image_dir}")

    measure_vram = not args.no_vram
    tmp_root = args.tmp_root / "seg"
    tmp_root.mkdir(parents=True, exist_ok=True)

    base_cmd = [
        "conda",
        "run",
        "-n",
        args.seg_conda_env,
        "python",
        "scripts/run_segmentation.py",
        "--mode",
        "infer",
        "--pad-repo",
        str(args.seg_pad_repo),
        "--config",
        str(args.seg_config),
        "--checkpoint",
        str(args.seg_checkpoint),
        "--encoder-checkpoint",
        str(args.seg_encoder_checkpoint),
        "--image-dir",
        str(args.seg_image_dir),
        "--num-frames",
        str(args.num_frames),
        "--batch-size",
        "1",
        "--mixed-precision",
        "--overwrite",
    ]

    warmup_cmd = base_cmd.copy()
    num_idx = warmup_cmd.index("--num-frames") + 1
    warmup_cmd[num_idx] = str(args.warmup_frames)
    warm_pred = tmp_root / "warmup_pred"
    warm_overlay = tmp_root / "warmup_overlay"
    warmup_cmd.extend(["--pred-dir", str(warm_pred), "--overlay-dir", str(warm_overlay)])
    warm_res = run_with_optional_vram(warmup_cmd, poll_seconds=args.vram_poll_seconds, measure_vram=measure_vram)
    if warm_res.returncode != 0:
        raise RuntimeError(f"Seg warmup failed:\nSTDOUT:\n{warm_res.stdout}\nSTDERR:\n{warm_res.stderr}")

    runs: list[dict[str, Any]] = []
    for idx in range(args.repeats):
        pred = tmp_root / f"pred_run{idx+1:02d}"
        ovl = tmp_root / f"overlay_run{idx+1:02d}"
        cmd = base_cmd + ["--pred-dir", str(pred), "--overlay-dir", str(ovl)]
        res = run_with_optional_vram(cmd, poll_seconds=args.vram_poll_seconds, measure_vram=measure_vram)
        if res.returncode != 0:
            raise RuntimeError(
                f"Seg benchmark run {idx+1} failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
            )

        payload = extract_json_block(res.stdout)
        inference = payload.get("inference", {})
        frames = int(inference.get("num_frames", args.num_frames))
        wall_fps = (frames / res.wall_seconds) if res.wall_seconds > 0 else None
        run = {
            "run_index": idx + 1,
            "num_frames": frames,
            "wall_seconds": res.wall_seconds,
            "wall_latency_ms_per_frame": (res.wall_seconds / frames * 1000.0) if frames > 0 else None,
            "wall_fps": wall_fps,
            "peak_vram_mib": res.peak_vram_mib,
            "forward_fps": inference.get("forward_fps"),
            "forward_latency_ms": inference.get("avg_forward_ms"),
            "device": inference.get("device"),
            "pred_dir": inference.get("pred_dir"),
        }
        runs.append(run)

    wall_fps_vals = [float(r["wall_fps"]) for r in runs if r["wall_fps"] is not None]
    wall_lat_vals = [float(r["wall_latency_ms_per_frame"]) for r in runs if r["wall_latency_ms_per_frame"] is not None]
    fwd_fps_vals = [float(r["forward_fps"]) for r in runs if r["forward_fps"] is not None]
    fwd_lat_vals = [float(r["forward_latency_ms"]) for r in runs if r["forward_latency_ms"] is not None]
    vram_vals = [int(r["peak_vram_mib"]) for r in runs if r["peak_vram_mib"] is not None]

    eval_data = read_optional_json(args.seg_eval_json)

    return {
        "mode": "benchmark",
        "model": "segmentation",
        "num_frames": args.num_frames,
        "warmup_frames": args.warmup_frames,
        "repeats": args.repeats,
        "env": args.seg_conda_env,
        "image_dir": str(args.seg_image_dir),
        "gt_dir": str(args.seg_gt_dir),
        "runs": runs,
        "summary": {
            "wall_fps_mean": safe_mean(wall_fps_vals),
            "wall_fps_std": safe_std(wall_fps_vals),
            "wall_latency_ms_mean": safe_mean(wall_lat_vals),
            "wall_latency_ms_std": safe_std(wall_lat_vals),
            "forward_fps_mean": safe_mean(fwd_fps_vals),
            "forward_fps_std": safe_std(fwd_fps_vals),
            "forward_latency_ms_mean": safe_mean(fwd_lat_vals),
            "forward_latency_ms_std": safe_std(fwd_lat_vals),
            "peak_vram_mib_max": max(vram_vals) if vram_vals else None,
        },
        "checkpoint": checkpoint_stats(args.seg_checkpoint),
        "accuracy": {
            "eval_json": str(args.seg_eval_json),
            "mIoU": eval_data.get("mIoU"),
            "pixel_accuracy": eval_data.get("pixel_accuracy"),
            "num_pairs_evaluated": eval_data.get("num_pairs_evaluated"),
        },
    }


def benchmark_detection(args: argparse.Namespace) -> dict[str, Any]:
    if not args.det_dataset_dir.is_dir():
        raise FileNotFoundError(f"Det dataset dir missing: {args.det_dataset_dir}")

    det_checkpoint = resolve_det_checkpoint(args)
    if not det_checkpoint.exists():
        raise FileNotFoundError(
            f"Det checkpoint not found: {det_checkpoint}. "
            "Pass --det-checkpoint explicitly or place checkpoint in data/checkpoints/"
        )

    measure_vram = not args.no_vram
    tmp_root = args.tmp_root / "det"
    tmp_root.mkdir(parents=True, exist_ok=True)

    base_cmd = [
        "conda",
        "run",
        "-n",
        args.det_conda_env,
        "python",
        "scripts/run_lidar_det.py",
        "--mode",
        "infer",
        "--openpcdet-repo",
        str(args.det_openpcdet_repo),
        "--cfg-file",
        str(args.det_cfg_file),
        "--checkpoint",
        str(det_checkpoint),
        "--dataset-dir",
        str(args.det_dataset_dir),
        "--split",
        args.det_split,
        "--num-frames",
        str(args.num_frames),
        "--batch-size",
        "1",
        "--score-thresh",
        str(args.det_score_thresh),
        "--overwrite",
    ]

    warmup_cmd = base_cmd.copy()
    num_idx = warmup_cmd.index("--num-frames") + 1
    warmup_cmd[num_idx] = str(args.warmup_frames)
    warm_pred = tmp_root / "warmup_pred"
    warmup_cmd.extend(["--pred-dir", str(warm_pred)])
    warm_res = run_with_optional_vram(warmup_cmd, poll_seconds=args.vram_poll_seconds, measure_vram=measure_vram)
    if warm_res.returncode != 0:
        raise RuntimeError(f"Det warmup failed:\nSTDOUT:\n{warm_res.stdout}\nSTDERR:\n{warm_res.stderr}")

    runs: list[dict[str, Any]] = []
    for idx in range(args.repeats):
        pred = tmp_root / f"pred_run{idx+1:02d}"
        cmd = base_cmd + ["--pred-dir", str(pred)]
        res = run_with_optional_vram(cmd, poll_seconds=args.vram_poll_seconds, measure_vram=measure_vram)
        if res.returncode != 0:
            raise RuntimeError(
                f"Det benchmark run {idx+1} failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
            )

        payload = extract_json_block(res.stdout)
        inference = payload.get("inference", {})
        frames = int(inference.get("num_frames", args.num_frames))
        wall_fps = (frames / res.wall_seconds) if res.wall_seconds > 0 else None
        run = {
            "run_index": idx + 1,
            "num_frames": frames,
            "wall_seconds": res.wall_seconds,
            "wall_latency_ms_per_frame": (res.wall_seconds / frames * 1000.0) if frames > 0 else None,
            "wall_fps": wall_fps,
            "peak_vram_mib": res.peak_vram_mib,
            "forward_fps": inference.get("forward_fps"),
            "forward_latency_ms": inference.get("avg_forward_ms"),
            "device": inference.get("device"),
            "pred_dir": inference.get("pred_dir"),
            "saved_boxes_score_ge_thresh": inference.get("saved_boxes_score_ge_thresh"),
        }
        runs.append(run)

    wall_fps_vals = [float(r["wall_fps"]) for r in runs if r["wall_fps"] is not None]
    wall_lat_vals = [float(r["wall_latency_ms_per_frame"]) for r in runs if r["wall_latency_ms_per_frame"] is not None]
    fwd_fps_vals = [float(r["forward_fps"]) for r in runs if r["forward_fps"] is not None]
    fwd_lat_vals = [float(r["forward_latency_ms"]) for r in runs if r["forward_latency_ms"] is not None]
    vram_vals = [int(r["peak_vram_mib"]) for r in runs if r["peak_vram_mib"] is not None]

    eval_data = read_optional_json(args.det_eval_json)

    return {
        "mode": "benchmark",
        "model": "detection",
        "num_frames": args.num_frames,
        "warmup_frames": args.warmup_frames,
        "repeats": args.repeats,
        "env": args.det_conda_env,
        "dataset_dir": str(args.det_dataset_dir),
        "split": args.det_split,
        "runs": runs,
        "summary": {
            "wall_fps_mean": safe_mean(wall_fps_vals),
            "wall_fps_std": safe_std(wall_fps_vals),
            "wall_latency_ms_mean": safe_mean(wall_lat_vals),
            "wall_latency_ms_std": safe_std(wall_lat_vals),
            "forward_fps_mean": safe_mean(fwd_fps_vals),
            "forward_fps_std": safe_std(fwd_fps_vals),
            "forward_latency_ms_mean": safe_mean(fwd_lat_vals),
            "forward_latency_ms_std": safe_std(fwd_lat_vals),
            "peak_vram_mib_max": max(vram_vals) if vram_vals else None,
        },
        "checkpoint": checkpoint_stats(det_checkpoint),
        "accuracy": {
            "eval_json": str(args.det_eval_json),
            "mAP": eval_data.get("mAP"),
            "metric": eval_data.get("metric"),
            "total_gt_boxes": eval_data.get("total_gt_boxes"),
            "num_frames_evaluated": eval_data.get("num_frames_evaluated"),
        },
    }


def run_vram_only(args: argparse.Namespace) -> dict[str, Any]:
    original_repeats = args.repeats
    original_num = args.num_frames
    args.repeats = 1
    args.num_frames = max(1, min(args.num_frames, 20))

    if args.model == "segmentation":
        result = benchmark_segmentation(args)
    elif args.model == "detection":
        result = benchmark_detection(args)
    else:
        raise ValueError("--model is required for vram mode")

    args.repeats = original_repeats
    args.num_frames = original_num

    peak = result.get("summary", {}).get("peak_vram_mib_max")
    out = {
        "mode": "vram",
        "model": args.model,
        "num_frames": result.get("num_frames"),
        "peak_vram_mib": peak,
        "source_result": result,
    }
    return out


def safe_mean(values: list[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def safe_std(values: list[float]) -> float | None:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0 if len(values) == 1 else None


def read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def format_float(x: float | None, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "N/A"
    return f"{x:.{nd}f}"


def format_size_mb(size_bytes: int | None) -> str:
    if size_bytes is None:
        return "N/A"
    return f"{size_bytes / (1024 * 1024):.2f} MB"


def generate_markdown_report(seg: dict[str, Any], det: dict[str, Any]) -> str:
    seg_sum = seg.get("summary", {})
    det_sum = det.get("summary", {})
    seg_ckpt = seg.get("checkpoint", {})
    det_ckpt = det.get("checkpoint", {})
    seg_acc = seg.get("accuracy", {})
    det_acc = det.get("accuracy", {})

    lines = []
    lines.append("# Stage 10 Benchmark Report")
    lines.append("")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Summary Table")
    lines.append("")
    lines.append("| Metric | ERFNet (Segmentation) | PointPillar (3D Detection) |")
    lines.append("|---|---:|---:|")
    lines.append(f"| Checkpoint size | {format_size_mb(seg_ckpt.get('size_bytes'))} | {format_size_mb(det_ckpt.get('size_bytes'))} |")
    lines.append(
        f"| Parameters | {format_float(seg_ckpt.get('params_million'), 3)} M | {format_float(det_ckpt.get('params_million'), 3)} M |"
    )
    lines.append(f"| Wall FPS (mean ± std) | {format_float(seg_sum.get('wall_fps_mean'))} ± {format_float(seg_sum.get('wall_fps_std'))} | {format_float(det_sum.get('wall_fps_mean'))} ± {format_float(det_sum.get('wall_fps_std'))} |")
    lines.append(f"| Wall latency ms/frame (mean ± std) | {format_float(seg_sum.get('wall_latency_ms_mean'))} ± {format_float(seg_sum.get('wall_latency_ms_std'))} | {format_float(det_sum.get('wall_latency_ms_mean'))} ± {format_float(det_sum.get('wall_latency_ms_std'))} |")
    lines.append(f"| Forward FPS (mean ± std) | {format_float(seg_sum.get('forward_fps_mean'))} ± {format_float(seg_sum.get('forward_fps_std'))} | {format_float(det_sum.get('forward_fps_mean'))} ± {format_float(det_sum.get('forward_fps_std'))} |")
    lines.append(f"| Forward latency ms (mean ± std) | {format_float(seg_sum.get('forward_latency_ms_mean'))} ± {format_float(seg_sum.get('forward_latency_ms_std'))} | {format_float(det_sum.get('forward_latency_ms_mean'))} ± {format_float(det_sum.get('forward_latency_ms_std'))} |")
    lines.append(f"| Peak VRAM (MiB, max) | {seg_sum.get('peak_vram_mib_max', 'N/A')} | {det_sum.get('peak_vram_mib_max', 'N/A')} |")
    lines.append(f"| Accuracy metric | mIoU={format_float(seg_acc.get('mIoU'), 4)} | mAP={format_float(det_acc.get('mAP'), 4)} |")
    lines.append("")
    lines.append("## Accuracy Sources")
    lines.append("")
    lines.append(f"- Segmentation eval JSON: `{seg_acc.get('eval_json', 'N/A')}`")
    lines.append(f"- Detection eval JSON: `{det_acc.get('eval_json', 'N/A')}`")
    lines.append("")
    lines.append("## Raw Benchmark JSON")
    lines.append("")
    lines.append(f"- Segmentation: `{seg.get('output_path', 'output/benchmark/seg_benchmark.json')}`")
    lines.append(f"- Detection: `{det.get('output_path', 'output/benchmark/det_benchmark.json')}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    resolve_args(args)

    if args.mode in ("benchmark", "vram") and not args.model:
        raise ValueError("--model is required for benchmark/vram mode")

    if args.mode == "benchmark":
        if args.model == "segmentation":
            result = benchmark_segmentation(args)
        elif args.model == "detection":
            result = benchmark_detection(args)
        else:
            raise ValueError(f"Unsupported model: {args.model}")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        print(f"[PASS] benchmark saved: {args.output}")
        return 0

    if args.mode == "vram":
        result = run_vram_only(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        print(f"[PASS] vram result saved: {args.output}")
        return 0

    seg = read_optional_json(args.seg_results)
    det = read_optional_json(args.det_results)
    if not seg:
        raise FileNotFoundError(f"Missing/empty seg results: {args.seg_results}")
    if not det:
        raise FileNotFoundError(f"Missing/empty det results: {args.det_results}")

    seg["output_path"] = str(args.seg_results)
    det["output_path"] = str(args.det_results)

    report_md = generate_markdown_report(seg, det)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(report_md, encoding="utf-8")

    result = {
        "mode": "report",
        "seg_results": str(args.seg_results),
        "det_results": str(args.det_results),
        "report_output": str(args.report_output),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"[PASS] benchmark report saved: {args.report_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
