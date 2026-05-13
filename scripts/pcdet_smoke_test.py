#!/usr/bin/env python3
"""Stage 05 OpenPCDet smoke test runner."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


DEFAULT_CKPT_URL = "https://drive.google.com/file/d/1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm/view"


def run_cmd(cmd: List[str], cwd: Path, env: Dict[str, str] | None = None, dry_run: bool = False) -> subprocess.CompletedProcess:
    print(f"[CMD] (cwd={cwd}) {' '.join(cmd)}")
    if dry_run:
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
    return subprocess.run(cmd, cwd=str(cwd), env=env, text=True, capture_output=True, check=True)


def read_gpu_mem_mib() -> int | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        ).strip().splitlines()
        return int(out[0])
    except Exception:
        return None


def install_gdown_if_missing(python_exe: str, dry_run: bool) -> None:
    if dry_run:
        return
    probe = subprocess.run([python_exe, "-m", "gdown", "--help"], text=True, capture_output=True)
    if probe.returncode == 0:
        return
    run_cmd([python_exe, "-m", "pip", "install", "--no-input", "gdown"], cwd=Path.cwd(), dry_run=False)


def ensure_checkpoint(args: argparse.Namespace) -> None:
    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if args.checkpoint_path.exists() or args.skip_download:
        return
    install_gdown_if_missing(args.python_exe, dry_run=args.dry_run)
    run_cmd(
        [
            args.python_exe,
            "-m",
            "gdown",
            "--fuzzy",
            args.checkpoint_url,
            "-O",
            str(args.checkpoint_path),
        ],
        cwd=args.repo_root,
        dry_run=args.dry_run,
    )


def ensure_sample_points(args: argparse.Namespace) -> None:
    args.sample_npy.parent.mkdir(parents=True, exist_ok=True)
    if args.sample_npy.exists() and not args.force_regenerate_sample:
        return

    rng = np.random.default_rng(args.sample_seed)
    n = args.sample_points
    points = np.zeros((n, 4), dtype=np.float32)
    points[:, 0] = rng.uniform(args.sample_x_min, args.sample_x_max, n)
    points[:, 1] = rng.uniform(args.sample_y_min, args.sample_y_max, n)
    points[:, 2] = rng.uniform(args.sample_z_min, args.sample_z_max, n)
    points[:, 3] = rng.uniform(0.0, 1.0, n)

    # Add a few denser clusters to increase chance of non-empty predictions.
    for cx, cy in [(12.0, -1.5), (25.0, 3.0), (35.0, -6.0)]:
        m = min(3500, n // 8)
        idx = rng.choice(n, size=m, replace=False)
        points[idx, 0] = rng.normal(cx, 1.3, m)
        points[idx, 1] = rng.normal(cy, 0.9, m)
        points[idx, 2] = rng.normal(-1.2, 0.3, m)
        points[idx, 3] = rng.uniform(0.2, 1.0, m)

    if not args.dry_run:
        np.save(args.sample_npy, points)


def verify_cuda_extensions(args: argparse.Namespace) -> Dict[str, object]:
    code = (
        "import torch\n"
        "from pcdet.ops.iou3d_nms import iou3d_nms_cuda\n"
        "from pcdet.ops.roiaware_pool3d import roiaware_pool3d_cuda\n"
        "from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_stack_cuda\n"
        "from pcdet.ops.pointnet2.pointnet2_batch import pointnet2_batch_cuda\n"
        "print('CUDA_EXT_OK')\n"
    )
    proc = run_cmd([args.python_exe, "-c", code], cwd=args.repo_root, dry_run=args.dry_run)
    return {
        "status": "ok",
        "stdout_tail": proc.stdout.splitlines()[-10:],
        "stderr_tail": proc.stderr.splitlines()[-10:],
    }


def _inject_argo2_stub() -> None:
    """Bypass optional Argo2 dependency chain in this repo snapshot."""
    mod_name = "repos.OpenPCDet.pcdet.datasets.argo2.argo2_dataset"
    if mod_name in sys.modules:
        return
    mod = types.ModuleType(mod_name)
    mod.Argo2Dataset = type("Argo2Dataset", (object,), {})
    sys.modules[mod_name] = mod


def _inject_dsvt_stub() -> None:
    """Bypass broken DSVT import path in this repo snapshot."""
    mod_name = "repos.OpenPCDet.pcdet.models.backbones_3d.dsvt"
    if mod_name in sys.modules:
        return
    mod = types.ModuleType(mod_name)
    mod.DSVT = type("DSVT", (object,), {})
    sys.modules[mod_name] = mod


def draw_bev_image(points: np.ndarray, boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 7), dpi=130)
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(points[:, 0], points[:, 1], s=0.06, c=points[:, 2], cmap="viridis", alpha=0.55)

    def rot_rect(cx: float, cy: float, dx: float, dy: float, yaw: float) -> np.ndarray:
        corners = np.array(
            [[dx / 2, dy / 2], [dx / 2, -dy / 2], [-dx / 2, -dy / 2], [-dx / 2, dy / 2], [dx / 2, dy / 2]],
            dtype=np.float32,
        )
        c, s = math.cos(yaw), math.sin(yaw)
        r = np.array([[c, -s], [s, c]], dtype=np.float32)
        return corners @ r.T + np.array([cx, cy], dtype=np.float32)

    colors = {1: "#ff3b30", 2: "#007aff", 3: "#34c759"}
    for i in range(boxes.shape[0]):
        x, y, _z, dx, dy, _dz, yaw = boxes[i, :7].tolist()
        poly = rot_rect(x, y, dx, dy, yaw)
        color = colors.get(int(labels[i]), "#ff9500")
        ax.plot(poly[:, 0], poly[:, 1], color=color, linewidth=1.2)
        ax.text(x, y, f"{float(scores[i]):.2f}", color=color, fontsize=6)

    ax.set_title("OpenPCDet PointPillar Smoke Test (BEV)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(0, 70)
    ax.set_ylim(-40, 40)
    ax.grid(alpha=0.15)
    ax.set_aspect("equal", adjustable="box")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run_inference(args: argparse.Namespace) -> Dict[str, object]:
    start_total = time.time()
    base_mem = read_gpu_mem_mib()
    peak_mem = base_mem

    # Ensure repo-root-style imports used by this OpenPCDet snapshot.
    if str(args.autonomous_root) not in sys.path:
        sys.path.insert(0, str(args.autonomous_root))

    _inject_argo2_stub()
    _inject_dsvt_stub()

    from repos.OpenPCDet.pcdet.config import cfg, cfg_from_yaml_file
    from repos.OpenPCDet.pcdet.datasets import DatasetTemplate
    from repos.OpenPCDet.pcdet.models import build_network, load_data_to_gpu
    from repos.OpenPCDet.pcdet.utils import common_utils
    import torch

    class DemoDataset(DatasetTemplate):
        def __init__(self, dataset_cfg, class_names, training, root_path, ext, logger):
            super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger)
            self.root_path = root_path
            self.ext = ext
            if self.root_path.is_dir():
                self.sample_file_list = sorted(self.root_path.glob(f"*{self.ext}"))
            else:
                self.sample_file_list = [self.root_path]

        def __len__(self):
            return len(self.sample_file_list)

        def __getitem__(self, index):
            sample = self.sample_file_list[index]
            if self.ext == ".bin":
                pts = np.fromfile(sample, dtype=np.float32).reshape(-1, 4)
            elif self.ext == ".npy":
                pts = np.load(sample)
            else:
                raise NotImplementedError(self.ext)
            inp = {"points": pts, "frame_id": index}
            return self.prepare_data(data_dict=inp)

    os.chdir(args.openpcdet_tools_dir)
    cfg_from_yaml_file(str(args.cfg_file), cfg)
    logger = common_utils.create_logger()

    dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        root_path=args.sample_npy,
        ext=args.sample_ext,
        logger=logger,
    )

    t_model_start = time.time()
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=str(args.checkpoint_path), logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    model_setup_seconds = time.time() - t_model_start

    last_pred = None
    forward_seconds = 0.0
    with torch.no_grad():
        for idx in range(len(dataset)):
            data_dict = dataset[idx]
            data_dict = dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            mem_now = read_gpu_mem_mib()
            if mem_now is not None:
                peak_mem = mem_now if peak_mem is None else max(peak_mem, mem_now)
            t_fw = time.time()
            pred_dicts, _ = model.forward(data_dict)
            forward_seconds += time.time() - t_fw
            last_pred = (data_dict, pred_dicts[0])

    if last_pred is None:
        raise RuntimeError("No prediction output produced")

    data_dict, pred = last_pred
    boxes = pred["pred_boxes"].detach().cpu().numpy()
    scores = pred["pred_scores"].detach().cpu().numpy()
    labels = pred["pred_labels"].detach().cpu().numpy()
    points_tensor = data_dict["points"][:, 1:5]  # remove batch-index column
    if hasattr(points_tensor, "detach"):
        points = points_tensor.detach().cpu().numpy()
    else:
        points = np.asarray(points_tensor)

    if boxes.shape[0] > args.max_boxes_to_draw:
        order = np.argsort(-scores)
        keep = order[: args.max_boxes_to_draw]
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    # Measure steady-state forward latency with warmup iterations on the same batch.
    bench_avg_ms = None
    bench_fps = None
    if args.bench_repeats > 0:
        import torch

        with torch.no_grad():
            for _ in range(args.bench_warmup):
                model.forward(data_dict)
            t_bench = time.time()
            for _ in range(args.bench_repeats):
                model.forward(data_dict)
            t_elapsed = time.time() - t_bench
        if t_elapsed > 0:
            bench_avg_ms = (t_elapsed / args.bench_repeats) * 1000.0
            bench_fps = args.bench_repeats / t_elapsed

    if not args.dry_run:
        draw_bev_image(points, boxes, scores, labels, args.doc_image)

    elapsed_total = round(time.time() - start_total, 4)
    elapsed_forward = round(forward_seconds, 4)
    fps_forward = (len(dataset) / forward_seconds) if forward_seconds > 0 else None
    delta_mem = None
    if base_mem is not None and peak_mem is not None:
        delta_mem = peak_mem - base_mem

    return {
        "num_input_points": int(points.shape[0]),
        "num_pred_boxes": int(boxes.shape[0]),
        "max_score": float(scores.max()) if scores.size > 0 else 0.0,
        "mean_score": float(scores.mean()) if scores.size > 0 else 0.0,
        "elapsed_total_seconds": elapsed_total,
        "model_setup_seconds": round(model_setup_seconds, 4),
        "forward_seconds": elapsed_forward,
        "forward_fps": float(fps_forward) if fps_forward is not None else None,
        "bench_repeats": int(args.bench_repeats),
        "bench_warmup": int(args.bench_warmup),
        "bench_avg_forward_ms": float(bench_avg_ms) if bench_avg_ms is not None else None,
        "bench_forward_fps": float(bench_fps) if bench_fps is not None else None,
        "baseline_mem_mib": base_mem,
        "peak_mem_mib": peak_mem,
        "delta_mem_mib": delta_mem,
        "doc_image": str(args.doc_image),
    }


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    autonomous_root = repo_root.parent
    openpcdet_repo = (autonomous_root / "repos/OpenPCDet").resolve()
    openpcdet_tools = (openpcdet_repo / "tools").resolve()

    parser = argparse.ArgumentParser(description="Stage 05 OpenPCDet smoke test runner")
    parser.add_argument("--openpcdet-repo", type=Path, default=openpcdet_repo)
    parser.add_argument("--cfg-file", type=Path, default=openpcdet_tools / "cfgs/kitti_models/pointpillar.yaml")
    parser.add_argument("--checkpoint-url", default=DEFAULT_CKPT_URL)
    parser.add_argument("--checkpoint-path", type=Path, default=repo_root / "data/checkpoints/pointpillar_7728.pth")
    parser.add_argument("--sample-npy", type=Path, default=repo_root / "data/processed/pcdet_test_data/sample_lidar.npy")
    parser.add_argument("--sample-ext", default=".npy")
    parser.add_argument("--sample-points", type=int, default=50000)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--sample-x-min", type=float, default=-40.0)
    parser.add_argument("--sample-x-max", type=float, default=40.0)
    parser.add_argument("--sample-y-min", type=float, default=-40.0)
    parser.add_argument("--sample-y-max", type=float, default=40.0)
    parser.add_argument("--sample-z-min", type=float, default=-3.0)
    parser.add_argument("--sample-z-max", type=float, default=1.0)
    parser.add_argument("--force-regenerate-sample", action="store_true")
    parser.add_argument("--doc-image", type=Path, default=repo_root / "docs/pcdet_smoke_test.png")
    parser.add_argument("--summary-json", type=Path, default=repo_root / "output/pcdet_smoke_test/stage05_summary.json")
    parser.add_argument("--max-boxes-to-draw", type=int, default=60)
    parser.add_argument("--bench-warmup", type=int, default=2)
    parser.add_argument("--bench-repeats", type=int, default=5)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.repo_root = repo_root.resolve()
    args.autonomous_root = autonomous_root.resolve()
    args.openpcdet_repo = args.openpcdet_repo.resolve()
    args.openpcdet_tools_dir = (args.openpcdet_repo / "tools").resolve()
    args.cfg_file = args.cfg_file.resolve()
    args.checkpoint_path = args.checkpoint_path.resolve()
    args.sample_npy = args.sample_npy.resolve()
    args.doc_image = args.doc_image.resolve()
    args.summary_json = args.summary_json.resolve()
    args.python_exe = sys.executable

    if not args.openpcdet_repo.exists():
        raise FileNotFoundError(f"OpenPCDet repo not found: {args.openpcdet_repo}")
    if not args.cfg_file.exists():
        raise FileNotFoundError(f"Config not found: {args.cfg_file}")

    return args


def main() -> int:
    args = parse_args()

    summary: Dict[str, object] = {
        "stage": "STAGE_05_OPENPCDET_SMOKE_TEST",
        "openpcdet_repo": str(args.openpcdet_repo),
        "cfg_file": str(args.cfg_file),
        "checkpoint_path": str(args.checkpoint_path),
        "sample_npy": str(args.sample_npy),
        "doc_image": str(args.doc_image),
        "dry_run": bool(args.dry_run),
    }

    ensure_checkpoint(args)
    ensure_sample_points(args)
    summary["cuda_extensions"] = verify_cuda_extensions(args)
    summary["inference"] = run_inference(args)

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[PASS] summary written: {args.summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
