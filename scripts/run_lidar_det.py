#!/usr/bin/env python3
"""Stage 08 LiDAR detection integration runner for OpenPCDet PointPillar."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import types
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OpenPCDet PointPillar inference/evaluation/visualization on CARLA LiDAR data",
    )
    parser.add_argument(
        "--mode",
        choices=["infer", "evaluate", "infer_and_evaluate", "visualize"],
        default="infer_and_evaluate",
        help="Pipeline mode",
    )

    parser.add_argument("--openpcdet-repo", type=Path, default=Path("../repos/OpenPCDet"), help="Path to OpenPCDet repo")
    parser.add_argument("--cfg-file", type=Path, default=Path("configs/carla_lidar.yaml"), help="PointPillar config")
    parser.add_argument("--checkpoint", type=Path, default=Path("data/checkpoints/pointpillar_7728.pth"), help="PointPillar checkpoint")

    parser.add_argument("--dataset-dir", type=Path, default=Path("data/processed/pcdet_format"), help="OpenPCDet custom dataset root")
    parser.add_argument("--split", choices=["train", "val"], default="val", help="Dataset split to run")
    parser.add_argument("--num-frames", type=int, default=0, help="Limit number of frames (0 = all)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (must be 1 on this hardware)")

    parser.add_argument("--pred-dir", type=Path, default=Path("output/detection_3d/predictions"), help="Directory to write per-frame prediction txt")
    parser.add_argument("--eval-output", type=Path, default=Path("output/detection_3d/eval_results.json"), help="Evaluation JSON output path")

    parser.add_argument("--vis-output-dir", type=Path, default=Path("output/detection_3d/visualizations"), help="Visualization output directory")
    parser.add_argument("--vis-max-frames", type=int, default=20, help="Max frames to visualize")
    parser.add_argument("--vis-size", type=int, default=1024, help="Visualization image size")
    parser.add_argument("--vis-range", type=float, default=75.2, help="BEV render range in meters")

    parser.add_argument("--overwrite", action="store_true", help="Overwrite prediction/visualization outputs")
    parser.add_argument("--score-thresh", type=float, default=0.1, help="Score threshold for saved predictions")
    parser.add_argument(
        "--allow-empty-gt",
        action="store_true",
        help="Allow evaluation to proceed when GT is fully empty (debug only).",
    )
    return parser.parse_args()


def _inject_argo2_stub() -> None:
    mod_name = "repos.OpenPCDet.pcdet.datasets.argo2.argo2_dataset"
    if mod_name in sys.modules:
        return
    mod = types.ModuleType(mod_name)
    mod.Argo2Dataset = type("Argo2Dataset", (object,), {})
    sys.modules[mod_name] = mod


def _inject_dsvt_stub() -> None:
    mod_name = "repos.OpenPCDet.pcdet.models.backbones_3d.dsvt"
    if mod_name in sys.modules:
        return
    mod = types.ModuleType(mod_name)
    mod.DSVT = type("DSVT", (object,), {})
    sys.modules[mod_name] = mod


def prepare_env(project_root: Path):
    autonomous_root = project_root.parent.resolve()
    if str(autonomous_root) not in sys.path:
        sys.path.insert(0, str(autonomous_root))

    _inject_argo2_stub()
    _inject_dsvt_stub()

    from repos.OpenPCDet.pcdet.config import cfg, cfg_from_yaml_file  # type: ignore
    from repos.OpenPCDet.pcdet.datasets.custom.custom_dataset import CustomDataset  # type: ignore
    from repos.OpenPCDet.pcdet.models import build_network, load_data_to_gpu  # type: ignore
    from repos.OpenPCDet.pcdet.utils import common_utils  # type: ignore

    return cfg, cfg_from_yaml_file, CustomDataset, build_network, load_data_to_gpu, common_utils


def ensure_infos_exist(dataset_dir: Path, split: str) -> None:
    needed = [dataset_dir / "custom_infos_train.pkl", dataset_dir / "custom_infos_val.pkl"]
    missing = [p for p in needed if not p.exists()]
    if missing:
        msg = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(
            f"Missing custom info files: {msg}. "
            "Generate them with scripts/convert_to_pcdet.py --generate-custom-infos"
        )

    split_file = dataset_dir / "ImageSets" / ("val.txt" if split == "val" else "train.txt")
    if not split_file.exists():
        raise FileNotFoundError(f"Missing split file: {split_file}")


def load_dataset_and_model(args: argparse.Namespace, project_root: Path):
    if args.batch_size != 1:
        raise ValueError("--batch-size must be 1 for Stage 08 hardware guardrails")

    openpcdet_repo = args.openpcdet_repo.resolve()
    if not openpcdet_repo.exists():
        raise FileNotFoundError(f"OpenPCDet repo not found: {openpcdet_repo}")

    cfg, cfg_from_yaml_file, CustomDataset, build_network, load_data_to_gpu, common_utils = prepare_env(project_root=project_root)

    cfg_path = args.cfg_file.resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    cfg_from_yaml_file(str(cfg_path), cfg)

    dataset_dir = args.dataset_dir.resolve()
    ensure_infos_exist(dataset_dir, split=args.split)

    cfg.DATA_CONFIG.DATA_PATH = str(dataset_dir)
    if args.split == "val":
        cfg.DATA_CONFIG.DATA_SPLIT.test = "val"
        cfg.DATA_CONFIG.INFO_PATH.test = ["custom_infos_val.pkl"]
    else:
        cfg.DATA_CONFIG.DATA_SPLIT.test = "train"
        cfg.DATA_CONFIG.INFO_PATH.test = ["custom_infos_train.pkl"]

    logger = common_utils.create_logger()
    dataset = CustomDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        root_path=dataset_dir,
        logger=logger,
    )

    ckpt = args.checkpoint.resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=str(ckpt), logger=logger, to_cpu=True)

    import torch

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return cfg, dataset, model, load_data_to_gpu, device


def write_prediction_txt(path: Path, anno: dict[str, Any], score_thresh: float) -> int:
    names = anno.get("name", np.array([]))
    scores = anno.get("score", np.array([]))
    boxes = anno.get("boxes_lidar", np.zeros((0, 7), dtype=np.float32))

    kept = 0
    with path.open("w", encoding="utf-8") as f:
        for i in range(len(names)):
            score = float(scores[i])
            if score < score_thresh:
                continue
            box = boxes[i]
            f.write(
                f"{box[0]:.6f} {box[1]:.6f} {box[2]:.6f} "
                f"{box[3]:.6f} {box[4]:.6f} {box[5]:.6f} {box[6]:.6f} "
                f"{score:.6f} {names[i]}\n"
            )
            kept += 1
    return kept


def run_inference(args: argparse.Namespace, project_root: Path) -> dict[str, Any]:
    cfg, dataset, model, load_data_to_gpu, device = load_dataset_and_model(args, project_root)

    pred_dir = args.pred_dir.resolve()
    pred_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for p in pred_dir.glob("*.txt"):
            p.unlink()

    import torch

    n_total = len(dataset)
    if args.num_frames > 0:
        n_total = min(n_total, args.num_frames)

    det_annos: list[dict[str, Any]] = []
    total_forward = 0.0
    saved_boxes = 0

    with torch.no_grad():
        for idx in range(n_total):
            data_dict = dataset[idx]
            batch_dict = dataset.collate_batch([data_dict])
            load_data_to_gpu(batch_dict)

            t0 = time.time()
            pred_dicts, _ = model.forward(batch_dict)
            total_forward += (time.time() - t0)

            annos = dataset.generate_prediction_dicts(batch_dict, pred_dicts, cfg.CLASS_NAMES)
            anno = annos[0]
            det_annos.append(anno)

            frame_id = str(anno.get("frame_id", f"{idx:06d}"))
            saved_boxes += write_prediction_txt(pred_dir / f"{frame_id}.txt", anno, score_thresh=args.score_thresh)

    fps = (n_total / total_forward) if total_forward > 0 else None
    avg_ms = (total_forward / n_total * 1000.0) if n_total > 0 else None

    packed_path = pred_dir / "predictions_compact.npz"
    np.savez_compressed(packed_path, det_annos=np.array(det_annos, dtype=object))

    return {
        "num_frames": int(n_total),
        "pred_dir": str(pred_dir),
        "predictions_npz": str(packed_path),
        "saved_boxes_score_ge_thresh": int(saved_boxes),
        "score_thresh": float(args.score_thresh),
        "avg_forward_ms": avg_ms,
        "forward_fps": fps,
        "device": str(device),
        "class_names": list(cfg.CLASS_NAMES),
    }


def read_prediction_txt(path: Path) -> list[dict[str, Any]]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 9:
            continue
        x, y, z, dx, dy, dz, heading, score = map(float, parts[:8])
        cls = parts[8]
        out.append(
            {
                "x": x,
                "y": y,
                "z": z,
                "dx": dx,
                "dy": dy,
                "dz": dz,
                "heading": heading,
                "score": score,
                "class_name": cls,
            }
        )
    return out


def read_gt_label_txt(path: Path) -> list[dict[str, Any]]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 8:
            continue
        x, y, z, dx, dy, dz, heading = map(float, parts[:7])
        cls = parts[7]
        out.append(
            {
                "x": x,
                "y": y,
                "z": z,
                "dx": dx,
                "dy": dy,
                "dz": dz,
                "heading": heading,
                "class_name": cls,
            }
        )
    return out


def _box_dict_to_arr(box: dict[str, Any]) -> np.ndarray:
    return np.array([box["x"], box["y"], box["z"], box["dx"], box["dy"], box["dz"], box["heading"]], dtype=np.float32)


def _ap_from_pr(recalls: np.ndarray, precisions: np.ndarray) -> float:
    if recalls.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _compute_ap_for_class(
    class_name: str,
    iou_thresh: float,
    frame_ids: list[str],
    pred_by_frame: dict[str, list[dict[str, Any]]],
    gt_by_frame: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    from repos.OpenPCDet.pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_bev_iou_cpu  # type: ignore

    total_gt = 0
    gt_used: dict[str, list[bool]] = {}
    for fid in frame_ids:
        gts = [g for g in gt_by_frame.get(fid, []) if g["class_name"] == class_name]
        total_gt += len(gts)
        gt_used[fid] = [False] * len(gts)

    preds_all = []
    for fid in frame_ids:
        for p in pred_by_frame.get(fid, []):
            if p["class_name"] == class_name:
                preds_all.append((fid, p))

    preds_all.sort(key=lambda x: float(x[1]["score"]), reverse=True)
    num_pred = len(preds_all)
    if num_pred == 0:
        return {
            "ap": 0.0,
            "num_gt": int(total_gt),
            "num_pred": 0,
            "precision": 0.0,
            "recall": 0.0,
            "iou_threshold": float(iou_thresh),
        }

    tp = np.zeros((num_pred,), dtype=np.float32)
    fp = np.zeros((num_pred,), dtype=np.float32)

    for i, (fid, pred) in enumerate(preds_all):
        gts = [g for g in gt_by_frame.get(fid, []) if g["class_name"] == class_name]
        if len(gts) == 0:
            fp[i] = 1.0
            continue

        pred_box = _box_dict_to_arr(pred).reshape(1, 7)
        gt_boxes = np.stack([_box_dict_to_arr(g) for g in gts], axis=0)
        ious = boxes_bev_iou_cpu(pred_box, gt_boxes).reshape(-1)

        best_idx = int(np.argmax(ious))
        best_iou = float(ious[best_idx])
        if best_iou >= iou_thresh and not gt_used[fid][best_idx]:
            tp[i] = 1.0
            gt_used[fid][best_idx] = True
        else:
            fp[i] = 1.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-8)
    recalls = tp_cum / max(total_gt, 1)
    ap = _ap_from_pr(recalls, precisions)

    return {
        "ap": float(ap),
        "num_gt": int(total_gt),
        "num_pred": int(num_pred),
        "precision": float(precisions[-1]) if precisions.size > 0 else 0.0,
        "recall": float(recalls[-1]) if recalls.size > 0 else 0.0,
        "iou_threshold": float(iou_thresh),
    }


def evaluate_predictions(args: argparse.Namespace, project_root: Path) -> dict[str, Any]:
    dataset_dir = args.dataset_dir.resolve()
    split_file = dataset_dir / "ImageSets" / ("val.txt" if args.split == "val" else "train.txt")
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    frame_ids = [x.strip() for x in split_file.read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.num_frames > 0:
        frame_ids = frame_ids[: args.num_frames]

    pred_by_frame: dict[str, list[dict[str, Any]]] = {}
    gt_by_frame: dict[str, list[dict[str, Any]]] = {}
    for fid in frame_ids:
        pred_by_frame[fid] = read_prediction_txt(args.pred_dir.resolve() / f"{fid}.txt")
        gt_by_frame[fid] = read_gt_label_txt(dataset_dir / "labels" / f"{fid}.txt")

    iou_thresholds = {"Vehicle": 0.7, "Pedestrian": 0.5, "Cyclist": 0.5}
    total_gt = int(sum(len(gt_by_frame[fid]) for fid in frame_ids))
    if total_gt == 0 and not args.allow_empty_gt:
        raise RuntimeError(
            "Evaluation aborted: no ground-truth boxes found in selected split. "
            "Use --allow-empty-gt only for explicit debug runs."
        )

    if total_gt == 0 and args.allow_empty_gt:
        class_results = {}
        for cls, thr in iou_thresholds.items():
            num_pred = int(sum(1 for fid in frame_ids for pred in pred_by_frame[fid] if pred["class_name"] == cls))
            class_results[cls] = {
                "ap": 0.0,
                "num_gt": 0,
                "num_pred": num_pred,
                "precision": 0.0,
                "recall": 0.0,
                "iou_threshold": float(thr),
            }
        result = {
            "num_frames_evaluated": len(frame_ids),
            "total_gt_boxes": 0,
            "allow_empty_gt": True,
            "metric": "BEV_AP_custom",
            "class_ap": class_results,
            "mAP": 0.0,
        }
        args.eval_output.parent.mkdir(parents=True, exist_ok=True)
        args.eval_output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    prepare_env(project_root=project_root)

    class_results: dict[str, Any] = {}
    for cls, thr in iou_thresholds.items():
        class_results[cls] = _compute_ap_for_class(
            class_name=cls,
            iou_thresh=thr,
            frame_ids=frame_ids,
            pred_by_frame=pred_by_frame,
            gt_by_frame=gt_by_frame,
        )

    total_gt = int(sum(v["num_gt"] for v in class_results.values()))

    valid_aps = [v["ap"] for v in class_results.values() if v["num_gt"] > 0]
    map_bev = float(np.mean(valid_aps)) if valid_aps else 0.0

    result = {
        "num_frames_evaluated": len(frame_ids),
        "total_gt_boxes": total_gt,
        "allow_empty_gt": bool(args.allow_empty_gt),
        "metric": "BEV_AP_custom",
        "class_ap": class_results,
        "mAP": map_bev,
    }

    args.eval_output.parent.mkdir(parents=True, exist_ok=True)
    args.eval_output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def world_to_pixel_xy(x: float, y: float, size: int, bev_range: float) -> tuple[int, int] | None:
    if abs(x) > bev_range or abs(y) > bev_range:
        return None
    u = int(((x + bev_range) / (2 * bev_range)) * (size - 1))
    v = int(((bev_range - y) / (2 * bev_range)) * (size - 1))
    return u, v


def rotated_rect(cx: float, cy: float, dx: float, dy: float, yaw: float) -> np.ndarray:
    corners = np.array(
        [[dx / 2, dy / 2], [dx / 2, -dy / 2], [-dx / 2, -dy / 2], [-dx / 2, dy / 2], [dx / 2, dy / 2]],
        dtype=np.float32,
    )
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return corners @ rot.T + np.array([cx, cy], dtype=np.float32)


def render_bev(points: np.ndarray, preds: list[dict[str, Any]], size: int, bev_range: float) -> np.ndarray:
    canvas = np.zeros((size, size, 3), dtype=np.uint8)

    xy = points[:, :2]
    mask = (
        (xy[:, 0] >= -bev_range)
        & (xy[:, 0] <= bev_range)
        & (xy[:, 1] >= -bev_range)
        & (xy[:, 1] <= bev_range)
    )
    pts = points[mask]

    if pts.shape[0] > 0:
        z = pts[:, 2]
        zmin = float(np.percentile(z, 5))
        zmax = float(np.percentile(z, 95))
        if zmax <= zmin:
            zmax = zmin + 1e-3
        norm = np.clip((z - zmin) / (zmax - zmin), 0.0, 1.0)
        colors = (norm * 255.0).astype(np.uint8)

        for i in range(pts.shape[0]):
            uv = world_to_pixel_xy(float(pts[i, 0]), float(pts[i, 1]), size, bev_range)
            if uv is None:
                continue
            u, v = uv
            canvas[v, u] = (colors[i], 255 - colors[i], 64)

    class_colors = {"Vehicle": (0, 0, 255), "Pedestrian": (255, 0, 0), "Cyclist": (0, 255, 0)}
    for obj in preds:
        poly = rotated_rect(obj["x"], obj["y"], obj["dx"], obj["dy"], obj["heading"])
        pts_px = []
        for p in poly:
            uv = world_to_pixel_xy(float(p[0]), float(p[1]), size, bev_range)
            if uv is None:
                pts_px = []
                break
            pts_px.append(uv)
        if len(pts_px) < 2:
            continue
        color = class_colors.get(obj["class_name"], (0, 255, 255))
        cv2.polylines(canvas, [np.array(pts_px, dtype=np.int32)], isClosed=False, color=color, thickness=2)
        cv2.putText(
            canvas,
            f"{obj['class_name']} {obj['score']:.2f}",
            pts_px[0],
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.putText(canvas, "BEV: x forward, y left (OpenPCDet)", (10, size - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
    return canvas


def run_visualization(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = args.dataset_dir.resolve()
    points_dir = dataset_dir / "points"
    pred_dir = args.pred_dir.resolve()
    vis_out = args.vis_output_dir.resolve()

    if not points_dir.exists():
        raise FileNotFoundError(f"Points directory not found: {points_dir}")
    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")

    vis_out.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for p in vis_out.glob("*.png"):
            p.unlink()

    point_files = sorted(points_dir.glob("*.npy"))
    if args.num_frames > 0:
        point_files = point_files[: args.num_frames]
    if args.vis_max_frames > 0:
        point_files = point_files[: args.vis_max_frames]

    written = 0
    for pf in point_files:
        preds = read_prediction_txt(pred_dir / f"{pf.stem}.txt")
        points = np.load(pf)
        bev = render_bev(points, preds, size=args.vis_size, bev_range=args.vis_range)
        cv2.imwrite(str(vis_out / f"{pf.stem}.png"), bev)
        written += 1

    return {"num_frames_visualized": written, "vis_output_dir": str(vis_out)}


def main() -> int:
    args = parse_args()
    project_root = Path.cwd().resolve()

    outputs: dict[str, Any] = {"mode": args.mode}

    if args.mode in ("infer", "infer_and_evaluate"):
        outputs["inference"] = run_inference(args, project_root)

    if args.mode in ("evaluate", "infer_and_evaluate"):
        outputs["evaluation"] = evaluate_predictions(args, project_root)

    if args.mode == "visualize":
        outputs["visualization"] = run_visualization(args)

    print("[PASS] run_lidar_det complete")
    print(json.dumps(outputs, indent=2))
    if "evaluation" in outputs:
        print(f"[PASS] evaluation saved: {args.eval_output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
