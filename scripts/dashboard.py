#!/usr/bin/env python3
"""Stage 09 dashboard compositor: RGB + segmentation + LiDAR detections."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


CITYSCAPES_COLORS_RGB = np.array(
    [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ],
    dtype=np.uint8,
)
CITYSCAPES_COLORS_BGR = CITYSCAPES_COLORS_RGB[:, ::-1]

DET_COLORS_BGR = {
    "Vehicle": (0, 200, 0),
    "Pedestrian": (0, 0, 220),
    "Cyclist": (0, 165, 255),
}


@dataclass(frozen=True)
class Detection:
    x: float
    y: float
    z: float
    dx: float
    dy: float
    dz: float
    heading: float
    score: float
    class_name: str


@dataclass(frozen=True)
class SensorRig:
    fx: float
    fy: float
    cx: float
    cy: float
    image_w: int
    image_h: int
    cam_tf: dict[str, float]
    lidar_tf: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Stage 09 dashboard frames/video from canonical outputs")
    parser.add_argument("--mode", choices=("frames", "video", "both", "world_model"), default="both", help="Output mode")
    parser.add_argument("--rgb-dir", type=Path, default=Path("data/raw/recording_001/rgb"), help="RGB frame directory")
    parser.add_argument("--seg-dir", type=Path, default=Path("output/segmentation/predictions"), help="Segmentation prediction directory")
    parser.add_argument(
        "--det-dir",
        type=Path,
        default=Path("output/detection_3d/predictions"),
        help="Detection prediction directory or parent containing predictions/",
    )
    parser.add_argument("--lidar-dir", type=Path, default=Path("data/raw/recording_001/lidar"), help="LiDAR .npy directory")
    parser.add_argument("--metadata-dir", type=Path, default=Path("data/raw/recording_001/metadata"), help="Metadata directory")
    parser.add_argument("--calib-json", type=Path, default=Path("data/raw/recording_001/calib/sensors.json"), help="Calibration JSON")
    parser.add_argument("--output", type=Path, default=Path("output/dashboard/dashboard_video.mp4"), help="Output MP4 path")
    parser.add_argument("--output-dir", type=Path, default=Path("output/dashboard/frames"), help="Output frame directory")
    parser.add_argument("--report-json", type=Path, default=Path("output/dashboard/dashboard_report.json"), help="Run report JSON")
    parser.add_argument("--fps", type=float, default=10.0, help="Video FPS")
    parser.add_argument("--num-frames", type=int, default=0, help="Max frame count (0 = all)")
    parser.add_argument("--start-index", type=int, default=0, help="Start index in sorted stems")
    parser.add_argument("--width", type=int, default=1280, help="Dashboard width")
    parser.add_argument("--height", type=int, default=720, help="Dashboard height")
    parser.add_argument("--seg-alpha", type=float, default=0.35, help="Segmentation overlay alpha")
    parser.add_argument("--det-score-thresh", type=float, default=0.1, help="Score threshold for rendering detections")
    parser.add_argument("--bev-range", type=float, default=75.2, help="BEV range in meters")
    parser.add_argument("--max-dets-per-frame", type=int, default=80, help="Max rendered detections per frame")
    parser.add_argument("--allow-partial-alignment", action="store_true", help="Use stem intersection instead of strict equality")
    parser.add_argument("--skip-projection", action="store_true", help="Skip 3D projection onto camera view")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite frame/video outputs")
    parser.add_argument("--wm-current-rgb-dir", type=Path, default=None, help="Stage 13 current RGB directory")
    parser.add_argument("--wm-recon-dir", type=Path, default=None, help="Stage 13 VAE reconstruction directory")
    parser.add_argument("--wm-dream-dir", type=Path, default=None, help="Stage 13 dream frame directory")
    parser.add_argument("--wm-metrics-json", type=Path, default=None, help="Stage 13 metrics JSON for text panel")
    return parser.parse_args()


def rot_matrix_from_rpy_deg(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)

    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)

    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def resolve_prediction_dir(det_dir: Path) -> Path:
    det_dir = det_dir.resolve()
    pred_sub = det_dir / "predictions"
    if pred_sub.is_dir():
        return pred_sub
    return det_dir


def collect_stems(path: Path, pattern: str) -> set[str]:
    if not path.is_dir():
        raise FileNotFoundError(f"Directory not found: {path}")
    return {p.stem for p in path.glob(pattern)}


def select_stems(args: argparse.Namespace, det_pred_dir: Path) -> tuple[list[str], dict[str, Any]]:
    rgb_stems = collect_stems(args.rgb_dir.resolve(), "*.png")
    seg_stems = collect_stems(args.seg_dir.resolve(), "*.png")
    det_stems = collect_stems(det_pred_dir, "*.txt")
    meta_stems = collect_stems(args.metadata_dir.resolve(), "*.json")

    if not args.allow_partial_alignment:
        if not (rgb_stems == seg_stems == det_stems == meta_stems):
            detail = {
                "rgb": len(rgb_stems),
                "seg": len(seg_stems),
                "det": len(det_stems),
                "meta": len(meta_stems),
                "only_rgb": sorted(rgb_stems - seg_stems - det_stems)[:10],
                "only_seg": sorted(seg_stems - rgb_stems - det_stems)[:10],
                "only_det": sorted(det_stems - rgb_stems - seg_stems)[:10],
                "only_meta": sorted(meta_stems - rgb_stems - seg_stems)[:10],
            }
            raise RuntimeError(f"Strict alignment failed: {detail}")
        stems = sorted(rgb_stems)
    else:
        stems = sorted(rgb_stems & seg_stems & det_stems & meta_stems)
        if not stems:
            raise RuntimeError("No aligned stems after intersection")

    if args.start_index < 0 or args.start_index >= len(stems):
        raise RuntimeError(f"start-index out of range: {args.start_index}, available={len(stems)}")

    stems = stems[args.start_index :]
    if args.num_frames > 0:
        stems = stems[: args.num_frames]

    alignment = {
        "rgb_count": len(rgb_stems),
        "seg_count": len(seg_stems),
        "det_count": len(det_stems),
        "meta_count": len(meta_stems),
        "selected_count": len(stems),
        "allow_partial_alignment": bool(args.allow_partial_alignment),
    }
    return stems, alignment


def parse_detection_file(path: Path, score_thresh: float) -> list[Detection]:
    out: list[Detection] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 9:
            continue
        x, y, z, dx, dy, dz, heading, score = map(float, parts[:8])
        class_name = parts[8]
        if score < score_thresh:
            continue
        out.append(Detection(x=x, y=y, z=z, dx=dx, dy=dy, dz=dz, heading=heading, score=score, class_name=class_name))
    out.sort(key=lambda d: d.score, reverse=True)
    return out


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    valid = (mask >= 0) & (mask < CITYSCAPES_COLORS_BGR.shape[0])
    out[valid] = CITYSCAPES_COLORS_BGR[mask[valid]]
    return out


def extract_sensor_tf(sensor_block: dict[str, Any], default_xyz: tuple[float, float, float]) -> dict[str, float]:
    tf = sensor_block.get("transform") if isinstance(sensor_block, dict) else {}
    tf = tf if isinstance(tf, dict) else {}
    return {
        "x": float(tf.get("x", default_xyz[0])),
        "y": float(tf.get("y", default_xyz[1])),
        "z": float(tf.get("z", default_xyz[2])),
        "roll": float(tf.get("roll", 0.0)),
        "pitch": float(tf.get("pitch", 0.0)),
        "yaw": float(tf.get("yaw", 0.0)),
    }


def load_sensor_rig(calib_json: Path, rgb_shape: tuple[int, int]) -> SensorRig:
    data = read_json(calib_json)
    sensors = data.get("sensors") if isinstance(data.get("sensors"), dict) else {}

    rgb_sensor = sensors.get("rgb_camera") if isinstance(sensors.get("rgb_camera"), dict) else {}
    lidar_sensor = sensors.get("lidar") if isinstance(sensors.get("lidar"), dict) else {}

    intr = rgb_sensor.get("intrinsics") if isinstance(rgb_sensor.get("intrinsics"), dict) else {}
    attrs = rgb_sensor.get("attributes") if isinstance(rgb_sensor.get("attributes"), dict) else {}

    h, w = rgb_shape
    fov = float(intr.get("fov_deg", attrs.get("fov", 110.0)))
    fx_fallback = w / (2.0 * math.tan(math.radians(fov) / 2.0))

    fx = float(intr.get("fx", fx_fallback))
    fy = float(intr.get("fy", fx_fallback))
    cx = float(intr.get("cx", w / 2.0))
    cy = float(intr.get("cy", h / 2.0))

    cam_tf = extract_sensor_tf(rgb_sensor, default_xyz=(1.5, 0.0, 2.4))
    lidar_tf = extract_sensor_tf(lidar_sensor, default_xyz=(0.0, 0.0, 2.5))

    return SensorRig(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        image_w=w,
        image_h=h,
        cam_tf=cam_tf,
        lidar_tf=lidar_tf,
    )


def pcdet_box_corners(det: Detection) -> np.ndarray:
    dx2 = det.dx / 2.0
    dy2 = det.dy / 2.0
    dz2 = det.dz / 2.0

    corners = np.array(
        [
            [dx2, dy2, -dz2],
            [dx2, -dy2, -dz2],
            [-dx2, -dy2, -dz2],
            [-dx2, dy2, -dz2],
            [dx2, dy2, dz2],
            [dx2, -dy2, dz2],
            [-dx2, -dy2, dz2],
            [-dx2, dy2, dz2],
        ],
        dtype=np.float64,
    )

    c, s = math.cos(det.heading), math.sin(det.heading)
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    rotated = corners @ rz.T
    rotated[:, 0] += det.x
    rotated[:, 1] += det.y
    rotated[:, 2] += det.z
    return rotated


def pcdet_to_carla_lidar(points_xyz: np.ndarray) -> np.ndarray:
    out = points_xyz.copy()
    out[:, 1] = -out[:, 1]
    return out


def lidar_to_camera(points_lidar: np.ndarray, rig: SensorRig) -> np.ndarray:
    r_l = rot_matrix_from_rpy_deg(rig.lidar_tf["roll"], rig.lidar_tf["pitch"], rig.lidar_tf["yaw"])
    t_l = np.array([rig.lidar_tf["x"], rig.lidar_tf["y"], rig.lidar_tf["z"]], dtype=np.float64)

    r_c = rot_matrix_from_rpy_deg(rig.cam_tf["roll"], rig.cam_tf["pitch"], rig.cam_tf["yaw"])
    t_c = np.array([rig.cam_tf["x"], rig.cam_tf["y"], rig.cam_tf["z"]], dtype=np.float64)

    points_ego = points_lidar @ r_l.T + t_l.reshape(1, 3)
    points_cam = (points_ego - t_c.reshape(1, 3)) @ r_c
    return points_cam


def project_camera(points_cam: np.ndarray, rig: SensorRig) -> tuple[np.ndarray, np.ndarray]:
    depth = points_cam[:, 0]
    valid = depth > 0.2

    u = np.zeros_like(depth)
    v = np.zeros_like(depth)

    u[valid] = rig.fx * (points_cam[valid, 1] / depth[valid]) + rig.cx
    v[valid] = rig.fy * (-points_cam[valid, 2] / depth[valid]) + rig.cy
    pix = np.stack([u, v], axis=1)
    return pix, valid


def draw_projected_box(img_bgr: np.ndarray, det: Detection, rig: SensorRig, color: tuple[int, int, int]) -> bool:
    corners_pcdet = pcdet_box_corners(det)
    corners_lidar = pcdet_to_carla_lidar(corners_pcdet)
    corners_cam = lidar_to_camera(corners_lidar, rig)
    pixels, valid = project_camera(corners_cam, rig)

    if np.count_nonzero(valid) < 8:
        return False

    h, w = img_bgr.shape[:2]
    pts = pixels.astype(np.int32)
    # Allow slightly out-of-image points to keep near-edge boxes.
    if np.any(pts[:, 0] < -w) or np.any(pts[:, 0] > 2 * w) or np.any(pts[:, 1] < -h) or np.any(pts[:, 1] > 2 * h):
        return False

    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    for i0, i1 in edges:
        p0 = (int(pts[i0, 0]), int(pts[i0, 1]))
        p1 = (int(pts[i1, 0]), int(pts[i1, 1]))
        cv2.line(img_bgr, p0, p1, color, 2, cv2.LINE_AA)

    front_center = np.mean(pts[[0, 1, 4, 5]], axis=0).astype(np.int32)
    rear_center = np.mean(pts[[2, 3, 6, 7]], axis=0).astype(np.int32)
    cv2.arrowedLine(
        img_bgr,
        (int(rear_center[0]), int(rear_center[1])),
        (int(front_center[0]), int(front_center[1])),
        color,
        2,
        cv2.LINE_AA,
        tipLength=0.25,
    )
    return True


def render_bev(points: np.ndarray, dets: list[Detection], bev_size: int, bev_range: float) -> np.ndarray:
    canvas = np.zeros((bev_size, bev_size, 3), dtype=np.uint8)

    if points.ndim == 2 and points.shape[1] >= 3 and points.shape[0] > 0:
        xy = points[:, :2]
        mask = (
            (xy[:, 0] >= -bev_range)
            & (xy[:, 0] <= bev_range)
            & (xy[:, 1] >= -bev_range)
            & (xy[:, 1] <= bev_range)
        )
        pts = points[mask]
        if pts.shape[0] > 0:
            if pts.shape[0] > 50000:
                stride = int(math.ceil(pts.shape[0] / 50000))
                pts = pts[::stride]

            z = pts[:, 2]
            zmin = float(np.percentile(z, 5))
            zmax = float(np.percentile(z, 95))
            if zmax <= zmin:
                zmax = zmin + 1e-3
            zn = np.clip((z - zmin) / (zmax - zmin), 0.0, 1.0)
            colors = (zn * 255.0).astype(np.uint8)

            u = ((pts[:, 1] + bev_range) / (2.0 * bev_range) * (bev_size - 1)).astype(np.int32)
            v = ((bev_range - pts[:, 0]) / (2.0 * bev_range) * (bev_size - 1)).astype(np.int32)
            valid = (u >= 0) & (u < bev_size) & (v >= 0) & (v < bev_size)
            u = u[valid]
            v = v[valid]
            c = colors[valid]
            canvas[v, u] = np.stack([c, 255 - c, np.full_like(c, 90)], axis=1)

    axis_color = (60, 60, 60)
    cv2.line(canvas, (bev_size // 2, 0), (bev_size // 2, bev_size - 1), axis_color, 1)
    cv2.line(canvas, (0, bev_size // 2), (bev_size - 1, bev_size // 2), axis_color, 1)

    for det in dets:
        color = DET_COLORS_BGR.get(det.class_name, (0, 255, 255))

        x = det.x
        y = -det.y  # PCDet y-left -> CARLA y-right
        heading = -det.heading
        dx2 = det.dx / 2.0
        dy2 = det.dy / 2.0

        local = np.array(
            [[dx2, dy2], [dx2, -dy2], [-dx2, -dy2], [-dx2, dy2], [dx2, dy2]],
            dtype=np.float64,
        )
        c, s = math.cos(heading), math.sin(heading)
        rot = np.array([[c, -s], [s, c]], dtype=np.float64)
        world = local @ rot.T + np.array([x, y], dtype=np.float64)

        pu = ((world[:, 1] + bev_range) / (2.0 * bev_range) * (bev_size - 1)).astype(np.int32)
        pv = ((bev_range - world[:, 0]) / (2.0 * bev_range) * (bev_size - 1)).astype(np.int32)
        poly = np.stack([pu, pv], axis=1)

        cv2.polylines(canvas, [poly], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)

        front = np.array([dx2, 0.0], dtype=np.float64) @ rot.T + np.array([x, y], dtype=np.float64)
        cu = int((y + bev_range) / (2.0 * bev_range) * (bev_size - 1))
        cvv = int((bev_range - x) / (2.0 * bev_range) * (bev_size - 1))
        fu = int((front[1] + bev_range) / (2.0 * bev_range) * (bev_size - 1))
        fv = int((bev_range - front[0]) / (2.0 * bev_range) * (bev_size - 1))
        cv2.arrowedLine(canvas, (cu, cvv), (fu, fv), color, 1, cv2.LINE_AA, tipLength=0.3)

    return canvas


def fit_image(src: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    out = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    if src is None or src.size == 0:
        return out
    h, w = src.shape[:2]
    scale = min(target_w / max(w, 1), target_h / max(h, 1))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_LINEAR)
    y0 = (target_h - nh) // 2
    x0 = (target_w - nw) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = resized
    return out


def draw_text_block(img: np.ndarray, lines: list[str], origin: tuple[int, int], color: tuple[int, int, int] = (230, 230, 230)) -> None:
    x, y = origin
    for i, line in enumerate(lines):
        cv2.putText(img, line, (x, y + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


def compose_dashboard(
    rgb_bgr: np.ndarray,
    seg_mask: np.ndarray,
    dets: list[Detection],
    points: np.ndarray,
    frame_meta: dict[str, Any],
    rig: SensorRig,
    frame_idx: int,
    frame_total: int,
    stem: str,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    h, w = rgb_bgr.shape[:2]
    if seg_mask.shape[:2] != (h, w):
        seg_mask = cv2.resize(seg_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    seg_color = colorize_mask(seg_mask)
    overlay = cv2.addWeighted(rgb_bgr, 1.0 - args.seg_alpha, seg_color, args.seg_alpha, 0.0)

    rendered_proj = 0
    if not args.skip_projection:
        for det in dets[: args.max_dets_per_frame]:
            color = DET_COLORS_BGR.get(det.class_name, (0, 255, 255))
            if draw_projected_box(overlay, det=det, rig=rig, color=color):
                rendered_proj += 1

    bev = render_bev(points, dets[: args.max_dets_per_frame], bev_size=320, bev_range=args.bev_range)

    canvas = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    top_h = int(round(args.height * 0.64))
    left_w = int(round(args.width * 0.75))
    right_w = args.width - left_w
    bottom_h = args.height - top_h

    main_view = fit_image(overlay, left_w, top_h)
    canvas[:top_h, :left_w] = main_view

    info = np.full((top_h, right_w, 3), 18, dtype=np.uint8)
    cv2.rectangle(info, (0, 0), (right_w - 1, top_h - 1), (50, 50, 50), 1)

    counts = {"Vehicle": 0, "Pedestrian": 0, "Cyclist": 0}
    for det in dets:
        counts[det.class_name] = counts.get(det.class_name, 0) + 1

    timestamp = frame_meta.get("timestamp")
    ts_text = f"{float(timestamp):.3f}s" if isinstance(timestamp, (int, float)) else "n/a"
    lines = [
        "STAGE 09 DASHBOARD",
        f"Frame: {frame_idx + 1}/{frame_total}",
        f"Stem: {stem}",
        f"Timestamp: {ts_text}",
        f"Detections: {len(dets)}",
        f"Projected: {rendered_proj}",
        f"Vehicle: {counts.get('Vehicle', 0)}",
        f"Pedestrian: {counts.get('Pedestrian', 0)}",
        f"Cyclist: {counts.get('Cyclist', 0)}",
        f"FPS target: {args.fps:.1f}",
        "Model: ERFNet + PointPillar",
    ]
    draw_text_block(info, lines, (14, 36))

    legend_y = top_h - 110
    for label, color in (("Vehicle", DET_COLORS_BGR["Vehicle"]), ("Pedestrian", DET_COLORS_BGR["Pedestrian"]), ("Cyclist", DET_COLORS_BGR["Cyclist"])):
        cv2.rectangle(info, (16, legend_y - 14), (36, legend_y + 6), color, -1)
        cv2.putText(info, label, (44, legend_y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
        legend_y += 28

    canvas[:top_h, left_w:] = info

    left_bottom_w = args.width // 2
    right_bottom_w = args.width - left_bottom_w
    panel_bg = np.full((bottom_h, left_bottom_w, 3), 12, dtype=np.uint8)
    cv2.putText(panel_bg, "LiDAR BEV + 3D Boxes", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (230, 230, 230), 2, cv2.LINE_AA)
    bev_fit = fit_image(bev, left_bottom_w - 20, bottom_h - 42)
    panel_bg[34 : 34 + bev_fit.shape[0], 10 : 10 + bev_fit.shape[1]] = bev_fit
    canvas[top_h:, :left_bottom_w] = panel_bg

    right_panel = np.full((bottom_h, right_bottom_w, 3), 10, dtype=np.uint8)
    half_w = right_bottom_w // 2
    raw_fit = fit_image(rgb_bgr, half_w - 16, bottom_h - 36)
    seg_fit = fit_image(seg_color, right_bottom_w - half_w - 16, bottom_h - 36)

    cv2.putText(right_panel, "RGB", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 2, cv2.LINE_AA)
    cv2.putText(right_panel, "Segmentation", (half_w + 12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 2, cv2.LINE_AA)

    right_panel[32 : 32 + raw_fit.shape[0], 8 : 8 + raw_fit.shape[1]] = raw_fit
    right_panel[32 : 32 + seg_fit.shape[0], half_w + 8 : half_w + 8 + seg_fit.shape[1]] = seg_fit

    canvas[top_h:, left_bottom_w:] = right_panel

    stats = {
        "num_dets": len(dets),
        "num_projected": rendered_proj,
        "vehicle": counts.get("Vehicle", 0),
        "pedestrian": counts.get("Pedestrian", 0),
        "cyclist": counts.get("Cyclist", 0),
    }
    return canvas, stats


def create_video_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    candidates = ("avc1", "mp4v", "XVID")
    for code in candidates:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*code), fps, size)
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError("Failed to open video writer with codecs: avc1, mp4v, XVID")


def world_model_stems(current_rgb_dir: Path, recon_dir: Path, dream_dir: Path) -> list[str]:
    rgb = {p.stem for p in current_rgb_dir.glob("*.png")}
    rec = {p.stem for p in recon_dir.glob("*.png")}
    dream = {p.stem for p in dream_dir.glob("*.png")}
    stems = sorted(rgb & rec & dream)
    if not stems:
        raise RuntimeError("No aligned world-model frames found among current/recon/dream dirs")
    return stems


def world_model_metrics_lines(metrics_json: Path | None) -> list[str]:
    if metrics_json is None or not metrics_json.exists():
        return ["WORLD MODEL METRICS", "No metrics JSON provided"]
    data = read_json(metrics_json)
    lines = ["WORLD MODEL METRICS"]
    for key in ("mean_planner_pred_reward", "mean_random_pred_reward", "planner_beats_random", "episodes", "horizon", "generations"):
        if key in data:
            lines.append(f"{key}: {data[key]}")
    if len(lines) == 1:
        lines.append("Metrics JSON found but keys are missing")
    return lines


def compose_world_model_dashboard(
    current_rgb: np.ndarray,
    reconstruction: np.ndarray,
    dream: np.ndarray,
    lines: list[str],
    args: argparse.Namespace,
    frame_idx: int,
    total_frames: int,
    stem: str,
) -> np.ndarray:
    canvas = np.full((args.height, args.width, 3), 12, dtype=np.uint8)
    half_w = args.width // 2
    half_h = args.height // 2

    panel1 = fit_image(current_rgb, half_w - 20, half_h - 40)
    panel2 = fit_image(reconstruction, half_w - 20, half_h - 40)
    panel3 = fit_image(dream, half_w - 20, half_h - 40)
    panel4 = np.full((half_h, half_w, 3), 20, dtype=np.uint8)

    draw_text_block(panel4, [f"Frame: {frame_idx + 1}/{total_frames}", f"Stem: {stem}", *lines], (12, 28))

    canvas[28 : 28 + panel1.shape[0], 10 : 10 + panel1.shape[1]] = panel1
    canvas[28 : 28 + panel2.shape[0], half_w + 10 : half_w + 10 + panel2.shape[1]] = panel2
    canvas[half_h + 20 : half_h + 20 + panel3.shape[0], 10 : 10 + panel3.shape[1]] = panel3
    canvas[half_h + 20 : half_h + 20 + panel4.shape[0], half_w : half_w + panel4.shape[1]] = panel4

    cv2.putText(canvas, "Current RGB", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 2, cv2.LINE_AA)
    cv2.putText(canvas, "VAE Reconstruction", (half_w + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Dream Future", (10, half_h + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Planner Trace", (half_w + 10, half_h + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 2, cv2.LINE_AA)
    return canvas


def run_world_model_dashboard(args: argparse.Namespace) -> int:
    if args.wm_current_rgb_dir is None or args.wm_recon_dir is None or args.wm_dream_dir is None:
        raise RuntimeError("world_model mode requires --wm-current-rgb-dir --wm-recon-dir --wm-dream-dir")

    current_rgb_dir = args.wm_current_rgb_dir.resolve()
    recon_dir = args.wm_recon_dir.resolve()
    dream_dir = args.wm_dream_dir.resolve()
    args.output = args.output.resolve()
    args.output_dir = args.output_dir.resolve()
    args.report_json = args.report_json.resolve()
    metrics_json = args.wm_metrics_json.resolve() if args.wm_metrics_json is not None else None
    stems = world_model_stems(current_rgb_dir, recon_dir, dream_dir)
    lines = world_model_metrics_lines(metrics_json)

    if args.num_frames > 0:
        stems = stems[: args.num_frames]
    if args.start_index > 0:
        stems = stems[args.start_index :]
    if not stems:
        raise RuntimeError("No world-model stems selected")

    if args.output_dir.exists() and args.overwrite:
        for p in args.output_dir.glob("*.png"):
            p.unlink()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.output.exists() and args.overwrite:
        args.output.unlink()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output video exists: {args.output}. Pass --overwrite")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    writer = create_video_writer(path=args.output, fps=args.fps, size=(args.width, args.height))
    frames_written = 0
    try:
        for i, stem in enumerate(stems):
            rgb = cv2.imread(str(current_rgb_dir / f"{stem}.png"), cv2.IMREAD_COLOR)
            rec = cv2.imread(str(recon_dir / f"{stem}.png"), cv2.IMREAD_COLOR)
            dream = cv2.imread(str(dream_dir / f"{stem}.png"), cv2.IMREAD_COLOR)
            if rgb is None or rec is None or dream is None:
                raise RuntimeError(f"Failed to load world-model panel frame: {stem}")

            frame = compose_world_model_dashboard(
                current_rgb=rgb,
                reconstruction=rec,
                dream=dream,
                lines=lines,
                args=args,
                frame_idx=i,
                total_frames=len(stems),
                stem=stem,
            )
            cv2.imwrite(str(args.output_dir / f"{stem}.png"), frame)
            writer.write(frame)
            frames_written += 1
    finally:
        writer.release()

    report = {
        "mode": "world_model",
        "num_frames": int(frames_written),
        "current_rgb_dir": str(current_rgb_dir),
        "recon_dir": str(recon_dir),
        "dream_dir": str(dream_dir),
        "output_video": str(args.output),
        "output_frames": str(args.output_dir),
        "metrics_json": str(metrics_json) if metrics_json is not None else None,
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"[PASS] world-model dashboard report saved: {args.report_json}")
    return 0


def main() -> int:
    args = parse_args()
    if args.mode == "world_model":
        return run_world_model_dashboard(args)

    args.rgb_dir = args.rgb_dir.resolve()
    args.seg_dir = args.seg_dir.resolve()
    args.det_dir = args.det_dir.resolve()
    args.lidar_dir = args.lidar_dir.resolve()
    args.metadata_dir = args.metadata_dir.resolve()
    args.calib_json = args.calib_json.resolve()
    args.output = args.output.resolve()
    args.output_dir = args.output_dir.resolve()
    args.report_json = args.report_json.resolve()

    det_pred_dir = resolve_prediction_dir(args.det_dir)
    stems, alignment = select_stems(args, det_pred_dir)

    if not stems:
        raise RuntimeError("No stems selected")

    if args.mode in ("frames", "both"):
        if args.output_dir.exists() and args.overwrite:
            for p in args.output_dir.glob("*.png"):
                p.unlink()
        elif args.output_dir.exists() and not args.overwrite:
            if any(args.output_dir.glob("*.png")):
                raise FileExistsError(f"Output frame directory already contains png files: {args.output_dir}")
        args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("video", "both"):
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists() and not args.overwrite:
            raise FileExistsError(f"Output video exists: {args.output}. Pass --overwrite")
        if args.output.exists() and args.overwrite:
            args.output.unlink()

    first_rgb = cv2.imread(str(args.rgb_dir / f"{stems[0]}.png"), cv2.IMREAD_COLOR)
    if first_rgb is None:
        raise RuntimeError(f"Failed to read first RGB frame: {stems[0]}")

    rig = load_sensor_rig(args.calib_json, rgb_shape=first_rgb.shape[:2])

    writer = None
    if args.mode in ("video", "both"):
        writer = create_video_writer(path=args.output, fps=args.fps, size=(args.width, args.height))

    t0 = time.time()
    frames_written = 0
    frames_with_proj = 0
    total_proj = 0
    total_det = 0

    try:
        for i, stem in enumerate(stems):
            rgb = cv2.imread(str(args.rgb_dir / f"{stem}.png"), cv2.IMREAD_COLOR)
            seg = cv2.imread(str(args.seg_dir / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
            if rgb is None:
                raise RuntimeError(f"Failed to read RGB frame: {stem}")
            if seg is None:
                raise RuntimeError(f"Failed to read seg frame: {stem}")
            if seg.ndim == 3:
                seg = seg[:, :, 0]

            dets = parse_detection_file(det_pred_dir / f"{stem}.txt", score_thresh=args.det_score_thresh)
            dets = dets[: args.max_dets_per_frame]

            lidar_path = args.lidar_dir / f"{stem}.npy"
            points = np.load(lidar_path) if lidar_path.exists() else np.zeros((0, 4), dtype=np.float32)
            if points.ndim != 2 or points.shape[1] < 3:
                points = np.zeros((0, 4), dtype=np.float32)

            frame_meta = read_json(args.metadata_dir / f"{stem}.json")

            dashboard, stats = compose_dashboard(
                rgb_bgr=rgb,
                seg_mask=seg,
                dets=dets,
                points=points,
                frame_meta=frame_meta,
                rig=rig,
                frame_idx=i,
                frame_total=len(stems),
                stem=stem,
                args=args,
            )

            total_det += stats["num_dets"]
            total_proj += stats["num_projected"]
            if stats["num_projected"] > 0:
                frames_with_proj += 1

            if args.mode in ("frames", "both"):
                cv2.imwrite(str(args.output_dir / f"{stem}.png"), dashboard)

            if writer is not None:
                writer.write(dashboard)

            frames_written += 1
    finally:
        if writer is not None:
            writer.release()

    elapsed = max(time.time() - t0, 1e-9)
    report = {
        "mode": args.mode,
        "rgb_dir": str(args.rgb_dir),
        "seg_dir": str(args.seg_dir),
        "det_pred_dir": str(det_pred_dir),
        "lidar_dir": str(args.lidar_dir),
        "metadata_dir": str(args.metadata_dir),
        "calib_json": str(args.calib_json),
        "output_video": str(args.output),
        "output_frames": str(args.output_dir),
        "alignment": alignment,
        "num_frames": frames_written,
        "fps_target": float(args.fps),
        "render_fps": float(frames_written / elapsed),
        "total_detections_rendered": int(total_det),
        "total_projected_boxes": int(total_proj),
        "frames_with_projected_boxes": int(frames_with_proj),
        "projection_enabled": not bool(args.skip_projection),
        "sensor_rig": {
            "intrinsics": {
                "fx": rig.fx,
                "fy": rig.fy,
                "cx": rig.cx,
                "cy": rig.cy,
                "width": rig.image_w,
                "height": rig.image_h,
            },
            "camera_transform": rig.cam_tf,
            "lidar_transform": rig.lidar_tf,
        },
    }

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"[PASS] dashboard report saved: {args.report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
