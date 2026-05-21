#!/usr/bin/env python3
"""Render a Stage 14 simulator demo video from a live planning rollout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

CARLA_010_PALETTE_BGR: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),  # Unlabeled
    1: (128, 64, 128),  # Road
    2: (232, 35, 244),  # Sidewalk
    3: (70, 70, 70),  # Building
    4: (156, 102, 102),  # Wall
    5: (40, 40, 100),  # Fence
    6: (153, 153, 153),  # Pole
    7: (30, 170, 250),  # Traffic light
    8: (0, 220, 220),  # Traffic sign
    9: (35, 142, 107),  # Vegetation
    10: (152, 251, 152),  # Terrain
    11: (180, 130, 70),  # Sky
    12: (60, 20, 220),  # Pedestrian
    13: (0, 0, 255),  # Rider
    14: (142, 0, 0),  # Car
    15: (70, 0, 0),  # Truck
    16: (100, 60, 0),  # Bus
    17: (100, 80, 0),  # Train
    18: (230, 0, 0),  # Motorcycle
    19: (32, 11, 119),  # Bicycle
    20: (160, 190, 110),  # Static
    21: (170, 120, 150),  # Dynamic
    22: (55, 90, 80),  # Other
    23: (150, 100, 100),  # Water / fallback bridge color
    24: (50, 234, 157),  # Road line
    25: (0, 81, 81),  # Ground
    26: (150, 100, 100),  # Bridge
    27: (140, 150, 230),  # Rail track
    28: (180, 165, 180),  # Guard rail
}

ROAD_TAGS = {1, 7, 24}
VEHICLE_TAGS = {10, 14, 15, 16}
WALKER_TAGS = {4, 12}
ACTOR_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "vehicle": (0, 220, 0),
    "walker": (0, 0, 240),
    "cyclist": (0, 220, 220),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Stage 14 simulator demo video")
    parser.add_argument("--rollout-dir", type=Path, required=True, help="Rollout directory with rgb/semantic/lidar/metadata")
    parser.add_argument("--output-video", type=Path, required=True, help="Output MP4 path")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--max-frames", type=int, default=0, help="Limit frame count (0=all)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quality-report-json", type=Path, help="Write visual-quality audit JSON")
    parser.add_argument("--min-semantic-tags", type=int, default=0, help="Fail unless this many semantic tags appear")
    parser.add_argument("--require-road", action="store_true", help="Fail unless road/roadline semantic tags appear")
    parser.add_argument("--min-vehicle-frames", type=int, default=0, help="Fail unless vehicle evidence appears in this many frames")
    parser.add_argument("--min-walker-frames", type=int, default=0, help="Fail unless pedestrian evidence appears in this many frames")
    parser.add_argument("--reject-stuck", action="store_true", help="Fail if metadata marks rollout stuck")
    return parser.parse_args()


def collect_stems(path: Path, pattern: str) -> set[str]:
    if not path.is_dir():
        raise FileNotFoundError(f"Directory not found: {path}")
    return {p.stem for p in path.glob(pattern)}


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def draw_title(panel: np.ndarray, title: str) -> None:
    cv2.putText(panel, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)


def semantic_tags_from_image(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.uint8)
    if img.ndim == 3 and img.shape[2] >= 3:
        blue = img[:, :, 0]
        green = img[:, :, 1]
        red = img[:, :, 2]
        if int(np.count_nonzero(blue)) == 0 and int(np.count_nonzero(green)) == 0:
            return red.astype(np.uint8)
    return cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.uint8)


def colorize_semantic_tags(tags: np.ndarray) -> np.ndarray:
    tags_u8 = tags.astype(np.uint8)
    view = np.zeros((*tags_u8.shape[:2], 3), dtype=np.uint8)
    for tag, color in CARLA_010_PALETTE_BGR.items():
        view[tags_u8 == tag] = color
    unknown = ~np.isin(tags_u8, np.array(list(CARLA_010_PALETTE_BGR), dtype=np.uint8))
    if np.any(unknown):
        fallback = cv2.applyColorMap(tags_u8, cv2.COLORMAP_TURBO)
        view[unknown] = fallback[unknown]
    return view


def render_semantic(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    view = colorize_semantic_tags(semantic_tags_from_image(img))
    return cv2.resize(view, size, interpolation=cv2.INTER_NEAREST)


def classify_actor_record(actor: dict[str, Any]) -> str | None:
    blob = " ".join(
        str(actor.get(key, ""))
        for key in ("class_name", "type_id", "label", "name")
    ).lower()
    if "pedestrian" in blob or "walker" in blob:
        return "walker"
    if "cyclist" in blob or "bicycle" in blob or "bike" in blob or "motorcycle" in blob:
        return "cyclist"
    if "vehicle" in blob or "car" in blob or "truck" in blob or "bus" in blob or "taxi" in blob:
        return "vehicle"

    semantic_tag = actor.get("semantic_tag")
    if semantic_tag in WALKER_TAGS:
        return "walker"
    if semantic_tag in VEHICLE_TAGS:
        return "vehicle"
    return None


def actor_counts(meta: dict[str, Any]) -> dict[str, int]:
    counts = {"vehicle": 0, "walker": 0, "cyclist": 0}
    actors = meta.get("actors", [])
    if not isinstance(actors, list):
        return counts
    for actor in actors:
        if not isinstance(actor, dict):
            continue
        cls = classify_actor_record(actor)
        if cls in counts:
            counts[cls] += 1
    return counts


def evaluate_visual_quality(semantic_tags: list[np.ndarray], metadata: list[dict[str, Any]]) -> dict[str, Any]:
    unique_tags: set[int] = set()
    road_frames = 0
    vehicle_semantic_frames = 0
    walker_semantic_frames = 0
    per_frame_tag_counts: list[int] = []

    for tags in semantic_tags:
        frame_tags = set(map(int, np.unique(tags)))
        unique_tags.update(frame_tags)
        per_frame_tag_counts.append(len(frame_tags))
        road_frames += bool(frame_tags & ROAD_TAGS)
        vehicle_semantic_frames += bool(frame_tags & VEHICLE_TAGS)
        walker_semantic_frames += bool(frame_tags & WALKER_TAGS)

    metadata_vehicle_frames = 0
    metadata_walker_frames = 0
    metadata_cyclist_frames = 0
    vehicle_counts: list[int] = []
    walker_counts: list[int] = []
    cyclist_counts: list[int] = []
    stuck_frames = 0
    done_reasons: set[str] = set()
    progress_values: list[float] = []

    for meta in metadata:
        counts = actor_counts(meta)
        vehicle_counts.append(counts["vehicle"])
        walker_counts.append(counts["walker"])
        cyclist_counts.append(counts["cyclist"])
        metadata_vehicle_frames += counts["vehicle"] > 0
        metadata_walker_frames += counts["walker"] > 0
        metadata_cyclist_frames += counts["cyclist"] > 0

        telemetry = meta.get("telemetry", {}) if isinstance(meta.get("telemetry"), dict) else {}
        wm = meta.get("world_model", {}) if isinstance(meta.get("world_model"), dict) else {}
        stuck = as_bool(telemetry.get("stuck", False)) or wm.get("done_reason") == "stuck"
        stuck_frames += bool(stuck)
        if wm.get("done_reason"):
            done_reasons.add(str(wm["done_reason"]))
        if "progress_m" in telemetry:
            try:
                progress_values.append(float(telemetry["progress_m"]))
            except (TypeError, ValueError):
                pass

    frames = max(len(semantic_tags), len(metadata))
    return {
        "frames": frames,
        "semantic_tag_count": len(unique_tags),
        "semantic_tags": sorted(unique_tags),
        "avg_semantic_tags_per_frame": float(np.mean(per_frame_tag_counts)) if per_frame_tag_counts else 0.0,
        "road_frames": int(road_frames),
        "vehicle_semantic_frames": int(vehicle_semantic_frames),
        "walker_semantic_frames": int(walker_semantic_frames),
        "metadata_vehicle_frames": int(metadata_vehicle_frames),
        "metadata_walker_frames": int(metadata_walker_frames),
        "metadata_cyclist_frames": int(metadata_cyclist_frames),
        "vehicle_evidence_frames": int(metadata_vehicle_frames),
        "walker_evidence_frames": int(metadata_walker_frames),
        "avg_metadata_vehicle_count": float(np.mean(vehicle_counts)) if vehicle_counts else 0.0,
        "avg_metadata_walker_count": float(np.mean(walker_counts)) if walker_counts else 0.0,
        "avg_metadata_cyclist_count": float(np.mean(cyclist_counts)) if cyclist_counts else 0.0,
        "stuck_frames": int(stuck_frames),
        "done_reasons": sorted(done_reasons),
        "max_progress_m": max(progress_values) if progress_values else 0.0,
    }


def check_visual_quality(
    report: dict[str, Any],
    *,
    min_semantic_tags: int = 0,
    require_road: bool = False,
    min_vehicle_frames: int = 0,
    min_walker_frames: int = 0,
    reject_stuck: bool = False,
) -> list[str]:
    failures: list[str] = []
    if min_semantic_tags > 0 and int(report["semantic_tag_count"]) < min_semantic_tags:
        failures.append(
            f"semantic tag diversity too low: {report['semantic_tag_count']} < {min_semantic_tags}"
        )
    if require_road and int(report["road_frames"]) <= 0:
        failures.append("road/roadline semantic evidence missing")
    if min_vehicle_frames > 0 and int(report["vehicle_evidence_frames"]) < min_vehicle_frames:
        failures.append(
            f"vehicle metadata evidence frames too low: {report['vehicle_evidence_frames']} < {min_vehicle_frames}"
        )
    if min_walker_frames > 0 and int(report["walker_evidence_frames"]) < min_walker_frames:
        failures.append(
            f"walker metadata evidence frames too low: {report['walker_evidence_frames']} < {min_walker_frames}"
        )
    if reject_stuck and int(report["stuck_frames"]) > 0:
        failures.append(f"rollout marked stuck in {report['stuck_frames']} frame(s)")
    return failures


def write_quality_report(path: Path, report: dict[str, Any], failures: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(report)
    payload["quality_failures"] = failures
    payload["quality_passed"] = not failures
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def float_from_nested(data: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(data.get(key, default))
    except (TypeError, ValueError):
        return default


def actor_local_xy(actor: dict[str, Any], ego_vehicle: dict[str, Any]) -> tuple[float, float] | None:
    actor_loc = actor.get("location", {})
    ego_loc = ego_vehicle.get("location", {})
    ego_rot = ego_vehicle.get("rotation", {})
    if not isinstance(actor_loc, dict) or not isinstance(ego_loc, dict) or not isinstance(ego_rot, dict):
        return None

    dx = float_from_nested(actor_loc, "x") - float_from_nested(ego_loc, "x")
    dy = float_from_nested(actor_loc, "y") - float_from_nested(ego_loc, "y")
    yaw = np.deg2rad(float_from_nested(ego_rot, "yaw"))
    forward = float(np.cos(yaw) * dx + np.sin(yaw) * dy)
    right = float(-np.sin(yaw) * dx + np.cos(yaw) * dy)
    return forward, right


def draw_actor_bev_overlay(
    canvas: np.ndarray,
    meta: dict[str, Any] | None,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> None:
    if not isinstance(meta, dict):
        return
    ego_vehicle = meta.get("ego_vehicle", {})
    actors = meta.get("actors", [])
    if not isinstance(ego_vehicle, dict) or not isinstance(actors, list):
        return

    height, width = canvas.shape[:2]
    for actor in actors:
        if not isinstance(actor, dict):
            continue
        cls = classify_actor_record(actor)
        if cls not in ACTOR_COLORS_BGR:
            continue
        local = actor_local_xy(actor, ego_vehicle)
        if local is None:
            continue
        x, y = local
        if not (x_min <= x <= x_max and y_min <= y <= y_max):
            continue

        u = int((y - y_min) / (y_max - y_min) * (width - 1))
        v = int(height - 1 - ((x - x_min) / (x_max - x_min) * (height - 1)))
        color = ACTOR_COLORS_BGR[cls]
        if cls == "vehicle":
            cv2.rectangle(canvas, (u - 6, v - 4), (u + 6, v + 4), color, -1)
            label = "V"
        elif cls == "walker":
            cv2.circle(canvas, (u, v), 5, color, -1)
            label = "P"
        else:
            pts = np.array([[u, v - 6], [u + 6, v], [u, v + 6], [u - 6, v]], dtype=np.int32)
            cv2.fillConvexPoly(canvas, pts, color)
            label = "C"
        cv2.putText(canvas, label, (u + 7, v + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def render_lidar_bev(points: np.ndarray, size: tuple[int, int], meta: dict[str, Any] | None = None) -> np.ndarray:
    width, height = size
    canvas = np.full((height, width, 3), 18, dtype=np.uint8)
    x_min, x_max = -15.0, 45.0
    y_min, y_max = -30.0, 30.0

    for x_m in range(-10, 41, 10):
        px = int((x_m - x_min) / (x_max - x_min) * (height - 1))
        y = height - 1 - px
        cv2.line(canvas, (0, y), (width - 1, y), (45, 45, 45), 1, cv2.LINE_AA)
    for y_m in range(-30, 31, 10):
        x = int((y_m - y_min) / (y_max - y_min) * (width - 1))
        cv2.line(canvas, (x, 0), (x, height - 1), (45, 45, 45), 1, cv2.LINE_AA)

    if points.ndim == 2 and points.shape[1] >= 3 and points.shape[0] > 0:
        xyz = points[:, :3].astype(np.float32)
        mask = np.isfinite(xyz).all(axis=1)
        xyz = xyz[mask]
        if xyz.size > 0:
            x = xyz[:, 0]
            y = xyz[:, 1]
            z = xyz[:, 2]
            in_range = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
            x = x[in_range]
            y = y[in_range]
            z = z[in_range]
            if x.size > 0:
                u = ((y - y_min) / (y_max - y_min) * (width - 1)).astype(np.int32)
                v = (height - 1 - ((x - x_min) / (x_max - x_min) * (height - 1))).astype(np.int32)
                z_norm = np.clip((z + 2.5) / 5.0, 0.0, 1.0)
                colors = (cv2.applyColorMap((z_norm * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)).reshape(-1, 3)
                canvas[v, u] = colors

    draw_actor_bev_overlay(canvas, meta, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

    ego = np.array(
        [
            [width // 2, int(height * 0.80)],
            [width // 2 - 10, int(height * 0.88)],
            [width // 2 + 10, int(height * 0.88)],
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(canvas, ego, (255, 255, 255))
    return canvas


def plot_trace(
    panel: np.ndarray,
    values: list[float],
    x0: int,
    y0: int,
    width: int,
    height: int,
    color: tuple[int, int, int],
) -> None:
    if len(values) < 2:
        return
    pts: list[tuple[int, int]] = []
    for idx, value in enumerate(values):
        x = int(x0 + (idx / max(1, len(values) - 1)) * (width - 1))
        v = float(np.clip(value, -1.0, 1.0))
        y = int(y0 + (1.0 - (v + 1.0) * 0.5) * (height - 1))
        pts.append((x, y))
    for p0, p1 in zip(pts[:-1], pts[1:]):
        cv2.line(panel, p0, p1, color, 2, cv2.LINE_AA)


def render_status_panel(
    size: tuple[int, int],
    frame_idx: int,
    total_frames: int,
    stem: str,
    meta: dict[str, Any],
    steer_hist: list[float],
    throttle_hist: list[float],
    brake_hist: list[float],
) -> np.ndarray:
    width, height = size
    panel = np.full((height, width, 3), 24, dtype=np.uint8)
    draw_title(panel, "Planner Control + Reward State")

    telemetry = meta.get("telemetry", {}) if isinstance(meta.get("telemetry"), dict) else {}
    control = meta.get("control", {}) if isinstance(meta.get("control"), dict) else {}
    wm = meta.get("world_model", {}) if isinstance(meta.get("world_model"), dict) else {}

    steer = float(control.get("steer", 0.0))
    throttle = float(control.get("throttle", 0.0))
    brake = float(control.get("brake", 0.0))
    speed = float(telemetry.get("speed_kmh", 0.0))
    progress = float(telemetry.get("progress_m", 0.0))
    reward = float(wm.get("reward", 0.0))
    done_reason = wm.get("done_reason")
    collision = float(telemetry.get("collision_intensity", 0.0))
    lane_inv = as_bool(telemetry.get("lane_invasion", False))
    offroad = as_bool(telemetry.get("offroad", False))
    stuck = as_bool(telemetry.get("stuck", False))
    counts = actor_counts(meta)

    lines = [
        f"Frame: {frame_idx + 1}/{total_frames}  (stem {stem})",
        f"Action [steer throttle brake]: [{steer:+.3f} {throttle:.3f} {brake:.3f}]",
        f"Speed: {speed:6.2f} km/h    Progress: {progress:7.2f} m",
        f"Reward: {reward:+.4f}    Done reason: {done_reason if done_reason else '-'}",
        f"Collision: {collision:.3f}    LaneInv: {lane_inv}    Offroad: {offroad}    Stuck: {stuck}",
        f"Actors: vehicles={counts['vehicle']} walkers={counts['walker']} cyclists={counts['cyclist']}",
    ]
    y = 58
    for text in lines:
        cv2.putText(panel, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (230, 230, 230), 1, cv2.LINE_AA)
        y += 34

    actors = meta.get("actors", [])
    if isinstance(actors, list) and actors:
        nearest = sorted(
            (a for a in actors if isinstance(a, dict)),
            key=lambda a: float(a.get("distance_to_ego_m", 1e9)),
        )[:3]
        for actor in nearest:
            cls = classify_actor_record(actor) or "actor"
            dist = actor.get("distance_to_ego_m", "?")
            try:
                dist_text = f"{float(dist):.1f}m"
            except (TypeError, ValueError):
                dist_text = "?m"
            cv2.putText(
                panel,
                f"Nearest {cls}: {dist_text}  {actor.get('type_id', '-')}",
                (18, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.54,
                (205, 205, 205),
                1,
                cv2.LINE_AA,
            )
            y += 26

    chart_x, chart_y = 18, 278
    chart_w, chart_h = width - 36, height - chart_y - 32
    cv2.rectangle(panel, (chart_x, chart_y), (chart_x + chart_w, chart_y + chart_h), (96, 96, 96), 1)
    mid_y = chart_y + chart_h // 2
    cv2.line(panel, (chart_x, mid_y), (chart_x + chart_w, mid_y), (60, 60, 60), 1, cv2.LINE_AA)

    plot_trace(panel, steer_hist, chart_x, chart_y, chart_w, chart_h, (255, 215, 0))
    thr_trace = [v * 2.0 - 1.0 for v in throttle_hist]
    brk_trace = [v * 2.0 - 1.0 for v in brake_hist]
    plot_trace(panel, thr_trace, chart_x, chart_y, chart_w, chart_h, (60, 220, 60))
    plot_trace(panel, brk_trace, chart_x, chart_y, chart_w, chart_h, (60, 60, 240))

    cv2.putText(panel, "steer", (24, chart_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 215, 0), 2, cv2.LINE_AA)
    cv2.putText(panel, "throttle", (104, chart_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 220, 60), 2, cv2.LINE_AA)
    cv2.putText(panel, "brake", (228, chart_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 240), 2, cv2.LINE_AA)
    cv2.putText(panel, "-1", (chart_x + 2, chart_y + chart_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(panel, "+1", (chart_x + 2, chart_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    return panel


def main() -> int:
    args = parse_args()
    rollout_dir = args.rollout_dir.resolve()
    rgb_dir = rollout_dir / "rgb"
    semantic_dir = rollout_dir / "semantic"
    lidar_dir = rollout_dir / "lidar"
    metadata_dir = rollout_dir / "metadata"

    rgb_stems = collect_stems(rgb_dir, "*.png")
    sem_stems = collect_stems(semantic_dir, "*.png")
    lidar_stems = collect_stems(lidar_dir, "*.npy")
    meta_stems = collect_stems(metadata_dir, "*.json")
    stems = sorted(rgb_stems & sem_stems & lidar_stems & meta_stems)
    if not stems:
        raise RuntimeError("No aligned stems found among rgb/semantic/lidar/metadata")
    if args.max_frames > 0:
        stems = stems[: args.max_frames]

    quality_requested = (
        args.quality_report_json is not None
        or args.min_semantic_tags > 0
        or args.require_road
        or args.min_vehicle_frames > 0
        or args.min_walker_frames > 0
        or args.reject_stuck
    )
    quality_report: dict[str, Any] | None = None
    quality_failures: list[str] = []
    if quality_requested:
        semantic_tag_frames: list[np.ndarray] = []
        metadata_frames: list[dict[str, Any]] = []
        for stem in stems:
            semantic = cv2.imread(str(semantic_dir / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
            if semantic is None:
                raise RuntimeError(f"Failed to load semantic frame for stem {stem}")
            semantic_tag_frames.append(semantic_tags_from_image(semantic))
            metadata_frames.append(load_json(metadata_dir / f"{stem}.json"))

        quality_report = evaluate_visual_quality(semantic_tag_frames, metadata_frames)
        quality_failures = check_visual_quality(
            quality_report,
            min_semantic_tags=args.min_semantic_tags,
            require_road=args.require_road,
            min_vehicle_frames=args.min_vehicle_frames,
            min_walker_frames=args.min_walker_frames,
            reject_stuck=args.reject_stuck,
        )
        if args.quality_report_json:
            write_quality_report(args.quality_report_json.resolve(), quality_report, quality_failures)
        if quality_failures:
            joined = "; ".join(quality_failures)
            raise RuntimeError(f"Visual quality gate failed: {joined}")

    panel_w = 960
    panel_h = 540
    frame_size = (panel_w * 2, panel_h * 2)

    args.output_video = args.output_video.resolve()
    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    if args.output_video.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output_video}. Pass --overwrite")
    if args.output_video.exists() and args.overwrite:
        args.output_video.unlink()

    writer = cv2.VideoWriter(
        str(args.output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.fps),
        frame_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {args.output_video}")

    steer_hist: list[float] = []
    throttle_hist: list[float] = []
    brake_hist: list[float] = []
    history_len = 60

    try:
        for idx, stem in enumerate(stems):
            rgb = cv2.imread(str(rgb_dir / f"{stem}.png"), cv2.IMREAD_COLOR)
            semantic = cv2.imread(str(semantic_dir / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
            points = np.load(lidar_dir / f"{stem}.npy")
            meta = load_json(metadata_dir / f"{stem}.json")
            if rgb is None or semantic is None:
                raise RuntimeError(f"Failed to load frame assets for stem {stem}")

            control = meta.get("control", {}) if isinstance(meta.get("control"), dict) else {}
            steer_hist.append(float(control.get("steer", 0.0)))
            throttle_hist.append(float(control.get("throttle", 0.0)))
            brake_hist.append(float(control.get("brake", 0.0)))
            steer_hist = steer_hist[-history_len:]
            throttle_hist = throttle_hist[-history_len:]
            brake_hist = brake_hist[-history_len:]

            rgb_panel = cv2.resize(rgb, (panel_w, panel_h), interpolation=cv2.INTER_LINEAR)
            sem_panel = render_semantic(semantic, (panel_w, panel_h))
            bev_panel = render_lidar_bev(points, (panel_w, panel_h), meta=meta)
            status_panel = render_status_panel(
                size=(panel_w, panel_h),
                frame_idx=idx,
                total_frames=len(stems),
                stem=stem,
                meta=meta,
                steer_hist=steer_hist,
                throttle_hist=throttle_hist,
                brake_hist=brake_hist,
            )

            draw_title(rgb_panel, "RGB Camera")
            draw_title(sem_panel, "Semantic / Lane / Road View")
            draw_title(bev_panel, "LiDAR BEV")

            top = np.concatenate([rgb_panel, sem_panel], axis=1)
            bottom = np.concatenate([bev_panel, status_panel], axis=1)
            canvas = np.concatenate([top, bottom], axis=0)
            writer.write(canvas)
    finally:
        writer.release()

    if not args.output_video.is_file() or args.output_video.stat().st_size <= 0:
        raise RuntimeError(f"Output video missing or empty: {args.output_video}")

    print("[PASS] Stage 14 simulator demo video rendered")
    print(f"frames: {len(stems)}")
    print(f"output: {args.output_video}")
    if quality_report is not None:
        print(f"semantic_tags: {quality_report['semantic_tags']}")
        print(
            "quality: "
            f"road_frames={quality_report['road_frames']} "
            f"vehicle_evidence_frames={quality_report['vehicle_evidence_frames']} "
            f"walker_evidence_frames={quality_report['walker_evidence_frames']} "
            f"stuck_frames={quality_report['stuck_frames']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
