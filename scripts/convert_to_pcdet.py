#!/usr/bin/env python3
"""Convert CARLA recorder outputs to OpenPCDet custom dataset format."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

CLASS_NAMES = ("Vehicle", "Pedestrian", "Cyclist")


@dataclass
class BoxLabel:
    x: float
    y: float
    z: float
    dx: float
    dy: float
    dz: float
    heading: float
    class_name: str


@dataclass(frozen=True)
class SourceSelection:
    input_dir: Path
    num_common_frames: int
    stems: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert CARLA LiDAR+metadata to OpenPCDet custom dataset format. "
            "Supports merging multiple raw datasets in source order."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Single CARLA raw recording directory (legacy option)",
    )
    parser.add_argument(
        "--input-dirs",
        type=Path,
        nargs="+",
        default=None,
        help="Multiple CARLA raw recording directories to merge in-order",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/pcdet_format"), help="Output dataset root")
    parser.add_argument(
        "--num-frames",
        type=int,
        default=0,
        help="Maximum number of frames per input dataset to convert (0 = all)",
    )
    parser.add_argument("--start-index", type=int, default=0, help="Start index in each sorted source frame list")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    parser.add_argument("--shuffle-split", action="store_true", help="Shuffle frame IDs before split")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output directory if exists")
    parser.add_argument(
        "--sensor-config",
        type=Path,
        default=Path("configs/sensor_config.yaml"),
        help="Sensor YAML config to read LiDAR transform defaults",
    )
    parser.add_argument(
        "--point-cloud-range",
        type=float,
        nargs=6,
        default=[-75.2, -75.2, -2.0, 75.2, 75.2, 4.0],
        metavar=("X_MIN", "Y_MIN", "Z_MIN", "X_MAX", "Y_MAX", "Z_MAX"),
        help="PCDet point cloud range for filtering labels",
    )
    parser.add_argument(
        "--max-range",
        type=float,
        default=75.0,
        help="Optional radial max range (meters) for actor label filtering",
    )
    parser.add_argument(
        "--min-points-per-class",
        type=str,
        nargs="*",
        default=[],
        metavar="CLASS:MIN",
        help=(
            "Optional minimum LiDAR points per GT box to keep label, e.g. "
            "'Vehicle:1 Pedestrian:1 Cyclist:1'. "
            "Classes omitted default to 0."
        ),
    )
    parser.add_argument(
        "--flip-point-y",
        action="store_true",
        default=True,
        help="Flip LiDAR point y-axis (CARLA y-right -> PCDet y-left). Default: enabled.",
    )
    parser.add_argument(
        "--no-flip-point-y",
        action="store_false",
        dest="flip_point_y",
        help="Disable LiDAR point y-axis flip (debug/compatibility only).",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path to save conversion summary JSON (default: <output-dir>/conversion_summary.json)",
    )
    parser.add_argument(
        "--manifest-json",
        type=Path,
        default=None,
        help="Optional path to save source manifest JSON (default: <output-dir>/conversion_manifest.json)",
    )
    parser.add_argument(
        "--generate-custom-infos",
        action="store_true",
        help="Generate custom_infos_{train,val}.pkl and custom_dbinfos_train.pkl in output-dir",
    )
    return parser.parse_args()


def parse_min_points_per_class(values: list[str]) -> dict[str, int]:
    out = {name: 0 for name in CLASS_NAMES}
    for item in values:
        text = str(item).strip()
        if not text:
            continue
        if ":" not in text:
            raise ValueError(f"Invalid --min-points-per-class entry '{item}', expected CLASS:MIN")
        cls, raw_min = text.split(":", 1)
        cls = cls.strip()
        if cls not in CLASS_NAMES:
            raise ValueError(f"Unknown class '{cls}' in --min-points-per-class (expected one of {CLASS_NAMES})")
        try:
            min_pts = int(raw_min.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid min points '{raw_min}' for class '{cls}'") from exc
        if min_pts < 0:
            raise ValueError(f"Min points must be >=0 for class '{cls}'")
        out[cls] = min_pts
    return out


def collect_common_stems(input_dir: Path) -> list[str]:
    lidar_dir = input_dir / "lidar"
    if not lidar_dir.is_dir():
        raise FileNotFoundError(f"Expected subdirectory 'lidar' in {input_dir}")

    lidar_stems = {p.stem for p in lidar_dir.glob("*.npy")}
    if not lidar_stems:
        raise RuntimeError(f"No LiDAR .npy files found in {lidar_dir}")

    rgb_dir = input_dir / "rgb"
    sem_dir = input_dir / "semantic"
    meta_dir = input_dir / "metadata"

    candidates = [lidar_stems]
    if rgb_dir.is_dir():
        candidates.append({p.stem for p in rgb_dir.glob("*.png")})
    if sem_dir.is_dir():
        candidates.append({p.stem for p in sem_dir.glob("*.png")})
    if meta_dir.is_dir():
        candidates.append({p.stem for p in meta_dir.glob("*.json")})

    common = sorted(set.intersection(*candidates))
    if not common:
        raise RuntimeError("No common frame stems among required inputs")
    return common


def resolve_input_dirs(args: argparse.Namespace) -> list[Path]:
    if args.input_dirs and args.input_dir:
        raise ValueError("Use either --input-dir or --input-dirs, not both")
    if not args.input_dirs and not args.input_dir:
        raise ValueError("Either --input-dir or --input-dirs is required")
    if args.input_dirs:
        return [p.resolve() for p in args.input_dirs]
    return [args.input_dir.resolve()]


def select_sources(input_dirs: list[Path], start_index: int, num_frames: int) -> list[SourceSelection]:
    selected: list[SourceSelection] = []
    for input_dir in input_dirs:
        common = collect_common_stems(input_dir)
        if start_index >= len(common):
            raise RuntimeError(
                f"start-index={start_index} is out of range for {input_dir} (num_common_frames={len(common)})"
            )
        if num_frames > 0:
            stems = common[start_index : start_index + num_frames]
        else:
            stems = common[start_index:]
        if not stems:
            raise RuntimeError(f"No frames selected from {input_dir}")
        selected.append(SourceSelection(input_dir=input_dir, num_common_frames=len(common), stems=stems))
    return selected


def prepare_output_dirs(output_dir: Path, overwrite: bool) -> tuple[Path, Path, Path]:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    points_dir = output_dir / "points"
    labels_dir = output_dir / "labels"
    imagesets_dir = output_dir / "ImageSets"
    points_dir.mkdir(parents=True, exist_ok=False)
    labels_dir.mkdir(parents=True, exist_ok=False)
    imagesets_dir.mkdir(parents=True, exist_ok=False)
    return points_dir, labels_dir, imagesets_dir


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def normalize_angle(rad: float) -> float:
    while rad > math.pi:
        rad -= 2.0 * math.pi
    while rad < -math.pi:
        rad += 2.0 * math.pi
    return rad


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


def world_to_local(point_w: np.ndarray, origin_w: np.ndarray, origin_rpy_deg: tuple[float, float, float]) -> np.ndarray:
    r = rot_matrix_from_rpy_deg(*origin_rpy_deg)
    return r.T @ (point_w - origin_w)


def map_actor_class(actor: dict[str, Any]) -> str | None:
    text_fields = [
        str(actor.get("type_id", "")),
        str(actor.get("class_name", "")),
        str(actor.get("label", "")),
        str(actor.get("name", "")),
    ]
    blob = " ".join(text_fields).lower()

    if "pedestrian" in blob or "walker" in blob:
        return "Pedestrian"
    if "cyclist" in blob or "bicycle" in blob or "bike" in blob or "motorcycle" in blob:
        return "Cyclist"
    if "vehicle" in blob or "car" in blob or "truck" in blob or "bus" in blob:
        return "Vehicle"

    semantic_tag = actor.get("semantic_tag")
    if semantic_tag == 4:
        return "Pedestrian"
    if semantic_tag == 10:
        return "Vehicle"
    return None


def extract_actor_records(meta: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("actors", "actor_states", "objects", "tracked_actors", "nearby_actors", "actor_bboxes"):
        value = meta.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]
    return []


def maybe_dict_xyz(obj: Any) -> tuple[float, float, float] | None:
    if not isinstance(obj, dict):
        return None
    if not all(k in obj for k in ("x", "y", "z")):
        return None
    return float(obj["x"]), float(obj["y"]), float(obj["z"])


def parse_box_lidar_direct(actor: dict[str, Any], point_cloud_range: np.ndarray) -> BoxLabel | None:
    keys = ("box_lidar", "bbox_lidar", "gt_box_lidar")
    arr = None
    for key in keys:
        value = actor.get(key)
        if isinstance(value, (list, tuple)) and len(value) >= 7:
            arr = value
            break
    if arr is None:
        return None

    class_name = map_actor_class(actor)
    if class_name is None:
        cname = str(actor.get("class_name", ""))
        class_name = cname if cname in CLASS_NAMES else None
    if class_name is None:
        return None

    x, y, z, dx, dy, dz, heading = [float(v) for v in arr[:7]]
    if dx <= 0 or dy <= 0 or dz <= 0:
        return None
    if not inside_range(np.array([x, y, z], dtype=np.float64), point_cloud_range):
        return None
    return BoxLabel(x=x, y=y, z=z, dx=dx, dy=dy, dz=dz, heading=normalize_angle(heading), class_name=class_name)


def inside_range(center_xyz: np.ndarray, point_cloud_range: np.ndarray) -> bool:
    x, y, z = center_xyz.tolist()
    return (
        point_cloud_range[0] <= x <= point_cloud_range[3]
        and point_cloud_range[1] <= y <= point_cloud_range[4]
        and point_cloud_range[2] <= z <= point_cloud_range[5]
    )


def parse_label_from_world_actor(
    actor: dict[str, Any],
    ego_state: dict[str, Any],
    lidar_tf: dict[str, float],
    point_cloud_range: np.ndarray,
    max_range: float,
) -> BoxLabel | None:
    class_name = map_actor_class(actor)
    if class_name is None:
        return None

    # Actor pose in world coordinates
    actor_loc_dict = actor.get("location") or actor.get("world_location") or actor.get("center")
    actor_rot_dict = actor.get("rotation") or actor.get("world_rotation") or {}

    actor_loc = maybe_dict_xyz(actor_loc_dict)
    ego_loc = maybe_dict_xyz(ego_state.get("location"))
    if actor_loc is None or ego_loc is None:
        return None

    ego_rot = ego_state.get("rotation") or {}
    ego_rpy = (
        float(ego_rot.get("roll", 0.0)),
        float(ego_rot.get("pitch", 0.0)),
        float(ego_rot.get("yaw", 0.0)),
    )

    lidar_offset = np.array([lidar_tf["x"], lidar_tf["y"], lidar_tf["z"]], dtype=np.float64)
    ego_rmat = rot_matrix_from_rpy_deg(*ego_rpy)
    lidar_origin_w = np.array(ego_loc, dtype=np.float64) + ego_rmat @ lidar_offset

    lidar_rpy = (
        ego_rpy[0] + float(lidar_tf["roll"]),
        ego_rpy[1] + float(lidar_tf["pitch"]),
        ego_rpy[2] + float(lidar_tf["yaw"]),
    )

    center_lidar_carla = world_to_local(np.array(actor_loc, dtype=np.float64), lidar_origin_w, lidar_rpy)

    # CARLA -> PCDet conversion: flip y-axis
    center_pcdet = center_lidar_carla.copy()
    center_pcdet[1] = -center_pcdet[1]

    if not inside_range(center_pcdet, point_cloud_range):
        return None
    if max_range > 0:
        if float(np.linalg.norm(center_pcdet[:2])) > max_range:
            return None

    bbox = actor.get("bounding_box") or actor.get("bbox") or {}
    extent_dict = bbox.get("extent") if isinstance(bbox, dict) else None
    size_dict = bbox.get("size") if isinstance(bbox, dict) else None

    dx = dy = dz = None
    if isinstance(extent_dict, dict) and all(k in extent_dict for k in ("x", "y", "z")):
        dx = 2.0 * float(extent_dict["x"])
        dy = 2.0 * float(extent_dict["y"])
        dz = 2.0 * float(extent_dict["z"])
    elif isinstance(size_dict, dict) and all(k in size_dict for k in ("x", "y", "z")):
        dx = float(size_dict["x"])
        dy = float(size_dict["y"])
        dz = float(size_dict["z"])

    if dx is None or dy is None or dz is None or dx <= 0 or dy <= 0 or dz <= 0:
        return None

    actor_yaw = float(actor_rot_dict.get("yaw", 0.0))
    lidar_yaw_world = lidar_rpy[2]
    rel_yaw_carla = math.radians(actor_yaw - lidar_yaw_world)

    # Y flip changes rotation handedness
    heading_pcdet = normalize_angle(-rel_yaw_carla)

    return BoxLabel(
        x=float(center_pcdet[0]),
        y=float(center_pcdet[1]),
        z=float(center_pcdet[2]),
        dx=float(dx),
        dy=float(dy),
        dz=float(dz),
        heading=float(heading_pcdet),
        class_name=class_name,
    )


def write_labels_file(path: Path, labels: list[BoxLabel]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in labels:
            f.write(
                f"{item.x:.6f} {item.y:.6f} {item.z:.6f} "
                f"{item.dx:.6f} {item.dy:.6f} {item.dz:.6f} "
                f"{item.heading:.6f} {item.class_name}\n"
            )


def parse_labels_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = path.read_text(encoding="utf-8").strip().splitlines() if path.exists() else []
    boxes = []
    names = []
    for line in lines:
        tokens = line.strip().split()
        if len(tokens) != 8:
            continue
        boxes.append([float(x) for x in tokens[:7]])
        names.append(tokens[7])

    if not boxes:
        return np.zeros((0, 7), dtype=np.float32), np.array([], dtype="<U1")
    return np.asarray(boxes, dtype=np.float32), np.asarray(names)


def points_in_oriented_box(points_xyz: np.ndarray, box: np.ndarray) -> np.ndarray:
    cx, cy, cz, dx, dy, dz, heading = [float(v) for v in box.tolist()]
    rel = points_xyz - np.array([cx, cy, cz], dtype=np.float32)
    c, s = math.cos(-heading), math.sin(-heading)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    rel_xy = rel[:, :2] @ rot.T
    cond = (
        (np.abs(rel_xy[:, 0]) <= dx / 2.0)
        & (np.abs(rel_xy[:, 1]) <= dy / 2.0)
        & (np.abs(rel[:, 2]) <= dz / 2.0)
    )
    return cond


def generate_custom_infos(output_dir: Path, train_ids: list[str], val_ids: list[str]) -> dict[str, Any]:
    points_dir = output_dir / "points"
    labels_dir = output_dir / "labels"

    def make_infos(ids: list[str]) -> list[dict[str, Any]]:
        infos = []
        for sample_id in ids:
            boxes, names = parse_labels_file(labels_dir / f"{sample_id}.txt")
            info = {
                "point_cloud": {"num_features": 4, "lidar_idx": sample_id},
                "annos": {"name": names, "gt_boxes_lidar": boxes},
            }
            infos.append(info)
        return infos

    train_infos = make_infos(train_ids)
    val_infos = make_infos(val_ids)

    train_info_path = output_dir / "custom_infos_train.pkl"
    val_info_path = output_dir / "custom_infos_val.pkl"
    with train_info_path.open("wb") as f:
        pickle.dump(train_infos, f)
    with val_info_path.open("wb") as f:
        pickle.dump(val_infos, f)

    # Build lightweight gt_database and dbinfos for compatibility.
    gt_db_dir = output_dir / "gt_database"
    gt_db_dir.mkdir(parents=True, exist_ok=True)
    dbinfos: dict[str, list[dict[str, Any]]] = {k: [] for k in CLASS_NAMES}

    for info in train_infos:
        sample_id = info["point_cloud"]["lidar_idx"]
        points = np.load(points_dir / f"{sample_id}.npy")
        boxes = info["annos"]["gt_boxes_lidar"]
        names = info["annos"]["name"]

        for obj_idx in range(boxes.shape[0]):
            box = boxes[obj_idx]
            name = str(names[obj_idx])
            if name not in dbinfos:
                continue
            mask = points_in_oriented_box(points[:, :3], box)
            gt_points = points[mask].copy()
            if gt_points.size == 0:
                continue
            gt_points[:, :3] -= box[:3]
            db_name = f"{sample_id}_{name}_{obj_idx}.bin"
            db_path = gt_db_dir / db_name
            gt_points.tofile(db_path)
            dbinfos[name].append(
                {
                    "name": name,
                    "path": str(Path("gt_database") / db_name),
                    "gt_idx": int(obj_idx),
                    "box3d_lidar": box,
                    "num_points_in_gt": int(gt_points.shape[0]),
                }
            )

    dbinfo_path = output_dir / "custom_dbinfos_train.pkl"
    with dbinfo_path.open("wb") as f:
        pickle.dump(dbinfos, f)

    return {
        "train_info_path": str(train_info_path),
        "val_info_path": str(val_info_path),
        "dbinfo_path": str(dbinfo_path),
        "db_counts": {k: len(v) for k, v in dbinfos.items()},
    }


def write_split_file(path: Path, frame_ids: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for frame_id in frame_ids:
            f.write(f"{frame_id}\n")


def class_histogram_template() -> dict[str, int]:
    return {name: 0 for name in CLASS_NAMES}


def main() -> int:
    args = parse_args()
    input_dirs = resolve_input_dirs(args)
    output_dir = args.output_dir.resolve()
    min_points_per_class = parse_min_points_per_class(args.min_points_per_class)

    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train-ratio must be in (0, 1)")
    if args.start_index < 0:
        raise ValueError("--start-index must be >= 0")
    if args.num_frames < 0:
        raise ValueError("--num-frames must be >= 0")

    sources = select_sources(
        input_dirs=input_dirs,
        start_index=args.start_index,
        num_frames=args.num_frames,
    )

    points_dir, labels_dir, imagesets_dir = prepare_output_dirs(output_dir, args.overwrite)

    config = read_yaml(args.sensor_config.resolve())
    lidar_tf_cfg = (
        config.get("sensors", {})
        .get("lidar", {})
        .get("transform", {})
        if isinstance(config, dict)
        else {}
    )
    lidar_tf = {
        "x": float(lidar_tf_cfg.get("x", 0.0)),
        "y": float(lidar_tf_cfg.get("y", 0.0)),
        "z": float(lidar_tf_cfg.get("z", 0.0)),
        "roll": float(lidar_tf_cfg.get("roll", 0.0)),
        "pitch": float(lidar_tf_cfg.get("pitch", 0.0)),
        "yaw": float(lidar_tf_cfg.get("yaw", 0.0)),
    }

    point_cloud_range = np.asarray(args.point_cloud_range, dtype=np.float64)
    nan_frames = 0
    empty_label_frames = 0
    total_labels = 0
    class_histogram = class_histogram_template()
    class_histogram_removed_by_points = class_histogram_template()
    labels_removed_by_points = 0

    converted = 0
    manifest_sources = []

    for source in sources:
        manifest_sources.append(
            {
                "input_dir": str(source.input_dir),
                "num_source_common_frames": int(source.num_common_frames),
                "num_selected_frames": int(len(source.stems)),
                "selected_first_stem": source.stems[0],
                "selected_last_stem": source.stems[-1],
            }
        )

        for stem in source.stems:
            out_id = f"{converted:06d}"

            src_lidar = source.input_dir / "lidar" / f"{stem}.npy"
            points = np.load(src_lidar)
            if points.ndim != 2 or points.shape[1] < 4:
                raise RuntimeError(f"Unexpected LiDAR shape {points.shape} in {src_lidar}")

            points = points[:, :4].astype(np.float32, copy=False)
            if args.flip_point_y:
                points[:, 1] = -points[:, 1]
            finite_mask = np.all(np.isfinite(points), axis=1)
            if not np.all(finite_mask):
                nan_frames += 1
                points = points[finite_mask]

            np.save(points_dir / f"{out_id}.npy", points)

            labels: list[BoxLabel] = []
            meta_path = source.input_dir / "metadata" / f"{stem}.json"
            if meta_path.is_file():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                actors = extract_actor_records(meta)
                ego_state = meta.get("ego_vehicle") if isinstance(meta.get("ego_vehicle"), dict) else {}

                for actor in actors:
                    direct = parse_box_lidar_direct(actor, point_cloud_range)
                    if direct is not None:
                        labels.append(direct)
                        continue

                    world_box = parse_label_from_world_actor(
                        actor=actor,
                        ego_state=ego_state,
                        lidar_tf=lidar_tf,
                        point_cloud_range=point_cloud_range,
                        max_range=float(args.max_range),
                    )
                    if world_box is not None:
                        labels.append(world_box)

            # Optional GT cleanup: keep boxes supported by at least N LiDAR points per class.
            if labels:
                filtered: list[BoxLabel] = []
                xyz = points[:, :3]
                for lbl in labels:
                    min_pts = int(min_points_per_class.get(lbl.class_name, 0))
                    if min_pts <= 0:
                        filtered.append(lbl)
                        continue
                    box = np.array([lbl.x, lbl.y, lbl.z, lbl.dx, lbl.dy, lbl.dz, lbl.heading], dtype=np.float32)
                    num_pts = int(points_in_oriented_box(xyz, box).sum())
                    if num_pts >= min_pts:
                        filtered.append(lbl)
                    else:
                        labels_removed_by_points += 1
                        class_histogram_removed_by_points[lbl.class_name] = (
                            class_histogram_removed_by_points.get(lbl.class_name, 0) + 1
                        )
                labels = filtered

            labels.sort(key=lambda x: x.class_name)
            total_labels += len(labels)
            if len(labels) == 0:
                empty_label_frames += 1
            for lbl in labels:
                class_histogram[lbl.class_name] = class_histogram.get(lbl.class_name, 0) + 1

            write_labels_file(labels_dir / f"{out_id}.txt", labels)
            converted += 1

    frame_ids = [f"{i:06d}" for i in range(converted)]
    split_ids = frame_ids.copy()
    if args.shuffle_split:
        rng = random.Random(args.seed)
        rng.shuffle(split_ids)

    train_count = int(len(split_ids) * args.train_ratio)
    train_ids = split_ids[:train_count]
    val_ids = split_ids[train_count:]

    write_split_file(imagesets_dir / "train.txt", train_ids)
    write_split_file(imagesets_dir / "val.txt", val_ids)

    sample_points = np.load(points_dir / "000000.npy")
    if sample_points.dtype != np.float32 or sample_points.ndim != 2 or sample_points.shape[1] != 4:
        raise RuntimeError(f"Converted sample points invalid: dtype={sample_points.dtype}, shape={sample_points.shape}")

    infos_summary = None
    if args.generate_custom_infos:
        infos_summary = generate_custom_infos(output_dir, train_ids, val_ids)

    summary = {
        "input_dir": str(input_dirs[0]) if len(input_dirs) == 1 else None,
        "input_dirs": [str(p) for p in input_dirs],
        "output_dir": str(output_dir),
        "num_source_datasets": len(sources),
        "num_source_common_frames_total": int(sum(s.num_common_frames for s in sources)),
        "num_converted_frames": converted,
        "start_index": args.start_index,
        "num_frames_requested_per_source": args.num_frames,
        "train_ratio": args.train_ratio,
        "train_frames": len(train_ids),
        "val_frames": len(val_ids),
        "shuffle_split": bool(args.shuffle_split),
        "seed": args.seed,
        "point_cloud_range": point_cloud_range.tolist(),
        "lidar_transform_from_config": lidar_tf,
        "frames_with_filtered_non_finite_points": nan_frames,
        "frames_with_empty_labels": empty_label_frames,
        "total_labels_written": total_labels,
        "empty_label_ratio": (empty_label_frames / converted) if converted > 0 else None,
        "label_class_histogram": class_histogram,
        "min_points_per_class": min_points_per_class,
        "labels_removed_by_min_points_total": int(labels_removed_by_points),
        "labels_removed_by_min_points_per_class": class_histogram_removed_by_points,
        "flip_point_y": bool(args.flip_point_y),
        "sample_points_shape": list(sample_points.shape),
        "sample_points_dtype": str(sample_points.dtype),
        "infos_summary": infos_summary,
    }

    summary_path = args.summary_json.resolve() if args.summary_json else output_dir / "conversion_summary.json"
    manifest_path = args.manifest_json.resolve() if args.manifest_json else output_dir / "conversion_manifest.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    manifest = {
        "output_dir": str(output_dir),
        "flip_point_y": bool(args.flip_point_y),
        "sources": manifest_sources,
        "frame_id_policy": "Output frame IDs are regenerated sequentially from 000000 in merged source order.",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("[PASS] PCDet conversion complete")
    print(json.dumps(summary, indent=2))
    print(f"[PASS] Summary saved: {summary_path}")
    print(f"[PASS] Manifest saved: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
