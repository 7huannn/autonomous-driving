#!/usr/bin/env python3
"""Record synchronized RGB, semantic segmentation, and LiDAR data from CARLA."""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Any

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from world_model import ensure_stage13_scope, load_yaml_config
from world_model.reward import (
    RewardConfig,
    TerminalConfig,
    TerminalState,
    advance_terminal_state,
    compute_reward,
    progress_m_from_locations,
    speed_kmh_from_velocity,
    terminal_decision,
)


LOGGER = logging.getLogger("carla_recorder")
SENSOR_NAMES = ("rgb_camera", "semantic_camera", "lidar")


@dataclass
class SensorPacket:
    name: str
    frame: int
    data: Any


@dataclass
class WorldModelRuntime:
    reward_cfg: RewardConfig
    terminal_cfg: TerminalConfig
    collision_intensity_scale: float
    enable_collision_sensor: bool
    enable_lane_invasion_sensor: bool
    mode: str
    policy: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record synchronized RGB, semantic segmentation, and LiDAR frames from CARLA.",
    )
    parser.add_argument(
        "--config",
        default="configs/sensor_config.yaml",
        help="Path to sensor YAML config (default: configs/sensor_config.yaml)",
    )
    parser.add_argument("--output-dir", help="Override recording.output_dir from config")
    parser.add_argument("--num-frames", type=int, help="Override recording.num_frames")
    parser.add_argument("--host", help="Override carla.host")
    parser.add_argument("--port", type=int, help="Override carla.port")
    parser.add_argument("--map", dest="map_name", help="Override carla.map")
    parser.add_argument("--timeout", type=float, help="Override carla.timeout")
    parser.add_argument("--tm-port", type=int, help="Override carla.tm_port")
    parser.add_argument("--width", type=int, help="Override camera image width")
    parser.add_argument("--height", type=int, help="Override camera image height")
    parser.add_argument("--traffic-vehicles", type=int, help="Override traffic vehicle count")
    parser.add_argument("--traffic-walkers", type=int, help="Override traffic walker count")
    parser.add_argument(
        "--prefer-cyclist-vehicles",
        type=int,
        help="Override target number of bike/motorcycle blueprints to spawn in traffic",
    )
    parser.add_argument("--lidar-pps", type=int, help="Override LiDAR points_per_second")
    parser.add_argument(
        "--actor-label-max-distance",
        type=float,
        help="Override max distance (meters) for actor bbox metadata collection",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing output directory before recording.",
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Load and validate config, then exit without connecting to CARLA.",
    )
    parser.add_argument(
        "--world-model-config",
        default="configs/world_model.yaml",
        help="Path to Stage 13 world-model config for reward/terminal + scope-gating",
    )
    parser.add_argument(
        "--allow-stage13-control",
        action="store_true",
        help="Required when world-model stage13.mode is live_planning or iterative.",
    )
    parser.add_argument(
        "--policy",
        default="autopilot_noise",
        help="Policy label written into recording summary and rollout manifest inputs.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for actor spawning")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return config


def deep_get(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def load_world_model_runtime(
    world_model_config_path: Path,
    sensor_config: dict[str, Any],
    allow_stage13_control: bool,
    policy: str,
) -> WorldModelRuntime:
    wm_cfg = load_yaml_config(world_model_config_path)
    mode = ensure_stage13_scope(wm_cfg, allow_stage13_control=allow_stage13_control)

    reward_cfg = wm_cfg.get("reward", {}) if isinstance(wm_cfg.get("reward"), dict) else {}
    terminal_cfg = wm_cfg.get("terminal", {}) if isinstance(wm_cfg.get("terminal"), dict) else {}
    recording_events = deep_get(sensor_config, "recording", "world_model_events", default={}) or {}
    recording_terminal = deep_get(sensor_config, "recording", "world_model_terminal", default={}) or {}

    terminal_merged = dict(terminal_cfg)
    terminal_merged.update({k: v for k, v in recording_terminal.items() if v is not None})

    return WorldModelRuntime(
        reward_cfg=RewardConfig(**{k: reward_cfg.get(k, getattr(RewardConfig(), k)) for k in RewardConfig.__dataclass_fields__}),
        terminal_cfg=TerminalConfig(
            **{k: terminal_merged.get(k, getattr(TerminalConfig(), k)) for k in TerminalConfig.__dataclass_fields__}
        ),
        collision_intensity_scale=float(recording_events.get("collision_intensity_scale", 0.01)),
        enable_collision_sensor=bool(recording_events.get("enable_collision_sensor", False)),
        enable_lane_invasion_sensor=bool(recording_events.get("enable_lane_invasion_sensor", False)),
        mode=mode,
        policy=str(policy),
    )


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = dict(config)
    config.setdefault("carla", {})
    config.setdefault("sensors", {})
    config.setdefault("recording", {})

    scalar_overrides = {
        ("recording", "output_dir"): args.output_dir,
        ("recording", "num_frames"): args.num_frames,
        ("recording", "traffic_vehicles"): args.traffic_vehicles,
        ("recording", "traffic_walkers"): args.traffic_walkers,
        ("recording", "prefer_cyclist_vehicles"): args.prefer_cyclist_vehicles,
        ("recording", "actor_label_max_distance_m"): args.actor_label_max_distance,
        ("carla", "host"): args.host,
        ("carla", "port"): args.port,
        ("carla", "map"): args.map_name,
        ("carla", "timeout"): args.timeout,
        ("carla", "tm_port"): args.tm_port,
    }
    for path, value in scalar_overrides.items():
        if value is None:
            continue
        section, key = path
        config[section][key] = value

    for sensor_name in ("rgb_camera", "semantic_camera"):
        attrs = config["sensors"].setdefault(sensor_name, {}).setdefault("attributes", {})
        if args.width is not None:
            attrs["image_size_x"] = args.width
        if args.height is not None:
            attrs["image_size_y"] = args.height

    if args.lidar_pps is not None:
        lidar_attrs = config["sensors"].setdefault("lidar", {}).setdefault("attributes", {})
        lidar_attrs["points_per_second"] = args.lidar_pps

    return config


def validate_config(config: dict[str, Any]) -> None:
    missing = []
    for top_key in ("carla", "sensors", "recording"):
        if top_key not in config:
            missing.append(top_key)
    for sensor_name in SENSOR_NAMES:
        if deep_get(config, "sensors", sensor_name) is None:
            missing.append(f"sensors.{sensor_name}")
        if deep_get(config, "sensors", sensor_name, "blueprint") is None:
            missing.append(f"sensors.{sensor_name}.blueprint")
    if missing:
        raise ValueError("Config is missing required keys: " + ", ".join(missing))

    num_frames = int(deep_get(config, "recording", "num_frames", default=0))
    if num_frames <= 0:
        raise ValueError("recording.num_frames must be > 0")


def resolve_output_dir(config: dict[str, Any]) -> Path:
    output_dir = Path(str(deep_get(config, "recording", "output_dir", default="data/raw/recording_001")))
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    return output_dir


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists; pass --overwrite: {output_dir}")
        shutil.rmtree(output_dir)
    for subdir in ("rgb", "semantic", "lidar", "metadata", "calib"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)


def _camera_intrinsics_from_attrs(attrs: dict[str, Any]) -> dict[str, float] | None:
    if "image_size_x" not in attrs or "image_size_y" not in attrs or "fov" not in attrs:
        return None
    width = float(attrs["image_size_x"])
    height = float(attrs["image_size_y"])
    fov_deg = float(attrs["fov"])
    if width <= 0 or height <= 0:
        return None
    fx = width / (2.0 * np.tan(np.radians(fov_deg) / 2.0))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    return {
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "width": int(width),
        "height": int(height),
        "fov_deg": float(fov_deg),
    }


def build_sensor_specs(config: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for sensor_name in SENSOR_NAMES:
        sensor_cfg = deep_get(config, "sensors", sensor_name, default={}) or {}
        attrs = dict(sensor_cfg.get("attributes", {}))
        spec = {
            "name": sensor_name,
            "blueprint": sensor_cfg.get("blueprint"),
            "transform": sensor_cfg.get("transform", {}),
            "attributes": attrs,
        }
        if sensor_name in ("rgb_camera", "semantic_camera"):
            intrinsics = _camera_intrinsics_from_attrs(attrs)
            if intrinsics is not None:
                spec["intrinsics"] = intrinsics
        out[sensor_name] = spec
    return out


def write_calibration_file(output_dir: Path, config: dict[str, Any]) -> Path:
    sensors_file = output_dir / "calib" / "sensors.json"
    payload = {
        "coordinate_convention": "CARLA/UE x-forward y-right z-up",
        "sensors": build_sensor_specs(config),
    }
    sensors_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return sensors_file


def write_scenario_file(output_dir: Path, scenario: dict[str, Any]) -> Path:
    scenario_file = output_dir / "scenario.json"
    scenario_file.write_text(json.dumps(scenario, indent=2), encoding="utf-8")
    return scenario_file


def validate_recording_integrity(output_dir: Path, expected_frames: int) -> dict[str, Any]:
    rgb_dir = output_dir / "rgb"
    sem_dir = output_dir / "semantic"
    lidar_dir = output_dir / "lidar"
    meta_dir = output_dir / "metadata"

    rgb_stems = {p.stem for p in rgb_dir.glob("*.png")}
    sem_stems = {p.stem for p in sem_dir.glob("*.png")}
    lidar_stems = {p.stem for p in lidar_dir.glob("*.npy")}
    meta_stems = {p.stem for p in meta_dir.glob("*.json")}
    common_stems = rgb_stems & sem_stems & lidar_stems & meta_stems

    counts = {
        "rgb": len(rgb_stems),
        "semantic": len(sem_stems),
        "lidar": len(lidar_stems),
        "metadata": len(meta_stems),
    }

    aligned = (
        len(rgb_stems) == len(sem_stems) == len(lidar_stems) == len(meta_stems) == len(common_stems)
        and len(common_stems) == expected_frames
    )

    timestamps_monotonic = True
    sensor_sync_all = True
    invalid_meta_files = 0
    missing_sensor_frame_keys = 0
    missing_world_model_fields = 0
    prev_ts = -1.0

    for stem in sorted(common_stems):
        meta_path = meta_dir / f"{stem}.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            ts = float(meta.get("timestamp", -1.0))
            if ts <= prev_ts:
                timestamps_monotonic = False
            prev_ts = ts

            sf = meta.get("sensor_frames", {})
            if not isinstance(sf, dict) or not all(k in sf for k in SENSOR_NAMES):
                missing_sensor_frame_keys += 1
                sensor_sync_all = False
            else:
                values = [int(sf[name]) for name in SENSOR_NAMES]
                if not (values[0] == values[1] == values[2]):
                    sensor_sync_all = False

            control = meta.get("control", {})
            telemetry = meta.get("telemetry", {})
            world_model = meta.get("world_model", {})
            valid_control = isinstance(control, dict) and all(k in control for k in ("steer", "throttle", "brake"))
            valid_telemetry = isinstance(telemetry, dict) and all(
                k in telemetry for k in ("speed_kmh", "progress_m", "collision_intensity", "lane_invasion", "offroad", "stuck")
            )
            valid_world_model = isinstance(world_model, dict) and "reward" in world_model and "done" in world_model
            if not (valid_control and valid_telemetry and valid_world_model):
                missing_world_model_fields += 1
        except Exception:
            invalid_meta_files += 1
            timestamps_monotonic = False
            sensor_sync_all = False

    complete = (
        aligned
        and timestamps_monotonic
        and sensor_sync_all
        and invalid_meta_files == 0
        and missing_sensor_frame_keys == 0
        and missing_world_model_fields == 0
    )
    return {
        "pass": bool(complete),
        "expected_frames": int(expected_frames),
        "counts": counts,
        "num_common_stems": len(common_stems),
        "aligned": bool(aligned),
        "timestamps_monotonic": bool(timestamps_monotonic),
        "sensor_sync_all": bool(sensor_sync_all),
        "invalid_meta_files": int(invalid_meta_files),
        "missing_sensor_frame_keys": int(missing_sensor_frame_keys),
        "missing_world_model_fields": int(missing_world_model_fields),
    }


def import_carla():
    try:
        import carla  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise RuntimeError(f"CARLA Python API import failed: {exc}") from exc
    return carla


def connect_to_carla(carla: Any, host: str, port: int, timeout: float, startup_wait_seconds: float) -> tuple[Any, Any]:
    deadline = time.monotonic() + max(0.0, startup_wait_seconds)
    last_exc: Exception | None = None
    while True:
        try:
            client = carla.Client(host, port)
            client.set_timeout(timeout)
            world = client.get_world()
            return client, world
        except Exception as exc:
            last_exc = exc
            if time.monotonic() >= deadline:
                break
            time.sleep(1.0)
    raise RuntimeError(
        f"Failed to connect to CARLA at {host}:{port} within {startup_wait_seconds:.1f}s: {last_exc}"
    ) from last_exc


def make_transform(carla: Any, transform_config: dict[str, Any] | None) -> Any:
    transform_config = transform_config or {}
    location = carla.Location(
        x=float(transform_config.get("x", 0.0)),
        y=float(transform_config.get("y", 0.0)),
        z=float(transform_config.get("z", 0.0)),
    )
    rotation = carla.Rotation(
        pitch=float(transform_config.get("pitch", 0.0)),
        yaw=float(transform_config.get("yaw", 0.0)),
        roll=float(transform_config.get("roll", 0.0)),
    )
    return carla.Transform(location, rotation)


def set_blueprint_attributes(blueprint: Any, attributes: dict[str, Any]) -> None:
    for key, value in attributes.items():
        blueprint.set_attribute(str(key), str(value))


def snapshot_actor_to_dict(actor_snapshot: Any) -> dict[str, Any]:
    transform = actor_snapshot.get_transform()
    velocity = actor_snapshot.get_velocity()
    acceleration = actor_snapshot.get_acceleration()
    return {
        "location": {"x": transform.location.x, "y": transform.location.y, "z": transform.location.z},
        "rotation": {
            "pitch": transform.rotation.pitch,
            "yaw": transform.rotation.yaw,
            "roll": transform.rotation.roll,
        },
        "velocity": {"x": velocity.x, "y": velocity.y, "z": velocity.z},
        "acceleration": {"x": acceleration.x, "y": acceleration.y, "z": acceleration.z},
    }


def sensor_callback(sensor_data: Any, queue: Queue[SensorPacket], sensor_name: str) -> None:
    queue.put(SensorPacket(name=sensor_name, frame=sensor_data.frame, data=sensor_data))


def wait_for_sensor_frame(
    sensor_name: str,
    queue: Queue[SensorPacket],
    frame: int,
    timeout: float,
) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            packet = queue.get(timeout=max(0.01, deadline - time.monotonic()))
        except Empty:
            break
        if packet.frame < frame:
            continue
        if packet.frame == frame:
            return packet.data
        raise RuntimeError(
            f"Sensor {sensor_name} skipped frame {frame}; next received frame was {packet.frame}"
        )
    raise TimeoutError(f"Timed out waiting for {sensor_name} frame {frame}")


def save_camera_image(image: Any, path: Path) -> None:
    array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    bgr = array[:, :, :3]
    if not cv2.imwrite(str(path), bgr):
        raise IOError(f"Failed to write image: {path}")


def save_lidar(lidar_data: Any, path: Path) -> np.ndarray:
    points = np.frombuffer(lidar_data.raw_data, dtype=np.float32).reshape((-1, 4)).copy()
    np.save(path, points)
    return points


def write_metadata(
    path: Path,
    frame: int,
    timestamp: float,
    ego_state: dict[str, Any],
    sensor_frames: dict[str, int],
    control: dict[str, Any],
    telemetry: dict[str, Any],
    world_model: dict[str, Any],
    actors: list[dict[str, Any]] | None = None,
) -> None:
    metadata = {
        "frame_id": frame,
        "timestamp": timestamp,
        "sensor_frames": sensor_frames,
        "ego_vehicle": ego_state,
        "control": control,
        "telemetry": telemetry,
        "world_model": world_model,
        "actors": actors or [],
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def classify_actor(type_id: str) -> str | None:
    tid = type_id.lower()
    if "walker.pedestrian" in tid:
        return "Pedestrian"
    if "bicycle" in tid or "bike" in tid or "cyclist" in tid or "motorcycle" in tid:
        return "Cyclist"
    if tid.startswith("vehicle."):
        return "Vehicle"
    return None


def collect_nearby_actor_boxes(
    world: Any,
    snapshot: Any,
    ego_vehicle: Any,
    max_distance_m: float = 75.0,
) -> list[dict[str, Any]]:
    ego_snapshot = snapshot.find(ego_vehicle.id)
    if ego_snapshot is None:
        return []

    ego_tf = ego_snapshot.get_transform()
    ego_loc = ego_tf.location
    actors_out: list[dict[str, Any]] = []

    for actor in world.get_actors():
        actor_id = int(actor.id)
        if actor_id == int(ego_vehicle.id):
            continue

        class_name = classify_actor(str(actor.type_id))
        if class_name is None:
            continue

        actor_snapshot = snapshot.find(actor_id)
        if actor_snapshot is None:
            continue

        actor_tf = actor_snapshot.get_transform()
        actor_loc = actor_tf.location
        distance = float(ego_loc.distance(actor_loc))
        if distance > max_distance_m:
            continue

        bbox = actor.bounding_box
        bbox_center_world = actor_tf.transform(bbox.location)
        semantic_tag = None
        try:
            tags = list(getattr(actor, "semantic_tags", []))
            semantic_tag = int(tags[0]) if tags else None
        except Exception:
            semantic_tag = None

        actors_out.append(
            {
                "id": actor_id,
                "type_id": str(actor.type_id),
                "class_name": class_name,
                "semantic_tag": semantic_tag,
                "distance_to_ego_m": distance,
                "location": {"x": actor_loc.x, "y": actor_loc.y, "z": actor_loc.z},
                "rotation": {
                    "pitch": actor_tf.rotation.pitch,
                    "yaw": actor_tf.rotation.yaw,
                    "roll": actor_tf.rotation.roll,
                },
                "velocity": {
                    "x": actor_snapshot.get_velocity().x,
                    "y": actor_snapshot.get_velocity().y,
                    "z": actor_snapshot.get_velocity().z,
                },
                "bounding_box": {
                    "extent": {
                        "x": float(bbox.extent.x),
                        "y": float(bbox.extent.y),
                        "z": float(bbox.extent.z),
                    },
                    "location": {
                        "x": float(bbox_center_world.x),
                        "y": float(bbox_center_world.y),
                        "z": float(bbox_center_world.z),
                    },
                    "rotation": {
                        "pitch": float(bbox.rotation.pitch),
                        "yaw": float(bbox.rotation.yaw),
                        "roll": float(bbox.rotation.roll),
                    },
                },
            }
        )

    return actors_out


def spawn_ego_vehicle(carla: Any, world: Any, seed: int) -> Any:
    rng = random.Random(seed)
    blueprints = list(world.get_blueprint_library().filter("vehicle.*"))
    blueprints = [bp for bp in blueprints if not bp.id.endswith("microlino")]
    if not blueprints:
        raise RuntimeError("No vehicle blueprints available")
    spawn_points = list(world.get_map().get_spawn_points())
    rng.shuffle(spawn_points)
    rng.shuffle(blueprints)
    for blueprint in blueprints:
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "hero")
        for spawn_point in spawn_points:
            vehicle = world.try_spawn_actor(blueprint, spawn_point)
            if vehicle is not None:
                return vehicle
    raise RuntimeError("Unable to spawn ego vehicle")


def spawn_traffic_vehicles(
    carla: Any,
    client: Any,
    world: Any,
    tm_port: int,
    count: int,
    seed: int,
    prefer_cyclist_vehicles: int = 0,
) -> list[Any]:
    if count <= 0:
        return []
    rng = random.Random(seed + 1)
    blueprints = list(world.get_blueprint_library().filter("vehicle.*"))
    cyclist_blueprints = [
        bp for bp in blueprints
        if any(token in bp.id.lower() for token in ("bicycle", "bike", "motorcycle", "cyclist"))
    ]
    spawn_points = list(world.get_map().get_spawn_points())
    rng.shuffle(spawn_points)
    rng.shuffle(blueprints)
    actors = []
    cyclists_spawned = 0
    for spawn_point in spawn_points:
        if len(actors) >= count:
            break
        if cyclists_spawned < prefer_cyclist_vehicles and cyclist_blueprints:
            blueprint = rng.choice(cyclist_blueprints)
        else:
            blueprint = rng.choice(blueprints)
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "autopilot")
        actor = world.try_spawn_actor(blueprint, spawn_point)
        if actor is None:
            continue
        actor.set_autopilot(True, tm_port)
        actors.append(actor)
        if any(token in blueprint.id.lower() for token in ("bicycle", "bike", "motorcycle", "cyclist")):
            cyclists_spawned += 1
    if len(actors) < count:
        LOGGER.warning("Requested %d traffic vehicles, spawned %d", count, len(actors))
    if prefer_cyclist_vehicles > 0 and cyclists_spawned < prefer_cyclist_vehicles:
        LOGGER.warning(
            "Requested %d cyclist-like vehicles, spawned %d",
            prefer_cyclist_vehicles,
            cyclists_spawned,
        )
    return actors


def spawn_walkers(carla: Any, client: Any, world: Any, count: int, seed: int) -> tuple[list[Any], list[Any]]:
    if count <= 0:
        return [], []
    rng = random.Random(seed + 2)
    walker_blueprints = list(world.get_blueprint_library().filter("walker.pedestrian.*"))
    controller_bp = world.get_blueprint_library().find("controller.ai.walker")
    walkers = []
    controllers = []
    for _ in range(count):
        location = world.get_random_location_from_navigation()
        if location is None:
            continue
        walker_bp = rng.choice(walker_blueprints)
        walker = world.try_spawn_actor(walker_bp, carla.Transform(location))
        if walker is None:
            continue
        controller = world.spawn_actor(controller_bp, carla.Transform(), walker)
        controller.start()
        target = world.get_random_location_from_navigation()
        if target is not None:
            controller.go_to_location(target)
        controller.set_max_speed(1.4)
        walkers.append(walker)
        controllers.append(controller)

    # Fallback for runtimes/maps where navmesh walker locations are unavailable.
    if len(walkers) < count:
        spawn_points = list(world.get_map().get_spawn_points())
        rng.shuffle(spawn_points)
        for spawn_point in spawn_points:
            if len(walkers) >= count:
                break
            walker_bp = rng.choice(walker_blueprints)
            fallback_loc = spawn_point.location
            fallback_loc.z += 0.5
            walker = world.try_spawn_actor(walker_bp, carla.Transform(fallback_loc, spawn_point.rotation))
            if walker is None:
                continue
            walkers.append(walker)

    if len(walkers) < count:
        LOGGER.warning("Requested %d walkers, spawned %d", count, len(walkers))
    return walkers, controllers


def spawn_sensors(carla: Any, world: Any, ego_vehicle: Any, config: dict[str, Any]) -> tuple[list[Any], dict[str, Queue[SensorPacket]]]:
    sensors = []
    queues: dict[str, Queue[SensorPacket]] = {}
    blueprint_library = world.get_blueprint_library()

    for sensor_name in SENSOR_NAMES:
        sensor_config = deep_get(config, "sensors", sensor_name, default={})
        blueprint = blueprint_library.find(sensor_config["blueprint"])
        set_blueprint_attributes(blueprint, sensor_config.get("attributes", {}))
        transform = make_transform(carla, sensor_config.get("transform"))
        sensor = world.spawn_actor(blueprint, transform, attach_to=ego_vehicle)
        sensor_queue: Queue[SensorPacket] = Queue()
        sensor.listen(lambda data, queue=sensor_queue, name=sensor_name: sensor_callback(data, queue, name))
        sensors.append(sensor)
        queues[sensor_name] = sensor_queue
    return sensors, queues


def spawn_event_sensors(
    world: Any,
    ego_vehicle: Any,
    runtime: WorldModelRuntime,
) -> tuple[list[Any], dict[str, Any]]:
    sensors: list[Any] = []
    state = {"collision_intensity": 0.0, "lane_invasion": False}
    library = world.get_blueprint_library()

    if runtime.enable_collision_sensor:
        collision_bp = library.find("sensor.other.collision")
        collision_sensor = world.spawn_actor(collision_bp, ego_vehicle.get_transform(), attach_to=ego_vehicle)

        def on_collision(event: Any) -> None:
            impulse = event.normal_impulse
            magnitude = (impulse.x**2 + impulse.y**2 + impulse.z**2) ** 0.5
            scaled = magnitude * runtime.collision_intensity_scale
            state["collision_intensity"] = max(float(state["collision_intensity"]), float(scaled))

        collision_sensor.listen(on_collision)
        sensors.append(collision_sensor)

    if runtime.enable_lane_invasion_sensor:
        lane_bp = library.find("sensor.other.lane_invasion")
        lane_sensor = world.spawn_actor(lane_bp, ego_vehicle.get_transform(), attach_to=ego_vehicle)

        def on_lane_invasion(_event: Any) -> None:
            state["lane_invasion"] = True

        lane_sensor.listen(on_lane_invasion)
        sensors.append(lane_sensor)

    return sensors, state


def carla_server_alive(client: Any) -> bool:
    try:
        client.get_world().get_snapshot()
        return True
    except Exception as exc:  # pragma: no cover - server-dependent
        LOGGER.warning("CARLA server is not reachable during cleanup: %s", exc)
        return False


def short_map_name(map_name: str) -> str:
    return map_name.rsplit("/", 1)[-1]


def resolve_runtime_map(client: Any, requested_map: str | None) -> tuple[str | None, list[str]]:
    available_maps = [str(path) for path in client.get_available_maps()]
    if not requested_map:
        return None, available_maps

    requested = str(requested_map).strip()
    by_short_name = {short_map_name(path): path for path in available_maps}
    if requested in by_short_name:
        return short_map_name(by_short_name[requested]), available_maps
    if requested in available_maps:
        return short_map_name(requested), available_maps

    available_short = ", ".join(sorted(short_map_name(path) for path in available_maps)) or "<none>"
    raise RuntimeError(
        f"Configured carla.map '{requested}' is not available in this runtime. "
        f"Available maps: {available_short}"
    )


def cleanup_actors(carla: Any, client: Any, sensors: list[Any], controllers: list[Any], actors: list[Any]) -> None:
    for sensor in sensors:
        try:
            sensor.stop()
        except Exception as exc:  # pragma: no cover - server-dependent cleanup
            LOGGER.warning("Failed to stop sensor %s: %s", getattr(sensor, "id", "?"), exc)

    for controller in controllers:
        try:
            controller.stop()
        except Exception as exc:  # pragma: no cover - server-dependent cleanup
            LOGGER.warning("Failed to stop walker controller %s: %s", getattr(controller, "id", "?"), exc)

    actor_ids: list[int] = []
    seen_ids: set[int] = set()
    for actor in [*sensors, *controllers, *actors]:
        actor_id = getattr(actor, "id", None)
        if actor_id is None or actor_id in seen_ids:
            continue
        seen_ids.add(actor_id)
        actor_ids.append(int(actor_id))

    if not actor_ids:
        return

    commands = [carla.command.DestroyActor(actor_id) for actor_id in actor_ids]
    try:
        client.apply_batch_sync(commands, True)
    except Exception as exc:  # pragma: no cover - server-dependent cleanup
        LOGGER.warning("Failed to batch-destroy actors: %s", exc)


def record(
    config: dict[str, Any],
    output_dir: Path,
    overwrite: bool,
    seed: int,
    world_model_runtime: WorldModelRuntime,
) -> dict[str, Any]:
    carla = import_carla()
    prepare_output_dir(output_dir, overwrite)

    host = str(deep_get(config, "carla", "host", default="localhost"))
    port = int(deep_get(config, "carla", "port", default=2000))
    map_name = deep_get(config, "carla", "map")
    weather_name = deep_get(config, "carla", "weather")
    timeout = float(deep_get(config, "carla", "timeout", default=10.0))
    startup_wait_seconds = float(deep_get(config, "carla", "startup_wait_seconds", default=30.0))
    tm_port = int(deep_get(config, "carla", "tm_port", default=10000))
    num_frames = int(deep_get(config, "recording", "num_frames", default=1000))
    traffic_vehicles = int(deep_get(config, "recording", "traffic_vehicles", default=0))
    traffic_walkers = int(deep_get(config, "recording", "traffic_walkers", default=0))
    prefer_cyclist_vehicles = int(deep_get(config, "recording", "prefer_cyclist_vehicles", default=0))
    fixed_delta_seconds = float(deep_get(config, "recording", "fixed_delta_seconds", default=0.1))
    warmup_ticks = int(deep_get(config, "recording", "warmup_ticks", default=5))
    sensor_timeout = float(deep_get(config, "recording", "sensor_timeout", default=5.0))
    actor_label_max_distance = float(
        deep_get(config, "recording", "actor_label_max_distance_m", default=75.0)
    )
    use_synchronous_mode = bool(deep_get(config, "recording", "use_synchronous_mode", default=True))
    native_recorder = bool(deep_get(config, "recording", "native_recorder", default=False))

    client, world = connect_to_carla(
        carla=carla,
        host=host,
        port=port,
        timeout=timeout,
        startup_wait_seconds=startup_wait_seconds,
    )
    target_map, _ = resolve_runtime_map(client, str(map_name) if map_name is not None else None)
    if target_map:
        current_map = short_map_name(world.get_map().name)
        if current_map != target_map:
            LOGGER.info("Loading map '%s' (current map: %s)", target_map, current_map)
            world = client.load_world(target_map)
        LOGGER.info("Using CARLA map: %s", short_map_name(world.get_map().name))
    else:
        LOGGER.info("Using current CARLA map: %s", short_map_name(world.get_map().name))

    if weather_name:
        weather_obj = getattr(carla.WeatherParameters, str(weather_name), None)
        if weather_obj is None:
            raise RuntimeError(f"Unknown weather preset: {weather_name}")
        world.set_weather(weather_obj)
        LOGGER.info("Using CARLA weather: %s", weather_name)
    original_settings = world.get_settings()
    traffic_manager = client.get_trafficmanager(tm_port)

    ego_vehicle = None
    sensors = []
    event_sensors: list[Any] = []
    traffic = []
    walkers = []
    controllers = []
    all_spawned_actors = []
    native_recording_started = False

    try:
        if use_synchronous_mode:
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = fixed_delta_seconds
            world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_random_device_seed(seed)

        if native_recorder:
            client.start_recorder(str(output_dir / "recording.log"))
            native_recording_started = True

        ego_vehicle = spawn_ego_vehicle(carla, world, seed)
        all_spawned_actors.append(ego_vehicle)
        ego_vehicle.set_autopilot(True, tm_port)
        traffic = spawn_traffic_vehicles(
            carla=carla,
            client=client,
            world=world,
            tm_port=tm_port,
            count=traffic_vehicles,
            seed=seed,
            prefer_cyclist_vehicles=prefer_cyclist_vehicles,
        )
        all_spawned_actors.extend(traffic)
        walkers, controllers = spawn_walkers(carla, client, world, traffic_walkers, seed)
        all_spawned_actors.extend(walkers)
        sensors, queues = spawn_sensors(carla, world, ego_vehicle, config)
        event_sensors, event_state = spawn_event_sensors(world, ego_vehicle, world_model_runtime)
        all_spawned_actors.extend(event_sensors)

        for _ in range(warmup_ticks):
            world.tick() if use_synchronous_mode else world.wait_for_tick()

        last_ego_state: dict[str, Any] | None = None
        last_control: dict[str, Any] | None = None
        terminal_state = TerminalState()
        prev_location: dict[str, float] | None = None
        lane_invasion_total = 0
        offroad_total = 0
        done_total = 0

        last_timestamp = -1.0
        started = time.monotonic()
        for frame_index in range(num_frames):
            try:
                frame = world.tick() if use_synchronous_mode else world.wait_for_tick().frame
            except Exception as exc:
                raise RuntimeError(f"world tick failed at frame_index={frame_index}: {exc}") from exc

            try:
                snapshot = world.get_snapshot()
                timestamp = float(snapshot.timestamp.elapsed_seconds)
            except Exception as exc:
                raise RuntimeError(f"world snapshot failed at frame_index={frame_index}, frame={frame}: {exc}") from exc

            if timestamp <= last_timestamp:
                raise RuntimeError(f"Timestamp not monotonic at frame {frame}: {timestamp} <= {last_timestamp}")
            last_timestamp = timestamp

            try:
                sensor_data = {
                    name: wait_for_sensor_frame(name, queues[name], frame, sensor_timeout)
                    for name in SENSOR_NAMES
                }
            except Exception as exc:
                raise RuntimeError(
                    f"sensor sync failed at frame_index={frame_index}, frame={frame}: {exc}"
                ) from exc

            stem = f"{frame_index:06d}"
            try:
                try:
                    ego_snapshot = snapshot.find(ego_vehicle.id)
                    if ego_snapshot is None:
                        raise RuntimeError(f"ego actor id={ego_vehicle.id} missing from snapshot")
                    ego_state = snapshot_actor_to_dict(ego_snapshot)
                    ego_state["state_valid"] = True
                    last_ego_state = ego_state
                except Exception as exc:
                    if last_ego_state is None:
                        raise
                    ego_state = dict(last_ego_state)
                    ego_state["state_valid"] = False
                    ego_state["state_error"] = str(exc)

                ctrl = ego_vehicle.get_control()
                control = {
                    "steer": float(ctrl.steer),
                    "throttle": float(ctrl.throttle),
                    "brake": float(ctrl.brake),
                    "reverse": bool(ctrl.reverse),
                    "hand_brake": bool(ctrl.hand_brake),
                    "manual_gear_shift": bool(ctrl.manual_gear_shift),
                }
                steer_delta = 0.0
                if last_control is not None:
                    steer_delta = float(control["steer"]) - float(last_control["steer"])
                last_control = control

                location = ego_state.get("location", {})
                velocity = ego_state.get("velocity", {})
                speed_kmh = speed_kmh_from_velocity(velocity if isinstance(velocity, dict) else {})
                progress_m = progress_m_from_locations(
                    prev_location,
                    location if isinstance(location, dict) else {"x": 0.0, "y": 0.0, "z": 0.0},
                )
                prev_location = location if isinstance(location, dict) else None

                offroad = False
                try:
                    waypoint = world.get_map().get_waypoint(ego_vehicle.get_location(), project_to_road=False)
                    offroad = waypoint is None
                except Exception:
                    offroad = False
                lane_invasion = bool(event_state.get("lane_invasion", False))
                collision_intensity = float(event_state.get("collision_intensity", 0.0))

                terminal_state = advance_terminal_state(
                    prev_state=terminal_state,
                    collision_intensity=collision_intensity,
                    speed_kmh=speed_kmh,
                    lane_invasion=lane_invasion,
                    offroad=offroad,
                    cfg=world_model_runtime.terminal_cfg,
                )
                telemetry = {
                    "speed_kmh": float(speed_kmh),
                    "progress_m": float(progress_m),
                    "collision_intensity": float(collision_intensity),
                    "lane_invasion": bool(lane_invasion),
                    "offroad": bool(offroad),
                    "stuck": bool(terminal_state.stuck_frames >= world_model_runtime.terminal_cfg.stuck_frames_threshold),
                }
                reward_value = compute_reward(telemetry=telemetry, steer_delta=steer_delta, cfg=world_model_runtime.reward_cfg)
                done, done_reason = terminal_decision(
                    state=terminal_state,
                    cfg=world_model_runtime.terminal_cfg,
                    rollout_end=(frame_index + 1 == num_frames),
                )
                world_model = {
                    "reward": float(reward_value),
                    "done": bool(done),
                    "done_reason": done_reason,
                }
                lane_invasion_total += int(lane_invasion)
                offroad_total += int(offroad)
                done_total += int(done)

                save_camera_image(sensor_data["rgb_camera"], output_dir / "rgb" / f"{stem}.png")
                save_camera_image(sensor_data["semantic_camera"], output_dir / "semantic" / f"{stem}.png")
                lidar_points = save_lidar(sensor_data["lidar"], output_dir / "lidar" / f"{stem}.npy")
                actor_records = collect_nearby_actor_boxes(
                    world=world,
                    snapshot=snapshot,
                    ego_vehicle=ego_vehicle,
                    max_distance_m=actor_label_max_distance,
                )
                write_metadata(
                    output_dir / "metadata" / f"{stem}.json",
                    frame=frame,
                    timestamp=timestamp,
                    ego_state=ego_state,
                    sensor_frames={name: int(sensor_data[name].frame) for name in SENSOR_NAMES},
                    control=control,
                    telemetry=telemetry,
                    world_model=world_model,
                    actors=actor_records,
                )
                event_state["collision_intensity"] = 0.0
                event_state["lane_invasion"] = False
            except Exception as exc:
                raise RuntimeError(
                    f"frame write failed at frame_index={frame_index}, frame={frame}: {exc}"
                ) from exc

            if frame_index == 0 or (frame_index + 1) % 100 == 0 or frame_index + 1 == num_frames:
                LOGGER.info(
                    "Recorded %d/%d frames (CARLA frame %d, lidar points %d)",
                    frame_index + 1,
                    num_frames,
                    frame,
                    len(lidar_points),
                )

        elapsed = time.monotonic() - started
        runtime_map = short_map_name(world.get_map().name)
        sensors_file = write_calibration_file(output_dir=output_dir, config=config)
        scenario_file = write_scenario_file(
            output_dir=output_dir,
            scenario={
                "map": runtime_map,
                "weather": weather_name,
                "seed": seed,
                "traffic_manager_port": tm_port,
                "traffic_vehicles_requested": traffic_vehicles,
                "traffic_walkers_requested": traffic_walkers,
                "prefer_cyclist_vehicles": prefer_cyclist_vehicles,
                "traffic_vehicles_spawned": len(traffic),
                "traffic_walkers_spawned": len(walkers),
                "use_synchronous_mode": use_synchronous_mode,
                "fixed_delta_seconds": fixed_delta_seconds,
                "warmup_ticks": warmup_ticks,
                "sensor_timeout": sensor_timeout,
                "startup_wait_seconds": startup_wait_seconds,
                "world_model": {
                    "policy": world_model_runtime.policy,
                    "stage13_mode": world_model_runtime.mode,
                    "reward_config": world_model_runtime.reward_cfg.__dict__,
                    "terminal_config": world_model_runtime.terminal_cfg.__dict__,
                    "events": {
                        "enable_collision_sensor": world_model_runtime.enable_collision_sensor,
                        "enable_lane_invasion_sensor": world_model_runtime.enable_lane_invasion_sensor,
                        "collision_intensity_scale": world_model_runtime.collision_intensity_scale,
                    },
                },
            },
        )
        integrity = validate_recording_integrity(output_dir=output_dir, expected_frames=num_frames)

        summary = {
            "output_dir": str(output_dir),
            "frames_target": num_frames,
            "frames_recorded": int(integrity["num_common_stems"]),
            "elapsed_seconds": elapsed,
            "fps_wall_clock": num_frames / elapsed if elapsed > 0 else None,
            "carla": {
                "host": host,
                "port": port,
                "map": runtime_map,
                "weather": weather_name,
                "tm_port": tm_port,
            },
            "traffic": {
                "vehicles_requested": traffic_vehicles,
                "vehicles_spawned": len(traffic),
                "walkers_requested": traffic_walkers,
                "walkers_spawned": len(walkers),
            },
            "recording": {
                "fixed_delta_seconds": fixed_delta_seconds,
                "warmup_ticks": warmup_ticks,
                "sensor_timeout": sensor_timeout,
                "startup_wait_seconds": startup_wait_seconds,
                "actor_label_max_distance_m": actor_label_max_distance,
                "prefer_cyclist_vehicles": prefer_cyclist_vehicles,
                "native_recorder": native_recorder,
                "use_synchronous_mode": use_synchronous_mode,
            },
            "world_model": {
                "stage13_mode": world_model_runtime.mode,
                "policy": world_model_runtime.policy,
                "reward_version": "carla_progress_v1",
                "terminal_version": "carla_terminal_v1",
                "lane_invasion_frames": lane_invasion_total,
                "offroad_frames": offroad_total,
                "done_frames": done_total,
            },
            "artifacts": {
                "scenario_file": str(scenario_file),
                "sensors_file": str(sensors_file),
            },
            "integrity": integrity,
            "complete": bool(integrity["pass"]),
        }
        with (output_dir / "recording_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        with (output_dir / "dataset_complete.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "complete": bool(integrity["pass"]),
                    "reason": None if integrity["pass"] else "Integrity checks failed",
                    "integrity": integrity,
                },
                handle,
                indent=2,
            )
        if not integrity["pass"]:
            raise RuntimeError(
                f"Dataset integrity failed for {output_dir}; inspect recording_summary.json and dataset_complete.json"
            )
        return summary
    finally:
        server_alive = carla_server_alive(client)
        if native_recording_started and server_alive:
            try:
                client.stop_recorder()
            except Exception as exc:  # pragma: no cover - CARLA cleanup best effort
                LOGGER.warning("Failed to stop CARLA native recorder: %s", exc)
        if server_alive:
            cleanup_actors(carla, client, sensors, controllers, all_spawned_actors)
            if use_synchronous_mode:
                try:
                    traffic_manager.set_synchronous_mode(False)
                except Exception as exc:  # pragma: no cover - server-dependent cleanup
                    LOGGER.warning("Failed to disable TM synchronous mode: %s", exc)
            try:
                world.apply_settings(original_settings)
            except Exception as exc:  # pragma: no cover - server-dependent cleanup
                LOGGER.warning("Failed to restore world settings: %s", exc)


def print_config_summary(config: dict[str, Any], output_dir: Path, world_model_runtime: WorldModelRuntime) -> None:
    print("[PASS] Config loaded")
    print(f"output_dir: {output_dir}")
    print(f"num_frames: {deep_get(config, 'recording', 'num_frames')}")
    print(f"carla: {deep_get(config, 'carla', 'host')}:{deep_get(config, 'carla', 'port')}")
    print(f"map: {deep_get(config, 'carla', 'map')}")
    print(f"weather: {deep_get(config, 'carla', 'weather')}")
    print(f"stage13_mode: {world_model_runtime.mode}")
    for sensor_name in SENSOR_NAMES:
        attrs = deep_get(config, "sensors", sensor_name, "attributes", default={})
        print(f"{sensor_name}: {deep_get(config, 'sensors', sensor_name, 'blueprint')} attrs={attrs}")


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    try:
        config = apply_overrides(load_config(Path(args.config)), args)
        validate_config(config)
        world_model_runtime = load_world_model_runtime(
            world_model_config_path=Path(args.world_model_config),
            sensor_config=config,
            allow_stage13_control=bool(args.allow_stage13_control),
            policy=str(args.policy),
        )
        output_dir = resolve_output_dir(config)
        if args.check_config:
            print_config_summary(config, output_dir, world_model_runtime)
            return 0
        summary = record(
            config,
            output_dir,
            overwrite=args.overwrite,
            seed=args.seed,
            world_model_runtime=world_model_runtime,
        )
    except Exception as exc:
        LOGGER.error("%s", exc)
        return 1

    print("[PASS] CARLA recording complete")
    print(f"output_dir: {summary['output_dir']}")
    print(f"frames_target: {summary['frames_target']}")
    print(f"frames_recorded: {summary['frames_recorded']}")
    print(f"traffic_vehicles_spawned: {summary['traffic']['vehicles_spawned']}")
    print(f"traffic_walkers_spawned: {summary['traffic']['walkers_spawned']}")
    print(f"stage13_mode: {summary['world_model']['stage13_mode']}")
    print(f"complete: {summary['complete']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
