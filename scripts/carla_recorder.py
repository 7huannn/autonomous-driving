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


LOGGER = logging.getLogger("carla_recorder")
SENSOR_NAMES = ("rgb_camera", "semantic_camera", "lidar")


@dataclass
class SensorPacket:
    name: str
    frame: int
    data: Any


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
    parser.add_argument("--timeout", type=float, help="Override carla.timeout")
    parser.add_argument("--tm-port", type=int, help="Override carla.tm_port")
    parser.add_argument("--width", type=int, help="Override camera image width")
    parser.add_argument("--height", type=int, help="Override camera image height")
    parser.add_argument("--traffic-vehicles", type=int, help="Override traffic vehicle count")
    parser.add_argument("--traffic-walkers", type=int, help="Override traffic walker count")
    parser.add_argument("--lidar-pps", type=int, help="Override LiDAR points_per_second")
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
        ("carla", "host"): args.host,
        ("carla", "port"): args.port,
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
    for subdir in ("rgb", "semantic", "lidar", "metadata"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)


def import_carla():
    try:
        import carla  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise RuntimeError(f"CARLA Python API import failed: {exc}") from exc
    return carla


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


def actor_transform_to_dict(actor: Any) -> dict[str, Any]:
    transform = actor.get_transform()
    velocity = actor.get_velocity()
    acceleration = actor.get_acceleration()
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


def write_metadata(path: Path, frame: int, timestamp: float, ego_vehicle: Any, sensor_frames: dict[str, int]) -> None:
    metadata = {
        "frame_id": frame,
        "timestamp": timestamp,
        "sensor_frames": sensor_frames,
        "ego_vehicle": actor_transform_to_dict(ego_vehicle),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


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


def spawn_traffic_vehicles(carla: Any, client: Any, world: Any, tm_port: int, count: int, seed: int) -> list[Any]:
    if count <= 0:
        return []
    rng = random.Random(seed + 1)
    blueprints = list(world.get_blueprint_library().filter("vehicle.*"))
    spawn_points = list(world.get_map().get_spawn_points())
    rng.shuffle(spawn_points)
    rng.shuffle(blueprints)
    actors = []
    for spawn_point in spawn_points:
        if len(actors) >= count:
            break
        blueprint = rng.choice(blueprints)
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "autopilot")
        actor = world.try_spawn_actor(blueprint, spawn_point)
        if actor is None:
            continue
        actor.set_autopilot(True, tm_port)
        actors.append(actor)
    if len(actors) < count:
        LOGGER.warning("Requested %d traffic vehicles, spawned %d", count, len(actors))
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


def cleanup_actors(actors: list[Any], controllers: list[Any] | None = None) -> None:
    # CARLA 0.10.0 on this host can throw std::exception during destroy/stop,
    # which aborts the whole recorder after data has already been written.
    # Prefer leaving cleanup to server shutdown rather than risking a crash.
    return None


def carla_server_alive(client: Any) -> bool:
    try:
        client.get_world().get_snapshot()
        return True
    except Exception as exc:  # pragma: no cover - server-dependent
        LOGGER.warning("CARLA server is not reachable during cleanup: %s", exc)
        return False


def record(config: dict[str, Any], output_dir: Path, overwrite: bool, seed: int) -> dict[str, Any]:
    carla = import_carla()
    prepare_output_dir(output_dir, overwrite)

    host = str(deep_get(config, "carla", "host", default="localhost"))
    port = int(deep_get(config, "carla", "port", default=2000))
    timeout = float(deep_get(config, "carla", "timeout", default=10.0))
    tm_port = int(deep_get(config, "carla", "tm_port", default=10000))
    num_frames = int(deep_get(config, "recording", "num_frames", default=1000))
    traffic_vehicles = int(deep_get(config, "recording", "traffic_vehicles", default=0))
    traffic_walkers = int(deep_get(config, "recording", "traffic_walkers", default=0))
    fixed_delta_seconds = float(deep_get(config, "recording", "fixed_delta_seconds", default=0.1))
    warmup_ticks = int(deep_get(config, "recording", "warmup_ticks", default=5))
    sensor_timeout = float(deep_get(config, "recording", "sensor_timeout", default=5.0))
    use_synchronous_mode = bool(deep_get(config, "recording", "use_synchronous_mode", default=True))
    native_recorder = bool(deep_get(config, "recording", "native_recorder", default=False))

    client = carla.Client(host, port)
    client.set_timeout(timeout)
    world = client.get_world()
    original_settings = world.get_settings()
    traffic_manager = client.get_trafficmanager(tm_port)

    ego_vehicle = None
    sensors = []
    traffic = []
    walkers = []
    controllers = []
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
        ego_vehicle.set_autopilot(True, tm_port)
        traffic = spawn_traffic_vehicles(carla, client, world, tm_port, traffic_vehicles, seed)
        walkers, controllers = spawn_walkers(carla, client, world, traffic_walkers, seed)
        sensors, queues = spawn_sensors(carla, world, ego_vehicle, config)

        for _ in range(warmup_ticks):
            world.tick() if use_synchronous_mode else world.wait_for_tick()

        last_timestamp = -1.0
        started = time.monotonic()
        for frame_index in range(num_frames):
            frame = world.tick() if use_synchronous_mode else world.wait_for_tick().frame
            snapshot = world.get_snapshot()
            timestamp = float(snapshot.timestamp.elapsed_seconds)
            if timestamp <= last_timestamp:
                raise RuntimeError(f"Timestamp not monotonic at frame {frame}: {timestamp} <= {last_timestamp}")
            last_timestamp = timestamp

            sensor_data = {
                name: wait_for_sensor_frame(name, queues[name], frame, sensor_timeout)
                for name in SENSOR_NAMES
            }

            stem = f"{frame_index:06d}"
            save_camera_image(sensor_data["rgb_camera"], output_dir / "rgb" / f"{stem}.png")
            save_camera_image(sensor_data["semantic_camera"], output_dir / "semantic" / f"{stem}.png")
            lidar_points = save_lidar(sensor_data["lidar"], output_dir / "lidar" / f"{stem}.npy")
            write_metadata(
                output_dir / "metadata" / f"{stem}.json",
                frame=frame,
                timestamp=timestamp,
                ego_vehicle=ego_vehicle,
                sensor_frames={name: int(sensor_data[name].frame) for name in SENSOR_NAMES},
            )

            if frame_index == 0 or (frame_index + 1) % 100 == 0 or frame_index + 1 == num_frames:
                LOGGER.info(
                    "Recorded %d/%d frames (CARLA frame %d, lidar points %d)",
                    frame_index + 1,
                    num_frames,
                    frame,
                    len(lidar_points),
                )

        elapsed = time.monotonic() - started
        summary = {
            "output_dir": str(output_dir),
            "frames": num_frames,
            "elapsed_seconds": elapsed,
            "fps_wall_clock": num_frames / elapsed if elapsed > 0 else None,
            "traffic_vehicles": len(traffic),
            "traffic_walkers": len(walkers),
        }
        with (output_dir / "recording_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        return summary
    finally:
        if native_recording_started and carla_server_alive(client):
            try:
                client.stop_recorder()
            except Exception as exc:  # pragma: no cover - CARLA cleanup best effort
                LOGGER.warning("Failed to stop CARLA native recorder: %s", exc)


def print_config_summary(config: dict[str, Any], output_dir: Path) -> None:
    print("[PASS] Config loaded")
    print(f"output_dir: {output_dir}")
    print(f"num_frames: {deep_get(config, 'recording', 'num_frames')}")
    print(f"carla: {deep_get(config, 'carla', 'host')}:{deep_get(config, 'carla', 'port')}")
    for sensor_name in SENSOR_NAMES:
        attrs = deep_get(config, "sensors", sensor_name, "attributes", default={})
        print(f"{sensor_name}: {deep_get(config, 'sensors', sensor_name, 'blueprint')} attrs={attrs}")


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    try:
        config = apply_overrides(load_config(Path(args.config)), args)
        validate_config(config)
        output_dir = resolve_output_dir(config)
        if args.check_config:
            print_config_summary(config, output_dir)
            return 0
        summary = record(config, output_dir, overwrite=args.overwrite, seed=args.seed)
    except Exception as exc:
        LOGGER.error("%s", exc)
        return 1

    print("[PASS] CARLA recording complete")
    print(f"output_dir: {summary['output_dir']}")
    print(f"frames: {summary['frames']}")
    print(f"traffic_vehicles: {summary['traffic_vehicles']}")
    print(f"traffic_walkers: {summary['traffic_walkers']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
