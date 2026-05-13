#!/usr/bin/env python3
"""Run CARLA Python examples with Traffic Manager default-port workaround.

Some CARLA 0.10.0 runtime environments fail on Traffic Manager default ports
8000/9000 with RuntimeError(std::exception). This wrapper remaps only those
defaults to a user-provided fallback port, without modifying upstream repos.
"""

from __future__ import annotations

import argparse
import runpy
import signal
import sys
import time
from pathlib import Path


class RuntimeLimitReached(Exception):
    """Raised inside the wrapped script when a bounded smoke run is complete."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an upstream CARLA example script with TM port remapping.",
    )
    parser.add_argument(
        "--script",
        required=True,
        help="Path to upstream script, e.g. repos/carla/PythonAPI/examples/manual_control.py",
    )
    parser.add_argument(
        "--tm-fallback-port",
        type=int,
        default=10000,
        help="Fallback TM port when script requests 8000/9000 (default: 10000)",
    )
    parser.add_argument(
        "--dummy-traffic-manager",
        action="store_true",
        help="Return a no-op Traffic Manager object instead of using CARLA TM.",
    )
    parser.add_argument(
        "--disable-autopilot",
        action="store_true",
        help="Patch vehicle.set_autopilot to a no-op to avoid TM-dependent failures.",
    )
    parser.add_argument(
        "--clamp-camera-width",
        type=int,
        default=640,
        help="Maximum camera image width to clamp sensor blueprints to (default: 640)",
    )
    parser.add_argument(
        "--clamp-camera-height",
        type=int,
        default=360,
        help="Maximum camera image height to clamp sensor blueprints to (default: 360)",
    )
    parser.add_argument(
        "--clamp-lidar-pps",
        type=int,
        default=20000,
        help="Maximum lidar points-per-second to clamp sensor blueprints to (default: 20000)",
    )
    parser.add_argument(
        "--max-runtime",
        type=float,
        default=0.0,
        help=(
            "Stop the wrapped script after this many seconds and return success. "
            "Useful for CARLA examples that run until a UI quit event."
        ),
    )
    parser.add_argument(
        "--safe-visualize-sensors",
        action="store_true",
        help=(
            "For visualize_multiple_sensors.py, run a bounded low-VRAM sensor "
            "path using the upstream display/sensor classes."
        ),
    )
    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to the upstream script. Use '--' before them.",
    )
    return parser.parse_args()


def patch_tm_port(fallback_port: int, dummy_tm: bool, disable_autopilot: bool) -> None:
    import carla

    original_get_tm = carla.Client.get_trafficmanager
    original_set_autopilot = carla.Vehicle.set_autopilot
    original_set_attr = carla.ActorBlueprint.set_attribute

    def _remap(port: int) -> int:
        if port in (8000, 9000):
            return fallback_port
        return port

    class _DummyTrafficManager:
        def set_synchronous_mode(self, _enabled):
            return None

        def set_random_device_seed(self, _seed):
            return None

        def set_hybrid_physics_mode(self, _enabled):
            return None

        def get_port(self):
            return fallback_port

    def patched_get_tm(self, port=8000):
        if dummy_tm:
            return _DummyTrafficManager()
        return original_get_tm(self, _remap(port))

    def patched_set_autopilot(self, enabled=True, tm_port=8000):
        if disable_autopilot:
            return None
        return original_set_autopilot(self, enabled, _remap(tm_port))

    def patched_set_attr(self, key, value):
        try:
            return original_set_attr(self, key, value)
        except RuntimeError:
            # UE5/driver combinations may reject a subset of camera attrs.
            print(f"[WARN] ignored unsupported blueprint attribute: {key}={value}")
            return None

    carla.Client.get_trafficmanager = patched_get_tm
    carla.Vehicle.set_autopilot = patched_set_autopilot
    carla.ActorBlueprint.set_attribute = patched_set_attr


def patch_sensor_clamps(max_w: int, max_h: int, max_pps: int) -> None:
    try:
        import carla
    except Exception:
        return

    orig_set_attr = carla.ActorBlueprint.set_attribute

    def clamped_set_attr(self, key, value):
        # clamp camera resolution
        try:
            if key == 'image_size_x':
                val = int(value)
                if val > max_w:
                    value = str(max_w)
            if key == 'image_size_y':
                val = int(value)
                if val > max_h:
                    value = str(max_h)
            if key == 'points_per_second':
                val = int(value)
                if val > max_pps:
                    value = str(max_pps)
        except Exception:
            pass
        return orig_set_attr(self, key, value)

    carla.ActorBlueprint.set_attribute = clamped_set_attr


def run_with_optional_runtime_limit(script_path: Path, max_runtime: float) -> None:
    if max_runtime <= 0:
        runpy.run_path(str(script_path), run_name="__main__")
        return

    def _handle_timeout(_signum, _frame):
        raise RuntimeLimitReached

    previous_handler = signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, max_runtime)
    try:
        runpy.run_path(str(script_path), run_name="__main__")
    except RuntimeLimitReached:
        print(f"[PASS] Runtime limit reached after {max_runtime:.1f}s without native abort")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def run_safe_visualize_sensors(script_path: Path, passthrough: list[str], max_runtime: float) -> None:
    import carla

    module_globals = runpy.run_path(str(script_path), run_name="__carla_wrapped__")
    display_manager_cls = module_globals["DisplayManager"]
    sensor_manager_cls = module_globals["SensorManager"]

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("-p", "--port", default=2000, type=int)
    parser.add_argument("--res", default="320x180")
    parser.add_argument("--filter", default="vehicle.*")
    parsed, _unknown = parser.parse_known_args(passthrough)
    width, height = [int(x) for x in parsed.res.split("x")]
    runtime = max_runtime if max_runtime > 0 else 20.0

    client = carla.Client(parsed.host, parsed.port)
    client.set_timeout(10.0)
    world = client.get_world()
    original_settings = world.get_settings()

    display_manager = None
    vehicle = None
    sensors = []

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        blueprints = list(world.get_blueprint_library().filter(parsed.filter))
        spawn_points = list(world.get_map().get_spawn_points())
        if not blueprints or not spawn_points:
            raise RuntimeError("No vehicle blueprints or spawn points available for sensor smoke run")

        for spawn_point in spawn_points:
            vehicle = world.try_spawn_actor(blueprints[0], spawn_point)
            if vehicle is not None:
                break
        if vehicle is None:
            raise RuntimeError("Unable to spawn vehicle for sensor smoke run")

        display_manager = display_manager_cls(grid_size=[1, 2], window_size=[width, height])
        sensors.append(sensor_manager_cls(
            world,
            display_manager,
            "Radar",
            carla.Transform(carla.Location(x=2.0, z=1.5), carla.Rotation(yaw=0)),
            vehicle,
            {"horizontal_fov": "35", "vertical_fov": "20", "range": "50"},
            display_pos=[0, 0],
        ))
        sensors.append(sensor_manager_cls(
            world,
            display_manager,
            "Radar",
            carla.Transform(carla.Location(x=0.0, z=1.5), carla.Rotation(yaw=90)),
            vehicle,
            {"horizontal_fov": "35", "vertical_fov": "20", "range": "30"},
            display_pos=[0, 1],
        ))

        deadline = time.monotonic() + runtime
        while time.monotonic() < deadline:
            world.tick()
            display_manager.render()

        missing = [idx for idx, sensor in enumerate(sensors) if sensor.tics_processing <= 0]
        if missing:
            raise RuntimeError(f"Sensor callbacks did not receive data: {missing}")
        print(f"[PASS] safe visualize sensor smoke ran {runtime:.1f}s with {len(sensors)} sensors")
    finally:
        if display_manager is not None:
            display_manager.destroy()
        if vehicle is not None:
            vehicle.destroy()
        world.apply_settings(original_settings)


def main() -> int:
    args = parse_args()
    script_path = Path(args.script).expanduser().resolve()
    if not script_path.is_file():
        print(f"ERROR: script not found: {script_path}", file=sys.stderr)
        return 2

    patch_tm_port(args.tm_fallback_port, args.dummy_traffic_manager, args.disable_autopilot)
    patch_sensor_clamps(args.clamp_camera_width, args.clamp_camera_height, args.clamp_lidar_pps)

    passthrough = list(args.script_args)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    sys.argv = [str(script_path)] + passthrough
    if args.safe_visualize_sensors:
        if script_path.name != "visualize_multiple_sensors.py":
            print("ERROR: --safe-visualize-sensors only supports visualize_multiple_sensors.py", file=sys.stderr)
            return 2
        run_safe_visualize_sensors(script_path, passthrough, args.max_runtime)
    else:
        run_with_optional_runtime_limit(script_path, args.max_runtime)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
