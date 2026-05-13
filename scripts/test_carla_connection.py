#!/usr/bin/env python3
"""Reusable CARLA connection smoke test for Stage 02.

This checks that a running CARLA server is reachable, the client can
retrieve basic world metadata, and the connection closes cleanly.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass


@dataclass
class ConnectionResult:
    server_version: str
    client_version: str
    map_name: str
    weather: str
    fixed_delta_seconds: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check that a CARLA server is reachable and world metadata can be read.",
    )
    parser.add_argument("--host", default="localhost", help="CARLA host (default: localhost)")
    parser.add_argument("--port", type=int, default=2000, help="CARLA RPC port (default: 2000)")
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Connection timeout in seconds (default: 10.0)",
    )
    return parser.parse_args()


def check_connection(host: str, port: int, timeout: float) -> ConnectionResult:
    try:
        import carla
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(f"ERROR: carla import failed: {exc}") from exc

    client = carla.Client(host, port)
    client.set_timeout(timeout)

    try:
        world = client.get_world()
        settings = world.get_settings()
        return ConnectionResult(
            server_version=client.get_server_version(),
            client_version=client.get_client_version(),
            map_name=world.get_map().name,
            weather=str(world.get_weather()),
            fixed_delta_seconds=settings.fixed_delta_seconds,
        )
    finally:
        try:
            del client
        except Exception:
            pass


def main() -> int:
    args = parse_args()
    try:
        result = check_connection(args.host, args.port, args.timeout)
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1

    print("[PASS] CARLA connection established")
    print(f"server_version: {result.server_version}")
    print(f"client_version: {result.client_version}")
    print(f"map_name: {result.map_name}")
    print(f"weather: {result.weather}")
    print(f"fixed_delta_seconds: {result.fixed_delta_seconds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())