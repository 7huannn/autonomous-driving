#!/usr/bin/env python3
"""Regenerate canonical Stage 03 raw datasets with Town10 class-gate fallback."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate canonical Stage 03 datasets")
    parser.add_argument("--python-exe", default=sys.executable, help="Python executable for recorder subprocess")
    parser.add_argument("--config", type=Path, default=Path("configs/sensor_config.yaml"), help="Recorder config path")

    parser.add_argument("--mine-map", default="Mine_01", help="CARLA map for main stable run")
    parser.add_argument("--mine-output", type=Path, default=Path("data/raw/stage03_mine_main_1000"))
    parser.add_argument("--mine-frames", type=int, default=1000)

    parser.add_argument("--town-map", default="Town10HD_Opt", help="CARLA map for aux diversity run")
    parser.add_argument("--town-output", type=Path, default=Path("data/raw/stage03_town10_aux_200"))
    parser.add_argument("--town-frames", type=int, default=200)

    parser.add_argument("--town-extra-root", type=Path, default=Path("data/raw/stage03_town10_aux_extra"))
    parser.add_argument("--town-extra-chunk", type=int, default=50)
    parser.add_argument("--town-extra-max", type=int, default=300)

    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--tm-port", type=int, default=10000)

    parser.add_argument("--mine-traffic-vehicles", type=int, default=0)
    parser.add_argument("--mine-traffic-walkers", type=int, default=0)
    parser.add_argument("--mine-prefer-cyclist-vehicles", type=int, default=0)
    parser.add_argument("--town-traffic-vehicles", type=int, default=20)
    parser.add_argument("--town-traffic-walkers", type=int, default=10)
    parser.add_argument("--town-prefer-cyclist-vehicles", type=int, default=3)

    parser.add_argument("--overwrite", action="store_true", help="Force overwrite target dataset directories")
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("output/stage_validation/stage03_regeneration_report.json"),
        help="Where to write regeneration report",
    )
    return parser.parse_args()


def run_cmd(cmd: list[str]) -> None:
    print("[CMD]", " ".join(cmd))
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"Command failed with code {proc.returncode}: {' '.join(cmd)}")


def run_recording(
    python_exe: str,
    config: Path,
    map_name: str,
    output_dir: Path,
    num_frames: int,
    timeout: float,
    tm_port: int,
    traffic_vehicles: int,
    traffic_walkers: int,
    prefer_cyclist_vehicles: int,
    overwrite: bool,
) -> None:
    cmd = [
        python_exe,
        "scripts/carla_recorder.py",
        "--config",
        str(config),
        "--map",
        map_name,
        "--output-dir",
        str(output_dir),
        "--num-frames",
        str(num_frames),
        "--timeout",
        str(timeout),
        "--tm-port",
        str(tm_port),
        "--traffic-vehicles",
        str(traffic_vehicles),
        "--traffic-walkers",
        str(traffic_walkers),
        "--prefer-cyclist-vehicles",
        str(prefer_cyclist_vehicles),
    ]
    if overwrite:
        cmd.append("--overwrite")
    run_cmd(cmd)


def inspect_recording_dir(output_dir: Path) -> dict[str, Any]:
    rgb = len(list((output_dir / "rgb").glob("*.png"))) if (output_dir / "rgb").is_dir() else 0
    sem = len(list((output_dir / "semantic").glob("*.png"))) if (output_dir / "semantic").is_dir() else 0
    lidar = len(list((output_dir / "lidar").glob("*.npy"))) if (output_dir / "lidar").is_dir() else 0
    meta = len(list((output_dir / "metadata").glob("*.json"))) if (output_dir / "metadata").is_dir() else 0
    aligned = (rgb == sem == lidar == meta and rgb > 0)
    return {
        "frames_recorded": rgb if aligned else 0,
        "aligned": aligned,
        "counts": {"rgb": rgb, "semantic": sem, "lidar": lidar, "metadata": meta},
    }


def run_recording_with_fallback(
    python_exe: str,
    config: Path,
    map_name: str,
    output_dir: Path,
    num_frames: int,
    timeout: float,
    tm_port: int,
    base_vehicles: int,
    base_walkers: int,
    base_prefer_cyclists: int,
    overwrite: bool,
    allow_partial: bool = False,
    min_partial_frames: int = 0,
) -> dict[str, Any]:
    profiles = [
        (base_vehicles, base_walkers, base_prefer_cyclists),
        (max(8, base_vehicles // 2), max(0, base_walkers // 2), max(1, base_prefer_cyclists // 2)),
        (6, 0, 2),
        (3, 0, 1),
    ]
    seen = set()
    deduped_profiles = []
    for profile in profiles:
        if profile in seen:
            continue
        seen.add(profile)
        deduped_profiles.append(profile)

    last_error: Exception | None = None
    for vehicles, walkers, prefer_cyclists in deduped_profiles:
        try:
            run_recording(
                python_exe=python_exe,
                config=config,
                map_name=map_name,
                output_dir=output_dir,
                num_frames=num_frames,
                timeout=timeout,
                tm_port=tm_port,
                traffic_vehicles=vehicles,
                traffic_walkers=walkers,
                prefer_cyclist_vehicles=prefer_cyclists,
                overwrite=overwrite,
            )
            info = inspect_recording_dir(output_dir)
            return {
                "traffic_vehicles": vehicles,
                "traffic_walkers": walkers,
                "prefer_cyclist_vehicles": prefer_cyclists,
                "partial": False,
                "frames_recorded": int(info["frames_recorded"]),
            }
        except Exception as exc:
            last_error = exc
            partial_info = inspect_recording_dir(output_dir)
            if allow_partial and partial_info["aligned"] and int(partial_info["frames_recorded"]) >= int(min_partial_frames):
                print(
                    f"[WARN] Accepting partial recording for {output_dir}: "
                    f"{partial_info['frames_recorded']} frames (aligned) after error: {exc}"
                )
                return {
                    "traffic_vehicles": vehicles,
                    "traffic_walkers": walkers,
                    "prefer_cyclist_vehicles": prefer_cyclists,
                    "partial": True,
                    "frames_recorded": int(partial_info["frames_recorded"]),
                    "error": str(exc),
                }
            if output_dir.exists():
                shutil.rmtree(output_dir, ignore_errors=True)
            print(
                f"[WARN] Recording failed for profile vehicles={vehicles}, walkers={walkers}, "
                f"prefer_cyclists={prefer_cyclists}: {exc}"
            )
    if last_error is None:
        raise RuntimeError("No fallback profiles available")
    raise RuntimeError(f"All recording fallback profiles failed for {output_dir}: {last_error}") from last_error


def aggregate_actor_hist(dataset_dirs: list[Path]) -> dict[str, Any]:
    class_hist: dict[str, int] = {}
    frames_total = 0
    frames_non_empty = 0

    for dataset_dir in dataset_dirs:
        meta_dir = dataset_dir / "metadata"
        files = sorted(meta_dir.glob("*.json")) if meta_dir.is_dir() else []
        for mf in files:
            frames_total += 1
            data = json.loads(mf.read_text(encoding="utf-8"))
            actors = data.get("actors", [])
            if isinstance(actors, list) and actors:
                frames_non_empty += 1
                for actor in actors:
                    if not isinstance(actor, dict):
                        continue
                    cls = str(actor.get("class_name", "Unknown"))
                    class_hist[cls] = class_hist.get(cls, 0) + 1

    has_vehicle = class_hist.get("Vehicle", 0) > 0
    has_secondary = (class_hist.get("Pedestrian", 0) > 0) or (class_hist.get("Cyclist", 0) > 0)

    return {
        "class_hist": class_hist,
        "frames_total": frames_total,
        "frames_non_empty": frames_non_empty,
        "has_vehicle": has_vehicle,
        "has_secondary": has_secondary,
        "gate_pass": has_vehicle and has_secondary,
    }


def main() -> int:
    args = parse_args()

    config = args.config.resolve()
    mine_output = args.mine_output.resolve()
    town_output = args.town_output.resolve()
    town_extra_root = args.town_extra_root.resolve()

    mine_profile = run_recording_with_fallback(
        python_exe=args.python_exe,
        config=config,
        map_name=args.mine_map,
        output_dir=mine_output,
        num_frames=args.mine_frames,
        timeout=args.timeout,
        tm_port=args.tm_port,
        base_vehicles=args.mine_traffic_vehicles,
        base_walkers=args.mine_traffic_walkers,
        base_prefer_cyclists=args.mine_prefer_cyclist_vehicles,
        overwrite=args.overwrite,
    )

    town_base_profile = run_recording_with_fallback(
        python_exe=args.python_exe,
        config=config,
        map_name=args.town_map,
        output_dir=town_output,
        num_frames=args.town_frames,
        timeout=args.timeout,
        tm_port=args.tm_port,
        base_vehicles=args.town_traffic_vehicles,
        base_walkers=args.town_traffic_walkers,
        base_prefer_cyclists=args.town_prefer_cyclist_vehicles,
        overwrite=args.overwrite,
        allow_partial=True,
        min_partial_frames=5,
    )

    town_datasets = [town_output]
    town_profiles = [town_base_profile]
    gate = aggregate_actor_hist(town_datasets)

    extra_requested_total = 0
    chunk_index = 0
    while (
        gate["frames_total"] < args.town_frames or not gate["gate_pass"]
    ) and extra_requested_total < args.town_extra_max:
        chunk_index += 1
        frames = min(args.town_extra_chunk, args.town_extra_max - extra_requested_total)
        extra_out = Path(f"{town_extra_root}_{chunk_index:02d}_{frames:03d}")
        profile = run_recording_with_fallback(
            python_exe=args.python_exe,
            config=config,
            map_name=args.town_map,
            output_dir=extra_out,
            num_frames=frames,
            timeout=args.timeout,
            tm_port=args.tm_port,
            base_vehicles=args.town_traffic_vehicles,
            base_walkers=args.town_traffic_walkers,
            base_prefer_cyclists=args.town_prefer_cyclist_vehicles,
            overwrite=args.overwrite,
            allow_partial=True,
            min_partial_frames=5,
        )
        town_datasets.append(extra_out)
        town_profiles.append(profile)
        extra_requested_total += frames
        gate = aggregate_actor_hist(town_datasets)

    report = {
        "mine_dataset": str(mine_output),
        "mine_profile_used": mine_profile,
        "town_base_dataset": str(town_output),
        "town_profile_used_base": town_base_profile,
        "town_datasets_used": [str(p) for p in town_datasets],
        "town_profiles_used": town_profiles,
        "town_extra_frames_requested_total": extra_requested_total,
        "town_frames_target_min": args.town_frames,
        "town_gate": gate,
        "town_gate_target": "Vehicle plus at least one of Pedestrian/Cyclist",
    }
    report["overall_ready"] = bool(gate["gate_pass"] and gate["frames_total"] >= args.town_frames)

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"[PASS] Report saved: {args.report_json.resolve()}")

    return 0 if report["overall_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
