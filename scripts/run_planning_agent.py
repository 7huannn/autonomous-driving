#!/usr/bin/env python3
"""Live Stage 13B CARLA planning agent (simulator-only, closed-loop)."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import carla_recorder as recorder
from world_model import ensure_stage13_scope, load_yaml_config
from world_model.mdrnn import MDRNN
from world_model.planners import PlannerSpec, RHEAPlanner, RMHCPlanner
from world_model.reward import TerminalState, advance_terminal_state, compute_reward, progress_m_from_locations, speed_kmh_from_velocity, terminal_decision
from world_model.vae import ConvVAE

LOGGER = logging.getLogger("run_planning_agent")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 13B live CARLA planning/control")
    parser.add_argument("--config", type=Path, default=Path("configs/sensor_config.yaml"), help="Sensor config")
    parser.add_argument("--world-model-config", type=Path, default=Path("configs/world_model.yaml"), help="World model config")
    parser.add_argument("--vae-checkpoint", type=Path, required=True)
    parser.add_argument("--mdrnn-checkpoint", type=Path, required=True)
    parser.add_argument("--planner", choices=("rmhc", "rhea"), default="rmhc")
    parser.add_argument("--policy", choices=("planner", "random", "autopilot", "autopilot_noise"), default="planner")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--allow-stage13-control", action="store_true", help="Required for live_planning/iterative mode")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stop-on-done", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", help="Validate configs/checkpoints and write run summary without CARLA")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_vae(cfg: dict[str, Any], checkpoint: Path, device: torch.device) -> ConvVAE:
    vae_cfg = cfg.get("vae", {}) if isinstance(cfg.get("vae"), dict) else {}
    model = ConvVAE(
        image_size=int(vae_cfg.get("image_size", 64)),
        channels=int(vae_cfg.get("channels", 3)),
        latent_size=int(vae_cfg.get("latent_size", 64)),
    ).to(device)
    payload = torch.load(checkpoint.resolve(), map_location=device)
    state = payload.get("model", payload)
    model.load_state_dict(state)
    model.eval()
    return model


def load_mdrnn(cfg: dict[str, Any], checkpoint: Path, device: torch.device) -> MDRNN:
    mdrnn_cfg = cfg.get("mdrnn", {}) if isinstance(cfg.get("mdrnn"), dict) else {}
    payload = torch.load(checkpoint.resolve(), map_location=device)
    state = payload.get("model", payload)
    action_size = int(mdrnn_cfg.get("action_size", 3))
    hidden_size = int(state["rnn.weight_hh_l0"].shape[1]) if "rnn.weight_hh_l0" in state else int(
        mdrnn_cfg.get("hidden_size_final", mdrnn_cfg.get("hidden_size_smoke", 256))
    )
    num_gaussians = int(state["pi_head.weight"].shape[0]) if "pi_head.weight" in state else int(
        mdrnn_cfg.get("num_gaussians", 5)
    )
    latent_size = int(state["rnn.weight_ih_l0"].shape[1] - action_size) if "rnn.weight_ih_l0" in state else int(
        mdrnn_cfg.get("latent_size", 64)
    )
    model = MDRNN(
        latent_size=latent_size,
        action_size=action_size,
        hidden_size=hidden_size,
        num_gaussians=num_gaussians,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def planner_spec_from_config(cfg: dict[str, Any]) -> PlannerSpec:
    planner_cfg = cfg.get("planner", {}) if isinstance(cfg.get("planner"), dict) else {}
    mut = planner_cfg.get("mutation_std", {})
    bounds = planner_cfg.get("action_bounds", {})
    action_order = tuple(planner_cfg.get("action_order", ["steer", "throttle", "brake"]))
    return PlannerSpec(
        action_order=action_order,
        horizon=int(planner_cfg.get("horizon", 20)),
        generations=int(planner_cfg.get("generations", 10)),
        mutation_std=np.array(
            [
                float(mut.get("steer", 0.15)),
                float(mut.get("throttle", 0.10)),
                float(mut.get("brake", 0.10)),
            ],
            dtype=np.float32,
        ),
        action_low=np.array(
            [
                float(bounds.get("steer", [-1.0, 1.0])[0]),
                float(bounds.get("throttle", [0.0, 1.0])[0]),
                float(bounds.get("brake", [0.0, 1.0])[0]),
            ],
            dtype=np.float32,
        ),
        action_high=np.array(
            [
                float(bounds.get("steer", [-1.0, 1.0])[1]),
                float(bounds.get("throttle", [0.0, 1.0])[1]),
                float(bounds.get("brake", [0.0, 1.0])[1]),
            ],
            dtype=np.float32,
        ),
    )


def build_planner(name: str, spec: PlannerSpec, seed: int) -> Any:
    rng = np.random.default_rng(seed)
    if name == "rmhc":
        return RMHCPlanner(spec=spec, rng=rng)
    return RHEAPlanner(spec=spec, rng=rng)


def encode_rgb_to_latent(frame_bgr: np.ndarray, vae: ConvVAE, device: torch.device, image_size: int) -> torch.Tensor:
    resized = cv2.resize(frame_bgr, (image_size, image_size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy((rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        mu, _logvar = vae.encode(x)
    return mu.squeeze(0)


def clone_hidden(hidden: tuple[torch.Tensor, torch.Tensor] | None) -> tuple[torch.Tensor, torch.Tensor] | None:
    if hidden is None:
        return None
    return hidden[0].clone(), hidden[1].clone()


def latent_rollout_score(
    model: MDRNN,
    start_latent: torch.Tensor,
    sequence: np.ndarray,
    device: torch.device,
    hidden: tuple[torch.Tensor, torch.Tensor] | None,
) -> float:
    with torch.no_grad():
        z = start_latent.clone()
        h = clone_hidden(hidden)
        total_reward = 0.0
        for action in sequence:
            z_t = z.view(1, 1, -1)
            a_t = torch.from_numpy(action.astype(np.float32)).to(device).view(1, 1, -1)
            preds, h = model(z_t, a_t, h)
            pi = torch.softmax(preds["pi_logits"][0, 0], dim=-1)
            k = int(torch.argmax(pi).item())
            z = preds["mu"][0, 0, k]
            total_reward += float(preds["reward"][0, 0].item())
            done_prob = float(torch.sigmoid(preds["done_logits"][0, 0]).item())
            if done_prob > 0.5:
                break
    return float(total_reward)


def action_to_control(carla: Any, action: np.ndarray) -> Any:
    return carla.VehicleControl(
        steer=float(action[0]),
        throttle=float(action[1]),
        brake=float(action[2]),
        hand_brake=False,
        reverse=False,
    )


def write_rollout_manifest(output_dir: Path, summary: dict[str, Any], world_model_cfg: dict[str, Any], policy: str) -> None:
    rollout_cfg = world_model_cfg.get("rollout", {}) if isinstance(world_model_cfg.get("rollout"), dict) else {}
    planner_cfg = world_model_cfg.get("planner", {}) if isinstance(world_model_cfg.get("planner"), dict) else {}
    manifest = {
        "rollout_id": output_dir.name,
        "policy": str(policy),
        "map": summary["carla"]["map"],
        "weather": summary["carla"]["weather"],
        "frames": int(summary["frames_recorded"]),
        "action_order": list(planner_cfg.get("action_order", rollout_cfg.get("action_order", ["steer", "throttle", "brake"]))),
        "reward_version": str(rollout_cfg.get("reward_version", "carla_progress_v1")),
        "terminal_version": str(rollout_cfg.get("terminal_version", "carla_terminal_v1")),
    }
    (output_dir / "rollout_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def run_live(args: argparse.Namespace, config: dict[str, Any], world_model_cfg: dict[str, Any], stage13_mode: str) -> dict[str, Any]:
    if args.max_frames <= 0:
        raise ValueError("--max-frames must be > 0")

    output_dir = args.output_dir.resolve()
    recorder.prepare_output_dir(output_dir, overwrite=bool(args.overwrite))

    world_model_runtime = recorder.load_world_model_runtime(
        world_model_config_path=args.world_model_config.resolve(),
        sensor_config=config,
        allow_stage13_control=bool(args.allow_stage13_control),
        policy=str(args.policy),
    )

    carla = recorder.import_carla()
    host = str(recorder.deep_get(config, "carla", "host", default="localhost"))
    port = int(recorder.deep_get(config, "carla", "port", default=2000))
    map_name = recorder.deep_get(config, "carla", "map")
    weather_name = recorder.deep_get(config, "carla", "weather")
    timeout = float(recorder.deep_get(config, "carla", "timeout", default=10.0))
    startup_wait_seconds = float(recorder.deep_get(config, "carla", "startup_wait_seconds", default=30.0))
    tm_port = int(recorder.deep_get(config, "carla", "tm_port", default=10000))
    fixed_delta_seconds = float(recorder.deep_get(config, "recording", "fixed_delta_seconds", default=0.1))
    sensor_timeout = float(recorder.deep_get(config, "recording", "sensor_timeout", default=5.0))
    warmup_ticks = int(recorder.deep_get(config, "recording", "warmup_ticks", default=5))
    use_synchronous_mode = bool(recorder.deep_get(config, "recording", "use_synchronous_mode", default=True))
    actor_label_max_distance = float(recorder.deep_get(config, "recording", "actor_label_max_distance_m", default=75.0))
    traffic_vehicles = int(recorder.deep_get(config, "recording", "traffic_vehicles", default=0))
    traffic_walkers = int(recorder.deep_get(config, "recording", "traffic_walkers", default=0))
    prefer_cyclist_vehicles = int(recorder.deep_get(config, "recording", "prefer_cyclist_vehicles", default=0))

    if not use_synchronous_mode:
        raise RuntimeError("Stage 13B live planning requires recording.use_synchronous_mode=true")

    device = resolve_device(args.device)
    vae = load_vae(world_model_cfg, args.vae_checkpoint, device)
    mdrnn = load_mdrnn(world_model_cfg, args.mdrnn_checkpoint, device)
    vae_cfg = world_model_cfg.get("vae", {}) if isinstance(world_model_cfg.get("vae"), dict) else {}
    image_size = int(vae_cfg.get("image_size", 64))

    spec = planner_spec_from_config(world_model_cfg)
    planner = build_planner(args.planner, spec=spec, seed=args.seed)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    client, world = recorder.connect_to_carla(
        carla=carla,
        host=host,
        port=port,
        timeout=timeout,
        startup_wait_seconds=startup_wait_seconds,
    )

    target_map, _ = recorder.resolve_runtime_map(client, str(map_name) if map_name is not None else None)
    if target_map:
        current_map = recorder.short_map_name(world.get_map().name)
        if current_map != target_map:
            LOGGER.info("Loading map '%s' (current map: %s)", target_map, current_map)
            world = client.load_world(target_map)

    if weather_name:
        weather_obj = getattr(carla.WeatherParameters, str(weather_name), None)
        if weather_obj is None:
            raise RuntimeError(f"Unknown weather preset: {weather_name}")
        world.set_weather(weather_obj)

    original_settings = world.get_settings()
    traffic_manager = client.get_trafficmanager(tm_port)

    ego_vehicle = None
    sensors: list[Any] = []
    event_sensors: list[Any] = []
    traffic: list[Any] = []
    walkers: list[Any] = []
    controllers: list[Any] = []
    all_spawned: list[Any] = []

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = fixed_delta_seconds
        world.apply_settings(settings)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_random_device_seed(args.seed)

        ego_vehicle = recorder.spawn_ego_vehicle(carla, world, args.seed)
        all_spawned.append(ego_vehicle)

        # Autopilot only for baseline policy.
        if args.policy in {"autopilot", "autopilot_noise"}:
            ego_vehicle.set_autopilot(True, tm_port)
        else:
            ego_vehicle.set_autopilot(False, tm_port)

        traffic = recorder.spawn_traffic_vehicles(
            carla=carla,
            client=client,
            world=world,
            tm_port=tm_port,
            count=traffic_vehicles,
            seed=args.seed,
            prefer_cyclist_vehicles=prefer_cyclist_vehicles,
        )
        all_spawned.extend(traffic)
        walkers, controllers = recorder.spawn_walkers(carla, client, world, traffic_walkers, args.seed)
        all_spawned.extend(walkers)

        sensors, queues = recorder.spawn_sensors(carla, world, ego_vehicle, config)
        event_sensors, event_state = recorder.spawn_event_sensors(world, ego_vehicle, world_model_runtime)
        all_spawned.extend(event_sensors)

        for _ in range(warmup_ticks):
            world.tick()

        terminal_state = TerminalState()
        prev_location: dict[str, float] | None = None
        prev_control: dict[str, float] | None = None
        prev_best_seq: np.ndarray | None = None
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None

        planner_scores: list[float] = []
        random_scores: list[float] = []
        episode_reward = 0.0
        done_reason: str | None = None
        frames_recorded = 0

        started = time.monotonic()
        for frame_index in range(args.max_frames):
            frame = world.tick()
            snapshot = world.get_snapshot()
            timestamp = float(snapshot.timestamp.elapsed_seconds)
            sensor_data = {
                name: recorder.wait_for_sensor_frame(name, queues[name], frame, sensor_timeout)
                for name in recorder.SENSOR_NAMES
            }

            rgb_array = np.frombuffer(sensor_data["rgb_camera"].raw_data, dtype=np.uint8).reshape(
                (sensor_data["rgb_camera"].height, sensor_data["rgb_camera"].width, 4)
            )[:, :, :3]
            z_t = encode_rgb_to_latent(rgb_array, vae=vae, device=device, image_size=image_size)

            if args.policy == "planner":
                eval_fn = lambda seq: latent_rollout_score(mdrnn, z_t, seq, device, hidden)
                result = planner.plan(evaluate_fn=eval_fn, prev_best_sequence=prev_best_seq)
                prev_best_seq = result.action_sequence
                action = result.action_sequence[0]
                planner_scores.append(float(result.score))
                baseline_trials = max(10, spec.generations)
                baseline = [float(eval_fn(planner.sample_sequence())) for _ in range(baseline_trials)]
                random_scores.append(float(np.mean(baseline)))
                ego_vehicle.apply_control(action_to_control(carla, action))
            elif args.policy == "random":
                sampled = planner.sample_sequence()
                action = sampled[0]
                ego_vehicle.apply_control(action_to_control(carla, action))
            else:
                ctrl = ego_vehicle.get_control()
                action = np.array([float(ctrl.steer), float(ctrl.throttle), float(ctrl.brake)], dtype=np.float32)

            if args.policy in {"planner", "random"}:
                with torch.no_grad():
                    z_input = z_t.view(1, 1, -1)
                    a_input = torch.from_numpy(action.astype(np.float32)).to(device).view(1, 1, -1)
                    _preds, hidden = mdrnn(z_input, a_input, hidden)

            ego_snapshot = snapshot.find(ego_vehicle.id)
            if ego_snapshot is None:
                raise RuntimeError(f"ego actor id={ego_vehicle.id} missing from snapshot")
            ego_state = recorder.snapshot_actor_to_dict(ego_snapshot)

            control = {
                "steer": float(action[0]),
                "throttle": float(action[1]),
                "brake": float(action[2]),
                "reverse": False,
                "hand_brake": False,
                "manual_gear_shift": False,
            }
            steer_delta = 0.0 if prev_control is None else float(control["steer"] - prev_control["steer"])
            prev_control = control

            location = ego_state.get("location", {}) if isinstance(ego_state.get("location", {}), dict) else {}
            velocity = ego_state.get("velocity", {}) if isinstance(ego_state.get("velocity", {}), dict) else {}
            speed_kmh = speed_kmh_from_velocity(velocity)
            progress_m = progress_m_from_locations(prev_location, location)
            prev_location = location

            waypoint = world.get_map().get_waypoint(ego_vehicle.get_location(), project_to_road=False)
            offroad = waypoint is None
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
                rollout_end=(frame_index + 1 == args.max_frames),
            )
            world_model = {
                "reward": float(reward_value),
                "done": bool(done),
                "done_reason": done_reason,
            }
            episode_reward += float(reward_value)

            stem = f"{frame_index:06d}"
            recorder.save_camera_image(sensor_data["rgb_camera"], output_dir / "rgb" / f"{stem}.png")
            recorder.save_camera_image(sensor_data["semantic_camera"], output_dir / "semantic" / f"{stem}.png")
            recorder.save_lidar(sensor_data["lidar"], output_dir / "lidar" / f"{stem}.npy")
            actors = recorder.collect_nearby_actor_boxes(
                world=world,
                snapshot=snapshot,
                ego_vehicle=ego_vehicle,
                max_distance_m=actor_label_max_distance,
            )
            recorder.write_metadata(
                output_dir / "metadata" / f"{stem}.json",
                frame=frame,
                timestamp=timestamp,
                ego_state=ego_state,
                sensor_frames={name: int(sensor_data[name].frame) for name in recorder.SENSOR_NAMES},
                control=control,
                telemetry=telemetry,
                world_model=world_model,
                actors=actors,
            )
            event_state["collision_intensity"] = 0.0
            event_state["lane_invasion"] = False
            frames_recorded += 1

            if done and args.stop_on_done:
                break

        elapsed = max(1e-9, time.monotonic() - started)
        runtime_map = recorder.short_map_name(world.get_map().name)
        sensors_file = recorder.write_calibration_file(output_dir=output_dir, config=config)
        scenario_file = recorder.write_scenario_file(
            output_dir=output_dir,
            scenario={
                "map": runtime_map,
                "weather": weather_name,
                "seed": args.seed,
                "traffic_manager_port": tm_port,
                "traffic_vehicles_requested": traffic_vehicles,
                "traffic_walkers_requested": traffic_walkers,
                "traffic_vehicles_spawned": len(traffic),
                "traffic_walkers_spawned": len(walkers),
                "use_synchronous_mode": True,
                "fixed_delta_seconds": fixed_delta_seconds,
                "sensor_timeout": sensor_timeout,
                "world_model": {
                    "policy": args.policy,
                    "stage13_mode": stage13_mode,
                    "planner": args.planner,
                },
            },
        )

        integrity = recorder.validate_recording_integrity(output_dir=output_dir, expected_frames=frames_recorded)
        planning_metrics = {
            "planner": args.planner,
            "policy": args.policy,
            "planner_scores": planner_scores,
            "random_scores": random_scores,
            "mean_planner_pred_reward": float(np.mean(planner_scores)) if planner_scores else None,
            "mean_random_pred_reward": float(np.mean(random_scores)) if random_scores else None,
            "planner_beats_random": bool(np.mean(planner_scores) > np.mean(random_scores)) if planner_scores and random_scores else None,
        }
        (output_dir / "planning_metrics.json").write_text(json.dumps(planning_metrics, indent=2), encoding="utf-8")

        summary = {
            "output_dir": str(output_dir),
            "frames_target": int(args.max_frames),
            "frames_recorded": int(frames_recorded),
            "elapsed_seconds": float(elapsed),
            "fps_wall_clock": float(frames_recorded / elapsed),
            "carla": {
                "host": host,
                "port": port,
                "map": runtime_map,
                "weather": weather_name,
                "tm_port": tm_port,
            },
            "recording": {
                "use_synchronous_mode": True,
                "fixed_delta_seconds": fixed_delta_seconds,
                "sensor_timeout": sensor_timeout,
            },
            "world_model": {
                "stage13_mode": stage13_mode,
                "policy": args.policy,
                "planner": args.planner,
                "episode_reward": float(episode_reward),
                "done_reason": done_reason,
                "reward_version": "carla_progress_v1",
                "terminal_version": "carla_terminal_v1",
            },
            "planning_metrics": planning_metrics,
            "artifacts": {
                "scenario_file": str(scenario_file),
                "sensors_file": str(sensors_file),
                "planning_metrics": str(output_dir / "planning_metrics.json"),
            },
            "integrity": integrity,
            "complete": bool(integrity["pass"]),
        }
        (output_dir / "recording_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (output_dir / "dataset_complete.json").write_text(
            json.dumps(
                {
                    "complete": bool(integrity["pass"]),
                    "reason": None if integrity["pass"] else "Integrity checks failed",
                    "integrity": integrity,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        write_rollout_manifest(output_dir, summary, world_model_cfg=world_model_cfg, policy=args.policy)
        return summary
    finally:
        server_alive = recorder.carla_server_alive(client)
        if server_alive:
            recorder.cleanup_actors(carla, client, sensors, controllers, all_spawned)
            try:
                traffic_manager.set_synchronous_mode(False)
            except Exception as exc:  # pragma: no cover - server-dependent cleanup
                LOGGER.warning("Failed to disable TM synchronous mode: %s", exc)
            try:
                world.apply_settings(original_settings)
            except Exception as exc:  # pragma: no cover - server-dependent cleanup
                LOGGER.warning("Failed to restore world settings: %s", exc)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    config = recorder.load_config(args.config.resolve())
    recorder.validate_config(config)
    world_model_cfg = load_yaml_config(args.world_model_config.resolve())
    stage13_mode = ensure_stage13_scope(world_model_cfg, allow_stage13_control=bool(args.allow_stage13_control))
    if stage13_mode not in {"live_planning", "iterative"}:
        raise ValueError(f"run_planning_agent.py requires stage13.mode live_planning/iterative, got '{stage13_mode}'")

    if not args.vae_checkpoint.resolve().is_file():
        raise FileNotFoundError(f"VAE checkpoint not found: {args.vae_checkpoint}")
    if not args.mdrnn_checkpoint.resolve().is_file():
        raise FileNotFoundError(f"MDRNN checkpoint not found: {args.mdrnn_checkpoint}")

    args.output_dir.resolve().mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        summary = {
            "output_dir": str(args.output_dir.resolve()),
            "dry_run": True,
            "stage13_mode": stage13_mode,
            "policy": args.policy,
            "planner": args.planner,
            "max_frames": int(args.max_frames),
        }
        (args.output_dir.resolve() / "recording_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("[PASS] run_planning_agent dry-run complete")
        print(json.dumps(summary, indent=2))
        return 0

    summary = run_live(args=args, config=config, world_model_cfg=world_model_cfg, stage13_mode=stage13_mode)
    print("[PASS] run_planning_agent complete")
    print(f"frames_recorded: {summary['frames_recorded']}")
    print(f"episode_reward: {summary['world_model']['episode_reward']:.6f}")
    print(f"done_reason: {summary['world_model']['done_reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
