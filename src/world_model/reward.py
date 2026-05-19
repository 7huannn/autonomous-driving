"""Reward and terminal helpers for Stage 13 world-model rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any


@dataclass(frozen=True)
class RewardConfig:
    progress_weight: float = 1.0
    speed_weight: float = 0.05
    speed_norm_kmh: float = 40.0
    collision_weight: float = 5.0
    lane_invasion_penalty: float = 1.0
    offroad_penalty: float = 2.0
    steer_delta_weight: float = 0.02
    stuck_penalty: float = 0.5


@dataclass(frozen=True)
class TerminalConfig:
    collision_intensity_threshold: float = 0.75
    stuck_speed_kmh_threshold: float = 1.0
    stuck_frames_threshold: int = 20
    lane_invasion_frames_threshold: int = 5
    offroad_frames_threshold: int = 5


@dataclass(frozen=True)
class TerminalState:
    collision_frames: int = 0
    stuck_frames: int = 0
    lane_invasion_frames: int = 0
    offroad_frames: int = 0


def reward_config_from_dict(config: dict[str, Any]) -> RewardConfig:
    reward_cfg = config.get("reward", {}) if isinstance(config, dict) else {}
    if not isinstance(reward_cfg, dict):
        reward_cfg = {}
    return RewardConfig(**{k: reward_cfg.get(k, getattr(RewardConfig(), k)) for k in RewardConfig.__dataclass_fields__})


def terminal_config_from_dict(config: dict[str, Any]) -> TerminalConfig:
    term_cfg = config.get("terminal", {}) if isinstance(config, dict) else {}
    if not isinstance(term_cfg, dict):
        term_cfg = {}
    return TerminalConfig(
        **{k: term_cfg.get(k, getattr(TerminalConfig(), k)) for k in TerminalConfig.__dataclass_fields__}
    )


def speed_kmh_from_velocity(velocity_xyz: dict[str, float]) -> float:
    vx = float(velocity_xyz.get("x", 0.0))
    vy = float(velocity_xyz.get("y", 0.0))
    vz = float(velocity_xyz.get("z", 0.0))
    return 3.6 * sqrt(vx * vx + vy * vy + vz * vz)


def progress_m_from_locations(prev_location: dict[str, float] | None, cur_location: dict[str, float]) -> float:
    if prev_location is None:
        return 0.0
    dx = float(cur_location.get("x", 0.0)) - float(prev_location.get("x", 0.0))
    dy = float(cur_location.get("y", 0.0)) - float(prev_location.get("y", 0.0))
    dz = float(cur_location.get("z", 0.0)) - float(prev_location.get("z", 0.0))
    return sqrt(dx * dx + dy * dy + dz * dz)


def compute_reward(telemetry: dict[str, Any], steer_delta: float, cfg: RewardConfig) -> float:
    speed_kmh = float(telemetry.get("speed_kmh", 0.0))
    progress_m = float(telemetry.get("progress_m", 0.0))
    collision_intensity = float(telemetry.get("collision_intensity", 0.0))
    lane_invasion = bool(telemetry.get("lane_invasion", False))
    offroad = bool(telemetry.get("offroad", False))
    stuck = bool(telemetry.get("stuck", False))

    reward = 0.0
    reward += progress_m * cfg.progress_weight
    reward += min(speed_kmh / max(cfg.speed_norm_kmh, 1e-6), 1.0) * cfg.speed_weight
    reward -= collision_intensity * cfg.collision_weight
    reward -= float(lane_invasion) * cfg.lane_invasion_penalty
    reward -= float(offroad) * cfg.offroad_penalty
    reward -= abs(float(steer_delta)) * cfg.steer_delta_weight
    reward -= float(stuck) * cfg.stuck_penalty
    return float(reward)


def advance_terminal_state(
    prev_state: TerminalState,
    collision_intensity: float,
    speed_kmh: float,
    lane_invasion: bool,
    offroad: bool,
    cfg: TerminalConfig,
) -> TerminalState:
    collision_frames = prev_state.collision_frames + (1 if collision_intensity >= cfg.collision_intensity_threshold else 0)
    stuck_now = speed_kmh < cfg.stuck_speed_kmh_threshold
    stuck_frames = prev_state.stuck_frames + 1 if stuck_now else 0
    lane_frames = prev_state.lane_invasion_frames + 1 if lane_invasion else 0
    offroad_frames = prev_state.offroad_frames + 1 if offroad else 0
    return TerminalState(
        collision_frames=collision_frames,
        stuck_frames=stuck_frames,
        lane_invasion_frames=lane_frames,
        offroad_frames=offroad_frames,
    )


def terminal_decision(state: TerminalState, cfg: TerminalConfig, rollout_end: bool = False) -> tuple[bool, str | None]:
    if state.collision_frames > 0:
        return True, "collision"
    if state.stuck_frames >= cfg.stuck_frames_threshold:
        return True, "stuck"
    if state.offroad_frames >= cfg.offroad_frames_threshold:
        return True, "offroad"
    if state.lane_invasion_frames >= cfg.lane_invasion_frames_threshold:
        return True, "lane_invasion"
    if rollout_end:
        return True, "rollout_end"
    return False, None
