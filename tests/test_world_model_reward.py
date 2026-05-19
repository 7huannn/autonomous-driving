from __future__ import annotations

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


def test_speed_and_progress_helpers() -> None:
    speed = speed_kmh_from_velocity({"x": 10.0, "y": 0.0, "z": 0.0})
    assert abs(speed - 36.0) < 1e-6

    progress = progress_m_from_locations({"x": 1.0, "y": 2.0, "z": 0.0}, {"x": 4.0, "y": 6.0, "z": 0.0})
    assert abs(progress - 5.0) < 1e-6


def test_reward_components_are_applied() -> None:
    cfg = RewardConfig()
    telemetry = {
        "speed_kmh": 20.0,
        "progress_m": 1.5,
        "collision_intensity": 0.2,
        "lane_invasion": True,
        "offroad": False,
        "stuck": True,
    }
    reward = compute_reward(telemetry=telemetry, steer_delta=0.5, cfg=cfg)
    expected = 0.0
    expected += 1.5 * cfg.progress_weight
    expected += (20.0 / 40.0) * cfg.speed_weight
    expected -= 0.2 * cfg.collision_weight
    expected -= 1.0 * cfg.lane_invasion_penalty
    expected -= 0.0 * cfg.offroad_penalty
    expected -= 0.5 * cfg.steer_delta_weight
    expected -= 1.0 * cfg.stuck_penalty
    assert abs(reward - expected) < 1e-9


def test_terminal_decision_priority() -> None:
    cfg = TerminalConfig(stuck_frames_threshold=3, offroad_frames_threshold=2, lane_invasion_frames_threshold=2)
    state = TerminalState()

    state = advance_terminal_state(
        prev_state=state,
        collision_intensity=0.0,
        speed_kmh=0.5,
        lane_invasion=False,
        offroad=False,
        cfg=cfg,
    )
    done, reason = terminal_decision(state, cfg)
    assert not done
    assert reason is None

    state = advance_terminal_state(
        prev_state=state,
        collision_intensity=0.0,
        speed_kmh=0.4,
        lane_invasion=True,
        offroad=False,
        cfg=cfg,
    )
    done, reason = terminal_decision(state, cfg)
    assert not done
    assert reason is None

    state = advance_terminal_state(
        prev_state=state,
        collision_intensity=0.0,
        speed_kmh=0.3,
        lane_invasion=True,
        offroad=False,
        cfg=cfg,
    )
    done, reason = terminal_decision(state, cfg)
    assert done
    assert reason == "stuck"


def test_collision_terminal_triggers_immediately() -> None:
    cfg = TerminalConfig(collision_intensity_threshold=0.75)
    state = advance_terminal_state(
        prev_state=TerminalState(),
        collision_intensity=1.0,
        speed_kmh=10.0,
        lane_invasion=False,
        offroad=False,
        cfg=cfg,
    )
    done, reason = terminal_decision(state, cfg)
    assert done
    assert reason == "collision"
