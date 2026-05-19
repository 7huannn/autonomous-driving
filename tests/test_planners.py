from __future__ import annotations

import numpy as np

from world_model.planners.base import PlannerSpec
from world_model.planners.rhea import RHEAPlanner
from world_model.planners.rmhc import RMHCPlanner


def toy_eval(sequence: np.ndarray) -> float:
    target = np.zeros_like(sequence)
    return -float(np.mean((sequence - target) ** 2))


def build_spec() -> PlannerSpec:
    return PlannerSpec(
        action_order=("steer", "throttle", "brake"),
        horizon=10,
        generations=40,
        mutation_std=np.array([0.2, 0.1, 0.1], dtype=np.float32),
        action_low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        action_high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
    )


def test_rmhc_beats_random_baseline() -> None:
    spec = build_spec()
    planner = RMHCPlanner(spec=spec, rng=np.random.default_rng(7))
    result = planner.plan(toy_eval)
    random_scores = [toy_eval(planner.sample_sequence()) for _ in range(64)]
    assert result.score > float(np.mean(random_scores))
    assert result.action_sequence.shape == (spec.horizon, 3)


def test_rhea_respects_action_bounds() -> None:
    spec = build_spec()
    planner = RHEAPlanner(spec=spec, rng=np.random.default_rng(11), population_size=12, elite_size=3)
    result = planner.plan(toy_eval)
    assert result.action_sequence.shape == (spec.horizon, 3)
    assert np.all(result.action_sequence[:, 0] >= -1.0)
    assert np.all(result.action_sequence[:, 0] <= 1.0)
    assert np.all(result.action_sequence[:, 1:] >= 0.0)
    assert np.all(result.action_sequence[:, 1:] <= 1.0)
