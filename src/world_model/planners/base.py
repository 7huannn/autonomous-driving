"""Planner interfaces for latent-space action sequence search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


EvaluateFn = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class PlannerSpec:
    action_order: tuple[str, str, str]
    horizon: int
    generations: int
    mutation_std: np.ndarray
    action_low: np.ndarray
    action_high: np.ndarray


@dataclass(frozen=True)
class PlannerResult:
    action_sequence: np.ndarray
    score: float


class PlannerBase:
    def __init__(self, spec: PlannerSpec, rng: np.random.Generator | None = None) -> None:
        self.spec = spec
        self.rng = rng or np.random.default_rng(42)

    def sample_sequence(self) -> np.ndarray:
        low = self.spec.action_low.reshape(1, -1)
        high = self.spec.action_high.reshape(1, -1)
        return self.rng.uniform(low=low, high=high, size=(self.spec.horizon, low.shape[1])).astype(np.float32)

    def clip_actions(self, seq: np.ndarray) -> np.ndarray:
        return np.clip(seq, self.spec.action_low, self.spec.action_high)

    def plan(self, evaluate_fn: EvaluateFn, prev_best_sequence: np.ndarray | None = None) -> PlannerResult:
        raise NotImplementedError
