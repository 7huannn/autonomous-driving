"""Shift-buffered rolling-horizon mutation hill climbing (RMHC)."""

from __future__ import annotations

import numpy as np

from .base import EvaluateFn, PlannerBase, PlannerResult


class RMHCPlanner(PlannerBase):
    def plan(self, evaluate_fn: EvaluateFn, prev_best_sequence: np.ndarray | None = None) -> PlannerResult:
        if prev_best_sequence is not None and prev_best_sequence.shape == (self.spec.horizon, len(self.spec.action_order)):
            seed = np.vstack([prev_best_sequence[1:], prev_best_sequence[-1:]]).astype(np.float32)
            sequence = self.clip_actions(seed)
        else:
            sequence = self.sample_sequence()

        best_sequence = sequence.copy()
        best_score = float(evaluate_fn(best_sequence))

        for _ in range(self.spec.generations):
            candidate = best_sequence.copy()
            t = int(self.rng.integers(0, self.spec.horizon))
            noise = self.rng.normal(loc=0.0, scale=self.spec.mutation_std, size=(len(self.spec.action_order),)).astype(np.float32)
            candidate[t] = candidate[t] + noise
            candidate = self.clip_actions(candidate)
            score = float(evaluate_fn(candidate))
            if score > best_score:
                best_score = score
                best_sequence = candidate

        return PlannerResult(action_sequence=best_sequence, score=best_score)
