"""Simple population-based Rolling Horizon Evolutionary Algorithm (RHEA)."""

from __future__ import annotations

import numpy as np

from .base import EvaluateFn, PlannerBase, PlannerResult


class RHEAPlanner(PlannerBase):
    def __init__(self, *args, population_size: int = 16, elite_size: int = 4, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.population_size = int(population_size)
        self.elite_size = int(elite_size)

    def plan(self, evaluate_fn: EvaluateFn, prev_best_sequence: np.ndarray | None = None) -> PlannerResult:
        population = [self.sample_sequence() for _ in range(self.population_size)]
        if prev_best_sequence is not None and prev_best_sequence.shape == (self.spec.horizon, len(self.spec.action_order)):
            shifted = np.vstack([prev_best_sequence[1:], prev_best_sequence[-1:]]).astype(np.float32)
            population[0] = self.clip_actions(shifted)

        best_seq = population[0]
        best_score = float("-inf")

        for _ in range(self.spec.generations):
            scores = np.array([float(evaluate_fn(seq)) for seq in population], dtype=np.float32)
            order = np.argsort(scores)[::-1]
            elites = [population[i].copy() for i in order[: self.elite_size]]
            if float(scores[order[0]]) > best_score:
                best_score = float(scores[order[0]])
                best_seq = population[int(order[0])].copy()

            new_population = elites.copy()
            while len(new_population) < self.population_size:
                parent = elites[int(self.rng.integers(0, len(elites)))].copy()
                t = int(self.rng.integers(0, self.spec.horizon))
                parent[t] = parent[t] + self.rng.normal(0.0, self.spec.mutation_std, size=parent.shape[1])
                new_population.append(self.clip_actions(parent.astype(np.float32)))
            population = new_population

        return PlannerResult(action_sequence=best_seq, score=best_score)
