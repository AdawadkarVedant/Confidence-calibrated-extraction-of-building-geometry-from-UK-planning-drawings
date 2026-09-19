"""Turns 4 confidence thresholds into 3 scores the optimizer can minimise.

The 4 thresholds: one per stage (triage, scale, geometry, link), each 0-1.
A record only counts as "accepted" if ALL FOUR of its confidences clear
their threshold. If any stage fails, the record abstains right there.

The 3 scores (all "lower is better"):
  1. error rate   - how often accepted records are actually wrong
  2. 1 - coverage - how many records got no automated answer at all
  3. review cost  - how expensive the abstentions are (abstaining late
                     costs more than abstaining early)

expected_cost() turns this into money: a wrong answer costs more than an
abstention, and the exact ratio is a business input, not something this
code decides.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from contracts import (
    MANUAL_SURVEY_COST,
    REVIEW_COST,
    REVIEW_EFFORT_PRICE,
    PropertyRecord,
    Stage,
)

# A record counts as wrong if area is off by more than this, or storeys don't match.
AREA_TOLERANCE = 0.10


@dataclass
class EvaluationMatrix:
    """All records converted to plain numpy arrays, for fast repeated scoring."""

    confidences: np.ndarray      # (n_records, n_stages)
    correct: np.ndarray          # (n_records,) bool -- output within tolerance
    review_cost: np.ndarray      # (n_stages,) cost of abstaining at each stage
    record_ids: list[str]

    @property
    def n_records(self) -> int:
        return self.confidences.shape[0]

    @classmethod
    def from_records(cls, records: list[PropertyRecord]) -> "EvaluationMatrix":
        stages = Stage.ordered()
        conf = np.array([r.confidence_vector() for r in records], dtype=float)

        correct = []
        for r in records:
            truth_area = r.truth.get("ground_floor_area_m2")
            pred_area = r.prediction("ground_floor_area_m2")
            truth_st = r.truth.get("storeys")
            pred_st = r.prediction("storeys")
            if truth_area is None or pred_area is None:
                correct.append(False)
                continue
            area_ok = abs(pred_area - truth_area) / truth_area <= AREA_TOLERANCE
            storey_ok = (truth_st is None) or (pred_st == truth_st)
            correct.append(bool(area_ok and storey_ok))

        return cls(
            confidences=conf,
            correct=np.array(correct, dtype=bool),
            review_cost=np.array([REVIEW_COST[s] for s in stages], dtype=float),
            record_ids=[r.record_id for r in records],
        )

    # ---------------------------------------------------------------- core

    def route(self, thresholds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Decide accept/abstain for every record given 4 thresholds.

        Returns (accepted, abstain_stage). abstain_stage is which stage
        failed first (-1 if accepted).
        """
        passes = self.confidences >= thresholds[None, :]
        failed = ~passes
        any_fail = failed.any(axis=1)
        first_fail = np.where(any_fail, failed.argmax(axis=1), -1)
        return ~any_fail, first_fail

    def evaluate(self, thresholds: np.ndarray) -> np.ndarray:
        """Return the three objectives for one threshold vector."""
        accepted, first_fail = self.route(thresholds)
        n = self.n_records
        n_acc = int(accepted.sum())

        if n_acc == 0:
            # Accepting nothing gives a fake "perfect" 0% error rate.
            # Penalise it so the optimizer can't cheat this way.
            return np.array([1.0, 1.0, self.review_cost.max()])

        error_rate = 1.0 - self.correct[accepted].mean()
        coverage = n_acc / n
        cost = self.review_cost[first_fail[~accepted]].sum() / n

        return np.array([error_rate, 1.0 - coverage, cost])

    def evaluate_population(self, pop: np.ndarray) -> np.ndarray:
        return np.array([self.evaluate(ind) for ind in pop])

    # ------------------------------------------------- commercial selection

    def expected_cost(
        self,
        thresholds: np.ndarray,
        cost_wrong: float = 20.0,
        cost_review: float = 1.0,
    ) -> float:
        """Average cost per record, in units of one manual survey.

        3 cases per record:
          accepted and correct -> costs 0
          accepted and wrong   -> costs cost_wrong (a mispriced policy)
          abstained            -> costs 1 survey + a bit of review effort

        Abstaining is never free - that's what stops the optimizer from
        just abstaining on everything to get a fake 0% error rate.
        """
        accepted, first_fail = self.route(thresholds)
        n = self.n_records
        n_wrong = int((accepted & ~self.correct).sum())
        n_abstain = int((~accepted).sum())
        effort = self.review_cost[first_fail[~accepted]].sum()

        cost = (
            n_wrong * cost_wrong
            + n_abstain * MANUAL_SURVEY_COST * cost_review
            + effort * REVIEW_EFFORT_PRICE * cost_review
        )
        return cost / n

    def select_operating_point(
        self,
        front: np.ndarray,
        cost_wrong: float = 20.0,
        cost_review: float = 1.0,
    ) -> tuple[int, float]:
        """Pick the front member minimising expected cost. Returns (index, cost)."""
        costs = [self.expected_cost(t, cost_wrong, cost_review) for t in front]
        best = int(np.argmin(costs))
        return best, float(costs[best])

    def sweep_cost_ratio(
        self, front: np.ndarray, ratios: list[float]
    ) -> list[dict]:
        """Best operating point at each cost ratio, as a table."""
        rows = []
        for r in ratios:
            idx, cost = self.select_operating_point(front, cost_wrong=r, cost_review=1.0)
            t = front[idx]
            objs = self.evaluate(t)
            rows.append(
                {
                    "cost_ratio": r,
                    "front_index": idx,
                    "thresholds": t.round(3).tolist(),
                    "error_rate": round(float(objs[0]), 4),
                    "coverage": round(float(1 - objs[1]), 4),
                    "review_cost": round(float(objs[2]), 4),
                    "expected_cost": round(cost, 4),
                }
            )
        return rows


def best_shared_threshold(
    matrix: EvaluationMatrix, cost_wrong: float, resolution: int = 200
) -> tuple[float, float]:
    """Try every value for ONE shared threshold, return the best one found.

    This is the fair comparison baseline: "what if we used one bar for
    all 4 stages instead of tuning them separately?"
    """
    grid = np.linspace(0.0, 0.99, resolution)
    costs = [matrix.expected_cost(np.full(4, t), cost_wrong, 1.0) for t in grid]
    i = int(np.argmin(costs))
    return float(grid[i]), float(costs[i])


def grid_baseline(matrix: EvaluationMatrix, levels: int = 40) -> np.ndarray:
    """The whole curve for one shared threshold, at many levels (for plotting)."""
    ts = np.linspace(0.0, 0.95, levels)
    rows = [matrix.evaluate(np.full(4, t)) for t in ts]
    return np.array([r for r in rows if r[1] < 1.0])
