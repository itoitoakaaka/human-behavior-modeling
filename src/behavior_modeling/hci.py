"""Minimal HCI evaluation helpers for intervention prototypes."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class HCIEvaluation:
    effectiveness: float
    perceived_usefulness: float
    cognitive_load: float
    autonomy_cost: float
    composite_score: float


def evaluate_intervention(
    outcome_before,
    outcome_after,
    usefulness_ratings,
    cognitive_load_ratings,
    autonomy_cost_ratings,
) -> HCIEvaluation:
    """Combine behavioral and subjective outcomes without hiding the trade-offs.

    All subjective ratings are expected on a 0-1 scale.
    """
    before = np.asarray(outcome_before, dtype=float)
    after = np.asarray(outcome_after, dtype=float)
    usefulness = np.asarray(usefulness_ratings, dtype=float)
    load = np.asarray(cognitive_load_ratings, dtype=float)
    autonomy = np.asarray(autonomy_cost_ratings, dtype=float)

    if before.shape != after.shape:
        raise ValueError("before and after must have the same shape")

    effectiveness = float(np.mean(after - before))
    perceived_usefulness = float(np.mean(usefulness))
    cognitive_load = float(np.mean(load))
    autonomy_cost = float(np.mean(autonomy))

    composite = (
        effectiveness
        + 0.35 * perceived_usefulness
        - 0.25 * cognitive_load
        - 0.25 * autonomy_cost
    )

    return HCIEvaluation(
        effectiveness=effectiveness,
        perceived_usefulness=perceived_usefulness,
        cognitive_load=cognitive_load,
        autonomy_cost=autonomy_cost,
        composite_score=float(composite),
    )
