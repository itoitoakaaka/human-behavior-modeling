"""Small human-centered intervention simulation.

The goal is not to claim an optimal policy for real people. It is to demonstrate how
behavioral parameters, intervention timing, and user burden can be represented explicitly.
"""

from dataclasses import dataclass
from typing import Iterable
import numpy as np


@dataclass(frozen=True)
class InterventionPolicy:
    name: str
    prompt_strength: float
    burden_cost: float
    present_bias_reduction: float


@dataclass(frozen=True)
class SimulationResult:
    policy: str
    completion_rate: float
    mean_burden: float
    utility: float


def simulate_completion(
    beta: float,
    delta: float,
    policy: InterventionPolicy,
    n_people: int = 2000,
    seed: int = 42,
) -> SimulationResult:
    """Synthetic completion simulation.

    A person completes a delayed task when discounted future benefit plus intervention
    support exceeds immediate effort cost. Heterogeneity is represented by random benefit
    and effort draws.
    """
    rng = np.random.default_rng(seed)
    beta_eff = min(1.0, beta + policy.present_bias_reduction)

    future_benefit = rng.normal(1.0, 0.20, n_people)
    effort_cost = rng.normal(0.72, 0.18, n_people)
    noise = rng.normal(0.0, 0.10, n_people)

    subjective_future = beta_eff * delta * future_benefit
    support = policy.prompt_strength
    completed = subjective_future + support + noise > effort_cost

    completion_rate = float(np.mean(completed))
    mean_burden = float(policy.burden_cost)
    utility = completion_rate - mean_burden

    return SimulationResult(
        policy=policy.name,
        completion_rate=completion_rate,
        mean_burden=mean_burden,
        utility=utility,
    )


def compare_policies(
    beta: float,
    delta: float,
    policies: Iterable[InterventionPolicy],
    n_people: int = 2000,
    seed: int = 42,
):
    return [
        simulate_completion(beta, delta, p, n_people=n_people, seed=seed)
        for p in policies
    ]
