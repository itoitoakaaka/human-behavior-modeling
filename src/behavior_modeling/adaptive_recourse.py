"""Synthetic adaptive algorithmic-recourse prototype.

Inspired by Tominaga, Yamashita, and Kurashima (IJCAI-ECAI 2026), who
experimentally studied psychological benefits and costs of recourse-set size
and diversity. This module does NOT reproduce their participant data.

The prospective extension asks:
Can a recourse interface adapt the number/diversity of options to a user's
latent engagement, cognitive-load, and acceptance state rather than presenting
a fixed recourse set?
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class UserState:
    """Latent user state used by the synthetic controller."""

    engagement: float = 0.50
    cognitive_load: float = 0.35
    acceptance: float = 0.50

    def clipped(self) -> "UserState":
        return UserState(
            engagement=float(np.clip(self.engagement, 0.0, 1.0)),
            cognitive_load=float(np.clip(self.cognitive_load, 0.0, 1.0)),
            acceptance=float(np.clip(self.acceptance, 0.0, 1.0)),
        )


@dataclass(frozen=True)
class RecoursePolicy:
    """Interface choice: how many options and how diverse they are."""

    set_size: int
    diversity: str

    def __post_init__(self) -> None:
        if self.set_size not in {1, 3, 7}:
            raise ValueError("set_size must be one of {1, 3, 7}")
        if self.diversity not in {"close", "diverse"}:
            raise ValueError("diversity must be 'close' or 'diverse'")
        if self.set_size == 1 and self.diversity == "diverse":
            raise ValueError("diversity is not meaningful for a single option")


@dataclass(frozen=True)
class ExpectedResponse:
    willingness_to_act: float
    decision_acceptance: float
    cognitive_load: float
    utility: float


CANDIDATE_POLICIES = (
    RecoursePolicy(1, "close"),
    RecoursePolicy(3, "close"),
    RecoursePolicy(3, "diverse"),
    RecoursePolicy(7, "close"),
    RecoursePolicy(7, "diverse"),
)


def expected_response(
    state: UserState,
    policy: RecoursePolicy,
    *,
    load_weight: float = 0.50,
) -> ExpectedResponse:
    """Compute a transparent synthetic response model.

    Coefficients are illustrative rather than fitted to the IJCAI-ECAI
    participant dataset. The qualitative structure reflects the published
    finding that diversification can be beneficial for smaller option sets,
    while large diverse sets can make cognitive load more salient.

    The prospective extension adds state dependence: users already carrying
    high cognitive load are penalized more strongly for complex recourse sets.
    """

    s = state.clipped()

    size_gain = {1: 0.00, 3: 0.12, 7: 0.18}[policy.set_size]
    base_load = {1: 0.05, 3: 0.16, 7: 0.34}[policy.set_size]

    diversity_gain = 0.0
    diversity_load = 0.0
    if policy.diversity == "diverse":
        diversity_gain = 0.12 if policy.set_size == 3 else 0.04
        diversity_load = 0.03 if policy.set_size == 3 else 0.20

    complexity = (policy.set_size - 1) / 6.0
    engagement_bonus = 0.08 * s.engagement * complexity
    overload_penalty = 0.28 * s.cognitive_load * complexity

    willingness = (
        0.34
        + 0.28 * s.engagement
        + size_gain
        + diversity_gain
        + engagement_bonus
        - overload_penalty
    )
    acceptance = (
        0.36
        + 0.30 * s.acceptance
        + 0.55 * size_gain
        + 0.40 * diversity_gain
        - 0.18 * s.cognitive_load * complexity
    )
    load = (
        0.12
        + 0.52 * s.cognitive_load
        + base_load
        + diversity_load
        - 0.07 * s.engagement
    )

    willingness = float(np.clip(willingness, 0.0, 1.0))
    acceptance = float(np.clip(acceptance, 0.0, 1.0))
    load = float(np.clip(load, 0.0, 1.0))

    # High-load users place more weight on reducing additional burden.
    effective_load_weight = load_weight + 0.90 * s.cognitive_load
    utility = (
        0.60 * willingness
        + 0.40 * acceptance
        - effective_load_weight * load
    )
    return ExpectedResponse(willingness, acceptance, load, float(utility))


def choose_policy(
    state: UserState,
    *,
    load_weight: float = 0.50,
) -> tuple[RecoursePolicy, ExpectedResponse]:
    """Choose the candidate policy with the largest expected utility."""

    scored = [
        (policy, expected_response(state, policy, load_weight=load_weight))
        for policy in CANDIDATE_POLICIES
    ]
    return max(scored, key=lambda item: item[1].utility)


def update_state(
    state: UserState,
    response: ExpectedResponse,
    *,
    retention: float = 0.80,
) -> UserState:
    """Synthetic trial-to-trial state update after observing an intervention."""

    s = state.clipped()
    one_minus = 1.0 - retention
    return UserState(
        engagement=retention * s.engagement + one_minus * response.willingness_to_act,
        cognitive_load=retention * s.cognitive_load + one_minus * response.cognitive_load,
        acceptance=retention * s.acceptance + one_minus * response.decision_acceptance,
    ).clipped()


def simulate_adaptive_session(
    initial_state: UserState,
    *,
    n_trials: int = 12,
    load_weight: float = 0.50,
) -> list[dict[str, float | int | str]]:
    """Run a closed-loop synthetic adaptive-recourse session."""

    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")

    state = initial_state.clipped()
    history: list[dict[str, float | int | str]] = []
    for trial in range(1, n_trials + 1):
        policy, response = choose_policy(state, load_weight=load_weight)
        history.append(
            {
                "trial": trial,
                "set_size": policy.set_size,
                "diversity": policy.diversity,
                "engagement": state.engagement,
                "cognitive_load_state": state.cognitive_load,
                "acceptance_state": state.acceptance,
                "willingness_to_act": response.willingness_to_act,
                "decision_acceptance": response.decision_acceptance,
                "predicted_load": response.cognitive_load,
                "utility": response.utility,
            }
        )
        state = update_state(state, response)
    return history
