import math

from behavior_modeling.decision_models import (
    IntertemporalOption,
    exponential_discount,
    quasi_hyperbolic_discount,
    probability_choose_a,
)
from behavior_modeling.intervention import InterventionPolicy, simulate_completion


def test_discount_at_zero_is_one():
    assert quasi_hyperbolic_discount(0, beta=0.5, delta=0.9) == 1.0


def test_exponential_discount_decreases_with_delay():
    assert exponential_discount(5, 0.95) < exponential_discount(1, 0.95)


def test_lower_beta_can_increase_preference_for_immediate_reward():
    sooner = IntertemporalOption(1.0, 0)
    later = IntertemporalOption(1.3, 7)
    p_low_beta = probability_choose_a(sooner, later, beta=0.5, delta=0.98)
    p_high_beta = probability_choose_a(sooner, later, beta=1.0, delta=0.98)
    assert p_low_beta > p_high_beta


def test_intervention_simulation_is_reproducible():
    policy = InterventionPolicy("prompt", 0.1, 0.02, 0.05)
    a = simulate_completion(0.7, 0.98, policy, n_people=500, seed=10)
    b = simulate_completion(0.7, 0.98, policy, n_people=500, seed=10)
    assert math.isclose(a.completion_rate, b.completion_rate)
