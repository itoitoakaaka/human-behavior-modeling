"""Simple decision-making models for transparent simulation and teaching.

These models are deliberately compact. They are intended to make assumptions explicit
before moving to richer behavioral models.
"""

from dataclasses import dataclass
import math


def exponential_discount(delay: float, delta: float) -> float:
    """Exponential discounting: V = delta^delay."""
    if delay < 0:
        raise ValueError("delay must be non-negative")
    if not 0 < delta <= 1:
        raise ValueError("delta must be in (0, 1]")
    return delta ** delay


def quasi_hyperbolic_discount(delay: float, beta: float, delta: float) -> float:
    """Beta-delta / quasi-hyperbolic discounting.

    At delay 0 the weight is 1.
    At delay >0 the weight is beta * delta^delay.
    """
    if delay < 0:
        raise ValueError("delay must be non-negative")
    if not 0 < beta <= 1:
        raise ValueError("beta must be in (0, 1]")
    if not 0 < delta <= 1:
        raise ValueError("delta must be in (0, 1]")
    return 1.0 if delay == 0 else beta * (delta ** delay)


def discounted_value(reward: float, delay: float, beta: float = 1.0, delta: float = 1.0) -> float:
    return reward * quasi_hyperbolic_discount(delay, beta, delta)


def softmax_choice(value_a: float, value_b: float, inverse_temperature: float = 5.0) -> float:
    """Probability of choosing option A over B."""
    if inverse_temperature < 0:
        raise ValueError("inverse_temperature must be non-negative")
    x = inverse_temperature * (value_a - value_b)
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


@dataclass(frozen=True)
class IntertemporalOption:
    reward: float
    delay: float


def probability_choose_a(
    option_a: IntertemporalOption,
    option_b: IntertemporalOption,
    beta: float,
    delta: float,
    inverse_temperature: float = 5.0,
) -> float:
    va = discounted_value(option_a.reward, option_a.delay, beta, delta)
    vb = discounted_value(option_b.reward, option_b.delay, beta, delta)
    return softmax_choice(va, vb, inverse_temperature)
