import pytest

from behavior_modeling.adaptive_recourse import (
    RecoursePolicy,
    UserState,
    choose_policy,
    expected_response,
    simulate_adaptive_session,
    update_state,
)


def test_diverse_three_improves_willingness_without_large_load_jump() -> None:
    state = UserState(engagement=0.6, cognitive_load=0.3, acceptance=0.5)
    close = expected_response(state, RecoursePolicy(3, "close"))
    diverse = expected_response(state, RecoursePolicy(3, "diverse"))
    assert diverse.willingness_to_act > close.willingness_to_act
    assert diverse.cognitive_load - close.cognitive_load < 0.10


def test_large_diverse_set_has_more_load_than_large_close_set() -> None:
    state = UserState(engagement=0.6, cognitive_load=0.3, acceptance=0.5)
    close = expected_response(state, RecoursePolicy(7, "close"))
    diverse = expected_response(state, RecoursePolicy(7, "diverse"))
    assert diverse.cognitive_load > close.cognitive_load


def test_high_load_user_gets_no_more_complex_policy_than_low_load_user() -> None:
    low = UserState(engagement=0.7, cognitive_load=0.1, acceptance=0.5)
    high = UserState(engagement=0.7, cognitive_load=0.9, acceptance=0.5)
    low_policy, _ = choose_policy(low)
    high_policy, _ = choose_policy(high)
    assert high_policy.set_size <= low_policy.set_size


def test_state_update_is_bounded() -> None:
    state = UserState(1.5, -0.2, 2.0)
    response = expected_response(state, RecoursePolicy(3, "diverse"))
    updated = update_state(state, response)
    assert 0.0 <= updated.engagement <= 1.0
    assert 0.0 <= updated.cognitive_load <= 1.0
    assert 0.0 <= updated.acceptance <= 1.0


def test_simulation_length_and_validation() -> None:
    history = simulate_adaptive_session(UserState(), n_trials=5)
    assert len(history) == 5
    with pytest.raises(ValueError):
        simulate_adaptive_session(UserState(), n_trials=0)
