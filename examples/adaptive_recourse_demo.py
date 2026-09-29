"""Run the synthetic adaptive-recourse extension."""

from behavior_modeling.adaptive_recourse import (
    CANDIDATE_POLICIES,
    UserState,
    choose_policy,
    expected_response,
    simulate_adaptive_session,
)


def main() -> None:
    states = {
        "low-load / engaged": UserState(engagement=0.80, cognitive_load=0.15, acceptance=0.55),
        "high-load / uncertain": UserState(engagement=0.35, cognitive_load=0.80, acceptance=0.35),
        "moderate": UserState(engagement=0.55, cognitive_load=0.45, acceptance=0.50),
    }

    print("Candidate policies by user state")
    print("=" * 72)
    for label, state in states.items():
        policy, response = choose_policy(state)
        print(
            f"{label:22s} -> {policy.set_size} options / {policy.diversity:7s} "
            f"| willingness={response.willingness_to_act:.3f} "
            f"| load={response.cognitive_load:.3f} "
            f"| utility={response.utility:.3f}"
        )

    print("\nPolicy table for the moderate state")
    print("=" * 72)
    state = states["moderate"]
    for policy in CANDIDATE_POLICIES:
        r = expected_response(state, policy)
        print(
            f"{policy.set_size} / {policy.diversity:7s} "
            f"willingness={r.willingness_to_act:.3f} "
            f"acceptance={r.decision_acceptance:.3f} "
            f"load={r.cognitive_load:.3f} utility={r.utility:.3f}"
        )

    print("\nClosed-loop session from a high-load state")
    print("=" * 72)
    for row in simulate_adaptive_session(states["high-load / uncertain"], n_trials=8):
        print(
            f"trial={row['trial']:2d} policy={row['set_size']}/{row['diversity']:7s} "
            f"state_load={row['cognitive_load_state']:.3f} "
            f"pred_load={row['predicted_load']:.3f} "
            f"utility={row['utility']:.3f}"
        )


if __name__ == "__main__":
    main()
