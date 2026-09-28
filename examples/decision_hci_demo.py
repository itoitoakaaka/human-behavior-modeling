"""Decision modeling + intervention + HCI demo.

Run:
    python examples/decision_hci_demo.py
"""

from behavior_modeling.decision_models import (
    IntertemporalOption,
    probability_choose_a,
)
from behavior_modeling.intervention import (
    InterventionPolicy,
    compare_policies,
)


def main():
    sooner = IntertemporalOption(reward=1.0, delay=0)
    later = IntertemporalOption(reward=1.35, delay=7)

    for beta in [1.0, 0.8, 0.6]:
        p_sooner = probability_choose_a(
            sooner,
            later,
            beta=beta,
            delta=0.98,
            inverse_temperature=8.0,
        )
        print(f"beta={beta:.1f}: P(choose sooner)={p_sooner:.3f}")

    policies = [
        InterventionPolicy(
            name="no_intervention",
            prompt_strength=0.00,
            burden_cost=0.00,
            present_bias_reduction=0.00,
        ),
        InterventionPolicy(
            name="light_prompt",
            prompt_strength=0.06,
            burden_cost=0.015,
            present_bias_reduction=0.03,
        ),
        InterventionPolicy(
            name="strong_prompt",
            prompt_strength=0.15,
            burden_cost=0.070,
            present_bias_reduction=0.08,
        ),
    ]

    print("\nSynthetic intervention comparison")
    for result in compare_policies(beta=0.65, delta=0.98, policies=policies):
        print(
            f"{result.policy:16s} "
            f"completion={result.completion_rate:.3f} "
            f"burden={result.mean_burden:.3f} "
            f"utility={result.utility:.3f}"
        )


if __name__ == "__main__":
    main()
