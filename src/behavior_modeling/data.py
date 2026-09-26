from __future__ import annotations

import numpy as np
import pandas as pd


def generate_behavior(
    n_subjects=24,
    n_trials=100,
    random_state=42,
):
    """Generate synthetic participant-level sensorimotor adaptation data."""
    rng = np.random.default_rng(random_state)
    rows = []

    for subject in range(n_subjects):
        experienced = int(subject < n_subjects // 2)
        condition = "Water" if subject % 2 else "Land"

        retention = 0.90 + 0.03 * experienced
        learning = 0.18 + 0.05 * experienced
        if condition == "Water":
            retention -= 0.02
            learning += 0.04

        state = 0.0
        targets = np.concatenate(
            [
                np.zeros(10),
                np.ones(max(1, n_trials - 30)),
                np.zeros(min(20, n_trials - 10)),
            ]
        )[:n_trials]

        for trial, target in enumerate(targets):
            observed = state + rng.normal(0.0, 0.04)
            error = target - observed

            rows.append(
                {
                    "subject": subject,
                    "trial": trial,
                    "condition": condition,
                    "experienced": experienced,
                    "target": target,
                    "output": observed,
                    "error": error,
                }
            )

            state = (
                retention * state
                + learning * error
                + rng.normal(0.0, 0.015)
            )

    return pd.DataFrame(rows)
