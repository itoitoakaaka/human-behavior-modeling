from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "target_t",
    "output_t",
    "error_t",
    "condition_water",
    "experienced",
]


def make_one_step_dataset(df):
    """Build next-trial prediction rows without crossing subject boundaries."""
    rows = []

    for subject, group in df.groupby("subject", sort=True):
        group = group.sort_values("trial").reset_index(drop=True)
        for idx in range(len(group) - 1):
            current = group.iloc[idx]
            nxt = group.iloc[idx + 1]

            rows.append(
                {
                    "subject": subject,
                    "trial": int(current["trial"]),
                    "target_t": float(current["target"]),
                    "output_t": float(current["output"]),
                    "error_t": float(current["error"]),
                    "condition_water": float(current["condition"] == "Water"),
                    "experienced": float(current["experienced"]),
                    "output_next": float(nxt["output"]),
                }
            )

    return pd.DataFrame(rows)


def make_sequences(df, window=8):
    """Build fixed-length trial-history sequences for recurrent models."""
    supervised = make_one_step_dataset(df)
    x, y, subjects = [], [], []

    for subject, group in supervised.groupby("subject", sort=True):
        values = group[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        targets = group["output_next"].to_numpy(dtype=np.float32)

        for end in range(window - 1, len(group)):
            start = end - window + 1
            x.append(values[start : end + 1])
            y.append(targets[end])
            subjects.append(subject)

    return (
        np.asarray(x, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        np.asarray(subjects, dtype=int),
    )
