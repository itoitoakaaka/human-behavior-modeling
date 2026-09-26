from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS


def persistence_predict(frame):
    """Naive baseline: predict next output as current output."""
    return frame["output_t"].to_numpy(dtype=float)


def fit_linear_baseline(train_frame):
    """Leakage-safe standardized ridge-regression baseline."""
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("regression", Ridge(alpha=1.0)),
        ]
    )
    model.fit(
        train_frame[FEATURE_COLUMNS],
        train_frame["output_next"],
    )
    return model


def predict_linear(model, frame):
    return np.asarray(model.predict(frame[FEATURE_COLUMNS]), dtype=float)
