import numpy as np

from behavior_modeling.data import generate_behavior
from behavior_modeling.features import make_one_step_dataset, make_sequences


def test_next_trial_dataset_does_not_cross_subjects():
    raw = generate_behavior(n_subjects=4, n_trials=30, random_state=1)
    supervised = make_one_step_dataset(raw)
    assert len(supervised) == 4 * (30 - 1)


def test_sequence_shapes():
    raw = generate_behavior(n_subjects=3, n_trials=30, random_state=2)
    x, y, subjects = make_sequences(raw, window=5)

    assert x.ndim == 3
    assert x.shape[1] == 5
    assert len(x) == len(y) == len(subjects)
    assert np.isfinite(x).all()
