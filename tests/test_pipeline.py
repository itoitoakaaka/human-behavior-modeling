import unittest

import numpy as np

from data import generate_behavior
from features import make_one_step_dataset, make_sequences


class HumanBehaviorTests(unittest.TestCase):
    def test_next_trial_dataset_does_not_cross_subjects(self):
        raw = generate_behavior(n_subjects=4, n_trials=30, random_state=1)
        supervised = make_one_step_dataset(raw)

        expected = 4 * (30 - 1)
        self.assertEqual(len(supervised), expected)

    def test_sequence_shapes(self):
        raw = generate_behavior(n_subjects=3, n_trials=30, random_state=2)
        x, y, subjects = make_sequences(raw, window=5)

        self.assertEqual(x.ndim, 3)
        self.assertEqual(x.shape[1], 5)
        self.assertEqual(len(x), len(y))
        self.assertEqual(len(y), len(subjects))
        self.assertTrue(np.isfinite(x).all())


if __name__ == "__main__":
    unittest.main()
