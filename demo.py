from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from baselines import fit_linear_baseline, persistence_predict, predict_linear
from data import generate_behavior
from evaluate import regression_metrics
from features import FEATURE_COLUMNS, make_one_step_dataset, make_sequences
from torch_models import (
    GRURegressor,
    MLPRegressorTorch,
    fit_torch_regressor,
    predict_torch,
    standardize_from_train,
)


def split_subjects(subject_ids, random_state=42):
    unique_subjects = np.unique(subject_ids)
    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(unique_subjects)

    n_test = max(1, int(round(len(shuffled) * 0.2)))
    n_val = max(1, int(round(len(shuffled) * 0.2)))

    test_subjects = shuffled[:n_test]
    val_subjects = shuffled[n_test : n_test + n_val]
    train_subjects = shuffled[n_test + n_val :]

    return train_subjects, val_subjects, test_subjects


def print_metrics(name, metrics):
    print(
        f"{name:12s} "
        f"MAE={metrics['mae']:.4f} "
        f"RMSE={metrics['rmse']:.4f} "
        f"R2={metrics['r2']:.4f}"
    )


def main():
    raw = generate_behavior()
    frame = make_one_step_dataset(raw)

    train_subjects, val_subjects, test_subjects = split_subjects(
        frame["subject"].to_numpy()
    )

    train_frame = frame[frame["subject"].isin(train_subjects)].copy()
    val_frame = frame[frame["subject"].isin(val_subjects)].copy()
    test_frame = frame[frame["subject"].isin(test_subjects)].copy()

    print("Subject split")
    print(f"  train: {sorted(train_subjects.tolist())}")
    print(f"  val  : {sorted(val_subjects.tolist())}")
    print(f"  test : {sorted(test_subjects.tolist())}")

    persistence = persistence_predict(test_frame)
    print_metrics(
        "Persistence",
        regression_metrics(test_frame["output_next"], persistence),
    )

    linear = fit_linear_baseline(train_frame)
    linear_pred = predict_linear(linear, test_frame)
    print_metrics(
        "Linear",
        regression_metrics(test_frame["output_next"], linear_pred),
    )

    x_train = train_frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_train = train_frame["output_next"].to_numpy(dtype=np.float32)
    x_val = val_frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_val = val_frame["output_next"].to_numpy(dtype=np.float32)
    x_test = test_frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)

    x_train, x_val = standardize_from_train(x_train, x_val)
    x_train_again, x_test = standardize_from_train(
        train_frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32),
        x_test,
    )
    x_train = x_train_again

    mlp = MLPRegressorTorch(len(FEATURE_COLUMNS))
    fit_torch_regressor(mlp, x_train, y_train, x_val, y_val)
    mlp_pred = predict_torch(mlp, x_test)
    print_metrics(
        "MLP",
        regression_metrics(test_frame["output_next"], mlp_pred),
    )

    x_seq, y_seq, seq_subjects = make_sequences(raw, window=8)
    train_mask = np.isin(seq_subjects, train_subjects)
    val_mask = np.isin(seq_subjects, val_subjects)
    test_mask = np.isin(seq_subjects, test_subjects)

    x_seq_train, x_seq_val = standardize_from_train(
        x_seq[train_mask],
        x_seq[val_mask],
    )
    x_seq_train_again, x_seq_test = standardize_from_train(
        x_seq[train_mask],
        x_seq[test_mask],
    )
    x_seq_train = x_seq_train_again

    gru = GRURegressor(x_seq.shape[-1])
    fit_torch_regressor(
        gru,
        x_seq_train,
        y_seq[train_mask],
        x_seq_val,
        y_seq[val_mask],
    )
    gru_pred = predict_torch(gru, x_seq_test)

    print_metrics(
        "GRU",
        regression_metrics(y_seq[test_mask], gru_pred),
    )


if __name__ == "__main__":
    main()
