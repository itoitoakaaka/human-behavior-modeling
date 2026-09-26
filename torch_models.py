from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class MLPRegressorTorch(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


class GRURegressor(nn.Module):
    def __init__(self, n_features, hidden_size=24):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.gru(x)
        return self.head(output[:, -1]).squeeze(-1)


def standardize_from_train(x_train, x_test):
    """Standardize feature dimensions using training data only."""
    axes = tuple(range(x_train.ndim - 1))
    mean = x_train.mean(axis=axes, keepdims=True)
    std = x_train.std(axis=axes, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (x_train - mean) / std, (x_test - mean) / std


def fit_torch_regressor(
    model,
    x_train,
    y_train,
    x_val,
    y_val,
    epochs=150,
    batch_size=64,
    learning_rate=1e-3,
    patience=20,
    random_state=42,
):
    """Train with early stopping on a held-out validation set."""
    torch.manual_seed(random_state)

    x_train_t = torch.as_tensor(x_train, dtype=torch.float32)
    y_train_t = torch.as_tensor(y_train, dtype=torch.float32)
    x_val_t = torch.as_tensor(x_val, dtype=torch.float32)
    y_val_t = torch.as_tensor(y_val, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(x_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    epochs_without_improvement = 0

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x_val_t), y_val_t).item())

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    model.load_state_dict(best_state)
    return model


def predict_torch(model, x):
    model.eval()
    with torch.no_grad():
        tensor = torch.as_tensor(x, dtype=torch.float32)
        return model(tensor).cpu().numpy()
