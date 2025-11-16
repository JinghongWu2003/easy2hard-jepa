"""Projection head implementations."""

from __future__ import annotations

from torch import nn


def mlp(in_dim: int, hidden_dim: int, out_dim: int, num_layers: int) -> nn.Sequential:
    layers = []
    current_dim = in_dim
    for layer in range(num_layers - 1):
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, out_dim))
    return nn.Sequential(*layers)


class ProjectorMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int) -> None:
        super().__init__()
        self.net = mlp(in_dim, hidden_dim, out_dim, num_layers)

    def forward(self, x):  # type: ignore[override]
        return self.net(x)
