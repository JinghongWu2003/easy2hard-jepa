"""Predictor network for JEPA."""

from __future__ import annotations

from torch import nn

from .projector import mlp


class PredictorMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int) -> None:
        super().__init__()
        self.net = mlp(in_dim, hidden_dim, out_dim, num_layers)

    def forward(self, x):  # type: ignore[override]
        return self.net(x)
