"""EMA utilities for momentum encoders."""

from __future__ import annotations

from torch import nn


def ema_update(model: nn.Module, ema_model: nn.Module, momentum: float) -> None:
    for param, ema_param in zip(model.parameters(), ema_model.parameters(), strict=True):
        ema_param.data = ema_param.data * momentum + param.data * (1 - momentum)
