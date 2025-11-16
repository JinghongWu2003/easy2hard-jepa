"""Difficulty estimation utilities."""

from __future__ import annotations

import torch


class DifficultyBuffer:
    """Maintain an exponential moving average of per-sample difficulties."""

    def __init__(self, size: int, alpha: float = 0.0, device: torch.device | None = None) -> None:
        self.alpha = alpha
        self.buffer = torch.zeros(size, dtype=torch.float32, device=device)
        self.initialized = torch.zeros(size, dtype=torch.bool, device=device)

    def update(self, indices: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        if indices.dtype != torch.long:
            indices = indices.long()
        if values.dim() != 1:
            values = values.view(-1)

        current = self.buffer[indices]
        mask = self.initialized[indices]
        if self.alpha > 0.0:
            blended = torch.where(
                mask,
                self.alpha * current + (1 - self.alpha) * values,
                values,
            )
        else:
            blended = values
        self.buffer[indices] = blended
        self.initialized[indices] = True
        return blended

    def get(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.dtype != torch.long:
            indices = indices.long()
        return self.buffer[indices]
