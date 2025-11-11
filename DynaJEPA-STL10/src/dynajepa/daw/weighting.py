"""Difficulty-aware weighting schemes."""

from __future__ import annotations

from typing import Literal

import torch

EPS = 1e-6


DifficultyNormMode = Literal["batch", "none"]


def _normalize(difficulty: torch.Tensor, mode: DifficultyNormMode) -> torch.Tensor:
    if mode == "none":
        return difficulty
    min_val = difficulty.min()
    max_val = difficulty.max()
    if torch.isclose(max_val, min_val):
        return torch.zeros_like(difficulty)
    return (difficulty - min_val) / (max_val - min_val + EPS)


def normalize_weights(weights: torch.Tensor, clip_min: float, clip_max: float) -> torch.Tensor:
    clipped = torch.clamp(weights, min=clip_min, max=clip_max)
    mean = clipped.mean().clamp_min(EPS)
    return clipped / mean


def softmax_weighting(
    difficulty: torch.Tensor,
    tau: float,
    clip_min: float,
    clip_max: float,
    norm: DifficultyNormMode = "batch",
) -> torch.Tensor:
    normalized = _normalize(difficulty, norm)
    scale = tau if tau > 0 else EPS
    weights = torch.softmax(normalized / scale, dim=0) * difficulty.numel()
    return normalize_weights(weights, clip_min, clip_max)


def power_weighting(
    difficulty: torch.Tensor,
    alpha: float,
    clip_min: float,
    clip_max: float,
    norm: DifficultyNormMode = "batch",
    reverse: bool = False,
) -> torch.Tensor:
    normalized = _normalize(difficulty, norm)
    if reverse:
        normalized = 1 - normalized
    weights = (normalized + EPS) ** alpha
    return normalize_weights(weights, clip_min, clip_max)


def inverse_weighting(
    difficulty: torch.Tensor,
    clip_min: float,
    clip_max: float,
    norm: DifficultyNormMode = "batch",
) -> torch.Tensor:
    normalized = _normalize(difficulty, norm)
    weights = 1.0 / (normalized + EPS)
    return normalize_weights(weights, clip_min, clip_max)
