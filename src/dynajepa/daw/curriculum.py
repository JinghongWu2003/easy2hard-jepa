"""Curriculum scheduling for difficulty-aware weighting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch

from .weighting import DifficultyNormMode, normalize_weights, power_weighting, softmax_weighting


@dataclass
class WeightScheduler:
    base_fn: Callable[[torch.Tensor], torch.Tensor]
    scheduler_fn: Optional[Callable[[int, torch.Tensor], torch.Tensor]] = None

    def __call__(self, difficulty: torch.Tensor, epoch: Optional[int] = None) -> torch.Tensor:
        if self.scheduler_fn is None or epoch is None:
            return self.base_fn(difficulty)
        return self.scheduler_fn(epoch, difficulty)


def build_base_weighting(daw_cfg: Dict) -> Callable[[torch.Tensor], torch.Tensor]:
    weighting_type = daw_cfg.get("weighting", "softmax")
    norm_mode: DifficultyNormMode = daw_cfg.get("difficulty_norm", "batch")
    clip_min = daw_cfg.get("clip_min", 0.1)
    clip_max = daw_cfg.get("clip_max", 10.0)

    if weighting_type == "softmax":
        tau = daw_cfg.get("softmax_tau", 0.25)

        def weight_fn(difficulty: torch.Tensor) -> torch.Tensor:
            return softmax_weighting(difficulty, tau, clip_min, clip_max, norm_mode)

    elif weighting_type == "power":
        alpha = daw_cfg.get("power_alpha", 1.0)

        def weight_fn(difficulty: torch.Tensor) -> torch.Tensor:
            return power_weighting(difficulty, alpha, clip_min, clip_max, norm_mode)

    else:
        raise ValueError(f"Unknown weighting type '{weighting_type}'")

    return weight_fn


def build_curriculum(daw_cfg: Dict, curriculum_cfg: Dict) -> WeightScheduler:
    base_weight_fn = build_base_weighting(daw_cfg)
    if not curriculum_cfg.get("enabled", False):
        return WeightScheduler(base_fn=base_weight_fn)

    easy_epochs = curriculum_cfg.get("easy_epochs", 0)
    uniform_epochs = curriculum_cfg.get("uniform_epochs", 0)
    hard_epochs = curriculum_cfg.get("hard_epochs", 0)
    easy_alpha = abs(curriculum_cfg.get("easy_alpha", 1.0))
    hard_alpha = curriculum_cfg.get("hard_alpha", 1.0)
    uniform_weight = curriculum_cfg.get("uniform_weight", 1.0)

    total_epochs = easy_epochs + uniform_epochs + hard_epochs
    if total_epochs <= 0:
        return WeightScheduler(base_fn=base_weight_fn)

    norm_mode: DifficultyNormMode = daw_cfg.get("difficulty_norm", "batch")
    clip_min = daw_cfg.get("clip_min", 0.1)
    clip_max = daw_cfg.get("clip_max", 10.0)

    def scheduler(epoch: int, difficulty: torch.Tensor) -> torch.Tensor:
        if epoch < easy_epochs:
            return power_weighting(difficulty, easy_alpha, clip_min, clip_max, norm_mode, reverse=True)
        if epoch < easy_epochs + uniform_epochs:
            weights = torch.full_like(difficulty, uniform_weight)
            return normalize_weights(weights, clip_min, clip_max)
        if epoch < total_epochs:
            return power_weighting(difficulty, hard_alpha, clip_min, clip_max, norm_mode)
        return base_weight_fn(difficulty)

    return WeightScheduler(base_fn=base_weight_fn, scheduler_fn=scheduler)
