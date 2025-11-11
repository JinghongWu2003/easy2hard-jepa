"""Checkpoint helpers for saving and loading training state."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


DEFAULT_CHECKPOINT_NAME = "last.pt"


def save_checkpoint(state: Dict[str, Any], output_dir: Path, filename: str | None = None) -> Path:
    """Save a training state dictionary to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / (filename or DEFAULT_CHECKPOINT_NAME)
    torch.save(state, path)
    return path


def load_checkpoint(path: Path | str, map_location: str | torch.device | None = None) -> Dict[str, Any]:
    """Load a checkpoint file."""
    return torch.load(path, map_location=map_location)
