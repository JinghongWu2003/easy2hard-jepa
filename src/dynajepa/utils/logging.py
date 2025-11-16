"""Logging helpers for training."""

from __future__ import annotations

import logging
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def setup_logging(output_dir: Path, run_name: str) -> tuple[logging.Logger, SummaryWriter]:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(run_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(log_dir / "training.log")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
    return logger, writer
