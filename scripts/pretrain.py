"""JEPA pretraining entrypoint."""

from __future__ import annotations

import argparse

import _path_setup  # noqa: F401

from dynajepa.training.trainer import PretrainTrainer
from dynajepa.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain JEPA on STL-10")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--override", nargs="*", default=None, help="Optional key=value overrides")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--run-name", type=str, default=None, help="Override run name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.override)
    trainer = PretrainTrainer(config, resume_path=args.resume, run_name_override=args.run_name)
    trainer.train()


if __name__ == "__main__":
    main()
