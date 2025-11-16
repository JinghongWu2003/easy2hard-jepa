"""Run a quick smoke test to verify the training loop."""

from __future__ import annotations

import argparse
from pathlib import Path

from dynajepa.training.trainer import PretrainTrainer
from dynajepa.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test for DynaJEPA")
    parser.add_argument("--config", type=str, default="configs/stl10_small_smoketest.yaml")
    parser.add_argument("--override", nargs="*", default=None)
    parser.add_argument("--run-name", type=str, default="smoke_test_run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.override)
    trainer = PretrainTrainer(config, run_name_override=args.run_name)
    trainer.train()

    output_dir = Path("outputs") / args.run_name
    checkpoint_dir = output_dir / "checkpoints"
    feature_dir = output_dir / "features"
    if not any(checkpoint_dir.glob("*.pt")):
        raise RuntimeError("Smoke test failed: no checkpoints saved")
    if not any(feature_dir.glob("*.pt")):
        raise RuntimeError("Smoke test failed: no features exported")
    print(f"Smoke test completed successfully. Outputs stored in {output_dir}")


if __name__ == "__main__":
    main()
