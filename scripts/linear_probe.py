"""Linear probe evaluation script."""

from __future__ import annotations

import argparse

import _path_setup  # noqa: F401

import torch

from dynajepa.data.stl10 import create_labeled_loader
from dynajepa.evals.linear_probe import LinearProbeConfig, train_linear_probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a linear probe on STL-10")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to pretraining checkpoint")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for probe training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of probe epochs")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate for probe")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for probe")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint.get("config", {})
    data_cfg = config.get("data", {})
    root = data_cfg.get("dataset_root", "./data")
    num_workers = data_cfg.get("num_workers", 8)
    pin_memory = data_cfg.get("pin_memory", True)

    train_loader = create_labeled_loader(
        root=root,
        split="train",
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    test_loader = create_labeled_loader(
        root=root,
        split="test",
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe_config = LinearProbeConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    train_acc, test_acc = train_linear_probe(checkpoint, train_loader, test_loader, device, probe_config)
    print(f"Linear probe accuracy - train: {train_acc:.4f}, test: {test_acc:.4f}")


if __name__ == "__main__":
    main()
