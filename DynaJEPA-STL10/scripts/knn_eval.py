"""Run k-NN evaluation on STL-10."""

from __future__ import annotations

import argparse

import torch

from dynajepa.data.stl10 import create_labeled_loader
from dynajepa.evals.knn import knn_accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="k-NN evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for feature extraction")
    parser.add_argument("--k", type=int, default=200, help="Number of neighbors")
    parser.add_argument("--temperature", type=float, default=0.07, help="Softmax temperature")
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
        shuffle=False,
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
    acc = knn_accuracy(checkpoint, train_loader, test_loader, device, k=args.k, temperature=args.temperature)
    print(f"k-NN accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
