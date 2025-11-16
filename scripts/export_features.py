"""Export features from a pretrained checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import _path_setup  # noqa: F401

import torch

from dynajepa.data.stl10 import create_labeled_loader
from dynajepa.evals.viz import extract_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export STL-10 features")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint.get("config", {})
    data_cfg = config.get("data", {})
    root = data_cfg.get("dataset_root", "./data")
    num_workers = data_cfg.get("num_workers", 8)
    pin_memory = data_cfg.get("pin_memory", True)

    loader = create_labeled_loader(
        root=root,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features, labels = extract_embeddings(checkpoint, loader, device, limit=args.limit)
    default_output = Path(args.checkpoint).resolve().parents[1] / "features" / f"{args.split}_features.pt"
    output_path = Path(args.output) if args.output else default_output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": features, "labels": labels}, output_path)
    print(f"Saved features to {output_path}")


if __name__ == "__main__":
    main()
