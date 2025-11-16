"""Generate t-SNE plots from pretrained checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import _path_setup  # noqa: F401

import torch

from dynajepa.data.stl10 import create_labeled_loader
from dynajepa.evals.viz import tsne_visualization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="t-SNE visualization")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint file")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for feature extraction")
    parser.add_argument("--limit", type=int, default=2000, help="Number of samples to visualize")
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity")
    parser.add_argument("--output", type=str, default=None, help="Optional output path")
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
        split="test",
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    default_output = Path(args.checkpoint).resolve().parents[1] / "figs" / "tsne.png"
    output_path = Path(args.output) if args.output else default_output
    tsne_visualization(checkpoint, loader, device, output_path, limit=args.limit, perplexity=args.perplexity)
    print(f"Saved t-SNE visualization to {output_path}")


if __name__ == "__main__":
    main()
