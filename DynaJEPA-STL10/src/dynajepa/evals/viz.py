"""Feature visualization utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from umap import UMAP

from ..models.jepa import build_model_from_config

STL10_CLASSES = (
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck",
)


def extract_embeddings(
    checkpoint: Dict,
    loader,
    device: torch.device,
    limit: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model = build_model_from_config(checkpoint.get("config", {}))
    model.load_state_dict(checkpoint["model"])
    encoder = model.get_encoder().to(device)
    encoder.eval()
    features = []
    labels = []
    total = 0
    with torch.no_grad():
        for images, label, _ in loader:
            images = images.to(device)
            feat = encoder(images).cpu()
            features.append(feat)
            labels.append(label)
            total += images.size(0)
            if limit is not None and total >= limit:
                break
    feats = torch.cat(features)[:limit]
    labs = torch.cat(labels)[:limit]
    return feats, labs


def plot_embeddings(
    embedding: torch.Tensor,
    labels: torch.Tensor,
    title: str,
    output_path: Path,
    class_names: Sequence[str] = STL10_CLASSES,
) -> None:
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap="tab10", s=5, alpha=0.8)
    handles = scatter.legend_elements()[0]
    plt.legend(handles, class_names, loc="best", fontsize="small", markerscale=2)
    plt.title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def tsne_visualization(
    checkpoint: Dict,
    loader,
    device: torch.device,
    output_path: Path,
    limit: Optional[int] = 2000,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> Path:
    features, labels = extract_embeddings(checkpoint, loader, device, limit)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    embedding = torch.from_numpy(tsne.fit_transform(features.numpy()))
    plot_embeddings(embedding, labels, "t-SNE", output_path)
    return output_path


def umap_visualization(
    checkpoint: Dict,
    loader,
    device: torch.device,
    output_path: Path,
    limit: Optional[int] = 2000,
    random_state: int = 42,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> Path:
    features, labels = extract_embeddings(checkpoint, loader, device, limit)
    reducer = UMAP(n_components=2, random_state=random_state, n_neighbors=n_neighbors, min_dist=min_dist)
    embedding = torch.from_numpy(reducer.fit_transform(features.numpy()))
    plot_embeddings(embedding, labels, "UMAP", output_path)
    return output_path
