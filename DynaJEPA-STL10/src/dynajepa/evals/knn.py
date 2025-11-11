"""k-NN evaluation utilities."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..models.jepa import build_model_from_config


def extract_features(
    checkpoint: Dict,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model = build_model_from_config(checkpoint.get("config", {}))
    model.load_state_dict(checkpoint["model"])
    encoder = model.get_encoder().to(device)
    encoder.eval()
    features = []
    labels = []
    with torch.no_grad():
        for images, label, _ in loader:
            images = images.to(device)
            feat = encoder(images)
            features.append(F.normalize(feat, dim=1).cpu())
            labels.append(label)
    return torch.cat(features), torch.cat(labels)


def knn_accuracy(
    checkpoint: Dict,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    k: int = 200,
    temperature: float = 0.07,
) -> float:
    train_features, train_labels = extract_features(checkpoint, train_loader, device)
    test_features, test_labels = extract_features(checkpoint, test_loader, device)

    similarities = test_features @ train_features.T / temperature
    topk = similarities.topk(k=k, dim=-1)
    topk_labels = train_labels[topk.indices]
    weights = torch.exp(topk.values)

    votes = torch.zeros(test_features.size(0), 10)
    for i in range(k):
        votes.scatter_add_(1, topk_labels[:, i : i + 1], weights[:, i : i + 1])
    predictions = votes.argmax(dim=1)
    accuracy = (predictions == test_labels).float().mean().item()
    return accuracy
