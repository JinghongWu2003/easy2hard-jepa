"""Linear probe evaluation for STL-10."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..models.jepa import build_model_from_config


@dataclass
class LinearProbeConfig:
    epochs: int = 50
    lr: float = 0.1
    weight_decay: float = 0.0
    momentum: float = 0.9


class LinearClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 10) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.fc(x)


def train_linear_probe(
    checkpoint: Dict,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    config: LinearProbeConfig,
) -> Tuple[float, float]:
    model = build_model_from_config(checkpoint.get("config", {}))
    model.load_state_dict(checkpoint["model"])
    encoder = model.get_encoder().to(device)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    classifier = LinearClassifier(in_dim=encoder.feature_dim, num_classes=10).to(device)  # type: ignore[attr-defined]
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        classifier.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )

    for epoch in range(config.epochs):
        classifier.train()
        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                features = encoder(images)
            logits = classifier(features)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    classifier.eval()
    train_acc = _evaluate(classifier, encoder, train_loader, device)
    test_acc = _evaluate(classifier, encoder, test_loader, device)
    return train_acc, test_acc


@torch.no_grad()
def _evaluate(
    classifier: LinearClassifier,
    encoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    correct = 0
    total = 0
    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)
        features = encoder(images)
        logits = classifier(features)
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)
