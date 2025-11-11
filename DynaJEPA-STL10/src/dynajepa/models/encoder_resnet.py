"""ResNet-based encoder backbone."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class ResNetEncoder(nn.Module):
    """ResNet-18 encoder returning a projection-ready feature vector."""

    def __init__(self, feature_dim: int = 512) -> None:
        super().__init__()
        backbone = models.resnet18(weights=None)
        modules = list(backbone.children())[:-1]
        self.backbone = nn.Sequential(*modules)
        self.out_dim = backbone.fc.in_features
        if feature_dim != self.out_dim:
            self.project = nn.Linear(self.out_dim, feature_dim)
        else:
            self.project = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        features = features.flatten(1)
        return self.project(features)

    @property
    def feature_dim(self) -> int:
        if isinstance(self.project, nn.Identity):
            return self.out_dim
        return self.project.out_features  # type: ignore[return-value]


def build_encoder(name: str, feature_dim: int) -> ResNetEncoder:
    if name.lower() != "resnet18":
        raise ValueError(f"Unsupported encoder '{name}'")
    return ResNetEncoder(feature_dim=feature_dim)
