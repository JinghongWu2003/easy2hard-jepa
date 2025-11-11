"""Joint-Embedding Predictive Architecture implementation."""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn
from torch.nn import functional as F

from .encoder_resnet import build_encoder
from .predictor import PredictorMLP
from .projector import ProjectorMLP


class JEPAModel(nn.Module):
    """Simplified JEPA with online and EMA target encoders."""

    def __init__(
        self,
        encoder_name: str,
        feature_dim: int,
        projector_cfg: Dict[str, int],
        predictor_cfg: Dict[str, int],
        ema_momentum: float = 0.996,
        loss_type: str = "cosine",
    ) -> None:
        super().__init__()
        self.online_encoder = build_encoder(encoder_name, feature_dim)
        self.target_encoder = build_encoder(encoder_name, feature_dim)
        self.projector = ProjectorMLP(
            in_dim=self.online_encoder.feature_dim,
            hidden_dim=projector_cfg["hidden_dim"],
            out_dim=projector_cfg["output_dim"],
            num_layers=projector_cfg["num_layers"],
        )
        self.target_projector = ProjectorMLP(
            in_dim=self.target_encoder.feature_dim,
            hidden_dim=projector_cfg["hidden_dim"],
            out_dim=projector_cfg["output_dim"],
            num_layers=projector_cfg["num_layers"],
        )
        self.predictor = PredictorMLP(
            in_dim=projector_cfg["output_dim"],
            hidden_dim=predictor_cfg["hidden_dim"],
            out_dim=predictor_cfg["output_dim"],
            num_layers=predictor_cfg["num_layers"],
        )
        self.ema_momentum = ema_momentum
        self.loss_type = loss_type
        self._init_target_weights()

    def _init_target_weights(self) -> None:
        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters(), strict=True):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False
        for param_o, param_t in zip(self.projector.parameters(), self.target_projector.parameters(), strict=True):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False

    @torch.no_grad()
    def update_target(self, momentum: float | None = None) -> None:
        m = momentum if momentum is not None else self.ema_momentum
        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters(), strict=True):
            param_t.data = param_t.data * m + param_o.data * (1 - m)
        for param_o, param_t in zip(self.projector.parameters(), self.target_projector.parameters(), strict=True):
            param_t.data = param_t.data * m + param_o.data * (1 - m)

    def forward(self, context: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        context_feat = self.online_encoder(context)
        context_proj = self.projector(context_feat)
        context_pred = self.predictor(context_proj)

        with torch.no_grad():
            target_feat = self.target_encoder(target)
            target_proj = self.target_projector(target_feat)

        if self.loss_type == "cosine":
            context_norm = F.normalize(context_pred, dim=-1)
            target_norm = F.normalize(target_proj, dim=-1)
            loss_vector = 1 - F.cosine_similarity(context_norm, target_norm, dim=-1)
        elif self.loss_type == "l2":
            loss_vector = F.mse_loss(context_pred, target_proj, reduction="none").mean(dim=-1)
        else:
            raise ValueError(f"Unknown loss type '{self.loss_type}'")

        loss = loss_vector.mean()
        return {
            "loss": loss,
            "loss_vector": loss_vector,
            "context_proj": context_pred.detach(),
            "target_proj": target_proj.detach(),
        }

    def get_encoder(self) -> nn.Module:
        return self.online_encoder


def build_model_from_config(config: Dict[str, Any]) -> JEPAModel:
    model_cfg = config.get("model", {})
    loss_cfg = config.get("loss", {})
    return JEPAModel(
        encoder_name=model_cfg.get("encoder", "resnet18"),
        feature_dim=model_cfg.get("feature_dim", 512),
        projector_cfg=model_cfg.get("projector", {"hidden_dim": 1024, "output_dim": 512, "num_layers": 2}),
        predictor_cfg=model_cfg.get("predictor", {"hidden_dim": 512, "output_dim": 512, "num_layers": 2}),
        ema_momentum=model_cfg.get("ema_momentum", 0.996),
        loss_type=loss_cfg.get("type", "cosine"),
    )
