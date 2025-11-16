"""Training loop for JEPA on STL-10."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import SGD
from tqdm import tqdm

from ..daw.curriculum import WeightScheduler, build_curriculum
from ..daw.difficulty import DifficultyBuffer
from ..daw.weighting import normalize_weights
from ..data.stl10 import DataConfig, create_labeled_loader, create_unlabeled_loader
from ..models.jepa import JEPAModel
from ..utils.checkpoint import DEFAULT_CHECKPOINT_NAME, load_checkpoint, save_checkpoint
from ..utils.distributed import is_primary
from ..utils.logging import setup_logging
from ..utils.seed import seed_everything


@dataclass
class OptimizerConfig:
    optimizer: str
    base_lr: float
    momentum: float
    weight_decay: float
    warmup_epochs: int
    epochs: int
    grad_clip: Optional[float]
    amp: bool
    save_interval: int
    log_interval: int
    knn_interval: int
    subset_knn_size: int
    device: str
    final_lr: float


class PretrainTrainer:
    def __init__(self, config: Dict, resume_path: Optional[str] = None, run_name_override: Optional[str] = None) -> None:
        self.config = config
        self.run_name = run_name_override or config.get("run_name", f"dynajepa_{int(time.time())}")
        self.output_dir = Path("outputs") / self.run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger, self.writer = setup_logging(self.output_dir, self.run_name)

        seed = config.get("seed", 0)
        seed_everything(seed)

        training_cfg = config.get("training", {})
        self.optim_cfg = OptimizerConfig(
            optimizer=training_cfg.get("optimizer", "sgd"),
            base_lr=training_cfg.get("base_lr", 0.1),
            momentum=training_cfg.get("momentum", 0.9),
            weight_decay=training_cfg.get("weight_decay", 1e-6),
            warmup_epochs=training_cfg.get("warmup_epochs", 0),
            epochs=training_cfg.get("epochs", 100),
            grad_clip=training_cfg.get("grad_clip"),
            amp=training_cfg.get("amp", True),
            save_interval=training_cfg.get("save_interval", 10),
            log_interval=training_cfg.get("log_interval", 10),
            knn_interval=training_cfg.get("knn_interval", 50),
            subset_knn_size=training_cfg.get("subset_knn_size", 2000),
            device=training_cfg.get("device", "cuda"),
            final_lr=training_cfg.get("final_lr", 0.0),
        )

        requested_device = self.optim_cfg.device
        if requested_device.startswith("cuda"):
            self.device = torch.device(requested_device if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(requested_device)

        data_cfg = config.get("data", {})
        data_config = DataConfig(
            dataset_root=data_cfg.get("dataset_root", "./data"),
            batch_size=data_cfg.get("batch_size", 256),
            num_workers=data_cfg.get("num_workers", 8),
            pin_memory=data_cfg.get("pin_memory", True),
            context_crop_ratio=data_cfg.get("context_crop_ratio", 0.6),
            target_crop_ratio=data_cfg.get("target_crop_ratio", 0.4),
            drop_last=data_cfg.get("drop_last", True),
            subset_size=data_cfg.get("subset_size"),
        )

        self.train_loader, self.dataset_size = create_unlabeled_loader(data_config)

        model_cfg = config.get("model", {})
        loss_cfg = config.get("loss", {})
        self.model = JEPAModel(
            encoder_name=model_cfg.get("encoder", "resnet18"),
            feature_dim=model_cfg.get("feature_dim", 512),
            projector_cfg=model_cfg.get("projector", {"hidden_dim": 1024, "output_dim": 512, "num_layers": 2}),
            predictor_cfg=model_cfg.get("predictor", {"hidden_dim": 512, "output_dim": 512, "num_layers": 2}),
            ema_momentum=model_cfg.get("ema_momentum", 0.996),
            loss_type=loss_cfg.get("type", "cosine"),
        ).to(self.device)

        if self.optim_cfg.optimizer.lower() == "lars":
            self.optimizer = torch.optim.LARS(
                self.model.parameters(),
                lr=self.optim_cfg.base_lr,
                momentum=self.optim_cfg.momentum,
                weight_decay=self.optim_cfg.weight_decay,
            )
        else:
            self.optimizer = SGD(
                self.model.parameters(),
                lr=self.optim_cfg.base_lr,
                momentum=self.optim_cfg.momentum,
                weight_decay=self.optim_cfg.weight_decay,
            )
        self.scaler = GradScaler(enabled=self.optim_cfg.amp)

        daw_cfg = config.get("daw", {"enabled": False})
        self.daw_enabled = daw_cfg.get("enabled", False)
        self.weight_clip_min = float(daw_cfg.get("clip_min", 0.1))
        self.weight_clip_max = float(daw_cfg.get("clip_max", 10.0))
        self.difficulty_buffer: Optional[DifficultyBuffer]
        if self.daw_enabled:
            alpha = float(daw_cfg.get("ema_alpha", 0.0))
            self.difficulty_buffer = DifficultyBuffer(self.dataset_size, alpha=alpha, device=self.device)
            self.weight_scheduler: WeightScheduler = build_curriculum(daw_cfg, config.get("curriculum", {}))
        else:
            self.difficulty_buffer = None
            self.weight_scheduler = WeightScheduler(base_fn=lambda x: torch.ones_like(x))

        self.start_epoch = 0
        self.global_step = 0
        if resume_path is not None:
            self._load_resume(resume_path)

        self.logger.info("Run %s starting with %d epochs", self.run_name, self.optim_cfg.epochs)

    def _load_resume(self, path: str) -> None:
        checkpoint = load_checkpoint(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scaler"):
            self.scaler.load_state_dict(checkpoint["scaler"])
        self.start_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        if self.difficulty_buffer is not None and "difficulty_buffer" in checkpoint:
            state = checkpoint["difficulty_buffer"]
            self.difficulty_buffer.buffer.copy_(state["buffer"])
            self.difficulty_buffer.initialized.copy_(state["initialized"])
        self.logger.info("Resumed from %s at epoch %d", path, self.start_epoch)

    def _save_checkpoint(self, epoch: int) -> Path:
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "config": self.config,
        }
        if self.difficulty_buffer is not None:
            state["difficulty_buffer"] = {
                "buffer": self.difficulty_buffer.buffer.detach().cpu(),
                "initialized": self.difficulty_buffer.initialized.detach().cpu(),
            }
        filename = (
            f"epoch_{epoch:03d}.pt"
            if (epoch % self.optim_cfg.save_interval == 0 or epoch == self.optim_cfg.epochs)
            else DEFAULT_CHECKPOINT_NAME
        )
        path = save_checkpoint(state, self.output_dir / "checkpoints", filename)
        self.logger.info("Saved checkpoint to %s", path)
        return path

    def _adjust_learning_rate(self, epoch: int) -> None:
        if epoch < self.optim_cfg.warmup_epochs:
            lr = self.optim_cfg.base_lr * float(epoch + 1) / max(1, self.optim_cfg.warmup_epochs)
        else:
            progress = (epoch - self.optim_cfg.warmup_epochs) / max(
                1, self.optim_cfg.epochs - self.optim_cfg.warmup_epochs
            )
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            lr = self.optim_cfg.final_lr + (self.optim_cfg.base_lr - self.optim_cfg.final_lr) * cosine
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def train(self) -> None:
        total_epochs = self.optim_cfg.epochs
        for epoch in range(self.start_epoch, total_epochs):
            self._adjust_learning_rate(epoch)
            epoch_loss = 0.0
            progress = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{total_epochs}", leave=False)
            for step, batch in enumerate(progress):
                context, target, indices = batch
                context = context.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                indices = torch.as_tensor(indices, device=self.device)

                self.optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=self.optim_cfg.amp):
                    outputs = self.model(context, target)
                    loss_vector = outputs["loss_vector"]
                    if self.daw_enabled and self.difficulty_buffer is not None:
                        difficulty_values = self.difficulty_buffer.update(indices, loss_vector.detach())
                        weights = self.weight_scheduler(difficulty_values, epoch=epoch).to(self.device)
                    else:
                        weights = torch.ones_like(loss_vector, device=self.device)
                    weights = normalize_weights(weights, self.weight_clip_min, self.weight_clip_max)
                    loss = torch.sum(weights * loss_vector) / torch.sum(weights)

                self.scaler.scale(loss).backward()
                if self.optim_cfg.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.optim_cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.model.update_target()

                epoch_loss += loss.item() * context.size(0)
                self.global_step += 1

                if step % self.optim_cfg.log_interval == 0 and is_primary():
                    mean_loss = loss_vector.mean().item()
                    self.logger.info(
                        "Epoch %d Step %d | loss %.4f | mean_weight %.3f",
                        epoch,
                        step,
                        mean_loss,
                        weights.mean().item(),
                    )
                    self.writer.add_scalar("train/loss", mean_loss, self.global_step)
                    self.writer.add_scalar("train/weight_mean", weights.mean().item(), self.global_step)
                    self.writer.add_scalar("train/weight_std", weights.std().item(), self.global_step)

            if (epoch + 1) % self.optim_cfg.save_interval == 0 or epoch + 1 == total_epochs:
                self._save_checkpoint(epoch + 1)

            if (epoch + 1) % self.optim_cfg.knn_interval == 0:
                self.export_embeddings(epoch + 1)

            avg_loss = epoch_loss / len(self.train_loader.dataset)
            self.logger.info("Epoch %d completed with avg loss %.4f", epoch + 1, avg_loss)

    @torch.no_grad()
    def export_embeddings(self, epoch: int) -> None:
        encoder = self.model.get_encoder().to(self.device)
        encoder.eval()
        loader = create_labeled_loader(
            root=self.config["data"].get("dataset_root", "./data"),
            split="train",
            batch_size=128,
            num_workers=self.config["data"].get("num_workers", 8),
            pin_memory=self.config["data"].get("pin_memory", True),
            shuffle=False,
        )
        features = []
        labels = []
        count = 0
        limit = self.optim_cfg.subset_knn_size
        for images, label, _ in loader:
            images = images.to(self.device)
            emb = encoder(images)
            features.append(emb.cpu())
            labels.append(label)
            count += images.size(0)
            if count >= limit:
                break
        feature_tensor = torch.cat(features)[:limit]
        label_tensor = torch.cat(labels)[:limit]
        export_path = self.output_dir / "features" / f"epoch_{epoch:03d}.pt"
        torch.save({"features": feature_tensor, "labels": label_tensor}, export_path)
        self.logger.info("Exported %d embeddings to %s", feature_tensor.size(0), export_path)
