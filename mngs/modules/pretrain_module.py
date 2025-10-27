from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import lightning as L
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
import torch.nn.functional as F

from mngs.datasets.mngs_set_dataset import Sample, augment_sample, mask_abundance, pad_collate_fn
from mngs.models.set_transformer import SetTransformerEncoder


@dataclass
class ContrastiveConfig:
    keep_ratio_range: Sequence[float] = (0.6, 0.9)
    drop_quantile: float = 0.05
    noise_sigma: float = 0.1


class SelfSupervisedModule(L.LightningModule):
    def __init__(
        self,
        encoder: SetTransformerEncoder,
        lr: float,
        weight_decay: float,
        loss_weights: Dict[str, float],
        mask_prob: float = 0.15,
        contrastive_temperature: float = 0.1,
        projection_dim: int = 128,
        scheduler: str = "none",
        max_steps: int | None = None,
        contrastive_cfg: ContrastiveConfig | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_weights = loss_weights
        self.mask_prob = mask_prob
        self.contrastive_temperature = contrastive_temperature
        self.scheduler = scheduler
        self.max_steps = max_steps
        self.pad_id = 0
        self.contrastive_cfg = contrastive_cfg or ContrastiveConfig()
        hidden_dim = encoder.pma.attn.embed_dim
        self.denoise_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.save_hyperparameters(ignore=["encoder"])

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        output = self.encoder(
            batch["species"],
            batch["genus"],
            batch["type"],
            batch["abundance"],
            batch["mask"],
            batch.get("depth"),
        )
        return output.set_embedding

    def _batch_to_samples(self, batch: Dict[str, torch.Tensor]) -> List[Sample]:
        samples: List[Sample] = []
        species = batch["species"].detach().cpu()
        genus = batch["genus"].detach().cpu()
        type_ids = batch["type"].detach().cpu()
        abundance = batch["abundance"].detach().cpu()
        mask = batch["mask"].detach().cpu()
        depth = batch["depth"].detach().cpu()
        for idx in range(species.size(0)):
            length = int(mask[idx].sum().item())
            if length <= 0:
                continue
            samples.append(
                Sample(
                    sample_id=str(idx),
                    species_ids=species[idx, :length].tolist(),
                    genus_ids=genus[idx, :length].tolist(),
                    type_ids=type_ids[idx, :length].tolist(),
                    abundance=abundance[idx, :length].tolist(),
                    depth=float(depth[idx].item()),
                )
            )
        return samples

    def _denoise_loss(self, batch: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, Dict[str, float]]:
        corrupted, noise_mask = mask_abundance(batch["abundance"], batch["mask"], mask_prob=self.mask_prob)
        output = self.encoder(
            batch["species"],
            batch["genus"],
            batch["type"],
            corrupted,
            batch["mask"],
            batch.get("depth"),
        )
        pred = self.denoise_head(output.token_embeddings).squeeze(-1)
        target = batch["abundance"]
        active = noise_mask & batch["mask"]
        if active.any():
            loss = F.smooth_l1_loss(pred[active], target[active])
        else:
            loss = torch.tensor(0.0, device=self.device)
        metrics = {
            "denoise_active_frac": active.float().mean().item(),
        }
        return loss, metrics

    def _collate_to_device(self, samples: List[Sample]) -> Dict[str, torch.Tensor]:
        collated = pad_collate_fn(samples, pad_id=self.pad_id)
        batch: Dict[str, torch.Tensor] = {}
        for key, value in collated.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)
            else:
                batch[key] = value
        return batch

    def _contrastive_loss(self, batch: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, Dict[str, float]]:
        samples = self._batch_to_samples(batch)
        if len(samples) == 0:
            return torch.tensor(0.0, device=self.device), {"contrastive_pairs": 0.0}
        view1 = [
            augment_sample(
                s,
                keep_ratio_range=self.contrastive_cfg.keep_ratio_range,
                drop_quantile=self.contrastive_cfg.drop_quantile,
                noise_sigma=self.contrastive_cfg.noise_sigma,
            )
            for s in samples
        ]
        view2 = [
            augment_sample(
                s,
                keep_ratio_range=self.contrastive_cfg.keep_ratio_range,
                drop_quantile=self.contrastive_cfg.drop_quantile,
                noise_sigma=self.contrastive_cfg.noise_sigma,
            )
            for s in samples
        ]
        batch1 = self._collate_to_device(view1)
        batch2 = self._collate_to_device(view2)
        z1 = self.encoder(
            batch1["species"],
            batch1["genus"],
            batch1["type"],
            batch1["abundance"],
            batch1["mask"],
            batch1.get("depth"),
        ).set_embedding
        z2 = self.encoder(
            batch2["species"],
            batch2["genus"],
            batch2["type"],
            batch2["abundance"],
            batch2["mask"],
            batch2.get("depth"),
        ).set_embedding
        z1 = F.normalize(self.projection_head(z1), dim=-1)
        z2 = F.normalize(self.projection_head(z2), dim=-1)
        logits = torch.matmul(z1, z2.t()) / self.contrastive_temperature
        labels = torch.arange(z1.size(0), device=self.device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) * 0.5
        metrics = {"contrastive_pairs": float(len(samples))}
        return loss, metrics

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = torch.tensor(0.0, device=self.device)
        logs: Dict[str, float] = {}
        if self.loss_weights.get("denoise", 0) > 0:
            denoise_loss, denoise_metrics = self._denoise_loss(batch)
            loss = loss + denoise_loss * self.loss_weights["denoise"]
            logs["train_denoise_loss"] = denoise_loss.detach()
            for key, value in denoise_metrics.items():
                logs[f"train_{key}"] = value
        if self.loss_weights.get("contrastive", 0) > 0:
            contrastive_loss, contrastive_metrics = self._contrastive_loss(batch)
            loss = loss + contrastive_loss * self.loss_weights["contrastive"]
            logs["train_contrastive_loss"] = contrastive_loss.detach()
            for key, value in contrastive_metrics.items():
                logs[f"train_{key}"] = value
        logs["train_loss"] = loss.detach()
        self.log_dict(logs, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        loss = torch.tensor(0.0, device=self.device)
        if self.loss_weights.get("denoise", 0) > 0:
            denoise_loss, _ = self._denoise_loss(batch)
            loss = loss + denoise_loss * self.loss_weights["denoise"]
            self.log("val_denoise_loss", denoise_loss, prog_bar=True, on_epoch=True, sync_dist=False)
        if self.loss_weights.get("contrastive", 0) > 0:
            contrastive_loss, _ = self._contrastive_loss(batch)
            loss = loss + contrastive_loss * self.loss_weights["contrastive"]
            self.log("val_contrastive_loss", contrastive_loss, prog_bar=True, on_epoch=True, sync_dist=False)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=False)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.scheduler == "none":
            return optimizer
        if self.scheduler == "cosine":
            if self.max_steps is None:
                raise ValueError("max_steps must be set for cosine scheduler")
            scheduler = CosineAnnealingLR(optimizer, T_max=self.max_steps)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        if self.scheduler == "onecycle":
            if self.max_steps is None:
                raise ValueError("max_steps must be set for onecycle scheduler")
            scheduler = OneCycleLR(optimizer, max_lr=self.lr, total_steps=self.max_steps)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        raise ValueError(f"Unknown scheduler: {self.scheduler}")
