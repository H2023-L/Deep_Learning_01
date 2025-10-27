from __future__ import annotations

from typing import Dict, Optional

import lightning as L
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from torchmetrics.classification import AUROC, AveragePrecision

from mngs.models.set_transformer import SetTransformerEncoder


class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        layers = []
        if hidden_dim and hidden_dim > 0:
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1)])
        else:
            layers.append(nn.Linear(input_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class FinetuneModule(L.LightningModule):
    def __init__(
        self,
        encoder: SetTransformerEncoder,
        lr: float,
        weight_decay: float,
        freeze_encoder: bool,
        hidden_dim: Optional[int],
        dropout: float,
        pos_weight: Optional[float] = None,
        loss_type: str = "bce",
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.lr = lr
        self.weight_decay = weight_decay
        self.freeze_encoder = freeze_encoder
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.pos_weight = pos_weight
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        input_dim = encoder.pma.attn.embed_dim
        self.head = ClassificationHead(input_dim, hidden_dim, dropout)
        if pos_weight is not None:
            self.register_buffer("_pos_weight", torch.tensor(pos_weight, dtype=torch.float32))
        else:
            self._pos_weight = None
        self.train_auc = AUROC(task="binary")
        self.val_auc = AUROC(task="binary")
        self.val_ap = AveragePrecision(task="binary")
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
        logits = self.head(output.set_embedding)
        return logits

    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "bce":
            if self._pos_weight is not None:
                return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self._pos_weight)
            return F.binary_cross_entropy_with_logits(logits, targets)
        if self.loss_type == "focal":
            probs = torch.sigmoid(logits)
            ce = F.binary_cross_entropy_with_logits(
                logits,
                targets,
                pos_weight=self._pos_weight,
                reduction="none",
            )
            p_t = targets * probs + (1 - targets) * (1 - probs)
            loss = ((1 - p_t) ** self.focal_gamma * ce).mean()
            return loss
        raise ValueError(f"Unsupported loss type: {self.loss_type}")

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        logits = self.forward(batch)
        targets = batch["outcome"]
        loss = self._loss(logits, targets)
        probs = torch.sigmoid(logits)
        self.train_auc.update(probs, targets.int())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        auc = self.train_auc.compute()
        self.log("train_auc", auc, prog_bar=True)
        self.train_auc.reset()

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        logits = self.forward(batch)
        targets = batch["outcome"]
        loss = self._loss(logits, targets)
        probs = torch.sigmoid(logits)
        self.val_auc.update(probs, targets.int())
        self.val_ap.update(probs, targets.int())
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        self.log("val_auc", self.val_auc.compute(), prog_bar=True)
        self.log("val_auprc", self.val_ap.compute(), prog_bar=False)
        self.val_auc.reset()
        self.val_ap.reset()

    def configure_optimizers(self):
        optimizer = AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
