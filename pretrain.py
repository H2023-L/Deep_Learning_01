from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from torch.utils.data import random_split
import torch
import yaml

from mngs.datasets.mngs_set_dataset import Sample, build_dataloader, sample_from_dict, truncate_sample
from mngs.models.set_transformer import SetTransformerEncoder
from mngs.modules.pretrain_module import ContrastiveConfig, SelfSupervisedModule
from mngs.utils import ensure_dir, load_json, set_seed


def parse_loss_weights(spec: str) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for part in spec.split(","):
        key, value = part.split("=")
        result[key.strip()] = float(value)
    return result


def load_samples(path: Path) -> List[Sample]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return [sample_from_dict(item) for item in data]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-supervised pretraining for mNGS set transformer")
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--ffn_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--depth_strategy", choices=["mlp", "add", "none"], default="mlp")
    parser.add_argument("--loss_weights", type=str, default="denoise=1.0,contrastive=0.5")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--scheduler", choices=["none", "cosine", "onecycle"], default="none")
    parser.add_argument("--mask_prob", type=float, default=0.15)
    parser.add_argument("--contrastive_temp", type=float, default=0.1)
    parser.add_argument("--projection_dim", type=int, default=128)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--contrastive_sigma", type=float, default=0.1)
    parser.add_argument("--contrastive_keep_min", type=float, default=0.6)
    parser.add_argument("--contrastive_keep_max", type=float, default=0.9)
    parser.add_argument("--contrastive_drop_quantile", type=float, default=0.05)
    return parser.parse_args()


def build_encoder(cache_dir: Path, args: argparse.Namespace) -> SetTransformerEncoder:
    vocab = load_json(cache_dir / "vocab.json")
    species_vocab = len(vocab["species"]["id_to_token"])
    genus_vocab = len(vocab["genus"]["id_to_token"])
    type_vocab = len(vocab["type"]["id_to_token"])
    encoder = SetTransformerEncoder(
        species_vocab_size=species_vocab,
        genus_vocab_size=genus_vocab,
        type_vocab_size=type_vocab,
        embedding_dim=args.embedding_dim,
        model_dim=args.d_model,
        num_heads=args.n_heads,
        num_layers=args.num_layers,
        ffn_dim=args.ffn_dim,
        dropout=0.1,
        depth_strategy=args.depth_strategy,
    )
    return encoder


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.out_dir)

    samples = load_samples(args.cache_dir / "samples_unlabeled.pkl")
    if len(samples) == 0:
        raise RuntimeError("No unlabeled samples found. Run preprocess.py first")
    loss_weights = parse_loss_weights(args.loss_weights)

    encoder = build_encoder(args.cache_dir, args)

    if args.max_len:
        samples = [truncate_sample(sample, args.max_len) for sample in samples]
    dataset = samples
    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("Validation split too large for dataset")
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = build_dataloader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = build_dataloader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)

    max_steps = args.epochs * len(train_loader)
    contrastive_cfg = ContrastiveConfig(
        keep_ratio_range=(args.contrastive_keep_min, args.contrastive_keep_max),
        drop_quantile=args.contrastive_drop_quantile,
        noise_sigma=args.contrastive_sigma,
    )

    module = SelfSupervisedModule(
        encoder=encoder,
        lr=args.lr,
        weight_decay=args.weight_decay,
        loss_weights=loss_weights,
        mask_prob=args.mask_prob,
        contrastive_temperature=args.contrastive_temp,
        projection_dim=args.projection_dim,
        scheduler=args.scheduler,
        max_steps=max_steps,
        contrastive_cfg=contrastive_cfg,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=args.out_dir,
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            filename="pretrain-{epoch:02d}-{val_loss:.4f}",
        ),
        EarlyStopping(monitor="val_loss", patience=args.early_stop, mode="min"),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        callbacks=callbacks,
        precision="bf16-mixed" if args.precision == "bf16" else args.precision,
        gradient_clip_val=args.grad_clip,
        log_every_n_steps=10,
        deterministic=False,
    )

    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_path = callbacks[0].best_model_path
    if best_path:
        state = torch.load(best_path, map_location="cpu", weights_only=False)
        module.load_state_dict(state["state_dict"])
    encoder_state = module.encoder.state_dict()
    torch.save(encoder_state, args.out_dir / "encoder.ckpt")
    data = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    (args.out_dir / "config.yaml").write_text(yaml.dump(data), encoding="utf-8")

    print(f"Pretraining complete. Encoder weights saved to {args.out_dir / 'encoder.ckpt'}")


if __name__ == "__main__":
    main()
