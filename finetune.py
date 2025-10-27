from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

from mngs.datasets.mngs_set_dataset import Sample, build_dataloader, sample_from_dict
from mngs.models.set_transformer import SetTransformerEncoder
from mngs.modules.finetune_module import FinetuneModule
from mngs.utils import compute_pos_weight, ensure_dir, load_json, save_json, set_seed, summarise_metrics


def load_samples(path: Path) -> List[Sample]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    samples = [sample_from_dict(item) for item in data]
    for sample in samples:
        if sample.outcome is None:
            raise ValueError(f"Outcome missing for sample {sample.sample_id}")
    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finetune Set Transformer on labeled mNGS data with k-fold CV")
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--freeze_encoder", type=str, default="true")
    parser.add_argument("--kfold", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--head_hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--loss", choices=["bce", "focal"], default="bce")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--pos_weight", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--early_stop", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--n_heads", type=int, default=None)
    parser.add_argument("--ffn_dim", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--embedding_dim", type=int, default=None)
    parser.add_argument("--depth_strategy", choices=["mlp", "add", "none"], default=None)
    return parser.parse_args()


def str_to_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "y"}


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


def infer_architecture(args: argparse.Namespace, pretrained_path: Path) -> None:
    config_path = pretrained_path.parent / "config.yaml"
    if config_path.exists():
        config = yaml.safe_load(config_path.read_text())
    else:
        config = {}
    defaults = {
        "d_model": config.get("d_model"),
        "n_heads": config.get("n_heads"),
        "ffn_dim": config.get("ffn_dim"),
        "num_layers": config.get("num_layers"),
        "embedding_dim": config.get("embedding_dim", 64),
        "depth_strategy": config.get("depth_strategy", "mlp"),
    }
    for key, value in defaults.items():
        if getattr(args, key) is None and value is not None:
            setattr(args, key, value)
    missing = [key for key in ["d_model", "n_heads", "ffn_dim", "num_layers", "embedding_dim", "depth_strategy"] if getattr(args, key) is None]
    if missing:
        raise ValueError(f"Missing architecture parameters: {missing}. Provide via CLI or ensure pretrain config.yaml exists")


def evaluate_predictions(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    metrics = {
        "auroc": roc_auc_score(y_true, probs),
        "auprc": average_precision_score(y_true, probs),
        "accuracy": accuracy_score(y_true, probs >= 0.5),
        "f1": f1_score(y_true, probs >= 0.5),
        "brier": brier_score_loss(y_true, probs),
    }
    return metrics


def youden_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    j_scores = tpr - fpr
    idx = np.argmax(j_scores)
    return float(thresholds[idx])


def plot_and_save(fold_dir: Path, y_true: np.ndarray, probs: np.ndarray) -> Dict[str, Path]:
    fold_dir.mkdir(parents=True, exist_ok=True)
    fpr, tpr, roc_thresholds = roc_curve(y_true, probs)
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thresholds})
    roc_csv = fold_dir / "roc.csv"
    roc_df.to_csv(roc_csv, index=False)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUROC={roc_auc_score(y_true, probs):.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    roc_png = fold_dir / "roc.png"
    plt.savefig(roc_png, dpi=200)
    plt.close()

    precision, recall, pr_thresholds = precision_recall_curve(y_true, probs)
    pr_df = pd.DataFrame({"precision": precision, "recall": recall})
    pr_csv = fold_dir / "pr.csv"
    pr_df.to_csv(pr_csv, index=False)
    plt.figure()
    plt.plot(recall, precision, label=f"AUPRC={average_precision_score(y_true, probs):.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    pr_png = fold_dir / "pr.png"
    plt.savefig(pr_png, dpi=200)
    plt.close()

    prob_true, prob_pred = calibration_curve(y_true, probs, strategy="quantile", n_bins=10)
    calib_df = pd.DataFrame({"prob_true": prob_true, "prob_pred": prob_pred})
    calib_csv = fold_dir / "calibration.csv"
    calib_df.to_csv(calib_csv, index=False)
    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Calibration Curve")
    calib_png = fold_dir / "calibration.png"
    plt.savefig(calib_png, dpi=200)
    plt.close()

    return {"roc_csv": roc_csv, "roc_png": roc_png, "pr_csv": pr_csv, "pr_png": pr_png, "calib_csv": calib_csv, "calib_png": calib_png}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.out_dir)
    infer_architecture(args, args.pretrained)

    samples = load_samples(args.cache_dir / "samples_labeled.pkl")
    labels = np.array([s.outcome for s in samples], dtype=np.int64)

    freeze_encoder = str_to_bool(args.freeze_encoder)
    loss_metrics: List[Dict[str, float]] = []
    preds_records: List[Dict[str, object]] = []
    best_model_path: Optional[Path] = None
    best_metric = -np.inf
    splits: List[Dict[str, List[str]]] = []

    skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros_like(labels), labels), start=1):
        fold_dir = args.out_dir / f"fold_{fold_idx}"
        train_samples = [samples[i] for i in train_idx]
        val_samples = [samples[i] for i in val_idx]
        splits.append({
            "fold": fold_idx,
            "train_ids": [s.sample_id for s in train_samples],
            "val_ids": [s.sample_id for s in val_samples],
        })

        train_loader = build_dataloader(train_samples, args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = build_dataloader(val_samples, args.batch_size, shuffle=False, num_workers=args.num_workers)

        encoder = build_encoder(args.cache_dir, args)
        state_dict = torch.load(args.pretrained, map_location="cpu")
        encoder.load_state_dict(state_dict, strict=False)

        train_labels = [s.outcome for s in train_samples]
        if args.pos_weight == "auto":
            pos_weight = compute_pos_weight(train_labels)
        elif args.pos_weight.lower() in {"none", "null"}:
            pos_weight = None
        else:
            pos_weight = float(args.pos_weight)

        module = FinetuneModule(
            encoder=encoder,
            lr=args.lr,
            weight_decay=args.weight_decay,
            freeze_encoder=freeze_encoder,
            hidden_dim=args.head_hidden,
            dropout=args.dropout,
            pos_weight=pos_weight,
            loss_type=args.loss,
            focal_gamma=args.focal_gamma,
        )

        checkpoint = ModelCheckpoint(
            dirpath=fold_dir,
            monitor="val_auc",
            mode="max",
            save_top_k=1,
            filename=f"finetune-fold{fold_idx:02d}-{{epoch:02d}}-{{val_auc:.4f}}",
        )
        early_stop = EarlyStopping(monitor="val_auc", mode="max", patience=args.early_stop)

        trainer = L.Trainer(
            max_epochs=args.epochs,
            accelerator="auto",
            devices=1,
            callbacks=[checkpoint, early_stop],
            precision=args.precision,
            gradient_clip_val=args.gradient_clip,
            log_every_n_steps=5,
        )
        trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

        best_path = checkpoint.best_model_path
        if not best_path:
            best_path = checkpoint.last_model_path
        state = torch.load(best_path, map_location="cpu")
        module.load_state_dict(state["state_dict"])
        module.eval()
        module.cpu()

        probs_list: List[float] = []
        labels_list: List[int] = []
        sample_ids: List[str] = []
        with torch.no_grad():
            for batch in val_loader:
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value
                logits = module.forward(batch)
                probs = torch.sigmoid(logits)
                probs_cpu = probs.cpu().tolist()
                labels_cpu = batch["outcome"].cpu().tolist()
                probs_list.extend(probs_cpu)
                labels_list.extend(labels_cpu)
                sample_ids.extend(batch["sample_id"])

        y_true = np.array(labels_list)
        probs = np.array(probs_list)
        metrics = evaluate_predictions(y_true, probs)
        metrics["threshold_youden"] = youden_threshold(y_true, probs)
        loss_metrics.append(metrics)
        plot_and_save(fold_dir, y_true, probs)

        for sid, prob, label in zip(sample_ids, probs_list, labels_list):
            preds_records.append(
                {
                    "fold": fold_idx,
                    "sample_id": sid,
                    "prob": float(prob),
                    "label": int(label),
                }
            )

        if metrics["auroc"] > best_metric:
            best_metric = metrics["auroc"]
            best_model_path = Path(best_path)

        save_json(metrics, fold_dir / "metrics.json")

    summary = {}
    for key in ["auroc", "auprc", "accuracy", "f1", "brier"]:
        values = [m[key] for m in loss_metrics]
        mean, ci95 = summarise_metrics(values)
        summary[key] = {"mean": mean, "ci95": ci95}
    summary["threshold_youden"] = float(np.mean([m["threshold_youden"] for m in loss_metrics]))
    save_json(summary, args.out_dir / "metrics_summary.json")

    preds_df = pd.DataFrame(preds_records)
    preds_df.to_csv(args.out_dir / "preds.csv", index=False)

    save_json({"folds": splits}, args.out_dir / "folds.json")

    if best_model_path is not None:
        torch.save(torch.load(best_model_path, map_location="cpu"), args.out_dir / "finetuned.ckpt")

    (args.out_dir / "config.yaml").write_text(yaml.dump(vars(args)), encoding="utf-8")

    print(f"Finetuning complete. Results saved to {args.out_dir}")


if __name__ == "__main__":
    main()
