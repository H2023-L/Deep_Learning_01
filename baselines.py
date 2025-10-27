from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from mngs.utils import ensure_dir, load_json, save_json

try:  # pragma: no cover - optional dependency
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline models for mNGS outcome prediction")
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=False, help="Path to folds.json from finetuning")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--kfold", type=int, default=5, help="Used if folds file missing")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_features(cache_dir: Path) -> Dict[str, np.ndarray]:
    path = cache_dir / "baseline_features_labeled.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    features = data["features"].astype(np.float32)
    if features.shape[1] > 0:
        features = features[:, 1:]
    return {
        "features": features,
        "sample_ids": data["sample_ids"].tolist(),
        "outcomes": data["outcomes"].astype(int),
    }


def compute_metrics(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    return {
        "auroc": roc_auc_score(y_true, probs),
        "auprc": average_precision_score(y_true, probs),
    }


def run_logistic(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray) -> np.ndarray:
    scaler = StandardScaler(with_mean=False)
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    clf = LogisticRegression(max_iter=5000, solver="saga", penalty="l2", class_weight="balanced")
    clf.fit(X_train_scaled, y_train)
    probs = clf.predict_proba(X_val_scaled)[:, 1]
    return probs


def run_xgboost(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray) -> np.ndarray:
    if xgb is None:
        raise RuntimeError("xgboost is not installed")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "eta": 0.05,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "seed": 42,
    }
    booster = xgb.train(params, dtrain, num_boost_round=200)
    return booster.predict(dval)


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)
    data = load_features(args.cache_dir)
    X = data["features"]
    y = data["outcomes"]
    sample_ids = data["sample_ids"]

    if args.folds and Path(args.folds).exists():
        folds_data = load_json(args.folds)
        folds = folds_data["folds"]
    else:
        skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
        folds = []
        for fold_idx, (_, val_idx) in enumerate(skf.split(X, y), start=1):
            folds.append({
                "fold": fold_idx,
                "train_ids": [sample_ids[i] for i in range(len(sample_ids)) if i not in val_idx],
                "val_ids": [sample_ids[i] for i in val_idx],
            })

    metrics_logreg: List[Dict[str, float]] = []
    metrics_xgb: List[Dict[str, float]] = []
    for fold in folds:
        train_mask = np.isin(sample_ids, fold["train_ids"])
        val_mask = np.isin(sample_ids, fold["val_ids"])
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        probs_lr = run_logistic(X_train, y_train, X_val)
        metrics_logreg.append(compute_metrics(y_val, probs_lr))
        if xgb is not None:
            probs_xgb = run_xgboost(X_train, y_train, X_val)
            metrics_xgb.append(compute_metrics(y_val, probs_xgb))

    results = {
        "logistic_regression": metrics_logreg,
    }
    if metrics_xgb:
        results["xgboost"] = metrics_xgb
    save_json(results, args.out_dir / "baseline_metrics.json")
    print(f"Baseline evaluation complete. Results saved to {args.out_dir}")


if __name__ == "__main__":
    main()
