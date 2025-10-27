from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from mngs.utils import ensure_dir, load_json, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate evaluation for finetuned models")
    parser.add_argument("--finetune_dir", type=Path, required=True)
    parser.add_argument("--plots_dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.plots_dir)
    preds_path = args.finetune_dir / "preds.csv"
    if not preds_path.exists():
        raise FileNotFoundError(preds_path)
    preds = pd.read_csv(preds_path)
    y_true = preds["label"].to_numpy()
    probs = preds["prob"].to_numpy()

    auroc = roc_auc_score(y_true, probs)
    auprc = average_precision_score(y_true, probs)

    fpr, tpr, thresholds = roc_curve(y_true, probs)
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})
    roc_csv = args.plots_dir / "roc_overall.csv"
    roc_df.to_csv(roc_csv, index=False)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUROC={auroc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Overall ROC")
    plt.legend()
    roc_png = args.plots_dir / "roc_overall.png"
    plt.savefig(roc_png, dpi=200)
    plt.close()

    precision, recall, pr_thresholds = precision_recall_curve(y_true, probs)
    pr_df = pd.DataFrame({"precision": precision, "recall": recall})
    pr_csv = args.plots_dir / "pr_overall.csv"
    pr_df.to_csv(pr_csv, index=False)
    plt.figure()
    plt.plot(recall, precision, label=f"AUPRC={auprc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Overall Precision-Recall")
    plt.legend()
    pr_png = args.plots_dir / "pr_overall.png"
    plt.savefig(pr_png, dpi=200)
    plt.close()

    prob_true, prob_pred = calibration_curve(y_true, probs, strategy="quantile", n_bins=10)
    calib_df = pd.DataFrame({"prob_true": prob_true, "prob_pred": prob_pred})
    calib_csv = args.plots_dir / "calibration_overall.csv"
    calib_df.to_csv(calib_csv, index=False)
    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Overall Calibration")
    calib_png = args.plots_dir / "calibration_overall.png"
    plt.savefig(calib_png, dpi=200)
    plt.close()

    summary_path = args.finetune_dir / "metrics_summary.json"
    if summary_path.exists():
        summary = load_json(summary_path)
    else:
        summary = {}
    summary.update(
        {
            "overall": {
                "auroc": auroc,
                "auprc": auprc,
            }
        }
    )
    save_json(summary, args.plots_dir / "evaluation_summary.json")
    print(f"Evaluation complete. Plots saved to {args.plots_dir}")


if __name__ == "__main__":
    main()
