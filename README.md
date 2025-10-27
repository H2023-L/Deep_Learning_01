# mNGS Set Transformer Pipeline

This repository implements an end-to-end workflow for learning patient-level representations from metagenomic next-generation sequencing (mNGS) results and predicting downstream clinical outcomes. The pipeline covers preprocessing, self-supervised pretraining with a Set Transformer encoder, supervised fine-tuning with cross-validation, evaluation/visualisation, and baseline comparisons.

## Key features

- **Set-aware modelling**: samples are modelled as variable-length sets of pathogen detections using a configurable Set Transformer encoder with SAB and PMA blocks.
- **Self-supervised objectives**: denoising reconstruction and contrastive NT-Xent losses for pretraining on large unlabeled cohorts.
- **Supervised fine-tuning**: lightweight classification head with k-fold cross-validation, calibration curves, ROC/PR plotting, and metrics summarisation with confidence intervals.
- **Baselines and diagnostics**: logistic regression and optional XGBoost bag-of-species baselines sharing the exact CV splits, plus aggregate evaluation utilities.
- **Reproducibility**: YAML/JSON artefacts, deterministic seeds, and CLI interfaces for every stage.

## Repository layout

```
├── baselines.py          # Baseline models (logistic regression / XGBoost)
├── evaluate.py           # Aggregate evaluation + plotting
├── finetune.py           # K-fold fine-tuning script
├── pretrain.py           # Self-supervised pretraining
├── preprocess.py         # Data preprocessing & caching
├── scripts/
│   └── generate_toy_data.py  # Toy data generator for smoke tests
├── mngs/
│   ├── datasets/
│   │   └── mngs_set_dataset.py
│   ├── models/
│   │   └── set_transformer.py
│   ├── modules/
│   │   ├── finetune_module.py
│   │   └── pretrain_module.py
│   └── utils.py
└── configs/
    └── default.yaml      # (placeholder for custom configs)
```

All artefacts produced by the pipeline are written to the `artifacts/` directory (configurable via CLI options).

## Installation

This project requires Python ≥ 3.10. Install dependencies with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install pandas numpy scikit-learn torch torchmetrics lightning matplotlib openpyxl pyyaml tqdm rich
# Optional
pip install xgboost lightgbm
```

## Toy data (for quick verification)

To spin up a miniature dataset:

```bash
python scripts/generate_toy_data.py
```

This creates `data/unlabeled.xlsx` (50 unlabeled samples) and `data/labeled.xlsx` (12 labeled samples with outcomes encoded as 0/1). All subsequent commands default to these locations in the examples below.

## 1. Preprocessing

Construct vocabularies, filter noisy detections, derive abundance features, and cache serialised samples for downstream scripts.

```bash
python preprocess.py \
  --unlabeled_xlsx ./data/unlabeled.xlsx \
  --labeled_xlsx   ./data/labeled.xlsx \
  --min_reads 2 \
  --abundance cpm_log1p \
  --out_dir ./artifacts/preprocessed
```

Key outputs:

- `samples_unlabeled.pkl`, `samples_labeled.pkl`: serialised sample dictionaries.
- `vocab.json`: Species/Genus/Type vocabularies (includes `<PAD>` / `<UNK>` tokens).
- `baseline_features_labeled.npz`: bag-of-species matrix for baseline models.
- `config.yaml`: preprocessing configuration snapshot.
- `outcome_mapping.json`: mapping of raw labels (e.g., `"yes"`, `"alive"`) to 0/1.

## 2. Self-supervised pretraining

Train the Set Transformer encoder on the unlabeled cohort with denoising + contrastive objectives:

```bash
python pretrain.py \
  --cache_dir ./artifacts/preprocessed \
  --d_model 256 --n_heads 4 --ffn_dim 512 --num_layers 3 \
  --loss_weights denoise=1.0,contrastive=0.5 \
  --batch_size 32 --max_len 512 \
  --epochs 5 --early_stop 2 \
  --lr 3e-4 --weight_decay 1e-2 \
  --precision bf16 --grad_clip 1.0 \
  --out_dir ./artifacts/pretrain
```

Outputs:

- `encoder.ckpt`: weights for the encoder only (ready for fine-tuning).
- `config.yaml`: run configuration snapshot.
- Lightning checkpoints and logs under `./artifacts/pretrain`.

## 3. Fine-tuning with k-fold CV

Fine-tune a classification head on the labeled cohort while reusing the pretrained encoder:

```bash
python finetune.py \
  --cache_dir ./artifacts/preprocessed \
  --pretrained ./artifacts/pretrain/encoder.ckpt \
  --freeze_encoder true \
  --kfold 5 --seed 42 \
  --head_hidden 128 \
  --loss bce --pos_weight auto \
  --epochs 20 --early_stop 5 \
  --lr 1e-3 --weight_decay 1e-3 \
  --out_dir ./artifacts/finetune
```

Per-fold artefacts (under `fold_*/`):

- `metrics.json`: AUROC, AUPRC, Accuracy, F1, Brier, Youden threshold.
- `roc.csv` / `roc.png`, `pr.csv` / `pr.png`, `calibration.csv` / `calibration.png`.

Global artefacts:

- `metrics_summary.json`: mean ± 95% CI across folds.
- `preds.csv`: per-sample predictions (Sample_ID, fold, probability, label).
- `folds.json`: exact train/validation splits for reproducibility.
- `finetuned.ckpt`: Lightning checkpoint (encoder + head).
- `config.yaml`: fine-tuning configuration.

## 4. Aggregate evaluation & plots

Combine fold predictions, recompute overall curves, and export summary plots:

```bash
python evaluate.py \
  --finetune_dir ./artifacts/finetune \
  --plots_dir ./artifacts/plots
```

Outputs include `roc_overall.*`, `pr_overall.*`, `calibration_overall.*`, and `evaluation_summary.json` (containing the per-fold metrics plus overall AUROC/AUPRC).

## 5. Baseline models

Run sparse bag-of-species baselines on the identical CV splits produced during fine-tuning:

```bash
python baselines.py \
  --cache_dir ./artifacts/preprocessed \
  --folds ./artifacts/finetune/folds.json \
  --out_dir ./artifacts/baselines
```

This evaluates:

- **Logistic regression** (L2 regularised, class-balanced).
- **XGBoost** (if `xgboost` is installed). The script gracefully skips this model when the dependency is absent.

Results are written to `baseline_metrics.json`, listing AUROC/AUPRC per fold for each baseline.

## Outcome encoding

During preprocessing, outcomes are normalised to binary labels using the following mapping (case-insensitive, whitespace trimmed):

- Negative → `{0, "no", "false", "neg", "negative", "alive", "absent"}`
- Positive → `{1, "yes", "true", "pos", "positive", "dead", "present"}`

Any unrecognised labels raise a descriptive error.

## Configuration tips

- **Abundance transformations**: choose between `cpm_log1p` (default) or `fraction_log1p` via `--abundance` in `preprocess.py`.
- **Noise filtering**: adjust `--min_reads`, `--min_relative_abundance`, and `--drop_tail_quantile` for domain-specific denoising.
- **Depth handling**: the encoder can combine sequencing depth via MLP concatenation (`--depth_strategy mlp`, default), additive fusion (`add`), or disable it (`none`).
- **Loss balancing**: the self-supervised loss weights accept comma-separated `key=value` pairs (e.g., `denoise=1.0,contrastive=0.25`).
- **Fine-tuning loss**: choose between `bce` (optionally weighted through `--pos_weight`) and `focal` (`--focal_gamma` controls focusing strength).

## Reproducibility

- Set seeds via `--seed` in `pretrain.py` and `finetune.py`.
- Every stage writes a `config.yaml` and the vocab/outcome mappings necessary to reproduce the experiment.
- K-fold assignments are persisted (`folds.json`) and reused by the baseline script.

## Next steps / extensibility

- **Temperature scaling**: `finetune.py` exposes logits and predictions; integrate temperature scaling or Platt calibration in `evaluate.py`.
- **Feature importance**: extend `baselines.py` or fine-tune module with SHAP/permutation hooks.
- **Prediction service**: wrap the encoder+head in a `predict.py` utility to score new Excel files (interface hooks are ready via the cached vocabularies and `Sample` serialisation).

## Troubleshooting

- Ensure Excel files have the expected columns: `Sample_ID`, `Species`, `Genus`, `Type`, `Reads`, and optionally `Outcome`.
- Samples with insufficient reads (after filtering) are dropped during preprocessing; check the preprocessing logs if counts look lower than expected.
- When GPU memory is constrained, reduce `--batch_size` and/or `--max_len`, or switch `--precision` to `32`.

## License

MIT License. See `LICENSE` (add your preferred licence file here if needed).
