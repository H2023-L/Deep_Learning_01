from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from mngs.datasets.mngs_set_dataset import Sample
from mngs.utils import OutcomeMapper, Vocabulary, ensure_dir, normalise_text, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess mNGS excel files for set transformer training")
    parser.add_argument("--unlabeled_xlsx", type=Path, required=True)
    parser.add_argument("--labeled_xlsx", type=Path, required=True)
    parser.add_argument("--min_reads", type=int, default=2)
    parser.add_argument("--min_relative_abundance", type=float, default=0.0)
    parser.add_argument(
        "--abundance",
        choices=["cpm_log1p", "fraction_log1p"],
        default="cpm_log1p",
        help="Transformation applied to abundance feature",
    )
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--min_items", type=int, default=1, help="Minimum number of entries per sample to keep")
    parser.add_argument(
        "--drop_tail_quantile",
        type=float,
        default=0.0,
        help="Drop entries below this within-sample abundance quantile (0-1)",
    )
    return parser.parse_args()


def read_excel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_excel(path)
    expected = {"Sample_ID", "Species", "Genus", "Type", "Reads"}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")
    df = df.copy()
    for col in ["Sample_ID", "Species", "Genus", "Type"]:
        df[col] = df[col].apply(normalise_text)
    df["Reads"] = pd.to_numeric(df["Reads"], errors="coerce").fillna(0).astype(float)
    return df


def build_vocabularies(dfs: List[pd.DataFrame]) -> Dict[str, Vocabulary]:
    species = []
    genus = []
    types = []
    for df in dfs:
        species.extend(df["Species"].tolist())
        genus.extend(df["Genus"].tolist())
        types.extend(df["Type"].tolist())
    vocab = {
        "species": Vocabulary.build(species),
        "genus": Vocabulary.build(genus),
        "type": Vocabulary.build(types),
    }
    return vocab


def compute_abundance(reads: np.ndarray, depth: float, mode: str) -> np.ndarray:
    if depth <= 0:
        raise ValueError("Sample depth must be positive")
    if mode == "cpm_log1p":
        cpm = (reads / depth) * 1e6
        return np.log1p(cpm)
    if mode == "fraction_log1p":
        fraction = reads / depth
        return np.log1p(fraction)
    raise ValueError(f"Unknown abundance mode: {mode}")


def filter_entries(reads: pd.Series, min_reads: int) -> pd.Series:
    mask = reads >= min_reads
    return reads[mask]


def build_samples(
    df: pd.DataFrame,
    vocab: Dict[str, Vocabulary],
    abundance_mode: str,
    min_reads: int,
    min_relative_abundance: float,
    min_items: int,
    drop_tail_quantile: float,
    mapper: OutcomeMapper | None = None,
) -> Tuple[List[Sample], Dict[str, int]]:
    samples: List[Sample] = []
    outcomes: Dict[str, int] = {}
    for sample_id, group in df.groupby("Sample_ID"):
        reads = filter_entries(group["Reads"], min_reads)
        if reads.empty:
            continue
        group = group.loc[reads.index]
        depth = float(reads.sum())
        if depth <= 0:
            continue
        rel_abundance = reads / depth
        if min_relative_abundance > 0:
            mask = rel_abundance >= min_relative_abundance
            group = group.loc[mask]
            reads = reads.loc[mask]
            rel_abundance = rel_abundance.loc[mask]
        if drop_tail_quantile > 0 and not reads.empty:
            threshold = reads.quantile(drop_tail_quantile)
            mask = reads >= threshold
            group = group.loc[mask]
            reads = reads.loc[mask]
            rel_abundance = rel_abundance.loc[mask]
        if len(group) < min_items:
            continue
        species_ids = [vocab["species"].lookup(x) for x in group["Species"].tolist()]
        genus_ids = [vocab["genus"].lookup(x) for x in group["Genus"].tolist()]
        type_ids = [vocab["type"].lookup(x) for x in group["Type"].tolist()]
        abundance = compute_abundance(reads.to_numpy(), depth, abundance_mode).tolist()
        sample = Sample(
            sample_id=str(sample_id),
            species_ids=species_ids,
            genus_ids=genus_ids,
            type_ids=type_ids,
            abundance=abundance,
            depth=depth,
        )
        if mapper is not None and "Outcome" in group.columns:
            unique_outcomes = group["Outcome"].dropna().unique()
            if len(unique_outcomes) == 0:
                raise ValueError(f"Missing outcome for sample {sample_id}")
            if len(set(unique_outcomes)) != 1:
                raise ValueError(f"Conflicting outcomes for sample {sample_id}: {unique_outcomes}")
            outcome_value = mapper(unique_outcomes[0])
            sample.outcome = outcome_value
            outcomes[str(sample_id)] = outcome_value
        samples.append(sample)
    return samples, outcomes


def build_baseline_matrix(samples: List[Sample], vocab: Vocabulary) -> Tuple[np.ndarray, List[str]]:
    num_samples = len(samples)
    num_species = len(vocab)
    matrix = np.zeros((num_samples, num_species), dtype=np.float32)
    sample_ids: List[str] = []
    for idx, sample in enumerate(samples):
        sample_ids.append(sample.sample_id)
        for species_id, abundance in zip(sample.species_ids, sample.abundance):
            matrix[idx, species_id] += float(abundance)
    return matrix, sample_ids


def serialize_samples(samples: List[Sample]) -> List[dict]:
    serialised = []
    for sample in samples:
        serialised.append(
            {
                "sample_id": sample.sample_id,
                "species_ids": sample.species_ids,
                "genus_ids": sample.genus_ids,
                "type_ids": sample.type_ids,
                "abundance": sample.abundance,
                "depth": sample.depth,
                "outcome": sample.outcome,
            }
        )
    return serialised


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)

    unlabeled_df = read_excel(args.unlabeled_xlsx)
    labeled_df = read_excel(args.labeled_xlsx)
    if "Outcome" not in labeled_df.columns:
        raise ValueError("labeled.xlsx must contain Outcome column")
    labeled_df = labeled_df.copy()
    mapper = OutcomeMapper()
    labeled_df["Outcome"] = labeled_df["Outcome"].apply(mapper)

    vocab = build_vocabularies([unlabeled_df, labeled_df])
    save_json({k: v.to_json() for k, v in vocab.items()}, args.out_dir / "vocab.json")
    save_json(mapper.to_json(), args.out_dir / "outcome_mapping.json")

    unlabeled_samples, _ = build_samples(
        unlabeled_df,
        vocab,
        args.abundance,
        args.min_reads,
        args.min_relative_abundance,
        args.min_items,
        args.drop_tail_quantile,
    )
    labeled_samples, outcomes = build_samples(
        labeled_df,
        vocab,
        args.abundance,
        args.min_reads,
        args.min_relative_abundance,
        args.min_items,
        args.drop_tail_quantile,
        mapper=mapper,
    )

    with open(args.out_dir / "samples_unlabeled.pkl", "wb") as f:
        pickle.dump(serialize_samples(unlabeled_samples), f)
    with open(args.out_dir / "samples_labeled.pkl", "wb") as f:
        pickle.dump(serialize_samples(labeled_samples), f)

    baseline_matrix, sample_ids = build_baseline_matrix(labeled_samples, vocab["species"])
    np.savez_compressed(
        args.out_dir / "baseline_features_labeled.npz",
        features=baseline_matrix,
        sample_ids=np.array(sample_ids),
        outcomes=np.array([s.outcome for s in labeled_samples]),
    )

    config = {
        "unlabeled_xlsx": str(args.unlabeled_xlsx),
        "labeled_xlsx": str(args.labeled_xlsx),
        "min_reads": args.min_reads,
        "min_relative_abundance": args.min_relative_abundance,
        "abundance": args.abundance,
        "min_items": args.min_items,
        "drop_tail_quantile": args.drop_tail_quantile,
        "num_unlabeled": len(unlabeled_samples),
        "num_labeled": len(labeled_samples),
    }
    (args.out_dir / "config.yaml").write_text(yaml.dump(config), encoding="utf-8")

    print(f"Saved {len(unlabeled_samples)} unlabeled samples and {len(labeled_samples)} labeled samples to {args.out_dir}")


if __name__ == "__main__":
    main()
