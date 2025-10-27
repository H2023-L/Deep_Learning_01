from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

SPECIES = [
    ("Escherichia coli", "Escherichia", "bacteria"),
    ("Staphylococcus aureus", "Staphylococcus", "bacteria"),
    ("Influenza A", "Influenza", "virus"),
    ("Candida albicans", "Candida", "fungi"),
    ("Klebsiella pneumoniae", "Klebsiella", "bacteria"),
    ("Human coronavirus", "Coronavirus", "virus"),
]


def make_sample(sample_id: int, with_outcome: bool) -> list[dict]:
    rows = []
    depth = random.randint(1000, 5000)
    chosen = random.sample(SPECIES, k=random.randint(2, min(4, len(SPECIES))))
    for idx, (species, genus, type_name) in enumerate(chosen, start=1):
        reads = random.randint(50, max(80, depth // (len(chosen) + 1)))
        rows.append(
            {
                "Sample_ID": f"S{sample_id:04d}",
                "Species": species,
                "Genus": genus,
                "Type": type_name,
                "Reads": reads,
                "Outcome": random.choice(["Yes", "No"]) if with_outcome and idx == 1 else None,
            }
        )
    if with_outcome and rows:
        outcome = random.choice([0, 1])
        for row in rows:
            row["Outcome"] = outcome
    return rows


def build_dataframe(num_samples: int, with_outcome: bool) -> pd.DataFrame:
    rows = []
    for sample_id in range(1, num_samples + 1):
        rows.extend(make_sample(sample_id, with_outcome))
    df = pd.DataFrame(rows)
    if not with_outcome:
        df = df.drop(columns=["Outcome"])
    return df


def main() -> None:
    random.seed(42)
    np.random.seed(42)
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)
    unlabeled = build_dataframe(50, with_outcome=False)
    labeled = build_dataframe(12, with_outcome=True)
    unlabeled.to_excel(out_dir / "unlabeled.xlsx", index=False)
    labeled.to_excel(out_dir / "labeled.xlsx", index=False)
    print(f"Toy data written to {out_dir}")


if __name__ == "__main__":
    main()
