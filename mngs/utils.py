"""Utility functions for the mNGS set transformer project."""
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def ensure_dir(path: os.PathLike | str) -> Path:
    """Create directory if it does not exist and return the path as :class:`Path`."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: dict, path: os.PathLike | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def load_json(path: os.PathLike | str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class Vocabulary:
    token_to_id: Dict[str, int]
    id_to_token: List[str]

    @classmethod
    def build(cls, tokens: Iterable[str]) -> "Vocabulary":
        sorted_tokens = sorted(set(tokens))
        id_to_token = [PAD_TOKEN, UNK_TOKEN]
        token_to_id = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        for token in sorted_tokens:
            if token in token_to_id:
                continue
            token_to_id[token] = len(id_to_token)
            id_to_token.append(token)
        return cls(token_to_id=token_to_id, id_to_token=id_to_token)

    def to_json(self) -> dict:
        return {"token_to_id": self.token_to_id, "id_to_token": self.id_to_token}

    @classmethod
    def from_json(cls, data: dict) -> "Vocabulary":
        return cls(token_to_id=data["token_to_id"], id_to_token=data["id_to_token"])

    def lookup(self, token: str) -> int:
        return self.token_to_id.get(token, self.token_to_id[UNK_TOKEN])

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.id_to_token)


class OutcomeMapper:
    """Robust mapper for various binary outcome encodings to {0, 1}."""

    NEGATIVE_ALIASES = {"0", "no", "false", "neg", "negative", "alive", "absent"}
    POSITIVE_ALIASES = {"1", "yes", "true", "pos", "positive", "dead", "present"}

    def __init__(self) -> None:
        self.mapping: Dict[str, int] = {}

    def __call__(self, value) -> int:
        if isinstance(value, (int, np.integer)):
            if value not in (0, 1):
                raise ValueError(f"Outcome integer must be 0 or 1, got {value!r}")
            return int(value)
        if isinstance(value, float):
            if value in (0.0, 1.0):
                return int(value)
            raise ValueError(f"Outcome float must be 0.0 or 1.0, got {value!r}")
        if value is None or (isinstance(value, float) and math.isnan(value)):
            raise ValueError("Outcome is missing")

        key = str(value).strip().lower()
        if key in self.mapping:
            return self.mapping[key]
        if key in self.NEGATIVE_ALIASES:
            self.mapping[key] = 0
            return 0
        if key in self.POSITIVE_ALIASES:
            self.mapping[key] = 1
            return 1
        raise ValueError(f"Unrecognised outcome label: {value!r}")

    def to_json(self) -> dict:
        return {"known_mapping": self.mapping}


def normalise_text(value: str | float | int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, int)):
        value = str(value)
    return value.strip().lower()


def summarise_metrics(metrics: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(metrics, dtype=np.float64)
    mean = float(arr.mean())
    if len(arr) <= 1:
        return mean, float("nan")
    stderr = arr.std(ddof=1) / math.sqrt(len(arr))
    ci95 = 1.96 * stderr
    return mean, ci95


def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def chunk_list(items: Sequence, chunk_size: int) -> List[Sequence]:
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def compute_pos_weight(labels: Sequence[int]) -> Optional[float]:
    labels = np.asarray(labels)
    positives = labels.sum()
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    return float(negatives / positives)


def write_csv(path: os.PathLike | str, df: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_vocab(path: os.PathLike | str) -> Dict[str, Vocabulary]:
    data = load_json(path)
    return {key: Vocabulary.from_json(value) for key, value in data.items()}
