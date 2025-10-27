"""Dataset utilities for mNGS set transformer training."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class Sample:
    sample_id: str
    species_ids: List[int]
    genus_ids: List[int]
    type_ids: List[int]
    abundance: List[float]
    depth: float
    outcome: Optional[int] = None

    def __post_init__(self) -> None:
        length = {len(self.species_ids), len(self.genus_ids), len(self.type_ids), len(self.abundance)}
        if len(length) != 1:
            raise ValueError("All feature lists must have the same length")

    @property
    def num_items(self) -> int:
        return len(self.species_ids)


class SetDataset(Dataset):
    """Dataset backed by a list of :class:`Sample`."""

    def __init__(self, samples: Sequence[Sample]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]


def pad_collate_fn(batch: Sequence[Sample], pad_id: int = 0) -> Dict[str, torch.Tensor]:
    max_len = max(sample.num_items for sample in batch)
    batch_size = len(batch)

    def init_long() -> torch.Tensor:
        return torch.full((batch_size, max_len), pad_id, dtype=torch.long)

    def init_float() -> torch.Tensor:
        return torch.zeros((batch_size, max_len), dtype=torch.float32)

    species = init_long()
    genus = init_long()
    type_ids = init_long()
    abundance = init_float()
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    depth = torch.zeros((batch_size,), dtype=torch.float32)
    outcome = torch.full((batch_size,), -1, dtype=torch.long)
    sample_ids: List[str] = []

    for i, sample in enumerate(batch):
        n = sample.num_items
        species[i, :n] = torch.tensor(sample.species_ids, dtype=torch.long)
        genus[i, :n] = torch.tensor(sample.genus_ids, dtype=torch.long)
        type_ids[i, :n] = torch.tensor(sample.type_ids, dtype=torch.long)
        abundance[i, :n] = torch.tensor(sample.abundance, dtype=torch.float32)
        mask[i, :n] = True
        depth[i] = float(sample.depth)
        if sample.outcome is not None:
            outcome[i] = int(sample.outcome)
        sample_ids.append(sample.sample_id)

    batch_dict: Dict[str, torch.Tensor] = {
        "species": species,
        "genus": genus,
        "type": type_ids,
        "abundance": abundance,
        "mask": mask,
        "depth": depth,
    }
    if (outcome >= 0).any():
        batch_dict["outcome"] = outcome.float()
    batch_dict["sample_id"] = sample_ids
    return batch_dict


def build_dataloader(
    samples: Sequence[Sample],
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pad_id: int = 0,
) -> DataLoader:
    dataset = SetDataset(samples)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: pad_collate_fn(batch, pad_id=pad_id),
        pin_memory=torch.cuda.is_available(),
    )


def apply_gaussian_noise(arr: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return arr
    noise = torch.randn_like(arr) * sigma
    noised = arr + noise
    return torch.clamp(noised, min=0.0)


def random_subsample_mask(length: int, keep_ratio: float) -> torch.Tensor:
    keep = max(1, int(math.ceil(length * keep_ratio)))
    indices = torch.randperm(length)[:keep]
    mask = torch.zeros(length, dtype=torch.bool)
    mask[indices] = True
    return mask


def drop_low_quantile(values: torch.Tensor, quantile: float) -> torch.Tensor:
    if values.numel() == 0:
        return torch.ones_like(values, dtype=torch.bool)
    if quantile <= 0:
        return torch.ones_like(values, dtype=torch.bool)
    cutoff = torch.quantile(values, torch.tensor(quantile, dtype=values.dtype, device=values.device))
    keep_mask = values >= cutoff
    if keep_mask.sum() == 0:
        keep_mask[values.argmax()] = True
    return keep_mask


def augment_sample(
    sample: Sample,
    keep_ratio_range: Sequence[float] = (0.6, 0.9),
    drop_quantile: float = 0.05,
    noise_sigma: float = 0.1,
) -> Sample:
    length = sample.num_items
    if length == 0:
        return sample

    keep_ratio = random.uniform(*keep_ratio_range)
    keep_mask = random_subsample_mask(length, keep_ratio)
    abundance_tensor = torch.tensor(sample.abundance, dtype=torch.float32)
    high_mask = drop_low_quantile(abundance_tensor, drop_quantile)
    combined_mask = keep_mask & high_mask
    if combined_mask.sum() == 0:
        combined_mask = keep_mask
    indices = combined_mask.nonzero(as_tuple=True)[0]
    indices = indices.sort()[0]

    def select(values: Sequence) -> List:
        return [values[i] for i in indices.tolist()]

    abundance = torch.tensor(select(sample.abundance), dtype=torch.float32)
    abundance = apply_gaussian_noise(abundance, noise_sigma)

    return Sample(
        sample_id=sample.sample_id,
        species_ids=select(sample.species_ids),
        genus_ids=select(sample.genus_ids),
        type_ids=select(sample.type_ids),
        abundance=abundance.tolist(),
        depth=sample.depth,
        outcome=sample.outcome,
    )


def mask_abundance(
    abundance: torch.Tensor,
    mask: torch.Tensor,
    mask_prob: float = 0.15,
    uniform: bool = True,
) -> torch.Tensor:
    noise_mask = (torch.rand_like(abundance) < mask_prob) & mask
    if not noise_mask.any():
        return abundance, noise_mask
    corrupted = abundance.clone()
    if uniform:
        corrupted[noise_mask] = corrupted[noise_mask] * torch.rand_like(corrupted[noise_mask])
    else:
        corrupted[noise_mask] = 0.0
    return corrupted, noise_mask


def sample_from_dict(data: Dict) -> Sample:
    return Sample(
        sample_id=str(data["sample_id"]),
        species_ids=list(data["species_ids"]),
        genus_ids=list(data["genus_ids"]),
        type_ids=list(data["type_ids"]),
        abundance=list(data["abundance"]),
        depth=float(data.get("depth", 0.0)),
        outcome=data.get("outcome"),
    )


def truncate_sample(sample: Sample, max_len: int) -> Sample:
    if sample.num_items <= max_len:
        return sample
    return Sample(
        sample_id=sample.sample_id,
        species_ids=sample.species_ids[:max_len],
        genus_ids=sample.genus_ids[:max_len],
        type_ids=sample.type_ids[:max_len],
        abundance=sample.abundance[:max_len],
        depth=sample.depth,
        outcome=sample.outcome,
    )
