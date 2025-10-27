"""Set Transformer encoder tailored for mNGS inputs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn


@dataclass
class EncoderOutput:
    token_embeddings: torch.Tensor
    set_embedding: torch.Tensor


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        attn_output, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = residual + self.dropout(attn_output)
        x = self.norm(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.net(x)
        return self.norm(residual + x)


class SAB(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.attn(x, key_padding_mask=key_padding_mask)
        x = self.ffn(x)
        return x


class PMA(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_seeds: int = 1, dropout: float = 0.0):
        super().__init__()
        self.seed_vectors = nn.Parameter(torch.randn(num_seeds, embed_dim))
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x.size(0)
        seeds = self.seed_vectors.unsqueeze(0).expand(batch_size, -1, -1)
        out, _ = self.attn(seeds, x, x, key_padding_mask=key_padding_mask)
        out = self.norm(seeds + self.dropout(out))
        return out


class SetTransformerEncoder(nn.Module):
    def __init__(
        self,
        species_vocab_size: int,
        genus_vocab_size: int,
        type_vocab_size: int,
        embedding_dim: int,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        ffn_dim: int,
        dropout: float = 0.1,
        depth_strategy: str = "mlp",
    ) -> None:
        super().__init__()
        self.species_embed = nn.Embedding(species_vocab_size, embedding_dim)
        self.genus_embed = nn.Embedding(genus_vocab_size, embedding_dim)
        self.type_embed = nn.Embedding(type_vocab_size, embedding_dim)
        token_dim = embedding_dim * 3 + 1
        self.input_projection = nn.Sequential(
            nn.Linear(token_dim, model_dim),
            nn.GELU(),
        )
        self.layers = nn.ModuleList([
            SAB(model_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)
        ])
        self.pma = PMA(model_dim, num_heads, num_seeds=1, dropout=dropout)
        self.depth_strategy = depth_strategy
        if depth_strategy == "mlp":
            self.depth_proj = nn.Sequential(
                nn.Linear(1, model_dim),
                nn.GELU(),
                nn.Linear(model_dim, model_dim),
            )
            self.combiner = nn.Linear(model_dim * 2, model_dim)
        elif depth_strategy == "add":
            self.depth_proj = nn.Sequential(
                nn.Linear(1, model_dim),
                nn.Sigmoid(),
            )
            self.combiner = None
        elif depth_strategy == "none":
            self.depth_proj = None
            self.combiner = None
        else:
            raise ValueError(f"Unknown depth strategy: {depth_strategy}")
        self.norm = nn.LayerNorm(model_dim)

    def forward(
        self,
        species: torch.Tensor,
        genus: torch.Tensor,
        type_ids: torch.Tensor,
        abundance: torch.Tensor,
        mask: torch.Tensor,
        depth: Optional[torch.Tensor] = None,
    ) -> EncoderOutput:
        species_emb = self.species_embed(species)
        genus_emb = self.genus_embed(genus)
        type_emb = self.type_embed(type_ids)
        abundance = abundance.unsqueeze(-1)
        token_repr = torch.cat([species_emb, genus_emb, type_emb, abundance], dim=-1)
        token_repr = self.input_projection(token_repr)
        key_padding_mask = ~mask
        for layer in self.layers:
            token_repr = layer(token_repr, key_padding_mask=key_padding_mask)
        pooled = self.pma(token_repr, key_padding_mask=key_padding_mask).squeeze(1)
        if self.depth_proj is not None and depth is not None:
            depth_feature = depth.unsqueeze(-1)
            depth_emb = self.depth_proj(depth_feature)
            if self.combiner is None:
                pooled = pooled + depth_emb
            else:
                pooled = torch.cat([pooled, depth_emb], dim=-1)
                pooled = self.combiner(pooled)
        pooled = self.norm(pooled)
        return EncoderOutput(token_embeddings=token_repr, set_embedding=pooled)

    def load_encoder_state(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.load_state_dict(state_dict)
