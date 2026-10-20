"""Per-shape latent codes for the auto-decoder."""
from __future__ import annotations

from torch import Tensor, nn


class LatentTable(nn.Module):
    """nn.Embedding of K latents of size d, initialised N(0, init_std^2)."""

    def __init__(self, n_shapes: int, latent_dim: int, init_std: float = 0.01):
        super().__init__()
        self.emb = nn.Embedding(n_shapes, latent_dim)
        nn.init.normal_(self.emb.weight, mean=0.0, std=init_std)

    @property
    def weight(self) -> Tensor:
        return self.emb.weight

    def forward(self, idx: Tensor) -> Tensor:
        return self.emb(idx)
