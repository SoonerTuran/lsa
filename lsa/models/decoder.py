"""DeepSDF decoder: MLP on [z, x] with a skip connection re-injecting the input."""
from __future__ import annotations

from typing import Callable, Sequence

import torch
from torch import Tensor, nn
from torch.nn.utils.parametrizations import weight_norm as _weight_norm


class Decoder(nn.Module):
    """f(z, x) -> sdf. ``n_layers`` hidden ReLU layers of width ``hidden_dim`` (weight
    normalised) and a linear output layer. Hidden layer i in ``skip_layers`` receives
    concat([h, z, x]); the layer before it outputs ``hidden_dim - (latent_dim+3)`` features
    so the concatenation is ``hidden_dim`` wide (as in DeepSDF)."""

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 512,
        n_layers: int = 8,
        skip_layers: Sequence[int] = (4,),
        weight_norm: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.skip_layers = frozenset(skip_layers)
        in_dim = latent_dim + 3
        layers = []
        d_in = in_dim
        for i in range(n_layers):
            skip_next = (i + 1) in self.skip_layers
            d_out = hidden_dim - in_dim if skip_next and hidden_dim > in_dim else hidden_dim
            lin = nn.Linear(d_in, d_out)
            layers.append(_weight_norm(lin) if weight_norm else lin)
            d_in = d_out + (in_dim if skip_next else 0)
        self.hidden = nn.ModuleList(layers)
        self.out = nn.Linear(d_in, 1)

    def forward(self, z: Tensor, x: Tensor) -> Tensor:
        """z (B,d) or (B,N,d), x (B,N,3) -> sdf (B,N)."""
        if z.dim() == 2:
            z = z[:, None, :].expand(-1, x.shape[1], -1)
        inp = torch.cat([z, x], dim=-1)
        h = inp
        for i, lin in enumerate(self.hidden):
            if i in self.skip_layers:
                h = torch.cat([h, inp], dim=-1)
            h = torch.relu(lin(h))
        return self.out(h)[..., 0]

    def sdf_fn(self, z: Tensor) -> Callable[[Tensor], Tensor]:
        """SDF callable (N,3) -> (N,) of a single latent z (d,), for viz / metrics."""
        z = z.detach().reshape(1, -1)

        def f(x: Tensor) -> Tensor:
            return self(z.to(x.device), x[None])[0]

        return f


def build_decoder(model_cfg: dict) -> Decoder:
    return Decoder(
        latent_dim=model_cfg.get("latent_dim", 128),
        hidden_dim=model_cfg.get("hidden_dim", 512),
        n_layers=model_cfg.get("n_layers", 8),
        skip_layers=tuple(model_cfg.get("skip_layers", (4,))),
        weight_norm=model_cfg.get("weight_norm", True),
    )
