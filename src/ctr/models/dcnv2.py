"""DCN-V2 (Wang et al., 2021), parallel form, on the same field ids as LR (ADR-006, ADR-012).

Each of the 39 field ids gets an `embed_dim` vector; their concatenation x0 feeds
two towers side by side:
- cross network: x_{l+1} = x0 ⊙ (W_l x_l + b_l) + x_l with full-rank W_l, which
  builds explicit feature interactions up to degree `cross_layers + 1`;
- deep network: an MLP over x0 (ReLU, optional dropout).
The last cross output and the last MLP output are concatenated into one linear
logit. That output layer starts at zero weight and bias logit(base_rate), so the
untrained model is the constant-CTR baseline, like LR.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn


class CrossLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        return x0 * self.linear(xl) + xl


class DCNv2(nn.Module):
    def __init__(
        self,
        field_sizes: Sequence[int],
        base_rate: float | None = None,
        embed_dim: int = 16,
        cross_layers: int = 3,
        hidden: Sequence[int] = (400, 400, 400),
        dropout: float = 0.0,
        embed_init_std: float = 1e-4,
    ):
        super().__init__()
        if not field_sizes or min(field_sizes) < 1:
            raise ValueError(f"field_sizes must be non-empty and positive, got {field_sizes}")
        if base_rate is not None and not 0 < base_rate < 1:
            raise ValueError(f"base_rate must be in (0, 1), got {base_rate}")
        if embed_dim < 1 or cross_layers < 0 or any(h < 1 for h in hidden):
            raise ValueError(
                f"bad sizes: embed_dim={embed_dim}, cross_layers={cross_layers}, hidden={hidden}"
            )
        if not 0 <= dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        offsets = torch.tensor([0, *field_sizes[:-1]]).cumsum(0)
        self.register_buffer("offsets", offsets, persistent=False)
        self.embedding = nn.Embedding(sum(field_sizes), embed_dim)
        nn.init.normal_(self.embedding.weight, std=embed_init_std)

        dim = len(field_sizes) * embed_dim
        self.cross = nn.ModuleList(CrossLayer(dim) for _ in range(cross_layers))
        layers: list[nn.Module] = []
        width = dim
        for h in hidden:
            linear = nn.Linear(width, h)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers += [linear, nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            width = h
        self.deep = nn.Sequential(*layers)
        self.has_deep = bool(hidden)

        self.out = nn.Linear(dim + (width if hidden else 0), 1)
        nn.init.zeros_(self.out.weight)
        bias = 0.0 if base_rate is None else math.log(base_rate / (1 - base_rate))
        nn.init.constant_(self.out.bias, bias)

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        """ids [batch, n_fields] -> x0 [batch, n_fields * embed_dim]."""
        return self.embedding(ids + self.offsets).flatten(1)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x0 = self.embed(ids)
        xl = x0
        for layer in self.cross:
            xl = layer(x0, xl)
        parts = [xl, self.deep(x0)] if self.has_deep else [xl]
        return self.out(torch.cat(parts, dim=1)).squeeze(-1)
