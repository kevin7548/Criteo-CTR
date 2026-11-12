"""Logistic regression as a sum of one-dimensional embeddings plus a bias (ADR-006).

Every field id has one weight; the logit of a row is the bias plus the weights of
its 39 ids. With weights at zero and the bias at logit(base_rate), the untrained
model is the constant-CTR baseline.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn


class LogisticRegression(nn.Module):
    def __init__(self, field_sizes: Sequence[int], base_rate: float | None = None):
        super().__init__()
        if not field_sizes or min(field_sizes) < 1:
            raise ValueError(f"field_sizes must be non-empty and positive, got {field_sizes}")
        if base_rate is not None and not 0 < base_rate < 1:
            raise ValueError(f"base_rate must be in (0, 1), got {base_rate}")

        offsets = torch.tensor([0, *field_sizes[:-1]]).cumsum(0)
        self.register_buffer("offsets", offsets, persistent=False)
        self.weights = nn.Embedding(sum(field_sizes), 1)
        nn.init.zeros_(self.weights.weight)
        bias = 0.0 if base_rate is None else math.log(base_rate / (1 - base_rate))
        self.bias = nn.Parameter(torch.tensor(bias))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: [batch, n_fields] local ids per field -> logits [batch]."""
        return self.weights(ids + self.offsets).squeeze(-1).sum(dim=1) + self.bias
