"""CTR models sharing one interface: local field ids [batch, n_fields] -> logits [batch]."""

from __future__ import annotations

from collections.abc import Sequence

from torch import nn

from ctr.models.lr import LogisticRegression

MODELS = {"lr": LogisticRegression}


def build_model(
    name: str, field_sizes: Sequence[int], base_rate: float | None = None, **kwargs
) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    return MODELS[name](field_sizes, base_rate=base_rate, **kwargs)


__all__ = ["MODELS", "LogisticRegression", "build_model"]
