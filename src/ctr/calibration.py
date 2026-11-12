"""Negative downsampling and its closed-form prior correction (ADR-005, ADR-011).

Keeping every positive and each negative with probability w turns a training CTR c
into c / (c + (1 − c)·w). A model fit on that sample predicts p on the sampled
scale; the correction q = p·w / (p·w + 1 − p), i.e. logit(q) = logit(p) + log w,
maps it back to the distribution the sample was drawn from. Here that is Criteo's
already-sampled distribution, not a production CTR (AGENTS.md).
"""

from __future__ import annotations

import numpy as np


def _check_rate(w: float) -> None:
    if not 0 < w <= 1:
        raise ValueError(f"negative keep rate must be in (0, 1], got {w}")


def negative_keep_mask(labels: np.ndarray, keep_rate: float, rng: np.random.Generator) -> np.ndarray:
    """Boolean mask that keeps every positive and each negative with prob. `keep_rate`."""
    _check_rate(keep_rate)
    labels = np.asarray(labels)
    if keep_rate == 1:
        return np.ones(labels.shape, dtype=bool)
    return (labels == 1) | (rng.random(labels.shape) < keep_rate)


def sampled_ctr(ctr: float, keep_rate: float) -> float:
    """Expected CTR after keeping negatives at `keep_rate`."""
    _check_rate(keep_rate)
    return ctr / (ctr + (1 - ctr) * keep_rate)


def prior_correct(p: np.ndarray, keep_rate: float) -> np.ndarray:
    """Map predictions of a model trained on the downsampled data back to the original
    scale: logit(q) = logit(p) + log(keep_rate). Monotone, so AUC is unchanged."""
    _check_rate(keep_rate)
    p = np.asarray(p, dtype=np.float64)
    pw = p * keep_rate
    return pw / (pw + (1 - p))
