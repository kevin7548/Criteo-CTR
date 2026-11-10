"""Evaluation metrics for CTR predictions, implemented directly (ADR-003).

Every function takes 1-D arrays `y` (labels in {0, 1}) and `p` (predicted click
probabilities in [0, 1]) and computes in float64, whatever dtype the model produced.

- LogLoss / NE: probability quality. NE divides by the LogLoss of a constant
  `base_rate` prediction, which callers set to the training-block CTR (ADR-008).
- AUC: ranking quality. Depends only on the order of `p`, so any strictly
  increasing transform (such as prior correction) leaves it unchanged.
- ECE: calibration error over 20 equal-count bins by default (ADR-004, ADR-008).
- PCOC: mean prediction / observed CTR. 1 means calibrated on average.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

EPS = 1e-15
DEFAULT_N_BINS = 20

Strategy = Literal["quantile", "uniform"]


def _validate(y: ArrayLike, p: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y)
    p = np.asarray(p, dtype=np.float64)
    if y.ndim != 1 or p.ndim != 1:
        raise ValueError(f"y and p must be 1-D, got shapes {y.shape} and {p.shape}")
    if len(y) != len(p):
        raise ValueError(f"length mismatch: len(y)={len(y)}, len(p)={len(p)}")
    if len(y) == 0:
        raise ValueError("y and p are empty")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("y must contain only 0 and 1")
    if not np.all(np.isfinite(p)) or np.any(p < 0) or np.any(p > 1):
        raise ValueError("p must be finite and within [0, 1]")
    return y.astype(np.int64), p


def logloss(y: ArrayLike, p: ArrayLike, eps: float = EPS) -> float:
    """Mean binary cross-entropy, with `p` clipped to [eps, 1 - eps]."""
    y, p = _validate(y, p)
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(np.where(y == 1, np.log(p), np.log1p(-p))))


def ne(y: ArrayLike, p: ArrayLike, base_rate: float) -> float:
    """Normalized entropy: LogLoss of `p` over LogLoss of predicting `base_rate` everywhere.

    `base_rate` has no default on purpose: it should be the training-block CTR,
    never the CTR of the split being evaluated (ADR-008).
    """
    if not 0 < base_rate < 1:
        raise ValueError(f"base_rate must be in (0, 1), got {base_rate}")
    y, p = _validate(y, p)
    return logloss(y, p) / logloss(y, np.full(len(y), base_rate))


def auc(y: ArrayLike, p: ArrayLike) -> float:
    """ROC AUC as the Mann-Whitney U statistic, with tied predictions given their mean rank.

    Ranks are kept doubled so that mean ranks of tie groups stay integers and the
    rank sum is exact in int64.
    """
    y, p = _validate(y, p)
    n = len(y)
    n_pos = int(y.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC needs both positive and negative labels")

    order = np.argsort(p, kind="stable")
    sorted_p, sorted_y = p[order], y[order]
    starts = np.concatenate(([0], np.flatnonzero(np.diff(sorted_p)) + 1))
    ends = np.append(starts[1:], n)
    # 1-based ranks start+1 .. end share the mean (start + 1 + end) / 2.
    doubled_mean_rank = starts + ends + 1
    positives_per_group = np.add.reduceat(sorted_y, starts)

    doubled_rank_sum = int(np.dot(positives_per_group, doubled_mean_rank))
    doubled_u = doubled_rank_sum - n_pos * (n_pos + 1)
    return doubled_u / (2 * n_pos * n_neg)


def pcoc(y: ArrayLike, p: ArrayLike) -> float:
    """Predicted CTR over observed CTR."""
    y, p = _validate(y, p)
    positives = int(y.sum())
    if positives == 0:
        raise ValueError("PCOC is undefined without positive labels")
    return float(p.mean() / (positives / len(y)))


@dataclass(frozen=True)
class CalibrationBins:
    """Per-bin row count, mean prediction and mean label, ordered by prediction."""

    count: np.ndarray
    mean_pred: np.ndarray
    mean_label: np.ndarray


def _bin_sums(
    index: np.ndarray, y: np.ndarray, p: np.ndarray, n_bins: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin count, prediction sum and label sum, with empty bins dropped."""
    count = np.bincount(index, minlength=n_bins)
    pred_sum = np.bincount(index, weights=p, minlength=n_bins)
    label_sum = np.bincount(index, weights=y, minlength=n_bins)
    keep = count > 0
    return count[keep], pred_sum[keep], label_sum[keep]


def calibration_bins(
    y: ArrayLike,
    p: ArrayLike,
    n_bins: int = DEFAULT_N_BINS,
    strategy: Strategy = "quantile",
) -> CalibrationBins:
    """Group rows into calibration bins.

    "quantile" (default): sort by `p` and cut into `n_bins` contiguous groups whose
    sizes differ by at most one (the sizes of `np.array_split`). Rows with equal `p`
    are never split: each tie group goes whole into the bin of its middle row, so with
    ties some bins grow and others may vanish (ADR-008).

    "uniform": bins [k/n, (k+1)/n) on [0, 1], with 1.0 in the last bin. Empty
    bins are dropped.
    """
    y, p = _validate(y, p)
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    if strategy == "quantile":
        n = len(y)
        if n < n_bins:
            raise ValueError(f"quantile bins need at least n_bins={n_bins} rows, got {n}")
        order = np.argsort(p, kind="stable")
        p, y = p[order], y[order]
        size, extra = divmod(n, n_bins)
        sizes = np.full(n_bins, size)
        sizes[:extra] += 1
        bin_of_position = np.repeat(np.arange(n_bins), sizes)
        # A tie group goes whole into the bin of its middle row, so the result never
        # depends on how tied rows happen to be ordered.
        group_start = np.concatenate(([0], np.flatnonzero(np.diff(p)) + 1))
        group_end = np.append(group_start[1:], n)
        group_middle = (group_start + group_end - 1) // 2
        index = np.repeat(bin_of_position[group_middle], group_end - group_start)
        count, pred_sum, label_sum = _bin_sums(index, y, p, n_bins)
    elif strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        index = np.searchsorted(edges[1:-1], p, side="right")
        count, pred_sum, label_sum = _bin_sums(index, y, p, n_bins)
    else:
        raise ValueError(f"unknown strategy {strategy!r}; use 'quantile' or 'uniform'")

    return CalibrationBins(
        count=count.astype(np.int64),
        mean_pred=pred_sum / count,
        mean_label=label_sum / count,
    )


def ece(
    y: ArrayLike,
    p: ArrayLike,
    n_bins: int = DEFAULT_N_BINS,
    strategy: Strategy = "quantile",
) -> float:
    """Expected calibration error: count-weighted mean of |mean prediction - mean label|."""
    bins = calibration_bins(y, p, n_bins=n_bins, strategy=strategy)
    weights = bins.count / bins.count.sum()
    return float(np.sum(weights * np.abs(bins.mean_pred - bins.mean_label)))


def summary(y: ArrayLike, p: ArrayLike, base_rate: float) -> dict:
    """All headline metrics for one split. AUC alone is never the comparison (AGENTS.md)."""
    y, p = _validate(y, p)
    return {
        "n": len(y),
        "positives": int(y.sum()),
        "ctr": float(y.mean()),
        "mean_pred": float(p.mean()),
        "logloss": logloss(y, p),
        "ne": ne(y, p, base_rate),
        "auc": auc(y, p),
        "ece": ece(y, p),
        "pcoc": pcoc(y, p),
    }


__all__ = [
    "DEFAULT_N_BINS",
    "EPS",
    "CalibrationBins",
    "auc",
    "calibration_bins",
    "ece",
    "logloss",
    "ne",
    "pcoc",
    "summary",
]
