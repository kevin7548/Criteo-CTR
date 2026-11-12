"""Predict a split in its original row order and score it with ctr.metrics.

NE uses the training-block CTR as its constant baseline (ADR-008), so every report
carries the same denominator as reports/baseline_constant.json.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ctr.baseline import TRAIN_BLOCKS
from ctr.data.dataset import iter_batches
from ctr.metrics import calibration_bins, summary


def resolve_device(device: str) -> str:
    """'auto' picks MPS on Apple Silicon, then CUDA, then CPU."""
    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def train_ctr(manifest: dict, blocks: Sequence[int] = TRAIN_BLOCKS) -> float:
    """Pooled CTR of the training blocks, from manifest counts."""
    infos = [manifest["blocks"][str(k)] for k in blocks]
    return sum(i["positives"] for i in infos) / sum(i["rows"] for i in infos)


@torch.no_grad()
def predict(
    model: nn.Module, files: Sequence[Path], batch_size: int, device: str
) -> tuple[np.ndarray, np.ndarray]:
    """Labels (int8) and click probabilities (float64) for `files`, in row order."""
    was_training = model.training
    model.eval()
    labels, probs = [], []
    try:
        for ids, y in iter_batches(files, batch_size):
            logits = model(torch.from_numpy(ids).to(device).long())
            probs.append(torch.sigmoid(logits.float()).cpu().numpy())
            labels.append(y)
    finally:
        model.train(was_training)
    return np.concatenate(labels), np.concatenate(probs).astype(np.float64)


def evaluate_split(y: np.ndarray, p: np.ndarray, base_rate: float) -> dict:
    """Headline metrics plus the 20 equal-count calibration bins behind ECE."""
    bins = calibration_bins(y, p)
    return {
        **summary(y, p, base_rate),
        "calibration_bins": {
            "count": bins.count.tolist(),
            "mean_pred": bins.mean_pred.tolist(),
            "mean_label": bins.mean_label.tolist(),
        },
    }
