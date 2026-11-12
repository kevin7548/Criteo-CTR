"""Mini-batches of encoded rows, read shard by shard so a split never has to fit in memory.

Evaluation reads shards in order and yields rows in their original order.
Training shuffles at two levels (ADR-010): the shard order is permuted, then
`shards_per_chunk` shards at a time are loaded together and their rows permuted,
so a batch mixes rows from several points in time.

Negative downsampling (ADR-011) is decided per shard before any shuffling: each
shard's mask comes from an RNG seeded by (sample_seed, the shard's position in
`files`), so the same rows are kept in every epoch, whatever the shuffle seed or
chunking, and a run on the first k shards keeps exactly what the full run keeps there.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import polars as pl

from ctr.calibration import negative_keep_mask
from ctr.data.split import block_files
from ctr.data.vocab import FIELDS

Batch = tuple[np.ndarray, np.ndarray]  # ids int32 [n, n_fields], labels int8 [n]


def split_files(
    encoded_dir: Path | str, blocks: Sequence[int], max_shards: int | None = None
) -> list[Path]:
    """Shards of `blocks` in chronological order, optionally only the first `max_shards`."""
    if max_shards is not None and max_shards < 1:
        raise ValueError(f"max_shards must be >= 1, got {max_shards}")
    files = [f for k in blocks for f in block_files(encoded_dir, k)]
    return files[:max_shards]


def load_arrays(files: Sequence[Path]) -> Batch:
    df = pl.read_parquet(list(files), columns=["label", *FIELDS])
    ids = np.ascontiguousarray(df.select(FIELDS).to_numpy(), dtype=np.int32)
    labels = df["label"].to_numpy().astype(np.int8, copy=False)
    return ids, labels


def _load_sampled(
    files: Sequence[Path], indices: Sequence[int], keep_rate: float, sample_seed: int
) -> Batch:
    """Rows of files[i] for i in `indices`, negatives kept at `keep_rate` per shard."""
    parts = []
    for i in indices:
        ids, labels = load_arrays([files[i]])
        keep = negative_keep_mask(labels, keep_rate, np.random.default_rng([sample_seed, i]))
        parts.append((ids[keep], labels[keep]))
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def iter_batches(
    files: Sequence[Path],
    batch_size: int,
    *,
    shuffle: bool = False,
    seed: int = 0,
    shards_per_chunk: int = 4,
    negative_keep_rate: float = 1.0,
    sample_seed: int = 0,
) -> Iterator[Batch]:
    """Yield (ids, labels) batches of at most `batch_size` rows over `files`.

    `negative_keep_rate < 1` is for training only; evaluation always reads every row.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if shards_per_chunk < 1:
        raise ValueError(f"shards_per_chunk must be >= 1, got {shards_per_chunk}")
    if not 0 < negative_keep_rate <= 1:
        raise ValueError(f"negative_keep_rate must be in (0, 1], got {negative_keep_rate}")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(files)) if shuffle else np.arange(len(files))
    chunk = shards_per_chunk if shuffle else 1
    for start in range(0, len(order), chunk):
        indices = order[start : start + chunk]
        if negative_keep_rate < 1:
            ids, labels = _load_sampled(files, indices, negative_keep_rate, sample_seed)
        else:
            ids, labels = load_arrays([files[i] for i in indices])
        if shuffle:
            perm = rng.permutation(len(labels))
            ids, labels = ids[perm], labels[perm]
        for b in range(0, len(labels), batch_size):
            yield ids[b : b + batch_size], labels[b : b + batch_size]
