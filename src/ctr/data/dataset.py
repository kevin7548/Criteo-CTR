"""Mini-batches of encoded rows, read shard by shard so a split never has to fit in memory.

Evaluation reads shards in order and yields rows in their original order.
Training shuffles at two levels (ADR-010): the shard order is permuted, then
`shards_per_chunk` shards at a time are loaded together and their rows permuted,
so a batch mixes rows from several points in time.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import polars as pl

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


def iter_batches(
    files: Sequence[Path],
    batch_size: int,
    *,
    shuffle: bool = False,
    seed: int = 0,
    shards_per_chunk: int = 4,
) -> Iterator[Batch]:
    """Yield (ids, labels) batches of at most `batch_size` rows over `files`."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if shards_per_chunk < 1:
        raise ValueError(f"shards_per_chunk must be >= 1, got {shards_per_chunk}")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(files)) if shuffle else np.arange(len(files))
    chunk = shards_per_chunk if shuffle else 1
    for start in range(0, len(order), chunk):
        ids, labels = load_arrays([files[i] for i in order[start : start + chunk]])
        if shuffle:
            perm = rng.permutation(len(labels))
            ids, labels = ids[perm], labels[perm]
        for b in range(0, len(labels), batch_size):
            yield ids[b : b + batch_size], labels[b : b + batch_size]
