import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ctr.data.split import CAT_COLS, COLUMNS, INT_COLS

N_BLOCKS = 7
SHARDS_PER_BLOCK = 2
ROWS_PER_SHARD = 400


def synthetic_shard(rng: np.random.Generator, n: int) -> pl.DataFrame:
    """Block-schema rows whose label depends on C1 and I1, with nulls in every field."""
    c1 = rng.integers(0, 8, n)
    i1 = rng.integers(-2, 200, n)
    logit = -1.5 + 2.0 * (c1 < 3) + 0.01 * i1
    label = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.int8)

    data: dict[str, pl.Series] = {"label": pl.Series(label, dtype=pl.Int8)}
    for j, c in enumerate(INT_COLS):
        values = i1 if c == "I1" else rng.integers(-3, 10 * (j + 1), n)
        data[c] = pl.Series(values, dtype=pl.Int32)
    for j, c in enumerate(CAT_COLS):
        values = c1 if c == "C1" else rng.integers(0, 3 + j % 6, n) * 2_654_435_761 % 2**32
        data[c] = pl.Series(values, dtype=pl.UInt32)
    df = pl.DataFrame(data).select(COLUMNS)
    null_mask = rng.random((n, len(COLUMNS) - 1)) < 0.05
    return df.with_columns(
        pl.when(pl.Series(null_mask[:, j])).then(None).otherwise(pl.col(c)).alias(c)
        for j, c in enumerate(COLUMNS[1:])
    )


def write_blocks(root: Path, shards: dict[int, list[pl.DataFrame]]) -> Path:
    """Write shards in the layout of ctr.data.split (ADR-007), manifest included."""
    blocks, bounds, offset = {}, [], 0
    for k in sorted(shards):
        block_dir = root / f"block_{k}"
        block_dir.mkdir(parents=True)
        for i, df in enumerate(shards[k]):
            df.write_parquet(block_dir / f"part-{i:05d}.parquet")
        rows = sum(df.height for df in shards[k])
        positives = sum(int(df["label"].sum()) for df in shards[k])
        blocks[str(k)] = {"rows": rows, "positives": positives, "shards": len(shards[k])}
        bounds.append([offset, offset + rows])
        offset += rows
    manifest = {
        "source": "synthetic",
        "n_rows": offset,
        "n_blocks": len(shards),
        "bounds": bounds,
        "columns": COLUMNS,
        "blocks": blocks,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return root


@pytest.fixture
def blocks_dir(tmp_path) -> Path:
    rng = np.random.default_rng(0)
    shards = {
        k: [synthetic_shard(rng, ROWS_PER_SHARD) for _ in range(SHARDS_PER_BLOCK)]
        for k in range(1, N_BLOCKS + 1)
    }
    return write_blocks(tmp_path / "blocks", shards)


@pytest.fixture
def make_blocks(tmp_path):
    """Write hand-made shards ({block: [DataFrame, ...]}) and return the blocks dir."""
    return lambda shards: write_blocks(tmp_path / "custom_blocks", shards)


@pytest.fixture
def make_shard():
    """synthetic_shard(rng, n) for tests that need to tweak rows themselves."""
    return synthetic_shard
