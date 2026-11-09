import json
import random
from itertools import pairwise
from pathlib import Path

import polars as pl
import pyarrow as pa
import pytest

from ctr.data.split import (
    CAT_COLS,
    COLUMNS,
    INT_COLS,
    block_bounds,
    count_rows,
    read_block,
    split_tsv_to_parquet,
)

# --- block_bounds ---------------------------------------------------------


def test_block_bounds_cover_all_rows_contiguously():
    bounds = block_bounds(100, 7)

    assert len(bounds) == 7
    assert bounds[0][0] == 0
    assert bounds[-1][1] == 100
    for (_, end), (start, _) in pairwise(bounds):
        assert end == start


def test_block_bounds_sizes_differ_by_at_most_one():
    sizes = [end - start for start, end in block_bounds(45_840_617, 7)]

    assert max(sizes) - min(sizes) <= 1
    assert sum(sizes) == 45_840_617


def test_block_bounds_rejects_fewer_rows_than_blocks():
    with pytest.raises(ValueError):
        block_bounds(6, 7)


# --- count_rows -----------------------------------------------------------


def test_count_rows_with_trailing_newline(tmp_path):
    p = tmp_path / "a.txt"
    p.write_bytes(b"a\nb\nc\n")

    assert count_rows(p) == 3


def test_count_rows_without_trailing_newline(tmp_path):
    p = tmp_path / "a.txt"
    p.write_bytes(b"a\nb\nc")

    assert count_rows(p) == 3


# --- split_tsv_to_parquet -------------------------------------------------


def make_rows(n: int, seed: int = 0) -> list[list[str]]:
    """Synthetic Criteo-shaped rows. I1 holds the row index so order is checkable."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        label = str(rng.randint(0, 1))
        ints = [str(i)] + [
            "" if rng.random() < 0.3 else str(rng.randint(-2, 1000)) for _ in range(12)
        ]
        cats = ["" if rng.random() < 0.3 else f"{rng.getrandbits(32):08x}" for _ in range(26)]
        rows.append([label, *ints, *cats])
    return rows


def write_tsv(path: Path, rows: list[list[str]]) -> Path:
    path.write_text("".join("\t".join(r) + "\n" for r in rows))
    return path


@pytest.fixture
def split_result(tmp_path):
    rows = make_rows(53)
    src = write_tsv(tmp_path / "train.txt", rows)
    out = tmp_path / "blocks"
    manifest = split_tsv_to_parquet(src, out, n_blocks=7, rows_per_shard=3, read_block_size=512)
    return rows, out, manifest


def test_blocks_preserve_original_row_order(split_result):
    rows, out, _ = split_result

    i1 = pl.concat([read_block(out, k) for k in range(1, 8)])["I1"].to_list()

    assert i1 == list(range(len(rows)))


def test_block_row_counts_match_bounds(split_result):
    rows, out, manifest = split_result

    for k, (start, end) in enumerate(block_bounds(len(rows), 7), start=1):
        df = read_block(out, k)
        assert df.height == end - start
        assert df["I1"].to_list() == list(range(start, end))
    assert manifest["n_rows"] == len(rows)


def test_shards_respect_rows_per_shard_and_do_not_cross_blocks(split_result):
    _, out, _ = split_result

    for k in range(1, 8):
        shards = sorted((out / f"block_{k}").glob("part-*.parquet"))
        assert shards
        assert all(pl.read_parquet(s).height <= 3 for s in shards)


def test_schema_and_dtypes(split_result):
    _, out, _ = split_result

    df = read_block(out, 1)

    assert df.columns == COLUMNS
    assert df.schema["label"] == pl.Int8
    assert all(df.schema[c] == pl.Int32 for c in INT_COLS)
    assert all(df.schema[c] == pl.UInt32 for c in CAT_COLS)


def test_values_and_missing_are_preserved(split_result):
    rows, out, _ = split_result

    df = pl.concat([read_block(out, k) for k in range(1, 8)])

    for i, row in enumerate(rows):
        got = df.row(i)
        assert got[0] == int(row[0])
        for j in range(1, 14):
            assert got[j] == (None if row[j] == "" else int(row[j]))
        for j in range(14, 40):
            assert got[j] == (None if row[j] == "" else int(row[j], 16))


def test_manifest_records_bounds_and_label_counts(split_result):
    rows, out, manifest = split_result

    on_disk = json.loads((out / "manifest.json").read_text())

    assert on_disk == manifest
    assert manifest["n_blocks"] == 7
    assert [tuple(b) for b in manifest["bounds"]] == block_bounds(len(rows), 7)
    for k, (start, end) in enumerate(block_bounds(len(rows), 7), start=1):
        info = manifest["blocks"][str(k)]
        assert info["rows"] == end - start
        assert info["positives"] == sum(int(r[0]) for r in rows[start:end])


def test_refuses_to_overwrite_existing_output(split_result, tmp_path):
    _, out, _ = split_result

    with pytest.raises(FileExistsError):
        split_tsv_to_parquet(tmp_path / "train.txt", out, n_blocks=7)


def test_malformed_row_raises(tmp_path):
    rows = make_rows(14)
    rows[5] = rows[5][:-1]  # one column short
    src = write_tsv(tmp_path / "bad.txt", rows)

    with pytest.raises(pa.ArrowInvalid):
        split_tsv_to_parquet(src, tmp_path / "out", n_blocks=7)
