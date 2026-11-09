"""Criteo train.txt (TSV) -> chronological blocks of sharded Parquet.

The file is streamed once; rows are never shuffled. Block k (1-based) holds rows
[bounds[k-1][0], bounds[k-1][1]) of the original file. See ADR-001 and ADR-007.

Output layout:
    out_dir/
      manifest.json
      block_1/part-00000.parquet, part-00001.parquet, ...
      ...
      block_7/...
"""

from __future__ import annotations

import argparse
import json
import shutil
from itertools import pairwise
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.csv as pacsv

INT_COLS = [f"I{i}" for i in range(1, 14)]
CAT_COLS = [f"C{i}" for i in range(1, 27)]
COLUMNS = ["label", *INT_COLS, *CAT_COLS]


def block_bounds(n_rows: int, n_blocks: int = 7) -> list[tuple[int, int]]:
    """Contiguous [start, end) row ranges whose sizes differ by at most one."""
    if n_rows < n_blocks:
        raise ValueError(f"n_rows={n_rows} is smaller than n_blocks={n_blocks}")
    cuts = [k * n_rows // n_blocks for k in range(n_blocks + 1)]
    return list(pairwise(cuts))


def count_rows(path: Path, chunk_size: int = 1 << 24) -> int:
    """Number of lines, counting a final line without a trailing newline."""
    n = 0
    last = b"\n"
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            n += chunk.count(b"\n")
            last = chunk[-1:]
    return n + (last != b"\n")


def _open_reader(src: Path, read_block_size: int) -> pacsv.CSVStreamingReader:
    return pacsv.open_csv(
        src,
        read_options=pacsv.ReadOptions(column_names=COLUMNS, block_size=read_block_size),
        parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
        convert_options=pacsv.ConvertOptions(
            column_types={
                "label": pa.int8(),
                **{c: pa.int64() for c in INT_COLS},
                **{c: pa.string() for c in CAT_COLS},
            },
            null_values=[""],
            strings_can_be_null=True,
        ),
    )


def _convert(batch: pa.RecordBatch) -> pl.DataFrame:
    """Integers -> Int32, 32-bit hex category hashes -> UInt32. Strict: overflow raises."""
    df = pl.from_arrow(pa.Table.from_batches([batch]))
    return df.with_columns(
        *[pl.col(c).cast(pl.Int32, strict=True) for c in INT_COLS],
        *[pl.col(c).str.to_integer(base=16, strict=True).cast(pl.UInt32) for c in CAT_COLS],
    )


def split_tsv_to_parquet(
    src: Path | str,
    out_dir: Path | str,
    n_blocks: int = 7,
    rows_per_shard: int = 1_000_000,
    read_block_size: int = 64 << 20,
    n_rows: int | None = None,
) -> dict:
    """Stream `src` into `n_blocks` chronological blocks of Parquet shards.

    Writes into `<out_dir>.tmp` and renames on success, so a crashed run never
    leaves a half-written `out_dir`.
    """
    src, out_dir = Path(src), Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} already exists; remove it to re-split")
    tmp_dir = out_dir.with_name(out_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    if n_rows is None:
        n_rows = count_rows(src)
    bounds = block_bounds(n_rows, n_blocks)
    blocks = {str(k): {"rows": 0, "positives": 0, "shards": 0} for k in range(1, n_blocks + 1)}

    block = 1
    offset = 0  # global row index of the next unread row
    pending: list[pl.DataFrame] = []
    pending_rows = 0

    def flush() -> None:
        nonlocal pending, pending_rows
        if not pending_rows:
            return
        df = pl.concat(pending)
        info = blocks[str(block)]
        block_dir = tmp_dir / f"block_{block}"
        block_dir.mkdir(parents=True, exist_ok=True)
        df.write_parquet(block_dir / f"part-{info['shards']:05d}.parquet", compression="zstd")
        info["rows"] += df.height
        info["positives"] += int(df["label"].sum())
        info["shards"] += 1
        pending, pending_rows = [], 0

    for batch in _open_reader(src, read_block_size):
        df = _convert(batch)
        while df.height:
            if block > n_blocks:
                raise ValueError(f"{src} has more than the {n_rows} rows counted")
            take = min(df.height, bounds[block - 1][1] - offset, rows_per_shard - pending_rows)
            pending.append(df.slice(0, take))
            pending_rows += take
            offset += take
            df = df.slice(take)
            if pending_rows == rows_per_shard:
                flush()
            if offset == bounds[block - 1][1]:
                flush()
                block += 1

    if offset != n_rows:
        raise ValueError(f"read {offset} rows but expected {n_rows}")

    manifest = {
        "source": src.name,
        "n_rows": n_rows,
        "n_blocks": n_blocks,
        "bounds": [list(b) for b in bounds],
        "columns": COLUMNS,
        "blocks": blocks,
    }
    (tmp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    tmp_dir.rename(out_dir)
    return manifest


def block_files(out_dir: Path | str, k: int) -> list[Path]:
    files = sorted((Path(out_dir) / f"block_{k}").glob("part-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no shards for block {k} under {out_dir}")
    return files


def scan_block(out_dir: Path | str, k: int) -> pl.LazyFrame:
    """Lazy view of block k with shards in original row order."""
    return pl.scan_parquet(block_files(out_dir, k))


def read_block(out_dir: Path | str, k: int) -> pl.DataFrame:
    return scan_block(out_dir, k).collect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-blocks", type=int, default=7)
    parser.add_argument("--rows-per-shard", type=int, default=1_000_000)
    args = parser.parse_args()

    manifest = split_tsv_to_parquet(
        args.src, args.out, n_blocks=args.n_blocks, rows_per_shard=args.rows_per_shard
    )
    for k, info in manifest["blocks"].items():
        ctr = info["positives"] / info["rows"]
        print(f"block {k}: rows={info['rows']:,} ctr={ctr:.4f} shards={info['shards']}")


if __name__ == "__main__":
    main()
