"""Per-field vocabularies built from training blocks only (ADR-002, ADR-009).

Every one of the 39 fields becomes a categorical id:

- I1–I13 are first bucketed: values <= 2 stay as they are, larger values become
  floor(ln(x)^2) + 3, so buckets never collide with the raw small values.
- C1–C26 keep their 32-bit hash value.

Within a field, id 0 is "missing" (null), id 1 is OOV (a value seen fewer than
`min_count` times in training, or never seen), and kept values get ids from 2 in
order of training count (descending, ties by value).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from ctr.data.split import CAT_COLS, INT_COLS

FIELDS = [*INT_COLS, *CAT_COLS]
MISSING_ID = 0
OOV_ID = 1
FIRST_VALUE_ID = 2


def numeric_token(x: pl.Expr) -> pl.Expr:
    """Bucket an integer feature: x if x <= 2 else floor(ln(x)^2) + 3. Nulls stay null."""
    x = x.cast(pl.Int64)
    big = (x.cast(pl.Float64).log() ** 2).floor().cast(pl.Int64) + 3
    return pl.when(x > 2).then(big).otherwise(x)


def tokenize(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Block-schema frame -> label plus one Int64 token column per field (nulls kept)."""
    return lf.select(
        pl.col("label"),
        *[numeric_token(pl.col(c)).alias(c) for c in INT_COLS],
        *[pl.col(c).cast(pl.Int64) for c in CAT_COLS],
    )


@dataclass
class Vocab:
    """Kept values and their training counts per field, in id order."""

    min_count: int
    values: dict[str, np.ndarray]
    counts: dict[str, np.ndarray]
    missing: dict[str, int]
    oov: dict[str, int]

    @property
    def sizes(self) -> list[int]:
        """Number of ids per field, in FIELDS order, including missing and OOV."""
        return [FIRST_VALUE_ID + len(self.values[f]) for f in FIELDS]

    def save(self, out_dir: Path | str) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pl.concat(
            [
                pl.DataFrame(
                    {
                        "field": pl.Series([f] * len(self.values[f]), dtype=pl.String),
                        "value": pl.Series(self.values[f], dtype=pl.Int64),
                        "count": pl.Series(self.counts[f], dtype=pl.Int64),
                    }
                )
                for f in FIELDS
            ]
        ).write_parquet(out_dir / "vocab.parquet", compression="zstd")
        meta = {
            "min_count": self.min_count,
            "fields": {
                f: {
                    "size": size,
                    "kept": len(self.values[f]),
                    "missing": self.missing[f],
                    "oov": self.oov[f],
                }
                for f, size in zip(FIELDS, self.sizes, strict=True)
            },
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    @classmethod
    def load(cls, vocab_dir: Path | str) -> Vocab:
        vocab_dir = Path(vocab_dir)
        meta = json.loads((vocab_dir / "meta.json").read_text())
        df = pl.read_parquet(vocab_dir / "vocab.parquet")
        values, counts = {}, {}
        for f in FIELDS:
            part = df.filter(pl.col("field") == f)  # row order is id order
            values[f] = part["value"].to_numpy()
            counts[f] = part["count"].to_numpy()
        fields = meta["fields"]
        return cls(
            min_count=meta["min_count"],
            values=values,
            counts=counts,
            missing={f: fields[f]["missing"] for f in FIELDS},
            oov={f: fields[f]["oov"] for f in FIELDS},
        )


def build_vocab(train: pl.LazyFrame, min_count: int) -> Vocab:
    """Count every field's tokens over `train`, which must hold training rows only."""
    if min_count < 1:
        raise ValueError(f"min_count must be >= 1, got {min_count}")
    tokens = tokenize(train)
    values, counts, missing, oov = {}, {}, {}, {}
    for f in FIELDS:
        freq = tokens.group_by(f).agg(pl.len().alias("count")).collect()
        nulls = freq.filter(pl.col(f).is_null())
        freq = freq.filter(pl.col(f).is_not_null())
        kept = freq.filter(pl.col("count") >= min_count).sort(
            ["count", f], descending=[True, False]
        )
        values[f] = kept[f].to_numpy()
        counts[f] = kept["count"].cast(pl.Int64).to_numpy()
        missing[f] = int(nulls["count"].sum())
        oov[f] = int(freq.filter(pl.col("count") < min_count)["count"].sum())
    return Vocab(min_count=min_count, values=values, counts=counts, missing=missing, oov=oov)


def encode(lf: pl.LazyFrame, vocab: Vocab) -> pl.LazyFrame:
    """Block-schema frame -> label (Int8) plus one Int32 id column per field."""
    tokens = tokenize(lf)
    exprs = []
    for f in FIELDS:
        ids = np.arange(FIRST_VALUE_ID, FIRST_VALUE_ID + len(vocab.values[f]), dtype=np.int32)
        col = pl.col(f)
        lookup = col.replace_strict(
            pl.Series(vocab.values[f], dtype=pl.Int64),
            pl.Series(ids, dtype=pl.Int32),
            default=OOV_ID,
            return_dtype=pl.Int32,
        )
        exprs.append(
            pl.when(col.is_null()).then(pl.lit(MISSING_ID, pl.Int32)).otherwise(lookup).alias(f)
        )
    return tokens.select(pl.col("label").cast(pl.Int8), *exprs)
