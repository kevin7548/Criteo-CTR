import math
from itertools import pairwise

import numpy as np
import polars as pl
import pytest

from ctr.data.split import CAT_COLS, INT_COLS
from ctr.data.vocab import (
    FIELDS,
    FIRST_VALUE_ID,
    MISSING_ID,
    OOV_ID,
    Vocab,
    build_vocab,
    encode,
    numeric_token,
    tokenize,
)


def frame(**cols) -> pl.DataFrame:
    """A block-schema frame; fields not given are all-null."""
    n = len(next(iter(cols.values())))
    data = {"label": pl.Series([0] * n, dtype=pl.Int8)}
    for c in INT_COLS:
        data[c] = pl.Series(cols.get(c, [None] * n), dtype=pl.Int32)
    for c in CAT_COLS:
        data[c] = pl.Series(cols.get(c, [None] * n), dtype=pl.UInt32)
    if "label" in cols:
        data["label"] = pl.Series(cols["label"], dtype=pl.Int8)
    return pl.DataFrame(data)


def tokens_of(values: list[int | None]) -> list[int | None]:
    df = pl.DataFrame({"x": pl.Series(values, dtype=pl.Int32)})
    return df.select(numeric_token(pl.col("x")))["x"].to_list()


# --- numeric_token ---------------------------------------------------------


def test_fields_are_numeric_then_categorical():
    assert FIELDS == [*INT_COLS, *CAT_COLS]
    assert len(FIELDS) == 39


def test_numeric_token_keeps_small_values_and_nulls():
    assert tokens_of([-3, -1, 0, 1, 2, None]) == [-3, -1, 0, 1, 2, None]


@pytest.mark.parametrize("x", [3, 4, 5, 10, 100, 12345, 2**31 - 1])
def test_numeric_token_buckets_large_values_by_squared_log(x):
    assert tokens_of([x]) == [math.floor(math.log(x) ** 2) + 3]


def test_numeric_token_buckets_never_collide_with_raw_small_values():
    big = tokens_of(list(range(3, 5000)))
    assert min(big) > 2


def test_numeric_token_is_monotone():
    xs = list(range(-3, 20000))
    tokens = tokens_of(xs)
    assert all(a <= b for a, b in pairwise(tokens))


# --- tokenize --------------------------------------------------------------


def test_tokenize_gives_int64_tokens_for_every_field_and_keeps_label():
    df = frame(label=[1, 0], I1=[5, None], C1=[0xFFFFFFFF, 7])
    out = tokenize(df.lazy()).collect()

    assert out.columns == ["label", *FIELDS]
    assert out["label"].to_list() == [1, 0]
    assert all(out[c].dtype == pl.Int64 for c in FIELDS)
    assert out["I1"].to_list() == [math.floor(math.log(5) ** 2) + 3, None]
    assert out["C1"].to_list() == [0xFFFFFFFF, 7]


# --- build_vocab -----------------------------------------------------------


def test_build_vocab_keeps_values_at_or_above_min_count():
    #       value 7 x3, value 8 x2, value 9 x1, null x2
    c1 = [7, 8, 7, 9, None, 8, 7, None]
    vocab = build_vocab(frame(C1=c1).lazy(), min_count=2)

    assert sorted(vocab.values["C1"].tolist()) == [7, 8]
    assert vocab.missing["C1"] == 2
    assert vocab.oov["C1"] == 1


def test_build_vocab_orders_by_count_then_value():
    c1 = [5, 3, 3, 4, 4, 5, 6, 6, 6]
    vocab = build_vocab(frame(C1=c1).lazy(), min_count=1)

    assert vocab.values["C1"].tolist() == [6, 3, 4, 5]
    assert vocab.counts["C1"].tolist() == [3, 2, 2, 2]


def test_build_vocab_sizes_reserve_missing_and_oov():
    vocab = build_vocab(frame(C1=[1, 1, 2], I1=[None, None, None]).lazy(), min_count=1)
    sizes = dict(zip(FIELDS, vocab.sizes, strict=True))

    assert sizes["C1"] == 2 + FIRST_VALUE_ID
    assert sizes["I1"] == FIRST_VALUE_ID
    assert vocab.missing["I1"] == 3


def test_build_vocab_rejects_min_count_below_one():
    with pytest.raises(ValueError):
        build_vocab(frame(C1=[1]).lazy(), min_count=0)


# --- encode ----------------------------------------------------------------


def test_encode_maps_missing_rare_unseen_and_kept_values():
    train = frame(C1=[7, 7, 7, 8, 8, 9, None])
    vocab = build_vocab(train.lazy(), min_count=2)

    evaluated = frame(label=[1, 0, 1, 0, 1], C1=[7, 8, 9, 123, None])
    out = encode(evaluated.lazy(), vocab).collect()

    assert out["C1"].to_list() == [FIRST_VALUE_ID, FIRST_VALUE_ID + 1, OOV_ID, OOV_ID, MISSING_ID]
    assert out["label"].to_list() == [1, 0, 1, 0, 1]


def test_encode_tokenizes_numeric_fields_before_lookup():
    train = frame(I1=[100, 101, 102, 1, 1])  # 100..102 share one bucket
    vocab = build_vocab(train.lazy(), min_count=2)

    out = encode(frame(I1=[105, 1, 50000]).lazy(), vocab).collect()

    assert vocab.values["I1"].tolist() == [math.floor(math.log(100) ** 2) + 3, 1]
    assert out["I1"].to_list() == [FIRST_VALUE_ID, FIRST_VALUE_ID + 1, OOV_ID]


def test_encode_output_schema():
    df = frame(C1=[1, 2], I3=[0, 4])
    out = encode(df.lazy(), build_vocab(df.lazy(), min_count=1)).collect()

    assert out.columns == ["label", *FIELDS]
    assert out["label"].dtype == pl.Int8
    assert all(out[c].dtype == pl.Int32 for c in FIELDS)


def test_encoded_ids_are_within_field_sizes():
    rng = np.random.default_rng(0)
    df = frame(C2=rng.integers(0, 50, 500).tolist(), I2=rng.integers(-3, 3000, 500).tolist())
    vocab = build_vocab(df.lazy(), min_count=3)
    out = encode(df.lazy(), vocab).collect()

    for field, size in zip(FIELDS, vocab.sizes, strict=True):
        assert out[field].min() >= 0
        assert out[field].max() < size


# --- save / load -----------------------------------------------------------


def test_vocab_round_trips_through_disk(tmp_path):
    df = frame(C1=[3, 3, 4, None], I1=[0, 0, 7, 7])
    vocab = build_vocab(df.lazy(), min_count=1)
    vocab.save(tmp_path / "vocab")
    loaded = Vocab.load(tmp_path / "vocab")

    assert loaded.min_count == 1
    assert loaded.sizes == vocab.sizes
    for f in FIELDS:
        assert loaded.values[f].tolist() == vocab.values[f].tolist()
        assert loaded.counts[f].tolist() == vocab.counts[f].tolist()
        assert loaded.missing[f] == vocab.missing[f]
        assert loaded.oov[f] == vocab.oov[f]
