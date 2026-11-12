import json

import numpy as np
import polars as pl
import pytest

from ctr.data.encode import encode_blocks
from ctr.data.split import read_block
from ctr.data.vocab import FIELDS, FIRST_VALUE_ID, OOV_ID, Vocab, build_vocab, encode


def test_encoded_blocks_mirror_shards_and_keep_row_order(blocks_dir, tmp_path):
    out = tmp_path / "encoded"
    encode_blocks(blocks_dir, out, min_count=2)

    for k in range(1, 8):
        src = sorted(p.name for p in (blocks_dir / f"block_{k}").glob("*.parquet"))
        dst = sorted(p.name for p in (out / f"block_{k}").glob("*.parquet"))
        assert dst == src
        assert read_block(out, k)["label"].to_list() == read_block(blocks_dir, k)["label"].to_list()


def test_encoded_blocks_match_encoding_with_train_only_vocab(blocks_dir, tmp_path):
    out = tmp_path / "encoded"
    encode_blocks(blocks_dir, out, min_count=2)

    train = pl.concat([read_block(blocks_dir, k) for k in range(1, 6)])
    expected_vocab = build_vocab(train.lazy(), min_count=2)
    vocab = Vocab.load(out / "vocab")
    assert vocab.sizes == expected_vocab.sizes

    for k in (1, 6, 7):
        expected = encode(read_block(blocks_dir, k).lazy(), expected_vocab).collect()
        assert read_block(out, k).equals(expected)


def test_values_seen_only_after_training_blocks_are_oov(make_blocks, make_shard, tmp_path):
    rng = np.random.default_rng(1)
    shards = {k: [make_shard(rng, 50)] for k in range(1, 8)}
    # A C2 value that appears 50 times, but only in validation and test blocks.
    for k in (6, 7):
        shards[k] = [shards[k][0].with_columns(pl.lit(424242, pl.UInt32).alias("C2"))]
    out = tmp_path / "encoded"
    encode_blocks(make_blocks(shards), out, min_count=1)

    assert 424242 not in Vocab.load(out / "vocab").values["C2"].tolist()
    assert set(read_block(out, 6)["C2"].to_list()) == {OOV_ID}
    assert set(read_block(out, 7)["C2"].to_list()) == {OOV_ID}


def test_manifest_records_source_counts_and_field_sizes(blocks_dir, tmp_path):
    out = tmp_path / "encoded"
    manifest = encode_blocks(blocks_dir, out, min_count=2)

    source = json.loads((blocks_dir / "manifest.json").read_text())
    on_disk = json.loads((out / "manifest.json").read_text())
    assert on_disk == manifest
    assert manifest["blocks"] == source["blocks"]
    assert manifest["vocab"]["train_blocks"] == [1, 2, 3, 4, 5]
    assert manifest["vocab"]["min_count"] == 2
    assert manifest["fields"] == FIELDS
    assert manifest["field_sizes"] == Vocab.load(out / "vocab").sizes
    assert all(s >= FIRST_VALUE_ID for s in manifest["field_sizes"])


def test_encode_blocks_refuses_to_overwrite(blocks_dir, tmp_path):
    out = tmp_path / "encoded"
    out.mkdir()
    with pytest.raises(FileExistsError):
        encode_blocks(blocks_dir, out, min_count=2)


def test_encode_blocks_leaves_no_partial_output_on_failure(blocks_dir, tmp_path):
    (blocks_dir / "block_7" / "part-00000.parquet").write_bytes(b"not parquet")
    out = tmp_path / "encoded"
    with pytest.raises(Exception):  # noqa: B017 - any read error will do
        encode_blocks(blocks_dir, out, min_count=2)
    assert not out.exists()
