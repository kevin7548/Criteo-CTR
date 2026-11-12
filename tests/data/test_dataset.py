import numpy as np
import polars as pl
import pytest

from ctr.data.dataset import iter_batches, load_arrays, split_files
from ctr.data.vocab import FIELDS


def write_encoded(root, sizes: dict[int, list[int]]):
    """Encoded-layout shards whose I1 column is a global row counter."""
    start = 0
    for k, shard_sizes in sizes.items():
        block = root / f"block_{k}"
        block.mkdir(parents=True)
        for i, n in enumerate(shard_sizes):
            rows = np.arange(start, start + n)
            df = pl.DataFrame(
                {
                    "label": pl.Series(rows % 2, dtype=pl.Int8),
                    **{f: pl.Series(rows * (j + 1), dtype=pl.Int32) for j, f in enumerate(FIELDS)},
                }
            )
            df.write_parquet(block / f"part-{i:05d}.parquet")
            start += n
    return root


@pytest.fixture
def encoded(tmp_path):
    return write_encoded(tmp_path / "enc", {1: [100, 100, 37], 2: [100, 64], 6: [50]})


def row_ids(batches):
    return np.concatenate([ids[:, 0] for ids, _ in batches])


# --- split_files -----------------------------------------------------------


def test_split_files_lists_blocks_in_chronological_order(encoded):
    files = split_files(encoded, (1, 2))
    assert [(f.parent.name, f.name) for f in files] == [
        ("block_1", "part-00000.parquet"),
        ("block_1", "part-00001.parquet"),
        ("block_1", "part-00002.parquet"),
        ("block_2", "part-00000.parquet"),
        ("block_2", "part-00001.parquet"),
    ]


def test_split_files_max_shards_takes_the_earliest_shards(encoded):
    files = split_files(encoded, (1, 2), max_shards=4)
    assert [(f.parent.name, f.name) for f in files][-1] == ("block_2", "part-00000.parquet")
    assert len(files) == 4


def test_split_files_rejects_non_positive_max_shards(encoded):
    with pytest.raises(ValueError):
        split_files(encoded, (1,), max_shards=0)


# --- load_arrays -----------------------------------------------------------


def test_load_arrays_returns_int32_ids_and_int8_labels(encoded):
    ids, labels = load_arrays(split_files(encoded, (6,)))

    assert ids.shape == (50, len(FIELDS))
    assert ids.dtype == np.int32 and ids.flags.c_contiguous
    assert labels.dtype == np.int8
    assert ids[:, 0].tolist() == list(range(401, 451))
    assert ids[3, 2] == ids[3, 0] * 3


# --- iter_batches ----------------------------------------------------------


def test_unshuffled_batches_concatenate_to_the_split_in_order(encoded):
    files = split_files(encoded, (1, 2))
    batches = list(iter_batches(files, batch_size=32))

    assert row_ids(batches).tolist() == list(range(401))
    assert all(len(ids) <= 32 for ids, _ in batches)
    assert all(len(ids) == len(labels) for ids, labels in batches)


def test_batches_keep_label_aligned_with_row(encoded):
    files = split_files(encoded, (1, 2))
    for shuffle in (False, True):
        for ids, labels in iter_batches(files, 16, shuffle=shuffle, seed=3):
            assert (labels == ids[:, 0] % 2).all()


def test_shuffled_epoch_visits_every_row_once(encoded):
    files = split_files(encoded, (1, 2))
    seen = row_ids(iter_batches(files, 32, shuffle=True, seed=0, shards_per_chunk=2))

    assert sorted(seen.tolist()) == list(range(401))
    assert seen.tolist() != list(range(401))


def test_shuffle_is_deterministic_per_seed(encoded):
    files = split_files(encoded, (1, 2))
    a = row_ids(iter_batches(files, 32, shuffle=True, seed=7))
    b = row_ids(iter_batches(files, 32, shuffle=True, seed=7))
    c = row_ids(iter_batches(files, 32, shuffle=True, seed=8))

    assert a.tolist() == b.tolist()
    assert a.tolist() != c.tolist()


def test_shuffle_mixes_rows_across_shards_in_a_chunk(encoded):
    files = split_files(encoded, (1,))[:2]  # rows 0..99 and 100..199
    first, _ = next(iter_batches(files, 64, shuffle=True, seed=0, shards_per_chunk=2))

    assert (first[:, 0] < 100).any() and (first[:, 0] >= 100).any()


def test_iter_batches_rejects_bad_arguments(encoded):
    files = split_files(encoded, (1,))
    with pytest.raises(ValueError):
        next(iter_batches(files, 0))
    with pytest.raises(ValueError):
        next(iter_batches(files, 8, shards_per_chunk=0))
    with pytest.raises(ValueError):
        next(iter_batches(files, 8, negative_keep_rate=0.0))


# --- negative downsampling (ADR-011) ----------------------------------------


def sampled(files, **kwargs):
    batches = iter_batches(files, 32, shuffle=True, negative_keep_rate=0.5, **kwargs)
    ids, labels = zip(*batches)
    return np.concatenate(ids)[:, 0], np.concatenate(labels)


def test_downsampling_keeps_all_positives_and_some_negatives(encoded):
    files = split_files(encoded, (1, 2))
    rows, labels = sampled(files, seed=0)

    assert sorted(rows[labels == 1].tolist()) == list(range(1, 401, 2))
    assert 0 < (labels == 0).sum() < 201
    assert (labels == rows % 2).all()


def test_downsampling_keeps_the_same_rows_whatever_the_shuffle(encoded):
    files = split_files(encoded, (1, 2))
    a, _ = sampled(files, seed=0, shards_per_chunk=2)
    b, _ = sampled(files, seed=5, shards_per_chunk=3)

    assert sorted(a.tolist()) == sorted(b.tolist())
    assert a.tolist() != b.tolist()


def test_downsampling_on_a_prefix_keeps_what_the_full_run_keeps_there(encoded):
    files = split_files(encoded, (1, 2))
    full, _ = sampled(files, seed=0)
    prefix, _ = sampled(files[:2], seed=1)

    assert sorted(prefix.tolist()) == sorted(r for r in full.tolist() if r < 200)


def test_downsampling_mask_depends_on_sample_seed(encoded):
    files = split_files(encoded, (1, 2))
    a, _ = sampled(files, seed=0, sample_seed=0)
    b, _ = sampled(files, seed=0, sample_seed=1)

    assert sorted(a.tolist()) != sorted(b.tolist())


def test_unshuffled_reads_ignore_nothing_by_default(encoded):
    files = split_files(encoded, (1, 2))
    assert len(row_ids(iter_batches(files, 32))) == 401
