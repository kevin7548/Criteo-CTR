import json

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import log_loss, roc_auc_score

from ctr.compare import (
    PairedBootstrap,
    load_run,
    percentile_ci,
    weighted_auc,
    weighted_logloss,
)
from ctr.metrics import auc, logloss


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    n = 3000
    y = rng.integers(0, 2, n).astype(np.int8)
    good = np.clip(0.5 + 0.3 * (y - 0.5) + rng.normal(0, 0.2, n), 0.01, 0.99)
    bad = np.clip(0.5 + 0.1 * (y - 0.5) + rng.normal(0, 0.2, n), 0.01, 0.99)
    return y, good, bad


# --- weighted metrics ------------------------------------------------------


def test_unit_weights_match_unweighted_metrics(data):
    y, p, _ = data
    ones = np.ones(len(y))
    assert weighted_logloss(y, p, ones) == pytest.approx(logloss(y, p), rel=1e-12)
    assert weighted_auc(y, p, ones) == pytest.approx(auc(y, p), rel=1e-12)


def test_integer_weights_equal_repeating_rows(data):
    y, p, _ = data
    w = np.random.default_rng(1).poisson(1.0, len(y)).astype(np.float64)
    reps = w.astype(int)
    y_rep, p_rep = np.repeat(y, reps), np.repeat(p, reps)

    assert weighted_logloss(y, p, w) == pytest.approx(log_loss(y_rep, p_rep), rel=1e-10)
    assert weighted_auc(y, p, w) == pytest.approx(roc_auc_score(y_rep, p_rep), rel=1e-10)


def test_weighted_auc_counts_ties_as_half():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.2, 0.2, 0.5, 0.9])
    w = np.array([2.0, 1.0, 1.0, 3.0])
    rep = np.repeat(np.arange(4), w.astype(int))

    assert weighted_auc(y, p, w) == pytest.approx(roc_auc_score(y[rep], p[rep]))


# --- paired bootstrap ------------------------------------------------------


def test_point_estimates_are_the_full_sample_metrics(data):
    y, good, bad = data
    boot = PairedBootstrap(y, {"good": good, "bad": bad})
    point = boot.point()

    assert point["good"]["logloss"] == pytest.approx(logloss(y, good))
    assert point["bad"]["auc"] == pytest.approx(auc(y, bad))
    assert point["good"]["pcoc"] == pytest.approx(good.mean() / y.mean())


def test_bootstrap_is_deterministic_per_seed(data):
    y, good, bad = data
    boot = PairedBootstrap(y, {"good": good, "bad": bad})
    a = boot.sample(20, seed=3)
    b = boot.sample(20, seed=3)
    c = boot.sample(20, seed=4)

    np.testing.assert_array_equal(a["good"]["auc"], b["good"]["auc"])
    assert not np.array_equal(a["good"]["auc"], c["good"]["auc"])
    assert a["good"]["auc"].shape == (20,)


def test_paired_difference_ci_excludes_zero_for_a_clearly_better_model(data):
    y, good, bad = data
    boot = PairedBootstrap(y, {"good": good, "bad": bad})
    diff = boot.compare("good", "bad", n_boot=200, seed=0)

    lo, hi = diff["auc"]["ci95"]
    assert lo <= diff["auc"]["diff"] <= hi
    assert lo > 0
    assert diff["logloss"]["ci95"][1] < 0


def test_paired_difference_ci_covers_zero_for_identical_models(data):
    y, good, _ = data
    boot = PairedBootstrap(y, {"a": good, "b": good.copy()})
    diff = boot.compare("a", "b", n_boot=50, seed=0)

    assert diff["auc"]["ci95"] == pytest.approx([0.0, 0.0], abs=1e-12)


def test_paired_ci_is_narrower_than_unpaired_spread(data):
    # Two correlated models see the same resampled rows, so the shared noise cancels.
    y, good, _ = data
    shrunk = 0.8 * good + 0.1
    boot = PairedBootstrap(y, {"good": good, "shrunk": shrunk})
    s = boot.sample(200, seed=0)
    paired = np.std(s["good"]["logloss"] - s["shrunk"]["logloss"])
    unpaired = np.hypot(np.std(s["good"]["logloss"]), np.std(s["shrunk"]["logloss"]))

    assert paired < 0.5 * unpaired


def test_percentile_ci():
    assert percentile_ci(np.arange(101.0)) == pytest.approx([2.5, 97.5])


def test_rejects_predictions_of_the_wrong_length(data):
    y, good, _ = data
    with pytest.raises(ValueError):
        PairedBootstrap(y, {"a": good[:-1]})


# --- load_run ----------------------------------------------------------------


def test_load_run_reads_predictions_and_prior_corrects_downsampled_runs(tmp_path):
    art = tmp_path / "art"
    art.mkdir()
    raw = np.array([0.5, 0.8], dtype=np.float32)
    np.save(art / "test_pred.npy", raw)
    report = tmp_path / "r.json"
    report.write_text(json.dumps({"config": {"artifacts": str(art)}, "negative_keep_rate": 0.25}))

    p = load_run(report, "test")
    np.testing.assert_allclose(p, [0.2, 0.5], rtol=1e-6)
    assert p.dtype == np.float64


def test_load_run_leaves_full_data_runs_as_is(tmp_path):
    art = tmp_path / "art"
    art.mkdir()
    np.save(art / "val_pred.npy", np.array([0.3], dtype=np.float32))
    report = tmp_path / "r.json"
    report.write_text(json.dumps({"config": {"artifacts": str(art)}, "negative_keep_rate": 1.0}))

    np.testing.assert_allclose(load_run(report, "val"), [0.3], rtol=1e-6)


def test_labels_are_read_in_row_order(tmp_path):
    from ctr.compare import load_labels

    block = tmp_path / "block_7"
    block.mkdir()
    pl.DataFrame({"label": pl.Series([1, 0], dtype=pl.Int8)}).write_parquet(
        block / "part-00000.parquet"
    )
    pl.DataFrame({"label": pl.Series([0, 0, 1], dtype=pl.Int8)}).write_parquet(
        block / "part-00001.parquet"
    )
    np.testing.assert_array_equal(load_labels(tmp_path, "test"), [1, 0, 0, 0, 1])
