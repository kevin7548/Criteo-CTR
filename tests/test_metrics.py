import math

import numpy as np
import pytest
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss, roc_auc_score

from ctr.baseline import constant_logloss
from ctr.metrics import EPS, auc, calibration_bins, ece, logloss, ne, pcoc, summary


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def logit(p):
    return np.log(p) - np.log1p(-p)


def synthetic(n: int, seed: int, base: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """Perfectly calibrated data: y ~ Bernoulli(p), logits spread around logit(base)."""
    rng = np.random.default_rng(seed)
    p = sigmoid(rng.normal(math.log(base / (1 - base)), 1.0, n))
    y = (rng.random(n) < p).astype(np.int8)
    return y, p


SEEDS = [0, 1, 2]


# --- input validation -----------------------------------------------------

INVALID_INPUTS = [
    pytest.param([0, 1], [0.5], id="length-mismatch"),
    pytest.param([], [], id="empty"),
    pytest.param([0, 2], [0.5, 0.5], id="label-not-binary"),
    pytest.param([0, 1], [0.5, -0.1], id="p-below-0"),
    pytest.param([0, 1], [0.5, 1.1], id="p-above-1"),
    pytest.param([0, 1], [0.5, float("nan")], id="p-nan"),
    pytest.param([[0, 1]], [[0.5, 0.5]], id="2d"),
]


@pytest.mark.parametrize("metric", [logloss, auc, pcoc, ece])
@pytest.mark.parametrize(("y", "p"), INVALID_INPUTS)
def test_metrics_reject_invalid_inputs(metric, y, p):
    with pytest.raises(ValueError):
        metric(y, p)


@pytest.mark.parametrize(("y", "p"), INVALID_INPUTS)
def test_ne_rejects_invalid_inputs(y, p):
    with pytest.raises(ValueError):
        ne(y, p, base_rate=0.25)


# --- LogLoss --------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_logloss_matches_sklearn(seed):
    y, p = synthetic(10_000, seed)

    assert logloss(y, p) == pytest.approx(log_loss(y, p), rel=1e-12)


@pytest.mark.parametrize(("positives", "n", "c"), [(3, 10, 0.3), (1, 7, 0.9), (0, 5, 0.2)])
def test_logloss_of_constant_prediction_matches_baseline(positives, n, c):
    y = np.array([1] * positives + [0] * (n - positives))

    assert logloss(y, np.full(n, c)) == pytest.approx(constant_logloss(positives, n, c))


def test_logloss_clips_certain_wrong_predictions():
    assert logloss([1], [0.0]) == pytest.approx(-math.log(EPS))
    assert logloss([0], [1.0]) == pytest.approx(-math.log(EPS), rel=1e-3)


def test_logloss_is_near_zero_for_confident_correct_predictions():
    assert logloss([0, 1], [1e-9, 1 - 1e-9]) < 1e-8


def test_logloss_handles_float32_probabilities_that_round_to_one():
    p = np.array([0.99999999, 0.5], dtype=np.float32)  # first value rounds to 1.0

    assert math.isfinite(logloss([0, 1], p))


# --- AUC ------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_auc_matches_sklearn(seed):
    y, p = synthetic(10_000, seed)

    assert auc(y, p) == pytest.approx(roc_auc_score(y, p), rel=1e-12)


@pytest.mark.parametrize("seed", SEEDS)
def test_auc_matches_sklearn_with_heavy_ties(seed):
    y, p = synthetic(10_000, seed)
    p = np.round(p, 2)

    assert auc(y, p) == pytest.approx(roc_auc_score(y, p), rel=1e-12)


def test_auc_counts_ties_as_half():
    # pairs (pos, neg): 0.5>0.1 win, 0.5=0.5 tie, 0.9>0.1 win, 0.9>0.5 win
    assert auc([0, 1, 0, 1], [0.1, 0.5, 0.5, 0.9]) == pytest.approx(3.5 / 4)


def test_auc_extremes():
    y = [0, 0, 1, 1]

    assert auc(y, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auc(y, [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert auc(y, [0.3, 0.3, 0.3, 0.3]) == 0.5


def test_auc_ignores_row_order():
    y, p = synthetic(5_000, 0)
    perm = np.random.default_rng(1).permutation(len(y))

    assert auc(y[perm], p[perm]) == auc(y, p)


@pytest.mark.parametrize("w", [0.5, 0.25, 0.1])
def test_auc_is_invariant_to_prior_correction(w):
    """Shifting every logit by log w keeps the order, so AUC cannot see it (ADR-003, ADR-005)."""
    y, p = synthetic(20_000, 0)
    shifted = sigmoid(logit(p) + math.log(w))
    order = np.argsort(p)
    assert np.all(np.diff(shifted[order]) > 0)  # the float transform kept strict order

    assert auc(y, shifted) == auc(y, p)


@pytest.mark.parametrize("y", [[0, 0, 0], [1, 1, 1]])
def test_auc_rejects_single_class(y):
    with pytest.raises(ValueError):
        auc(y, [0.1, 0.5, 0.9])


# --- calibration bins and ECE ---------------------------------------------


def test_ece_hand_computed_example():
    # sorted by p: bin 1 = p 0.1..0.4 (mean 0.25), labels 1,0,0,0 (mean 0.25) -> gap 0
    #              bin 2 = p 0.6..0.9 (mean 0.75), labels 1,1,1,1 (mean 1.0)  -> gap 0.25
    p = [0.9, 0.1, 0.6, 0.3, 0.8, 0.2, 0.7, 0.4]
    y = [1, 1, 1, 0, 1, 0, 1, 0]

    assert ece(y, p, n_bins=2) == pytest.approx(0.125)


def test_quantile_bins_have_equal_counts_and_increasing_predictions():
    y, p = synthetic(1_003, 0)

    bins = calibration_bins(y, p, n_bins=20)

    assert len(bins.count) == 20
    assert bins.count.sum() == 1_003
    assert bins.count.max() - bins.count.min() <= 1
    assert np.all(np.diff(bins.mean_pred) >= 0)


def test_default_is_twenty_quantile_bins():
    y, p = synthetic(10_000, 0)

    assert len(calibration_bins(y, p).count) == 20
    assert ece(y, p) == ece(y, p, n_bins=20, strategy="quantile")


def test_bins_report_mean_prediction_and_label():
    p = [0.1, 0.2, 0.3, 0.4]
    y = [0, 1, 1, 1]

    bins = calibration_bins(y, p, n_bins=2)

    np.testing.assert_allclose(bins.mean_pred, [0.15, 0.35])
    np.testing.assert_allclose(bins.mean_label, [0.5, 1.0])
    np.testing.assert_array_equal(bins.count, [2, 2])


@pytest.mark.parametrize("strategy", ["quantile", "uniform"])
def test_ece_of_constant_prediction_is_its_bias(strategy):
    y = np.array([1] * 30 + [0] * 70)

    assert ece(y, np.full(100, 0.25), strategy=strategy) == pytest.approx(0.05)


def test_ece_ignores_row_order():
    y, p = synthetic(5_000, 0)
    perm = np.random.default_rng(1).permutation(len(y))

    assert ece(y[perm], p[perm]) == pytest.approx(ece(y, p), abs=1e-15)


def test_ece_ignores_row_order_with_heavy_ties():
    y, p = synthetic(5_000, 0)
    p = np.round(p, 2)
    perm = np.random.default_rng(1).permutation(len(y))

    assert ece(y[perm], p[perm]) == pytest.approx(ece(y, p), abs=1e-15)


def test_quantile_bins_never_split_a_tie_group():
    # positional bins of 2 would cut the four 0.3s apart; the group stays whole in the
    # bin of its middle row (sorted positions 1-4 -> position 2 -> bin 1)
    p = [0.1, 0.3, 0.3, 0.3, 0.3, 0.9]
    y = [0, 1, 1, 0, 0, 1]

    bins = calibration_bins(y, p, n_bins=3)

    np.testing.assert_array_equal(bins.count, [1, 4, 1])
    np.testing.assert_allclose(bins.mean_pred, [0.1, 0.3, 0.9])
    np.testing.assert_allclose(bins.mean_label, [0.0, 0.5, 1.0])


def test_constant_prediction_fills_a_single_quantile_bin():
    y = np.array([1] * 30 + [0] * 70)

    np.testing.assert_array_equal(calibration_bins(y, np.full(100, 0.25)).count, [100])


def test_ece_is_small_for_calibrated_predictions():
    y, p = synthetic(200_000, 0)

    assert ece(y, p) < 0.01


def test_uniform_bins_match_sklearn_calibration_curve():
    y, p = synthetic(10_000, 0)

    bins = calibration_bins(y, p, n_bins=10, strategy="uniform")
    prob_true, prob_pred = calibration_curve(y, p, n_bins=10, strategy="uniform")

    np.testing.assert_allclose(bins.mean_label, prob_true)
    np.testing.assert_allclose(bins.mean_pred, prob_pred)


def test_uniform_bins_skip_empty_bins_and_keep_one_in_last_bin():
    bins = calibration_bins([0, 1, 0, 1], [0.05, 0.15, 0.25, 1.0], n_bins=10, strategy="uniform")

    np.testing.assert_array_equal(bins.count, [1, 1, 1, 1])
    np.testing.assert_allclose(bins.mean_pred, [0.05, 0.15, 0.25, 1.0])


def test_quantile_bins_need_at_least_one_row_per_bin():
    with pytest.raises(ValueError):
        calibration_bins([0, 1, 0], [0.1, 0.2, 0.3], n_bins=4)


@pytest.mark.parametrize(("n_bins", "strategy"), [(0, "quantile"), (20, "kmeans")])
def test_calibration_bins_reject_bad_options(n_bins, strategy):
    y, p = synthetic(100, 0)

    with pytest.raises(ValueError):
        calibration_bins(y, p, n_bins=n_bins, strategy=strategy)


# --- PCOC -----------------------------------------------------------------


def test_pcoc_is_mean_prediction_over_ctr():
    assert pcoc([0, 1], [0.2, 0.4]) == pytest.approx(0.3 / 0.5)


def test_pcoc_of_constant_prediction():
    y = np.array([1] * 10 + [0] * 90)

    assert pcoc(y, np.full(100, 0.24)) == pytest.approx(0.24 / 0.10)


def test_pcoc_rejects_no_positives():
    with pytest.raises(ValueError):
        pcoc([0, 0], [0.1, 0.2])


# --- NE -------------------------------------------------------------------


def test_ne_is_one_for_the_constant_base_rate():
    y = np.array([1] * 10 + [0] * 90)

    assert ne(y, np.full(100, 0.24), base_rate=0.24) == pytest.approx(1.0)


def test_ne_denominator_is_the_constant_baseline_logloss():
    y, p = synthetic(10_000, 0)
    positives = int(y.sum())

    expected = logloss(y, p) / constant_logloss(positives, len(y), 0.255)

    assert ne(y, p, base_rate=0.255) == pytest.approx(expected)


def test_ne_below_one_for_an_informative_model():
    y, p = synthetic(50_000, 0)

    assert ne(y, p, base_rate=float(y.mean())) < 0.95


@pytest.mark.parametrize("base_rate", [0.0, 1.0, -0.1, 1.5, float("nan")])
def test_ne_rejects_base_rate_outside_open_interval(base_rate):
    with pytest.raises(ValueError):
        ne([0, 1], [0.3, 0.7], base_rate=base_rate)


# --- the project's core claim in miniature --------------------------------


def test_inflated_predictions_keep_auc_but_break_calibration():
    """A model trained with negatives kept at rate w predicts sigmoid(logit(p) - log w)."""
    y, p = synthetic(200_000, 0)
    inflated = sigmoid(logit(p) - math.log(0.1))

    assert auc(y, inflated) == pytest.approx(auc(y, p), abs=1e-9)
    assert pcoc(y, inflated) > 2.0
    assert ece(y, inflated) > 10 * ece(y, p)
    assert logloss(y, inflated) > logloss(y, p) + 0.1


# --- summary --------------------------------------------------------------


def test_summary_collects_every_metric():
    y, p = synthetic(10_000, 0)

    result = summary(y, p, base_rate=0.255)

    assert result == {
        "n": 10_000,
        "positives": int(y.sum()),
        "ctr": pytest.approx(float(y.mean())),
        "mean_pred": pytest.approx(float(p.mean())),
        "logloss": pytest.approx(logloss(y, p)),
        "ne": pytest.approx(ne(y, p, base_rate=0.255)),
        "auc": pytest.approx(auc(y, p)),
        "ece": pytest.approx(ece(y, p)),
        "pcoc": pytest.approx(pcoc(y, p)),
    }
