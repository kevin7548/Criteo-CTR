import numpy as np
import pytest

from ctr.calibration import negative_keep_mask, prior_correct, sampled_ctr
from ctr.metrics import auc

# --- negative_keep_mask ----------------------------------------------------


def test_keep_mask_keeps_every_positive():
    labels = np.array([1, 0, 1, 0, 0, 1] * 500, dtype=np.int8)
    keep = negative_keep_mask(labels, 0.1, np.random.default_rng(0))

    assert keep.dtype == bool and keep.shape == labels.shape
    assert keep[labels == 1].all()


def test_keep_mask_keeps_negatives_at_the_requested_rate():
    labels = np.zeros(200_000, dtype=np.int8)
    keep = negative_keep_mask(labels, 0.25, np.random.default_rng(0))

    assert keep.mean() == pytest.approx(0.25, abs=0.005)


def test_keep_rate_one_keeps_every_row():
    labels = np.array([0, 1, 0, 0], dtype=np.int8)
    assert negative_keep_mask(labels, 1.0, np.random.default_rng(0)).all()


def test_keep_mask_is_deterministic_per_rng_seed():
    labels = np.zeros(1000, dtype=np.int8)
    a = negative_keep_mask(labels, 0.5, np.random.default_rng([3, 1]))
    b = negative_keep_mask(labels, 0.5, np.random.default_rng([3, 1]))
    c = negative_keep_mask(labels, 0.5, np.random.default_rng([3, 2]))

    np.testing.assert_array_equal(a, b)
    assert (a != c).any()


@pytest.mark.parametrize("rate", [0.0, -0.5, 1.5])
def test_keep_mask_rejects_rates_outside_half_open_unit_interval(rate):
    with pytest.raises(ValueError):
        negative_keep_mask(np.zeros(3, dtype=np.int8), rate, np.random.default_rng(0))


# --- sampled_ctr -----------------------------------------------------------


def test_sampled_ctr_counts_all_positives_and_a_fraction_of_negatives():
    # 25 positives, 75 negatives; keep 1/3 of negatives -> 25 / (25 + 25).
    assert sampled_ctr(0.25, 1 / 3) == pytest.approx(0.5)
    assert sampled_ctr(0.25, 1.0) == pytest.approx(0.25)


def test_sampled_ctr_matches_an_actual_sample():
    rng = np.random.default_rng(0)
    labels = (rng.random(400_000) < 0.2).astype(np.int8)
    keep = negative_keep_mask(labels, 0.1, rng)

    assert labels[keep].mean() == pytest.approx(sampled_ctr(labels.mean(), 0.1), abs=0.005)


# --- prior_correct ---------------------------------------------------------


@pytest.mark.parametrize("w", [0.5, 0.25, 0.1])
def test_prior_correct_inverts_sampled_ctr(w):
    for c in (0.01, 0.2553, 0.9):
        assert prior_correct(np.array([sampled_ctr(c, w)]), w)[0] == pytest.approx(c)


def test_prior_correct_adds_log_w_to_the_logit():
    p = np.array([0.05, 0.3, 0.7, 0.99])
    w = 0.25
    q = prior_correct(p, w)

    logit = lambda x: np.log(x / (1 - x))
    np.testing.assert_allclose(logit(q), logit(p) + np.log(w), rtol=1e-12)


def test_prior_correct_with_w_one_is_identity():
    p = np.array([0.1, 0.5, 0.9])
    np.testing.assert_allclose(prior_correct(p, 1.0), p)


def test_prior_correct_is_monotone_so_auc_is_unchanged():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 1000)
    p = rng.random(1000)

    assert auc(y, prior_correct(p, 0.1)) == pytest.approx(auc(y, p), abs=1e-12)


def test_prior_correct_keeps_extremes_inside_the_unit_interval():
    q = prior_correct(np.array([0.0, 1e-12, 1 - 1e-12, 1.0]), 0.1)
    assert q[0] == 0.0 and q[-1] == 1.0
    assert 0 < q[1] < q[2] < 1


@pytest.mark.parametrize("w", [0.0, 1.5])
def test_prior_correct_rejects_bad_rate(w):
    with pytest.raises(ValueError):
        prior_correct(np.array([0.5]), w)
