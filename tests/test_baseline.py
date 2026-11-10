import json
import math

import pytest

from ctr.baseline import constant_baseline, constant_logloss, main


def row_by_row_logloss(positives: int, n: int, p: float) -> float:
    labels = [1] * positives + [0] * (n - positives)
    return -sum(math.log(p) if y else math.log(1 - p) for y in labels) / n


def manifest_with(counts: dict[int, tuple[int, int]]) -> dict:
    """counts maps block number -> (rows, positives)."""
    return {"blocks": {str(k): {"rows": r, "positives": p} for k, (r, p) in counts.items()}}


COUNTS = {
    1: (100, 20),
    2: (100, 25),
    3: (100, 30),
    4: (100, 25),
    5: (100, 20),
    6: (100, 30),
    7: (100, 10),
}


# --- constant_logloss -----------------------------------------------------


@pytest.mark.parametrize(
    ("positives", "n", "p"), [(3, 10, 0.3), (3, 10, 0.5), (1, 7, 0.9), (0, 5, 0.2), (5, 5, 0.2)]
)
def test_constant_logloss_matches_row_by_row_sum(positives, n, p):
    assert constant_logloss(positives, n, p) == pytest.approx(row_by_row_logloss(positives, n, p))


def test_constant_logloss_is_minimized_at_empirical_rate():
    best = constant_logloss(25, 100, 0.25)

    assert best < constant_logloss(25, 100, 0.24)
    assert best < constant_logloss(25, 100, 0.26)


@pytest.mark.parametrize("p", [0.0, 1.0, -0.1, 1.1])
def test_constant_logloss_rejects_probability_outside_open_interval(p):
    with pytest.raises(ValueError):
        constant_logloss(1, 2, p)


@pytest.mark.parametrize(("positives", "n"), [(1, 0), (-1, 5), (6, 5)])
def test_constant_logloss_rejects_invalid_counts(positives, n):
    with pytest.raises(ValueError):
        constant_logloss(positives, n, 0.5)


# --- constant_baseline ----------------------------------------------------


def test_prediction_is_pooled_ctr_of_train_blocks():
    result = constant_baseline(manifest_with(COUNTS))

    assert result["prediction"] == pytest.approx(120 / 500)
    assert result["train_blocks"] == [1, 2, 3, 4, 5]


def test_prediction_ignores_validation_and_test_counts():
    changed = dict(COUNTS) | {6: (900, 800), 7: (900, 1)}

    original = constant_baseline(manifest_with(COUNTS))["prediction"]

    assert constant_baseline(manifest_with(changed))["prediction"] == original


def test_split_metrics():
    test = constant_baseline(manifest_with(COUNTS))["splits"]["test"]
    p = 120 / 500

    assert test["blocks"] == [7]
    assert test["rows"] == 100
    assert test["positives"] == 10
    assert test["ctr"] == pytest.approx(0.10)
    assert test["logloss"] == pytest.approx(row_by_row_logloss(10, 100, p))
    assert test["entropy"] == pytest.approx(row_by_row_logloss(10, 100, 0.10))
    assert test["pcoc"] == pytest.approx(p / 0.10)


def test_baseline_logloss_never_beats_the_split_entropy():
    result = constant_baseline(manifest_with(COUNTS))

    for split in result["splits"].values():
        assert split["logloss"] >= split["entropy"]


def test_train_split_is_exactly_calibrated():
    train = constant_baseline(manifest_with(COUNTS))["splits"]["train"]

    assert train["pcoc"] == pytest.approx(1.0)
    assert train["logloss"] == pytest.approx(train["entropy"])


def test_missing_train_block_raises():
    counts = {k: v for k, v in COUNTS.items() if k != 3}

    with pytest.raises(KeyError):
        constant_baseline(manifest_with(counts))


# --- CLI ------------------------------------------------------------------


def test_main_writes_report_from_manifest(tmp_path, monkeypatch, capsys):
    blocks = tmp_path / "blocks"
    blocks.mkdir()
    (blocks / "manifest.json").write_text(json.dumps(manifest_with(COUNTS)))
    out = tmp_path / "reports" / "baseline.json"
    monkeypatch.setattr("sys.argv", ["baseline", "--blocks", str(blocks), "--out", str(out)])

    main()

    assert json.loads(out.read_text()) == constant_baseline(manifest_with(COUNTS))
    assert "test" in capsys.readouterr().out
