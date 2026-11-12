import json

import numpy as np
import pytest

from ctr.calibration import sampled_ctr
from ctr.data.dataset import split_files
from ctr.data.encode import encode_blocks
from ctr.evaluate import train_ctr
from ctr.train import TrainConfig, load_config, run


@pytest.fixture
def encoded(blocks_dir, tmp_path):
    out = tmp_path / "encoded"
    encode_blocks(blocks_dir, out, min_count=2)
    return out


def make_config(tmp_path, encoded, **overrides) -> TrainConfig:
    values = {
        "encoded": encoded,
        "artifacts": tmp_path / "artifacts",
        "report": tmp_path / "reports" / "lr.json",
        "device": "cpu",
        "batch_size": 64,
        "lr": 0.02,
        "epochs": 3,
        "shards_per_chunk": 2,
        "eval_batch_size": 256,
        **overrides,
    }
    return TrainConfig(**values)


# --- load_config -----------------------------------------------------------


def test_load_config_reads_toml_sections(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text(
        """
seed = 3
device = "cpu"

[model]
name = "lr"

[data]
encoded = "data/encoded"
max_train_shards = 5
max_val_shards = 2
evaluate_test = false

[train]
batch_size = 1024
lr = 0.002
epochs = 2

[output]
artifacts = "artifacts/x"
report = "reports/x.json"
"""
    )
    cfg = load_config(path)

    assert cfg.seed == 3
    assert cfg.model == "lr"
    assert cfg.max_train_shards == 5
    assert cfg.max_val_shards == 2
    assert cfg.evaluate_test is False
    assert cfg.batch_size == 1024
    assert cfg.lr == pytest.approx(0.002)
    assert str(cfg.report) == "reports/x.json"
    assert cfg.patience == TrainConfig.patience  # default kept


def test_load_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text("[train]\nlearning_rate = 0.1\n")
    with pytest.raises(ValueError, match="learning_rate"):
        load_config(path)


# --- run -------------------------------------------------------------------


def test_run_learns_signal_and_writes_report(tmp_path, encoded):
    cfg = make_config(tmp_path, encoded)
    report = run(cfg)

    assert json.loads(cfg.report.read_text()) == report
    manifest = json.loads((encoded / "manifest.json").read_text())
    assert report["base_rate"] == pytest.approx(train_ctr(manifest))
    assert report["val"]["ne"] < 0.95
    assert report["test"]["ne"] < 0.95
    assert report["val"]["n"] == manifest["blocks"]["6"]["rows"]
    assert report["test"]["n"] == manifest["blocks"]["7"]["rows"]
    assert 1 <= len(report["history"]) <= cfg.epochs


def test_run_saves_model_and_predictions_in_row_order(tmp_path, encoded):
    cfg = make_config(tmp_path, encoded)
    report = run(cfg)

    assert (cfg.artifacts / "model.pt").exists()
    val = np.load(cfg.artifacts / "val_pred.npy")
    test = np.load(cfg.artifacts / "test_pred.npy")
    assert len(val) == report["val"]["n"]
    assert len(test) == report["test"]["n"]
    assert val.mean() == pytest.approx(report["val"]["mean_pred"], rel=1e-5)


def test_run_reports_the_best_epoch(tmp_path, encoded):
    report = run(make_config(tmp_path, encoded, epochs=4, patience=4))

    losses = [h["val"]["logloss"] for h in report["history"]]
    best = report["best_epoch"]
    assert losses[best - 1] == min(losses)
    assert report["val"]["logloss"] == pytest.approx(losses[best - 1])


def test_run_stops_after_patience_epochs_without_improvement(tmp_path, encoded):
    # A huge learning rate overfits from the first epoch on.
    report = run(make_config(tmp_path, encoded, lr=5.0, epochs=6, patience=1))
    losses = [h["val"]["logloss"] for h in report["history"]]

    assert len(losses) < 6
    assert losses[-1] >= min(losses)


def test_run_limits_shards_and_can_skip_test(tmp_path, encoded):
    cfg = make_config(tmp_path, encoded, max_train_shards=3, max_val_shards=1, evaluate_test=False)
    report = run(cfg)

    train_files = split_files(encoded, (1, 2, 3, 4, 5), max_shards=3)
    assert report["train_rows"] == 3 * 400
    assert report["train_files"] == [f"{f.parent.name}/{f.name}" for f in train_files]
    assert report["val"]["n"] == 400
    assert "test" not in report
    assert not (cfg.artifacts / "test_pred.npy").exists()


def test_run_is_deterministic_on_cpu(tmp_path, encoded):
    a = run(make_config(tmp_path / "a", encoded, epochs=1))
    b = run(make_config(tmp_path / "b", encoded, epochs=1))

    assert a["val"]["logloss"] == b["val"]["logloss"]
    np.testing.assert_array_equal(
        np.load(tmp_path / "a" / "artifacts" / "val_pred.npy"),
        np.load(tmp_path / "b" / "artifacts" / "val_pred.npy"),
    )


# --- negative downsampling (ADR-005, ADR-011) ------------------------------


def test_load_config_reads_negative_keep_rate(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text("[data]\nnegative_keep_rate = 0.25\n")
    assert load_config(path).negative_keep_rate == pytest.approx(0.25)
    assert TrainConfig().negative_keep_rate == 1.0


def test_untrained_downsampled_model_starts_at_sampled_ctr(tmp_path, encoded):
    # lr = 0 leaves the model at its initial bias: logit of the sampled training CTR.
    # Prior correction maps that back to the training-block CTR.
    report = run(make_config(tmp_path, encoded, lr=0.0, epochs=1, negative_keep_rate=0.25))

    assert report["train_ctr_sampled"] == pytest.approx(sampled_ctr(report["base_rate"], 0.25))
    assert report["val"]["mean_pred"] == pytest.approx(report["train_ctr_sampled"])
    assert report["val_corrected"]["mean_pred"] == pytest.approx(report["base_rate"])
    assert report["val_corrected"]["ne"] == pytest.approx(1.0, abs=0.02)


def test_downsampled_run_trains_on_fewer_rows_but_evaluates_on_all(tmp_path, encoded):
    cfg = make_config(tmp_path, encoded, negative_keep_rate=0.25)
    report = run(cfg)
    manifest = json.loads((encoded / "manifest.json").read_text())

    assert report["negative_keep_rate"] == 0.25
    assert report["train_positives"] < report["train_rows_sampled"] < report["train_rows"]
    for split, block in (("val", "6"), ("test", "7")):
        for key in (split, f"{split}_corrected"):
            assert report[key]["n"] == manifest["blocks"][block]["rows"]
            assert report[key]["positives"] == manifest["blocks"][block]["positives"]


def test_prior_correction_fixes_the_mean_but_not_the_ranking(tmp_path, encoded):
    report = run(make_config(tmp_path, encoded, negative_keep_rate=0.25))

    for split in ("val", "test"):
        raw, fixed = report[split], report[f"{split}_corrected"]
        assert raw["pcoc"] > 1.3  # sampled_ctr(c, 0.25) / c is ~1.7 on this data
        assert abs(fixed["pcoc"] - 1) < abs(raw["pcoc"] - 1)
        assert fixed["logloss"] < raw["logloss"]
        assert fixed["auc"] == pytest.approx(raw["auc"], abs=1e-12)

    val = np.load(tmp_path / "artifacts" / "val_pred.npy")
    assert val.mean() == pytest.approx(report["val"]["mean_pred"], rel=1e-5)


def test_downsampled_run_keeps_the_epoch_with_best_corrected_val_logloss(tmp_path, encoded):
    report = run(make_config(tmp_path, encoded, negative_keep_rate=0.25, epochs=4, patience=4))

    losses = [h["val_corrected"]["logloss"] for h in report["history"]]
    assert losses[report["best_epoch"] - 1] == min(losses)
    assert report["val_corrected"]["logloss"] == pytest.approx(min(losses))


def test_keep_rate_one_reports_identical_raw_and_corrected(tmp_path, encoded):
    report = run(make_config(tmp_path, encoded, epochs=1))

    assert report["negative_keep_rate"] == 1.0
    assert report["train_rows_sampled"] == report["train_rows"]
    assert report["val_corrected"] == report["val"]

