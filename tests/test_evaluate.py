import json

import numpy as np
import pytest
import torch

from ctr.data.dataset import load_arrays, split_files
from ctr.data.encode import encode_blocks
from ctr.evaluate import evaluate_split, predict, resolve_device, train_ctr
from ctr.metrics import DEFAULT_N_BINS, summary
from ctr.models.lr import LogisticRegression


@pytest.fixture
def encoded(blocks_dir, tmp_path):
    out = tmp_path / "encoded"
    encode_blocks(blocks_dir, out, min_count=2)
    return out


def test_train_ctr_pools_training_blocks_only():
    manifest = {
        "blocks": {
            "1": {"rows": 10, "positives": 1},
            "2": {"rows": 30, "positives": 9},
            "6": {"rows": 100, "positives": 90},
        }
    }
    assert train_ctr(manifest, blocks=(1, 2)) == pytest.approx(10 / 40)


def test_predict_keeps_row_order_and_matches_sigmoid_of_logits(encoded):
    sizes = json.loads((encoded / "manifest.json").read_text())["field_sizes"]
    torch.manual_seed(0)
    model = LogisticRegression(sizes, base_rate=0.3)
    with torch.no_grad():
        model.weights.weight.normal_(std=0.3)

    files = split_files(encoded, (6,))
    y, p = predict(model, files, batch_size=97, device="cpu")
    ids, labels = load_arrays(files)
    with torch.no_grad():
        expected = torch.sigmoid(model(torch.from_numpy(ids).long())).double().numpy()

    assert y.tolist() == labels.tolist()
    assert p.dtype == np.float64
    np.testing.assert_allclose(p, expected, rtol=1e-6)


def test_predict_leaves_model_in_its_previous_mode(encoded):
    sizes = json.loads((encoded / "manifest.json").read_text())["field_sizes"]
    model = LogisticRegression(sizes)
    model.train()
    predict(model, split_files(encoded, (7,)), batch_size=128, device="cpu")
    assert model.training


def test_evaluate_split_reports_summary_and_calibration_bins():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.6, 5000)
    y = (rng.random(5000) < p).astype(np.int8)
    out = evaluate_split(y, p, base_rate=0.3)

    for key, value in summary(y, p, base_rate=0.3).items():
        assert out[key] == pytest.approx(value)
    bins = out["calibration_bins"]
    assert len(bins["count"]) == DEFAULT_N_BINS
    assert sum(bins["count"]) == 5000
    assert len(bins["mean_pred"]) == len(bins["mean_label"]) == DEFAULT_N_BINS


def test_resolve_device():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in {"cpu", "mps", "cuda"}


def test_predict_keeps_precision_for_confident_rows(encoded):
    # float32 sigmoid rounds 1 - p near p = 1; prior correction then magnifies that error.
    sizes = json.loads((encoded / "manifest.json").read_text())["field_sizes"]
    model = LogisticRegression(sizes)
    with torch.no_grad():
        model.bias.fill_(12.0)

    _, p = predict(model, split_files(encoded, (6,)), batch_size=128, device="cpu")
    np.testing.assert_allclose(1 - p, 1 / (1 + np.exp(12.0)), rtol=1e-6)
