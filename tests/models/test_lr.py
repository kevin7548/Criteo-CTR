import math

import pytest
import torch

from ctr.models import build_model
from ctr.models.lr import LogisticRegression

SIZES = [3, 5, 4]


def test_forward_returns_one_logit_per_row():
    model = LogisticRegression(SIZES)
    ids = torch.tensor([[0, 1, 2], [2, 4, 3]])

    assert model(ids).shape == (2,)


def test_initial_logit_is_logit_of_base_rate():
    model = LogisticRegression(SIZES, base_rate=0.25)
    out = model(torch.tensor([[0, 0, 0], [2, 4, 3]]))

    assert torch.allclose(out, torch.full((2,), math.log(0.25 / 0.75)))


def test_initial_logit_without_base_rate_is_zero():
    model = LogisticRegression(SIZES)
    assert model(torch.tensor([[1, 1, 1]])).item() == 0.0


def test_logit_is_bias_plus_one_weight_per_field():
    model = LogisticRegression(SIZES)
    with torch.no_grad():
        model.weights.weight.copy_(torch.arange(sum(SIZES), dtype=torch.float32).unsqueeze(1))
        model.bias.fill_(0.5)

    # Field offsets are 0, 3, 8: global ids 1, 3 + 4, 8 + 2.
    out = model(torch.tensor([[1, 4, 2]]))
    assert out.item() == pytest.approx(0.5 + 1 + 7 + 10)


def test_same_local_id_in_different_fields_uses_different_weights():
    model = LogisticRegression(SIZES)
    with torch.no_grad():
        model.weights.weight.normal_()
    a = model(torch.tensor([[1, 0, 0]]))
    b = model(torch.tensor([[0, 1, 0]]))
    c = model(torch.tensor([[0, 0, 0]]))

    assert (a - c).item() != pytest.approx((b - c).item())


def test_only_looked_up_weights_receive_gradient():
    model = LogisticRegression(SIZES)
    model(torch.tensor([[1, 4, 0]])).sum().backward()
    grad = model.weights.weight.grad.squeeze(1)

    touched = {1, 3 + 4, 8 + 0}
    assert set(torch.nonzero(grad).flatten().tolist()) == touched


def test_fits_a_one_field_signal():
    torch.manual_seed(0)
    ids = torch.stack([torch.randint(0, s, (2000,)) for s in SIZES], dim=1)
    y = (ids[:, 1] < 2).float()
    model = LogisticRegression(SIZES, base_rate=0.5)
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    for _ in range(100):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(ids), y)
        loss.backward()
        opt.step()

    assert loss.item() < 0.05


@pytest.mark.parametrize("bad", [[], [3, 0]])
def test_rejects_empty_or_zero_sized_fields(bad):
    with pytest.raises(ValueError):
        LogisticRegression(bad)


@pytest.mark.parametrize("rate", [0.0, 1.0, -0.1])
def test_rejects_base_rate_outside_open_unit_interval(rate):
    with pytest.raises(ValueError):
        LogisticRegression(SIZES, base_rate=rate)


def test_build_model_by_name():
    model = build_model("lr", SIZES, base_rate=0.3)
    assert isinstance(model, LogisticRegression)
    with pytest.raises(ValueError):
        build_model("nope", SIZES)
