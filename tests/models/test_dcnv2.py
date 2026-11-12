import math

import pytest
import torch

from ctr.models import build_model
from ctr.models.dcnv2 import CrossLayer, DCNv2

SIZES = [3, 5, 4]


def small(**kwargs) -> DCNv2:
    args = {"embed_dim": 4, "cross_layers": 2, "hidden": (8, 8), **kwargs}
    return DCNv2(SIZES, **args)


def test_forward_returns_one_logit_per_row():
    model = small()
    assert model(torch.tensor([[0, 1, 2], [2, 4, 3]])).shape == (2,)


def test_untrained_model_predicts_the_base_rate_for_every_row():
    # Output weights start at zero, so training starts from the constant baseline (ADR-010).
    torch.manual_seed(0)
    model = small(base_rate=0.25)
    out = model(torch.stack([torch.randint(0, s, (16,)) for s in SIZES], dim=1))

    assert torch.allclose(out, torch.full((16,), math.log(0.25 / 0.75)))


def test_cross_layer_formula():
    # x_{l+1} = x0 * (W x_l + b) + x_l
    layer = CrossLayer(3)
    with torch.no_grad():
        layer.linear.weight.copy_(torch.eye(3) * 2)
        layer.linear.bias.fill_(1.0)
    x0 = torch.tensor([[1.0, 2.0, 3.0]])
    xl = torch.tensor([[0.5, -1.0, 0.0]])

    expected = x0 * (2 * xl + 1) + xl
    assert torch.allclose(layer(x0, xl), expected)


def test_cross_layer_has_full_rank_weight():
    assert CrossLayer(6).linear.weight.shape == (6, 6)


def test_same_local_id_in_different_fields_uses_different_embeddings():
    model = small()
    a = model.embed(torch.tensor([[1, 0, 0]]))
    b = model.embed(torch.tensor([[0, 1, 0]]))

    assert not torch.allclose(a[:, :4], b[:, 4:8])


def test_embedding_output_is_fields_times_dim():
    assert small().embed(torch.tensor([[0, 0, 0]])).shape == (1, len(SIZES) * 4)


def test_seed_fixes_initialisation():
    torch.manual_seed(1)
    a = small().embedding.weight.clone()
    torch.manual_seed(1)
    b = small().embedding.weight.clone()
    torch.manual_seed(2)
    c = small().embedding.weight.clone()

    assert torch.equal(a, b) and not torch.equal(a, c)


def test_learns_an_interaction_lr_cannot():
    # Label = XOR of two fields' parity: no additive model separates it.
    torch.manual_seed(0)
    sizes = [2, 2, 3]
    ids = torch.stack([torch.randint(0, s, (2000,)) for s in sizes], dim=1)
    y = (ids[:, 0] ^ ids[:, 1]).float()

    def fit(model):
        opt = torch.optim.Adam(model.parameters(), lr=0.02)
        for _ in range(300):
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(ids), y)
            loss.backward()
            opt.step()
        return loss.item()

    dcn = fit(DCNv2(sizes, base_rate=0.5, embed_dim=4, cross_layers=2, hidden=(16,)))
    lr = fit(build_model("lr", sizes, base_rate=0.5))

    assert dcn < 0.1
    assert lr > 0.6


def test_works_without_deep_tower():
    model = small(hidden=())
    assert model(torch.tensor([[0, 1, 2]])).shape == (1,)


@pytest.mark.parametrize(
    "kwargs", [{"embed_dim": 0}, {"cross_layers": -1}, {"hidden": (8, 0)}, {"dropout": 1.0}]
)
def test_rejects_bad_hyperparameters(kwargs):
    with pytest.raises(ValueError):
        small(**kwargs)


@pytest.mark.parametrize("rate", [0.0, 1.0])
def test_rejects_base_rate_outside_open_unit_interval(rate):
    with pytest.raises(ValueError):
        small(base_rate=rate)


def test_build_model_passes_hyperparameters():
    model = build_model("dcnv2", SIZES, base_rate=0.3, embed_dim=4, cross_layers=1, hidden=[8])
    assert isinstance(model, DCNv2)
    assert len(model.cross) == 1
    assert model.embedding.weight.shape == (sum(SIZES), 4)
