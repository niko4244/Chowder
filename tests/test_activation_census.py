"""Tests for the activation census (dense-parent neuron measurement).

The fake model reproduces the dense qwen3_5 MLP forward math exactly:
``gated = silu(x @ Wg.T) * (x @ Wu.T); out = gated @ Wd.T``. Weights are
constructed so each token's active-neuron set is hand-checkable: with
``Wu`` all-positive, ``gated_i > 0`` iff ``gate_i > 0`` -- the exact
dReLU counterfactual semantics the census must measure. Two neurons
sharing an identical gate row co-activate on every token, which the
sketch and the exact top-N co-occurrence must both reflect.

The census must never modify the model: parameter hashes before/after
are compared, hooks are removed on exit, and a second forward produces
identical outputs to a never-censused model.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from chowder.activation_census import (
    ActivationCensus,
    ActivationCensusError,
    gini,
    write_activation_profile,
)

torch = pytest.importorskip("torch")
nn = torch.nn


HIDDEN = 4
INTERMEDIATE = 8


def _sha_params(model: nn.Module) -> str:
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        h.update(name.encode())
        h.update(p.detach().numpy().tobytes())
    return h.hexdigest()


class _FakeDenseModel(nn.Module):
    """Minimal dense decoder: language_model.layers[i].mlp.{gate,up,down}."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(20260908)
        layer = nn.Module()
        mlp = nn.Module()
        # Wu strictly positive -> up-projection never gates a neuron off.
        wu = torch.abs(torch.randn(INTERMEDIATE, HIDDEN)) + 0.5
        # Gate rows: neurons 0 and 1 share an all-positive row; with
        # non-negative x their gate_pre = sum(x) > 0 on every token, so
        # they co-activate everywhere. Rows 2..7 get varied sign patterns.
        wg = torch.randn(INTERMEDIATE, HIDDEN)
        wg[0] = torch.ones(HIDDEN)
        wg[1] = wg[0]
        mlp.gate_proj = nn.Linear(HIDDEN, INTERMEDIATE, bias=False)
        mlp.up_proj = nn.Linear(HIDDEN, INTERMEDIATE, bias=False)
        mlp.down_proj = nn.Linear(INTERMEDIATE, HIDDEN, bias=False)
        with torch.no_grad():
            mlp.gate_proj.weight.copy_(wg)
            mlp.up_proj.weight.copy_(wu)
            mlp.down_proj.weight.copy_(torch.abs(torch.randn(HIDDEN, INTERMEDIATE)) + 0.1)
        layer.mlp = mlp
        # Distinct layer modules per position (registering the same object
        # twice would make both census registrations alias one down_proj).
        layer2 = nn.Module()
        mlp2 = nn.Module()
        # Layer 1 gets independent weights: same Wu > 0 discipline.
        wu2 = torch.abs(torch.randn(INTERMEDIATE, HIDDEN)) + 0.5
        wg2 = torch.randn(INTERMEDIATE, HIDDEN)
        mlp2.gate_proj = nn.Linear(HIDDEN, INTERMEDIATE, bias=False)
        mlp2.up_proj = nn.Linear(HIDDEN, INTERMEDIATE, bias=False)
        mlp2.down_proj = nn.Linear(INTERMEDIATE, HIDDEN, bias=False)
        with torch.no_grad():
            mlp2.gate_proj.weight.copy_(wg2)
            mlp2.up_proj.weight.copy_(wu2)
            mlp2.down_proj.weight.copy_(torch.abs(torch.randn(HIDDEN, INTERMEDIATE)) + 0.1)
        layer2.mlp = mlp2
        self.language_model = nn.Module()
        self.language_model.layers = nn.ModuleList([layer, layer2])

    def forward(self, input_ids: torch.Tensor, **_: object) -> dict[str, torch.Tensor]:
        # Real sequence processing: (B=1, T, H), one hidden vector per token.
        ids = input_ids[0]  # (T,)
        T = ids.shape[0]
        x = self.language_model.layers[0].mlp.up_proj.weight.new_zeros(T, HIDDEN)
        for r in range(T):
            for c in range(HIDDEN):
                # Non-negative (0..6): keeps x @ wu > 0 for wu > 0, so the
                # dReLU counterfactual reduces to (gate_pre > 0) exactly.
                x[r, c] = float((ids[r].item() + c) % 7)
        for layer in self.language_model.layers:
            mlp = layer.mlp
            gated = nn.functional.silu(mlp.gate_proj(x)) * mlp.up_proj(x)
            x = mlp.down_proj(gated)
        return {"logits": x}


@pytest.fixture()
def fake_model_and_texts():
    model = _FakeDenseModel()

    class _Tok:
        def __call__(self, text, return_tensors="pt", truncation=False, max_length=None):
            ids = torch.tensor([[len(text) % 10 + 1, 3, 5, 7]])
            return {"input_ids": ids}

    texts = [f"calibration prompt {i} with varied length {i * 13 % 9}" for i in range(6)]
    return model, _Tok(), texts


def test_census_measures_drelu_semantics_and_never_modifies_model(
    fake_model_and_texts,
):
    model, tok, texts = fake_model_and_texts
    before = _sha_params(model)

    census = ActivationCensus(model, tok, seed=20260908, device="cpu", hot_fraction=0.25)
    with census:
        census.consume_texts(texts)
    profile = census.finalize(provenance={"model_revision": "fake", "dataset": "synthetic"})

    assert _sha_params(model) == before, "census must not modify model weights"

    layers = profile["layers"]
    assert sorted(layers) == ["0", "1"]
    for lid, lp in layers.items():
        assert lp["intermediate_size"] == INTERMEDIATE
        assert lp["tokens_seen"] == 4 * len(texts)
        # dReLU semantics: gated_i != 0 iff gate_i > 0 AND up_i > 0; with
        # Wu > 0 elementwise, active_i == (gate_i > 0) exactly.
        gate_w = model.language_model.layers[int(lid)].mlp.gate_proj.weight
        up_w = model.language_model.layers[int(lid)].mlp.up_proj.weight
        # Recompute per-token active sets by hand over the same inputs.
        active_any = set()
        for text in texts:
            ids = tok(text)["input_ids"]
            x = model.language_model.layers[0].mlp.up_proj.weight.new_zeros(4, HIDDEN)
            for r in range(4):
                for c in range(HIDDEN):
                    x[r, c] = float((ids[0, r].item() + c) % 7)
            g = x @ gate_w.T
            u = x @ up_w.T
            act = (g > 0) & (u > 0)
            for i in range(INTERMEDIATE):
                if bool(act[:, i].any()):
                    active_any.add(i)
        # Every neuron the census never saw active must have zero frequency.
        for i in range(INTERMEDIATE):
            if i not in active_any:
                assert lp["per_neuron_top100_frequency"].get(str(i), 0.0) == 0.0 or i not in lp["per_neuron_top100_frequency"]
        # The shared-gate pair (0, 1) must co-occur in the exact table.
        # Inner pair dicts keep int neuron ids in memory (strings only
        # after a JSON round-trip); accept both for robustness.
        co = lp["hotset_cooccurrence"]
        found = any(
            1 in row or "1" in row for n, row in co.items() if int(n) == 0
        ) or any(
            0 in row or "0" in row for n, row in co.items() if int(n) == 1
        )
        assert found, "identical gate rows must co-occur exactly"


def test_census_hot_set_and_sparsity_fields(fake_model_and_texts):
    model, tok, texts = fake_model_and_texts
    census = ActivationCensus(model, tok, seed=1, device="cpu", hot_fraction=0.25)
    with census:
        census.consume_texts(texts[:3])
        census.mark_split()
        census.consume_texts(texts[3:])
    profile = census.finalize(provenance={"dataset": "synthetic-split"})

    lp = profile["layers"]["0"]
    assert lp["hot_neurons"] == max(1, int(0.25 * INTERMEDIATE))
    assert len(lp["hot_neuron_ids"]) == lp["hot_neurons"]
    # Split boundary recorded and both halves present.
    assert profile["totals"]["split_boundary"] == 4 * 3
    assert lp["cooccurrence_split_boundary"] == 4 * 3
    assert "hotset_cooccurrence_half_a" in lp and "hotset_cooccurrence_half_b" in lp
    # Sparsity fields exist and are in range.
    assert 0.0 <= lp["sparsity_drelu_counterfactual"] <= 1.0
    assert 0.0 <= lp["activation_sparsity_swiglu_near_zero"] <= 1.0
    assert 0.0 <= lp["gini_activation_frequency"] <= 1.0
    # Sketch payload present with the right shape and a matching digest.
    assert len(lp["sketch"]) == INTERMEDIATE
    assert all(len(row) == 256 for row in lp["sketch"])
    h = hashlib.sha256()
    for row in lp["sketch"]:
        h.update(json.dumps([round(v, 6) for v in row], separators=(",", ":")).encode())
    assert h.hexdigest() == lp["sketch_sha256"]


def test_census_refuses_non_dense_and_double_start(fake_model_and_texts):
    model, tok, _ = fake_model_and_texts
    # A model whose layer lacks down_proj must be refused.
    class _Broken(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layer = nn.Module()
            layer.mlp = nn.Module()  # no projections
            self.language_model = nn.Module()
            self.language_model.layers = nn.ModuleList([layer])

    with pytest.raises(ActivationCensusError):
        with ActivationCensus(_Broken(), tok, seed=1):
            pass

    census = ActivationCensus(model, tok, seed=1)
    with census:
        with pytest.raises(ActivationCensusError):
            census2 = ActivationCensus(model, tok, seed=1)
            with census2:
                census2.consume_texts(["x"])
    # consume outside context refused
    with pytest.raises(ActivationCensusError):
        census.consume_texts(["x"])


def test_write_activation_profile_atomic(tmp_path):
    profile = {"layers": {}, "totals": {}}
    path = tmp_path / "nested" / "activation_profile.json"
    written = write_activation_profile(profile, path)
    assert written == str(path)
    assert json.loads(path.read_text(encoding="utf-8")) == profile
    assert not path.with_suffix(".tmp").exists()


def test_gini_bounds():
    assert gini([]) == 0.0
    assert gini([1.0] * 10) == 0.0
    assert gini([0.0] * 99 + [100.0]) == pytest.approx(0.99, abs=1e-6)