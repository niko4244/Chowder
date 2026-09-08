"""Sidecar-path tests (large intermediate sizes, real-parent scale).

1. A census over a layer with I > INLINE_SKETCH_MAX (constant
   monkeypatched small) must write per-layer float64 .npy sidecars whose
   digest matches the profile, and must NOT inline the sketch.
2. make_grouping_sketch_cluster on a sidecar layer must take the numpy
   path and recover planted round-robin blocks exactly like the inline
   path.
"""

from __future__ import annotations

import hashlib
import json
import random

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
nn = torch.nn

from chowder import activation_census as ac
from chowder.activation_experiments import (
    make_grouping_sketch_cluster,
    run_grouping_comparison,
)

HIDDEN = 4
INTERMEDIATE = 8


class _Tiny(nn.Module):
    """Same shape discipline as the census fixture, one layer."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(11)
        layer = nn.Module()
        mlp = nn.Module()
        wg = torch.randn(INTERMEDIATE, HIDDEN)
        wg[0] = torch.ones(HIDDEN)
        wg[1] = wg[0]
        mlp.gate_proj = nn.Linear(HIDDEN, INTERMEDIATE, bias=False)
        mlp.up_proj = nn.Linear(HIDDEN, INTERMEDIATE, bias=False)
        mlp.down_proj = nn.Linear(INTERMEDIATE, HIDDEN, bias=False)
        with torch.no_grad():
            mlp.gate_proj.weight.copy_(wg)
            mlp.up_proj.weight.copy_(torch.abs(torch.randn(INTERMEDIATE, HIDDEN)) + 0.5)
            mlp.down_proj.weight.copy_(torch.abs(torch.randn(HIDDEN, INTERMEDIATE)) + 0.1)
        layer.mlp = mlp
        self.language_model = nn.Module()
        self.language_model.layers = nn.ModuleList([layer])

    def forward(self, input_ids: torch.Tensor, **_: object) -> dict[str, torch.Tensor]:
        ids = input_ids[0]
        T = ids.shape[0]
        x = self.language_model.layers[0].mlp.up_proj.weight.new_zeros(T, HIDDEN)
        for r in range(T):
            for c in range(HIDDEN):
                x[r, c] = float((ids[r].item() + c) % 7)
        mlp = self.language_model.layers[0].mlp
        gated = nn.functional.silu(mlp.gate_proj(x)) * mlp.up_proj(x)
        return {"logits": mlp.down_proj(gated)}


def test_large_i_census_writes_verified_sidecar(tmp_path, monkeypatch):
    monkeypatch.setattr(ac, "INLINE_SKETCH_MAX", 4)  # force sidecar at I=8

    class _Tok:
        def __call__(self, text, return_tensors="pt", truncation=False, max_length=None):
            return {"input_ids": torch.tensor([[len(text) % 10 + 1, 3, 5, 7]])}

    model = _Tiny()
    census = ac.ActivationCensus(model, _Tok(), seed=5, device="cpu")
    with census:
        census.consume_texts(["alpha passage", "beta passage", "gamma passage"])
    provenance = {"output_dir": str(tmp_path), "dataset": "sidecar-test"}
    profile = census.finalize(provenance=provenance)

    lp = profile["layers"]["0"]
    assert lp["sketch_inline"] is False
    assert lp["sketch_shape"] == [INTERMEDIATE, ac.SKETCH_DIM]
    assert "sketch" not in lp
    sidecar = tmp_path / "sketch_layer_0.f64.npy"
    assert sidecar.is_file()
    arr = np.load(sidecar, allow_pickle=False)
    assert arr.shape == (INTERMEDIATE, ac.SKETCH_DIM)
    h = hashlib.sha256()
    h.update(arr.astype("<f8").tobytes())
    assert h.hexdigest() == lp["sketch_sha256"]
    # The rounded-5 inline digest path must NOT have run; sidecar digest
    # is over raw f64 bytes (rounding would change them).


def test_numpy_clustering_recovers_planted_sidecar_blocks(tmp_path):
    """Sidecar layer + planted round-robin blocks -> numpy path recovers them."""
    # pytest rootdir imports: tests share a conftest-less namespace via
    # rootdir insertion; import by file path is the portable route.
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "t_activation_experiments",
        Path(__file__).parent / "test_activation_experiments.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _synthetic_layer = mod._synthetic_layer

    layer = _synthetic_layer()
    mat = np.array(layer.pop("sketch"), dtype="<f8")
    sidecar = tmp_path / "sketch_layer_0.f64.npy"
    np.save(sidecar, mat, allow_pickle=False)
    layer["sketch_inline"] = False
    layer["sketch_dtype"] = "<f8"
    layer["sketch_shape"] = [mat.shape[0], mat.shape[1]]
    layer["sketch_sidecar"] = str(sidecar)
    h = hashlib.sha256()
    h.update(mat.astype("<f8").tobytes())
    layer["sketch_sha256"] = h.hexdigest()

    profile = {
        "census_version": 1,
        "layers": {"0": layer},
        "totals": {"num_layers": 1, "split_boundary": 250},
    }
    result = run_grouping_comparison(profile, num_experts=4, layer_subset=[0])
    verdict = result["verdict"]["per_grouping"]["sketch-cluster"]
    assert verdict["passes"], f"planted sidecar structure must be found: {verdict}"

    assignment = make_grouping_sketch_cluster(4)(layer["intermediate_size"], layer, random.Random(0))
    labels_b0 = {assignment[i] for i in range(0, layer["intermediate_size"], 4)}
    assert len(labels_b0) == 1, f"sidecar block 0 scattered: {labels_b0}"


def test_sidecar_digest_mismatch_refused(tmp_path):
    from chowder.activation_experiments import ActivationExperimentError, _sketch_matrix

    mat = np.random.default_rng(0).standard_normal((16, 8)).astype("<f8")
    sidecar = tmp_path / "sketch.f64.npy"
    np.save(sidecar, mat, allow_pickle=False)
    layer = {
        "sketch_inline": False,
        "sketch_shape": [16, 8],
        "sketch_sidecar": str(sidecar),
        "sketch_sha256": "0" * 64,  # wrong on purpose
    }
    with pytest.raises(ActivationExperimentError, match="digest mismatch"):
        _sketch_matrix(layer)
