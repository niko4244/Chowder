"""The training-leg routing census: rung 4's T4a producer.

T4a thresholds ``dead_experts_after < 428``, so the number must come from a
declared, reproducible measurement rather than from whatever corpus happened to
be loaded. These tests pin the properties that make the claim honest:

- the census basis is always named, because a census over the *training* corpus
  flatters the router and is not comparable to a held-out one;
- a census corpus is hash-checked, so the measured basis cannot drift;
- the census runs in eval mode, so dropout cannot make the same trained router
  report different expert usage run to run;
- dead experts are counted from real top-1 routing logits.

They are the producer-side counterpart to the judge's T4a check.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from chowder.backends.router_healing import RouterHealingRunSpec
from chowder.backends.router_healing_worker import (
    _census_blocks,
    _census_metrics,
    _routing_census,
)


# --- a tiny router whose gate logits are deterministic ----------------------


class _TinyRouter(torch.nn.Module):
    """Two ``...mlp.gate`` modules, each emitting real routing logits."""

    def __init__(self, layers: int = 2, experts: int = 4, width: int = 8) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for index in range(layers):
            layer = torch.nn.Module()
            layer.mlp = torch.nn.Module()
            torch.manual_seed(index)
            layer.mlp.gate = torch.nn.Linear(width, experts, bias=False)
            self.layers.append(layer)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        # The signal varies per token so different rows can route differently.
        base = input_ids.to(torch.float32).unsqueeze(-1)
        x = base.repeat(1, 1, 8) + torch.arange(8, dtype=torch.float32)
        for layer in self.layers:
            layer.mlp.gate(x)
        return SimpleNamespace(loss=None)


def _spec(**overrides) -> RouterHealingRunSpec:
    base = {
        "base_model_dir": "F:/models/base",
        "base_content_sha256": "a" * 64,
        "corpus_path": "F:/data/corpus.txt",
        "corpus_sha256": "b" * 64,
        "output_dir": "F:/out",
        "max_steps": 4,
        "learning_rate": 0.05,
        "seq_len": 4,
        "batch_size": 1,
        "seed": 1,
        "probe_window": 2,
        "max_tokens": 16,
    }
    base.update(overrides)
    return RouterHealingRunSpec(**base)


# --- the spec refuses a half-declared measurement basis ---------------------


def test_a_census_corpus_without_its_hash_is_refused():
    """A basis nobody can check is not a basis."""
    with pytest.raises(ValueError, match="must be declared together"):
        _spec(census_corpus_path="F:/data/holdout.txt")


def test_a_census_hash_without_its_corpus_is_refused():
    with pytest.raises(ValueError, match="must be declared together"):
        _spec(census_corpus_sha256="c" * 64)


def test_a_non_positive_census_block_count_is_refused():
    with pytest.raises(ValueError, match="census_blocks must be a positive integer"):
        _spec(census_blocks=0)


def test_a_census_corpus_is_accepted_only_with_a_sha256():
    with pytest.raises(ValueError, match="sha256 hex digest"):
        _spec(census_corpus_path="F:/data/holdout.txt", census_corpus_sha256="short")


# --- the basis is named, and a declared corpus is hash-checked --------------


def test_without_a_declared_corpus_the_census_says_it_used_training_data():
    """The fallback must be labelled, because it is the flattering basis."""
    training = [[1, 2, 3, 4], [5, 6, 7, 8]]
    blocks, basis, sha = _census_blocks(_spec(), SimpleNamespace(), training)
    assert blocks == training
    assert basis == "training-corpus"
    assert sha == "b" * 64


def test_a_declared_census_corpus_is_read_hashed_and_packed(tmp_path: Path):
    corpus = tmp_path / "holdout.txt"
    corpus.write_text("one two three four five six seven eight\n", encoding="utf-8")
    digest = hashlib.sha256(corpus.read_bytes()).hexdigest()

    class _Tokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": list(range(len(text.split())))}

    spec = _spec(
        census_corpus_path=str(corpus),
        census_corpus_sha256=digest,
        seq_len=4,
    )
    blocks, basis, sha = _census_blocks(spec, _Tokenizer(), [[9, 9, 9, 9]])
    assert basis == "declared-census-corpus"
    assert sha == digest
    assert blocks == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_a_drifted_census_corpus_is_refused(tmp_path: Path):
    """The measured basis must be the frozen one, byte for byte."""
    corpus = tmp_path / "holdout.txt"
    corpus.write_text("alpha beta gamma delta\n", encoding="utf-8")
    spec = _spec(
        census_corpus_path=str(corpus),
        census_corpus_sha256="d" * 64,
        seq_len=4,
    )

    class _Tokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [0, 1, 2, 3]}

    with pytest.raises(RuntimeError, match="census corpus hash mismatch"):
        _census_blocks(spec, _Tokenizer(), [])


# --- the census itself reads real routing logits ----------------------------


def test_the_census_tallies_top1_experts_per_gate():
    model = _TinyRouter(layers=2, experts=4)
    counts = _routing_census(
        torch, model, [[0, 1, 2, 3], [4, 5, 6, 7]], torch.device("cpu"), limit=2
    )
    assert set(counts) == {"layers.0.mlp.gate", "layers.1.mlp.gate"}
    for layer_counts in counts.values():
        assert len(layer_counts) == 4
        # routing is tallied per *token*: two blocks of four tokens, batch 1
        assert sum(layer_counts) == 8


def test_the_census_restores_the_model_training_mode():
    """A measurement must not silently change how the run trains."""
    model = _TinyRouter()
    model.train()
    _routing_census(torch, model, [[0, 1, 2, 3]], torch.device("cpu"), limit=1)
    assert model.training is True

    model.eval()
    _routing_census(torch, model, [[0, 1, 2, 3]], torch.device("cpu"), limit=1)
    assert model.training is False


def test_the_census_bounds_its_work_by_the_declared_block_limit():
    """Cost is declared, not 'however long the corpus happens to be'."""
    model = _TinyRouter()
    counts = _routing_census(
        torch, model, [[0, 1, 2, 3]] * 10, torch.device("cpu"), limit=3
    )
    # three of the ten blocks, four tokens each
    assert sum(counts["layers.0.mlp.gate"]) == 12


def test_the_census_metrics_count_dead_experts_and_name_the_basis():
    counts = {
        "layers.0.mlp.gate": [5, 0, 0, 3],
        "layers.1.mlp.gate": [0, 0, 0, 0],
    }
    metrics = _census_metrics(
        counts,
        basis="declared-census-corpus",
        corpus_sha256="e" * 64,
        blocks_used=2,
        blocks_available=7,
    )
    assert metrics["dead_experts_after"] == 6
    assert metrics["expert_slots"] == 8
    assert metrics["census_basis"] == "declared-census-corpus"
    assert metrics["census_corpus_sha256"] == "e" * 64
    assert metrics["census_blocks_used"] == 2
    assert metrics["census_blocks_available"] == 7
    assert metrics["dead_experts_per_layer"] == {
        "layers.0.mlp.gate": 2,
        "layers.1.mlp.gate": 4,
    }


def test_a_fully_collapsed_router_reports_every_slot_dead():
    """The pathological case T4a exists to catch."""
    counts = {"layers.0.mlp.gate": [9, 0, 0, 0], "layers.1.mlp.gate": [8, 0, 0, 0]}
    metrics = _census_metrics(
        counts,
        basis="training-corpus",
        corpus_sha256="f" * 64,
        blocks_used=1,
        blocks_available=1,
    )
    assert metrics["dead_experts_after"] == 6
    assert metrics["expert_slots"] == 8
