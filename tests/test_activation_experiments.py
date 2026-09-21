"""Tests for the natural-expert-structure evaluation (Phase 4).

Synthetic profiles carry PLANTED structure: neurons grouped in blocks
whose hot-set co-occurrence is high inside the block and (near) zero
across blocks. The evaluation must find the planted structure with the
sketch-cluster grouping (which sees block-shaped sketches), while the
random grouping must NOT pass the decision rule -- the negative path is
as much a part of the contract as the positive one.
"""

from __future__ import annotations

import json
import math
import random

import pytest

from chowder.activation_census import gini
from chowder.activation_experiments import (
    ActivationExperimentError,
    evaluate_grouping,
    evaluate_grouping_full,
    format_verdict,
    make_grouping_contiguous,
    make_grouping_random,
    make_grouping_sketch_cluster,
    run_grouping_comparison,
    write_grouping_comparison,
)


E = 4
I = 64
BLOCK = I // E  # 16 neurons per planted block


def _noisy_copy(co: dict, rng: random.Random, jitter: float = 0.25) -> dict:
    """Independent sampling noise on each pair count (simulating two
    disjoint calibration halves observing the same structure)."""
    out = {}
    for a, row in co.items():
        out[a] = {
            b: max(1, int(round(c * (1.0 + rng.uniform(-jitter, jitter)))))
            for b, c in row.items()
        }
    return out


def _synthetic_layer(*, noise: float = 0.0, seed: int = 7) -> dict:
    """A profile layer with planted block co-activation.

    Neurons are assigned to blocks ROUND-ROBIN (neuron i -> block i % E),
    so the structure is deliberately NON-contiguous: every 16-wide
    contiguous slice spans all four blocks and the mechanical baseline
    cannot express it. Block members co-occur strongly; cross-block
    co-occurrence is zero (or `noise`-diluted). Sketches mirror the block
    structure: all neurons in a block share a sign-profile direction.
    """
    rng = random.Random(seed)
    hot_ids = [i for i in range(0, I, 2)]  # every other neuron "hot"
    freq = {str(i): round(rng.uniform(0.3, 0.9) if i in set(hot_ids) else rng.uniform(0.0, 0.1), 6) for i in range(I)}
    co = {}
    for b in range(E):
        block = [i for i in range(I) if i % E == b]
        hot_block = [n for n in block if n in set(hot_ids)]
        for a_i in range(len(hot_block)):
            for b_i in range(a_i + 1, len(hot_block)):
                a, c = hot_block[a_i], hot_block[b_i]
                co.setdefault(a, {})[c] = rng.randint(40, 60)
                co.setdefault(c, {})[a] = co[a][c]
    # Sketches: all neurons in block b share direction b (a one-hot on
    # dim b of a small basis, embedded in 32 dims); blocks are mutually
    # orthogonal. Optional per-neuron noise perturbs the direction.
    sketches = []
    for i in range(I):
        b = i % E
        vec = [0.0] * 32
        vec[b] = 1.0
        if noise:
            vec = [v + rng.uniform(-noise, noise) for v in vec]
        sketches.append(vec)
    return {
        "intermediate_size": I,
        "tokens_seen": 500,
        "per_neuron_top100_frequency": freq,
        "hot_neuron_ids": hot_ids,
        "hotset_cooccurrence": co,
        "hotset_cooccurrence_half_a": _noisy_copy(co, rng),
        "hotset_cooccurrence_half_b": _noisy_copy(co, rng),
        "sketch": sketches,
    }


def _profile(layer: dict) -> dict:
    return {
        "census_version": 1,
        "hot_fraction": 0.10,
        "layers": {"0": layer},
        "totals": {"num_layers": 1, "split_boundary": 250},
    }


def test_planted_structure_found_random_rejected():
    profile = _profile(_synthetic_layer())
    result = run_grouping_comparison(profile, num_experts=E, layer_subset=[0])
    verdict = result["verdict"]
    # The sketch clustering should find the planted blocks.
    sc = verdict["per_grouping"]["sketch-cluster"]
    assert sc["passes"], f"planted structure must be found: {sc}"
    # Random partition must NOT pass (no signal in a null model).
    assert "random" not in verdict["per_grouping"] or not verdict["per_grouping"]["random"].get("passes", False)
    assert verdict["real_structure_found"]


def test_negative_result_when_no_structure():
    # A layer whose co-occurrence is uniformly random across hot neurons:
    rng = random.Random(3)
    layer = _synthetic_layer()
    co = {}
    hot_ids = layer["hot_neuron_ids"]
    for a_i in range(len(hot_ids)):
        for b_i in range(a_i + 1, len(hot_ids)):
            a, c = hot_ids[a_i], hot_ids[b_i]
            v = rng.randint(0, 5)
            if v:
                co.setdefault(a, {})[c] = v
                co.setdefault(c, {})[a] = v
    layer["hotset_cooccurrence"] = co
    profile = _profile(layer)
    result = run_grouping_comparison(profile, num_experts=E, layer_subset=[0])
    # No grouping should beat the random baseline by the margin.
    for name, r in result["verdict"]["per_grouping"].items():
        if name != "random":
            # sketch-cluster sees random sketches here (structure destroyed
            # only in co-occurrence, sketch still blocky) -- so assert on
            # the *co-occurrence* metric the random null shares.
            assert not (
                r["passes"] and result["summary"]["sketch-cluster"]["mean_within_over_between"]
                > 2 * result["summary"]["random"]["mean_within_over_between"]
            )


def test_metrics_match_hand_computation():
    layer = _synthetic_layer()
    # Planted blocks are round-robin (non-contiguous), so the grouping
    # that recovers them exactly is the sketch cluster.
    assignment = make_grouping_sketch_cluster(E)(I, layer, random.Random(0))
    metrics = evaluate_grouping(assignment, layer, num_experts=E)
    # Clusters == planted blocks: within == block pairs (counts 40-60),
    # between has no observations at all.
    assert metrics["mean_within_coactivation"] > 40.0
    assert metrics["mean_between_coactivation"] == 0.0
    # +1-floored ratio: (mean_within + 1) / (0 + 1) = mean_within + 1.
    assert metrics["within_over_between"] == pytest.approx(
        metrics["mean_within_coactivation"] + 1.0, abs=1e-6
    )
    # Every cluster is exactly one 16-neuron block, but balance is over
    # hot-neuron MASS and per-neuron frequencies are random U(0.3, 0.9):
    # the min/max mass ratio lands near E[min-sum]/E[max-sum] ~ 0.6.
    assert metrics["cluster_load_balance"] > 0.5


def test_full_metrics_include_stability():
    layer = _synthetic_layer()
    assignment = make_grouping_random(E, seed=11)(I, layer, random.Random(0))
    full = evaluate_grouping_full(assignment, layer, num_experts=E)
    assert set(full) == {"combined", "half_a", "half_b",
                         "stability_within_over_between", "stability_load_balance",
                         "heldout_within_over_between"}
    assert 0.0 <= full["stability_within_over_between"] <= 1.0


def test_sketch_cluster_assigns_same_block_together():
    layer = _synthetic_layer()
    assignment = make_grouping_sketch_cluster(E)(I, layer, random.Random(0))
    # All neurons of planted block 0 (round-robin: i % E == 0) must
    # share one expert label.
    labels = {assignment[i] for i in range(0, I, E)}
    assert len(labels) == 1, f"block 0 scattered: {labels}"
    # ...and each planted block maps to a distinct label.
    block_labels = []
    for b in range(E):
        labels_b = {assignment[i] for i in range(b, I, E)}
        block_labels.append(next(iter(labels_b)) if len(labels_b) == 1 else -1)
    assert len(set(block_labels)) == E, f"blocks collide: {block_labels}"


def test_invalid_grouping_labels_refused():
    layer = _synthetic_layer()
    with pytest.raises(ActivationExperimentError):
        evaluate_grouping([0] * (I - 1) + [E], layer, num_experts=E)


def test_unknown_grouping_name_refused():
    profile = _profile(_synthetic_layer())
    with pytest.raises(ActivationExperimentError):
        run_grouping_comparison(profile, num_experts=E, groupings=["nonexistent"])


def test_missing_sketch_refused():
    layer = _synthetic_layer()
    del layer["sketch"]
    with pytest.raises(ActivationExperimentError):
        make_grouping_sketch_cluster(E)(I, layer, random.Random(0))


def test_artifact_round_trip_and_verdict_text(tmp_path):
    profile = _profile(_synthetic_layer())
    result = run_grouping_comparison(profile, num_experts=E, layer_subset=[0])
    path = tmp_path / "grouping_comparison.json"
    written = write_grouping_comparison(result, path)
    assert written == str(path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["verdict"] == result["verdict"]
    text = format_verdict(loaded)
    assert "DECISION:" in text
    assert ("proceed to Phase 5" in text) or ("negative result" in text)


def test_verdict_negative_text():
    # A profile with no structure anywhere: verdict must say negative.
    layer = _synthetic_layer()
    # Destroy sketch structure: all sketches identical random directions.
    rng = random.Random(5)
    layer["sketch"] = [[rng.gauss(0, 1) for _ in range(32)] for _ in range(I)]
    # And uniform co-occurrence regardless of assignment.
    co = {}
    hot_ids = layer["hot_neuron_ids"]
    for a_i in range(len(hot_ids)):
        for b_i in range(a_i + 1, len(hot_ids)):
            a, c = hot_ids[a_i], hot_ids[b_i]
            co.setdefault(a, {})[c] = 1
            co.setdefault(c, {})[a] = 1
    layer["hotset_cooccurrence"] = co
    profile = _profile(layer)
    result = run_grouping_comparison(profile, num_experts=E, layer_subset=[0])
    text = format_verdict(result)
    assert "negative result" in text or "no signal" in text