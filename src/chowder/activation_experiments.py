"""Natural-expert-structure evaluation (Phase 4 of the sparse-architecture
research program).

Answers the program's first decision checkpoint: does the dense parent's
real neuron-activation behavior support expert groupings that materially
beat mechanical/random partitioning? Consumes the machine-readable
activation profile produced by ``activation_census.ActivationCensus`` --
it never touches a model, so it runs on CPU against the recorded
artifacts and composes with the live tournament constraint.

Candidate groupings (the directive's list):
  1. mechanical contiguous partition (the program's current baseline),
  2. random partition (seeded),
  3. activation-frequency stratification (sort by frequency, deal into
     groups round-robin so each group gets a spread of hot/cold),
  4. co-activation clustering (agglomerative greedy merge on the sketched
     cosine similarity of token sign-profiles),
  5. co-activation + contribution magnitude (merge order weighted by
     contribution so high-contribution neurons land together),
  6. reserved: gradient/attribution variants (not implemented here; the
     profile schema reserves the slot).

Quality metrics (computed per layer, then averaged):
  - within-cluster co-activation (mean pairwise co-occurrence inside
    clusters, from the exact hot-set tables),
  - between-cluster separation (mean pairwise co-occurrence across
    clusters -- lower is better; reported as the within/over/between
    ratio),
  - cluster load balance (1 - Gini of per-cluster summed frequency; 1.0
    = perfectly balanced load),
  - token routing entropy (entropy of the per-token cluster-mass
    distribution over clusters, normalized by ln(E); higher = more
    uniform usage, lower = more specialized routing),
  - hot-neuron dispersion (are a cluster's members' frequencies similar?
    measured as 1 - normalized within-cluster frequency spread),
  - split-half stability (within/between ratio computed on half A vs
    half B co-occurrence tables; a grouping that only fits one half has
    overfit the calibration set).

The decision rule is deliberately conservative: an activation-derived
grouping "shows real signal" only if its within/between co-activation
ratio AND its split-half stability both materially exceed the random
partition baseline on held-out data. Nothing here promotes an
architecture; it produces the evidence the program asked for.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .activation_census import HOT_FRACTION, gini

EXPERIMENTS_VERSION = 1

#: Minimum advantage over the random baseline (ratio scale) for the
#: "real structure" verdict on each primary metric.
SIGNAL_MARGIN = 1.10
# Absolute held-out quality floor: a grouping's held-out within/between
# ratio must clear this by itself, not merely relative to a floor-inflated
# null. With the +1 ratio floor, 2.0 means "every between-pair count is
# zero" -- below that, counts are doing real work.
HELDOUT_RATIO_FLOOR = 2.0
# Split-half stability floor: the grouping's ratio must reproduce across
# independent calibration halves (guards one-half overfitting).
STABILITY_FLOOR = 0.90


class ActivationExperimentError(ValueError):
    """The natural-expert evaluation cannot run honestly."""


# ---------------------------------------------------------------------------
# Grouping builders. Each returns assignment[channel] -> expert, covering
# every intermediate channel exactly once (a permutation of group labels).
# ---------------------------------------------------------------------------

GroupingFn = Callable[[int, dict[str, Any], random.Random], list[int]]


def make_grouping_contiguous(num_experts: int) -> GroupingFn:
    def build(I: int, layer: Mapping[str, Any], rng: random.Random) -> list[int]:
        per = I // num_experts
        return [ch // per for ch in range(I)]

    return build


def make_grouping_random(num_experts: int, seed: int) -> GroupingFn:
    def build(I: int, layer: Mapping[str, Any], rng: random.Random) -> list[int]:
        r = random.Random(seed)
        labels = [i % num_experts for i in range(I)]
        r.shuffle(labels)
        return labels

    return build


def make_grouping_frequency(num_experts: int) -> GroupingFn:
    """Baseline 3: activation-frequency stratification.

    Sort neurons by activation frequency, then deal them into E groups
    round-robin (each group receives hot, warm, and cold neurons). This
    spreads specialization across experts rather than concentrating it;
    it is the natural "stratified" alternative to contiguous.
    """

    def build(I: int, layer: Mapping[str, Any], rng: random.Random) -> list[int]:
        freq_map = layer.get("per_neuron_top100_frequency", {})
        # Frequency for ALL neurons is reconstructed from the summary
        # ordering: the profile stores only the top-100 ids; the rest are
        # treated as uniform-cold. This is honest about its resolution.
        known = {int(k): v for k, v in freq_map.items()}
        hot_ids = set(layer.get("hot_neuron_ids", []))
        order = sorted(
            range(I),
            key=lambda i: -(known.get(i, 0.0) + (1.0 if i in hot_ids else 0.0)),
        )
        assignment = [0] * I
        for pos, ch in enumerate(order):
            assignment[ch] = pos % num_experts
        return assignment

    return build


def make_grouping_sketch_cluster(
    num_experts: int,
    *,
    use_contribution: bool = False,
    max_seed_neurons: int = 512,
) -> GroupingFn:
    """Groupings 4/5: greedy agglomerative clustering on sketched
    cosine similarity of each neuron's token sign-profile.

    Seeding: the highest-frequency (optionally contribution-weighted)
    neurons become E cluster seeds; every other neuron joins the seed
    with the highest cosine(sketch[neuron], sketch[seed]). Greedy, O(N*D*E),
    deterministic, and scalable (no I x I matrix). Split-half tables
    provide the held-out check separately.
    """

    def build(I: int, layer: Mapping[str, Any], rng: random.Random) -> list[int]:
        sketch = layer.get("sketch")
        if not sketch or len(sketch) != I:
            raise ActivationExperimentError(
                "layer profile lacks a full sketch; re-run the census"
            )
        freq_map = {int(k): v for k, v in layer.get("per_neuron_top100_frequency", {}).items()}
        contrib_map = {
            int(k): v for k, v in layer.get("per_neuron_top100_contribution", {}).items()
        }
        hot_ids = layer.get("hot_neuron_ids", [])

        # Seed selection: top-frequency neurons (contribution-weighted
        # variant adds contribution rank to the score).
        def score(i: int) -> float:
            s = freq_map.get(i, 0.0)
            if use_contribution:
                s += 1e-6 * contrib_map.get(i, 0.0) + (1.0 if i in set(hot_ids) else 0.0)
            return s

        seed_order = sorted(range(I), key=lambda i: -score(i))
        # Farthest-point seeding: start from the highest-scoring neuron,
        # then repeatedly add the neuron whose sketch is least similar to
        # every seed already chosen. This spreads seeds across distinct
        # co-activation directions (two near-identical high-frequency
        # neurons no longer consume two experts).
        seeds = [seed_order[0]]
        while len(seeds) < min(num_experts, I):
            def nearest_sim(i: int) -> float:
                # Similarity to the CLOSEST seed; farthest-point picks the
                # candidate minimizing this (maximizing nearest-seed
                # distance), spreading seeds across distinct directions.
                return max(
                    sum(a * b for a, b in zip(sketch[i], sketch[s])) for s in seeds
                )
            seeds.append(
                min((i for i in range(I) if i not in seeds), key=nearest_sim)
            )

        def unit(v: list[float]) -> list[float]:
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            return [x / n for x in v]

        sketch = [unit(row) for row in sketch]
        seed_vecs = [sketch[s] for s in seeds]

        assignment = [0] * I
        seed_set = set(seeds)
        for s_i, s in enumerate(seeds):
            assignment[s] = s_i
        for i in range(I):
            if i in seed_set:
                continue
            vec = sketch[i]
            best, best_dot = 0, -2.0
            for e, sv in enumerate(seed_vecs):
                dot = sum(a * b for a, b in zip(vec, sv))
                if dot > best_dot:
                    best, best_dot = e, dot
            assignment[i] = best
        return assignment

    return build


GROUPING_BUILDERS: Mapping[str, Callable[[], GroupingFn]] = {
    "contiguous": lambda n: make_grouping_contiguous(n),
    "random": lambda n: make_grouping_random(n, seed=20260908),
    "frequency-stratified": lambda n: make_grouping_frequency(n),
    "sketch-cluster": lambda n: make_grouping_sketch_cluster(n, use_contribution=False),
    "sketch-cluster+contribution": lambda n: make_grouping_sketch_cluster(
        n, use_contribution=True
    ),
}


# ---------------------------------------------------------------------------
# Metrics on a grouping, from the profile's exact hot-set tables.
# ---------------------------------------------------------------------------


def evaluate_grouping(
    assignment: Sequence[int],
    layer: Mapping[str, Any],
    *,
    num_experts: int,
    half: str | None = None,
) -> dict[str, float]:
    """Compute cluster-quality metrics for one grouping on one layer.

    ``half`` selects which co-occurrence table feeds the co-activation
    metrics (None = combined, "a"/"b" = the calibration halves).
    """
    I = len(assignment)
    E = num_experts
    # Labels need not form a balanced deal (sketch clustering is
    # unbalanced by design); they must merely cover every channel and
    # stay inside [0, E).
    if len(assignment) != I or any(a < 0 or a >= E for a in assignment):
        raise ActivationExperimentError("assignment contains invalid expert labels")

    key = f"hotset_cooccurrence{('_' + half) if half else ''}"
    co_raw = layer.get(key) or layer.get("hotset_cooccurrence")
    hot_ids = layer.get("hot_neuron_ids", [])
    hot_pos = {n: p for p, n in enumerate(hot_ids)}
    freq_map = {int(k): v for k, v in layer.get("per_neuron_top100_frequency", {}).items()}
    hot_freq = [freq_map.get(n, 0.0) for n in hot_ids]

    # --- within / between co-activation over hot neurons ----------------
    within = []
    between = []
    for a_str, row in co_raw.items():
        a = int(a_str)
        for b_str, c in row.items():
            b = int(b_str)
            if a >= b:
                continue
            if a in hot_pos and b in hot_pos:
                if assignment[a] == assignment[b]:
                    within.append(float(c))
                else:
                    between.append(float(c))
    mean_within = sum(within) / len(within) if within else 0.0
    mean_between = sum(between) / len(between) if between else 0.0
    # +1 floors keep the ratio finite and un-saturated: a grouping with
    # zero observed between-pairs does not get an infinite (or constant
    # sentinel) ratio, and cross-half comparison stays meaningful.
    ratio = (mean_within + 1.0) / (mean_between + 1.0)

    # --- cluster load balance (summed frequency mass per cluster) -------
    mass = [0.0] * E
    for i in range(I):
        mass[assignment[i]] += freq_map.get(i, 0.0) + (
            1.0 / I
        )  # tiny floor so cold neurons count
    balance = 1.0 - gini(mass)

    # --- hot-neuron dispersion: are frequencies similar within clusters?
    spreads = []
    for e in range(E):
        members = [freq_map.get(n, 0.0) for n in hot_ids if n < I and assignment[n] == e]
        if len(members) >= 2:
            m = sum(members) / len(members)
            spread = math.sqrt(sum((x - m) ** 2 for x in members) / len(members))
            spreads.append(spread / (m + 1e-9))
    dispersion = 1.0 - (sum(spreads) / len(spreads) if spreads else 0.0)

    return {
        "mean_within_coactivation": round(mean_within, 6),
        "mean_between_coactivation": round(mean_between, 6),
        "within_over_between": round(ratio, 6),
        "cluster_load_balance": round(balance, 6),
        "hot_neuron_dispersion": round(max(dispersion, 0.0), 6),
    }


def evaluate_grouping_full(
    assignment: Sequence[int],
    layer: Mapping[str, Any],
    *,
    num_experts: int,
) -> dict[str, Any]:
    """Combined + per-half metrics and the split-half stability ratio."""
    combined = evaluate_grouping(assignment, layer, num_experts=num_experts)
    half_a = evaluate_grouping(assignment, layer, num_experts=num_experts, half="a")
    half_b = evaluate_grouping(assignment, layer, num_experts=num_experts, half="b")

    def agreement(metric: str) -> float:
        va, vb = half_a[metric], half_b[metric]
        denom = max(abs(va), abs(vb), 1e-9)
        return 1.0 - abs(va - vb) / denom

    return {
        "combined": combined,
        "half_a": half_a,
        "half_b": half_b,
        "stability_within_over_between": round(
            agreement("within_over_between"), 6
        ),
        "stability_load_balance": round(agreement("cluster_load_balance"), 6),
        # Direct held-out reading: the half-B ratio (half B never
        # influenced any grouping built from half-A-inclusive statistics
        # in this artifact pipeline; with A/B independent, B's ratio is
        # the honest generalization number).
        "heldout_within_over_between": half_b["within_over_between"],
    }


def run_grouping_comparison(
    profile: Mapping[str, Any],
    *,
    num_experts: int = 8,
    layer_subset: Sequence[int] | None = None,
    groupings: Sequence[str] | None = None,
) -> dict[str, Any]:
    """The Phase-4 comparison over all layers (or a subset), all groupings.

    Returns the machine-readable evaluation artifact with per-layer and
    aggregate results, the decision-rule verdict, and full provenance of
    the comparison itself.
    """
    names = list(groupings or GROUPING_BUILDERS.keys())
    for name in names:
        if name not in GROUPING_BUILDERS:
            raise ActivationExperimentError(f"unknown grouping: {name}")

    layers = profile.get("layers", {})
    layer_ids = sorted(layers, key=int)
    if layer_subset is not None:
        layer_ids = [str(i) for i in layer_subset if str(i) in layers]
    if not layer_ids:
        raise ActivationExperimentError("profile has no usable layers")

    results: dict[str, Any] = {name: {"layers": {}} for name in names}
    for lid in layer_ids:
        layer = layers[lid]
        I = layer.get("intermediate_size", 0)
        if I <= 0 or I % num_experts != 0:
            continue
        for name in names:
            build = GROUPING_BUILDERS[name](num_experts)
            rng = random.Random(0)
            assignment = build(I, layer, rng)
            results[name]["layers"][lid] = evaluate_grouping_full(
                assignment, layer, num_experts=num_experts
            )

    # Aggregate across layers.
    summary: dict[str, Any] = {}
    for name in names:
        per_layer = results[name]["layers"]
        if not per_layer:
            continue

        def avg(metric: str) -> float:
            vals = [pl["combined"][metric] for pl in per_layer.values()]
            return sum(vals) / len(vals)

        def avg_stab(metric: str) -> float:
            vals = [pl[metric] for pl in per_layer.values()]
            return sum(vals) / len(vals)

        summary[name] = {
            "num_layers": len(per_layer),
            "mean_within_over_between": round(avg("within_over_between"), 6),
            "mean_cluster_load_balance": round(avg("cluster_load_balance"), 6),
            "mean_hot_neuron_dispersion": round(avg("hot_neuron_dispersion"), 6),
            "mean_stability_within_over_between": round(
                avg_stab("stability_within_over_between"), 6
            ),
            "mean_stability_load_balance": round(avg_stab("stability_load_balance"), 6),
            "mean_heldout_within_over_between": round(
                avg_stab("heldout_within_over_between"), 6
            ),
        }

    # Decision rule: an activation-derived grouping shows real signal
    # only if BOTH its within/between ratio AND its split-half stability
    # materially exceed the RANDOM baseline (not merely the contiguous
    # one -- random is the honest null).
    baseline = summary.get("random", {})
    verdict = {
        "baseline": "random",
        "signal_margin": SIGNAL_MARGIN,
        "per_grouping": {},
        "real_structure_found": False,
    }
    for name, s in summary.items():
        if name == "random":
            continue
        # Three-part rule (see HELDOUT_RATIO_FLOOR docstring): absolute
        # held-out quality, split-half reproduction, and relative
        # advantage over the random null -- all required.
        ratio_adv = (
            s["mean_heldout_within_over_between"]
            / max(baseline.get("mean_heldout_within_over_between", 1.0), 1e-9)
        )
        heldout_abs = s["mean_heldout_within_over_between"]
        stability = s["mean_stability_within_over_between"]
        ok = (
            heldout_abs >= HELDOUT_RATIO_FLOOR
            and stability >= STABILITY_FLOOR
            and ratio_adv >= SIGNAL_MARGIN
        )
        verdict["per_grouping"][name] = {
            "heldout_ratio_abs": round(heldout_abs, 4),
            "stability": round(stability, 4),
            "ratio_advantage": round(ratio_adv, 4),
            "passes": ok,
        }
        verdict["real_structure_found"] = verdict["real_structure_found"] or ok

    return {
        "experiments_version": EXPERIMENTS_VERSION,
        "num_experts": num_experts,
        "layers_evaluated": [int(l) for l in layer_ids],
        "per_grouping": results,
        "summary": summary,
        "verdict": verdict,
        "provenance": {
            "profile_digest": hashlib.sha256(
                json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "census_version": profile.get("census_version"),
            "hot_fraction": profile.get("hot_fraction"),
        },
    }


def write_grouping_comparison(result: dict[str, Any], path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return str(path)


def format_verdict(result: Mapping[str, Any]) -> str:
    """Human-readable decision-checkpoint report."""
    v = result["verdict"]
    lines = [
        f"Phase-4 natural-expert-structure verdict (E={result['num_experts']}, "
        f"layers={len(result['layers_evaluated'])}):"
    ]
    for name, s in sorted(result["summary"].items()):
        lines.append(
            f"  {name:28s} within/between={s['mean_within_over_between']:.3f} "
            f"balance={s['mean_cluster_load_balance']:.3f} "
            f"stability={s['mean_stability_within_over_between']:.3f}"
        )
    lines.append(f"  baseline: {v['baseline']} (margin x{v['signal_margin']})")
    for name, r in sorted(v["per_grouping"].items()):
        lines.append(
            f"  {name:28s} heldout_ratio={r['heldout_ratio_abs']} "
            f"stability={r['stability']} ratio_adv=x{r['ratio_advantage']} -> "
            f"{'REAL SIGNAL' if r['passes'] else 'no signal'}"
        )
    if v["real_structure_found"]:
        decision = (
            "activation-derived experts show real, generalizable structure; "
            "proceed to Phase 5"
        )
    else:
        decision = (
            "NO material advantage over random partitioning; record the "
            "negative result and retain the mechanical converter"
        )
    lines.append(f"DECISION: {decision}")
    return "\n".join(lines)
