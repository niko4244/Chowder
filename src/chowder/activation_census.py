"""Activation census for dense Qwen3.8 parents (sparse-architecture research).

Measures how the dense parent's FFN neurons actually behave on a
calibration corpus, without modifying the model: forward hooks only,
``no_grad``, eval mode, weights untouched. This is the measurement half
of the TurboSparse/PowerInfer investigation (program research note:
``docs/TURBOSPARSE_POWERINFER_RESEARCH.md``) and the input to the
natural-expert-structure evaluation (``activation_experiments.py``).

What is measured per decoder layer (methodology aligned with Eqs. 3-4
of arXiv:2406.05955 and the hot/cold analysis of arXiv:2312.12456):
  - per-neuron activation frequency (gated activation != 0, i.e. the
    dReLU-active set under an applied-dReLU counterfactual),
  - per-neuron mean |activation| magnitude,
  - per-neuron contribution (magnitude x down-column mass),
  - activation sparsity (the dense SwiGLU near-zero fraction and the
    counterfactual dReLU-active fraction),
  - per-token active-neuron counts (sampled),
  - hot/cold-neuron split (power-law concentration),
  - Gini coefficient of the activation-frequency distribution,
  - top-N active-set co-occurrence among hot neurons, tracked
    separately per calibration half so held-out generalization is
    measured, not asserted,
  - a Johnson-Lindenstrauss random-projection sketch of every neuron's
    token-activation profile (exact I x I co-activation is prohibitive;
    each batch's projection is an independent unbiased estimator of the
    sign-profile inner products, so the accumulated sketch preserves
    cosine similarity within the JL bound).

Because silu is not positively homogeneous, the census is taken on the
REAL SwiGLU gated activation ``silu(gate)*up`` -- the quantity the
converted model's experts actually compute -- not on gate alone.
dReLU semantics fall out exactly: ``silu(g)*u != 0`` iff ``g > 0 and
u > 0`` (up to float underflow), so the nonzero mask of the gated
activation IS the dReLU-active set.

Never imports torch at module import time (torch-gated at call time,
mirroring moe_instrumentation). Performs no model surgery: hooks only,
and every hook is removed in ``__exit__``.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

CENSUS_VERSION = 1

#: JL sketch dimensionality for per-neuron token-profile embeddings.
#: 256 dims preserves cosine similarity of token profiles to within a
#: few percent for I~10k neurons (JL bound), at I*256 floats per layer
#: (~10 MB fp32) -- vs I*I (~400 MB) exact per layer.
SKETCH_DIM = 256

#: Hot-neuron fraction for the PowerInfer-style hot/cold split.
HOT_FRACTION = 0.10

#: top-N active neurons per token for the exact co-occurrence matrix.
COOCCURRENCE_TOPN = 64

#: Cap on buffered per-token active sets (memory bound).
MAX_TOKEN_SETS = 200_000


class ActivationCensusError(RuntimeError):
    """The census cannot be recorded honestly."""


@dataclass
class _LayerCensus:
    """Mutable per-layer accumulators. Dimensions: I = intermediate."""

    intermediate_size: int = 0
    tokens_seen: int = 0
    # Per-neuron accumulators (length I each):
    active_count: list[int] = field(default_factory=list)
    magnitude_sum: list[float] = field(default_factory=list)
    # SwiGLU near-zero element count (|act| < 1e-12) and element total:
    near_zero_count: int = 0
    elements_seen: int = 0
    # Per-token active counts (dReLU counterfactual), sampled:
    token_active_counts: list[int] = field(default_factory=list)
    # (global_token_index, top-N active neuron set) buffers:
    token_active_sets: list[tuple[int, set[int]]] = field(default_factory=list)
    # JL sketch: one (I, SKETCH_DIM) accumulator over sign profiles.
    sketch: list[list[float]] = field(default_factory=list)
    # down-column magnitude (I,), captured once from the live weight:
    down_mag: list[float] | None = None
    # Hot neurons fixed at finalize() from active_count:
    hot_neurons: tuple[int, ...] = ()

    def to_summary(self) -> dict[str, Any]:
        n = max(self.tokens_seen, 1)
        total_elements = self.tokens_seen * self.intermediate_size
        freq = [c / n for c in self.active_count] if self.active_count else []
        mean_active = (
            sum(self.token_active_counts) / len(self.token_active_counts)
            if self.token_active_counts
            else 0.0
        )
        mean_freq = (sum(freq) / len(freq)) if freq else 0.0
        return {
            "intermediate_size": self.intermediate_size,
            "tokens_seen": self.tokens_seen,
            "activation_sparsity_swiglu_near_zero": (
                self.near_zero_count / total_elements if total_elements else 0.0
            ),
            "mean_active_fraction_drelu_counterfactual": mean_freq,
            "sparsity_drelu_counterfactual": 1.0 - mean_freq,
            "mean_active_neurons_per_token_drelu": mean_active,
            "gini_activation_frequency": gini(freq),
            "hot_neurons": len(self.hot_neurons),
            "hot_fraction": HOT_FRACTION,
        }


def gini(values: Sequence[float]) -> float:
    """Gini coefficient (0 = uniform, 1 = maximally concentrated)."""
    if not values:
        return 0.0
    s = sorted(values)
    total = sum(s)
    if total <= 0:
        return 0.0
    n = len(s)
    cumulative = 0.0
    for i, v in enumerate(s):
        cumulative += (i + 1) * v
    return (2.0 * cumulative) / (n * total) - (n + 1.0) / n


def _sketch_digest(sketch: list[list[float]]) -> str:
    h = hashlib.sha256()
    for row in sketch:
        h.update(json.dumps([round(v, 5) for v in row], separators=(",", ":")).encode())
    return h.hexdigest()


def _pair_stats(
    pairs: dict[int, dict[int, int]],
) -> dict[str, float]:
    """Summary of a hot-set co-occurrence table: concentration measures."""
    counts = [c for row in pairs.values() for c in row.values()]
    if not counts:
        return {"pairs": 0, "mean_pair_count": 0.0, "max_pair_count": 0}
    return {
        "pairs": len(counts),
        "mean_pair_count": sum(counts) / len(counts),
        "max_pair_count": max(counts),
    }


class ActivationCensus:
    """Hook-based census recorder for ONE dense model instance.

    Usage::

        census = ActivationCensus(model, tokenizer, seed=20260908)
        with census:
            census.consume_texts(half_a)
            census.mark_split()
            census.consume_texts(half_b)
        profile = census.finalize(provenance={...})
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        seed: int,
        device: str = "cpu",
        max_length: int = 512,
        sketch_dim: int = SKETCH_DIM,
        hot_fraction: float = HOT_FRACTION,
        cooccurrence_topn: int = COOCCURRENCE_TOPN,
        token_active_sample_rate: float = 0.25,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._seed = seed
        self._device = device
        self._max_length = max_length
        self._sketch_dim = sketch_dim
        self._hot_fraction = hot_fraction
        self._cooccurrence_topn = cooccurrence_topn
        self._token_active_sample_rate = token_active_sample_rate
        self._rng = random.Random(seed)
        self._handles: list[Any] = []
        self._layers: dict[int, _LayerCensus] = {}
        self._token_index = 0
        self._started = False
        self._active = False
        # Per-layer gate/up pre-activation capture for the dReLU
        # counterfactual mask; keyed by layer idx, consumed by the
        # down_proj pre-hook on the same forward pass.
        self._preacts: dict[int, tuple[Any, Any]] = {}
        # Calibration split boundary (0 = no split recorded).
        self._split_boundary = 0

    # -- layer discovery (dense qwen3_5 layout; refuse surprises) ---------

    def _discover_layers(self) -> list[tuple[int, Any]]:
        layers = []
        base = self._model
        inner = getattr(base, "language_model", None) or getattr(base, "model", None) or base
        decoder = getattr(inner, "layers", None)
        if decoder is None:
            raise ActivationCensusError(
                "could not locate decoder layers (model.language_model.layers); "
                "refusing to guess the layout"
            )
        for idx, layer in enumerate(decoder):
            mlp = getattr(layer, "mlp", None)
            gate = getattr(mlp, "gate_proj", None) if mlp is not None else None
            up = getattr(mlp, "up_proj", None) if mlp is not None else None
            down = getattr(mlp, "down_proj", None) if mlp is not None else None
            if gate is None or up is None or down is None:
                raise ActivationCensusError(
                    f"layer {idx} lacks a dense gate/up/down mlp; this census is "
                    "for DENSE parents only (use moe_instrumentation for MoE models)"
                )
            layers.append((idx, mlp))
        if not layers:
            raise ActivationCensusError("no decoder layers discovered")
        return layers

    def __enter__(self) -> "ActivationCensus":
        import torch

        if self._started:
            raise ActivationCensusError("census already started")
        if getattr(self._model, "_chowder_census_active", False):
            raise ActivationCensusError(
                "another ActivationCensus is already active on this model; "
                "concurrent censuses would double-count tokens"
            )
        self._started = True
        self._active = True
        self._torch = torch
        self._model._chowder_census_active = True
        self._model.eval()
        for idx, mlp in self._discover_layers():
            # Capture gate/up pre-activations (needed for the dReLU
            # counterfactual) and down_proj's INPUT: exactly the gated
            # activation act(gate)*up (I-dim per token) -- the true
            # expert-computed quantity, captured with zero re-computation.
            self._handles.append(
                mlp.gate_proj.register_forward_hook(self._capture_gate(idx))
            )
            self._handles.append(
                mlp.up_proj.register_forward_hook(self._capture_up(idx))
            )
            self._handles.append(
                mlp.down_proj.register_forward_pre_hook(self._make_hook(idx, mlp))
            )
            self._layers[idx] = _LayerCensus()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._active = False
        for h in self._handles:
            h.remove()
        self._handles.clear()
        try:
            delattr(self._model, "_chowder_census_active")
        except AttributeError:
            pass

    # -- hook ------------------------------------------------------------

    def _capture_gate(self, layer_idx: int):
        def hook(module: Any, args: tuple[Any, ...], output: Any) -> None:
            self._preacts.setdefault(layer_idx, [None, None])[0] = output.detach()
        return hook

    def _capture_up(self, layer_idx: int):
        def hook(module: Any, args: tuple[Any, ...], output: Any) -> None:
            self._preacts.setdefault(layer_idx, [None, None])[1] = output.detach()
        return hook

    def _make_hook(self, layer_idx: int, mlp: Any):
        torch = self._torch
        device = next(self._model.parameters()).device

        def hook(module: Any, args: tuple[Any, ...]) -> None:
            gated = args[0].detach()  # (B, T, I): silu(gate)*up
            two_d = gated.reshape(-1, gated.shape[-1])
            I = two_d.shape[-1]
            n_tokens = two_d.shape[0]
            lc = self._layers[layer_idx]
            if lc.intermediate_size == 0:
                lc.intermediate_size = I
                lc.active_count = [0] * I
                lc.magnitude_sum = [0.0] * I
                lc.sketch = [[0.0] * self._sketch_dim for _ in range(I)]
            elif lc.intermediate_size != I:
                raise ActivationCensusError(
                    f"layer {layer_idx} intermediate size changed mid-census"
                )
            pre = self._preacts.pop(layer_idx, None)
            if pre is None or pre[0] is None or pre[1] is None:
                raise ActivationCensusError(
                    f"layer {layer_idx}: gate/up pre-activations were not captured"
                )
            gate_pre = pre[0].reshape(-1, I)
            up_pre = pre[1].reshape(-1, I)
            if gate_pre.shape[0] != n_tokens:
                raise ActivationCensusError(
                    f"layer {layer_idx}: pre-activation token count mismatch "
                    f"({gate_pre.shape[0]} vs {n_tokens})"
                )

            with torch.no_grad():
                # dReLU counterfactual activity: a neuron "fires" iff BOTH
                # paths' pre-activations are positive (TurboSparse Eq. 5
                # semantics on the dense SwiGLU parent).
                active_mask = (gate_pre > 0) & (up_pre > 0)
                active_counts = active_mask.sum(dim=0)  # (I,)
                mags = two_d.abs().sum(dim=0)  # (I,)
                lc.near_zero_count += int((two_d.abs() < 1e-12).sum().item())
                lc.elements_seen += int(two_d.numel())

                if self._rng.random() < self._token_active_sample_rate:
                    per_token = active_mask.sum(dim=1)
                    lc.token_active_counts.extend(int(v) for v in per_token.tolist())

                if len(lc.token_active_sets) < MAX_TOKEN_SETS:
                    topn = min(self._cooccurrence_topn, I)
                    _, top_idx = torch.topk(two_d.abs(), topn, dim=1)
                    base_index = self._token_index - n_tokens
                    for r, row in enumerate(top_idx.tolist()):
                        lc.token_active_sets.append((base_index + r, set(row)))

                # JL sketch update: sign profiles projected by per-batch
                # shared random vectors. Each batch's projection is an
                # independent unbiased estimator of the sign-profile
                # inner products; per-batch seeds decorrelate noise.
                signs = torch.sign(two_d)  # (n_tokens, I)
                g = torch.Generator(device="cpu")
                g.manual_seed(self._seed + self._token_index)
                proj = torch.randn(n_tokens, self._sketch_dim, generator=g)
                proj = (proj / math.sqrt(self._sketch_dim)).to(device)
                update = signs.t() @ proj  # (I, D)
                update_list = update.tolist()
                for i in range(I):
                    row = lc.sketch[i]
                    upd = update_list[i]
                    for d in range(self._sketch_dim):
                        row[d] += upd[d]

                lc.tokens_seen += n_tokens
                for i in range(I):
                    lc.active_count[i] += int(active_counts[i].item())
                    lc.magnitude_sum[i] += float(mags[i].item())
                # down-column magnitude captured once (contribution term)
                if lc.down_mag is None:
                    w = mlp.down_proj.weight.detach()  # (hidden, I)
                    lc.down_mag = w.abs().sum(dim=0).tolist()

        return hook

    def mark_split(self) -> None:
        """Record the current token index as the calibration-split boundary.

        Call once between the calibration halves. finalize() then counts
        hot-set co-occurrence separately per half, making cross-half
        agreement a real held-out test.
        """
        if not self._started:
            raise ActivationCensusError("mark_split requires an active census")
        self._split_boundary = self._token_index

    def consume_texts(self, texts: Sequence[str]) -> None:
        import torch

        if not self._active:
            raise ActivationCensusError(
                "enter the census context before consuming texts "
                "(and not after exit)"
            )
        for text in texts:
            enc = self._tokenizer(
                text, return_tensors="pt", truncation=True, max_length=self._max_length
            )
            enc = {k: v.to(self._device) for k, v in enc.items()}
            self._token_index += int(enc["input_ids"].shape[-1])
            with torch.no_grad():
                self._model(**enc)

    # -- finalization -----------------------------------------------------

    def finalize(self, *, provenance: Mapping[str, Any]) -> dict[str, Any]:
        """Aggregate everything the hooks saw into the machine-readable
        activation profile. The model is never modified."""
        out: dict[str, Any] = {
            "census_version": CENSUS_VERSION,
            "sketch_dim": self._sketch_dim,
            "hot_fraction": self._hot_fraction,
            "cooccurrence_topn": self._cooccurrence_topn,
            "token_active_sample_rate": self._token_active_sample_rate,
            "provenance": dict(provenance),
            "layers": {},
        }
        for idx, lc in sorted(self._layers.items()):
            n = max(lc.tokens_seen, 1)
            freq = [c / n for c in lc.active_count]
            mag = [m / n for m in lc.magnitude_sum]
            down = lc.down_mag or [1.0] * lc.intermediate_size
            contrib = [m * d for m, d in zip(mag, down)]

            order = sorted(range(lc.intermediate_size), key=lambda i: -freq[i])
            n_hot = max(1, int(self._hot_fraction * lc.intermediate_size))
            hot = tuple(sorted(order[:n_hot]))
            lc.hot_neurons = hot
            hot_set = set(hot)

            boundary = self._split_boundary
            co: dict[int, dict[int, int]] = {h: {} for h in hot}
            co_a: dict[int, dict[int, int]] = {h: {} for h in hot}
            co_b: dict[int, dict[int, int]] = {h: {} for h in hot}
            for tok_i, tset in lc.token_active_sets:
                hit = sorted(tset & hot_set)
                target = co_a if (boundary and tok_i < boundary) else co_b
                for a_i in range(len(hit)):
                    for b_i in range(a_i + 1, len(hit)):
                        a, b = hit[a_i], hit[b_i]
                        co[a][b] = co[a].get(b, 0) + 1
                        co[b][a] = co[b].get(a, 0) + 1
                        target[a][b] = target[a].get(b, 0) + 1
                        target[b][a] = target[b].get(a, 0) + 1

            out["layers"][str(idx)] = {
                **lc.to_summary(),
                "per_neuron_top100_frequency": {
                    str(i): round(freq[i], 6) for i in order[:100]
                },
                "per_neuron_top100_contribution": {
                    str(i): round(contrib[i], 6)
                    for i in sorted(
                        range(lc.intermediate_size), key=lambda i: -contrib[i]
                    )[:100]
                },
                "hot_neuron_ids": list(hot),
                "hotset_cooccurrence": {str(k): v for k, v in co.items()},
                "hotset_cooccurrence_half_a": {str(k): v for k, v in co_a.items()},
                "hotset_cooccurrence_half_b": {str(k): v for k, v in co_b.items()},
                "hotset_pair_stats": _pair_stats(co),
                "hotset_pair_stats_half_a": _pair_stats(co_a),
                "hotset_pair_stats_half_b": _pair_stats(co_b),
                "cooccurrence_split_boundary": boundary,
                "sketch": [[round(v, 5) for v in row] for row in lc.sketch],
                "sketch_sha256": _sketch_digest(lc.sketch),
            }
        out["totals"] = {
            "num_layers": len(self._layers),
            "total_tokens": max((l.tokens_seen for l in self._layers.values()), default=0),
            "total_token_sets": sum(len(l.token_active_sets) for l in self._layers.values()),
            "split_boundary": self._split_boundary,
        }
        return out


def write_activation_profile(profile: dict[str, Any], path: str | Path) -> str:
    """Atomically write the activation profile JSON; returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return str(path)
