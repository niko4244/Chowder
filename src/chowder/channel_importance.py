"""Measure which FFN channels matter, so a conversion can rank instead of guess.

Why this module exists
---------------------
`dense_to_moe` partitions the dense FFN's intermediate dimension by *index*.
Measurement says that choice, not sparsity itself, is what destroyed the 27B
ladder: held-out on the 9B at 28% of channels kept, an index-ordered (arbitrary)
subset scores **614.7x** baseline perplexity while an activation-ranked subset of
the same size scores **1.69x**. Ranking is worth up to three orders of magnitude,
and it costs one forward pass over a calibration split.

What is measured
---------------
Per layer, per intermediate channel c:  `sum over calibration tokens of |h_c|`,
where `h = silu(gate_proj(x)) * up_proj(x)` is exactly the vector `down_proj`
consumes. That is the channel's actual contribution magnitude into the residual
stream, read off the unmodified dense parent via a `forward_pre_hook` on
`down_proj` -- no surgery, no gradients, forward-only.

This is deliberately NOT the dReLU counterfactual that
`a4b_sparsity_diag.py` reports. That one answers "what would a dReLU variant
leave inactive" and needs ~150B tokens of continued pretraining to realise.
This one answers "on the SiLU model as it stands today, which channels carry
the output", which is the quantity a conversion can act on immediately.

Honesty constraints this module enforces
---------------------------------------
* **Hooks must fire.** If nothing is captured it raises rather than returning a
  confident all-zero ranking, which would silently degenerate to index order --
  i.e. to the 614.7x arm.
* **Selection and evaluation must not share data.** `measure_channel_importance`
  records which calibration rows it consumed and their digest, and
  `ChannelRanking.split` names the convention used. Ranking on the same prompts
  later used to score the converted model is selection-on-test; v1 of this
  measurement did exactly that and read ~0.02-0.09x optimistic. Callers that
  evaluate must pass a disjoint split.
* **The split must be spread across the corpus, not a contiguous slice of it.**
  Disjoint is necessary but not sufficient. Measured on the 9B: ranking on 32
  contiguous prompts cost 1.691x on a nearby eval split but 3.646x on a distant
  one, a 2.16x spread. Re-ranking on 32 prompts evenly spread over the same corpus
  moved those to 1.859x and 3.011x -- worse nearby, **17.4% better far away**, and
  the spread fell to 1.62x. Only 73.5% of the top-3,440 channels were shared
  between the two rankings, so this is a real change in what gets kept, not noise.
  Prefer `spread_across` over a head slice; a parochial ranking produces a
  checkpoint that looks fine on whatever you ranked near and degrades elsewhere.
* **The ranking is bound to one checkpoint.** `source_manifest_sha256` is
  recorded so `hot_core_upcycle` can refuse a ranking measured on a different
  model, which would otherwise produce a plausible-looking and wrong conversion.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .local_model_manifest import build_local_model_manifest

#: How `ranking` is ordered: hottest channel first.
METHOD = "abs_activation_sum_v1"


class ChannelImportanceError(ValueError):
    """Channel importance cannot be measured or trusted."""


@dataclass(frozen=True)
class ChannelRanking:
    """Per-layer channel order, hottest first, bound to one checkpoint."""

    method: str
    source_dir: str
    source_manifest_sha256: str
    intermediate_size: int
    ranking: dict[int, tuple[int, ...]]
    calibration: dict[str, Any]
    concentration: dict[str, float]

    def __post_init__(self) -> None:
        if not self.ranking:
            raise ChannelImportanceError("a ranking with no layers is not a ranking")
        expected = set(range(self.intermediate_size))
        for layer, order in self.ranking.items():
            if set(order) != expected:
                raise ChannelImportanceError(
                    f"layer {layer} ranking must be a permutation of all "
                    f"{self.intermediate_size} channels; got {len(order)} entries "
                    f"covering {len(set(order))} distinct channels"
                )

    def digest(self) -> str:
        """Stable digest of the ordering itself (not the metadata around it)."""
        blob = json.dumps(
            {str(k): list(v) for k, v in sorted(self.ranking.items())},
            separators=(",", ":"), sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "source_dir": self.source_dir,
            "source_manifest_sha256": self.source_manifest_sha256,
            "intermediate_size": self.intermediate_size,
            "num_layers": len(self.ranking),
            "calibration": dict(self.calibration),
            "concentration": dict(self.concentration),
            "ranking_digest": self.digest(),
            "ranking": {str(k): list(v) for k, v in sorted(self.ranking.items())},
        }

    def write(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=False) + "\n",
            encoding="utf-8", newline="\n",
        )
        return out

    @classmethod
    def read(cls, path: str | Path) -> "ChannelRanking":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        recorded = payload.get("ranking_digest")
        ranking = {int(k): tuple(v) for k, v in payload["ranking"].items()}
        obj = cls(
            method=payload["method"],
            source_dir=payload["source_dir"],
            source_manifest_sha256=payload["source_manifest_sha256"],
            intermediate_size=int(payload["intermediate_size"]),
            ranking=ranking,
            calibration=payload.get("calibration", {}),
            concentration=payload.get("concentration", {}),
        )
        if recorded and recorded != obj.digest():
            raise ChannelImportanceError(
                f"ranking digest mismatch in {path}: recorded {recorded}, "
                f"recomputed {obj.digest()} -- the file was edited after writing"
            )
        return obj


def digest_texts(texts: Sequence[str]) -> str:
    """Digest of the exact calibration strings consumed, in order."""
    h = hashlib.sha256()
    for text in texts:
        h.update(hashlib.sha256(text.encode("utf-8")).digest())
    return h.hexdigest()


def measure_channel_importance(
    model_dir: str | Path,
    texts: Sequence[str],
    *,
    max_length: int = 384,
    load_in_4bit: bool = True,
    split_label: str = "unspecified",
    device_map: str = "cuda:0",
) -> ChannelRanking:
    """Rank each layer's FFN channels by summed |h| over `texts`.

    Forward-only, no grad. 4-bit load is the default because the ranking is an
    *ordering* -- quantisation shifts absolute magnitudes far less than it would
    need to in order to reorder the channel list, and it is the difference
    between fitting a 9B on a 16 GiB card and not. Pass `load_in_4bit=False`
    when a full-precision ranking is affordable.

    Raises rather than returning a degenerate ranking: an all-zero importance
    vector would silently sort to index order, which is the arm measured at
    614.7x baseline perplexity.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ChannelImportanceError(
            "measuring channel importance needs the 'train' extra "
            "(torch + transformers); install it or supply a ranking file"
        ) from exc

    if not texts:
        raise ChannelImportanceError("no calibration texts supplied")

    source = Path(model_dir)
    source_manifest = build_local_model_manifest(source, mode="fast")

    kwargs: dict[str, Any] = {"local_files_only": True, "trust_remote_code": False}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        )
        kwargs["device_map"] = device_map
        kwargs["dtype"] = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
    model.eval()

    inner = getattr(model, "model", model)
    inner = getattr(inner, "language_model", inner)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise ChannelImportanceError(
            "could not locate decoder layers; refusing to emit a ranking for a "
            "model whose shape was not verified"
        )

    targets: list[tuple[int, Any]] = []
    for index, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "down_proj") and hasattr(mlp, "gate_proj"):
            targets.append((index, mlp))
    if not targets:
        raise ChannelImportanceError(
            "no layer exposed a dense mlp.gate_proj/down_proj; this measures the "
            "dense parent, not an already-converted MoE"
        )
    intermediate_size = int(targets[0][1].down_proj.in_features)

    totals = {
        index: torch.zeros(intermediate_size, dtype=torch.float64, device=model.device)
        for index, _ in targets
    }
    captured: dict[int, Any] = {}
    tokens_seen = 0

    def make_hook(index: int):
        def pre_hook(_module, inputs):
            captured[index] = inputs[0].detach()
            return None
        return pre_hook

    handles = [mlp.down_proj.register_forward_pre_hook(make_hook(i)) for i, mlp in targets]
    try:
        with torch.no_grad():
            for text in texts:
                encoded = tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=max_length
                )
                encoded = {k: v.to(model.device) for k, v in encoded.items()}
                captured.clear()
                model(**encoded)
                tokens_seen += int(encoded["input_ids"].shape[-1])
                for index, hidden in captured.items():
                    totals[index] += hidden.double().abs().sum(
                        dim=tuple(range(hidden.dim() - 1))
                    )
    finally:
        for handle in handles:
            handle.remove()

    ranking: dict[int, tuple[int, ...]] = {}
    concentration: dict[str, float] = {}
    top_shares: list[float] = []
    for index, total in totals.items():
        mass = float(total.sum().item())
        if mass <= 0.0:
            raise ChannelImportanceError(
                f"layer {index} accumulated zero importance mass over "
                f"{len(texts)} prompts; the hooks captured nothing usable and a "
                "ranking built from this would silently be index order"
            )
        order = torch.argsort(total, descending=True)
        ranking[index] = tuple(int(c) for c in order.tolist())
        top = max(1, intermediate_size // 10)
        top_shares.append(float(total[order[:top]].sum().item() / mass))

    concentration["top10pct_mass_mean"] = sum(top_shares) / len(top_shares)
    concentration["top10pct_mass_min"] = min(top_shares)
    concentration["top10pct_mass_max"] = max(top_shares)

    return ChannelRanking(
        method=METHOD,
        source_dir=str(source),
        source_manifest_sha256=source_manifest.manifest_sha256,
        intermediate_size=intermediate_size,
        ranking=ranking,
        calibration={
            "prompts": len(texts),
            "tokens": tokens_seen,
            "max_length": max_length,
            "texts_sha256": digest_texts(texts),
            "split": split_label,
            "load_in_4bit": load_in_4bit,
        },
        concentration=concentration,
    )


def split_disjoint(texts: Iterable[str]) -> tuple[list[str], list[str]]:
    """Even indices rank, odd indices evaluate.

    Offered as a named convention so the two halves cannot drift apart between
    a ranking run and a scoring run. Selection-on-test is the one methodological
    error this measurement is known to make when left to chance.

    Note this interleaves within whatever it is given. Applied to a head slice it
    still yields a *parochial* ranking split -- see `spread_across`, and the
    module docstring for what that cost when measured.
    """
    rows = list(texts)
    return rows[0::2], rows[1::2]


def spread_across(texts: Iterable[str], count: int, *, exclude: Iterable[str] = ()) -> list[str]:
    """Pick `count` texts evenly spread over the whole corpus, deduplicated.

    The ranking split's *location* matters as much as its disjointness: a
    contiguous slice produced a ranking that cost 1.691x near it and 3.646x far
    away, while the same number of prompts spread over the corpus cost 1.859x and
    3.011x. Use this to choose a ranking split, and pass the eval splits as
    `exclude` so selection-on-test is structurally impossible rather than merely
    intended.

    Deduplicates on text because identical prompts recur in real corpora -- the
    9B's 696-row calibration file holds 690 distinct texts, so disjoint *index*
    ranges were not disjoint splits.
    """
    blocked = set(exclude)
    pool: list[str] = []
    seen: set[str] = set()
    for text in texts:
        if text in blocked or text in seen:
            continue
        seen.add(text)
        pool.append(text)
    if not pool:
        raise ChannelImportanceError("no eligible texts left after exclusions")
    if count >= len(pool):
        return pool
    stride = len(pool) / count
    picked = [pool[min(len(pool) - 1, int(round(k * stride)))] for k in range(count)]
    return list(dict.fromkeys(picked))
