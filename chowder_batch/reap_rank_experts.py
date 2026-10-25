"""Publish the deterministic, source-bound REAP128_v2 expert ranking.

This is a CPU-only selection operation, NOT a model build or quality gate.
The canonical REAP score and stable tie-breaking remain in reap_prune.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from chowder_batch.reap_build_preflight import build_plan
from chowder_batch.reap_prune import kept_experts

SCHEMA = "chowder.reap128-v2-expert-selection.v1"
N_LAYERS = 40
N_EXPERTS = 256
KEEP = 128


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _layer_details(observations: Path, selected: dict[int, list[int]]) -> dict[str, dict]:
    layers: dict[str, dict] = {}
    for i in range(N_LAYERS):
        path = observations / f"layer-{i:02d}.pt"
        stats = torch.load(path, map_location="cpu", weights_only=True)
        count, saliency_sum = stats["count"].double(), stats["reap_sum"].double()
        scores = saliency_sum / count.clamp(min=1)
        scores = torch.where(count > 0, scores, torch.full_like(scores, -1.0))
        ranked = torch.argsort(scores, descending=True, stable=True).tolist()
        retained = selected[i]
        if set(ranked[:KEEP]) != set(retained):
            raise ValueError(f"layer {i}: canonical REAP ranking disagreement")
        mapping = [-1] * N_EXPERTS
        for destination, source in enumerate(retained):
            mapping[source] = destination
        retained_sum = math.fsum(float(saliency_sum[j]) for j in retained)
        total_sum = math.fsum(float(v) for v in saliency_sum.tolist())
        layers[str(i)] = {
            "stats_sha256": _sha256(path),
            "ranked_source_experts": ranked,
            "retained_source_experts": retained,
            "destination_to_source_expert": retained,
            "source_to_destination_expert": mapping,
            "routed_token_assignments": int(count.sum().item()),
            "experts_used": int((count > 0).sum().item()),
            "score_cutoff_retained": float(scores[ranked[KEEP - 1]]),
            "score_best_discarded": float(scores[ranked[KEEP]]),
            "score_boundary_margin": float(scores[ranked[KEEP - 1]] - scores[ranked[KEEP]]),
            "reap_sum_retained": retained_sum,
            "reap_sum_discarded": total_sum - retained_sum,
            "reap_sum_retained_fraction": retained_sum / total_sum if total_sum > 0 else None,
            "zero_count_experts": [j for j in range(N_EXPERTS) if count[j] == 0],
            "max_expert_routing_fraction": float(count.max() / count.sum()),
        }
    return layers


def generate(*, model_dir: Path, weights_dir: Path, calibration: Path,
             observations: Path, prune_out: Path) -> dict:
    """Fail closed with the actual registered build preflight, then rank."""
    plan = build_plan(
        arm="reap128_v2", model_dir=model_dir, weights_dir=weights_dir,
        calibration=calibration, observations=observations, out=prune_out,
    )
    # Reuse the exact function the model pruner executes; no new REAP definition.
    selected = kept_experts(observations, N_LAYERS, N_EXPERTS, KEEP, "reap", 20261017)
    layers = _layer_details(observations, selected)
    return {
        "schema": SCHEMA,
        "status": "EXPERTS_SELECTED_NOT_BUILT_OR_QUALIFIED",
        "arm": "reap128_v2",
        "method": "reap",
        "score_definition": "reap_sum / count for count > 0; stable descending; expert-id tie-break",
        "layers_count": N_LAYERS,
        "source_experts_per_layer": N_EXPERTS,
        "retained_experts_per_layer": KEEP,
        "observer_sha256": plan["calibration"]["observer_sha256"],
        "calibration_sha256": plan["calibration"]["calibration_sha256"],
        "source_model_config_sha256": plan["model"]["config_sha256"],
        "source_model_index_sha256": plan["model"]["index_sha256"],
        "saliency_layer_digest_sha256": plan["saliency_layer_digest_sha256"],
        "source_model_dir": plan["model"]["directory"],
        "source_weights_dir": plan["model"]["weights_directory"],
        "observation_dir": str(observations.resolve()),
        "frozen_order_predecessors": plan["frozen_order_predecessors"],
        "intended_prune_out": plan["output_directory_intended_only"],
        "layers": layers,
        "qualification": plan["qualification"],
    }


def validate_selection(document: dict, *, model_dir: Path,
                       weights_dir: Path, observations: Path,
                       kept: dict[int, list[int]]) -> None:
    """Pruner-side check that an external selection matches its exact inputs."""
    if not isinstance(document, dict) or any((
        document.get("schema") != SCHEMA,
        document.get("arm") != "reap128_v2",
        document.get("method") != "reap",
        document.get("layers_count") != N_LAYERS,
        document.get("source_experts_per_layer") != N_EXPERTS,
        document.get("retained_experts_per_layer") != KEEP,
        document.get("calibration_sha256") != json.loads(
            (observations / "observation.json").read_text(encoding="utf-8")
        ).get("calibration_sha256"),
        document.get("source_model_dir") != str(model_dir.resolve()),
        document.get("source_weights_dir") != str(weights_dir.resolve()),
        document.get("observation_dir") != str(observations.resolve()),
        document.get("observer_sha256") != _sha256(observations / "observation.json"),
        document.get("source_model_config_sha256") != _sha256(model_dir / "config.json"),
        document.get("source_model_index_sha256") != _sha256(model_dir / "model.safetensors.index.json"),
    )):
        raise ValueError("REAP128_v2 selection identity differs from build inputs")
    layers = document.get("layers")
    if not isinstance(layers, dict) or set(layers) != {str(i) for i in range(N_LAYERS)}:
        raise ValueError("REAP128_v2 selection must contain exactly 40 layers")
    for i in range(N_LAYERS):
        entry = layers[str(i)]
        ids = kept[i]
        path = observations / f"layer-{i:02d}.pt"
        stats = torch.load(path, map_location="cpu", weights_only=True)
        count, reap_sum = stats["count"], stats["reap_sum"]
        scores = reap_sum.double() / count.double().clamp(min=1)
        scores = torch.where(count > 0, scores, torch.full_like(scores, -1.0))
        ranking = torch.argsort(scores, descending=True, stable=True).tolist()
        if (not isinstance(entry, dict)
                or entry.get("stats_sha256") != _sha256(path)
                or entry.get("ranked_source_experts") != ranking
                or entry.get("retained_source_experts") != ids
                or entry.get("destination_to_source_expert") != ids
                or entry.get("source_to_destination_expert") != [
                    ids.index(j) if j in ids else -1 for j in range(N_EXPERTS)
                ]):
            raise ValueError(f"REAP128_v2 selection differs from canonical layer {i} ranking")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--prune-out", type=Path, required=True,
                        help="future model build directory (must not exist)")
    parser.add_argument("--out", type=Path, required=True,
                        help="new selection.json file (must not exist)")
    args = parser.parse_args(argv)
    if not args.out.parent.is_dir() or args.out.exists():
        parser.error("selection output parent must exist and selection file must be new")
    if any(args.out.resolve().is_relative_to(source.resolve()) for source in (
        args.model_dir, args.weights_dir, args.calibration, args.observations, args.prune_out,
    )):
        parser.error("selection output must be outside every immutable input and prune destination")
    document = generate(model_dir=args.model_dir, weights_dir=args.weights_dir,
                        calibration=args.calibration, observations=args.observations,
                        prune_out=args.prune_out)
    # Verify the consumer can read precisely the published mapping before writing.
    kept = kept_experts(args.observations, N_LAYERS, N_EXPERTS, KEEP, "reap", 20261017)
    validate_selection(document, model_dir=args.model_dir, weights_dir=args.weights_dir,
                       observations=args.observations, kept=kept)
    payload = json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with args.out.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
    print(json.dumps({"status": document["status"], "out": str(args.out.resolve()),
                      "sha256": _sha256(args.out), "layers": N_LAYERS,
                      "experts_per_layer": KEEP}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
