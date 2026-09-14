"""Score a published router payload in a process of its own.

The training worker cannot evaluate itself: it holds the optimizer state, the
data order and the loss it was optimizing, and its own report is a claim about
what it did rather than a measurement of what the result is worth. This process
starts from the base weights and the published payload only.

What "independent" means here, concretely
-----------------------------------------
* The base is loaded fresh from disk and its **content** identity must match the
  one the payload was trained against; a mismatch refuses before scoring.
* The holdout corpus is a separate file with its own recorded hash, so scoring
  fit instead of capability is a refusal rather than a quieter number.
* The payload is re-verified from disk -- manifest, per-tensor hashes, shapes,
  dtypes -- and applied to exactly the declared parameter set.
* The **application control** is measured, not inherited: a probe input is run
  before and after applying the payload, and the resulting logits are digested.
  A payload that changed parameters but not the model's output proves the
  routing path is not wired to those tensors, and that is a refusal rather than
  a score.

Two metrics are reported, and their provenance is recorded separately so they
can never be read as the same kind of number:

* ``holdout_loss`` -- measured, a real forward pass over held-out data.
* ``experts_per_token`` -- read from the loaded model's own configuration. It
  describes the routing the model is configured for, and is labelled as a
  configuration read in ``metric_sources``, never presented as a behaviour.
* ``dead_experts`` -- measured, by counting which experts the router actually
  selected over the holdout blocks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any
from typing import Mapping

from ..base_identity import resolve_base_identity
from ..lifecycle import (
    PHASE_BASELINE_GENERATION,
    PHASE_CANDIDATE_GENERATION,
    PHASE_STEADY_STEPS,
    PhaseTimer,
    cuda_synchronize,
    sampling_device,
    training_lifecycle_ledger,
)
from ..router_payload import apply_router_payload, load_router_payload, payload_matches_model
from ..trainability import _tensor_digest, utilization_by_expert
from ..worker_env import chowder_source_identity
from .device_preflight import GIB
from .router_healing import EVAL_WORKER_RESULT_KIND, QUALIFIED_DEVICES, RouterHealingEvalSpec
from .router_healing_load import load_with_policy

#: The probe whose logits demonstrate the payload reached the routing path.
_PROBE_TEXT = "the router chooses an expert"

# Routing-invariance tolerance for the uniform-shift identity control.
# Softmax over unchanged gate logits is mathematically identical; the only
# differences a routing-invariant payload can produce here are float
# rounding. 1e-6 is orders of magnitude above fp32 softmax rounding noise
# and orders below any routing change a real payload makes (the test
# fixture's smallest real routing change is ~5e-2).
_ROUTING_ROUNDING_TOLERANCE = 1e-6


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_holdout(spec: RouterHealingEvalSpec) -> str:
    path = Path(spec.holdout_corpus_path)
    if not path.is_file():
        raise RuntimeError(f"holdout corpus not found: {path}")
    actual = _sha256_file(path)
    if actual != spec.holdout_corpus_sha256:
        raise RuntimeError(
            "holdout corpus hash mismatch: the spec declares "
            f"{spec.holdout_corpus_sha256!r} but {path} is {actual!r}"
        )
    return path.read_text(encoding="utf-8")


def _pack_blocks(input_ids: list[int], seq_len: int) -> list[list[int]]:
    usable = len(input_ids) - (len(input_ids) % seq_len)
    if usable < seq_len:
        raise RuntimeError(
            f"the holdout corpus encodes to {len(input_ids)} tokens, fewer than one "
            f"sequence of {seq_len}; refusing to report a metric from nothing"
        )
    return [input_ids[start : start + seq_len] for start in range(0, usable, seq_len)]


def _score(torch: Any, model: Any, batches: list[Any]) -> float:
    """Mean cross-entropy over the scored blocks. Synchronized where relevant."""
    total = 0.0
    with torch.no_grad():
        for batch in batches:
            total += float(model(input_ids=batch, labels=batch).loss.detach())
    return total / len(batches)


def _logits_fingerprint(torch: Any, model: Any, probe: Any) -> dict[str, Any]:
    """Digest of the model's own output for one fixed probe input.

    Uses the shared tensor digest rather than a bespoke hash, so the identity
    control benefits from the same bounded, device-safe read as everything else.
    """
    with torch.no_grad():
        logits = model(input_ids=probe).logits
    record = _tensor_digest(logits)
    return {"digest": record["digest"], "strategy": record["strategy"], "shape": list(logits.shape)}


def _routing_fingerprint(model: Any, spec: Any, tokenizer: Any = None) -> dict[str, Any]:
    """Routing behaviour on the probe input, before vs after a payload lands.

    Bit-identical logits cannot be the identity control: softmax's uniform-
    shift invariance holds in exact arithmetic, and in float32 a uniform gate
    shift can round one ulp away in the rest of the network -- depending on
    the host's elementwise kernel dispatch, which is how a green control
    became a CI coin flip. The invariant that actually distinguishes
    "cannot change routing" from "changed routing" is the routing itself:

    * ``top1_decisions``: per-layer top-1 expert choice per probe position,
      from the gate logits the model actually computed (same hook convention
      as ``_routing_counts``),
    * ``routing_weights``: the softmax over those same logits, so a payload
      that keeps the argmax but rescales expert weights is still caught.

    Deterministic on the fixed probe, device-safe (compared on-device, only
    a few floats cross to the host), and honest about what it is not: it does
    not certify the holdout score, only that the routing path consumed the
    payload.
    """
    import torch

    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            spec.base_model_dir, local_files_only=True
        )
    probe_ids = tokenizer(_PROBE_TEXT, add_special_tokens=False)["input_ids"][
        : spec.seq_len
    ]
    # The probe must live where the model lives: the flow may have loaded on
    # cpu while a CUDA device exists, and torch.cuda.is_available() would
    # silently place the input on the wrong device.
    device = next(model.parameters()).device
    probe = torch.tensor([probe_ids], device=device)
    layer_logits: dict[str, Any] = {}
    handles = []

    def _hook(name: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            logits = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(logits) or logits.dim() < 2:
                return
            layer_logits[name] = logits.detach()

        return hook

    for name, module in model.named_modules():
        if name.endswith("mlp.gate"):
            handles.append(module.register_forward_hook(_hook(name)))
    try:
        with torch.no_grad():
            model(input_ids=probe)
    finally:
        for handle in handles:
            handle.remove()

    if not layer_logits:
        raise RuntimeError(
            "the model exposes no mlp.gate modules, so a routing fingerprint "
            "cannot be taken; refusing to claim routing invariance from nothing"
        )

    decisions: dict[str, list[int]] = {}
    weights: dict[str, list[list[float]]] = {}
    for name in sorted(layer_logits):
        logits = layer_logits[name]
        flat = logits.reshape(-1, logits.shape[-1])
        decisions[name] = flat.argmax(dim=-1).tolist()
        weights[name] = torch.softmax(flat, dim=-1).tolist()
    return {
        "top1_decisions": decisions,
        "routing_weights": weights,
        "probe_length": len(probe_ids),
    }


def _routing_fingerprint_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> float:
    """Largest absolute routing-weight change between two fingerprints.

    ``inf`` when the layers or positions being compared do not line up at
    all: a structural change is the largest possible change, not a missing
    measurement.
    """
    before_layers = before.get("routing_weights") or {}
    after_layers = after.get("routing_weights") or {}
    if sorted(before_layers) != sorted(after_layers):
        return math.inf
    worst = 0.0
    for name, before_rows in before_layers.items():
        after_rows = after_layers[name]
        if len(before_rows) != len(after_rows):
            return math.inf
        for before_row, after_row in zip(before_rows, after_rows):
            if len(before_row) != len(after_row):
                return math.inf
            for before_value, after_value in zip(before_row, after_row):
                worst = max(worst, abs(after_value - before_value))
    return worst


def _routing_counts(torch: Any, model: Any, batches: list[Any]) -> dict[str, list[int]]:
    """Per-layer top-1 expert counts, measured from the router's own logits.

    A forward hook on each ``mlp.gate`` reads the routing logits the model
    actually computed. This is behaviour, not configuration: a router that has
    collapsed onto one expert shows up here and nowhere else.
    """
    counts: dict[str, list[int]] = {}
    handles = []

    def _hook(name: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            logits = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(logits) or logits.dim() < 2:
                return
            rows = logits.detach().reshape(-1, logits.shape[-1]).argmax(dim=-1)
            tally = torch.bincount(rows.cpu(), minlength=int(logits.shape[-1])).tolist()
            existing = counts.get(name)
            if existing is None:
                counts[name] = [int(value) for value in tally]
            else:
                counts[name] = [a + int(b) for a, b in zip(existing, tally)]

        return hook

    for name, module in model.named_modules():
        if name.endswith("mlp.gate"):
            handles.append(module.register_forward_hook(_hook(name)))
    try:
        with torch.no_grad():
            for batch in batches:
                model(input_ids=batch, labels=batch)
    finally:
        for handle in handles:
            handle.remove()
    return counts


def evaluate(spec: RouterHealingEvalSpec) -> dict[str, Any]:
    """Apply a verified payload and measure the result."""
    if spec.device not in QUALIFIED_DEVICES:
        raise RuntimeError(
            f"device {spec.device!r} is not qualified for router evaluation; qualified: "
            f"{list(QUALIFIED_DEVICES)}"
        )

    import torch

    synchronize = cuda_synchronize(torch)
    device = torch.device(spec.device)
    accelerator_count = 0 if device.type == "cpu" else 1
    started = time.perf_counter()

    # P11 rung 2: on a non-CPU device, measure the device before scoring. The
    # free-memory reading is the budget context the score was measured under;
    # peak accounting starts here so the score's own footprint is measured.
    device_preflight: dict[str, Any] | None = None
    if device.type != "cpu":
        device_preflight = {
            "device": str(device),
            "free_memory_bytes": int(torch.cuda.mem_get_info(device)[0]),
        }
        torch.cuda.reset_peak_memory_stats(device)

    base_identity = resolve_base_identity(spec.base_model_dir)
    if base_identity["content_sha256"] != spec.base_content_sha256:
        raise RuntimeError(
            "base identity mismatch: the spec scores against content "
            f"{spec.base_content_sha256!r} but {spec.base_model_dir} is "
            f"{base_identity['content_sha256']!r}"
        )

    model_load = PhaseTimer(synchronize=synchronize)
    with model_load:
        model, tokenizer, load_report = load_with_policy(
            spec.base_model_dir, load_policy=spec.load_policy, device=str(device)
        )
        model.eval()
    load_policy_report: dict[str, Any] = dict(load_report)

    text = _read_holdout(spec)
    encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
    blocks = _pack_blocks(list(encoded), spec.seq_len)
    scored = blocks[: spec.batches]
    if len(scored) < spec.batches:
        raise RuntimeError(
            f"the holdout corpus yields {len(blocks)} blocks but {spec.batches} were "
            "declared; refusing to score fewer blocks than the spec claims"
        )
    batches = [torch.tensor([block], device=device) for block in scored]
    probe = torch.tensor(
        [tokenizer(_PROBE_TEXT, add_special_tokens=False)["input_ids"][: spec.seq_len]],
        device=device,
    )

    baseline_timer = PhaseTimer()
    baseline_timer.__enter__()
    base_loss = _score(torch, model, batches)
    before = _logits_fingerprint(torch, model, probe)
    routing_before = _routing_fingerprint(model, spec, tokenizer)
    baseline_timer.__exit__(None, None, None)

    payload: dict[str, Any] | None = None
    comparison: dict[str, Any] | None = None
    apply_report: dict[str, Any] | None = None
    routing_after: dict[str, Any] | None = None
    candidate_timer: PhaseTimer | None = None
    if spec.payload_dir is None:
        # The base arm. Nothing is loaded and nothing is applied; the score below
        # is the untouched model's, measured on the same holdout blocks. The
        # second generation leg is recorded as *unavailable with a reason* rather
        # than as a zero-duration phase, so this cannot be read as a candidate
        # comparison that happened to be instant.
        after = before
        candidate_loss = base_loss
    else:
        payload = load_router_payload(
            spec.payload_dir, expected_base_content_sha256=spec.base_content_sha256
        )
        comparison = payload_matches_model(model, payload)
        declared = tuple(str(name) for name in spec.expected_parameter_paths)
        provided = tuple(str(name) for name in payload["tensors"])
        if sorted(declared) != sorted(provided):
            raise RuntimeError(
                "the payload does not carry exactly the declared parameter set: "
                f"declared={sorted(declared)}, payload={sorted(provided)}"
            )

        candidate_timer = PhaseTimer()
        candidate_timer.__enter__()
        apply_report = apply_router_payload(
            model, payload, expected_parameter_paths=spec.expected_parameter_paths
        )
        after = _logits_fingerprint(torch, model, probe)
        routing_after = _routing_fingerprint(model, spec, tokenizer)
        candidate_loss = _score(torch, model, batches)
        candidate_timer.__exit__(None, None, None)
        routing_top1_equal = (
            routing_before["top1_decisions"] == routing_after["top1_decisions"]
        )
        max_abs_routing_weight_delta = _routing_fingerprint_delta(
            routing_before, routing_after
        )
        routing_unchanged = bool(
            routing_top1_equal
            and max_abs_routing_weight_delta <= _ROUTING_ROUNDING_TOLERANCE
        )

    counts = _routing_counts(torch, model, batches)
    utilization = utilization_by_expert(counts or None)
    dead_experts = (
        sum(
            sum(1 for count in layer if count == 0)
            for layer in counts.values()
        )
        if counts
        else 0
    )
    experts_per_tok = int(getattr(model.config, "num_experts_per_tok", 0))

    total_wall = time.perf_counter() - started
    # The shared builder defines what a complete lifecycle looks like, so it is
    # reused rather than re-listed; the two entries that describe a *training*
    # run are then corrected, because a wrong reason recorded beside a real
    # number is exactly the kind of plausible-looking evidence this codebase
    # refuses. Unlike the text evaluator, this process measures both arms, so
    # neither is left unknown for being "another process".
    ledger = training_lifecycle_ledger(
        accelerator_count=accelerator_count,
        model_load=model_load,
        steady_state_steps_seconds=None,
        detailed_timing_enabled=spec.detailed_timing,
        closeout_seconds=None,
    )
    ledger.record_unavailable(
        PHASE_STEADY_STEPS, "this process scores a model; it runs no training loop"
    )
    ledger.record(PHASE_BASELINE_GENERATION, baseline_timer.seconds, synchronized=False)
    if candidate_timer is None:
        ledger.record_unavailable(
            PHASE_CANDIDATE_GENERATION,
            "the base arm measures the untouched model only, so there is no candidate "
            "generation leg to time",
        )
    else:
        ledger.record(PHASE_CANDIDATE_GENERATION, candidate_timer.seconds, synchronized=False)

    payload_arm = payload is not None
    if payload_arm:
        assert comparison is not None and apply_report is not None
        application_control = {
            "payload_kind": payload["payload_kind"],
            "parameters_changed": bool(comparison["differing"]),
            "outputs_changed": before["digest"] != after["digest"],
            "identity_payload": comparison["is_identity"],
            "parameters_comparing_equal": comparison["identical"],
            "parameters_differing": comparison["differing"],
            "applied_parameters": apply_report["applied_parameters"],
            "logits_before": before,
            "logits_after": after,
            "routing_fingerprint_before": routing_before,
            "routing_fingerprint_after": routing_after,
            "routing_top1_equal": routing_top1_equal,
            "max_abs_routing_weight_delta": max_abs_routing_weight_delta,
            "routing_unchanged_beyond_rounding": routing_unchanged,
        }
        payload_verification = {
            "payload_dir": payload["payload_dir"],
            "manifest_sha256": payload["manifest_sha256"],
            "payload_kind": payload["payload_kind"],
            "base_content_sha256": payload["base_content_sha256"],
            "parameter_names": payload["parameter_names"],
        }
    else:
        # `kind: none` is a positive statement, not an absent field: a consumer
        # that reads this document can distinguish "measured the base" from
        # "forgot to record what was applied".
        application_control = {
            "payload_kind": "none",
            "parameters_changed": False,
            "outputs_changed": False,
            "identity_payload": False,
            "parameters_comparing_equal": None,
            "parameters_differing": [],
            "applied_parameters": [],
            "logits_before": before,
            "logits_after": before,
        }
        payload_verification = None

    return {
        "kind": EVAL_WORKER_RESULT_KIND,
        "arm": "candidate" if payload_arm else "base",
        "spec_digest": spec.digest(),
        "metrics": {
            "holdout_loss": candidate_loss,
            "experts_per_token": float(experts_per_tok),
            "dead_experts": float(dead_experts),
        },
        "metric_sources": {
            "holdout_loss": (
                "measured: mean cross-entropy over held-out blocks"
                if payload_arm
                else "measured: mean cross-entropy over held-out blocks, untouched base"
            ),
            "experts_per_token": (
                "configuration read: model.config.num_experts_per_tok, not a measured "
                "routing behaviour"
            ),
            "dead_experts": (
                "measured: experts receiving no top-1 assignment across the holdout blocks"
            ),
        },
        "base_holdout_loss": base_loss,
        "candidate_holdout_loss": candidate_loss if payload_arm else None,
        "holdout_loss_delta": candidate_loss - base_loss if payload_arm else None,
        "application_control": application_control,
        "payload_verification": payload_verification,
        "routing": {"per_layer_top1": counts, "utilization": utilization},
        "base_identity": base_identity,
        "load_policy_report": load_policy_report,
        "lifecycle": ledger.to_dict(),
        "holdout": {
            "path": spec.holdout_corpus_path,
            "sha256": spec.holdout_corpus_sha256,
            "blocks_available": len(blocks),
            "blocks_scored": len(scored),
            "seq_len": spec.seq_len,
        },        "source_identity": chowder_source_identity(),
        "device_preflight": device_preflight,
        "resource_usage": {
            "wall_seconds": total_wall,
            "active_accelerator_count": accelerator_count,
            "visible_accelerator_count": accelerator_count,
            "peak_vram_gb_by_accelerator": (
                {}
                if device.type == "cpu"
                else {"0": round(torch.cuda.max_memory_allocated(device) / GIB, 6)}
            ),
            "sampling_device": sampling_device(torch),
        },
    }

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--chowder-identity", default=None)
    args = parser.parse_args()

    from ..worker_env import verify_source_identity

    if args.chowder_identity is not None:
        verify_source_identity(json.loads(Path(args.chowder_identity).read_text(encoding="utf-8")))
    else:
        print(
            "WARNING: no --chowder-identity supplied; the evaluator's source identity is "
            "unverified for this run",
            file=sys.stderr,
        )

    spec = RouterHealingEvalSpec(**json.loads(Path(args.spec).read_text(encoding="utf-8")))
    result = evaluate(spec)
    Path(args.result).write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
