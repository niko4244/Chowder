"""How much a placement step retains for backward, measured one layer at a time.

  python chowder_batch/moe_activation_probe.py --base-model <dir> \
      [--layer-kind linear_attention|full_attention|both] [--seq-len 512] [--out act.json]

The prereg's device budget had never been asked this question. The census sized
the weights (host RAM, and what the card must hold); the activation term was
"unsized" because how much autograd keeps per layer is an implementation detail
of the model classes. It is, but it is a *measurable* one, and it turns out to
dominate: at seq 512 the weights floor is 5.5 GiB while 40 layers retain 12.4.

How it measures, and why that is legitimate without the artifact's weights:

* The model is built from the artifact's **text config** with a single decoder
  layer of the requested kind, on ``meta`` and then ``to_empty`` on the CPU. The
  weights are therefore arbitrary -- but every tensor autograd saves is sized by
  shapes and dtypes, not by values, so the byte count is the real one. Only the
  routing *assignment* depends on weights, and it does not change the totals: a
  layer gathers ``tokens x top_k`` rows either way.
* The layer runs with the placement's own forward installed
  (``install_int8_expert_forward``), not a dense one. That matters: the int8
  transient Function saves three small tensors and recomputes its experts in
  backward, where a dense layer would retain the whole MLP's intermediates.
* ``saved_tensors_hooks`` counts **forward** saves only. The Function's backward
  re-enters autograd to recompute the layer, and those saves are transient;
  counting them would report churn as residency.
* Saves are deduplicated by storage start, because one buffer packed by several
  ops costs the device once.

Two sequence lengths give a slope and an intercept: ``bytes = fixed + slope *
seq``, which is how the per-chunk tensors of the linear-attention block scale
(one chunk is a fixed slice of tokens). That is what makes a projection to the
run's own seq_len an extrapolation with a stated basis rather than a guess.

The numbers are *upper bounds* on retention: overlapping views of one storage are
counted separately, and a tensor an op saved that its own backward frees before
the rest of the model's backward arrives would still be counted.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # noqa: E402

#: ``layer_types`` names this artifact uses; every decoder layer is one or the other.
LAYER_KINDS = ("linear_attention", "full_attention")
EXPERTS_CLASSES = ("Qwen3MoeExperts", "Qwen3_5MoeExperts")
SCHEMA = "chowder.moe-activation-probe.v1"


def _log(message: str) -> None:
    print(message, flush=True)


def build_layer_model(base: str, kind: str, dtype: Any) -> Any:
    """One decoder layer of ``kind`` at the artifact's shapes, on the CPU."""
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(base)
    text = config.get_text_config().to_dict()
    text["num_hidden_layers"] = 1
    text["layer_types"] = [kind]
    composite = config.to_dict()
    composite["text_config"] = text
    one_layer = type(config).from_dict(composite)
    # Weights are never read: only shapes matter to a saved-tensor count, and the
    # artifact's 22 B parameters would not fit in this process anyway.
    import torch

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(one_layer, dtype=dtype)
    model.to_empty(device="cpu")
    return model


def install_placement(model: Any) -> int:
    """The int8 placement on this one-layer model: int8 buffers, frozen experts."""
    import torch

    from chowder.backends import moe_int8_experts as mie

    patched = 0
    for _name, module in model.named_modules():
        if type(module).__name__ not in EXPERTS_CLASSES:
            continue
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        gate_up, down = module.gate_up_proj, module.down_proj
        for buffer, shape, buffer_dtype in (
            ("gate_up_q", gate_up.shape, torch.int8),
            ("gate_up_scale", (*gate_up.shape[:-1], 1), torch.float32),
            ("down_q", down.shape, torch.int8),
            ("down_scale", (*down.shape[:-1], 1), torch.float32),
        ):
            module.register_buffer(buffer, torch.ones(shape, dtype=buffer_dtype), persistent=False)
        patched += 1
    mie.install_int8_expert_forward(model)
    return patched


def measure(
    base: str,
    kind: str,
    seq_len: int,
    *,
    dtype_name: str = "bfloat16",
    gradient_checkpointing: bool = False,
) -> dict[str, Any]:
    import torch

    dtype = getattr(torch, dtype_name)
    started = time.perf_counter()
    model = build_layer_model(base, kind, dtype)
    built = time.perf_counter() - started
    if gradient_checkpointing:
        # What a worker run does by default: each layer re-runs its forward in
        # backward instead of keeping its activations, so what is *retained*
        # between layers collapses to the layer's input.
        model.gradient_checkpointing_enable()
    patched = install_placement(model)
    text = model.model
    torch.manual_seed(0)
    tokens = torch.randint(0, text.config.vocab_size, (1, seq_len))

    saved = {"bytes": 0, "tensors": 0, "packed": 0, "largest": 0}
    seen: set[tuple[int, int, int]] = set()
    forward = {"open": True}

    def pack(tensor: Any) -> Any:
        if forward["open"]:
            saved["packed"] += 1
            size = tensor.numel() * tensor.element_size()
            key = (
                tensor.untyped_storage().data_ptr(),
                tensor.storage_offset() * tensor.element_size(),
                size,
            )
            if key not in seen:
                seen.add(key)
                saved["bytes"] += size
                saved["tensors"] += 1
                saved["largest"] = max(saved["largest"], size)
        return tensor

    started = time.perf_counter()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        hidden = text.get_input_embeddings()(tokens).detach().requires_grad_(True)
        output = text(inputs_embeds=hidden, use_cache=False)
        forward["open"] = False
        loss = output.last_hidden_state.float().square().mean()
        loss.backward()
    elapsed = time.perf_counter() - started

    return {
        "layer_kind": kind,
        "seq_len": seq_len,
        "dtype": dtype_name,
        "placement": "cpu-transient-grad-int8",
        "gradient_checkpointing": bool(gradient_checkpointing),
        "experts_modules_patched": patched,
        "saved_bytes": saved["bytes"],
        "saved_gib": round(saved["bytes"] / 2**30, 4),
        "saved_tensors": saved["tensors"],
        "packed_tensors": saved["packed"],
        "largest_saved_gib": round(saved["largest"] / 2**30, 4),
        "build_seconds": round(built, 1),
        "step_seconds": round(elapsed, 1),
    }


def fit(points: list[dict[str, Any]]) -> dict[str, Any]:
    """``bytes = fixed + slope * seq`` from the two **shortest** measured lengths.

    A third point is a check, not a fit: including it would make the line agree
    with itself. ``predicted_at`` reports every point against the line so the
    check is visible rather than assumed.
    """
    if len(points) < 2:
        return {}
    (first, second) = sorted(points, key=lambda point: point["seq_len"])[:2]
    delta_seq = second["seq_len"] - first["seq_len"]
    slope = (second["saved_bytes"] - first["saved_bytes"]) / delta_seq
    fixed = first["saved_bytes"] - slope * first["seq_len"]
    return {
        "fixed_bytes": round(fixed),
        "bytes_per_token": round(slope, 1),
        "measured_at": [first["seq_len"], second["seq_len"]],
        "predicted_at": [
            {
                "seq_len": point["seq_len"],
                "predicted_gib": round((fixed + slope * point["seq_len"]) / 2**30, 4),
                "measured_gib": point["saved_gib"],
            }
            for point in points
        ],
    }


def model_totals(
    fits: dict[str, dict[str, Any]], counts: dict[str, int], kinds: tuple[str, ...], seq_lens: list[int]
) -> dict[str, float | None]:
    """The whole model's retention at each measured length, or ``None``.

    ``None`` rather than 0.0 when a kind has no line to extrapolate: a single
    measured point cannot be projected, and a zero there would read as "this kind
    costs nothing" instead of "this kind was not fitted".
    """
    totals: dict[str, float | None] = {}
    for seq_len in sorted(set(seq_lens)):
        if any(not fits.get(kind) for kind in kinds):
            totals[str(seq_len)] = None
            continue
        total = sum(
            (fits[kind]["fixed_bytes"] + fits[kind]["bytes_per_token"] * seq_len) * counts[kind]
            for kind in kinds
        )
        totals[str(seq_len)] = round(total / 2**30, 2)
    return totals


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model", required=True, help="local checkpoint directory (config only is read)")
    parser.add_argument("--layer-kind", default="both", choices=(*LAYER_KINDS, "both"))
    parser.add_argument("--seq-len", type=int, nargs="+", default=[128, 512], help="one or more sequence lengths")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="recompute each layer's forward in backward, as a worker run does by default",
    )
    parser.add_argument("--layers-per-kind", default="", help="override the artifact's own layer counts, e.g. 30,10")
    parser.add_argument("--out", default="", help="where to write the record (JSON)")
    args = parser.parse_args(argv)

    base = Path(args.base_model)
    if not base.is_dir():
        print(f"base model directory not found: {base}", file=sys.stderr)
        return 2

    kinds = LAYER_KINDS if args.layer_kind == "both" else (args.layer_kind,)
    points: list[dict[str, Any]] = []
    for kind in kinds:
        for seq_len in sorted(set(args.seq_len)):
            _log(f"measuring {kind} at seq {seq_len} ...")
            point = measure(
                str(base),
                kind,
                seq_len,
                dtype_name=args.dtype,
                gradient_checkpointing=args.gradient_checkpointing,
            )
            points.append(point)
            _log(f"  {point['saved_gib']} GiB retained, {point['saved_tensors']} tensors")

    counts = {"linear_attention": 30, "full_attention": 10}
    if args.layers_per_kind:
        values = [int(value) for value in args.layers_per_kind.split(",")]
        counts = dict(zip(LAYER_KINDS, values))
    fits = {
        kind: fit([point for point in points if point["layer_kind"] == kind]) for kind in kinds
    }
    record = {
        "schema": SCHEMA,
        "base_model": str(base),
        "dtype": args.dtype,
        # The record says which step it measured: a step measured with per-layer
        # recompute and one measured without retain different amounts, and a
        # consumer that had to infer the mode from a filename would eventually
        # read one as the other.
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "points": points,
        "fits": fits,
        "layers_per_kind": counts,
        "model_total_gib": model_totals(
            fits, counts, kinds, [point["seq_len"] for point in points]
        ),
        "note": (
            "Forward-retained, deduplicated by storage, with the int8 placement's "
            "forward installed. Upper bound: overlapping views of one storage count "
            "separately. The model total is the fitted line extrapolated to each "
            "measured seq_len and multiplied by the layer counts."
        ),
    }
    payload = json.dumps(record, indent=2)
    if args.out:
        Path(args.out).write_text(payload, encoding="utf-8")
        _log(f"wrote {args.out}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
