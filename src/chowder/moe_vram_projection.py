"""What an int8 placement step *should* cost, from the checkpoint's headers alone.

The placement's preregistration argues with arithmetic: both censuses were run on
the CPU while a training run held the card, so ``memory_vram`` still reads "not
measured" and the claim about what the dtype fix does to a step lives in prose.
Prose cannot be checked, and a number in prose is indistinguishable from a guess.

This module makes that claim a command. It reads the safetensors **headers** --
a few hundred kilobytes for 26 shards, no tensor data, no model, no device -- and
derives what each part of a placement step must occupy. Two of the terms are not
estimates at all:

* **The int8 grid.** The loader allocates exactly one int8 byte per expert
  parameter and one fp32 scale per weight row (``_empty_int8_experts``), so
  summing ``numel`` and ``prod(shape[:-1]) * 4`` over the expert keys has to
  reproduce the census' ``expert_bytes`` to the byte. ``disagreements`` treats a
  mismatch as a refusal rather than a rounding difference: it would mean this
  module and the loader disagree about what the placement is, and until they
  agree neither number can be trusted.
* **The resident non-expert parameters.** ``numel * itemsize(dtype)`` at the dtype
  the loader resolves -- the same rule that produced 4.55 GiB where the first
  census held 9.09.

The rest is bounded arithmetic with named terms: one layer's experts dequantized
at the compute dtype, the row-block scratch that dequantisation allocates on the
card, and -- for the gate's own loss expression only -- the logits and their fp32
copies.

What it does **not** size is the activation term. How much autograd retains per
layer is an implementation detail of the model classes (two decoder layer kinds
on this hybrid, 40 layers of it), and inventing a number there is how a
projection becomes a fake measurement. So the reported step figure is a *floor*
and the row that would make it a peak says ``unsized``.

The imports below are the loader's own constants, not copies: a projection that
disagreed with the loader about which keys are ignored, or about how wide a row
block is, would disagree about the bytes.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

from chowder.backends.moe_int8_experts import (
    DEQUANT_ELEMENTS_PER_BLOCK,
    _UNUSED_TOWER_PREFIXES,
)
from chowder.backends.moe_transient_grad import EXPERT_PARAM_INFIX

#: Element sizes by safetensors dtype name. Only these appear in a checkpoint; a
#: name outside the table is a refusal, not a zero.
DTYPE_SIZES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}

#: Torch-side names the loader resolves to, for the same reason.
TORCH_DTYPE_SIZES = {
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "float64": 8,
}

#: A scale is one fp32 value per weight row (``_empty_int8_experts``).
SCALE_BYTES = 4

#: A projection that misses a recorded measurement by more than this is a
#: disagreement, not a rounding difference. 50 MiB, far above the two-decimal
#: rounding in the gate table and far below anything structural.
TOLERANCE_GIB = 0.05

GIB = 2**30


def shard_paths(base: str | Path) -> list[Path]:
    """The checkpoint's shards: the index's order when there is an index."""
    base = Path(base)
    index_path = base / "model.safetensors.index.json"
    shards: list[Path] = []
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        for name in (index.get("weight_map") or {}).values():
            path = base / str(name)
            if path not in shards:
                shards.append(path)
    if not shards:
        shards = sorted(base.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors shards under {str(base)!r}")
    return shards


def header(path: str | Path) -> dict[str, Any]:
    """One shard's header: ``name -> {dtype, shape, data_offsets}``, no tensors.

    The first eight bytes are the header length, then that many bytes of JSON.
    Reading it costs the header and nothing else -- on reap160 the whole
    inventory is well under a megabyte across 26 shards.
    """
    with Path(path).open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        raw = json.loads(handle.read(size))
    raw.pop("__metadata__", None)
    return raw


def _numel(shape: tuple[int, ...]) -> int:
    total = 1
    for extent in shape:
        total *= int(extent)
    return total


def dequant_block_bytes(
    in_features: int,
    *,
    itemsize: int,
    rows_available: int | None = None,
    scale_bytes: int = SCALE_BYTES,
) -> int:
    """Peak card-side scratch for dequantizing one matrix, one row block at a time.

    ``_dequantize_block`` holds the staged int8 block (1 byte/element), the fp32
    product (4), and a cast copy in the destination dtype before ``copy_`` writes
    it into the layer (``itemsize``), plus one fp32 scale per row. The matrix
    being filled is counted separately as the layer's own dequantized weight.

    A block is at most ``DEQUANT_ELEMENTS_PER_BLOCK`` elements *and* at most the
    whole matrix, so a small matrix is never charged for a block it cannot fill.
    """
    rows = max(1, DEQUANT_ELEMENTS_PER_BLOCK // max(1, in_features))
    if rows_available:
        rows = max(1, min(rows, int(rows_available)))
    elements = rows * max(1, in_features)
    return elements * (1 + 4 + itemsize) + rows * scale_bytes


def inventory(base: str | Path, *, dtype: str) -> dict[str, Any]:
    """What the checkpoint holds, by role, read from the headers.

    ``dtype`` is the torch-side name the loader resolved for the non-expert
    parameters; expert weights are summed at their int8 size regardless of what
    the shards store, because that is what the loader allocates for them.
    """
    if dtype not in TORCH_DTYPE_SIZES:
        raise ValueError(f"unknown dtype {dtype!r}; expected one of {sorted(TORCH_DTYPE_SIZES)}")
    itemsize = TORCH_DTYPE_SIZES[dtype]

    base = Path(base)
    shards = shard_paths(base)
    layers: dict[str, list[dict[str, Any]]] = {}
    non_expert_numel = 0
    tower_bytes = 0
    source_dtypes: set[str] = set()
    tensors = 0
    for shard in shards:
        for name, meta in header(shard).items():
            tensors += 1
            shape = tuple(int(extent) for extent in meta.get("shape") or ())
            source = str(meta.get("dtype"))
            source_dtypes.add(source)
            if source not in DTYPE_SIZES:
                raise ValueError(f"{name!r} in {shard.name} has dtype {source!r}, which this inventory cannot size")
            numel = _numel(shape)
            if EXPERT_PARAM_INFIX in name:
                layer = name.rpartition(EXPERT_PARAM_INFIX)[0] + EXPERT_PARAM_INFIX.rstrip(".")
                layers.setdefault(layer, []).append(
                    {
                        "name": name,
                        "shape": shape,
                        "numel": numel,
                        "rows": _numel(shape[:-1]) if len(shape) > 1 else 1,
                        "in_features": int(shape[-1]) if shape else 1,
                    }
                )
                continue
            if name.startswith(_UNUSED_TOWER_PREFIXES):
                tower_bytes += numel * DTYPE_SIZES[source]
                continue
            non_expert_numel += numel

    if not layers:
        raise ValueError(
            f"no routed-expert tensors under {str(base)!r}: this is not a MoE checkpoint "
            f"(looked for {EXPERT_PARAM_INFIX!r}), so the int8 placement says nothing about it"
        )

    expert_int8 = sum(tensor["numel"] for tensors_ in layers.values() for tensor in tensors_)
    expert_scale = sum(tensor["rows"] for tensors_ in layers.values() for tensor in tensors_) * SCALE_BYTES
    layer_numel = {layer: sum(tensor["numel"] for tensor in tensors_) for layer, tensors_ in layers.items()}
    block = max(
        dequant_block_bytes(
            tensor["in_features"], itemsize=itemsize, rows_available=int(tensor["rows"])
        )
        for tensors_ in layers.values()
        for tensor in tensors_
    )
    return {
        "shards": len(shards),
        "tensors": tensors,
        "source_dtypes": sorted(source_dtypes),
        "dtype": dtype,
        "non_expert_numel": non_expert_numel,
        "non_expert_bytes": non_expert_numel * itemsize,
        "expert_layers": len(layers),
        "expert_matrices_per_layer": sorted({len(tensors_) for tensors_ in layers.values()}),
        "expert_int8_bytes": expert_int8,
        "expert_scale_bytes": expert_scale,
        "expert_bytes": expert_int8 + expert_scale,
        "layer_expert_numel_max": max(layer_numel.values()),
        "layer_dequant_bytes": max(layer_numel.values()) * itemsize,
        "dequant_block_bytes": block,
        "tower_bytes": tower_bytes,
    }


def gate_loss_bytes(*, seq_len: int, vocab_size: int, itemsize: int) -> int:
    """The peak of the gate's own ``logits.float().square().mean()``.

    With fp32 logits ``.float()`` is a no-op and ``.square()`` adds one fp32 copy.
    With bf16 logits it materialises a second fp32 tensor, which is why this term
    grows when the dtype fix lands -- a real cross-entropy step would not pay it.
    """
    elements = int(seq_len) * int(vocab_size)
    copies = 1 if itemsize == 4 else 2
    return elements * itemsize + copies * elements * 4


def project(
    inventory_record: dict[str, Any],
    *,
    seq_len: int = 2048,
    vocab_size: int | None = None,
    activation_bytes: int | None = None,
) -> dict[str, Any]:
    """The placement step's memory terms, in bytes, from an inventory.

    ``step_floor_bytes`` is a lower bound on peak device memory: the resident
    non-expert parameters plus one layer's dequantized experts plus that
    dequantisation's row-block scratch. It is a floor because the activation term
    is not sized here; ``activations_bytes`` is where a measured one goes, and
    nothing else in this module pretends to fill it.
    """
    itemsize = TORCH_DTYPE_SIZES[inventory_record["dtype"]]
    non_expert = inventory_record["non_expert_bytes"]
    layer_dequant = inventory_record["layer_dequant_bytes"]
    block = inventory_record["dequant_block_bytes"]
    resident_host = inventory_record["expert_bytes"] + non_expert

    logits = None
    if vocab_size:
        logits = gate_loss_bytes(seq_len=int(seq_len), vocab_size=int(vocab_size), itemsize=itemsize)

    step_floor = non_expert + layer_dequant + block
    return {
        "dtype": inventory_record["dtype"],
        "seq_len": int(seq_len),
        "vocab_size": int(vocab_size) if vocab_size else None,
        "host_expert_bytes": inventory_record["expert_bytes"],
        "host_expert_int8_bytes": inventory_record["expert_int8_bytes"],
        "host_expert_scale_bytes": inventory_record["expert_scale_bytes"],
        "host_non_expert_bytes": non_expert,
        "host_resident_bytes": resident_host,
        "layer_dequant_bytes": layer_dequant,
        "dequant_block_bytes": block,
        "gate_loss_bytes": logits,
        "step_floor_bytes": step_floor,
        "step_floor_with_gate_loss_bytes": step_floor + (logits or 0),
        "activations_bytes": activation_bytes,
        "inventory": {
            key: inventory_record[key]
            for key in (
                "shards",
                "tensors",
                "source_dtypes",
                "expert_layers",
                "expert_matrices_per_layer",
                "layer_expert_numel_max",
                "non_expert_numel",
                "tower_bytes",
            )
        },
    }


def gib(nbytes: int | float | None) -> float | None:
    return None if nbytes is None else round(float(nbytes) / GIB, 3)


def _cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    return f"{value:.2f}"


def render(
    projection: dict[str, Any],
    recorded: dict[str, Any] | None = None,
    *,
    artifact: dict[str, Any] | None = None,
) -> str:
    """The projection as a table, with whatever has been recorded beside it."""
    recorded = recorded or {}
    proj_host_expert = gib(projection["host_expert_bytes"])
    proj_non_expert = gib(projection["host_non_expert_bytes"])
    proj_resident = gib(projection["host_resident_bytes"])
    proj_layer = gib(projection["layer_dequant_bytes"])
    proj_block = gib(projection["dequant_block_bytes"])
    proj_gate = gib(projection["gate_loss_bytes"])
    proj_floor = gib(projection["step_floor_bytes"])
    proj_floor_gate = gib(projection["step_floor_with_gate_loss_bytes"])
    proj_act = gib(projection["activations_bytes"])

    def delta(projected: float | None, measure: float | None) -> str:
        if projected is None or measure is None:
            return "-"
        return f"{measure - projected:+.2f}"

    rows: list[tuple[str, str, str, str, str]] = [
        (
            "int8 + scales, host resident",
            _cell(proj_host_expert),
            _cell(recorded.get("int8_plus_scales")),
            delta(proj_host_expert, recorded.get("int8_plus_scales")),
            "load_census",
        ),
        (
            "non-expert parameters, host resident",
            _cell(proj_non_expert),
            _cell(recorded.get("non_expert")),
            delta(proj_non_expert, recorded.get("non_expert")),
            "census non_expert_bytes" if recorded.get("non_expert") is not None else "not recorded",
        ),
        (
            "host resident total (the two above)",
            _cell(proj_resident),
            _cell(recorded.get("host_peak")),
            delta(proj_resident, recorded.get("host_peak")),
            "host_rss; the gap is interpreter + torch",
        ),
        (
            "one layer's experts, dequantized on the card",
            _cell(proj_layer),
            "-",
            "-",
            f"{projection['dtype']} at the compute dtype",
        ),
        (
            "dequant block scratch, one matrix at a time",
            _cell(proj_block),
            "-",
            "-",
            "int8 block + fp32 product + cast",
        ),
        (
            "gate loss: logits + fp32 copy + square",
            _cell(proj_gate),
            "-",
            "-",
            "the gate's own expression, not a CE step",
        ),
        (
            "step VRAM floor (weights + one layer + block)",
            _cell(proj_floor),
            _cell(recorded.get("step_vram")),
            delta(proj_floor, recorded.get("step_vram")),
            "a peak cannot be below this",
        ),
        (
            "step VRAM floor with the gate's loss term",
            _cell(proj_floor_gate),
            _cell(recorded.get("step_vram")),
            delta(proj_floor_gate, recorded.get("step_vram")),
            "still a floor",
        ),
        (
            "activations, per layer, retained for backward",
            _cell(proj_act) if proj_act is not None else "unsized",
            "-",
            "-",
            "unsized on purpose" if proj_act is None else "measured",
        ),
    ]

    width = max(len(row[0]) for row in rows)
    lines = [
        "projection from shard headers only (no weights read, no device touched)",
    ]
    if artifact:
        shards = artifact.get("shards")
        shards_note = "" if artifact.get("shards") is None else f", {shards} shards"
        sha = (artifact.get("config_sha256") or "unknown")[:12]
        lines.append(f"  artifact  {artifact.get('path') or ''}  config sha256 {sha}{shards_note}")
    lines.append(
        f"  dtype     {projection['dtype']}, seq {projection['seq_len']}"
        + (f", vocab {projection['vocab_size']}" if projection.get("vocab_size") else "")
        + f", {projection['inventory']['expert_layers']} expert layers"
    )
    lines.append(
        f"  units     GiB; recorded: {recorded.get('source') or 'nothing on record'}"
    )
    lines.append("")
    lines.append(f"  {'component':<{width}}  {'projected':>9}  {'recorded':>9}  {'delta':>7}  note")
    for label, projected, measured, difference, note in rows:
        lines.append(f"  {label:<{width}}  {projected:>9}  {measured:>9}  {difference:>7}  {note}")
    lines.append("")
    lines.append(
        "  memory_vram stays not measured: the rows above are a floor, and the "
        "activation term is what would turn it into a peak."
    )
    if recorded.get("seq_len") and int(recorded["seq_len"]) != int(projection["seq_len"]):
        lines.append(
            f"  the recorded step was measured at seq {recorded['seq_len']}, this "
            f"projection at seq {projection['seq_len']}: compare the step rows only after matching"
        )
    return "\n".join(lines)


def disagreements(
    projection: dict[str, Any],
    recorded: dict[str, Any] | None,
    *,
    tolerance_gib: float = TOLERANCE_GIB,
) -> list[str]:
    """Where the projection and what was measured cannot both be right.

    Two rules, because the two quantities fail differently: a resident size the
    projection *derives* (int8 + scales, non-expert parameters) must match a
    measurement of the same thing in both directions, while a host peak and a
    device peak are upper bounds on their floors and may only be *too low*.
    """
    recorded = recorded or {}
    problems: list[str] = []

    def two_sided(label: str, projected_bytes: int, measured: float | None) -> None:
        if measured is None:
            return
        projected = gib(projected_bytes)
        if abs(measured - projected) > tolerance_gib:
            problems.append(
                f"{label}: projected {projected} GiB, recorded {measured:.2f} GiB "
                f"({measured - projected:+.2f}); the header arithmetic and the measurement "
                "do not describe the same placement"
            )

    def measured_at_least(label: str, floor_bytes: int, measured: float | None, field: str) -> None:
        if measured is None:
            return
        floor = gib(floor_bytes)
        if measured < floor - tolerance_gib:
            problems.append(
                f"{label}: recorded {field} {measured:.2f} GiB is below the projected floor "
                f"{floor} GiB; a measured peak cannot be smaller than what has to be resident"
            )

    two_sided("int8 + scales", projection["host_expert_bytes"], recorded.get("int8_plus_scales"))
    two_sided("non-expert parameters", projection["host_non_expert_bytes"], recorded.get("non_expert"))
    measured_at_least("host resident", projection["host_resident_bytes"], recorded.get("host_peak"), "host_rss")
    measured_at_least("step VRAM", projection["step_floor_bytes"], recorded.get("step_vram"), "memory_vram")
    return problems
