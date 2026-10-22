"""int8 frozen experts for the 16 GB MoE training path (option B).

``cpu-transient-grad-bulk`` keeps the routed experts in host RAM as bf16 and
copies one layer's matrices to the training device per pass. That works while
the host has room: for REAP-160 Qwen3.6 (160 experts/layer, 42.9 GB checkpoint)
the bf16 experts are ~40 GB, and on a 64 GB machine with ~24-33 GB available the
training step pages to disk.

This module is option B: keep the same frozen experts in RAM as **int8 with one
per-output-row scale**, so the resident expert bytes halve (int8 weights plus an
fp32 scale per output row is 8.5 bits/weight, ~20 GB for the same artifact),
copy the int8 layer to the device per pass and dequantize *there*. The placement
is ``cpu-transient-grad-int8``; the forward/backward contract is the bulk one
(one copy per layer per pass, recompute in backward, nothing but activations and
routing saved), so gradients still reach the hidden states -- and every LoRA
leaf below the layer -- and ``top_k_weights``.

Two things this module exists to guarantee:

* **Peak host RAM never holds the bf16 experts.** ``load_model_int8_experts``
  builds the model skeleton on ``meta``, then streams the safetensors shards
  tensor by tensor: an expert tensor is quantized as soon as it is read and the
  bf16 values are dropped before the next one is touched, so at most one expert
  tensor is live at a time. Everything else is loaded to the device.
  Quantisation itself runs in row blocks straight into the preallocated int8 and
  scale stores (``QUANTIZE_ELEMENTS_PER_BLOCK``), because a *fused* expert
  tensor is layer-sized -- ~1 GB of bf16 for reap160's 160 experts -- and the
  ``round(w.float() / scale).clamp(...)`` chain needs roughly nine times a
  tensor's bf16 bytes in fp32 intermediates. Doing that per tensor is what put
  the reap160 load at 37.8 GB of private bytes against the ~20 GB the int8
  experts should occupy, and the host heap keeps that high-water mark committed
  long after the load returns. The load reports its progress on stderr, so a
  long one is observable.
* **Nothing bf16 is left for the experts.** The expert parameters are removed
  from each experts module (before the skeleton is materialised, so they are
  never allocated) and replaced by four CPU buffers: ``gate_up_q``,
  ``gate_up_scale``, ``down_q``, ``down_scale``.

Numerically this is not the bulk path: each expert weight row is rounded to a
127-level grid, so an expert's output carries up to ~0.4% relative error per
element (half a step of ``absmax/127``). The router and every LoRA leaf still see
gradients computed through the dequantized weights, exactly as the bulk path
computes them through the bf16 weights.

Nothing layer-sized is dequantized but the weights themselves: ``dequantize_rows``
and ``_dequantized_on`` walk row blocks, so the card holds one block of int8 plus
one block's fp32 product at a time, never an fp32 copy of the layer beside its
bf16 weights.

Design: docs/DESIGN-MOE-16GB-TRAINING-BACKEND-2026-10-18.md (option B). This
module is the mechanism only; the placement's preregistered gates must be
re-measured on the real artifact before any campaign declares it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

try:  # the light CI job has no safetensors, and this module's constants are read there
    from safetensors import safe_open
except ImportError:  # pragma: no cover - exercised by the dev-only CI jobs
    safe_open = None  # type: ignore[assignment]

from .moe_transient_grad import (
    EXPERTS_CLASSES,
    EXPERT_PARAM_INFIX,
    INT8_PLACEMENT,
    _moe_layer,
)

#: The placement name for this path (also in ``moe_transient_grad.EXPERT_PLACEMENTS``).
INT8_PLACEMENTS = (INT8_PLACEMENT,)

#: The four CPU buffers an int8 experts module carries instead of its bf16 params.
INT8_EXPERT_BUFFERS = ("gate_up_q", "gate_up_scale", "down_q", "down_scale")

#: The quantisation grid: ``max |w_row| / 127`` -- the largest magnitude a signed
#: int8 row can express, so the worst per-element error is half of one step.
INT8_MAX = 127


#: Quantisation streams a matrix through row blocks of at most this many
#: elements. The working set of a block is the block itself plus one fp32
#: scratch buffer (``abs`` copy, fp32 copy, then round/clamp in place), so ~8M
#: elements is ~48 MB -- not the ~9x a whole layer-sized tensor would need.
#: Whole-layer blocks are what held reap160 at 37.8 GB private after load: a
#: 1 GB fused ``gate_up_proj`` quantised in one go needs ~8 GB of fp32
#: intermediates, and the host heap keeps that high-water mark committed.
QUANTIZE_ELEMENTS_PER_BLOCK = 1 << 23


def _rows_per_block(row_elements: int, limit: int) -> int:
    """How many rows of ``row_elements`` fit in ``limit`` elements (at least one)."""
    return max(1, limit // max(1, row_elements))


def _quantize_block(block: Any) -> tuple[Any, Any]:
    """``(int8, fp32 scale)`` for a 2-D row block, with one fp32 scratch buffer.

    The arithmetic is the same expression the whole-tensor version used --
    ``round(w.float() / scale).clamp(-127, 127)`` -- with ``/``, ``round`` and
    ``clamp`` done in place on a single fp32 copy, so the int8 bytes are
    identical to a block-agnostic quantisation.
    """
    import torch

    absmax = block.abs().amax(dim=-1, keepdim=True).float()
    # A row that is entirely zero keeps a scale of 1.0 rather than 0.0: dividing
    # by zero would produce NaNs, while an all-zero int8 row dequantizes back to
    # 0 with either scale.
    scale = torch.where(absmax > 0, absmax / float(INT8_MAX), torch.ones_like(absmax))
    # ``to(float32)`` aliases its input when it is already fp32, and the rest of
    # this function mutates in place, so the copy has to be explicit.
    scratch = block.to(torch.float32, copy=True)
    scratch.div_(scale)
    scratch.round_()
    scratch.clamp_(-INT8_MAX, INT8_MAX)
    return scratch.to(torch.int8), scale


def _quantize_rows_into(q_out: Any, scale_out: Any, rows: Any) -> None:
    """Quantize ``rows`` (2-D) into the preallocated int8/scale stores, in blocks.

    The destination is written block by block, so neither an int8 copy of the
    input nor an fp32 copy of it ever exists whole: only one block of each.
    """
    n_rows = rows.shape[0]
    if n_rows == 0:
        return
    block = _rows_per_block(rows.shape[1], QUANTIZE_ELEMENTS_PER_BLOCK)
    for start in range(0, n_rows, block):
        stop = start + block
        q_block, scale_block = _quantize_block(rows[start:stop])
        q_out[start:stop].copy_(q_block)
        scale_out[start:stop].copy_(scale_block)


def quantize_rows(weight: Any, *, scale_dtype: Any = None) -> tuple[Any, Any]:
    """Symmetric per-output-row int8 quantisation of a weight matrix.

    The last dimension is the *input* dimension (``F.linear`` contracts it), so
    each row over the last dim is one output feature: ``scale = max|row| / 127``
    and ``q = round(w / scale)`` clamped to ``[-127, 127]``.

    A row that is entirely zero keeps a scale of 1.0 rather than 0.0: dividing by
    zero would produce NaNs, while an all-zero int8 row dequantizes back to 0
    with either scale.

    Returns ``(q[int8], scale)`` with the scale in ``scale_dtype`` (fp32 by
    default: it is one value per output row, 0.5 byte per 4 weights). The matrix
    is quantised in row blocks (``QUANTIZE_ELEMENTS_PER_BLOCK``), so this never
    allocates a whole-matrix fp32 intermediate; ``q`` is the destination itself.
    """
    import torch

    if scale_dtype is None:
        scale_dtype = torch.float32
    features = weight.shape[-1]
    q = torch.empty(weight.shape, dtype=torch.int8)
    scale = torch.empty((*weight.shape[:-1], 1), dtype=scale_dtype)
    # The stores are the only matrix-sized allocations; the input is reshaped,
    # never copied, so a whole-matrix fp32 intermediate never exists.
    _quantize_rows_into(
        q.reshape(-1, features),
        scale.reshape(-1, 1),
        weight.detach().reshape(-1, features),
    )
    return q, scale


#: Dequantisation works through row blocks of at most this many elements, so a
#: block's fp32 product stays under ~32 MB. The fp32 product of a whole expert
#: matrix would be twice its bf16 bytes on the card, which the 16 GB budget
#: cannot spend on top of the copy.
DEQUANT_ELEMENTS_PER_BLOCK = 1 << 23

#: How many elements one host->device copy of the int8 store moves. Separate from
#: the dequantisation block above because the two limits answer different
#: questions: the copy wants few, large transfers (each pageable one blocks the
#: stream), the fp32 product wants a small working set. 2**26 elements is 64 MiB
#: of int8 per copy -- 32 rows of the artifact's ``gate_up_proj`` (5 copies a
#: layer instead of 40) and 4x that of ``down_proj`` -- while the dequantised
#: output stays capped by ``DEQUANT_ELEMENTS_PER_BLOCK`` whatever this is.
DEQUANT_COPY_ELEMENTS = 1 << 26


#: Whether the backward reuses the weights the recompute dequantised. Three passes
#: dequantise a layer (checkpointed forward, recompute, backward); the recompute's
#: copy is the one whose backward follows immediately, so keeping it saves a whole
#: pass. **On by default since 2026-10-21**, because it is now measured rather than
#: projected: on REAP-160 at seq 2048 the A/B in
#: ``docs/quals/MOE_INT8_THROUGHPUT_2026-10-20.md`` measured 32.63 -> 27.48 s/step
#: (15.9%) for **0.000 GiB** of extra peak -- the retained buffer replaces the
#: backward's own dequantisation instead of sitting beside it, so it is not the
#: ~0.94 GiB the first design note assumed. Opt out with
#: ``CHOWDER_INT8_RETAIN_WEIGHTS=0``; ``retain_recomputed_weights(False)`` still
#: forces it off for a phase, which is how the A/B measures the other arm.
_RETAIN_ENV = "CHOWDER_INT8_RETAIN_WEIGHTS"
_retain_override: bool | None = None


def retain_recomputed_weights(enabled: bool | None = None) -> bool:
    """Read, or set for the process, whether backward reuses the recompute's weights."""
    global _retain_override
    if enabled is not None:
        _retain_override = bool(enabled)
        return _retain_override
    if _retain_override is not None:
        return _retain_override
    import os

    # Unset means on. An explicitly empty value is still an opt-out, so
    # ``CHOWDER_INT8_RETAIN_WEIGHTS=`` in a shell profile cannot silently turn the
    # optimisation off and make a run look slower than the default.
    raw = os.environ.get(_RETAIN_ENV)
    if raw is None:
        return True
    return raw.strip().lower() not in ("", "0", "false", "off")


def _in_recompute_pass() -> bool:
    """Whether this forward call is the activation-checkpoint recompute.

    Only the recompute's weights are immediately consumed by their own backward,
    so only they are worth holding. The signal has to be read in the module's
    ``forward`` and passed into the autograd Function: a custom Function's own
    ``forward`` always runs with grad disabled, so grad mode cannot tell the two
    passes apart even from inside, and at the module level BOTH the checkpointed
    first pass and the recompute run with grad enabled. What does differ is the
    autograd graph task: -1 in the first pass, non-negative during the backward
    that triggers the recompute. Measured on torch 2.11; a torch without the
    introspection retains nothing, which only costs the optimization.
    """
    import torch

    graph_task_id = getattr(torch._C, "_current_graph_task_id", None)
    if graph_task_id is None:  # pragma: no cover - depends on the torch build
        return False
    return graph_task_id() >= 0


#: The one layer whose weights the checkpoint recompute just dequantised, waiting
#: for the backward that follows it: ``(id(gate_up_q), gate_up, down)``.
#:
#: This is a side channel, not ``ctx``, because the recompute's ctx is not the one
#: whose backward runs: the non-reentrant checkpoint replays the FIRST forward's
#: graph and uses the recompute only to refill the saved tensors it dropped, so the
#: node that reaches ``backward`` was built by the pass that ran with grad enabled
#: *before* the slot was worth filling. One slot, because the engine runs a layer's
#: recompute and its backward back to back; a slot that does not match the layer
#: being backwarded is dropped rather than kept, so a missed hand-off costs one
#: copy -- never memory, and never a wrong weight: the int8 buffers are frozen, so
#: even a stale entry would dequantise to the same bytes.
_recomputed_weights: tuple[int, Any, Any] | None = None


def _take_recomputed_weights(gate_up_q: Any) -> tuple[Any, Any] | None:
    """Pop the recompute's hand-off if it belongs to this layer; always clears."""
    global _recomputed_weights
    held, _recomputed_weights = _recomputed_weights, None
    if held is None or held[0] != id(gate_up_q):
        return None
    return held[1], held[2]


def _dequantize_block(out_slice: Any, q_block: Any, scale_block: Any) -> None:
    """``out_slice = (q * scale)`` for one block, multiplied in fp32.

    The multiply is done in fp32 and cast once to the destination dtype: the
    scale is a per-row fp32 value and casting it to bf16 first would move the
    weights by up to half a bf16 ulp before any arithmetic happens.
    """
    import torch

    out_slice.copy_((q_block.to(torch.float32) * scale_block.to(torch.float32)).to(out_slice.dtype))


def _dequantize_into(out: Any, q: Any, scale: Any) -> None:
    """Fill ``out`` (already allocated, ``q``'s shape) from ``q`` and ``scale``.

    Row blocks only: nothing but one block's fp32 product is ever allocated, so
    the caller may hold the layer's dequantized weights exactly once (``out``)
    without a second, fp32, copy of the layer beside it.
    """
    if q.numel() == 0 or out.numel() == 0:
        return
    block = _rows_per_block(q[0].numel(), DEQUANT_ELEMENTS_PER_BLOCK)
    for start in range(0, q.shape[0], block):
        rows = slice(start, start + block)
        _dequantize_block(out[rows], q[rows], scale[rows])


def dequantize_rows(q: Any, scale: Any, dtype: Any = None) -> Any:
    """``q * scale`` with the multiply in fp32, returned in ``dtype``.

    Works on any device: the arithmetic happens wherever ``q`` lives, so a
    caller that copied the int8 weights to the card dequantizes on the card.
    The product is accumulated in row blocks (``DEQUANT_ELEMENTS_PER_BLOCK``),
    so a layer's worth of fp32 never exists at once.
    """
    import torch

    out_dtype = torch.float32 if dtype is None else dtype
    out = torch.empty(q.shape, dtype=out_dtype, device=q.device)
    _dequantize_into(out, q, scale)
    return out


def resolve_device(device: Any = 0) -> Any:
    """``device=0`` means "the training card" -- except where there is none.

    ``CHOWDER_NO_GPU`` (or no CUDA) resolves to the CPU so the CPU tests and any
    no-GPU inspection run the same loader instead of skipping it.
    """
    import os

    import torch

    disabled = os.environ.get("CHOWDER_NO_GPU", "").strip().lower() not in ("", "0", "false")
    if device is None:
        resolved = torch.device("cuda", 0)
    elif isinstance(device, int):
        resolved = torch.device("cuda", device)
    else:
        resolved = torch.device(device)
    if resolved.type != "cuda":
        return resolved
    if disabled or not torch.cuda.is_available():
        return torch.device("cpu")
    return resolved


#: Where a composite (multimodal) config hides the text model's ``dtype``. The
#: REAP-160 artifact says ``text_config.dtype: bfloat16`` and nothing at the top
#: level, so reading only ``config.dtype`` resolved to fp32 and the loader
#: materialised 4.55 GiB of bf16 non-expert weights as 9.09 GiB (2026-10-20
#: census). ``from_pretrained`` keeps them bf16.
_SUB_CONFIG_ATTRS = ("text_config", "llm_config", "language_config", "decoder")

#: safetensors header dtype names -> torch dtypes, for the checkpoint fallback.
_CHECKPOINT_DTYPES = {
    "BF16": "bfloat16",
    "F16": "float16",
    "F32": "float32",
    "F64": "float64",
}


def _torch_dtype(value: Any) -> Any:
    """``value`` as a ``torch.dtype``, or ``None`` if it does not name one."""
    import torch

    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        mapped = getattr(torch, value, None)
        if isinstance(mapped, torch.dtype):
            return mapped
    return None


def _config_dtype(config: Any) -> tuple[Any, str] | None:
    """``(dtype, where)`` from the config or its sub-configs, or ``None``.

    The top level wins, as it does for ``from_pretrained``; a composite config
    that keeps its text model's dtype one level down is read too.
    """
    resolved = _torch_dtype(getattr(config, "dtype", None))
    if resolved is not None:
        return resolved, "config"
    for attribute in _SUB_CONFIG_ATTRS:
        sub = getattr(config, attribute, None)
        if sub is None:
            continue
        resolved = _torch_dtype(getattr(sub, "dtype", None))
        if resolved is not None:
            return resolved, attribute
    return None


def _checkpoint_dtype(shard: Path) -> Any:
    """The first float dtype declared in a shard's header, read without weights.

    ``from_pretrained`` defaults to the checkpoint's own dtype when no config
    names one (verified against transformers 5.16: bf16 shards load as bf16 with
    a dtype-less config), so this is what that default resolves to. Header only:
    no tensor is read, so asking costs a few kilobytes.
    """
    import json
    import struct

    import torch

    with shard.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
    header.pop("__metadata__", None)
    for name in sorted(header):
        mapped = _CHECKPOINT_DTYPES.get(str(header[name].get("dtype")))
        if mapped is not None:
            return getattr(torch, mapped)
    return None


def _resolve_weight_dtype(config: Any, shards: list[Path], dtype: Any = None) -> tuple[Any, str]:
    """The dtype the loader materialises non-expert weights in, and its source.

    The order is the one ``from_pretrained`` uses: an explicit dtype, then the
    config (top level, then the text model's), then the checkpoint's own dtype.
    fp32 remains the fallback for a checkpoint that declares nothing at all.
    """
    import torch

    explicit = _torch_dtype(dtype) if dtype is not None else None
    if dtype is not None and explicit is None:
        raise ValueError(f"dtype must be a torch.dtype, got {dtype!r}")
    if explicit is not None:
        return explicit, "argument"
    from_config = _config_dtype(config)
    if from_config is not None:
        return from_config
    for shard in shards:
        from_checkpoint = _checkpoint_dtype(shard)
        if from_checkpoint is not None:
            return from_checkpoint, "checkpoint"
    return torch.float32, "default"


def resolve_non_expert_dtype(
    base_dir: str | Path,
    *,
    dtype: Any = None,
    config: Any = None,
    revision: str | None = None,
    local_files_only: bool = True,
    trust_remote_code: bool = False,
) -> tuple[Any, str]:
    """The dtype the loader materialises non-expert weights in, loading no weights.

    ``load_model_int8_experts`` resolves this inline from the config it already
    holds; this is the same call for a caller that wants the answer without
    building the model, e.g. a projection that sizes the placement from the
    shard headers. It resolves the dtype with ``_resolve_weight_dtype`` rather
    than re-deriving the rule: two definitions of "what dtype is this checkpoint"
    would eventually disagree, and then every byte figure would too.
    """
    if config is None:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            str(base_dir),
            revision=revision,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
    return _resolve_weight_dtype(config, _shard_files(Path(base_dir)), dtype)


def _experts_modules(model: Any) -> list[tuple[str, Any]]:
    return [
        (name, module)
        for name, module in model.named_modules()
        if type(module).__name__ in EXPERTS_CLASSES
    ]


#: Report host memory every this many expert layers, and at every shard.
PROGRESS_LAYERS = 4


def _host_private_bytes() -> int | None:
    """This process's private (committed) bytes, or ``None`` if unmeasurable.

    Never raises: a progress line must not be able to kill a 40-minute load.
    """
    try:
        import psutil

        info = psutil.Process().memory_info()
        private = getattr(info, "private", None)
        if private is not None:
            return int(private)
        # POSIX has no "private bytes"; unique set size is the same question.
        return int(psutil.Process().memory_full_info().uss)
    except Exception:  # pragma: no cover - telemetry is never fatal
        return None


def _host_private_gb() -> str:
    private = _host_private_bytes()
    return "n/a" if private is None else f"{private / 1024**3:.2f}"


def _log_load(message: str) -> None:
    """A load progress line on stderr, flushed so a long load is observable."""
    print(f"[int8-load] {message}", file=sys.stderr, flush=True)


def _shard_files(base: Path) -> list[Path]:
    """Every safetensors shard of a local checkpoint, in index order when there
    is an index (a sharded checkpoint's order is the index's, not the name's)."""
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
    missing = [str(path) for path in shards if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"the safetensors index names shards that are missing: {missing[:3]}")
    return shards


def _empty_int8_experts(module: Any, *, scale_dtype: Any) -> dict[str, Any]:
    """The int8 storage for one layer, shaped from the skeleton's own params."""
    import torch

    gate_up = module.gate_up_proj
    down = module.down_proj
    return {
        "gate_up_q": torch.empty(gate_up.shape, dtype=torch.int8),
        "gate_up_scale": torch.empty((*gate_up.shape[:-1], 1), dtype=scale_dtype),
        "down_q": torch.empty(down.shape, dtype=torch.int8),
        "down_scale": torch.empty((*down.shape[:-1], 1), dtype=scale_dtype),
    }


def _quantize_into(store: dict[str, Any], which: str, rows: Any, expert: int | None = None, offset: int = 0) -> None:
    """Quantize ``rows`` and write them into the layer's int8 buffers.

    ``expert`` selects the 3-D slice, ``offset`` the row block inside it: a
    legacy checkpoint's ``gate_proj`` and ``up_proj`` are the two halves of the
    fused ``gate_up_proj`` (``chunk(2, dim=-1)`` -> gate first), so they land in
    rows ``[0:inter]`` and ``[inter:2*inter]`` of the fused matrix.

    The read tensor is quantised in row blocks straight into the store, so a
    whole-layer checkpoint tensor (a fused ``gate_up_proj`` is ~1 GB of bf16 on
    reap160) never gets a whole-layer int8 or fp32 companion -- the store is
    already the int8 copy, and only one block of fp32 exists at a time.
    """
    q_target = store[f"{which}_q"]
    s_target = store[f"{which}_scale"]
    if expert is not None:
        q_target = q_target[expert]
        s_target = s_target[expert]
    features = rows.shape[-1]
    # A fused per-layer key arrives 3-D (experts x out x in) and a per-expert key
    # 2-D; both are quantised as a flat row list, which is the same thing.
    total_rows = rows.numel() // features
    q_target = q_target[offset:offset + total_rows]
    s_target = s_target[offset:offset + total_rows]
    _quantize_rows_into(
        q_target.reshape(-1, features),
        s_target.reshape(-1, 1),
        rows.detach().reshape(total_rows, features),
    )


#: Checkpoint key prefixes transformers' own loader rewrites and this streaming
#: loader must rewrite itself. A Qwen3.5/3.6 conditional-generation checkpoint
#: nests the text model one level deeper than ``...ForCausalLM`` does, and REAP
#: exports keep the source keys byte-for-byte (see chowder_batch/reap_prune.py,
#: which matches ``model.language_model.layers.<i>.mlp.experts.gate_up_proj``).
_WRAPPER_PREFIXES = (
    ("model.language_model.", "model."),
    ("language_model.", "model."),
)

#: Tensors a causal-LM skeleton has no module for, recorded rather than treated
#: as a load failure: a conditional-generation export carries its vision tower
#: and MTP head in the same shards.
_UNUSED_TOWER_PREFIXES = (
    "model.visual.",
    "visual.",
    "model.vision_tower.",
    "model.multi_modal_projector.",
    "mtp.",
)


def _model_key(raw: str, params: dict[str, Any], buffers: dict[str, Any]) -> str:
    """The skeleton's name for a checkpoint key, or the key unchanged.

    An exact match wins; otherwise the text model's wrapper prefix is removed
    when the skeleton names that tensor without it.
    """
    if raw in params or raw in buffers:
        return raw
    for prefix, replacement in _WRAPPER_PREFIXES:
        if raw.startswith(prefix):
            candidate = replacement + raw[len(prefix):]
            if candidate in params or candidate in buffers or EXPERT_PARAM_INFIX in candidate:
                return candidate
    return raw


def _expert_key_target(key: str) -> tuple[str, str, int | None, int]:
    """Map an expert checkpoint key to ``(layer_prefix, which, expert, offset)``.

    Accepts the fused layout transformers 5 keeps on the model
    (``...mlp.experts.gate_up_proj`` / ``...mlp.experts.down_proj``) and the
    per-expert layout its own ``save_pretrained`` writes, and older REAP exports
    carry (``...mlp.experts.<e>.gate_proj.weight`` and friends).
    """
    prefix, _, tail = key.rpartition(EXPERT_PARAM_INFIX)
    experts_module = prefix + EXPERT_PARAM_INFIX.rstrip(".")
    if tail in ("gate_up_proj", "down_proj"):
        return experts_module, ("gate_up" if tail == "gate_up_proj" else "down"), None, 0
    stem = tail[:-len(".weight")] if tail.endswith(".weight") else tail
    head, _, leaf = stem.partition(".")
    if head.isdigit() and leaf in ("gate_proj", "up_proj", "down_proj"):
        expert = int(head)
        if leaf == "down_proj":
            return experts_module, "down", expert, 0
        # gate/up are fused with gate first, exactly as ``chunk(2, dim=-1)`` reads it.
        return experts_module, "gate_up", expert, 0 if leaf == "gate_proj" else -1
    raise ValueError(
        f"unrecognised expert tensor key {key!r}: expected {EXPERT_PARAM_INFIX}"
        "gate_up_proj/down_proj or <index>.gate_proj/up_proj/down_proj.weight"
    )


def int8_expert_census(model: Any) -> dict[str, Any]:
    """What the int8 experts actually occupy: buffers, bytes, and their device."""
    int8_bytes = 0
    scale_bytes = 0
    devices: set[str] = set()
    modules = 0
    buffers = 0
    for _, module in _experts_modules(model):
        modules += 1
        for name in INT8_EXPERT_BUFFERS:
            tensor = getattr(module, name, None)
            if tensor is None:
                continue
            buffers += 1
            devices.add(tensor.device.type)
            nbytes = tensor.numel() * tensor.element_size()
            if name.endswith("_q"):
                int8_bytes += nbytes
            else:
                scale_bytes += nbytes
    return {
        "expert_modules": modules,
        "expert_int8_buffers": buffers,
        "expert_int8_bytes": int8_bytes,
        "expert_scale_bytes": scale_bytes,
        "expert_bytes": int8_bytes + scale_bytes,
        "expert_buffer_devices": sorted(devices),
    }


def load_model_int8_experts(
    base_dir: str | Path,
    *,
    device: Any = 0,
    dtype: Any = None,
    scale_dtype: Any = None,
    revision: str | None = None,
    local_files_only: bool = True,
    trust_remote_code: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Load a local MoE checkpoint with its routed experts quantized to int8 in RAM.

    The host never holds the bf16 experts: the skeleton is built on ``meta``, the
    expert parameters are dropped from it before materialisation, and each
    shard's tensors are read one at a time -- expert tensors are quantized
    immediately and dropped before the next key is touched.

    ``dtype`` defaults to what ``from_pretrained`` would use, which is not always
    the config's top-level ``dtype``: a composite checkpoint that keeps it in
    ``text_config`` is read there, and a checkpoint that names none loads in its
    own dtype. The census records which of those applied, because materialising
    bf16 weights as fp32 doubles them -- on REAP-160 that was 4.55 GiB of
    non-expert weights become 9.09 GiB.

    Returns ``(model, census)``. The census carries the expert bytes resident in
    RAM (int8 + scales), the non-expert parameters' device and dtype, the number
    of expert modules the loader found, and any missing/unexpected keys.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    base = Path(base_dir)
    if not base.is_dir():
        raise ValueError(
            f"load_model_int8_experts streams local safetensors shards, so base_dir must be a "
            f"local checkpoint directory; got {str(base_dir)!r}"
        )
    target_device = resolve_device(device)
    config = AutoConfig.from_pretrained(
        str(base),
        revision=revision,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    shards = _shard_files(base)
    resolved_dtype, dtype_source = _resolve_weight_dtype(config, shards, dtype)
    if scale_dtype is None:
        scale_dtype = torch.float32

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=resolved_dtype)

    # Drop the expert parameters BEFORE anything is materialised: ``to_empty``
    # would otherwise allocate their bf16 bytes on the host, which is the whole
    # memory this placement exists to avoid.
    experts = _experts_modules(model)
    if not experts:
        raise RuntimeError(
            f"no experts module ({', '.join(EXPERTS_CLASSES)}) in {str(base)!r}; "
            f"expert_placement={INT8_PLACEMENT!r} needs a routed-expert MoE checkpoint"
        )
    stores: dict[str, dict[str, Any]] = {}
    for name, module in experts:
        stores[name] = _empty_int8_experts(module, scale_dtype=scale_dtype)
        for param_name in ("gate_up_proj", "down_proj"):
            if param_name in module._parameters:
                del module._parameters[param_name]

    model.to_empty(device=target_device)

    # A tied head is one tensor, as ``from_pretrained`` leaves it, and the
    # checkpoint may then carry only the embedding.
    if getattr(config, "tie_word_embeddings", False):
        model.tie_weights()

    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    loaded: set[str] = set()
    unexpected: list[str] = []
    ignored: list[str] = []
    renamed = 0

    # A tensor can have several owners (a tied head is the embedding), and some
    # models initialise a child's tensor from the parent's ``_init_weights``
    # (Qwen3_5MoeSparseMoeBlock re-inits ``gate.weight``). So every owner AND its
    # ancestor chain must be marked, or a later re-initialisation would overwrite
    # the value just loaded.
    owners: dict[int, list[str]] = {}
    for module_name, module in model.named_modules():
        for tensor in list(module._parameters.values()) + list(module._buffers.values()):
            if tensor is not None:
                owners.setdefault(id(tensor), []).append(module_name)

    def mark_initialized(target: Any) -> None:
        """Tell ``initialize_weights`` these modules' tensors came from the
        checkpoint, so only the modules with nothing loaded (the rotary
        buffers) get initialised -- ``from_pretrained``'s own bookkeeping."""
        for module_name in owners.get(id(target), ()):
            parts = module_name.split(".") if module_name else []
            for depth in range(1, len(parts) + 1):
                model.get_submodule(".".join(parts[:depth]))._is_hf_initialized = True

    # Layers whose expert tensors have been streamed so far, in first-seen order;
    # this is the progress counter the load line reports.
    quantized_layers: list[str] = []
    for shard_index, shard in enumerate(shards, start=1):
        _log_load(
            f"shard {shard_index}/{len(shards)}, layers quantized {len(quantized_layers)}, "
            f"host private {_host_private_gb()} GB"
        )
        if safe_open is None:  # the guarded import at the top of the module
            raise ImportError(
                "reading shards needs safetensors; the placement's header arithmetic "
                "imports without it, but loading one does not"
            )
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for raw_key in handle.keys():
                key = _model_key(raw_key, params, buffers)
                if key != raw_key:
                    renamed += 1
                if (
                    key not in params
                    and key not in buffers
                    and key.startswith(_UNUSED_TOWER_PREFIXES)
                ):
                    ignored.append(raw_key)
                    continue
                tensor = handle.get_tensor(raw_key)
                if EXPERT_PARAM_INFIX in key:
                    layer, which, expert, offset = _expert_key_target(key)
                    if layer not in quantized_layers:
                        quantized_layers.append(layer)
                        if len(quantized_layers) % PROGRESS_LAYERS == 0:
                            _log_load(
                                f"shard {shard_index}/{len(shards)}, layers quantized "
                                f"{len(quantized_layers)}, host private {_host_private_gb()} GB"
                            )
                    store = stores.get(layer)
                    if store is None:
                        raise RuntimeError(
                            f"expert tensor {raw_key!r} belongs to {layer!r}, which is not an experts module "
                            f"on this model (experts modules: {sorted(stores)[:3]})"
                        )
                    if offset == -1:
                        offset = store[f"{which}_q"].shape[1] // 2
                    _quantize_into(store, which, tensor, expert=expert, offset=offset)
                    del tensor
                    loaded.add(key)
                    continue
                target = params.get(key)
                if target is None:
                    target = buffers.get(key)
                if target is None:
                    unexpected.append(raw_key)
                    del tensor
                    continue
                with torch.no_grad():
                    # ``copy_`` does the cast and the host->device move itself.
                    # ``tensor.to(device)`` would materialise a second,
                    # device-sized copy of every non-expert tensor first.
                    target.copy_(tensor)
                mark_initialized(target)
                del tensor
                loaded.add(key)

    _log_load(
        f"shard {len(shards)}/{len(shards)}, layers quantized {len(quantized_layers)}, "
        f"host private {_host_private_gb()} GB"
    )
    if unexpected:
        raise RuntimeError(
            f"{len(unexpected)} checkpoint tensor(s) have no home on the model skeleton, "
            f"e.g. {unexpected[:3]}; this loader does not skip tensors silently"
        )

    for name, module in experts:
        store = stores.pop(name)
        for buffer_name in INT8_EXPERT_BUFFERS:
            module.register_buffer(buffer_name, store[buffer_name], persistent=False)
        module._is_hf_initialized = True

    # Non-persistent buffers (the rotary inverse frequencies) are in no
    # checkpoint: ``initialize_weights`` fills exactly the modules this loader
    # did not mark, which is what ``from_pretrained`` does for missing keys.
    model.initialize_weights()
    # ``from_pretrained`` records where the weights came from on the config, and
    # generation/tooling reads it; ``from_config`` blanks it.
    model.config._name_or_path = str(base)

    missing = [
        name
        for name in params
        if name not in loaded and not (getattr(config, "tie_word_embeddings", False) and name == "lm_head.weight")
    ]
    if missing:
        raise RuntimeError(
            f"{len(missing)} model parameter(s) were not in the checkpoint, e.g. {missing[:3]}; "
            "loading random weights for them would be silently wrong"
        )

    census: dict[str, Any] = {
        "loader": "int8-streaming",
        "shards": len(shards),
        "device": str(target_device),
        "dtype": str(resolved_dtype),
        "dtype_source": dtype_source,
        "scale_dtype": str(scale_dtype),
        **int8_expert_census(model),
        "non_expert_params": len(params),
        "non_expert_params_on_device": sum(
            parameter.device.type == target_device.type for parameter in params.values()
        ),
        "non_expert_bytes": sum(
            parameter.numel() * parameter.element_size() for parameter in params.values()
        ),
        "tensors_loaded": len(loaded),
        "keys_renamed": renamed,
        "ignored_keys": sorted(ignored)[:20],
        "ignored_key_count": len(ignored),
        "unexpected_keys": [],
        # Modules that received int8 storage instead of their bf16 parameters;
        # ``prepare_int8_placement`` overwrites this with the patched-forward
        # count once the forward is installed.
        "patched_expert_modules": len(experts),
    }
    return model, census


def _dequantized_on(tensor: Any, scale: Any, device: Any, dtype: Any) -> Any:
    """Dequantize one CPU-resident int8 matrix onto ``device``, in row blocks.

    The result (the layer's weights in ``dtype``, i.e. bf16 in training) is the
    only layer-sized allocation: the int8 matrix is staged to the device one
    *copy block* at a time and each copy's fp32 product is freed before the next,
    so the forward never needs an int8 *and* an fp32 copy of the whole layer
    beside the weights. On the 16 GB card that difference is what decides whether
    the layer fits or spills to WDDM.

    The copy granularity and the dequantisation granularity are separate: a copy
    is as large as ``DEQUANT_COPY_ELEMENTS`` allows (few host->device calls, each
    of which blocks the stream while a pageable source is staged), while the fp32
    product is still capped at ``DEQUANT_ELEMENTS_PER_BLOCK``. Reading them as one
    number is what made a layer cost 60 blocking copies of 8 MiB: measured on the
    card, that granularity ran at 5.52 GiB/s against 5.82 GiB/s for the 10x block,
    and the pinned variant was no faster than the pageable one (5.52 both), so the
    link -- not the pinning -- is the ceiling.
    """
    import torch

    out = torch.empty(tensor.shape, dtype=dtype, device=device)
    if tensor.numel() == 0:
        return out
    row = tensor[0].numel()
    block = _rows_per_block(row, DEQUANT_ELEMENTS_PER_BLOCK)
    copy = max(block, _rows_per_block(row, DEQUANT_COPY_ELEMENTS))
    for start in range(0, tensor.shape[0], copy):
        stop = min(start + copy, tensor.shape[0])
        q_block = tensor[start:stop].to(device, non_blocking=True)
        scale_block = scale[start:stop].to(device, non_blocking=True)
        for sub in range(0, stop - start, block):
            rows = slice(start + sub, min(start + sub + block, stop))
            _dequantize_block(out[rows], q_block[sub:sub + block], scale_block[sub:sub + block])
        del q_block, scale_block
    return out


def _int8_layer_function() -> Any:
    import torch

    class TransientLayerExpertsInt8(torch.autograd.Function):
        """A whole layer's frozen int8 experts: copied in, dequantized on the
        device, recomputed in backward, never saved."""

        @staticmethod
        def forward(
            ctx: Any,
            hidden_states: Any,
            top_k_index: Any,
            top_k_weights: Any,
            gate_up_q: Any,
            gate_up_scale: Any,
            down_q: Any,
            down_scale: Any,
            act_fn: Any,
            num_experts: int,
            reuse_weights: bool,
        ) -> Any:
            global _recomputed_weights
            ctx.save_for_backward(hidden_states, top_k_index, top_k_weights)
            ctx.weights = (gate_up_q, gate_up_scale, down_q, down_scale)  # CPU-resident int8
            ctx.act_fn = act_fn
            ctx.num_experts = num_experts
            dev, dtype = hidden_states.device, hidden_states.dtype
            with torch.no_grad():
                gate_up = _dequantized_on(gate_up_q, gate_up_scale, dev, dtype)
                down = _dequantized_on(down_q, down_scale, dev, dtype)
                out = _moe_layer(
                    hidden_states, top_k_index, top_k_weights, gate_up, down, act_fn, num_experts
                )
            if reuse_weights:
                # This call is the checkpoint recompute; the backward that follows
                # it is the consumer (see _recomputed_weights for why not ctx).
                _recomputed_weights = (id(gate_up_q), gate_up, down)
            else:
                del gate_up, down
            return out

        @staticmethod
        def backward(ctx: Any, grad_out: Any) -> Any:
            hidden_states, top_k_index, top_k_weights = ctx.saved_tensors
            gate_up_q, gate_up_scale, down_q, down_scale = ctx.weights
            reused = _take_recomputed_weights(gate_up_q)
            with torch.enable_grad():
                x = hidden_states.detach().requires_grad_(True)
                w = top_k_weights.detach().requires_grad_(True)
                dev, dtype = x.device, x.dtype
                if reused is not None:
                    gate_up, down = reused
                else:
                    gate_up = _dequantized_on(gate_up_q, gate_up_scale, dev, dtype)
                    down = _dequantized_on(down_q, down_scale, dev, dtype)
                y = _moe_layer(x, top_k_index, w, gate_up, down, ctx.act_fn, ctx.num_experts)
                grad_x, grad_w = torch.autograd.grad(y, (x, w), grad_out)
            del gate_up, down, reused
            return grad_x, None, grad_w, None, None, None, None, None, None, None

    return TransientLayerExpertsInt8


def install_int8_expert_forward(model: Any) -> int:
    """Patch every experts module with the int8 transient forward.

    Returns the number of patched modules. Refuses a module that is missing its
    int8 buffers, or that still carries trainable expert parameters: this path
    gives no weight gradient, so training them would be a silent no-op.
    """
    import torch

    function = _int8_layer_function()

    def forward(self: Any, hidden_states: Any, top_k_index: Any, top_k_weights: Any) -> Any:
        # Read here, where the checkpoint's two passes are distinguishable; the
        # Function's own forward cannot tell them apart (see _in_recompute_pass).
        reuse = retain_recomputed_weights() and _in_recompute_pass()
        return function.apply(
            hidden_states,
            top_k_index,
            top_k_weights,
            self.gate_up_q,
            self.gate_up_scale,
            self.down_q,
            self.down_scale,
            self.act_fn,
            self.num_experts,
            reuse,
        )

    patched = 0
    for name, module in _experts_modules(model):
        missing = [buffer for buffer in INT8_EXPERT_BUFFERS if getattr(module, buffer, None) is None]
        if missing:
            raise RuntimeError(
                f"{name} has no int8 expert buffers {missing}; load it with "
                "load_model_int8_experts before installing this forward"
            )
        trainable = [param for param in module.parameters() if param.requires_grad]
        if trainable:
            raise RuntimeError(
                f"{name} has trainable expert weights; the transient-grad path keeps "
                "experts frozen and gives them no gradient, so freeze them "
                "(requires_grad=False) rather than train them to no effect"
            )
        module.forward = forward.__get__(module)
        patched += 1
    return patched


def prepare_int8_placement(model: Any, *, device_type: str = "cuda") -> dict[str, Any]:
    """Census the int8 placement, then install the int8 transient forward.

    Raises when the placement did not happen: no experts module, a bf16 expert
    parameter left behind, an int8 buffer off the CPU, or a non-expert
    parameter off the device all mean the request was not honoured.
    """
    census = int8_expert_census(model)
    leftover = [name for name, _ in model.named_parameters() if EXPERT_PARAM_INFIX in name]
    others = [(name, parameter) for name, parameter in model.named_parameters() if EXPERT_PARAM_INFIX not in name]
    census["remaining_expert_params"] = len(leftover)
    census["non_expert_params"] = len(others)
    census["non_expert_params_on_device"] = sum(
        parameter.device.type == device_type for _, parameter in others
    )
    problems: list[str] = []
    if census["expert_modules"] == 0:
        problems.append("no experts module was found")
    if leftover:
        problems.append(f"{len(leftover)} bf16 expert parameter(s) remain, e.g. {leftover[0]}")
    if census["expert_buffer_devices"] not in ([], ["cpu"]):
        problems.append(f"int8 expert buffers are not CPU-resident: {census['expert_buffer_devices']}")
    if census["non_expert_params_on_device"] != len(others):
        problems.append(
            f"{len(others) - census['non_expert_params_on_device']} non-expert parameter(s) are not on "
            f"{device_type!r}"
        )
    if problems:
        raise RuntimeError(
            f"the {INT8_PLACEMENT} placement census failed on the live model: {'; '.join(problems)}; "
            "a placement that did not happen must not be believed because it was requested"
        )
    census["patched_expert_modules"] = install_int8_expert_forward(model)
    if census["patched_expert_modules"] == 0:
        raise RuntimeError("int8 expert buffers exist but no experts module was patched")
    return census
