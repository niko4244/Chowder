"""Hot-core upcycling: the dense->MoE init the measurements actually support.

This sits beside `dense_to_moe` rather than replacing it. That module implements
the plan's exactness-preserving partition and has real artifacts behind it; it is
left untouched. This module implements a different trade, chosen because it was
measured (evidence: `A4B-UPCYCLING-DESIGN.md` in the frontier repo, and the
numbers quoted below are held-out on the real dense 9B, ratios against an
identical nf4 load).

The three things measurement changed
------------------------------------
1. **Rank channels, never index them.** Held-out at 28% of channels kept, an
   index-ordered subset scores **614.7x** baseline perplexity; an
   activation-ranked one scores **1.69x**. `dense_to_moe` partitions by index.
2. **Do not pre-scale `down_proj` by E.** A partition needs xE to be exact at
   `top_k == E`, but `Qwen3_5MoeTopKRouter` renormalises unconditionally, so at
   `top_k = k < E` the surviving channels come out multiplied by E/k -- a blowup
   of a residual stream the norms never saw. That factor, not sparsity, is the
   second half of the 27B ladder's collapse to 243,981. Here cold `down_proj` is
   pre-scaled by `top_k` instead, so the *intended operating point* has unit
   scale and no other point is privileged.
3. **Put the hot core in `shared_expert`, which `dense_to_moe` fills with
   zeros.** Replicating a hot core inside every routed expert is never better
   than a plain static prune at equal active compute, because `top_k = k`
   recomputes the core k times: active cost `k*(h+c)`, coverage only `h + k*c`.
   In the shared expert the core is computed once, unconditionally, so active
   cost equals coverage.

What this init is and is not
---------------------------
It is **not exact**. At init the converted model is not bit-equal to the dense
parent: only `h + top_k * c` of the `intermediate_size` channels are present for
a given token. That is the deliberate trade. The exactness contract bought
`top_k == E`, which is dense compute and therefore zero saving, while this init
degrades gracefully -- measured **1.81x** baseline perplexity at h = 75% of a
3,440-channel active budget, against a per-token-oracle ceiling of 1.03x. A
trainable starting point, which 243,981 never was.

Two mechanical details that are easy to get wrong, both verified against the
installed `transformers` source (`models/qwen3_5_moe/modeling_qwen3_5_moe.py`):

* The shared expert is **gated**: the block computes
  `expert_output + sigmoid(shared_expert_gate(x)) * shared_expert(x)`. With the
  gate zero-init, `sigmoid(0)` is exactly 0.5, so a hot core placed there would
  arrive at HALF scale. Its `down_proj` is therefore pre-scaled by 2, which is an
  exact power-of-two bump, and leaves the gate at 0 where its derivative is
  largest -- the most learnable point, not a corner.
* Routed weights are renormalised to sum to 1 over the selected experts, so at
  init each selected expert carries exactly `1/top_k`. The exactness requirement
  for bf16 is therefore on **`top_k`** being a power of two, NOT on
  `num_experts`. Unlike `dense_to_moe`, E is free here, which matters because
  storage is not the binding constraint at these active budgets and finer
  granularity means more reachable channel combinations.

Like `dense_to_moe` this is stdlib-only byte surgery and never imports torch --
converting a checkpoint is not a model load. Arbitrary column gathers on a
`(hidden, intermediate)` `down_proj` are done with `array`'s strided slice
assignment (`out[j::c] = src[ch::intermediate]`), which runs in C, so no array
dependency is introduced for what is fundamentally a byte permutation.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .channel_importance import ChannelRanking
from .dense_to_moe import (
    _DTYPE_BYTES,
    _GATE_SUFFIX,
    _LAYERS_PREFIX,
    _MLP_SUFFIXES,
    _PASSTHROUGH_FILES,
    _UP_SUFFIX,
    _read_header,
    _scale_bytes_exact,
    _tensor_entries,
)
from .local_model_manifest import build_local_model_manifest, write_manifest_file
from .provenance import sha256_file

#: Opaque bit-moving codes: we permute elements, never interpret them.
_MOVE_CODE: Mapping[int, str] = {2: "H", 4: "I", 8: "Q"}

#: sigmoid(0) == 0.5 exactly, so the gated shared expert needs x2 to reach unit
#: scale at init. See module docstring.
_SHARED_GATE_INIT_COMPENSATION = 2


class HotCoreUpcycleError(ValueError):
    """A hot-core conversion cannot be performed honestly."""


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


@dataclass(frozen=True)
class HotCoreScheme:
    """The recorded channel destination map, derived from a measured ranking.

    `hot` are the channels that go to the always-on shared expert, hottest
    first. `cold_by_expert[e]` are the channels routed expert e carries.
    """

    num_experts: int
    top_k: int
    intermediate_size: int
    hot_core_size: int
    hot: tuple[int, ...]
    cold_by_expert: tuple[tuple[int, ...], ...]
    ranking_digest: str
    cold_assignment: str

    def __post_init__(self) -> None:
        if not _is_power_of_two(self.top_k):
            raise HotCoreUpcycleError(
                f"top_k must be a power of two so the uniform routed weight "
                f"1/top_k and the x{self.top_k} down_proj pre-scale are both exact "
                f"in bf16; got {self.top_k}"
            )
        if not 1 <= self.top_k <= self.num_experts:
            raise HotCoreUpcycleError(
                f"top_k {self.top_k} must be in 1..num_experts {self.num_experts}"
            )
        if not 0 < self.hot_core_size < self.intermediate_size:
            raise HotCoreUpcycleError(
                f"hot_core_size must be in 1..{self.intermediate_size - 1}; got "
                f"{self.hot_core_size}. A zero core is the all-cold design "
                "measured at 7364.69x baseline perplexity; a full core leaves no "
                "routed channels and so no router to train"
            )
        cold_total = self.intermediate_size - self.hot_core_size
        if cold_total % self.num_experts != 0:
            raise HotCoreUpcycleError(
                f"{cold_total} cold channels do not divide evenly across "
                f"{self.num_experts} experts; the expert bank would be ragged. "
                f"Adjust hot_core_size (e.g. to "
                f"{self.intermediate_size - (cold_total // self.num_experts) * self.num_experts + self.hot_core_size})"
            )
        if len(self.hot) != self.hot_core_size:
            raise HotCoreUpcycleError("hot must list exactly hot_core_size channels")
        if len(self.cold_by_expert) != self.num_experts:
            raise HotCoreUpcycleError("cold_by_expert must have one entry per expert")
        per = cold_total // self.num_experts
        if any(len(group) != per for group in self.cold_by_expert):
            raise HotCoreUpcycleError("every expert must carry the same cold width")
        seen = list(self.hot) + [c for group in self.cold_by_expert for c in group]
        if sorted(seen) != list(range(self.intermediate_size)):
            raise HotCoreUpcycleError(
                "hot + cold must partition every intermediate channel exactly once; "
                "a channel dropped here is a channel the model can never compute"
            )

    @property
    def moe_intermediate_size(self) -> int:
        return (self.intermediate_size - self.hot_core_size) // self.num_experts

    @property
    def active_channels(self) -> int:
        """Channels computed for one token: the core once, plus top_k slices."""
        return self.hot_core_size + self.top_k * self.moe_intermediate_size

    @classmethod
    def from_ranking(
        cls,
        ranking: ChannelRanking,
        layer: int,
        *,
        num_experts: int,
        top_k: int,
        hot_core_size: int,
    ) -> "HotCoreScheme":
        """Build one layer's scheme: top-h to the core, rest round-robin.

        Cold channels are dealt **round-robin by rank** (`cold[e::E]`), not in
        contiguous rank blocks. Contiguous blocks would make expert 0 the warm
        one and expert E-1 nearly dead, which gives a router every reason to
        collapse onto expert 0 and no reason to differentiate. Round-robin gives
        every expert the same importance profile, so differentiation has to be
        learned from the data rather than handed over at init.
        """
        order = ranking.ranking.get(layer)
        if order is None:
            raise HotCoreUpcycleError(
                f"ranking has no entry for layer {layer}; refusing to fall back to "
                "index order, which is the arm measured at 614.7x"
            )
        hot = tuple(order[:hot_core_size])
        cold = list(order[hot_core_size:])
        groups = tuple(tuple(cold[e::num_experts]) for e in range(num_experts))
        return cls(
            num_experts=num_experts,
            top_k=top_k,
            intermediate_size=ranking.intermediate_size,
            hot_core_size=hot_core_size,
            hot=hot,
            cold_by_expert=groups,
            ranking_digest=ranking.digest(),
            cold_assignment="round_robin_by_rank",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "intermediate_size": self.intermediate_size,
            "hot_core_size": self.hot_core_size,
            "moe_intermediate_size": self.moe_intermediate_size,
            "active_channels": self.active_channels,
            "cold_assignment": self.cold_assignment,
            "ranking_digest": self.ranking_digest,
            "hot": list(self.hot),
            "cold_by_expert": [list(g) for g in self.cold_by_expert],
        }

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True).encode("utf-8")
        ).hexdigest()


def _as_moveable(raw: bytes, width: int) -> array:
    """View raw bytes as fixed-width opaque elements for permutation."""
    code = _MOVE_CODE.get(width)
    if code is None:
        raise HotCoreUpcycleError(
            f"no {width}-byte element code for permutation; refusing to guess"
        )
    if sys.byteorder != "little":
        raise HotCoreUpcycleError(
            "safetensors is little-endian and this permutation moves native-order "
            f"elements; refusing to run on a {sys.byteorder}-endian host"
        )
    buf = array(code)
    buf.frombytes(raw)
    return buf


def _gather_rows(raw: bytes, channels: tuple[int, ...], *, hidden: int, width: int) -> bytes:
    """Rows of a (intermediate, hidden) tensor, in `channels` order.

    gate_proj/up_proj are nn.Linear (out, in) = (intermediate, hidden), so a
    channel is a contiguous ROW and this is a plain slice gather. Never scaled:
    silu is applied to the gate factor and is not positively homogeneous, so
    scaling it would change the function, not its scale.
    """
    stride = hidden * width
    out = bytearray()
    for channel in channels:
        start = channel * stride
        out += raw[start : start + stride]
    return bytes(out)


def _gather_columns(
    raw: bytes, channels: tuple[int, ...], *, hidden: int, intermediate: int, width: int
) -> bytes:
    """Columns of a (hidden, intermediate) tensor, as (hidden, len(channels)).

    down_proj is nn.Linear (out, in) = (hidden, intermediate), so a channel is a
    strided COLUMN. `src[ch::intermediate]` is that column and `out[j::n]` is its
    destination column; both are C-level strided copies, which is what makes an
    arbitrary gather affordable in stdlib at 32 layers x 4096 rows.
    """
    src = _as_moveable(raw, width)
    expected = hidden * intermediate
    if len(src) != expected:
        raise HotCoreUpcycleError(
            f"down_proj has {len(src)} elements, expected hidden*intermediate "
            f"= {expected}; the source layout is not what this converter verified"
        )
    count = len(channels)
    code = _MOVE_CODE[width]
    out = array(code, bytes(hidden * count * width))
    for position, channel in enumerate(channels):
        out[position::count] = src[channel::intermediate]
    return out.tobytes()


@dataclass(frozen=True)
class HotCorePlan:
    """Shape and budget arithmetic, before any bytes move."""

    hidden_size: int
    intermediate_size: int
    num_layers: int
    weight_dtype: str
    num_experts: int
    top_k: int
    hot_core_size: int
    moe_intermediate_size: int
    active_channels: int
    params_per_channel_all_layers: int
    stored_ffn_params: int
    active_ffn_params: int
    dense_ffn_params: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_layers": self.num_layers,
            "weight_dtype": self.weight_dtype,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "hot_core_size": self.hot_core_size,
            "moe_intermediate_size": self.moe_intermediate_size,
            "active_channels": self.active_channels,
            "active_channel_fraction": self.active_channels / self.intermediate_size,
            "params_per_channel_all_layers": self.params_per_channel_all_layers,
            "stored_ffn_params": self.stored_ffn_params,
            "active_ffn_params": self.active_ffn_params,
            "dense_ffn_params": self.dense_ffn_params,
            "stored_vs_dense_ffn": self.stored_ffn_params / self.dense_ffn_params,
            "active_vs_dense_ffn": self.active_ffn_params / self.dense_ffn_params,
        }


def plan_hot_core_conversion(
    model_dir: str | Path,
    ranking: ChannelRanking,
    *,
    num_experts: int,
    top_k: int,
    hot_core_size: int,
) -> HotCorePlan:
    """Read the source headers and price the conversion. Moves no bytes."""
    source = Path(model_dir)
    config_path = source / "config.json"
    if not config_path.is_file():
        raise HotCoreUpcycleError(f"no config.json in {source}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    text_config = config.get("text_config", config)
    hidden = int(text_config["hidden_size"])
    intermediate = int(text_config["intermediate_size"])

    if ranking.intermediate_size != intermediate:
        raise HotCoreUpcycleError(
            f"ranking was measured at intermediate_size {ranking.intermediate_size} "
            f"but this checkpoint has {intermediate}"
        )
    source_manifest = build_local_model_manifest(source, mode="fast")
    if ranking.source_manifest_sha256 != source_manifest.manifest_sha256:
        raise HotCoreUpcycleError(
            "the ranking was measured on a different checkpoint "
            f"({ranking.source_manifest_sha256[:12]}...) than the one being "
            f"converted ({source_manifest.manifest_sha256[:12]}...). Converting "
            "with a foreign ranking silently produces a wrong-but-plausible model"
        )

    dtypes: set[str] = set()
    layers: set[int] = set()
    for shard in sorted(source.glob("*.safetensors")):
        for name, entry in _tensor_entries(_read_header(shard)).items():
            if name.startswith(_LAYERS_PREFIX) and name.endswith(_MLP_SUFFIXES):
                layers.add(int(name[len(_LAYERS_PREFIX):].split(".")[0]))
                dtypes.add(entry["dtype"])
    if not layers:
        raise HotCoreUpcycleError(
            f"no dense MLP tensors found under {_LAYERS_PREFIX} in {source}"
        )
    if len(dtypes) != 1:
        raise HotCoreUpcycleError(
            f"mixed MLP dtypes {sorted(dtypes)}; refusing to convert a "
            "heterogeneous FFN"
        )
    weight_dtype = dtypes.pop()
    missing = sorted(layers - set(ranking.ranking))
    if missing:
        raise HotCoreUpcycleError(
            f"ranking is missing layers {missing[:8]}{'...' if len(missing) > 8 else ''}; "
            "refusing to index-order any layer"
        )

    moe_int = (intermediate - hot_core_size) // num_experts
    num_layers = len(layers)
    per_channel = 3 * hidden * num_layers
    stored = (hot_core_size + num_experts * moe_int) * per_channel
    active = (hot_core_size + top_k * moe_int) * per_channel
    return HotCorePlan(
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_layers=num_layers,
        weight_dtype=weight_dtype,
        num_experts=num_experts,
        top_k=top_k,
        hot_core_size=hot_core_size,
        moe_intermediate_size=moe_int,
        active_channels=hot_core_size + top_k * moe_int,
        params_per_channel_all_layers=per_channel,
        stored_ffn_params=stored,
        active_ffn_params=active,
        dense_ffn_params=intermediate * per_channel,
    )


def convert_checkpoint_hot_core(
    model_dir: str | Path,
    output_dir: str | Path,
    ranking: ChannelRanking,
    *,
    num_experts: int,
    top_k: int,
    hot_core_size: int,
) -> dict[str, Any]:
    """Convert a dense qwen3_5 checkpoint with a measured hot core.

    Two-pass streaming, mirroring `dense_to_moe.convert_checkpoint`: headers are
    scanned once to build the output layout and a regenerated index, then each
    output shard is written atomically (temp file + `os.replace`). Everything
    outside the MLPs -- attention, `linear_attn`, embeddings, vision, norms -- is
    copied byte-identical.

    Returns the provenance dict, which records that this init is NOT exact and
    what it is instead.
    """
    source = Path(model_dir)
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise HotCoreUpcycleError(f"output dir exists and is not empty: {out}")

    plan = plan_hot_core_conversion(
        source, ranking, num_experts=num_experts, top_k=top_k, hot_core_size=hot_core_size
    )
    hidden = plan.hidden_size
    intermediate = plan.intermediate_size
    moe_int = plan.moe_intermediate_size
    width = _DTYPE_BYTES[plan.weight_dtype]
    source_manifest = build_local_model_manifest(source, mode="fast")

    shard_data_starts: dict[str, int] = {}
    for shard_file in sorted(source.glob("*.safetensors")):
        with shard_file.open("rb") as probe:
            shard_data_starts[shard_file.name] = 8 + struct.unpack("<Q", probe.read(8))[0]
    src_handles: dict[str, Any] = {}

    def read_part(shard: str, entry: Mapping[str, Any]) -> bytes:
        handle = src_handles.get(shard)
        if handle is None:
            handle = (source / shard).open("rb")
            src_handles[shard] = handle
        begin, end = entry["data_offsets"]
        handle.seek(shard_data_starts[shard] + begin)
        return handle.read(end - begin)

    # ---- pass 1: headers -> output layout -----------------------------------
    shard_headers: dict[str, dict[str, dict[str, Any]]] = {
        shard.name: {} for shard in source.glob("*.safetensors")
    }
    layer_sources: dict[str, dict[str, Any]] = {}
    for shard in sorted(source.glob("*.safetensors")):
        for name, entry in _tensor_entries(_read_header(shard)).items():
            if name.startswith(_LAYERS_PREFIX) and name.endswith(_MLP_SUFFIXES):
                layer = name[len(_LAYERS_PREFIX):].split(".")[0]
                layer_sources.setdefault(layer, {"shard": None, "parts": {}})
                layer_sources[layer]["parts"][name] = {"shard": shard.name, "entry": entry}
                if name.endswith(_GATE_SUFFIX):
                    layer_sources[layer]["shard"] = shard.name
                continue
            shard_headers[shard.name][name] = {"dtype": entry["dtype"], "shape": entry["shape"]}

    schemes: dict[str, HotCoreScheme] = {}
    # Exact output-name -> (layer, kind). Dispatch in pass 2 is a lookup in this
    # map, never a substring test: `".mlp." in name` also matches the vision
    # tower's `model.visual.blocks.N.mlp.linear_fc1`, and this converter promises
    # vision is copied byte-identical.
    moe_targets: dict[str, tuple[str, str]] = {}
    for layer, info in sorted(layer_sources.items()):
        schemes[layer] = HotCoreScheme.from_ranking(
            ranking, int(layer),
            num_experts=num_experts, top_k=top_k, hot_core_size=hot_core_size,
        )
        target = shard_headers[info["shard"]]
        prefix = f"{_LAYERS_PREFIX}{layer}.mlp."
        for suffix, kind, shape in (
            ("experts.gate_up_proj", "cold_gate_up", [num_experts, 2 * moe_int, hidden]),
            ("experts.down_proj", "cold_down", [num_experts, hidden, moe_int]),
            ("gate.weight", "router", [num_experts, hidden]),
            ("shared_expert.gate_proj.weight", "hot_gate", [hot_core_size, hidden]),
            ("shared_expert.up_proj.weight", "hot_up", [hot_core_size, hidden]),
            ("shared_expert.down_proj.weight", "hot_down", [hidden, hot_core_size]),
            ("shared_expert_gate.weight", "shared_gate", [1, hidden]),
        ):
            target[prefix + suffix] = {"dtype": plan.weight_dtype, "shape": shape}
            moe_targets[prefix + suffix] = (layer, kind)

    index_weight_map: dict[str, str] = {}
    for shard_name, entries in shard_headers.items():
        offset = 0
        for name in sorted(entries):
            count = 1
            for dim in entries[name]["shape"]:
                count *= dim
            nbytes = count * _DTYPE_BYTES[entries[name]["dtype"]]
            entries[name]["data_offsets"] = [offset, offset + nbytes]
            offset += nbytes
            index_weight_map[name] = shard_name

    # ---- pass 2: stream tensors, write shards atomically --------------------
    def mlp_part(layer: str, suffix: str) -> bytes:
        part = layer_sources[layer]["parts"][f"{_LAYERS_PREFIX}{layer}.mlp.{suffix}"]
        return read_part(part["shard"], part["entry"])

    file_hashes: dict[str, str] = {}
    for shard in sorted(source.glob("*.safetensors")):
        shard_name = shard.name
        source_header = _read_header(shard)
        new_header = shard_headers[shard_name]
        header_blob = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
        header_blob += b" " * ((8 - len(header_blob) % 8) % 8)

        out_path = out / shard_name
        tmp_path = out_path.with_suffix(".tmp")
        out.mkdir(parents=True, exist_ok=True)
        with tmp_path.open("wb") as dst:
            dst.write(struct.pack("<Q", len(header_blob)))
            dst.write(header_blob)
            for name in sorted(new_header):
                kinded = moe_targets.get(name)
                if kinded is None:
                    # Everything this converter did not create: attention,
                    # linear_attn, embeddings, norms, MTP, and the vision tower
                    # (whose blocks also carry `.mlp.` tensors) -- byte-identical.
                    dst.write(read_part(shard_name, source_header[name]))
                    continue
                layer, kind = kinded
                scheme = schemes[layer]

                if kind == "cold_gate_up":
                    gate_raw = mlp_part(layer, "gate_proj.weight")
                    up_raw = mlp_part(layer, "up_proj.weight")
                    for group in scheme.cold_by_expert:
                        dst.write(_gather_rows(gate_raw, group, hidden=hidden, width=width))
                        dst.write(_gather_rows(up_raw, group, hidden=hidden, width=width))
                elif kind == "cold_down":
                    down_raw = mlp_part(layer, "down_proj.weight")
                    for group in scheme.cold_by_expert:
                        # xtop_k, not xE: each selected expert carries 1/top_k.
                        dst.write(_scale_bytes_exact(
                            _gather_columns(down_raw, group, hidden=hidden,
                                            intermediate=intermediate, width=width),
                            plan.weight_dtype, top_k,
                        ))
                elif kind == "hot_gate":
                    dst.write(_gather_rows(mlp_part(layer, "gate_proj.weight"),
                                           scheme.hot, hidden=hidden, width=width))
                elif kind == "hot_up":
                    dst.write(_gather_rows(mlp_part(layer, "up_proj.weight"),
                                           scheme.hot, hidden=hidden, width=width))
                elif kind == "hot_down":
                    # x2 undoes sigmoid(0)=0.5 on the gated shared branch.
                    dst.write(_scale_bytes_exact(
                        _gather_columns(mlp_part(layer, "down_proj.weight"), scheme.hot,
                                        hidden=hidden, intermediate=intermediate, width=width),
                        plan.weight_dtype, _SHARED_GATE_INIT_COMPENSATION,
                    ))
                elif kind == "router":
                    dst.write(bytes(num_experts * hidden * width))
                elif kind == "shared_gate":
                    dst.write(bytes(hidden * width))
                else:  # pragma: no cover - moe_targets and this dispatch are one unit
                    raise HotCoreUpcycleError(
                        f"unhandled MoE tensor kind {kind!r} for {name}; refusing "
                        "to emit a shard with an unwritten tensor"
                    )
        os.replace(tmp_path, out_path)
        file_hashes[shard_name] = sha256_file(out_path)

    for handle in src_handles.values():
        handle.close()

    # ---- companion files ----------------------------------------------------
    for file_name in _PASSTHROUGH_FILES:
        candidate = source / file_name
        if candidate.is_file():
            (out / file_name).write_bytes(candidate.read_bytes())
    actual_bytes = sum((out / name).stat().st_size for name in file_hashes)
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": actual_bytes},
                    "weight_map": index_weight_map}, indent=2),
        encoding="utf-8", newline="\n",
    )
    _write_hot_core_config(source, out, plan)

    scheme_digests = {layer: s.digest() for layer, s in sorted(schemes.items())}
    provenance = {
        "converter": "hot_core_upcycle",
        "source_dir": str(source),
        "source_manifest_sha256": source_manifest.manifest_sha256,
        "ranking": {
            "method": ranking.method,
            "digest": ranking.digest(),
            "calibration": dict(ranking.calibration),
            "concentration": dict(ranking.concentration),
        },
        "plan": plan.to_dict(),
        "scheme_digests": scheme_digests,
        "exactness_contract": (
            "NONE -- this init is deliberately not exact. Only hot_core_size + "
            "top_k * moe_intermediate_size of intermediate_size channels are "
            "present for a given token, so the converted model does not reproduce "
            "the dense function bit-for-bit. The exactness contract it replaces "
            "required top_k == num_experts, which is dense compute and therefore "
            "zero saving; this trades that for graceful degradation (measured "
            "1.81x baseline perplexity at a 75% core over a 3,440-channel active "
            "budget, against a per-token-oracle ceiling of 1.03x). Scaling: cold "
            "down_proj x top_k because renormalised routed weights are 1/top_k "
            "each; shared down_proj x2 because the shared branch is gated by "
            "sigmoid(shared_expert_gate(x)) and a zero-init gate gives exactly "
            "0.5. gate/up are copied verbatim -- silu is not positively "
            "homogeneous, so scaling the gate factor would change the function."
        ),
        "router_init": (
            "zeros. Logits are therefore tied and which top_k experts win is an "
            "implementation detail of torch.topk, not a guarantee -- so at step 0 "
            "experts outside the tie-break winners may receive no token. This is "
            "reproducible but cold-starts the bank; a training arm should break "
            "the tie deliberately rather than rely on the tie order."
        ),
        "output_file_sha256": file_hashes,
        "storage": {
            "source_bytes": sum(
                f.stat().st_size for f in source.glob("*.safetensors")
            ),
            "actual_output_bytes": actual_bytes,
        },
    }
    manifest = build_local_model_manifest(out, mode="full")
    write_manifest_file(manifest, out / "conversion.manifest.json")
    provenance["output_manifest_sha256"] = manifest.manifest_sha256
    (out / "conversion.provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8", newline="\n",
    )
    return provenance


def _write_hot_core_config(source: Path, out: Path, plan: HotCorePlan) -> None:
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    text_config = config.get("text_config", config)
    kept = {k: v for k, v in config.items() if k not in {"architectures", "model_type"}}
    kept["architectures"] = ["Qwen3_5MoeForConditionalGeneration"]
    kept["model_type"] = "qwen3_5_moe"
    new_text = dict(text_config)
    new_text["model_type"] = "qwen3_5_moe_text"
    new_text["num_experts"] = plan.num_experts
    # top_k, NOT num_experts: the whole point is that this init does not require
    # every expert to run, and the down_proj pre-scale is matched to this value.
    new_text["num_experts_per_tok"] = plan.top_k
    new_text["moe_intermediate_size"] = plan.moe_intermediate_size
    new_text["shared_expert_intermediate_size"] = plan.hot_core_size
    kept["text_config"] = new_text
    (out / "config.json").write_text(
        json.dumps(kept, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n"
    )
