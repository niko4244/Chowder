"""Weight-preserving dense-to-MoE conversion (program Phase 6).

Implements docs/PHASE6_CONVERSION_PLAN.md's partition-conversion: the
dense FFN's intermediate dimension is *partitioned* into E channel
groups (experts); no expert weights are invented, no expert is copied.

Exactness contract (and the plan erratum it corrects)
-----------------------------------------------------
The dense SwiGLU output is the sum over *all* intermediate channels, so
a converted model computes the dense function exactly only when every
expert runs for every token — **top-k = E**. The plan's §2 suggestion
that top-1 rungs could be init-exact was wrong (dropping a group drops
its contribution; top-1 exactness would need E copies, the E×-storage
upcycling the plan rejects). Recorded as an erratum in the plan.

Mechanics with the real `Qwen3_5MoeTopKRouter`: zero router logits are
bitwise-uniform in every float dtype, so softmax gives exactly 1/E and
the top-k renormalization (`w /= w.sum()`) keeps 1/E exactly (1/8, 1/16,
1/32 are exact powers of two in bf16 — E must be a power of two). Each
expert's SwiGLU computes its channel slice; because silu is not
positively homogeneous, only the *linear* leg (down_proj) is pre-scaled
by E — gate/up slices are copied verbatim — which makes the 1/E-weighted
sum of E slices equal the dense sum exactly.
For BF16/F16 the ×E is an exponent-field bump on raw bytes — exact for
all finites (subnormals promote exactly as a real multiply), with
overflow producing signed infinity and inf/nan passed through verbatim,
both matching true multiplication. F32/F64 scale by exact value
multiply (power-of-two factor). Any other dtype is refused, never
approximated. Source layout verified from parent A's real headers:
`gate_proj (intermediate, hidden)`, `up_proj (intermediate, hidden)`,
`down_proj (hidden, intermediate)` (nn.Linear (out, in)), all BF16.

Preserved means provably preserved: MTP/vision/attention/embeddings are
copied byte-identical; a layer's three MLP tensors may live in
different shards (measured on the real parent A: 63/64 layers
co-locate gate/up/down in one shard, layer 15 does not) and each part
is streamed from whichever shard holds it; the output gets a regenerated
`model.safetensors.index.json`, a converted config, per-file hashes, a
provenance block, and a full-mode manifest. Writes are atomic per shard
(temp file + `os.replace`).

This module is stdlib-only and never imports torch; conversion of a
52 GiB checkpoint is a byte-surgery pass, not a model load. Loading and
forward-proofing happen in `conversion_exactness` (torch-gated).
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .local_model_manifest import build_local_model_manifest, write_manifest_file
from .provenance import sha256_file


class DenseToMoeError(ValueError):
    """A conversion cannot be performed honestly."""


_DTYPE_BYTES: Mapping[str, int] = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "U64": 8,
    "I32": 4, "U32": 4, "I16": 2, "U16": 2, "I8": 1, "U8": 1,
    "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1,
}

_EXACT_SCALABLE: frozenset[str] = frozenset({"BF16", "F16", "F32", "F64"})

_SHARED_EXPERT_INTERMEDIATE = 512

_LAYERS_PREFIX = "model.language_model.layers."
_GATE_SUFFIX = "mlp.gate_proj.weight"
_UP_SUFFIX = "mlp.up_proj.weight"
_DOWN_SUFFIX = "mlp.down_proj.weight"
_MLP_SUFFIXES = (_GATE_SUFFIX, _UP_SUFFIX, _DOWN_SUFFIX)

#: Files copied byte-identical into the converted directory.
_PASSTHROUGH_FILES: tuple[str, ...] = (
    "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
    "special_tokens_map.json", "added_tokens.json", "chat_template.jinja",
    "preprocessor_config.json", "video_preprocessor_config.json",
    "generation_config.json", "LICENSE", "README.md",
)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


@dataclass(frozen=True)
class PartitionScheme:
    """The recorded channel→group map. Contiguous by default; invertible
    by plain concatenation, which the exactness harness re-checks."""

    num_experts: int
    intermediate_size: int
    assignment: tuple[int, ...]
    contiguous: bool
    seed: int | None

    def __post_init__(self) -> None:
        if not _is_power_of_two(self.num_experts):
            raise DenseToMoeError(
                f"num_experts must be a power of two for exact rescaling "
                f"(the plan's ladder is 8/16/32); got {self.num_experts}"
            )
        if self.intermediate_size % self.num_experts != 0:
            raise DenseToMoeError(
                f"intermediate_size {self.intermediate_size} is not divisible by "
                f"num_experts {self.num_experts}; the partition would be ragged"
            )
        if len(self.assignment) != self.intermediate_size:
            raise DenseToMoeError("assignment must cover every intermediate channel")
        per_group = self.intermediate_size // self.num_experts
        if any(
            self.assignment[channel] != channel // per_group
            for channel in range(self.intermediate_size)
        ):
            raise DenseToMoeError("a contiguous scheme's assignment must be contiguous")

    @classmethod
    def contiguous_scheme(cls, num_experts: int, intermediate_size: int) -> "PartitionScheme":
        per_group = intermediate_size // num_experts
        return cls(
            num_experts=num_experts,
            intermediate_size=intermediate_size,
            assignment=tuple(channel // per_group for channel in range(intermediate_size)),
            contiguous=True,
            seed=None,
        )

    def to_dict(self) -> dict[str, Any]:
        assignment_digest = hashlib.sha256(
            json.dumps(list(self.assignment), separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return {
            "num_experts": self.num_experts,
            "intermediate_size": self.intermediate_size,
            "contiguous": self.contiguous,
            "seed": self.seed,
            "assignment_sha256": assignment_digest,
        }

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


def _scale_bytes_exact(raw: bytes, dtype: str, factor: int) -> bytes:
    """Multiply float elements by a power-of-two `factor`, exactly.

    BF16/F16: per-element exponent bump with true-multiply edge
    semantics (inf/nan verbatim; overflow → signed infinity; subnormal
    promotion exact). F32/F64: exact value multiply. Integer dtypes and
    fp8 refuse — approximation is never a conversion step.
    """
    if dtype not in _EXACT_SCALABLE:
        raise DenseToMoeError(
            f"dtype {dtype} has no exact x{factor} semantics in this converter; "
            "refusing to approximate (fail closed)"
        )
    if dtype in {"BF16", "F16"}:
        mbits = 7 if dtype == "BF16" else 10
        ebits = 8 if dtype == "BF16" else 5
        emask = (1 << ebits) - 1
        mmask = (1 << mbits) - 1
        bump = factor.bit_length() - 1           # log2(factor)
        out = bytearray(raw)
        for offset in range(0, len(raw) - len(raw) % 2, 2):
            word = out[offset] | (out[offset + 1] << 8)
            sign = word & 0x8000
            exponent = (word >> mbits) & emask
            mantissa = word & mmask
            if exponent == emask:                # inf/nan: x2^k is inf/nan
                continue
            if exponent > 0:                     # normal: add k to the exponent, exact
                new_exponent = exponent + bump
                if new_exponent >= emask:        # overflow -> signed infinity
                    word = sign | (emask << mbits)
                else:
                    word = sign | (new_exponent << mbits) | mantissa
            else:                                # subnormal or zero
                scaled = mantissa << bump
                if scaled == 0:                  # +/-0 x 2^k is bit-identical
                    continue
                if scaled <= mmask:              # still subnormal
                    word = sign | scaled
                else:                            # promoted to normal, exactly
                    leading = scaled.bit_length() - 1
                    new_exponent = leading + 1 - mbits
                    if new_exponent >= emask:    # unreachable for bf16/f16; fail closed
                        raise DenseToMoeError(
                            f"x{factor} overflows a {dtype} subnormal; refusing to approximate"
                        )
                    # scaled = mantissa << bump, so scaled - 2**leading carries
                    # >= (leading - mbits) trailing zeros: the shift is lossless.
                    word = (
                        sign
                        | (new_exponent << mbits)
                        | ((scaled - (1 << leading)) >> (leading - mbits))
                    )
            out[offset] = word & 0xFF
            out[offset + 1] = (word >> 8) & 0xFF
        return bytes(out)
    import array

    code = "f" if dtype == "F32" else "d"
    width = _DTYPE_BYTES[dtype]
    values = array.array(code)
    values.frombytes(raw[: len(raw) - len(raw) % width])
    pack = struct.Struct("<" + code).pack
    return b"".join(pack(value * factor) for value in values) + raw[len(raw) - len(raw) % width :]


def _fuse_gate_up(
    gate_raw: bytes,
    up_raw: bytes,
    *,
    hidden: int,
    channels_per_expert: int,
    num_experts: int,
    width: int,
) -> bytes:
    """Interleave verbatim gate/up channel blocks per expert.

    gate/up are NEVER scaled: silu is applied to the gate factor and is
    not positively homogeneous, so scaling it would change the function.
    Output layout: for each expert e, gate rows [e*cpe, (e+1)*cpe) then
    up rows [e*cpe, (e+1)*cpe) — the fused (E, 2*moe_int, hidden) shape.
    """
    per = channels_per_expert * hidden * width
    out = bytearray()
    for expert in range(num_experts):
        lo = expert * per
        out += gate_raw[lo : lo + per]
        out += up_raw[lo : lo + per]
    return bytes(out)


def _fuse_down(
    down_scaled_raw: bytes,
    *,
    hidden: int,
    channels_per_expert: int,
    num_experts: int,
    width: int,
) -> bytes:
    """Strided column gather of the xE-scaled down tensor.

    down_proj is (hidden, intermediate) — nn.Linear (out, in) — so an
    intermediate channel is a COLUMN. Fused (e, r, j) = down_scaled(r,
    e*cpe + j), emitted as (E, hidden, moe_int).
    """
    in_row = channels_per_expert * num_experts * width
    out_row = channels_per_expert * width
    out = bytearray()
    for expert in range(num_experts):
        col = expert * channels_per_expert * width
        for row in range(hidden):
            start = row * in_row + col
            out += down_scaled_raw[start : start + out_row]
    return bytes(out)


def _read_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise DenseToMoeError(f"{path.name}: truncated header-length prefix")
        (header_len,) = struct.unpack("<Q", prefix)
        raw = handle.read(header_len)
        if len(raw) != header_len:
            raise DenseToMoeError(f"{path.name}: truncated header")
    return json.loads(raw)


def _tensor_entries(header: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {name: entry for name, entry in header.items() if name != "__metadata__"}


@dataclass(frozen=True)
class ConversionPlan:
    """What a conversion would do, measured before any byte is written."""

    source_dir: str
    num_experts: int
    intermediate_size: int
    hidden_size: int
    scheme_digest: str
    num_layers_converted: int
    routed_source_tensors: int
    passthrough_tensors: int
    source_bytes: int
    estimated_output_bytes: int
    storage_delta_bytes: int
    weight_dtype: str

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def plan_conversion(model_dir: str | Path, num_experts: int) -> ConversionPlan:
    """Measure a dense checkpoint; refuse anything conversion could not handle.

    Reads headers and config only. The refusals here are the whole truth
    about what a real run would hit: wrong model_type, non-divisible
    intermediate size, non-power-of-two E, incomplete MLP triples,
    inconsistent dtypes or shapes, or a dtype without exact scaling
    semantics.
    """
    root = Path(model_dir)
    if not root.is_dir():
        raise DenseToMoeError(f"model dir is not an existing directory: {root}")
    if not _is_power_of_two(num_experts):
        raise DenseToMoeError(f"num_experts must be a power of two; got {num_experts}")

    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3_5":
        raise DenseToMoeError(
            f"expected dense qwen3_5 source, found model_type {config.get('model_type')!r}"
        )
    text_config = config.get("text_config", config)
    intermediate = text_config.get("intermediate_size")
    hidden = text_config.get("hidden_size")
    layers = text_config.get("num_hidden_layers")
    if not all(isinstance(v, int) and v > 0 for v in (intermediate, hidden, layers)):
        raise DenseToMoeError("config lacks usable intermediate_size/hidden_size/num_hidden_layers")
    if intermediate % num_experts != 0:
        raise DenseToMoeError(
            f"intermediate_size {intermediate} is not divisible by num_experts {num_experts}"
        )

    shards = sorted(root.glob("*.safetensors"))
    if not shards:
        raise DenseToMoeError(f"{root}: no safetensors shards")

    # one pass over headers: locate every layer's three MLP tensors
    mlp_locations: dict[str, dict[str, tuple[str, str]]] = {}
    passthrough_count = 0
    weight_dtype: str | None = None
    source_bytes = 0
    for shard in shards:
        source_bytes += shard.stat().st_size
        for name, entry in _tensor_entries(_read_header(shard)).items():
            if name.startswith(_LAYERS_PREFIX) and name.endswith(_MLP_SUFFIXES):
                layer = name[len(_LAYERS_PREFIX):].split(".")[0]
                which = name.split(".")[-2]
                mlp_locations.setdefault(layer, {})[which] = (shard.name, entry["dtype"])
                if weight_dtype is None:
                    weight_dtype = entry["dtype"]
                elif entry["dtype"] != weight_dtype:
                    raise DenseToMoeError(
                        f"MLP weight dtypes differ across tensors ({weight_dtype} vs {entry['dtype']}); "
                        "the fused layout requires one dtype per layer set"
                    )
            else:
                passthrough_count += 1

    missing = sorted(layer for layer, parts in mlp_locations.items() if len(parts) != 3)
    if missing:
        raise DenseToMoeError(f"layers without a full gate/up/down triple: {missing[:5]}")
    # A layer's three MLP tensors may live in different shards (measured on
    # the real parent A: 63/64 layers co-locate, layer 15 does not). Each
    # part is read from its own shard; the fused tensors are emitted into
    # the shard hosting the layer's gate tensor.
    if len(mlp_locations) != layers:
        raise DenseToMoeError(
            f"config says {layers} layers but MLP triples were found for {len(mlp_locations)}"
        )
    if weight_dtype not in _EXACT_SCALABLE:
        raise DenseToMoeError(
            f"MLP weight dtype {weight_dtype} cannot be scaled exactly; conversion refused"
        )

    scheme = PartitionScheme.contiguous_scheme(num_experts, intermediate)
    # routed: 3T in, 3T out (disjoint slices) — delta is router + shared expert only
    delta_params_per_layer = (num_experts + 3 * _SHARED_EXPERT_INTERMEDIATE + 1) * hidden
    width = _DTYPE_BYTES[weight_dtype]
    delta_bytes = delta_params_per_layer * width * layers
    return ConversionPlan(
        source_dir=str(root),
        num_experts=num_experts,
        intermediate_size=intermediate,
        hidden_size=hidden,
        scheme_digest=scheme.digest(),
        num_layers_converted=len(mlp_locations),
        routed_source_tensors=3 * len(mlp_locations),
        passthrough_tensors=passthrough_count,
        source_bytes=source_bytes,
        storage_delta_bytes=delta_bytes,
        estimated_output_bytes=source_bytes + delta_bytes,
        weight_dtype=weight_dtype,
    )


def convert_checkpoint(
    model_dir: str | Path,
    output_dir: str | Path,
    num_experts: int,
) -> dict[str, Any]:
    """Convert a dense qwen3_5 checkpoint into the qwen3_5_moe layout.

    Two-pass streaming: headers are scanned once to build the output
    layout (and the regenerated index), then each output shard is
    written atomically by streaming each source tensor once — routed
    MLP weights scaled by E and fused, router/shared-expert zeros
    emitted, everything else copied byte-identical. Returns the
    provenance dict.
    """
    source = Path(model_dir)
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise DenseToMoeError(f"output dir exists and is not empty: {out}")
    plan = plan_conversion(source, num_experts)
    scheme = PartitionScheme.contiguous_scheme(num_experts, plan.intermediate_size)
    moe_int = plan.intermediate_size // num_experts
    channels_per_expert = moe_int
    hidden = plan.hidden_size
    width = _DTYPE_BYTES[plan.weight_dtype]
    factor = num_experts

    source_manifest = build_local_model_manifest(source, mode="fast")
    out.mkdir(parents=True, exist_ok=True)

    # Data-region start of every source shard + lazily opened read handles,
    # so a layer's parts can be streamed from whichever shard holds them.
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

    # ---- pass 1: headers -> output layout ------------------------------------
    shard_headers: dict[str, dict[str, dict[str, Any]]] = {shard.name: {} for shard in source.glob("*.safetensors")}
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

    for layer, info in sorted(layer_sources.items()):
        shard_name = info["shard"]
        parts = info["parts"]
        target = shard_headers[shard_name]
        target[f"{_LAYERS_PREFIX}{layer}.mlp.experts.gate_up_proj"] = {
            "dtype": plan.weight_dtype, "shape": [num_experts, 2 * moe_int, hidden],
        }
        target[f"{_LAYERS_PREFIX}{layer}.mlp.experts.down_proj"] = {
            "dtype": plan.weight_dtype, "shape": [num_experts, hidden, moe_int],
        }
        target[f"{_LAYERS_PREFIX}{layer}.mlp.gate.weight"] = {
            "dtype": plan.weight_dtype, "shape": [num_experts, hidden],
        }
        target[f"{_LAYERS_PREFIX}{layer}.mlp.shared_expert.gate_proj.weight"] = {
            "dtype": plan.weight_dtype, "shape": [_SHARED_EXPERT_INTERMEDIATE, hidden],
        }
        target[f"{_LAYERS_PREFIX}{layer}.mlp.shared_expert.up_proj.weight"] = {
            "dtype": plan.weight_dtype, "shape": [_SHARED_EXPERT_INTERMEDIATE, hidden],
        }
        target[f"{_LAYERS_PREFIX}{layer}.mlp.shared_expert.down_proj.weight"] = {
            "dtype": plan.weight_dtype, "shape": [hidden, _SHARED_EXPERT_INTERMEDIATE],
        }
        target[f"{_LAYERS_PREFIX}{layer}.mlp.shared_expert_gate.weight"] = {
            "dtype": plan.weight_dtype, "shape": [1, hidden],
        }

    # assign data offsets per output shard (pure arithmetic from shapes)
    index_weight_map: dict[str, str] = {}
    for shard_name, entries in shard_headers.items():
        offset = 0
        for name in sorted(entries):
            shape = entries[name]["shape"]
            count = 1
            for dim in shape:
                count *= dim
            nbytes = count * _DTYPE_BYTES[entries[name]["dtype"]]
            entries[name]["data_offsets"] = [offset, offset + nbytes]
            offset += nbytes
            index_weight_map[name] = shard_name

    # ---- pass 2: stream tensors, write output shards atomically ---------------
    file_hashes: dict[str, str] = {}
    for shard in sorted(source.glob("*.safetensors")):
        shard_name = shard.name
        source_header = _read_header(shard)

        out_path = out / shard_name
        tmp_path = out_path.with_suffix(".tmp")
        new_header = shard_headers[shard_name]
        header_blob = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
        header_blob += b" " * ((8 - len(header_blob) % 8) % 8)
        with tmp_path.open("wb") as dst:
            dst.write(struct.pack("<Q", len(header_blob)))
            dst.write(header_blob)
            for name in sorted(new_header):
                entry = new_header[name]
                fused_name = name
                if fused_name.endswith("mlp.experts.gate_up_proj"):
                    layer = fused_name[len(_LAYERS_PREFIX):].split(".")[0]
                    parts = layer_sources[layer]["parts"]
                    # gate/up fused VERBATIM: silu is applied to the
                    # unscaled gate factor and is not positively
                    # homogeneous, so only the linear leg (down) carries xE.
                    dst.write(_fuse_gate_up(
                        read_part(
                            parts[f"{_LAYERS_PREFIX}{layer}.mlp.gate_proj.weight"]["shard"],
                            parts[f"{_LAYERS_PREFIX}{layer}.mlp.gate_proj.weight"]["entry"],
                        ),
                        read_part(
                            parts[f"{_LAYERS_PREFIX}{layer}.mlp.up_proj.weight"]["shard"],
                            parts[f"{_LAYERS_PREFIX}{layer}.mlp.up_proj.weight"]["entry"],
                        ),
                        hidden=hidden,
                        channels_per_expert=channels_per_expert,
                        num_experts=num_experts,
                        width=width,
                    ))
                elif fused_name.endswith("mlp.experts.down_proj"):
                    layer = fused_name[len(_LAYERS_PREFIX):].split(".")[0]
                    parts = layer_sources[layer]["parts"]
                    dst.write(_fuse_down(
                        _scale_bytes_exact(
                            read_part(
                                parts[f"{_LAYERS_PREFIX}{layer}.mlp.down_proj.weight"]["shard"],
                                parts[f"{_LAYERS_PREFIX}{layer}.mlp.down_proj.weight"]["entry"],
                            ),
                            plan.weight_dtype,
                            factor,
                        ),
                        hidden=hidden,
                        channels_per_expert=channels_per_expert,
                        num_experts=num_experts,
                        width=width,
                    ))
                elif fused_name.endswith("mlp.gate.weight"):
                    dst.write(bytes(num_experts * hidden * width))
                elif fused_name.endswith("mlp.shared_expert_gate.weight"):
                    dst.write(bytes(hidden * width))
                elif ".mlp.shared_expert." in fused_name:
                    shape = entry["shape"]
                    count = 1
                    for dim in shape:
                        count *= dim
                    dst.write(bytes(count * width))
                else:
                    dst.write(read_part(shard_name, source_header[name]))
        os.replace(tmp_path, out_path)
        file_hashes[shard_name] = sha256_file(out_path)

    for handle in src_handles.values():
        handle.close()

    # ---- companion files -------------------------------------------------------
    for file_name in _PASSTHROUGH_FILES:
        candidate = source / file_name
        if candidate.is_file():
            (out / file_name).write_bytes(candidate.read_bytes())
    # The converted model always carries a complete index, built from the
    # headers pass 1 recorded -- output completeness never depends on the
    # source shipping its own index.
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": plan.estimated_output_bytes}, "weight_map": index_weight_map}, indent=2),
        encoding="utf-8", newline="\n",
    )
    _write_converted_config(source, out, num_experts, moe_int)

    provenance = {
        "source_dir": str(source),
        "source_manifest_sha256": source_manifest.manifest_sha256,
        "scheme": scheme.to_dict(),
        "scheme_digest": scheme.digest(),
        "num_experts": num_experts,
        "moe_intermediate_size": moe_int,
        "weight_dtype": plan.weight_dtype,
        "exactness_contract": (
            "top_k == num_experts; router zeros (uniform 1/E after softmax+renorm); "
            "down_proj slices pre-scaled xE by exact exponent shift (gate/up left "
            "unscaled: silu is not positively homogeneous, so scaling the gate "
            "factor would change the function); shared expert zero-init; dense "
            "output recovered exactly when every expert runs"
        ),
        "output_file_sha256": file_hashes,
        "storage": {
            "source_bytes": plan.source_bytes,
            "estimated_output_bytes": plan.estimated_output_bytes,
            "storage_delta_bytes": plan.storage_delta_bytes,
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


def _write_converted_config(source: Path, out: Path, num_experts: int, moe_int: int) -> None:
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    text_config = config.get("text_config", config)
    kept_top = {k: v for k, v in config.items() if k not in {"architectures", "model_type"}}
    kept_top["architectures"] = ["Qwen3_5MoeForConditionalGeneration"]
    kept_top["model_type"] = "qwen3_5_moe"
    new_text = dict(text_config)
    new_text["model_type"] = "qwen3_5_moe_text"
    new_text["num_experts"] = num_experts
    new_text["num_experts_per_tok"] = num_experts
    new_text["moe_intermediate_size"] = moe_int
    new_text["shared_expert_intermediate_size"] = _SHARED_EXPERT_INTERMEDIATE
    kept_top["text_config"] = new_text
    (out / "config.json").write_text(
        json.dumps(kept_top, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n"
    )
