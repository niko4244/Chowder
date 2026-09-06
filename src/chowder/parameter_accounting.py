"""Active-parameter accounting for real model directories (program Phase 11).

docs/QWEN38_SPARSE_PROGRAM.md Phase 11 requires *authoritative accounting*
of total vs per-token-active parameters, with separate routed/shared/MTP/
vision/embedding rows, and states the gate this module enforces
mechanically: "Do not label a model 'A4B' unless measured routing
geometry supports the claim."

How measurement happens — headers, not hypotheses
-------------------------------------------------
Every parameter count in a `ParameterAccounting` is computed from real
safetensors headers read out of the directory's shard files: the 8-byte
header-length prefix and the JSON header that follows it carry each
tensor's dtype and shape, which is all that is needed for a census. The
safetensors *library* is deliberately not imported (it is not a base
dependency and accounting must never silently require the training
runtime): parsing the header format with the standard library is exact,
not an approximation.

The only values taken from `config.json` are `model_type` (for provenance)
and `num_experts_per_tok` for sparse models — the one number headers
cannot supply, since routing *width* is a runtime choice recorded in
config, not a tensor shape. For a sparse model, a missing or
non-integer `num_experts_per_tok` is a hard `ParameterAccountingError`,
never a defaulted value: an A-label built on an assumed top-k would be
exactly the fabrication Phase 11 exists to prevent.

Classification (each tensor lands in exactly one category)
----------------------------------------------------------
`mtp`, `vision`, `embedding`, `shared_expert`, `router`, `routed_expert`,
`dense_ffn`, `attention_and_deltanet`, `layernorm`, `other`. Category
names come from the real tensor namespaces observed on parent A
(`model.language_model.layers.*` with `self_attn.*` / `linear_attn.*` /
`mlp.*`, plus `model.visual.*`, `mtp.*`, `lm_head`,
`embed_tokens`) — not from a generic heuristic. Anything unrecognized
is *counted* in `other` (never silently dropped) and its names stay
visible in the report.

Active-parameter semantics, stated honestly
-------------------------------------------
- Dense model: every parameter is active on every token;
  `active_parameters == total_parameters` with `active_definition`
  saying exactly that.
- Sparse model: routed experts run only for their selected tokens and
  the router runs to select them, so
  `active = total − routed_experts − router`. The shared expert,
  attention/GatedDeltaNet, embeddings, MTP, vision, and norms are all
  computed for every token and therefore count as active. The definition
  string is carried on the object so a report can never drop it.

The a-label gate
----------------
`ParameterAccounting.a_label()` refuses to return anything unless
measured routing geometry exists (fused-expert tensors were actually
counted, their shapes agreed across layers, and `num_experts_per_tok`
was actually recorded in config). "A4B" as a *string label* is therefore
inconstructible without evidence. There is no flag to override this.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

# Bytes per element for the dtypes safetensors defines. An unknown dtype
# is a hard stop: guessing a size would fabricate the accounting.
_DTYPE_BYTES: Mapping[str, int] = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "U64": 8,
    "I32": 4,
    "U32": 4,
    "I16": 2,
    "U16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
}

_HEADER_KEY = "__metadata__"


class ParameterAccountingError(ValueError):
    """A model directory cannot be accounted honestly."""


def _numel(shape: list[Any]) -> int:
    total = 1
    for dim in shape:
        if isinstance(dim, bool) or not isinstance(dim, int) or dim < 0:
            raise ParameterAccountingError(f"invalid tensor dimension {dim!r} in shape {shape!r}")
        total *= dim
    return total


def _tensor_parameters(dtype: str, shape: list[Any]) -> int:
    width = _DTYPE_BYTES.get(dtype)
    if width is None:
        raise ParameterAccountingError(
            f"unknown safetensors dtype {dtype!r}; refusing to guess an element size"
        )
    return _numel(shape)


def read_safetensors_header(path: str | Path) -> dict[str, dict[str, Any]]:
    """Parse one safetensors file's header without importing safetensors.

    Returns tensor-name -> {"dtype": str, "shape": [int, ...]}. Data is
    never read past the header, so this works on arbitrarily large shards
    in bounded memory.
    """
    shard = Path(path)
    if not shard.is_file():
        raise ParameterAccountingError(f"shard is not an existing file: {shard}")
    with shard.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ParameterAccountingError(f"{shard.name}: truncated header-length prefix")
        (header_len,) = struct.unpack("<Q", prefix)
        raw = handle.read(header_len)
        if len(raw) != header_len:
            raise ParameterAccountingError(f"{shard.name}: truncated header")
    try:
        header = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ParameterAccountingError(f"{shard.name}: header is not valid JSON ({exc})") from exc
    tensors: dict[str, dict[str, Any]] = {}
    for name, entry in header.items():
        if name == _HEADER_KEY:
            continue
        if not isinstance(entry, dict) or "dtype" not in entry or "shape" not in entry:
            raise ParameterAccountingError(f"{shard.name}: malformed header entry for {name!r}")
        tensors[name] = {"dtype": entry["dtype"], "shape": list(entry["shape"])}
    return tensors


def _classify(name: str) -> str:
    """Classify one real tensor name into its Phase 11 category.

    Order matters: MTP/vision namespaces win over everything, then
    embedding identities, then within-layer specialization (experts →
    shared expert → router → dense FFN → attention/GatedDeltaNet →
    norms), with `other` as the never-silent remainder.
    """
    if name == "lm_head.weight" or ".embed_tokens." in name or name.startswith("embed_tokens."):
        return "embedding"
    if name.startswith("mtp."):
        return "mtp"
    if name.startswith("model.visual.") or ".visual." in name:
        return "vision"
    if ".mlp.experts." in name:
        return "routed_expert"
    if ".mlp.shared_expert" in name or ".shared_expert_gate" in name:
        return "shared_expert"
    if ".mlp.gate." in name:
        return "router"
    if ".mlp." in name:
        return "dense_ffn"
    if ".self_attn." in name or ".linear_attn." in name:
        return "attention_and_deltanet"
    if name.endswith("layernorm.weight") or name.endswith(".norm.weight") or name == "model.language_model.norm.weight":
        return "layernorm"
    return "other"


@dataclass(frozen=True)
class CategoryTotals:
    """Parameter totals for one category. Absence is zeros, not None."""

    tensors: int
    parameters: int
    bytes: int

    def __post_init__(self) -> None:
        for label in ("tensors", "parameters", "bytes"):
            value = getattr(self, label)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"category {label} must be a non-negative int")

    def to_dict(self) -> dict[str, int]:
        return {"tensors": self.tensors, "parameters": self.parameters, "bytes": self.bytes}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CategoryTotals":
        return cls(tensors=data["tensors"], parameters=data["parameters"], bytes=data["bytes"])


@dataclass(frozen=True)
class RouterGeometry:
    """Routing geometry measured from real tensor shapes — not config defaults.

    `num_experts` and `moe_intermediate_size` come from the fused expert
    tensors' shapes (`(E, 2*moe_int, H)` / `(E, H, moe_int)`),
    cross-checked against the router's `(E, H)` weight. `top_k` comes
    from config and is required for a sparse model.
    """

    num_experts: int
    moe_intermediate_size: int
    top_k: int

    def __post_init__(self) -> None:
        for label in ("num_experts", "moe_intermediate_size", "top_k"):
            value = getattr(self, label)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"router geometry {label} must be a positive int")
        if self.top_k > self.num_experts:
            raise ValueError("router geometry top_k cannot exceed num_experts")

    def to_dict(self) -> dict[str, int]:
        return {
            "num_experts": self.num_experts,
            "moe_intermediate_size": self.moe_intermediate_size,
            "top_k": self.top_k,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RouterGeometry":
        return cls(
            num_experts=data["num_experts"],
            moe_intermediate_size=data["moe_intermediate_size"],
            top_k=data["top_k"],
        )


@dataclass(frozen=True)
class ParameterAccounting:
    """Measured Phase 11 accounting for one model directory."""

    model_dir: str
    model_type: str
    total_parameters: int
    total_bytes: int
    num_tensors: int
    categories: Mapping[str, CategoryTotals]
    active_parameters: int | None
    active_definition: str | None
    router_geometry: RouterGeometry | None

    def __post_init__(self) -> None:
        if not isinstance(self.model_dir, str) or not self.model_dir.strip():
            raise ValueError("model_dir must be a non-empty string")
        if not isinstance(self.model_type, str) or not self.model_type.strip():
            raise ValueError("model_type must be a non-empty string")
        if self.num_tensors <= 0:
            raise ParameterAccountingError("an accounting over zero tensors is not evidence")
        if self.total_parameters <= 0 or self.total_bytes <= 0:
            raise ParameterAccountingError("accounted parameters/bytes must be positive")
        summed = sum(category.parameters for category in self.categories.values())
        if summed != self.total_parameters:
            raise ParameterAccountingError(
                f"category parameters sum to {summed}, total says {self.total_parameters}"
            )
        if (self.active_parameters is None) != (self.active_definition is None):
            raise ParameterAccountingError(
                "active_parameters and active_definition must be set together"
            )
        if self.active_parameters is not None and not (0 < self.active_parameters <= self.total_parameters):
            raise ParameterAccountingError(
                f"active parameters {self.active_parameters} outside (0, total {self.total_parameters}]"
            )

    # -- the Phase 11 gate ---------------------------------------------------

    @property
    def is_sparse(self) -> bool:
        return self.categories.get("routed_expert", CategoryTotals(0, 0, 0)).tensors > 0

    def a_label(self) -> str:
        """Return the honest active-parameter label, or raise.

        Raises `ParameterAccountingError` unless measured routing
        geometry exists — a sparse *count* alone is not enough, the
        top-k must have been recorded in config. Dense models label by
        total parameters. Fractional billions are rounded to one
        decimal and the raw number is always included.
        """
        if self.active_parameters is None or self.router_geometry is None:
            raise ParameterAccountingError(
                "no a-label is constructible: this accounting has no measured "
                "routing geometry (top-k and fused-expert shapes). Phase 11 "
                "forbids labeling active parameters without it."
            )
        active_b = self.active_parameters / 1e9
        return f"A{active_b:.1f}B ({self.active_parameters:,} active parameters/token; top-{self.router_geometry.top_k} of {self.router_geometry.num_experts})"

    # -- reporting -----------------------------------------------------------

    def format_phase11_report(self) -> str:
        """The program doc's Phase 11 report shape, from measured numbers."""
        lines = [
            f"Model directory:          {self.model_dir}",
            f"Model type:               {self.model_type}",
            f"Total parameters:         {self.total_parameters:,}",
        ]
        for name in (
            "embedding",
            "attention_and_deltanet",
            "dense_ffn",
            "routed_expert",
            "shared_expert",
            "router",
            "mtp",
            "vision",
            "layernorm",
            "other",
        ):
            totals = self.categories.get(name)
            if totals is not None and totals.tensors:
                lines.append(
                    f"{name + ':':<26}{totals.parameters:,} params "
                    f"({totals.tensors} tensors, {totals.bytes / 2**30:.2f} GiB)"
                )
        if self.active_parameters is not None:
            lines.append(f"Active/token:             {self.active_parameters:,}")
            assert self.active_definition is not None
            lines.append(f"Active definition:        {self.active_definition}")
        if self.router_geometry is not None:
            geometry = self.router_geometry
            lines.append(
                f"Routing geometry:         top-{geometry.top_k} of {geometry.num_experts} experts "
                f"(moe_intermediate_size {geometry.moe_intermediate_size})"
            )
            lines.append(f"A-label:                  {self.a_label()}")
        else:
            lines.append("A-label:                  none constructible (no measured routing geometry)")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_dir": self.model_dir,
            "model_type": self.model_type,
            "total_parameters": self.total_parameters,
            "total_bytes": self.total_bytes,
            "num_tensors": self.num_tensors,
            "categories": {name: totals.to_dict() for name, totals in sorted(self.categories.items())},
            "active_parameters": self.active_parameters,
            "active_definition": self.active_definition,
            "router_geometry": self.router_geometry.to_dict() if self.router_geometry else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ParameterAccounting":
        geometry = data.get("router_geometry")
        return cls(
            model_dir=data["model_dir"],
            model_type=data["model_type"],
            total_parameters=data["total_parameters"],
            total_bytes=data["total_bytes"],
            num_tensors=data["num_tensors"],
            categories={
                name: CategoryTotals.from_dict(entry)
                for name, entry in data["categories"].items()
            },
            active_parameters=data["active_parameters"],
            active_definition=data["active_definition"],
            router_geometry=RouterGeometry.from_dict(geometry) if geometry else None,
        )


def account_parameters(model_dir: str | Path) -> ParameterAccounting:
    """Account a real model directory from its safetensors headers.

    Reads every `*.safetensors` shard's header (never tensor data),
    classifies every tensor, cross-checks the safetensors index when one
    exists, and — for sparse models — measures routing geometry from the
    expert/router shapes and requires `num_experts_per_tok` in config.
    """
    root = Path(model_dir)
    if not root.is_dir():
        raise ParameterAccountingError(f"model dir is not an existing directory: {root}")

    config_path = root / "config.json"
    if not config_path.is_file():
        raise ParameterAccountingError(f"{root}: config.json is required for accounting")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ParameterAccountingError(f"{root}: config.json is not valid JSON ({exc})") from exc
    text_config = config.get("text_config", config)
    model_type = config.get("model_type")
    if not isinstance(model_type, str) or not model_type.strip():
        raise ParameterAccountingError(f"{root}: config.json has no usable model_type")

    shards = sorted(path for path in root.glob("*.safetensors") if path.is_file())
    if not shards:
        raise ParameterAccountingError(f"{root}: no .safetensors shards found")

    seen: dict[str, tuple[str, str]] = {}
    category_tensors: dict[str, dict[str, dict[str, Any]]] = {}
    for shard in shards:
        for name, entry in read_safetensors_header(shard).items():
            if name in seen:
                raise ParameterAccountingError(
                    f"tensor {name!r} appears in both {seen[name][0]} and {shard.name}; "
                    "a model directory with duplicated tensor names cannot be accounted"
                )
            seen[name] = (shard.name, entry["dtype"])
            category_tensors.setdefault(_classify(name), {})[name] = entry

    # cross-check against the index when the directory ships one
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ParameterAccountingError(f"index is not valid JSON ({exc})") from exc
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, Mapping):
            raise ParameterAccountingError("index weight_map is missing or malformed")
        indexed = set(weight_map)
        headered = set(seen)
        if indexed != headered:
            missing = sorted(indexed - headered)[:5]
            extra = sorted(headered - indexed)[:5]
            raise ParameterAccountingError(
                "index/shard mismatch: "
                f"{len(indexed - headered)} indexed tensors absent from shard headers "
                f"(e.g. {missing}); {len(headered - indexed)} header tensors absent from "
                f"the index (e.g. {extra})"
            )

    categories: dict[str, CategoryTotals] = {}
    total_parameters = 0
    total_bytes = 0
    for name, entries in category_tensors.items():
        params = 0
        nbytes = 0
        for entry in entries.values():
            count = _tensor_parameters(entry["dtype"], entry["shape"])
            params += count
            nbytes += count * _DTYPE_BYTES[entry["dtype"]]
        categories[name] = CategoryTotals(
            tensors=len(entries), parameters=params, bytes=nbytes
        )
        total_parameters += params
        total_bytes += nbytes

    routed = categories.get("routed_expert", CategoryTotals(0, 0, 0))
    if routed.tensors == 0:
        return ParameterAccounting(
            model_dir=str(root),
            model_type=model_type,
            total_parameters=total_parameters,
            total_bytes=total_bytes,
            num_tensors=len(seen),
            categories=categories,
            active_parameters=total_parameters,
            active_definition=(
                "dense model: every parameter is computed for every token"
            ),
            router_geometry=None,
        )

    geometry = _measure_router_geometry(root, category_tensors, text_config)
    router = categories.get("router", CategoryTotals(0, 0, 0))
    active = total_parameters - routed.parameters - router.parameters
    definition = (
        "total - routed_experts - router "
        f"(-{routed.parameters:,} routed, -{router.parameters:,} router); "
        "shared expert, attention/GatedDeltaNet, embeddings, MTP, vision and "
        "norms run on every token and count as active"
    )
    return ParameterAccounting(
        model_dir=str(root),
        model_type=model_type,
        total_parameters=total_parameters,
        total_bytes=total_bytes,
        num_tensors=len(seen),
        categories=categories,
        active_parameters=active,
        active_definition=definition,
        router_geometry=geometry,
    )


def _measure_router_geometry(
    root: Path,
    category_tensors: Mapping[str, Mapping[str, dict[str, Any]]],
    text_config: Mapping[str, Any],
) -> RouterGeometry:
    """Derive (E, moe_int, top_k) from real shapes; fail closed on gaps.

    The fused-expert naming verified for qwen3_5_moe in
    transformers 5.16.1 is used to *read shapes*: gate_up_proj entries
    must be 3-D `(E, 2*moe_int, H)` with a consistent H across layers,
    and every routed tensor's first dimension must be the same E. The
    router weight `(E, H)` corroborates E when present. top_k must come
    from config (`num_experts_per_tok`) — it is not guessable from
    shapes, and a sparse accounting without it is an error, not a None.
    """
    gate_up_shapes: list[list[int]] = []
    down_shapes: list[list[int]] = []
    for name, entry in sorted(category_tensors.get("routed_expert", {}).items()):
        shape = entry["shape"]
        if name.endswith(".gate_up_proj.weight"):
            gate_up_shapes.append(shape)
        elif name.endswith(".down_proj.weight"):
            down_shapes.append(shape)

    if not gate_up_shapes or not down_shapes:
        raise ParameterAccountingError(
            "routed tensors exist but no fused gate_up_proj/down_proj shapes were "
            "found; this module measures the verified qwen3_5_moe fused layout "
            "(.mlp.experts.{gate_up_proj,down_proj}.weight) and refuses to guess "
            "geometry from other layouts"
        )

    def _dims(shapes: list[list[int]], label: str) -> tuple[int, int, int]:
        first = shapes[0]
        if len(first) != 3:
            raise ParameterAccountingError(f"fused {label} shape {first!r} is not 3-D")
        if any(shape != first for shape in shapes[1:]):
            raise ParameterAccountingError(
                f"fused {label} shapes differ across layers ({shapes[0]!r} vs "
                f"{shapes[1]!r}); per-layer geometry is not supported without "
                "explicit evidence"
            )
        return first[0], first[1], first[2]

    experts, two_moe_int, hidden_gu = _dims(gate_up_shapes, "gate_up_proj")
    experts_down, hidden_down, moe_int_down = _dims(down_shapes, "down_proj")
    if experts != experts_down:
        raise ParameterAccountingError(
            f"expert-count mismatch: gate_up_proj says {experts}, down_proj says {experts_down}"
        )
    if two_moe_int % 2 != 0:
        raise ParameterAccountingError(
            f"gate_up_proj second dim {two_moe_int} is odd; expected 2*moe_intermediate_size"
        )
    moe_int = two_moe_int // 2
    if moe_int != moe_int_down or hidden_gu != hidden_down:
        raise ParameterAccountingError(
            "expert shapes disagree: "
            f"gate_up_proj ({experts}, {two_moe_int}, {hidden_gu}) vs "
            f"down_proj ({experts_down}, {hidden_down}, {moe_int_down})"
        )

    router_entries = category_tensors.get("router", {})
    for entry in router_entries.values():
        shape = entry["shape"]
        if len(shape) == 2 and shape[0] != experts:
            raise ParameterAccountingError(
                f"router weight first dim {shape[0]} != measured expert count {experts}"
            )

    config_experts = text_config.get("num_experts")
    if isinstance(config_experts, int) and config_experts != experts:
        raise ParameterAccountingError(
            f"config num_experts {config_experts} != measured expert count {experts}"
        )
    config_moe_int = text_config.get("moe_intermediate_size")
    if isinstance(config_moe_int, int) and config_moe_int != moe_int:
        raise ParameterAccountingError(
            f"config moe_intermediate_size {config_moe_int} != shape-derived {moe_int}"
        )

    top_k = text_config.get("num_experts_per_tok")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ParameterAccountingError(
            "config has no usable num_experts_per_tok; per-token active parameters "
            "cannot be measured without it, and Phase 11 forbids assuming a top-k"
        )
    if not (0 < top_k <= experts):
        raise ParameterAccountingError(
            f"config num_experts_per_tok {top_k} outside (0, {experts}]"
        )
    return RouterGeometry(
        num_experts=experts, moe_intermediate_size=moe_int, top_k=top_k
    )


def write_accounting_json(accounting: ParameterAccounting, output_path: str | Path) -> str:
    """Persist the accounting as evidence; returns the file's sha256."""
    import hashlib

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(accounting.to_dict(), indent=2, sort_keys=True, ensure_ascii=False)
    path.write_text(payload, encoding="utf-8", newline="\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()
