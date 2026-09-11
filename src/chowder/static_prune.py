"""Static FFN channel pruning — the thing that actually won.

`hot_core_upcycle` builds a routed MoE whose active budget is split between a
fixed hot core and routed cold slices. Measured at f=0.28 on the 9B, it loses to
simply *deleting* the unranked channels: geometric-mean 3.008x dense for the MoE
init against **2.366x for this**, at identical active compute, with no router and
no healing (`docs/HOT_CORE_VS_STATIC_PRUNE.md`).

This module is therefore the recommended path, and it is also the simplest
correctness story in the program:

* **No scaling, anywhere.** A pruned SwiGLU FFN is just a smaller SwiGLU FFN:
  the output is `sum over kept channels of down[:,c] * silu(gate[c]@x) * (up[c]@x)`,
  so dropping channels drops terms and the survivors keep their own weights. There
  is no renormalisation to compensate for, which is the entire class of bug that
  produced the partition converter's x(E/k) blowup and the hot core's x2
  sigmoid(0) correction.
* **No architecture change.** The output is still `qwen3_5` /
  `Qwen3_5ForConditionalGeneration` with a smaller `intermediate_size`. Nothing
  about MoE loading, router tie-breaks, raw-`nn.Parameter` expert banks or
  quantisation-skip lists applies. On the real 9B exactly 96 tensors mention
  `intermediate_size` (32 layers x gate/up/down) and one config field, and the
  vision tower carries its own `intermediate_size` which is untouched.
* **Storage falls with compute.** Unlike the MoE, which stored all 12,288 channels
  and activated 3,440, this stores only what it runs. Pruning the 9B to 3,440
  channels takes total parameters from 9.410B to **5.931B**, which is also the
  active count because nothing is conditional.

Channels are emitted in **rank order, hottest first**. The FFN sum is invariant
under any consistent permutation of (gate rows, up rows, down columns), so this
costs nothing and buys something: a further prune is then a pure prefix slice of
an already-pruned checkpoint, no re-ranking required.

Like the other converters this is stdlib-only byte surgery and never imports
torch.
"""
from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .channel_importance import ChannelRanking
from .dense_to_moe import (
    _DTYPE_BYTES,
    _DOWN_SUFFIX,
    _GATE_SUFFIX,
    _LAYERS_PREFIX,
    _MLP_SUFFIXES,
    _PASSTHROUGH_FILES,
    _UP_SUFFIX,
    _read_header,
    _tensor_entries,
)
from .hot_core_upcycle import _gather_columns, _gather_rows
from .local_model_manifest import build_local_model_manifest, write_manifest_file
from .provenance import sha256_file


class StaticPruneError(ValueError):
    """A prune cannot be performed honestly."""


@dataclass(frozen=True)
class StaticPrunePlan:
    """Shapes and budget, before any bytes move."""

    hidden_size: int
    intermediate_size: int
    keep_channels: int
    num_layers: int
    weight_dtype: str
    params_per_channel_all_layers: int
    dense_ffn_params: int
    pruned_ffn_params: int
    always_on_params: int

    @property
    def total_params(self) -> int:
        """Also the ACTIVE count: nothing in a pruned dense model is conditional."""
        return self.always_on_params + self.pruned_ffn_params

    def to_dict(self) -> dict[str, Any]:
        return {
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "keep_channels": self.keep_channels,
            "keep_fraction": self.keep_channels / self.intermediate_size,
            "num_layers": self.num_layers,
            "weight_dtype": self.weight_dtype,
            "params_per_channel_all_layers": self.params_per_channel_all_layers,
            "dense_ffn_params": self.dense_ffn_params,
            "pruned_ffn_params": self.pruned_ffn_params,
            "always_on_params": self.always_on_params,
            "total_params": self.total_params,
            "active_params": self.total_params,
            "ffn_params_removed": self.dense_ffn_params - self.pruned_ffn_params,
        }


def plan_static_prune(
    model_dir: str | Path, ranking: ChannelRanking, *, keep_channels: int
) -> StaticPrunePlan:
    """Read the source headers and price the prune. Moves no bytes."""
    source = Path(model_dir)
    config_path = source / "config.json"
    if not config_path.is_file():
        raise StaticPruneError(f"no config.json in {source}")
    text_config = json.loads(config_path.read_text(encoding="utf-8"))
    text_config = text_config.get("text_config", text_config)
    hidden = int(text_config["hidden_size"])
    intermediate = int(text_config["intermediate_size"])

    if not 0 < keep_channels < intermediate:
        raise StaticPruneError(
            f"keep_channels must be in 1..{intermediate - 1}; got {keep_channels}. "
            "Keeping every channel is a copy, not a prune"
        )
    if ranking.intermediate_size != intermediate:
        raise StaticPruneError(
            f"ranking was measured at intermediate_size {ranking.intermediate_size} "
            f"but this checkpoint has {intermediate}"
        )
    source_manifest = build_local_model_manifest(source, mode="fast")
    if ranking.source_manifest_sha256 != source_manifest.manifest_sha256:
        raise StaticPruneError(
            "the ranking was measured on a different checkpoint "
            f"({ranking.source_manifest_sha256[:12]}...) than the one being pruned "
            f"({source_manifest.manifest_sha256[:12]}...). Pruning with a foreign "
            "ranking deletes the wrong channels and the result still loads"
        )

    dtypes: set[str] = set()
    layers: set[int] = set()
    total_params = 0
    for shard in sorted(source.glob("*.safetensors")):
        for name, entry in _tensor_entries(_read_header(shard)).items():
            count = 1
            for dim in entry["shape"]:
                count *= dim
            total_params += count
            if name.startswith(_LAYERS_PREFIX) and name.endswith(_MLP_SUFFIXES):
                layers.add(int(name[len(_LAYERS_PREFIX):].split(".")[0]))
                dtypes.add(entry["dtype"])
    if not layers:
        raise StaticPruneError(
            f"no dense MLP tensors found under {_LAYERS_PREFIX} in {source}"
        )
    if len(dtypes) != 1:
        raise StaticPruneError(
            f"mixed MLP dtypes {sorted(dtypes)}; refusing to prune a heterogeneous FFN"
        )
    missing = sorted(layers - set(ranking.ranking))
    if missing:
        raise StaticPruneError(
            f"ranking is missing layers {missing[:8]}{'...' if len(missing) > 8 else ''}; "
            "refusing to fall back to index order, which deletes the wrong channels"
        )

    num_layers = len(layers)
    per_channel = 3 * hidden * num_layers
    dense_ffn = intermediate * per_channel
    return StaticPrunePlan(
        hidden_size=hidden,
        intermediate_size=intermediate,
        keep_channels=keep_channels,
        num_layers=num_layers,
        weight_dtype=dtypes.pop(),
        params_per_channel_all_layers=per_channel,
        dense_ffn_params=dense_ffn,
        pruned_ffn_params=keep_channels * per_channel,
        always_on_params=total_params - dense_ffn,
    )


def prune_checkpoint(
    model_dir: str | Path,
    output_dir: str | Path,
    ranking: ChannelRanking,
    *,
    keep_channels: int,
) -> dict[str, Any]:
    """Delete all but the top `keep_channels` FFN channels, by measured rank.

    Two-pass streaming like the other converters: headers are scanned once to
    build the output layout and a regenerated index, then each shard is written
    atomically. Everything outside the 3 MLP projections per layer -- attention,
    `linear_attn`, embeddings, norms, the vision tower (which has its own
    `intermediate_size`) -- is copied byte-identical.
    """
    source = Path(model_dir)
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise StaticPruneError(f"output dir exists and is not empty: {out}")

    plan = plan_static_prune(source, ranking, keep_channels=keep_channels)
    hidden = plan.hidden_size
    intermediate = plan.intermediate_size
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
    # exact output name -> (layer, which projection); never a substring test, since
    # the vision tower's blocks also carry `.mlp.` tensors
    pruned_targets: dict[str, tuple[str, str]] = {}
    for shard in sorted(source.glob("*.safetensors")):
        for name, entry in _tensor_entries(_read_header(shard)).items():
            target = shard_headers[shard.name]
            if name.startswith(_LAYERS_PREFIX) and name.endswith(_MLP_SUFFIXES):
                layer = name[len(_LAYERS_PREFIX):].split(".")[0]
                if name.endswith(_DOWN_SUFFIX):
                    shape, which = [hidden, keep_channels], "down"
                else:
                    shape = [keep_channels, hidden]
                    which = "gate" if name.endswith(_GATE_SUFFIX) else "up"
                target[name] = {"dtype": entry["dtype"], "shape": shape}
                pruned_targets[name] = (layer, which)
            else:
                target[name] = {"dtype": entry["dtype"], "shape": entry["shape"]}

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

    kept: dict[str, tuple[int, ...]] = {
        layer: tuple(ranking.ranking[int(layer)][:keep_channels])
        for layer, _ in set(pruned_targets.values())
    }

    # ---- pass 2: stream, writing each shard atomically ----------------------
    file_hashes: dict[str, str] = {}
    out.mkdir(parents=True, exist_ok=True)
    for shard in sorted(source.glob("*.safetensors")):
        shard_name = shard.name
        source_header = _read_header(shard)
        new_header = shard_headers[shard_name]
        header_blob = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
        header_blob += b" " * ((8 - len(header_blob) % 8) % 8)

        out_path = out / shard_name
        tmp_path = out_path.with_suffix(".tmp")
        with tmp_path.open("wb") as dst:
            dst.write(struct.pack("<Q", len(header_blob)))
            dst.write(header_blob)
            for name in sorted(new_header):
                kinded = pruned_targets.get(name)
                if kinded is None:
                    dst.write(read_part(shard_name, source_header[name]))
                    continue
                layer, which = kinded
                raw = read_part(shard_name, source_header[name])
                channels = kept[layer]
                if which == "down":
                    # down_proj is (hidden, intermediate): a channel is a column
                    dst.write(_gather_columns(
                        raw, channels, hidden=hidden,
                        intermediate=intermediate, width=width))
                else:
                    # gate/up are (intermediate, hidden): a channel is a row.
                    # Verbatim -- no scaling exists in a prune.
                    dst.write(_gather_rows(raw, channels, hidden=hidden, width=width))
        os.replace(tmp_path, out_path)
        file_hashes[shard_name] = sha256_file(out_path)

    for handle in src_handles.values():
        handle.close()

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
    _write_pruned_config(source, out, keep_channels)

    provenance = {
        "converter": "static_prune",
        "source_dir": str(source),
        "source_manifest_sha256": source_manifest.manifest_sha256,
        "ranking": {
            "method": ranking.method,
            "digest": ranking.digest(),
            "calibration": dict(ranking.calibration),
            "concentration": dict(ranking.concentration),
        },
        "plan": plan.to_dict(),
        "channel_order": (
            "rank order, hottest first. The FFN sum is invariant under any "
            "consistent permutation of gate rows / up rows / down columns, so this "
            "is free and makes a further prune a pure prefix slice."
        ),
        "scaling": (
            "NONE. A pruned SwiGLU FFN is a smaller SwiGLU FFN -- dropping channels "
            "drops terms and the survivors keep their own weights. No "
            "renormalisation exists to compensate for, unlike the partition "
            "converter's xE (and its x(E/k) blowup at smaller top_k) or the hot "
            "core's x2 sigmoid(0) correction."
        ),
        "exactness_contract": (
            "NONE, by construction: the pruned model computes the dense function "
            "restricted to the kept channels. It is not an approximation of the "
            "dense FFN, it is a smaller FFN. Measured cost on the 9B at "
            "keep=3440 with a corpus-wide ranking: 1.859x dense perplexity on one "
            "held-out split and 3.011x on another."
        ),
        "output_file_sha256": file_hashes,
        "storage": {
            "source_bytes": sum(f.stat().st_size for f in source.glob("*.safetensors")),
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


def _write_pruned_config(source: Path, out: Path, keep_channels: int) -> None:
    """Only `intermediate_size` changes. The architecture is unchanged."""
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if "text_config" in config:
        config["text_config"] = dict(config["text_config"])
        config["text_config"]["intermediate_size"] = keep_channels
    else:
        config["intermediate_size"] = keep_channels
    (out / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n"
    )
