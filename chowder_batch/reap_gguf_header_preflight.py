"""CPU-only GGUF MoE topology preflight against a pruned BF16 model record.

This inspects GGUF *metadata* through a memory map, not weight values, and is
NOT a checkpoint-content attestation, inference load, speed measurement, or
registered frozen gate. It refuses inconsistent expert/tensor topology before
a separate authorized GPU or CPU inference experiment is considered.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


class GGUFHeaderRefusal(ValueError):
    """A GGUF header does not support the declared REAP model topology."""


_CONFIG_KEYS = {
    "qwen35moe.block_count": "num_hidden_layers",
    "qwen35moe.expert_count": "num_experts",
    "qwen35moe.expert_used_count": "num_experts_per_tok",
    "qwen35moe.expert_feed_forward_length": "moe_intermediate_size",
    "qwen35moe.embedding_length": "hidden_size",
    "qwen35moe.vocab_size": "vocab_size",
}


def verify_header_topology(
    fields: Mapping[str, Any], tensors: Mapping[str, tuple[int, ...]],
    model_config: Mapping[str, Any], prune: Mapping[str, Any],
    *, expected_file_type: int = 15,
) -> dict[str, Any]:
    """Validate all routed layers and expert-shape axes, without model load."""
    if model_config.get("model_type") != "qwen3_5_moe":
        raise GGUFHeaderRefusal("expected a Qwen3.5/3.6 MoE model config")
    text = model_config.get("text_config")
    if not isinstance(text, Mapping):
        raise GGUFHeaderRefusal("model config lacks text_config")
    if fields.get("general.architecture") != "qwen35moe":
        raise GGUFHeaderRefusal("GGUF architecture does not match Qwen35 MoE")
    if expected_file_type not in (7, 15, 32):
        raise GGUFHeaderRefusal("only preregistered BF16/Q8_0/Q4_K_M GGUF types are supported")
    if (fields.get("general.file_type") != expected_file_type
            or fields.get("general.quantization_version") != 2):
        raise GGUFHeaderRefusal("GGUF file type or quantization version does not match frozen stage")
    for gguf_key, cfg_key in _CONFIG_KEYS.items():
        if cfg_key == "vocab_size" and fields.get(gguf_key) is None:
            # llama.cpp vocab-size metadata is optional, but embedding tensor
            # shape is still checked below against the source vocabulary.
            continue
        desired = text.get(cfg_key)
        if (isinstance(desired, bool) or not isinstance(desired, int)
                or desired <= 0 or fields.get(gguf_key) != desired):
            raise GGUFHeaderRefusal(f"{gguf_key}: GGUF/model config mismatch")
    blocks = text["num_hidden_layers"]
    experts = text["num_experts"]
    hidden = text["hidden_size"]
    intermediate = text["moe_intermediate_size"]
    vocab = text["vocab_size"]
    if (not isinstance(prune.get("keep"), int) or prune.get("keep") != experts
            or not isinstance(prune.get("of"), int) or prune["of"] <= experts):
        raise GGUFHeaderRefusal("pruning parent/expert count is incompatible")
    selected = prune.get("kept_experts")
    if not isinstance(selected, Mapping) or set(selected) != {str(i) for i in range(blocks)}:
        raise GGUFHeaderRefusal("pruning record lacks an exact routed-layer census")
    for i in range(blocks):
        row = selected[str(i)]
        if (not isinstance(row, list) or len(row) != experts
                or any(type(x) is not int or x < 0 or x >= prune["of"] for x in row)
                or len(set(row)) != experts):
            raise GGUFHeaderRefusal(f"pruning expert survivor census invalid for block {i}")
    expected: dict[str, tuple[int, ...]] = {
        "output.weight": (hidden, vocab),
        "token_embd.weight": (hidden, vocab),
    }
    for i in range(blocks):
        prefix = f"blk.{i}."
        expected.update({
            prefix + "ffn_gate_inp.weight": (hidden, experts),
            prefix + "ffn_down_exps.weight": (intermediate, hidden, experts),
            prefix + "ffn_gate_exps.weight": (hidden, intermediate, experts),
            prefix + "ffn_up_exps.weight": (hidden, intermediate, experts),
        })
    for name, shape in expected.items():
        if tensors.get(name) != shape:
            raise GGUFHeaderRefusal(f"missing/mismatched routed tensor {name}: expected {shape}")
    for name in tensors:
        if name.startswith("blk."):
            parts = name.split(".")
            if len(parts) > 1 and parts[1].isdigit() and int(parts[1]) >= blocks:
                raise GGUFHeaderRefusal(f"GGUF contains out-of-range model block: {name}")
    return {
        "status": "GGUF_METADATA_TOPOLOGY_MATCH_ONLY",
        "architecture": "qwen35moe", "file_type": fields["general.file_type"],
        "blocks": blocks, "experts_per_block": experts,
        "top_k": text["num_experts_per_tok"],
        "checked_tensor_shapes": len(expected),
        "total_gguf_tensor_headers": len(tensors),
        "limits": "Not a SHA-256 of GGUF weight bytes, checkpoint ancestry proof, "
                  "actual llama.cpp load, decode test or speed/fit gate",
    }


def _field(reader: Any, key: str) -> Any:
    field = reader.get_field(key)
    if field is None:
        return None
    if len(field.data) != 1:
        raise GGUFHeaderRefusal(f"GGUF metadata {key} must be a single scalar")
    part = field.parts[field.data[0]]
    if len(part) > 1:
        return bytes(part).decode("utf-8", "strict")
    return int(part[0])


def _sha256_small(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_preflight(gguf: Path, model_dir: Path, *, expected_file_type: int = 15) -> dict[str, Any]:
    """Check a real GGUF against local model metadata, without tensor loads."""
    try:
        from gguf import GGUFReader
    except ImportError as exc:
        raise GGUFHeaderRefusal("install optional gguf metadata reader first") from exc
    config_path = model_dir / "config.json"
    prune_path = model_dir / "reap-prune.json"
    if not gguf.is_file() or not config_path.is_file() or not prune_path.is_file():
        raise GGUFHeaderRefusal("GGUF file, model config, or pruning record missing")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    prune = json.loads(prune_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(prune, dict):
        raise GGUFHeaderRefusal("model config/pruning record must be JSON objects")
    reader = GGUFReader(str(gguf), "r")  # mmap, never tensors' .data or pretrained load
    fields = {k: _field(reader, k) for k in (*_CONFIG_KEYS, "general.architecture",
                                               "general.file_type", "general.quantization_version")}
    tensors: dict[str, tuple[int, ...]] = {}
    for tensor in reader.tensors:
        if tensor.name in tensors:
            raise GGUFHeaderRefusal(f"GGUF has duplicate tensor header {tensor.name}")
        tensors[tensor.name] = tuple(int(v) for v in tensor.shape)
    receipt = verify_header_topology(
        fields, tensors, config, prune, expected_file_type=expected_file_type,
    )
    return {
        "schema": "chowder.reap-gguf-header-preflight.v1",
        **receipt,
        "gguf_path": str(gguf.resolve()), "gguf_size_bytes": gguf.stat().st_size,
        "bf16_config_sha256": _sha256_small(config_path),
        "prune_record_sha256": _sha256_small(prune_path),
        "gguf_weight_sha256": "NOT_REHASHED_ARCHIVAL_IDENTITY_NOT_REVERIFIED",
        "gpu_hours_consumed": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--bf16-model", type=Path, required=True)
    parser.add_argument("--expected-file-type", type=int, choices=(7, 15, 32), default=15)
    parser.add_argument("--out", type=Path, help="new output receipt (never overwrite)")
    args = parser.parse_args(argv)
    try:
        report = run_preflight(
            args.gguf, args.bf16_model, expected_file_type=args.expected_file_type,
        )
    except (OSError, ValueError, UnicodeError) as exc:
        parser.exit(2, f"REAP GGUF header preflight REFUSED: {exc}\n")
    encoded = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
