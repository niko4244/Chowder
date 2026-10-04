"""Architecture and resource inventory for the low-rank vocabulary experiment.

Audits an actual checkpoint on disk -- identity, tensor shapes, tying status,
quantization, parameter accounting -- without loading any weights, and derives
the resource estimates (VRAM, RAM, FLOPs, workspace) that gate the rest of the
experiment. Emits a JSON report.

Usage:
    python chowder_batch/low_rank_inventory.py --model F:/llm-models/Qwen3.8-9B-abliterated-25-bf16 --out F:/chowder-campaign/low-rank-embed/inventory.json
"""
from __future__ import annotations

import argparse
import json
import os
import struct
from pathlib import Path
from typing import Any

BYTES_PER = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1, "I8": 1}

EMBED_KEYS = (
    "model.language_model.embed_tokens.weight",
    "model.embed_tokens.weight",
    "embed_tokens.weight",
)
HEAD_KEYS = ("lm_head.weight",)


def _read_header(shard: Path) -> dict[str, Any]:
    with open(shard, "rb") as handle:
        (header_len,) = struct.unpack("<Q", handle.read(8))
        return json.loads(handle.read(header_len))


def _load_config(model_dir: Path) -> dict[str, Any]:
    return json.loads((model_dir / "config.json").read_text(encoding="utf-8"))


def _flatten_index(model_dir: Path) -> dict[str, str] | None:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        return None
    return json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]


def audit_checkpoint(model_dir: str) -> dict[str, Any]:
    model_dir = Path(model_dir)
    cfg = _load_config(model_dir)
    text_cfg = cfg.get("text_config", cfg)
    weight_map = _flatten_index(model_dir)
    if weight_map is None:
        raise FileNotFoundError(f"experiment requires a sharded checkpoint index in {model_dir}")

    shard_names = sorted(set(weight_map.values()))
    headers: dict[str, dict[str, Any]] = {}
    dtypes: set[str] = set()
    total_tensor_bytes = 0
    for shard in shard_names:
        header = _read_header(model_dir / shard)
        headers[shard] = header
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            dtypes.add(meta["dtype"])
            start, stop = meta["data_offsets"]
            total_tensor_bytes += stop - start

    def find(candidates: tuple[str, ...]) -> dict[str, Any] | None:
        for key in candidates:
            if key in weight_map:
                meta = headers[weight_map[key]][key]
                return {"key": key, "shard": weight_map[key], **meta}
        return None

    embed = find(EMBED_KEYS)
    head = find(HEAD_KEYS)
    if embed is None or head is None:
        raise RuntimeError("embedding or lm_head tensor not found in checkpoint")

    dtype_bytes = sum(BYTES_PER.get(d, 2) for d in dtypes) / len(dtypes)
    quant = cfg.get("quantization_config")

    vocab, hidden = head["shape"]
    embed_vocab, embed_hidden = embed["shape"]
    tied = embed["key"] == head["key"]

    # Unique vs per-token parameters.
    unique_params = round(total_tensor_bytes / dtype_bytes)
    # A token's forward pass touches one embedding row (hidden params) and the
    # full lm_head matrix (vocab x hidden params), plus every decoder weight.
    decoder_keys = {k for k in weight_map if ".layers." in k}
    decoder_params = 0
    for key in decoder_keys:
        meta = headers[weight_map[key]][key]
        start, stop = meta["data_offsets"]
        decoder_params += (stop - start) // dtype_bytes
    used_per_token = (
        hidden  # embedding row
        + vocab * head["shape"][1]  # full output projection
        + decoder_params  # all decoder weights are used every token
    )

    # FLOPs for the two compression-target matrices, per token.
    embed_flops = 2 * hidden
    head_flops = 2 * vocab * hidden

    report: dict[str, Any] = {
        "model_dir": str(model_dir),
        "identity": {
            "architectures": cfg.get("architectures"),
            "model_type": cfg.get("model_type"),
            "text_model_type": text_cfg.get("model_type"),
            "torch_dtype": cfg.get("torch_dtype") or cfg.get("dtype") or text_cfg.get("dtype"),
            "tokenizer_files": sorted(
                f for f in os.listdir(model_dir) if f.startswith("tokenizer") or f in {"vocab.json", "merges.txt"}
            ),
            "quantization_config": quant,
            "is_quantized": quant is not None,
            "source_weight_format": "safetensors (dense, trainable)" if quant is None else "quantized container",
            "gguf_only": False,
        },
        "architecture": {
            "vocab_size": vocab,
            "hidden_size": hidden,
            "num_hidden_layers": text_cfg.get("num_hidden_layers"),
            "intermediate_size": text_cfg.get("intermediate_size"),
            "head_dim": text_cfg.get("head_dim"),
            "tie_word_embeddings": text_cfg.get("tie_word_embeddings", tied),
            "observed_tied": tied,
            "layer_types": text_cfg.get("layer_types", [])[:4] + ["..."],
        },
        "tensors": {
            "input_embedding": {
                "key": embed["key"],
                "shape": embed["shape"],
                "dtype": embed["dtype"],
                "parameters": embed_vocab * embed_hidden,
                "bytes": (embed["data_offsets"][1] - embed["data_offsets"][0]),
            },
            "output_projection": {
                "key": head["key"],
                "shape": head["shape"],
                "dtype": head["dtype"],
                "parameters": vocab * hidden,
                "bytes": (head["data_offsets"][1] - head["data_offsets"][0]),
            },
            "total_tensors": len(weight_map),
            "total_bytes": total_tensor_bytes,
            "dtype_samples": sorted(dtypes),
        },
        "parameters": {
            "unique_total": unique_params,
            "embedding_unique": embed_vocab * embed_hidden,
            "lm_head_unique": vocab * hidden,
            "decoder_unique": decoder_params,
            "used_per_token": used_per_token,
            "embedding_used_per_token": hidden,
            "lm_head_used_per_token": vocab * hidden,
            "note": "lm_head participates in full every decode step; the embedding contributes only the looked-up row",
        },
        "flops_per_token": {
            "embedding": embed_flops,
            "lm_head": head_flops,
            "embedding_at_rank": {
                str(r): 2 * (hidden + r) for r in (2048, 1536, 1024, 512)
            },
            "lm_head_at_rank": {
                str(r): 2 * (vocab + hidden) * r for r in (2048, 1536, 1024, 512)
            },
        },
        "resources": _resource_estimates(total_tensor_bytes, dtype_bytes, vocab, hidden),
    }
    return report


def _resource_estimates(
    total_bytes: int, dtype_bytes: float, vocab: int, hidden: int
) -> dict[str, Any]:
    gb = 1024**3
    matrix_bytes = vocab * hidden * dtype_bytes
    return {
        "teacher_load_vram_gb": round(total_bytes * 1.15 / gb, 2),
        "teacher_load_ram_gb": round(total_bytes * 1.30 / gb, 2),
        "one_matrix_copy_gb": round(matrix_bytes / gb, 2),
        "matrix_as_f32_gb": round(vocab * hidden * 4 / gb, 2),
        "svd_gram_workspace_gb": round(hidden * hidden * 4 / gb, 4),
        "svd_block_workspace_gb": round(65536 * hidden * 4 / gb, 2),
        "factor_store_gb": {
            str(r): round((vocab * r + r * hidden) * 2 / gb, 3) for r in (2048, 1536, 1024, 512)
        },
        "candidate_total_bytes_at_dtype": {
            str(r): round(total_bytes - 2 * matrix_bytes + 2 * (vocab * r + r * hidden) * dtype_bytes)
            for r in (2048, 1536, 1024, 512)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = audit_checkpoint(args.model)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["identity"], indent=2))
    print(json.dumps(report["tensors"]["input_embedding"], indent=2))
    print(json.dumps(report["tensors"]["output_projection"], indent=2))
    print(json.dumps(report["resources"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
