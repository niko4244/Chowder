"""CPU-only per-layer Qwen3.6 REAP128_v2 BF16 inference/model-load smoke.

The 40 decoder layers are loaded one at a time rather than materializing a
35-GiB model into a busy host's RAM. This verifies that the *actual* pruned
checkpoint runs with local tokenizer, embeddings, attention, MoE experts,
final RMSNorm and LM head. It is NOT a Q8/Q4 benchmark or a quality gate.
Do not run concurrently with source hashing or CPU model construction.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from chowder_batch.reap_observe import (
    PREFIX, ShardReader, _load_layer_without_duplicate_weights,
)
from chowder_batch.reap_hidden_checkpoint import sha256_file

CHECKPOINT_LAYERS = (0, 9, 19, 29)


def _activation_identity(model: Path, prompt: str, ids: list[int],
                         full_verification_sha256: str | None) -> dict:
    return {
        "schema": "chowder.reap128-v2-cpu-activation.v1",
        "model": str(model),
        "index_sha256": sha256_file(model / "model.safetensors.index.json"),
        "prune_receipt_sha256": (
            sha256_file(model / "reap-prune.json")
            if (model / "reap-prune.json").is_file() else None
        ),
        "config_sha256": sha256_file(model / "config.json"),
        "input_prompt": prompt,
        "input_tokens": ids,
        "full_verification_sha256": full_verification_sha256,
    }


def _checkpoint_paths(directory: Path, layer: int) -> tuple[Path, Path]:
    return (directory / f"layer-{layer:02d}.safetensors",
            directory / f"layer-{layer:02d}.receipt.json")


def _recover_activation(directory: Path, identity: dict,
                        expected_shape: tuple[int, ...]) -> tuple[int, torch.Tensor | None]:
    from safetensors.torch import load_file

    highest = -1
    recovered = None
    missing_predecessor = False
    for layer in CHECKPOINT_LAYERS:
        saved, receipt = _checkpoint_paths(directory, layer)
        if saved.exists() != receipt.exists():
            raise ValueError(f"incomplete immutable CPU activation checkpoint at layer {layer}")
        if not saved.exists():
            missing_predecessor = True
            continue
        if missing_predecessor:
            raise ValueError("activation checkpoint sequence contains a missing earlier stage")
        if saved.is_symlink() or receipt.is_symlink():
            raise ValueError("activation checkpoint may not be a symlink")
        record = json.loads(receipt.read_text(encoding="utf-8"))
        if (record.get("identity") != identity or record.get("layer") != layer
                or record.get("saved_sha256") != sha256_file(saved)):
            raise ValueError(f"stale/corrupt BF16 activation checkpoint at layer {layer}")
        x = load_file(str(saved), device="cpu").get("hidden")
        if (not isinstance(x, torch.Tensor) or tuple(x.shape) != expected_shape
                or x.dtype != torch.bfloat16 or not bool(torch.isfinite(x).all())):
            raise ValueError(f"invalid saved hidden state at layer {layer}")
        highest, recovered = layer, x
    return highest, recovered


def _save_activation(directory: Path, layer: int, hidden: torch.Tensor,
                     identity: dict) -> None:
    from safetensors.torch import save_file
    import uuid

    saved, receipt = _checkpoint_paths(directory, layer)
    if saved.exists() or receipt.exists():
        raise ValueError(f"refuse overwrite of activation checkpoint at layer {layer}")
    tmp = directory / f".layer-{layer:02d}.{uuid.uuid4().hex}.safetensors"
    save_file({"hidden": hidden.detach().contiguous()}, str(tmp))
    # os.rename refuses an existing destination on Windows. Unlike replace,
    # it cannot silently overwrite a checkpoint published by another worker.
    tmp.rename(saved)
    record = {"identity": identity, "layer": layer, "saved_sha256": sha256_file(saved)}
    with receipt.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, indent=2, sort_keys=True) + "\n")


def run(model: Path, *, prompt: str, max_tokens: int = 16,
        progress_dir: Path | None = None,
        full_verification_sha256: str | None = None) -> dict:
    if not prompt:
        raise ValueError("smoke prompt cannot be empty")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("-1", ""):
        raise ValueError("CPU-only streaming smoke requires CUDA_VISIBLE_DEVICES=-1")
    import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as M
    from transformers import AutoTokenizer
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

    model = model.resolve()
    cfg = json.loads((model / "config.json").read_text(encoding="utf-8"))["text_config"]
    if cfg.get("num_hidden_layers") != 40 or cfg.get("num_experts") != 128:
        raise ValueError("expected completed REAP128_v2 40-layer/128-expert model")
    tc = Qwen3_5MoeTextConfig(**cfg)
    tc._attn_implementation = "sdpa"
    tokenizer = AutoTokenizer.from_pretrained(
        model, local_files_only=True, trust_remote_code=False,
    )
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if not 1 <= len(ids) <= max_tokens:
        raise ValueError("prompt token count exceeds bounded CPU smoke budget")
    token_ids = torch.tensor(ids, dtype=torch.long)
    resumed_from = -1
    state = None
    activation_identity = None
    if progress_dir is not None:
        progress_dir = progress_dir.resolve()
        if (not progress_dir.is_dir() or progress_dir.is_relative_to(model)
                or model.is_relative_to(progress_dir)):
            raise ValueError("progress directory must exist outside the candidate model")
        if (not isinstance(full_verification_sha256, str)
                or len(full_verification_sha256) != 64):
            raise ValueError("resumable inference must bind a full BF16 verification digest")
        activation_identity = _activation_identity(
            model, prompt, ids, full_verification_sha256,
        )
        resumed_from, state = _recover_activation(
            progress_dir, activation_identity, (1, len(ids), tc.hidden_size),
        )
    reader = ShardReader(model)
    t0 = time.monotonic()
    torch.set_num_threads(2)
    with torch.inference_mode():
        if state is None:
            embed = reader.load(PREFIX + "embed_tokens.", "cpu")["weight"]
            x = F.embedding(token_ids, embed).unsqueeze(0)
            del embed
        else:
            x = state
        rotary = M.Qwen3_5MoeTextRotaryEmbedding(tc)
        pos = torch.arange(len(ids), dtype=torch.long).view(1, -1)
        position_embeddings = rotary(x, pos[None].expand(3, 1, -1))
        layer_types: list[str] = list(tc.layer_types[:resumed_from + 1])
        for layer_id in range(resumed_from + 1, 40):
            layer = _load_layer_without_duplicate_weights(
                M.Qwen3_5MoeDecoderLayer, tc, layer_id, torch.bfloat16,
                reader.load(PREFIX + f"layers.{layer_id}.", "cpu"),
            ).eval()
            if layer.block_type == "full_attention":
                mask = M.create_causal_mask(
                    config=tc, inputs_embeds=x, attention_mask=None,
                    past_key_values=None, position_ids=pos,
                )
            else:
                mask = M.create_recurrent_attention_mask(
                    config=tc, inputs_embeds=x, attention_mask=None,
                )
            y = layer(x, position_embeddings=position_embeddings,
                      attention_mask=mask, position_ids=pos)
            if not bool(torch.isfinite(y).all()):
                raise ValueError(f"non-finite actual pruned BF16 decoder output at layer {layer_id}")
            x = y
            layer_types.append(layer.block_type)
            del y, layer, mask
            gc.collect()
            if progress_dir is not None and layer_id in CHECKPOINT_LAYERS:
                _save_activation(progress_dir, layer_id, x, activation_identity)
            if layer_id in (0, 9, 19, 29, 39):
                print(json.dumps({"event": "STREAMING_LAYER_PASS",
                                  "layer": layer_id, "type": layer_types[-1]}), flush=True)
        norm = M.Qwen3_5MoeRMSNorm(tc.hidden_size, eps=tc.rms_norm_eps)
        norm.weight.data.copy_(reader.load(PREFIX + "norm.", "cpu")["weight"])
        head = reader.load("lm_head.", "cpu")["weight"]
        logits = F.linear(norm(x)[0, -1:], head).float().squeeze(0)
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("non-finite final real BF16 checkpoint logits")
        winner = int(torch.argmax(logits).item())
        strongest = torch.topk(logits, k=5).indices.tolist()
        logits_sha256 = hashlib.sha256(logits.contiguous().numpy().tobytes()).hexdigest()
    return {
        "schema": "chowder.reap128-v2-cpu-streaming-smoke.v1",
        "status": "CPU_BF16_END_TO_END_SMOKE_PASS_NOT_QUALITY_GATE",
        "checkpoint": str(model),
        "prompt": prompt,
        "input_tokens": ids,
        "layer_count": len(layer_types),
        "layer_types": {"linear_attention": layer_types.count("linear_attention"),
                        "full_attention": layer_types.count("full_attention")},
        "predicted_token_id": winner,
        "predicted_token_text": tokenizer.decode([winner]),
        "top_5_token_ids": strongest,
        "final_logits_sha256": logits_sha256,
        "elapsed_seconds": round(time.monotonic() - t0, 3),
        "resumed_after_layer": resumed_from,
        "activation_progress_dir": str(progress_dir) if progress_dir else None,
        "full_verification_sha256": full_verification_sha256,
        "verification_scope": "single bounded greedy next-token BF16 forward only",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt", default="The answer is")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--out", type=Path, required=True, help="new receipt, never overwrite")
    parser.add_argument("--progress-dir", type=Path,
                        help="resume from immutable BF16 activation checksums in this external dir")
    parser.add_argument("--full-verification-sha256",
                        help="required SHA-256 of complete 26-shard BF16 proof for resumable inference")
    args = parser.parse_args(argv)
    if args.out.exists() or not args.out.parent.is_dir():
        parser.error("receipt output must be new and parent must exist")
    try:
        result = run(
            args.model, prompt=args.prompt, max_tokens=args.max_tokens,
            progress_dir=args.progress_dir,
            full_verification_sha256=args.full_verification_sha256,
        )
        data = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
        with args.out.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(data)
        print(json.dumps({"status": result["status"],
                          "receipt": str(args.out), "token": result["predicted_token_id"]}))
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as exc:
        print(f"real BF16 streaming smoke refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
