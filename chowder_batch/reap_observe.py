"""Layer-by-layer REAP observation for Qwen3.6-35B-A3B on a single 16 GB card.

The 72 GB BF16 checkpoint never needs to be resident: every calibration
sequence's hidden state is held on CPU (~4 KB/token), and each decoder layer is
materialised on the GPU from its safetensors shards, run over every sequence,
observed, and freed. The forward replicates ``Qwen3_5MoeTextModel.forward``
(rotary over 3-axis text positions, causal mask for full attention, recurrent
mask for Gated DeltaNet) with the library's own layer class, so the arithmetic
is the reference implementation's -- only the scheduling differs.

Per routed expert j of every layer it accumulates, over the tokens routed to j:
  count_j, sum_j g_j(x), sum_j ||f_j(x)||, and sum_j g_j(x)*||f_j(x)||,
where g_j is the renormalised top-k router weight and f_j the expert output
before weighting. REAP saliency is  S_j = sum g*||f|| / count  (arXiv 2510.13999).

End-to-end check (gate D0): after the last layer the final norm and lm_head
score next-token NLL on the calibration text. A pipeline defect shows up as
an implausible perplexity.

Usage:
  python chowder_batch/reap_observe.py --model-dir <bf16 dir> \
      --calibration <calibration.jsonl> --out <dir> [--limit-seqs N] [--layers 0-3]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import weakref
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

PREFIX = "model.language_model."
# Bound the final vocabulary projection, not the observed calibration census.
# Qwen's 248,320-word head otherwise allocates ~1.9 GiB for the fp32 logits
# of a single 2,048-token sequence, on top of its bf16 projection and weights.
NLL_CHUNK_POSITIONS = 64


def _calibration_snapshot(calibration: Path) -> tuple[list[dict], str]:
    """Retain the exact bytes supplying an expensive registered observation.

    The prior observer read the rows before running but re-read the path *only
    after* 40 CUDA layers to record its SHA. A changed file could therefore
    falsely attribute the observed activations to later, different inputs.
    """
    raw = calibration.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    try:
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    except (UnicodeError, ValueError) as exc:
        raise ValueError("observer calibration JSONL is invalid") from exc
    if not rows or any(not isinstance(row, dict)
                       or not isinstance(row.get("text"), str)
                       or not row["text"]
                       or not isinstance(row.get("source"), str)
                       or not row["source"] for row in rows):
        raise ValueError("observer calibration contains invalid/empty sequence or source")
    return rows, sha


def _assert_calibration_unchanged(calibration: Path, witnessed_sha256: str) -> None:
    """Refuse new observation attribution if calibration bytes changed."""
    if hashlib.sha256(calibration.read_bytes()).hexdigest() != witnessed_sha256:
        raise RuntimeError("observer calibration bytes changed during measurement; "
                           "partial layer output is not qualified evidence")


def _require_frozen_d0_calibration(
    calibration_sha256: str, row_count: int, limit_seqs: int,
) -> None:
    from chowder_batch.reap_v2_source_identity import FROZEN_V2_BASE_SHA256
    if (limit_seqs or row_count != 704
            or calibration_sha256 != FROZEN_V2_BASE_SHA256):
        raise ValueError("full BF16 Gate D0 requires frozen untruncated 704-row calibration")


def _require_frozen_d0_model(tc, n_tokens: int) -> None:
    if (tc.num_hidden_layers != 40 or tc.hidden_size != 2048
            or tc.num_experts != 256 or tc.num_experts_per_tok != 8
            or n_tokens != 1_192_632):
        raise ValueError("full BF16 Gate D0 requires exact registered 704x40 model/token census")


def _prepare_observation_destination(
    out: Path, *, calibration: Path, model_dir: Path, weights_dir: Path,
) -> None:
    """Require a new disjoint experiment directory without erasing evidence."""
    intended = out.resolve()
    protected = [calibration.resolve(), model_dir.resolve(), weights_dir.resolve()]
    for item in protected:
        if intended == item or intended.is_relative_to(item) or item.is_relative_to(intended):
            raise ValueError("observer output overlaps immutable calibration or model inputs")
    if intended.exists():
        raise FileExistsError("observer output directory already exists; refusing overwrite/reuse")
    intended.mkdir(parents=True, exist_ok=False)


class ShardReader:
    def __init__(self, model_dir: Path) -> None:
        index = json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
        self.dir = model_dir
        self.weight_map: dict[str, str] = index["weight_map"]

    def load(self, prefix: str, device: str) -> dict[str, torch.Tensor]:
        by_file: dict[str, list[str]] = {}
        for key, shard in self.weight_map.items():
            if key.startswith(prefix):
                by_file.setdefault(shard, []).append(key)
        out: dict[str, torch.Tensor] = {}
        for shard, keys in sorted(by_file.items()):
            with safe_open(str(self.dir / shard), framework="pt", device="cpu") as handle:
                for key in keys:
                    out[key[len(prefix):]] = handle.get_tensor(key).to(device, non_blocking=True)
        if not out:
            raise KeyError(f"no tensors under {prefix!r}")
        return out


def observing_experts_forward(experts, stats: dict[str, torch.Tensor]):
    """The library's Qwen3_5MoeExperts.forward, plus REAP accumulation."""
    # This closure is installed as experts.forward. Capturing the module itself
    # would form a reference cycle, retaining a CUDA layer after ``del layer``.
    expert_ref = weakref.ref(experts)

    def forward(hidden_states, top_k_index, top_k_weights):
        module = expert_ref()
        if module is None:
            raise RuntimeError("observer expert layer was released")
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=module.num_experts).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in hit:
            e = int(expert_idx[0])
            top_k_pos, token_idx = torch.where(mask[e])
            state = hidden_states[token_idx]
            gate, up = F.linear(state, module.gate_up_proj[e]).chunk(2, dim=-1)
            out = F.linear(module.act_fn(gate) * up, module.down_proj[e])
            weight = top_k_weights[token_idx, top_k_pos, None]
            norm = out.float().norm(dim=-1)
            g = weight.float().squeeze(-1)
            stats["count"][e] += token_idx.numel()
            stats["gate_sum"][e] += g.sum()
            stats["norm_sum"][e] += norm.sum()
            stats["reap_sum"][e] += (g * norm).sum()
            final.index_add_(0, token_idx, (out * weight).to(final.dtype))
        return final

    return forward


def _load_layer_without_duplicate_weights(layer_class, config, layer_id, dtype, state):
    """Assign checkpoint storage to a meta layer, avoiding a second GPU copy.

    Regular ``to_empty(device='cuda')`` followed by ``load_state_dict`` holds
    both the allocated destination parameters and the full CUDA checkpoint
    state during loading. ``assign=True`` preserves the checkpoint tensors.
    """
    with torch.device("meta"):
        layer = layer_class(config, layer_id)
    layer = layer.to(dtype)
    expected = layer.state_dict()
    for name, tensor in state.items():
        if name in expected and tensor.dtype != expected[name].dtype:
            raise TypeError(f"observer layer {layer_id} checkpoint dtype mismatch: {name}")
    layer.load_state_dict(state, strict=True, assign=True)
    if any(t.is_meta for t in (*layer.parameters(), *layer.buffers())):
        raise RuntimeError(f"observer layer {layer_id} contains unmaterialized tensors")
    return layer


@torch.inference_mode()
def _next_token_nll_chunked(
    norm: torch.nn.Module,
    hidden_states: torch.Tensor,
    tokens: torch.Tensor,
    lm_head: torch.Tensor,
    *,
    chunk_positions: int = NLL_CHUNK_POSITIONS,
) -> tuple[float, int]:
    """Same next-token cross-entropy as the full projection, in position chunks.

    Only positions [0, length-1) predict labels [1, length). Normalization is
    token-local; slicing before RMSNorm and F.linear therefore leaves the
    frozen loss definition unchanged. Each float32 vocabulary-logit matrix is
    at most chunk_positions x vocab_size, rather than length x vocab_size.
    """
    if type(chunk_positions) is not int or chunk_positions <= 0:
        raise ValueError("NLL chunk_positions must be a positive integer")
    if (hidden_states.ndim != 3 or hidden_states.shape[0] != 1
            or tokens.ndim != 1 or hidden_states.shape[1] != tokens.numel()):
        raise ValueError("NLL requires one hidden-state sequence and aligned token ids")
    positions = tokens.numel() - 1
    if positions <= 0:
        return 0.0, 0
    total = torch.zeros((), dtype=torch.float64, device=lm_head.device)
    for start in range(0, positions, chunk_positions):
        stop = min(start + chunk_positions, positions)
        # The observer retains hidden states on the CPU. Never move the full
        # sequence to CUDA while projecting through the large vocabulary head.
        states = hidden_states[:, start:stop].to(device=lm_head.device, dtype=lm_head.dtype)
        logits = F.linear(norm(states)[0], lm_head).float()
        targets = tokens[start + 1:stop + 1].to(device=lm_head.device)
        total += F.cross_entropy(logits, targets, reduction="sum").double()
    return total.item(), positions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="config, tokenizer and safetensors index")
    parser.add_argument("--weights-dir", default="", help="where the shards live (default: --model-dir)")
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit-seqs", type=int, default=0)
    parser.add_argument("--layers", default="", help="e.g. 0-3 for a smoke run (skips the NLL check)")
    parser.add_argument(
        "--hidden-disk-dir", default="",
        help="NEW scratch directory for atomic, two-buffer BF16 checkpoints",
    )
    parser.add_argument(
        "--resume-hidden-disk-dir", default="",
        help="resume a committed two-buffer BF16 checkpoint (same original --out)",
    )
    args = parser.parse_args()

    from transformers import AutoConfig, AutoTokenizer
    import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as M

    model_dir = Path(args.model_dir)
    out = Path(args.out)
    calibration = Path(args.calibration)
    rows, calibration_sha256 = _calibration_snapshot(calibration)
    # A partial smoke test may exercise arbitrary inputs, but a full registered
    # D0 checkpoint must never silently substitute a truncated calibration.
    # The actual token count is independently checked after HF tokenization.
    checkpoint_dir = args.hidden_disk_dir or args.resume_hidden_disk_dir
    if checkpoint_dir and not args.layers:
        _require_frozen_d0_calibration(calibration_sha256, len(rows), args.limit_seqs)
        from chowder_batch.reap_original_bf16_pin import validate_registered_source_metadata
        validate_registered_source_metadata(
            model_dir, Path(args.weights_dir or args.model_dir),
        )
    if args.hidden_disk_dir and args.resume_hidden_disk_dir:
        raise ValueError("choose either NEW checkpoint or resume, not both")
    resuming = bool(args.resume_hidden_disk_dir)
    if checkpoint_dir:
        scratch = Path(checkpoint_dir)
        candidate = scratch.resolve()
        for protected in (calibration, model_dir, Path(args.weights_dir or args.model_dir), out):
            existing = protected.resolve()
            if (candidate == existing or candidate.is_relative_to(existing)
                    or existing.is_relative_to(candidate)):
                raise ValueError("hidden scratch overlaps source or output evidence")
        if scratch.exists() and not resuming:
            raise FileExistsError("hidden scratch already exists; refusing reuse")
        if resuming and not scratch.is_dir():
            raise FileNotFoundError("checkpoint scratch directory does not exist")
    if resuming:
        if not out.is_dir() or (out / "observation.json").exists():
            raise FileExistsError("resume requires existing unfinished observer output")
    else:
        _prepare_observation_destination(
            out, calibration=calibration, model_dir=model_dir,
            weights_dir=Path(args.weights_dir or args.model_dir),
        )
    config = AutoConfig.from_pretrained(model_dir)
    tc = config.text_config
    tc._attn_implementation = "sdpa"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    reader = ShardReader(model_dir)
    if args.weights_dir:
        reader.dir = Path(args.weights_dir)
    dev = "cuda"
    dtype = torch.bfloat16

    if args.limit_seqs:
        rows = rows[: args.limit_seqs]
    seqs = [torch.tensor(tokenizer(r["text"], add_special_tokens=False)["input_ids"]) for r in rows]
    n_tokens = sum(len(s) for s in seqs)
    if checkpoint_dir and not args.layers:
        _require_frozen_d0_model(tc, n_tokens)

    disk_hidden = None
    if checkpoint_dir:
        # When invoked as an ordinary script the sibling module is on sys.path.
        # Full BF16 payload hashing deliberately runs before any model load.
        from reap_hidden_checkpoint import BF16CheckpointHidden, fingerprint_inputs

        identity = fingerprint_inputs(
            model_dir=model_dir, weights_dir=reader.dir,
            calibration_sha256=calibration_sha256, seqs=seqs,
            layer_ids=(list(range(tc.num_hidden_layers)) if not args.layers
                       else list(range(int(args.layers.split("-")[0]),
                                       int(args.layers.split("-")[1]) + 1))),
            out=out,
            num_experts=tc.num_experts, top_k=tc.num_experts_per_tok,
            require_registered_original=not bool(args.layers),
        )
        disk_hidden = BF16CheckpointHidden(
            [len(s) for s in seqs], tc.hidden_size, Path(checkpoint_dir),
            identity, resume=resuming,
        )
    hidden = [] if disk_hidden is None else disk_hidden
    if disk_hidden is None or disk_hidden.generation == -1:
        embed = reader.load(PREFIX + "embed_tokens.", dev)["weight"]
        for index, tokens in enumerate(seqs):
            row_hidden = F.embedding(tokens.to(dev), embed).unsqueeze(0).to("cpu")
            if disk_hidden is None:
                hidden.append(row_hidden)
            else:
                hidden[index] = row_hidden
            del row_hidden
        del embed
        if disk_hidden is not None:
            disk_hidden.finish_embeddings()
    rotary = M.Qwen3_5MoeTextRotaryEmbedding(tc).to(dev)

    if args.layers:
        lo, hi = (int(x) for x in args.layers.split("-"))
        layer_ids = list(range(lo, hi + 1))
    else:
        layer_ids = list(range(tc.num_hidden_layers))
    if layer_ids[0] != 0:
        raise SystemExit("observation must start at layer 0 (hidden states are computed, not loaded)")

    if disk_hidden is not None and identity["layers"] != layer_ids:
        raise RuntimeError("checkpoint layer selection changed")
    timings = [] if disk_hidden is None else disk_hidden.timings
    if disk_hidden is not None:
        if disk_hidden.generation > len(layer_ids):
            raise RuntimeError("checkpoint generation beyond expected layer range")
        print(json.dumps({
            "bf16_checkpoint": ("resumed" if resuming else "new"),
            "committed_generation": disk_hidden.generation,
            "next_layer": disk_hidden.generation,
            "gate_d0": "NOT_MEASURED",
        }), flush=True)
        for committed_layer in range(disk_hidden.generation):
            disk_hidden.mirror_layer_stats(out, committed_layer)
    for i in layer_ids[0 if disk_hidden is None else disk_hidden.generation:]:
        t0 = time.time()
        if disk_hidden is not None:
            disk_hidden.begin_layer(i)
        torch.cuda.reset_peak_memory_stats()
        layer = _load_layer_without_duplicate_weights(
            M.Qwen3_5MoeDecoderLayer, tc, i, dtype,
            reader.load(f"{PREFIX}layers.{i}.", dev),
        )
        layer.eval()
        E = tc.num_experts
        stats = {k: torch.zeros(E, dtype=torch.float64, device=dev) for k in ("count", "gate_sum", "norm_sum", "reap_sum")}
        layer.mlp.experts.forward = observing_experts_forward(layer.mlp.experts, stats)
        t_load = time.time() - t0
        with torch.inference_mode():
            for s in range(len(seqs)):
                h = hidden[s]
                x = h.to(dev, dtype)
                L = x.shape[1]
                pos = torch.arange(L, device=dev).view(1, -1)
                pe = rotary(x, pos[None].expand(3, 1, -1))
                if layer.block_type == "full_attention":
                    mask = M.create_causal_mask(config=tc, inputs_embeds=x, attention_mask=None,
                                                past_key_values=None, position_ids=pos)
                else:
                    mask = M.create_recurrent_attention_mask(config=tc, inputs_embeds=x, attention_mask=None)
                y = layer(x, position_embeddings=pe, attention_mask=mask, position_ids=pos)
                if not torch.isfinite(y).all():
                    raise SystemExit(f"non-finite hidden state at layer {i}, sequence {s}")
                hidden[s] = y.to("cpu")
                # The last sequence's CUDA x/y/PE/attention tensors must not
                # survive into the next decoder's checkpoint allocation.
                # The former loop retained those bindings across del layer
                # and empty_cache, increasing peak GPU memory unnecessarily.
                del x, y, pos, pe, mask, h
        _assert_calibration_unchanged(calibration, calibration_sha256)
        routed = int(stats["count"].sum().item())
        if routed != n_tokens * tc.num_experts_per_tok:
            raise RuntimeError(f"layer {i} routed-token census incomplete ({routed})")
        experts_hit = int((stats["count"] > 0).sum().item())
        del layer
        torch.cuda.empty_cache()
        dt = time.time() - t0
        global_free, _global_total = torch.cuda.mem_get_info()
        timing = {"layer": i, "block_type": tc.layer_types[i], "load_s": round(t_load, 1), "total_s": round(dt, 1),
                        "routed_tokens": routed, "experts_hit": experts_hit,
                        "torch_cuda_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
                        "torch_cuda_allocated_after_mib": round(torch.cuda.memory_allocated() / 2**20, 1),
                        "torch_cuda_reserved_after_mib": round(torch.cuda.memory_reserved() / 2**20, 1),
                        "global_cuda_free_after_mib": round(global_free / 2**20, 1)}
        stats_cpu = {k: v.cpu() for k, v in stats.items()}
        if disk_hidden is not None:
            disk_hidden.finish_layer(i, stats_cpu, timing)
            disk_hidden.mirror_layer_stats(out, i)
        else:
            torch.save(stats_cpu, out / f"layer-{i:02d}.pt")
            timings.append(timing)
        print(json.dumps(timings[-1]), flush=True)

    checkpoint_receipt = disk_hidden.sealed_receipt() if disk_hidden is not None else None
    result = {"layers": timings, "n_seqs": len(seqs), "n_tokens": n_tokens,
              "calibration_sha256": calibration_sha256}
    if disk_hidden is not None:
        result["hidden_storage"] = {
            "type": "dual file-backed CPU BF16 bits, atomic layer checkpoint",
            "bytes": 2 * disk_hidden.required_bytes,
            "last_committed_generation": disk_hidden.generation,
            "numeric_conversions": False,
        }
        result["checkpoint_receipt"] = checkpoint_receipt
    if args.layers:
        result["gate_d0"] = "NOT_MEASURED (explicit partial-layer smoke)"
    if not args.layers:
        if disk_hidden is not None:
            if disk_hidden.generation != tc.num_hidden_layers:
                raise RuntimeError("NLL requires all frozen layers committed")
            disk_hidden.begin_final_read()
        norm_w = reader.load(PREFIX + "norm.", dev)["weight"]
        norm = M.Qwen3_5MoeRMSNorm(tc.hidden_size, eps=tc.rms_norm_eps).to(dev)
        norm.weight.data.copy_(norm_w)
        lm_head = reader.load("lm_head.", dev)["weight"]
        nll_sum, count = 0.0, 0
        per_source: dict[str, list[float]] = {}
        with torch.inference_mode():
            for row, s, h in zip(rows, seqs, hidden):
                nll, predicted = _next_token_nll_chunked(norm, h, s, lm_head)
                nll_sum += nll
                count += predicted
                per_source.setdefault(row["source"], [0.0, 0])
                per_source[row["source"]][0] += nll
                per_source[row["source"]][1] += predicted
        result["calibration_ppl"] = math.exp(nll_sum / count)
        result["calibration_ppl_by_source"] = {k: math.exp(v[0] / v[1]) for k, v in per_source.items()}
        print("calibration perplexity", result["calibration_ppl"], result["calibration_ppl_by_source"], flush=True)
    _assert_calibration_unchanged(calibration, calibration_sha256)
    if disk_hidden is not None:
        if disk_hidden.sealed_receipt() != checkpoint_receipt:
            raise RuntimeError("BF16 checkpoint changed between final NLL and publication")
        # Full payload recheck before a completed result is attributed to the
        # original model. Resume separately performs the same preflight.
        if fingerprint_inputs(
            model_dir=model_dir, weights_dir=reader.dir,
            calibration_sha256=calibration_sha256, seqs=seqs,
            layer_ids=layer_ids, out=out,
            num_experts=tc.num_experts, top_k=tc.num_experts_per_tok,
            require_registered_original=not bool(args.layers),
        ) != identity:
            raise RuntimeError("frozen checkpoint/model/runtime identity changed")
        from reap_hidden_checkpoint import _atomic_bytes
        _atomic_bytes(out / "observation.json",
                      (json.dumps(result, indent=2, allow_nan=False) + "\n").encode())
        disk_hidden.close()
    else:
        (out / "observation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
