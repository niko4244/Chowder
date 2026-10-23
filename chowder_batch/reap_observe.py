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
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

PREFIX = "model.language_model."
# Bound the final vocabulary projection, not the observed calibration census.
# Qwen's 248,320-word head otherwise allocates ~1.9 GiB for the fp32 logits
# of a single 2,048-token sequence, on top of its bf16 projection and weights.
NLL_CHUNK_POSITIONS = 64


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

    def forward(hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=experts.num_experts).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in hit:
            e = int(expert_idx[0])
            top_k_pos, token_idx = torch.where(mask[e])
            state = hidden_states[token_idx]
            gate, up = F.linear(state, experts.gate_up_proj[e]).chunk(2, dim=-1)
            out = F.linear(experts.act_fn(gate) * up, experts.down_proj[e])
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
    args = parser.parse_args()

    from transformers import AutoConfig, AutoTokenizer
    import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as M

    model_dir = Path(args.model_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = AutoConfig.from_pretrained(model_dir)
    tc = config.text_config
    tc._attn_implementation = "sdpa"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    reader = ShardReader(model_dir)
    if args.weights_dir:
        reader.dir = Path(args.weights_dir)
    dev = "cuda"
    dtype = torch.bfloat16

    rows = [json.loads(l) for l in Path(args.calibration).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit_seqs:
        rows = rows[: args.limit_seqs]
    seqs = [torch.tensor(tokenizer(r["text"], add_special_tokens=False)["input_ids"]) for r in rows]
    n_tokens = sum(len(s) for s in seqs)

    embed = reader.load(PREFIX + "embed_tokens.", dev)["weight"]
    hidden = [F.embedding(s.to(dev), embed).unsqueeze(0).to("cpu") for s in seqs]
    del embed
    rotary = M.Qwen3_5MoeTextRotaryEmbedding(tc).to(dev)

    if args.layers:
        lo, hi = (int(x) for x in args.layers.split("-"))
        layer_ids = list(range(lo, hi + 1))
    else:
        layer_ids = list(range(tc.num_hidden_layers))
    if layer_ids[0] != 0:
        raise SystemExit("observation must start at layer 0 (hidden states are computed, not loaded)")

    timings = []
    for i in layer_ids:
        t0 = time.time()
        with torch.device("meta"):
            layer = M.Qwen3_5MoeDecoderLayer(tc, i)
        layer = layer.to(dtype).to_empty(device=dev)
        layer.load_state_dict(reader.load(f"{PREFIX}layers.{i}.", dev), strict=True)
        layer.eval()
        E = tc.num_experts
        stats = {k: torch.zeros(E, dtype=torch.float64, device=dev) for k in ("count", "gate_sum", "norm_sum", "reap_sum")}
        layer.mlp.experts.forward = observing_experts_forward(layer.mlp.experts, stats)
        t_load = time.time() - t0
        with torch.inference_mode():
            for s, h in enumerate(hidden):
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
        torch.save({k: v.cpu() for k, v in stats.items()}, out / f"layer-{i:02d}.pt")
        del layer
        torch.cuda.empty_cache()
        dt = time.time() - t0
        timings.append({"layer": i, "block_type": tc.layer_types[i], "load_s": round(t_load, 1), "total_s": round(dt, 1),
                        "routed_tokens": int(stats["count"].sum().item()), "experts_hit": int((stats["count"] > 0).sum().item())})
        print(json.dumps(timings[-1]), flush=True)

    result = {"layers": timings, "n_seqs": len(seqs), "n_tokens": n_tokens,
              "calibration_sha256": hashlib.sha256(Path(args.calibration).read_bytes()).hexdigest()}
    if not args.layers:
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
    (out / "observation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
