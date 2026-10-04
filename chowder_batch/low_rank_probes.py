"""Phase-3 matrix-level quality probes for the factorized vocabulary matrices.

For each persisted rank this script measures, on the real matrices:

* per-token-row relative reconstruction error, bucketed by token class
  (digits, ASCII words, CJK, whitespace/punct, rare unicode);
* logit-level divergence using *real* embedding rows as hidden states (at the
  first decoder position the residual stream equals the embedding row, so this
  is a faithful first-order probe of output-projection damage);
* top-1 agreement, top-5 recall, and KL divergence between original and
  factorized next-token distributions;
* head-projection latency at batch 1 (prefill is where the lm_head FLOPs land).

Text-level KL with full hidden states requires the teacher pass and is tracked
separately (Phase 4); these probes are matrix-level by construction.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chowder.low_rank_checkpoint import read_tensor  # noqa: E402

RANKS = (2048, 1536, 1024, 512)


def classify_tokens(tokenizer) -> dict[str, torch.Tensor]:
    """Bucket every vocabulary ID into a token class."""
    vocab_size = len(tokenizer)
    classes: dict[str, list[int]] = {"digits": [], "ascii_word": [], "cjk": [], "space_punct": [], "rare_unicode": []}
    for token_id in range(vocab_size):
        text = tokenizer.convert_ids_to_tokens(token_id)
        if text is None:
            classes["rare_unicode"].append(token_id)
            continue
        if any(ch.isdigit() for ch in text):
            classes["digits"].append(token_id)
        elif all(("a" <= ch <= "z") or ("A" <= ch <= "Z") for ch in text) and text:
            classes["ascii_word"].append(token_id)
        elif any("\u4e00" <= ch <= "\u9fff" for ch in text):
            classes["cjk"].append(token_id)
        elif all(not ch.isalpha() for ch in text):
            classes["space_punct"].append(token_id)
        else:
            classes["rare_unicode"].append(token_id)
    return {name: torch.tensor(ids, dtype=torch.long) for name, ids in classes.items()}


def row_errors(weight: torch.Tensor, u: torch.Tensor, v: torch.Tensor, chunk: int = 32768) -> torch.Tensor:
    """Per-row relative error ||W_i - (UV)_i|| / ||W_i||, computed in chunks."""
    errors = torch.empty(weight.shape[0], dtype=torch.float32)
    for start in range(0, weight.shape[0], chunk):
        stop = min(start + chunk, weight.shape[0])
        w = weight[start:stop]
        approx = u[start:stop] @ v
        errors[start:stop] = (w - approx).norm(dim=1) / w.norm(dim=1).clamp_min(1e-12)
    return errors


def logit_probe(head: torch.Tensor, u: torch.Tensor, v: torch.Tensor, hidden_rows: torch.Tensor, *, temperature: float = 1.0, chunk_rows: int = 512) -> dict[str, float]:
    """Compare logits from the original and factorized head on given rows.

    Chunked over rows: a full-vocab softmax on 248320 tokens is ~1 GB per
    fp32 tensor per 1024 rows, so unchunked runs overflow an 16 GB card.
    """
    kl_parts: list[torch.Tensor] = []
    correct = 0
    recall_sum = 0.0
    max_delta = 0.0
    total = hidden_rows.shape[0]
    with torch.no_grad():
        for start in range(0, total, chunk_rows):
            rows = hidden_rows[start : start + chunk_rows]
            original = rows @ head.T
            approx = (rows @ v.T) @ u.T
            p = torch.softmax(original / temperature, dim=-1)
            q = torch.softmax(approx / temperature, dim=-1)
            kl_parts.append(torch.xlogy(p, p / q.clamp_min(1e-12)).sum(-1))
            correct += float((original.argmax(-1) == approx.argmax(-1)).sum())
            top5_orig = original.topk(5, dim=-1).indices.tolist()
            top5_approx = approx.topk(5, dim=-1).indices.tolist()
            recall_sum += sum(len(set(a) & set(b)) / 5.0 for a, b in zip(top5_orig, top5_approx))
            max_delta = max(max_delta, float((original - approx).abs().max()))
            del original, approx, p, q
    kl = torch.cat(kl_parts)
    return {
        "kl_mean": float(kl.mean()),
        "kl_p95": float(kl.quantile(0.95)),
        "top1_agreement": correct / total,
        "top5_recall": recall_sum / total,
        "max_abs_logit_delta": max_delta,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--factors", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("loading real matrices once...", flush=True)
    embed = read_tensor(args.model, "model.language_model.embed_tokens.weight").to(torch.float32)
    head = read_tensor(args.model, "lm_head.weight").to(torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    classes = classify_tokens(tokenizer)
    print({name: int(ids.numel()) for name, ids in classes.items()}, flush=True)

    # Hidden-state surrogate: real embedding rows (frequent ASCII words, digits,
    # and code-ish punctuation) stand in for the residual stream.
    sample_ids = torch.cat([
        classes["ascii_word"][:4096],
        classes["digits"][:1024],
        classes["space_punct"][:1024],
    ])
    hidden_rows = embed[sample_ids].to(device)

    report: dict[str, dict] = {"model": args.model, "device": str(device), "ranks": {}}
    head_gpu = head.to(device)
    for rank in RANKS:
        path = Path(args.factors) / f"lm_head_rank{rank}_factors.safetensors"
        with safe_open(str(path), framework="pt") as handle:
            u = handle.get_tensor("u").to(torch.float32).to(device)
            v = handle.get_tensor("v").to(torch.float32).to(device)
        entry: dict[str, object] = {}

        # Per-class row error of the head matrix.
        row_err = row_errors(head_gpu, u, v)
        entry["row_error_overall"] = float(row_err.mean())
        entry["row_error_by_class"] = {
            name: float(row_err[ids].mean()) for name, ids in classes.items() if ids.numel()
        }
        entry["row_error_p95"] = float(row_err.quantile(0.95))

        # Logit divergence on real embedding rows.
        entry["logit_probe"] = logit_probe(head_gpu, u, v, hidden_rows)

        # Timing: prefill-style full-vocab projection, batch 1, fp32 GPU.
        one_row = hidden_rows[:1]
        for _ in range(2):
            one_row @ head_gpu.T
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            one_row @ head_gpu.T
        if device.type == "cuda":
            torch.cuda.synchronize()
        full_ms = (time.time() - t0) * 100
        t0 = time.time()
        for _ in range(10):
            (one_row @ v.T) @ u.T
        if device.type == "cuda":
            torch.cuda.synchronize()
        fact_ms = (time.time() - t0) * 100
        entry["head_latency_ms_full"] = round(full_ms, 3)
        entry["head_latency_ms_rank"] = round(fact_ms, 3)
        report["ranks"][str(rank)] = entry
        print(f"rank {rank}: row_err={entry['row_error_overall']:.4f} top1={entry['logit_probe']['top1_agreement']:.3f} kl={entry['logit_probe']['kl_mean']:.4f}", flush=True)
        del u, v, row_err
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
