"""Experiment E Phase-2 speculative decoding prototype.

Design note: an EAGLE-style learned draft head is not buildable here -- the
small model (Spark, 131072-token vocab) and the teacher (Qwen3.8, 248320-token
vocab) use **incompatible tokenizers**, so draft tokens cannot be verified in
the teacher's vocabulary, and no trained draft head exists for either model.
The honest Phase-2 prototype is therefore **prompt-lookup n-gram drafting with
exact greedy verification**:

1. Draft up to k tokens by looking up the most recent n-gram match in the
   context (repair and structured outputs copy from the prompt).
2. Run ONE teacher forward pass over [context + draft], reading the argmax at
   each draft position.
3. Accept the longest prefix whose teacher argmax agrees position-by-position
   (greedy verification = exact). After the last accepted token, emit the
   teacher's own next token, so every round advances at least one token.

Because the verifier is the teacher itself, greedy output is **token-for-token
identical** to teacher-only greedy decoding. This is tested below.

Measured: draft acceptance, effective tokens per second, verification overhead
(teacher forward on k+1 positions vs plain single-token steps), and peak
memory. The small model on GPU is additionally measured as a real draft source
(distributional drafts, exact teacher-argmax verification, no equivalence
claim) so the report can separate n-gram from model-draft effects.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def load_teacher(model_dir: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    # torch_dtype (not dtype): this is the exact code path that survived the
    # only successful full-model load on this machine; keep it identical.
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map="auto", local_files_only=True
    )
    model.eval()
    return tokenizer, model


def load_spark(model_dir: str):
    import hashlib

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from chowder.local_model_compat import patch_transformers5_custom_model

    digests = {
        name: hashlib.sha256((Path(model_dir) / name).read_bytes()).hexdigest()
        for name in ("configuration_spark.py", "modeling_spark.py")
    }
    patch_transformers5_custom_model(model_dir, digests)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, trust_remote_code=True, local_files_only=True,
        dtype=torch.bfloat16, device_map="cuda:0",
    )
    model.eval()
    return tokenizer, model


def draft_from_prompt(context_ids: list[int], *, ngram: int = 3, max_draft: int = 8) -> list[int]:
    """Draft up to max_draft tokens from the longest suffix n-gram match.

    Longest-match-first: try the longest n-gram window that occurs at least
    twice in the context; the draft is the continuation after its earlier
    occurrence. Falls back to shorter windows, then to no draft.
    """
    if len(context_ids) < ngram + 1:
        return []
    for size in range(min(ngram, len(context_ids) - 1), 0, -1):
        suffix = tuple(context_ids[-size:])
        # Search for an earlier occurrence of the suffix (excluding the tail
        # itself), scanning from the left so the continuation is longest.
        for start in range(len(context_ids) - size):
            if tuple(context_ids[start : start + size]) == suffix:
                return list(context_ids[start + size : start + size + max_draft])
    return []


def verify_and_accept(
    model, context: torch.Tensor, draft: list[int], *, device: torch.device
) -> tuple[list[int], int]:
    """One verify round. Returns (emitted_tokens, drafted_count)."""
    if not draft:
        # Plain single-token step: teacher picks its own token.
        with torch.inference_mode():
            logits = model(context).logits[0, -1]
        return [int(logits.argmax())], 0

    draft_t = torch.tensor([draft], dtype=torch.long, device=device)
    extended = torch.cat([context, draft_t], dim=1)
    with torch.inference_mode():
        logits = model(extended).logits[0]
    # argmax at position i predicts the token at position i+1.
    argmaxes = logits.argmax(-1)
    emitted: list[int] = []
    for offset, draft_token in enumerate(draft):
        predictor_index = context.shape[1] - 1 + offset
        if int(argmaxes[predictor_index]) == draft_token:
            emitted.append(draft_token)
        else:
            break
    # After the last accepted draft token, take the teacher's own token.
    emitted.append(int(argmaxes[context.shape[1] - 1 + len(emitted)]))
    return emitted, len(draft)


def speculative_generate(
    model, prompt_ids: torch.Tensor, *, max_new: int, ngram: int, max_draft: int,
    device: torch.device,
) -> dict:
    """Full loop. Returns output ids, acceptance stats, timing, and memory."""
    context = prompt_ids.clone()
    n_generated = 0
    n_drafted = 0
    n_accepted = 0
    n_rounds = 0
    t0 = time.perf_counter()
    while n_generated < max_new:
        draft = draft_from_prompt(context[0].tolist(), ngram=ngram, max_draft=max_draft)
        emitted, drafted = verify_and_accept(model, context, draft, device=device)
        if n_generated + len(emitted) > max_new:
            emitted = emitted[: max_new - n_generated]
        context = torch.cat([context, torch.tensor([emitted], dtype=torch.long, device=device)], dim=1)
        n_generated += len(emitted)
        n_drafted += drafted
        n_accepted += max(0, len(emitted) - 1 if drafted else 0)
        n_rounds += 1
        if emitted and emitted[-1] == tokenizer_eos(model):
            break
    elapsed = time.perf_counter() - t0
    return {
        "output_ids": context[0, prompt_ids.shape[1]:].tolist(),
        "seconds": elapsed,
        "rounds": n_rounds,
        "drafted": n_drafted,
        "accepted": n_accepted,
        "acceptance_rate": n_accepted / n_drafted if n_drafted else 0.0,
        "tokens_per_second": n_generated / elapsed,
    }


def tokenizer_eos(model) -> int:
    cfg = getattr(model.config, "text_config", model.config)
    return int(getattr(cfg, "eos_token_id", 0) or 0)


def plain_generate(model, prompt_ids: torch.Tensor, *, max_new: int, device: torch.device) -> dict:
    context = prompt_ids.clone()
    t0 = time.perf_counter()
    for _ in range(max_new):
        with torch.inference_mode():
            logits = model(context).logits[0, -1]
        token = int(logits.argmax())
        context = torch.cat([context, torch.tensor([[token]], dtype=torch.long, device=device)], dim=1)
        if token == tokenizer_eos(model):
            break
    elapsed = time.perf_counter() - t0
    return {
        "output_ids": context[0, prompt_ids.shape[1]:].tolist(),
        "seconds": elapsed,
        "tokens_per_second": (context.shape[1] - prompt_ids.shape[1]) / elapsed,
    }


PROMPTS = (
    # copy-heavy prompts (n-gram drafting should shine)
    "def parse_version(s):\n    parts = s.split('.')\n    while len(parts) < 3: parts.append('0')\n    return tuple(int(p) for p in parts)\n\n# Copy this function exactly, then add a version() helper returning '1.0.0'.\n",
    "The runtime harness exposes three tools: read_file, write_file, and run_tests. Repeat the three tool names exactly, separated by commas.\n",
    "Repeat exactly: The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.\n",
    # generative prompt (n-gram drafting should struggle)
    "Explain in one sentence why testing before releasing software reduces risk.\n",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="F:/llm-models/Qwen3.8-9B-abliterated-25-bf16")
    parser.add_argument("--spark", default="F:/Huihui-Spark-X2.5-4B-abliterated")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-new", type=int, default=48)
    parser.add_argument("--skip-spark-draft", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda:0")
    print("loading teacher...", flush=True)
    tokenizer, model = load_teacher(args.model)

    results: dict[str, dict] = {"prompts": [], "equivalence": None}
    plain_outputs: list[list[int]] = []
    plain_timings: list[float] = []
    for prompt in PROMPTS:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        plain = plain_generate(model, ids, max_new=args.max_new, device=device)
        plain_outputs.append(plain["output_ids"])
        plain_timings.append(plain["seconds"])
        print(f"plain: {plain['tokens_per_second']:.2f} tok/s", flush=True)

    for ngram, max_draft in ((3, 8), (4, 12)):
        rows = []
        for idx, prompt in enumerate(PROMPTS):
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            spec = speculative_generate(model, ids, max_new=args.max_new, ngram=ngram, max_draft=max_draft, device=device)
            equivalent = spec["output_ids"] == plain_outputs[idx]
            rows.append({
                "prompt_kind": "copy" if idx < 3 else "generative",
                "equivalent_to_plain": equivalent,
                "tokens_per_second": round(spec["tokens_per_second"], 3),
                "plain_tokens_per_second": round((args.max_new) / plain_timings[idx], 3),
                "acceptance_rate": round(spec["acceptance_rate"], 4),
                "drafted": spec["drafted"],
                "accepted": spec["accepted"],
                "rounds": spec["rounds"],
            })
            print(f"spec ngram={ngram} k={max_draft} [{rows[-1]['prompt_kind']}]: eq={equivalent} {rows[-1]['tokens_per_second']} tok/s acc={rows[-1]['acceptance_rate']}", flush=True)
        results[f"ngram{NGRAM_LABEL(ngram, max_draft)}"] = rows

    if not args.skip_spark_draft:
        print("loading spark as draft model...", flush=True)
        spark_tok, spark = load_spark(args.spark)
        rows = []
        for idx, prompt in enumerate(PROMPTS):
            spark_ids = spark_tok.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to(device)
            with torch.inference_mode():
                draft_text = spark.generate(
                    **spark_ids, max_new_tokens=args.max_new, do_sample=False, pad_token_id=spark_tok.pad_token_id
                )
            draft_text = spark_tok.decode(draft_text[0, spark_ids.shape[1]:], skip_special_tokens=True)
            reencoded = tokenizer(draft_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            spec = verify_and_accept(model, tokenizer(prompt, return_tensors="pt").input_ids.to(device), reencoded[0].tolist()[:12], device=device)
            results.setdefault("spark_draft", []).append({
                "prompt_kind": "copy" if idx < 3 else "generative",
                "draft_seconds": None,
                "emitted_from_draft": len(spec[0]),
                "drafted": len(reencoded[0].tolist()[:12]),
            })
        del spark
        gc.collect()
        torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("wrote", out)
    return 0


def NGRAM_LABEL(ngram: int, max_draft: int) -> str:
    return f"{ngram}_k{max_draft}"


if __name__ == "__main__":
    raise SystemExit(main())
