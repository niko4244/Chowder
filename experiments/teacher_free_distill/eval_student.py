"""Bounded student evaluation for the teacher-free pilot (Condition A).

Measures ONE model per invocation — the untouched base, or the base plus a
trained LoRA adapter — on three axes that answer different questions:

  * ``gsm8k``      — held-out grade-school math generation (never in training
                     data): a forgetting probe for reasoning ability.
  * ``mmlu``       — held-out general knowledge, scored by log-likelihood over
                     the four options: a second, non-math forgetting axis.
  * ``dev_ppl``    — completion-only perplexity on pilot_v3's dev split, which
                     the adapter was NEVER trained on: the learning signal.

Both models are scored on the SAME fixed item slices (ids are recorded), so the
result is a paired comparison, not two independent numbers. Every run records
its provenance: base revision, adapter digest, dataset revisions, device,
contention snapshot, package versions, and wall time. Nothing is interpolated
and no metric is reported for a slice that failed to load — failures are
recorded as failures.

Usage:
  python eval_student.py --label base --out eval_base.json
  python eval_student.py --label cond_a --adapter C:/.../cond_a/adapter --out eval_cond_a.json
  python eval_student.py --compare eval_base.json eval_cond_a.json
  python eval_student.py --label base --only mmlu --mmlu-style lead_in --out eval_base_mmlu.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_pilot  # noqa: E402  (exclusivity policy + student alias table)

#: Evaluation floor: a 1.7B bf16 model is ~3.4 GB of frozen weights plus KV
#: cache for short greedy generations. This is deliberately NOT the 8 GB
#: training floor -- evaluation loads no optimizer state and trains nothing.
EVAL_MIN_FREE_VRAM_GB = 5.0
BASE_REPO = "Qwen/Qwen3-1.7B"
BASE_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DEV_ROWS = Path(r"C:\Users\nikma\chowder_teacher_free\pilot_v3\dev.jsonl")
CHOICES = ["A", "B", "C", "D"]
#: Text appended before the option letters are scored (see run_mmlu).
MMLU_LEAD_IN = "The correct option is"


def _load_evaluate():
    spec = importlib.util.spec_from_file_location("tfd_evaluate", HERE / "evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


evaluate = _load_evaluate()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def paired_table(base_items: dict[str, bool], cand_items: dict[str, bool]) -> dict:
    """Paired per-item outcome table over the shared slice (McNemar-style)."""
    shared = sorted(set(base_items) & set(cand_items))
    both = sum(1 for k in shared if base_items[k] and cand_items[k])
    base_only = sum(1 for k in shared if base_items[k] and not cand_items[k])
    cand_only = sum(1 for k in shared if cand_items[k] and not base_items[k])
    neither = sum(1 for k in shared if not base_items[k] and not cand_items[k])
    return {"items": len(shared), "both_correct": both, "base_only": base_only,
            "candidate_only": cand_only, "neither": neither,
            "base_accuracy": (both + base_only) / len(shared) if shared else None,
            "candidate_accuracy": (both + cand_only) / len(shared) if shared else None}


def render(tokenizer, messages: list[dict], *, add_generation_prompt: bool,
           enable_thinking: bool = False) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
    if enable_thinking is False:
        kwargs["enable_thinking"] = False
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:  # template without thinking support
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


class StopAtAnswer:
    """Stop once the answer is COMPLETE (lm-eval-harness ``until``, but safe).

    Condition A was trained on long reasoning traces, so greedy decoding can
    run to the token cap on every item; without a stop rule the same slice can
    take hours. Two traps are avoided here:

      * stopping AT the marker truncates the number (\"\\boxed{\" with no
        digits) -- the first version of this class scored 0/3 that way;
      * stopping at the first digit of a multi-digit number reports a
        truncated answer (\"#### 1\" for 18).

    So a braced answer (``\\boxed{18}``) must be closed, and a bare
    ``#### 18`` must be followed by a character that ends the number. The
    token cap remains the ceiling. Both models use the same rule, so the
    comparison stays paired.
    """

    BOXED = re.compile(r"\\boxed\{\s*-?\$?[\d,]+(?:\.\d+)?\s*\}")
    BARE = re.compile(r"####\s*-?\$?[\d,]+(?:\.\d+)?")

    def __init__(self, tokenizer, prompt_len: int):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        text = self.tokenizer.decode(input_ids[0, self.prompt_len:],
                                     skip_special_tokens=True)
        if self.BOXED.search(text):
            return True
        m = self.BARE.search(text)
        if m:
            after = text[m.end():]
            # The number may still grow only while digits follow it.
            if after and not after[0].isdigit() and after[0] != ",":
                return True
        return False


def dataset_cache_fingerprint(names: list[str]) -> dict:
    """Which cached dataset config each slice came from (provenance, not a claim).

    The slices are the canonical test splits, taken in order; recording the
    cache fingerprint makes the exact files auditable without a network call.
    """
    import os
    cache = Path(os.environ.get("HF_DATASETS_CACHE")
                 or (Path.home() / ".cache" / "huggingface" / "datasets"))
    out: dict[str, dict] = {}
    for name in names:
        dirs = sorted(cache.glob(f"{name}/*/*/*"), key=lambda p: p.stat().st_mtime,
                      reverse=True)
        if not dirs:
            out[name] = {"cache_dir": None}
            continue
        out[name] = {"cache_dir": str(dirs[0]),
                     "files": sorted(p.name for p in dirs[0].glob("*.arrow"))}
    return out


def load_model(adapter: Path | None, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_REPO, revision=BASE_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_REPO, revision=BASE_REVISION, dtype=torch.bfloat16)
    model.to(device)
    model.eval()
    adapter_digest = None
    if adapter is not None:
        from peft import PeftModel
        weights = adapter / "adapter_model.safetensors"
        adapter_digest = sha256_file(weights) if weights.is_file() else None
        model = PeftModel.from_pretrained(model, str(adapter))
        model.to(device)
        model.eval()
    return model, tokenizer, adapter_digest


def gsm8k_slice(limit: int) -> list[dict]:
    """Fixed, ordered slice of the official GSM8K test split (same for every model)."""
    from datasets import load_dataset
    try:
        # Canonical id: the short "gsm8k" alias needs a hub round-trip to resolve,
        # which fails under HF_HUB_OFFLINE on a cache that only holds this id.
        ds = load_dataset("openai/gsm8k", "main", split="test")
    except Exception:  # noqa: BLE001 - fall back to the alias for older caches
        ds = load_dataset("gsm8k", "main", split="test")
    rows = []
    for i in range(min(limit, len(ds))):
        row = ds[i]
        rows.append({"item_id": f"gsm8k-test-{i}", "question": row["question"],
                     "answer": row["answer"].split("####")[-1].strip()})
    return rows


def mmlu_slice(limit: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    rows = []
    for i in range(min(limit, len(ds))):
        row = ds[i]
        rows.append({"item_id": f"mmlu-all-test-{i}", "question": row["question"],
                     "choices": list(row["choices"]),
                     "answer_index": int(row["answer"])})
    return rows


@torch.no_grad()
def run_gsm8k(model, tokenizer, items: list[dict], device: str, max_new_tokens: int,
              *, stop_at_answer: bool = True) -> dict:
    from transformers import StoppingCriteriaList
    correct: dict[str, bool] = {}
    samples: list[dict] = []
    gen_tokens: list[int] = []
    for n, item in enumerate(items, 1):
        prompt = render(tokenizer, [{"role": "user", "content":
                        "Solve the problem. Give the final numeric answer after '#### '.\n\n"
                        + item["question"]}], add_generation_prompt=True)
        ids = tokenizer(prompt, return_tensors="pt").to(device)
        stopper = StoppingCriteriaList([StopAtAnswer(tokenizer, ids["input_ids"].shape[1])]) \
            if stop_at_answer else None
        out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                             stopping_criteria=stopper,
                             pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        ok = evaluate.gsm8k_correct(text, item["answer"])
        correct[item["item_id"]] = ok
        if n <= 5 or not ok:
            samples.append({"item_id": item["item_id"], "ok": ok,
                            "predicted_tail": evaluate.gsm8k_extract(text),
                            "gold": item["answer"], "text_tail": text[-160:]})
        gen_tokens.append(int(out.shape[1] - ids["input_ids"].shape[1]))
        if n % 25 == 0:
            print(f"  gsm8k {n}/{len(items)} acc={sum(correct.values())/len(correct):.3f} "
                  f"mean_gen_tokens={sum(gen_tokens)/len(gen_tokens):.0f}", flush=True)
    acc = sum(correct.values()) / len(correct) if correct else None
    return {"accuracy": acc, "items": len(correct), "per_item": correct,
            "stop_at_answer": stop_at_answer,
            "mean_generated_tokens": (sum(gen_tokens) / len(gen_tokens)) if gen_tokens else None,
            "samples": samples[:20]}


@torch.no_grad()
def run_mmlu(model, tokenizer, items: list[dict], device: str,
             style: str = "lead_in") -> dict:
    """Score the four option letters as continuations of the same prompt.

    ``style`` decides what the letters continue:
      * ``lead_in``   — the prompt is extended with "The correct option is" and
                        the letters are scored as " A"/" B"/... continuations.
      * ``bare``      — the letters continue the chat template's generation
                        prompt directly (the first pass measured this way).
    The first pass showed why this matters: with ``bare`` every letter logs
    about -25 with near-uniform scores, i.e. the model does not answer a bare
    letter there and the metric is close to noise. Same items, same greedy
    protocol, same scoring code — only the continuation context differs.
    """
    correct: dict[str, bool] = {}
    samples: list[dict] = []
    for n, item in enumerate(items, 1):
        prompt = (item["question"] + "\n" + "\n".join(
            f"{CHOICES[i]}. {c}" for i, c in enumerate(item["choices"]))
            + "\nAnswer with the letter of the correct option.")
        prefix = render(tokenizer, [{"role": "user", "content": prompt}],
                        add_generation_prompt=True)
        if style == "lead_in":
            prefix = prefix + MMLU_LEAD_IN
        prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(device)
        scores = []
        for letter in CHOICES:
            text = f" {letter}" if style == "lead_in" else letter
            cont = tokenizer(text, add_special_tokens=False,
                             return_tensors="pt").input_ids.to(device)
            full = torch.cat([prefix_ids, cont], dim=1)
            logits = model(full).logits[:, :-1, :]
            logprobs = torch.log_softmax(logits.float(), dim=-1)
            picked = logprobs[0, -cont.shape[1]:, :]
            idx = full[0, -cont.shape[1]:]
            scores.append(float(picked[torch.arange(cont.shape[1]), idx].sum()))
        guess = max(range(len(CHOICES)), key=lambda i: scores[i])
        correct[item["item_id"]] = guess == item["answer_index"]
        if n <= 5 or guess != item["answer_index"]:
            samples.append({"item_id": item["item_id"], "ok": guess == item["answer_index"],
                            "chosen": CHOICES[guess], "gold": CHOICES[item["answer_index"]],
                            "logprobs": [round(s, 3) for s in scores]})
        if n % 50 == 0:
            print(f"  mmlu {n}/{len(items)} acc={sum(correct.values())/len(correct):.3f}", flush=True)
    acc = sum(correct.values()) / len(correct) if correct else None
    return {"accuracy": acc, "items": len(correct), "per_item": correct,
            "style": style, "samples": samples[:20]}


@torch.no_grad()
def run_dev_perplexity(model, tokenizer, rows: list[dict], device: str, limit: int) -> dict:
    """Completion-only NLL over the assistant turn of never-trained-on dev rows."""
    total_nll = 0.0
    total_tokens = 0
    used = 0
    for row in rows[:limit]:
        messages = [m for m in (row.get("messages") or [])
                    if isinstance(m, dict) and isinstance(m.get("content"), str)]
        if len(messages) < 2 or messages[-1].get("role") != "assistant":
            continue
        assistant = messages[-1]["content"]
        prefix_text = render(tokenizer, messages[:-1], add_generation_prompt=True)
        full_text = render(tokenizer, messages, add_generation_prompt=False)
        if not assistant or not full_text.startswith(prefix_text):
            continue
        prefix_ids = tokenizer(prefix_text, add_special_tokens=False).input_ids
        full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
        if len(full_ids) <= len(prefix_ids) + 1:
            continue
        ids = torch.tensor([full_ids], device=device)
        logits = model(ids).logits[:, :-1, :]
        logprobs = torch.log_softmax(logits.float(), dim=-1)[0]
        # logprobs[j] scores token j+1, so the first completion token (index
        # `start`) is scored by row `start - 1`; align rows and targets alike.
        start = len(prefix_ids)
        target = ids[0, 1:]
        picked = logprobs[torch.arange(start - 1, target.shape[0]), target[start - 1:]]
        total_nll += float(-picked.sum())
        total_tokens += int(picked.numel())
        used += 1
    mean_nll = total_nll / total_tokens if total_tokens else None
    return {"rows": used, "completion_tokens": total_tokens, "mean_nll": mean_nll,
            "perplexity": (float(torch.exp(torch.tensor(mean_nll)))
                           if mean_nll is not None else None)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", help="model label for the result record")
    ap.add_argument("--adapter", type=Path, help="LoRA adapter dir (omit for the untouched base)")
    ap.add_argument("--out", type=Path, help="result JSON path")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--gsm8k-limit", type=int, default=200)
    ap.add_argument("--mmlu-limit", type=int, default=200)
    ap.add_argument("--ppl-limit", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--compare", nargs=2, type=Path,
                    help="two result JSONs: print the paired comparison and exit")
    ap.add_argument("--only", choices=["all", "gsm8k", "mmlu", "ppl"], default="all",
                    help="run one phase (cheap re-scoring of a single axis)")
    ap.add_argument("--mmlu-style", choices=["lead_in", "bare"], default="lead_in")
    ap.add_argument("--no-stop-at-answer", action="store_true",
                    help="decode to the token cap instead of stopping at the answer marker")
    args = ap.parse_args()

    if args.compare:
        base = json.loads(args.compare[0].read_text(encoding="utf-8"))
        cand = json.loads(args.compare[1].read_text(encoding="utf-8"))
        base_metrics = {k: v for k, v in base.get("metrics", {}).items()
                        if isinstance(v, (int, float))}
        cand_metrics = {k: v for k, v in cand.get("metrics", {}).items()
                        if isinstance(v, (int, float))}
        out = {"base_label": base.get("label"), "candidate_label": cand.get("label"),
               "metric_values": {"base": base_metrics, "candidate": cand_metrics},
               "metric_deltas": {k: round(cand_metrics[k] - base_metrics[k], 4)
                                 for k in sorted(set(base_metrics) & set(cand_metrics))},
               # evaluate.compare covers the REPAIR metrics; the capability
               # metrics above are the paired numbers this command exists for.
               "scalars": evaluate.compare(base_metrics, cand_metrics)}
        for name in ("gsm8k", "mmlu"):
            b = (base.get("details", {}).get(name) or {}).get("per_item")
            c = (cand.get("details", {}).get(name) or {}).get("per_item")
            if b and c:
                out[f"{name}_paired"] = paired_table(b, c)
        ppl_base = base.get("metrics", {}).get("dev_mean_nll")
        ppl_cand = cand.get("metrics", {}).get("dev_mean_nll")
        if ppl_base and ppl_cand:
            out["dev_nll_delta"] = ppl_cand - ppl_base
        print(json.dumps(out, indent=2, sort_keys=True))
        if args.out:
            args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
        return 0

    if not args.label or not args.out:
        ap.error("--label and --out are required unless --compare is used")

    device_index = int(args.device.split(":")[-1]) if ":" in args.device else 0
    exclusivity = train_pilot.check_device_exclusivity(
        device_index, min_free_gb=EVAL_MIN_FREE_VRAM_GB)
    print(f"[exclusivity] {json.dumps(exclusivity)[:300]}", flush=True)

    started = time.perf_counter()
    model, tokenizer, adapter_digest = load_model(args.adapter, args.device)
    dev_rows = [json.loads(l) for l in DEV_ROWS.read_text(encoding="utf-8").splitlines()
                if l.strip()]

    details: dict[str, dict] = {}
    metrics: dict[str, float] = {}
    failures: dict[str, str] = {}

    gsm_items: list[dict] = []
    if args.only in ("all", "gsm8k"):
        gsm_items = gsm8k_slice(args.gsm8k_limit)
        details["gsm8k"] = run_gsm8k(model, tokenizer, gsm_items, args.device,
                                     args.max_new_tokens,
                                     stop_at_answer=not args.no_stop_at_answer)
        if details["gsm8k"]["accuracy"] is not None:
            metrics["gsm8k_accuracy"] = details["gsm8k"]["accuracy"]

    if args.only in ("all", "mmlu"):
        try:
            mmlu_items = mmlu_slice(args.mmlu_limit)
            details["mmlu"] = run_mmlu(model, tokenizer, mmlu_items, args.device,
                                       args.mmlu_style)
            if details["mmlu"]["accuracy"] is not None:
                metrics["mmlu_accuracy"] = details["mmlu"]["accuracy"]
        except Exception as exc:  # noqa: BLE001 - record the failure, never a number
            failures["mmlu"] = f"{type(exc).__name__}: {exc}"
            print(f"[warn] mmlu unavailable: {failures['mmlu'][:200]}", flush=True)

    if args.only in ("all", "ppl"):
        details["dev_perplexity"] = run_dev_perplexity(model, tokenizer, dev_rows,
                                                       args.device, args.ppl_limit)
        if details["dev_perplexity"]["mean_nll"] is not None:
            metrics["dev_mean_nll"] = details["dev_perplexity"]["mean_nll"]
            metrics["dev_perplexity"] = details["dev_perplexity"]["perplexity"]

    record = {
        "label": args.label,
        "adapter": str(args.adapter) if args.adapter else None,
        "adapter_model_sha256": adapter_digest,
        "base_model": BASE_REPO,
        "base_revision": BASE_REVISION,
        "device": args.device,
        "exclusivity": exclusivity,
        # What the comparison is actually against, stated so nobody can read a
        # "gen-2" claim into it: the gen-2 campaign is a 9B ablation with no
        # trained 1.7B artifact (docs/gen2/gen2_campaign.json, docs/HANDOFF.md
        # line 54 "Gen2 has not been trained."), so the reference for this
        # student is the untouched base on identical item slices.
        "protocol": {
            "reference_model": "untouched base (Qwen3-1.7B @ base_revision)",
            "gen2_reference_available": False,
            "gen2_note": ("no 1.7B-scale gen-2 artifact exists; gen-2 is an untrained 9B "
                          "campaign, so this is a paired base-vs-student comparison"),
            "paired_item_slices": True,
            "decoding": "greedy (do_sample=False)",
            "thinking": "disabled (enable_thinking=False)",
            "mmlu_scoring": "sum of log-probs over the option letter",
            "dev_ppl_scoring": "completion-only NLL over the assistant turn",
        },
        "datasets": dataset_cache_fingerprint(["openai___gsm8k", "cais___mmlu"]),
        "phase": args.only,
        "mmlu_style": args.mmlu_style,
        "stop_at_answer": not args.no_stop_at_answer,
        "slices": {"gsm8k": len(gsm_items),
                   "mmlu": len(details.get("mmlu", {}).get("per_item", {})),
                   "dev_rows_scored": details.get("dev_perplexity", {}).get("rows", 0)},
        "metrics": metrics,
        "details": details,
        "failures": failures,
        "wall_seconds": round(time.perf_counter() - started, 1),
        "versions": {"torch": torch.__version__},
    }
    args.out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({"label": args.label, "metrics": metrics,
                      "failures": failures, "out": str(args.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
