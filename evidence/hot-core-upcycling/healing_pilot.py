"""Hot-core router-healing pilot. Spec: docs/HOT_CORE_HEALING_PILOT_PREREG.md.

Narrow, falsifiable question: does training the router move perplexity at all?
Thresholds were fixed before the run (see the prereg) and are restated here as
constants so the verdict is computed, not chosen.

Attribution is the weak point of this design: `shared_expert_gate` is in the
trainable scope and controls the hot core's contribution, so a perplexity gain
could come from re-weighting the core rather than from routing. Both tensors
start at EXACTLY zero, which makes attribution measurable: the run records each
group's weight norm over time and, at the end, re-evaluates three ways --
both trained, router-only (gate reset to 0), gate-only (router reset to 0).
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\Users\nikma\Chowder-router-healing\src")

MODEL = r"F:\llm-models\Qwen3.8-9B-HotCore-E16-k2-h2176"
CALIB = Path(r"C:\Users\nikma\frontier-lowram-autoresearch\training\data\grpo_prompts_borderline_696.jsonl")
OUT = Path(r"F:\llm-models\_a4b\healing-pilot")

TRAIN_SEQ, EVAL_MAXLEN = 768, 384      # eval matches the 9.8444 baseline exactly
STEPS, EVAL_EVERY = 150, 25
LR, WARMUP, CLIP = 1e-3, 10, 1.0

# pre-registered reference points (all measured, held out)
INIT_PPL = 9.8444                      # converted model, eval A
STATIC_PRUNE_PPL = 8.9096              # static hot prune at f=0.28, no router
PASS_PPL = 9.648                       # 2% below init
DENSE_PPL = 5.3137


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from chowder.router_healing import (
        QUANTIZATION_SKIP_MODULES, assert_trainable_gradients_reachable,
        freeze_for_router_healing, select_trainable_parameter_names,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l)["prompt"] for l in
            CALIB.read_text(encoding="utf-8").splitlines() if l.strip()]
    # The file has 690 distinct texts in 696 rows: 6 prompts appear twice. Index
    # ranges alone are therefore NOT disjoint splits -- 3 of eval B's texts also
    # occur inside the train range. Dedup on TEXT, and drop the overlap from
    # train rather than from the eval sets, so eval A stays byte-identical to the
    # split that produced the recorded 9.8444 baseline.
    eval_a = rows[1:64:2]          # the split the 9.8444 baseline came from
    eval_b = rows[632:696]         # never seen by ranking or training
    rank_split = set(rows[0:64:2])
    held_out = rank_split | set(eval_a) | set(eval_b)
    seen: set[str] = set()
    train_rows = []
    for text in rows[64:632]:
        if text in held_out or text in seen:
            continue
        seen.add(text)
        train_rows.append(text)
    dropped = 568 - len(train_rows)
    assert not (set(eval_a) & set(train_rows)), "eval A leaked into train"
    assert not (set(eval_b) & set(train_rows)), "eval B leaked into train"
    assert not (rank_split & set(train_rows)), "rank split leaked into train"
    log(f"splits: rank 32 | evalA {len(eval_a)} | train {len(train_rows)} "
        f"(dropped {dropped} duplicate/held-out) | evalB {len(eval_b)}")

    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=list(QUANTIZATION_SKIP_MODULES)),
        dtype=torch.bfloat16, device_map="cuda:0",
        local_files_only=True, trust_remote_code=False)

    summary = freeze_for_router_healing(model)
    assert_trainable_gradients_reachable(
        model, select_trainable_parameter_names(list(model.named_parameters())))
    log(f"trainable {summary.trainable_param_count/1e6:.3f}M over "
        f"{len(summary.trainable_param_names)} tensors; reachability PASS")
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    router_params = {n: p for n, p in model.named_parameters()
                     if p.requires_grad and n.endswith("mlp.gate.weight")}
    gate_params = {n: p for n, p in model.named_parameters()
                   if p.requires_grad and n.endswith("mlp.shared_expert_gate.weight")}
    log(f"router tensors {len(router_params)}, shared-gate tensors {len(gate_params)}")
    for p in list(router_params.values()) + list(gate_params.values()):
        assert float(p.detach().float().abs().max()) == 0.0, "expected zero init"
    log("confirmed: both groups start at exactly zero (makes attribution measurable)")

    def group_norm(group) -> float:
        return math.sqrt(sum(float(p.detach().float().norm()) ** 2 for p in group.values()))

    # ---- training blocks: concatenate train prompts, chunk to TRAIN_SEQ ----
    ids = tok("\n\n".join(train_rows), return_tensors="pt")["input_ids"][0]
    blocks = [ids[i:i + TRAIN_SEQ] for i in range(0, len(ids) - 1, TRAIN_SEQ)]
    blocks = [b for b in blocks if len(b) >= 64]
    log(f"train stream {len(ids)} tokens -> {len(blocks)} blocks of <= {TRAIN_SEQ}")

    def ppl(prompts) -> float:
        model.eval()
        was = model.config.use_cache
        model.config.use_cache = False
        total, n_tok = 0.0, 0
        with torch.no_grad():
            for text in prompts:
                enc = tok(text, return_tensors="pt", truncation=True, max_length=EVAL_MAXLEN)
                enc = {k: v.to(model.device) for k, v in enc.items()}
                if enc["input_ids"].shape[1] < 8:
                    continue
                loss = model(**enc, labels=enc["input_ids"]).loss.float().item()
                if not math.isfinite(loss):
                    return float("inf")
                n = enc["input_ids"].shape[1] - 1
                total += loss * n
                n_tok += n
        model.config.use_cache = was
        model.train()
        return math.exp(total / n_tok)

    trainable = list(router_params.values()) + list(gate_params.values())
    optimizer = torch.optim.AdamW(trainable, lr=LR, betas=(0.9, 0.999), weight_decay=0.01)

    def lr_at(step: int) -> float:
        if step < WARMUP:
            return (step + 1) / WARMUP
        prog = (step - WARMUP) / max(1, STEPS - WARMUP)
        return 0.5 * (1 + math.cos(math.pi * prog))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)

    trajectory = []

    def record(step: int, loss: float | None) -> None:
        a, b = ppl(eval_a), ppl(eval_b)
        row = {"step": step, "eval_a_ppl": a, "eval_b_ppl": b,
               "eval_a_ratio_to_dense": a / DENSE_PPL,
               "router_norm": group_norm(router_params),
               "shared_gate_norm": group_norm(gate_params),
               "train_loss": loss}
        trajectory.append(row)
        log(f"  step {step:>4}  evalA {a:>8.4f} ({a/DENSE_PPL:>5.3f}x)  evalB {b:>8.4f}  "
            f"|W| router {row['router_norm']:>7.4f} gate {row['shared_gate_norm']:>7.4f}"
            + (f"  loss {loss:.4f}" if loss is not None else ""))

    log("baseline (step 0) -- must reproduce the recorded 9.8444 on eval A")
    record(0, None)
    drift = abs(trajectory[0]["eval_a_ppl"] - INIT_PPL)
    if drift > 0.05:
        log(f"WARNING: step-0 eval A is {trajectory[0]['eval_a_ppl']:.4f}, recorded "
            f"{INIT_PPL} (drift {drift:.4f}). Investigate before trusting deltas.")
    else:
        log(f"  reproduces within {drift:.4f} -- eval path matches the baseline")

    model.train()
    recent: list[float] = []
    t0 = time.time()
    for step in range(STEPS):
        block = blocks[step % len(blocks)].unsqueeze(0).to(model.device)
        out = model(input_ids=block, labels=block)
        loss = out.loss
        if not math.isfinite(loss.item()):
            log(f"non-finite loss at step {step}; stopping")
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, CLIP)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        recent.append(loss.item())
        if (step + 1) % EVAL_EVERY == 0:
            record(step + 1, sum(recent[-EVAL_EVERY:]) / len(recent[-EVAL_EVERY:]))
    log(f"trained {len(recent)} steps in {(time.time()-t0)/60:.1f} min")

    # ---- attribution: which group produced any gain? ----
    log("attribution: re-evaluating with each group reset to its zero init")
    router_state = {n: p.detach().clone() for n, p in router_params.items()}
    gate_state = {n: p.detach().clone() for n, p in gate_params.items()}
    with torch.no_grad():
        for n, p in gate_params.items():
            p.zero_()
    router_only = {"eval_a_ppl": ppl(eval_a), "eval_b_ppl": ppl(eval_b)}
    log(f"  router only (gate reset)  evalA {router_only['eval_a_ppl']:.4f}  "
        f"evalB {router_only['eval_b_ppl']:.4f}")
    with torch.no_grad():
        for n, p in gate_params.items():
            p.copy_(gate_state[n])
        for n, p in router_params.items():
            p.zero_()
    gate_only = {"eval_a_ppl": ppl(eval_a), "eval_b_ppl": ppl(eval_b)}
    log(f"  gate only (router reset)  evalA {gate_only['eval_a_ppl']:.4f}  "
        f"evalB {gate_only['eval_b_ppl']:.4f}")
    with torch.no_grad():
        for n, p in router_params.items():
            p.copy_(router_state[n])

    # ---- pre-registered verdict, computed not chosen ----
    best = min(trajectory, key=lambda r: r["eval_a_ppl"])
    init_a, init_b = trajectory[0]["eval_a_ppl"], trajectory[0]["eval_b_ppl"]
    a_improved = best["eval_a_ppl"] <= PASS_PPL
    b_improved = best["eval_b_ppl"] < init_b
    milestone = best["eval_a_ppl"] <= STATIC_PRUNE_PPL
    router_gain = init_a - router_only["eval_a_ppl"]
    gate_gain = init_a - gate_only["eval_a_ppl"]
    attribution = ("router" if router_gain > 2 * max(gate_gain, 0.0)
                   else "shared_expert_gate" if gate_gain > 2 * max(router_gain, 0.0)
                   else "both/unclear")
    if not a_improved:
        verdict = "FAIL"
    elif attribution == "shared_expert_gate":
        verdict = "ATTRIBUTION FAIL"
    elif milestone and b_improved:
        verdict = "THESIS MILESTONE"
    elif b_improved:
        verdict = "PASS"
    else:
        verdict = "WEAK PASS"

    log("")
    log(f"best eval A {best['eval_a_ppl']:.4f} at step {best['step']} "
        f"(init {init_a:.4f}, pass <= {PASS_PPL}, milestone <= {STATIC_PRUNE_PPL})")
    log(f"eval B {best['eval_b_ppl']:.4f} vs init {init_b:.4f}")
    log(f"attribution: router gain {router_gain:+.4f}, gate gain {gate_gain:+.4f} "
        f"-> {attribution}")
    log(f"VERDICT: {verdict}")

    (OUT / "pilot-result.json").write_text(json.dumps({
        "model": MODEL, "prereg": "docs/HOT_CORE_HEALING_PILOT_PREREG.md",
        "recipe": {"train_seq": TRAIN_SEQ, "eval_maxlen": EVAL_MAXLEN, "steps": STEPS,
                   "lr": LR, "warmup": WARMUP, "clip": CLIP, "optimizer": "AdamW",
                   "trainable_params": summary.trainable_param_count},
        "splits": {"rank": 32, "eval_a": len(eval_a), "train": len(train_rows), "train_dropped_duplicates": dropped,
                   "eval_b": len(eval_b), "train_tokens": int(len(ids)),
                   "train_blocks": len(blocks)},
        "references": {"dense_ppl": DENSE_PPL, "init_ppl": INIT_PPL,
                       "static_prune_ppl": STATIC_PRUNE_PPL, "pass_ppl": PASS_PPL},
        "trajectory": trajectory,
        "attribution": {"router_only": router_only, "gate_only": gate_only,
                        "router_gain": router_gain, "gate_gain": gate_gain,
                        "verdict": attribution},
        "best": best, "verdict": verdict,
    }, indent=2) + "\n", encoding="utf-8")
    log(f"wrote {OUT/'pilot-result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
