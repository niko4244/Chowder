# Handoff — A4b conclusion, Chowder training verification, first real training run

Written 2026-09-11 ~20:15. **One run is still in flight — see §1 first.**

Repos and branches:
* Chowder: `C:\Users\nikma\Chowder-router-healing` on `feat/hot-core-upcycling`,
  HEAD `9f66960`, pushed, open as **PR #158**. Clean tree.
* Frontier: `C:\Users\nikma\frontier-lowram-autoresearch` on
  `feat/16gbvram-glm52-deepseek-recipe`, HEAD `40cdffb`, pushed. **5 unrelated
  v4-scoring files remain modified in the working tree and were deliberately never
  staged** — `evals/mmlu_pro_eval.py`, `kaggle/cycle4_math_kernel/run_cycle4_math_train.py`,
  `tests/test_mmlu_pro.py`, `training/adapter_delta.py`, `training/train_dpo_trl.py`.
  Leave them alone unless you own that work.

---

## 1. IN FLIGHT: first real training run (GSM8K SFT on the pruned 9B)

**Still running as of 20:13.** PID ~111220 (2.1 GB RSS), GPU 6.5 GiB / ~25%.

* Launcher: `scratchpad/wait_then_train.sh` → `scratchpad/real_train_gsm8k.py`
* Log: `scratchpad/realtrain4.log`
* Work dir: `F:\llm-models\_a4b\realtrain-gsm8k`
* Pre-registration (committed **before** the run, `9f66960`):
  `docs/PRUNED_9B_REAL_TRAINING_PREREG.md`

Progress: automatic baseline eval **28 of 50** problems at 20:13. Then 500 training
steps (~15–20 min), then the candidate eval (~85 min). **Expect completion around
22:30–23:00.**

It is slow for a reason that is itself the headline finding below: the model never
emits EOS, so every problem burns the full 768-token budget (~1.7 min each).

### To pick it up

```bash
tail -f "C:/Users/nikma/AppData/Local/Temp/claude/C--Users-nikma/d2ee79ac-3407-48b9-8397-d6ba96c24813/scratchpad/realtrain4.log"
```

When it finishes, the verdict lands in
`F:\llm-models\_a4b\realtrain-gsm8k\level2-report.json` (keys: `registry_results`,
`gate_promoted`, `telemetry`, `resolved_target_modules`). Judge it against the
pre-registered outcomes — do **not** invent new thresholds.

### Interim finding, already solid and important

**The pruned model degenerates into repetition loops and scored 0.00 on the first
28 GSM8K problems.** Sample tail: *"The number of eggs for the farmers' market is 2."*
repeated to the token cap.

This **corrects the static-prune recommendation**: perplexity 1.863×/3.014× dense
looked tolerable, but on multi-step generation the model is not usable.
**Perplexity was a misleading proxy for generative reasoning**, and every
prune-vs-MoE comparison in this session was made on perplexity alone.

### The control this finding still needs — run it first

Unrun, prepared, and **required before believing the above**:
`scratchpad/control_dense_vs_pruned_generation.py`. It generates on the same 8
prompts with the dense parent and the pruned model, same settings and scorer, and
reports GSM8K, duplicate-line ratio, and token-cap-hit rate. If the **dense** model
also loops, the prompt/harness is at fault, not pruning. Run it after the main run
releases the GPU:

```bash
cd /c/Users/nikma/Chowder-router-healing && PYTHONPATH=src python "<scratchpad>/control_dense_vs_pruned_generation.py"
```

---

## 2. A4b concluded: take the static prune, not the MoE

Full writeup: `docs/HOT_CORE_VS_STATIC_PRUNE.md`. Evidence:
`evidence/hot-core-upcycling/`.

| | eval A | eval B | geo-mean ×dense |
|---|---:|---:|---:|
| dense parent | 5.2707 | 4.3044 | 1.000× |
| hot-core MoE init (corpus-wide rank) | 13.4224 | 15.2936 | 3.008× |
| **static prune (corpus-wide rank)** | **9.9235** | **13.0258** | **2.387×** |

The MoE costs ~26% more at identical active compute. It spends 2,176 of 3,440 active
channels on a fixed core, so at init it trades the next-best channels for a scattered
cold sample — and 150 healing steps did not earn that back (pre-registered
attribution: the router contributed nothing; the gain was the shared-expert gate).
**The core-share sweep had already said this** — more core was better at every
budget, and 100% core *is* static pruning. I read a monotone result as a starting
point instead of a conclusion, which cost two training runs and a second conversion.

**Deliverable:** `F:\llm-models\Qwen3.8-9B-Pruned-CW-3456` — `qwen3_5` unchanged,
total = active = 5.937B (from 9.410B), 11.87 GB on disk, peak VRAM 5.46 GiB (less
than the dense parent), 3,456 = 54×64 so bitsandbytes uses its fast kernel.
**Caveat: §1 shows it cannot generate coherently. Do not ship it on the perplexity
numbers alone.**

Also established: **rankings must be corpus-wide**. A 32-prompt contiguous ranking
cost 1.691× near its own data and 3.646× far from it; the same count spread across
the corpus cost 1.859×/3.011× — 17.4% better out of distribution, sharing only 73.5%
of its top-3,440 channels. `channel_importance.spread_across` enforces this.

**≤3.5B active remains unreachable** and needs the always-on axis (tie embeddings,
drop vision, narrow hidden): the always-on floor is 4.578B before any FFN runs.

---

## 3. Chowder's training path: four silent failures, now loud or fixed

The tests that prove Chowder trains are gated behind `CHOWDER_REAL_ML_SMOKE=1` /
`CHOWDER_REAL_UNSLOTH_SMOKE=1` and sit among the **77 skipped**. Every "1495 passed"
in this session said nothing about training until those gates were opened.

| # | defect | fix | commit |
|---|---|---|---|
| 1 | Workers resolved `import chowder` through an editable `.pth` → from any worktree they ran **different code than the parent** | `chowder/worker_env.py`, all 11 launch sites | `b7c1710` |
| 2 | An adapter whose keys all mismatched loaded "successfully" and was **inert** (logit delta 0.000000) while provenance said `adapter_loaded: true` | `chowder/adapter_guard.py`, all 4 load sites | `3621965` |
| 3 | Unsloth loaded the VLM wrapper, so its adapter keys could never match the evaluator's model | `text_only=True` (Unsloth's own supported text-decoder path) | `bf190e6` |
| 4 | An explicit target list silently adapted **128 of 200** modules — all 72 `linear_attn` skipped | `chowder/target_coverage.py` (refuses), then a suffix-match **regex** instead of a list (fixes) | `81cfe05`, `0bd62ed` |

Gated suites after the fixes: Transformers real-ML **4/4** (was 1/4); both Unsloth
real tests pass, one **for the first time ever** (it had always skipped) and the
repair-loop test is now re-runnable (it had passed once in PR #135 and collided with
its own registry ever since).

Both engines now train the pruned model at full 200-module coverage:

| | Transformers | Unsloth |
|---|---|---|
| modules adapted | 200 | 200 |
| peak VRAM | 11.66 GB | **5.96 GB** |
| training time | 137 s | **65 s** |

**Two things any future trainer of this architecture needs.** PEFT has **no
auto-detection for `qwen3_5`**, so `target_modules` must be explicit, and Chowder's
curated `attention_and_mlp` preset has no entry either. The verified list (matches
this repo's own `train_grpo_minimal.py` BROAD_MODULES): `q/k/v/o_proj` (8
full-attention layers), `in_proj_qkv`/`in_proj_z`/`out_proj` (24 `linear_attn`),
`gate/up/down_proj` (all 32). For Unsloth it must go through as a **regex**, not a
list.

**Flagged for re-checking:** prior Unsloth results on models with a VLM wrapper
shape predate fix #2/#3 and may have silently scored base-model performance.

---

## 4. Open work, in the order I would take it

1. **Finish §1 and run its control.** The control decides whether the repetition
   finding is about pruning or about my harness. Everything else about the pruned
   checkpoint's usability depends on it.
2. **If pruning is confirmed as the cause:** the static-prune recommendation needs a
   generative metric, not just perplexity. Re-evaluate at milder pruning (f=0.56 cost
   only 1.14× perplexity) and find the fraction where generation survives.
3. **Curated `attention_and_mlp` preset entry for `qwen3_5`** — the module list is
   now verified three ways, so this is safe and removes a footgun. Parked only for
   scope.
4. **A coverage check for partial-vs-requested on the *transformers* engine's
   presets**, same spirit as #4 above.
5. Earlier open items, unchanged: matched dense-vs-converted perplexity at the
   unproven 5.44 baseline; the full 1162-passage census; parent D's evidence gap;
   the uncensored-objective-vs-parent-A tension.

## 5. Method notes worth keeping

* **On this platform `torch.cuda.OutOfMemoryError` never fires** — Windows WDDM
  oversubscribes into system RAM. Judge fit by headroom plus step-time blowup. A
  probe of mine reported "FITS at seq 2048" while that step took 137 s against 7 s
  at 768.
* **`peft.prepare_model_for_kbit_training` upcasts every non-quantized parameter to
  fp32**, including a frozen raw-`nn.Parameter` MoE expert bank: +11.20 GiB measured,
  predicted +11.19.
* **A measurement gap is not a defect.** Both new guards record "unknown" rather
  than failing when they cannot measure (unreadable weights; an unreported coverage
  count). Without that, every fake-worker test failed spuriously — which is how I
  found the rule.
* Pre-registration earned its keep twice: it produced an `ATTRIBUTION FAIL` I would
  otherwise have been tempted to read as success, and it stopped me trimming `n` and
  the token budget mid-run in §1 to save time.
