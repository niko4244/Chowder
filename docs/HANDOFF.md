# Session Handoff

Continuity document for agent sessions working on Chowder. **Every session
that lands work on `main` updates this document in the same session** —
it is the delta-layer on top of the repo's own truth, not a replacement
for it:

- What is proven vs. still open: [`ROADMAP.md`](ROADMAP.md) (with PR
  numbers). This doc never restates roadmap status; it points at it.
- Teacher Fabric architecture and slice plan:
  [`TEACHER_FABRIC.md`](TEACHER_FABRIC.md). The verbatim Priority-8
  mission brief (15 regression rules, 10-signal taxonomy, exact Slice A
  scope, slices B–J) is preserved at
  [`TEACHER_FABRIC_BRIEF.md`](TEACHER_FABRIC_BRIEF.md) — read it before
  any Teacher Fabric slice; it is the source of the non-negotiable rules.

## Current state (updated 2026-09-11) — READ THE HEADLINE FIRST

**The sparse program's central mechanism does not work, and this is now
measured, not suspected.** The dense→MoE conversion itself is validated
(perplexity 5.44 at top_k=16, a healthy number — the first numerical
proof the conversion is sound on the real 27B checkpoint). But reducing
top_k to buy compute destroys the model: 67.8 at k=8, 51,030 at k=4,
**243,981 at k=3** — the setting the program was aiming for. Full
measurement: `Chowder-Protected/runs/v3-20260909/topk-ladder-finding.md`
and `topk-ladder-screen.json`.

**Root cause (structural, not a tuning problem).** The conversion
partitions one FFN into 16 *disjoint* channel slices, so the experts are
**complementary, not redundant**: the dense output is a sum over all
17,408 channels, and any k<16 subset is a partial sum of one computation
rather than an alternative computation. A natively-trained MoE's experts
are each individually competent; these are not. **Therefore router
healing with frozen experts cannot work here regardless of training
budget** — it can only choose which slices to keep, and the ceiling is
fixed by construction. Partial sums would only suffice under genuine
activation sparsity, and Qwen3.8 is SiLU (no hard zeros, every channel
contributes). That also independently explains the Phase 4 census
negative result: there was no sparsity structure to exploit, which is why
no clustering strategy beat random.

**Target relabelled.** A3B/A4B is retired as unreachable on two
independent grounds — parameter arithmetic (always-on floor 11.17B, so
12.24B at top_k=1 / 14.38B at top_k=3) and measured quality. Honest
labels: **A28B measured today**, **A12–A14B aspirational** if the
redundancy problem is solved. See `QWEN38_SPARSE_PROGRAM.md`'s relabel
note. Phase 11 accounting now exists beside the artifact
(`Qwen3.8-27B-MoE-E16.accounting.json`) — it previously could not be
produced at all, because `parameter_accounting.py` demanded a `.weight`
suffix the real raw-`nn.Parameter` expert tensors do not carry.

**Four honest paths, none chosen, none cheap:** (1) train the expert
weights — real MoE upcycling, makes experts individually competent;
(2) convert activations to dReLU first to create the sparsity the
partition needs (~150B tokens per the research note); (3) a conversion
yielding redundant rather than complementary experts, trading storage for
droppability; (4) accept a dense ~28B model with MoE structure and no
sparsity win.

**Corrections to earlier claims in this document's history — do not
propagate these:**
- "Byte-exact to the dry-run estimate" was **false**. Real output is
  56,580,816,384 bytes vs the 56,580,780,856 estimate: **+35,528 bytes**
  of safetensors header padding the shape arithmetic does not model. No
  byte-check had ever been performed — provenance recorded only the
  estimate. `convert_checkpoint` now records `actual_output_bytes`.
- "D's tokenizer gate passed — real, recorded evidence" was
  **overstated**. That result exists only in an ephemeral Temp log, not in
  any persisted artifact, and the selection packet covers only A/B/C.
- "Behavior 0.0 is a scorer bug, not bad behaviour" holds **for A only**.
  The v4 rescore shows **B and C verdict "comply" on all three harmful
  prompts** — for them the 0.0 partly reflects genuinely permissive
  behaviour. Also A's 1.0 is softer than it looks: one of its six items
  scored via the classifier's `empty_generation_verdict: "refuse"` default
  on a truncated empty answer, not an observed refusal.
- "retry7 was never modified" was **unverifiable** — no baseline hash
  existed, and the DB's mtime did move (a WAL-mode read-write open;
  content verified intact). Now sealed:
  `Chowder-Protected/registry-baseline-hashes.json`.

**Unresolved, needs a user decision:** this program's stated aim is an
*uncensored* model and names B its primary development parent, but the
frozen parent is **A**, the official control — which scores as the parent
that refuses *most* (v4 rescore: A=1.0 vs B=C=0.5). Nothing records that
the uncensored objective was superseded.

### Chowder now trains this workload end to end (2026-09-11, measured)

The gap an audit named — "the only working healing run was a standalone
Temp script that recorded no experiment, accounted no GPU-hours, and never
reached the gate" — is closed. `router_healing_orchestrator.
run_router_healing_experiment` drives a healing experiment through the
existing registry/gate/budget lifecycle, and a **bounded engineering run
on real hardware proved it**: experiment `exp-router-healing-
25f3a7b390dcf546`, own registry at `Chowder-Protected/
engineering-healing.registry.db`, artifacts under
`Chowder-Protected/runs/engineering-healing-20260911/`.

This run's purpose was to prove the PATH, not to improve the model, and
its success criterion was explicitly "trustworthy lifecycle even if the
model fails the gate".

| acceptance condition | evidence |
|---|---|
| 1. preflight measures real formats/memory | 38.455 GiB resident, **quantized_fraction 0.1091** |
| 2. intended tensors can receive gradient | `nonzero-grad tensors=64/64` on both steps |
| 3. delta + resumable state saved | 118 MB delta + `.resume.json`, written pre-evaluation |
| 4. independent reload, versioned protocol | delta re-read from disk, base manifest `ff9c0e84…` verified |
| 5. cost/outcome/verdict in the registry | `status=rejected`, `gpu_hours=0.277221` measured |

- **The "4-bit" label is now measured as misleading, not just argued.**
  Only **10.91%** of resident bytes are quantized: `routed_expert` is
  **31.88 GiB of bfloat16** (128 tensors) on a 16 GiB card, because
  bitsandbytes replaces `nn.Linear` only and `Qwen3_5MoeExperts` holds
  `gate_up_proj`/`down_proj` as raw `nn.Parameter`. Attention/embedding
  *are* largely uint8. This is the real budget and the reason steps cost
  ~5 min (310.5s and 308.2s measured).
- **Scope was narrowed honestly, not bypassed.** Trained
  `ROUTER_ONLY_SUFFIXES` (64 × `mlp.gate.weight` = 5,242,880 params;
  `layers_with_trainable_shared_expert_gate: 0`) against 22,893,387,264
  frozen. The shared-expert gate is provably unlearnable on this
  checkpoint (zero-init frozen shared expert ⇒ exactly zero derivative;
  CPU-probed 0.0 vs 0.16 with non-zero weights), so designating it would
  correctly trip the new reachability check. `require_reachable=False`
  exists but was NOT used.
- **The gate rejected, and that is condition 5 passing.** Evaluation was
  held-out perplexity (passages 200-204, 1020 tokens, **5.7057**) rather
  than the ~2h 54-item suite, so the gate returned `rejected: evaluation
  evidence is incomplete` and named all nine missing protected dimensions.
  A promotion authority that cannot be satisfied by partial evidence is
  the property we want; do not "fix" this by loosening the goal.
- **No quality claim.** Two steps is a lifecycle proof. Perplexity moved
  5.4436 → 5.7057, which at this step count is noise, not a result, and
  the 5.4436 reference itself still lacks a matched dense-parent
  comparison (see the corrections above).

**Defect found and fixed in the same cycle**, by an independent audit of
this branch's own work: `freeze_for_router_healing` designated 64
shared-expert gates that could never learn, and the earlier pilot's
proof-of-life (one *global* grad-norm plus the router's weight norm, both
dominated by `mlp.gate.weight`) was structurally incapable of noticing.
Per-tensor reachability is now a precondition of the freeze rather than
something a human might spot in a log.

**Next, per the roadmap** (`docs/superpowers/plans/
2026-09-11-chowder-training-first-9b-a35b.md`): the ≤10B-total/≤3.5B-active
north star belongs to the **9B successor line**, not this 27B conversion —
and it is not reachable by FFN routing alone there either. The measured 9B
always-on floor is 4.578B (4.122B text-only), of which embeddings are
2.034B and vision 0.456B, so the plan needs an **always-on reduction
axis** beside init/granularity/routing. A candidate shape that does fit
(32 layers, backbone 4096→3072, E=16 @ width 768, top-2, 512 shared) was
shape-checked at 6.477B total / 3.306B active — a viable budget, NOT a
trained model. It may simply be wrong for the
program's purpose.

---

## Prior entry (2026-09-10) — protocol-v3 A/B/C tournament,
parent A frozen by explicit decision, first real dense→MoE conversion

**Headline: the sparse program has its first real artifact.** Parent A
(native Qwen3.8-27B control) was converted dense→MoE at E=16, loaded for
real on the RTX 5060 Ti under 4-bit NF4, passed `audit_moe_architecture`
(64/64 layers converted, `num_experts=16`, `num_experts_per_tok=16` —
every expert still fires, exactness-preserving as designed), and generated
coherent real text (`"The capital of France is Paris. The capital of
Germany is Berlin..."`). This is NOT yet a sparse/efficient model — top_k
still equals E, so there is zero compute saving yet. Router healing
(below, "Next steps") is what would actually make it sparse.

- **Protocol v3 tournament (PR #151 census fixes, #152 canonical-rendering
  + behavioral tokenizer gate, #153 this doc's prior sha update — all
  merged, CI green at `beffd23e`):**
  - **A, B, C all COMPLETE under v3** — real 9-dimension/54-item evaluation,
    protocol digest `6a18a4e4f03df8ca...` identical across all three,
    behavioral tokenizer gate passed for B and C against A (proves C's
    genuinely-different serialized tokenizer identity is behaviorally
    irrelevant — exactly what v3 was built to establish). Registry:
    `Chowder-Protected/tournament-v3.registry.db`, runs under
    `Chowder-Protected/runs/v3-20260909/`.
  - **D never produced scored evidence.** Attempt 1 correctly refused by
    the 80 GiB commit-headroom preflight (79.1 GiB measured) — the gate
    working as designed. Attempt 2 crashed on `FileExistsError`: the
    relaunch reused attempt 1's `run_dir`, and `evaluate_parent`'s
    `mkdir(..., exist_ok=False)` collided with its stale `eval-spec.json`.
    Real driver-script bug (scratch script, not `src/chowder/`), not a
    protocol or model defect — D's behavioral tokenizer gate DID pass
    before the crash, so D remains a plausible future candidate, just
    without scored evidence. By explicit user decision, D was not
    relaunched; the tournament proceeded on A/B/C.
  - **v2-vs-v3 A/B diagnostic** (`Chowder-Protected/runs/v3-20260909/
    v2-vs-v3-ab-diagnostic.md`): zero movement anywhere — every dimension
    mean, all 54 per-item scores, and even the raw generated text are
    byte-identical between v2 and v3 for A and B. Root cause confirmed
    (not inferred): B's own `chat_template.jinja` hashes to the exact
    canonical-template digest (`c3cf9e34abf4f9e3...`), so v3's canonical
    rendering was a no-op for A/B specifically. v3's real value is
    admitting C (and would have admitted D) via the behavioral gate, not
    anything it changed for A/B. The prior calibration gap (B 0.500 vs A
    0.833) and the `behavior`-dimension scoring artifact (raw scorer
    literal-matches `"comply"`/`"refuse"` against free-text generation,
    so both score 0.0 regardless of actual refusal quality) both persist
    identically in v2 and v3 — separately diagnosed via
    `behavior-rescore-v4-parent-{a,b,c}.json` (rescoring the SAME
    predictions with a refusal classifier, not re-running inference):
    real separation found there, A=1.0, B=C=0.5.
  - **Real gap found and fixed in the freeze machinery**: `parent_freeze.py`
    (PR #142) predates v3 (PR #152) and its `_tokenizer_gate` still did the
    v2-era byte-identity check — would have wrongly fail-closed C/D despite
    proven behavioral equivalence. Fixed in
    **[PR #154](https://github.com/niko4244/Chowder/pull/154) (open, NOT
    merged)**: a v3-aware branch that accepts behavioral-equivalence
    evidence in place of byte-identity when every role carries it; falls
    back to the original strict check otherwise (v2 packets unaffected).
    19/19 tests green (16 existing + 3 new), lint clean. **Merge this
    before the next time D rejoins a freeze packet** — A/B/C alone didn't
    strictly need it (their tokenizers already matched byte-for-byte
    apart from C, which the fix does cover), but a real four-parter will.
  - **No automatic parent selection exists.** `freeze_selected_parent`
    was run against the real A/B/C packet and correctly raised
    `MissingParentEvidenceError: missing evidence for role(s): ('D',)`
    rather than inventing a winner — exactly the fail-closed behavior the
    mission requires. Full decision state:
    `Chowder-Protected/runs/v3-20260909/four-parent-decision-state.md`.
    Real A/B/C evidence: **A holds the only clear-difference advantage
    anywhere (calibration, over both B and C, who tie each other there)**;
    no dimension favors B or C over A; C is weak-signal softer on
    coding/agentic, weak-signal stronger on self_correction. B's
    historical "primary development parent" label played no role.
  - **Parent A selected by explicit user decision, recorded as a manual
    override — not an automatic tournament outcome.**
    `Chowder-Protected/runs/v3-20260909/parent-freeze-manual-override.json`.
    Reopenable: if D's real evidence is later obtained and shows a
    clear-difference advantage over A, revisit.

- **Phase 6 conversion (ladder item 3 + 4 in `PHASE6_CONVERSION_PLAN.md`,
  both now done with real evidence — the plan doc's own top banner
  claiming "no conversion code exists yet" / "parent A still downloading"
  is STALE, left over from before the module existed; ignore it, the code
  and evidence below are real):**
  - **Profile-only dry run** (no writes) clean for E=8/16/32 on parent A:
    64/64 layers have complete gate/up/down triples, BF16 confirmed
    exactly-scalable, 55.56 GiB source -> ~56.58-56.59 GiB estimated
    output depending on E. `Chowder-Protected/runs/v3-20260909/
    phase6-parent-a-profile-only-dry-run.json`.
  - **Real conversion run, E=16** (user's explicit choice — lands near the
    3-4B active-FFN-param target at top_k=3 of 16):
    `F:\Local Models\HuggingFace\Qwen\Qwen3.8-27B-MoE-E16`, 71 minutes,
    18 shards streamed (never loaded 55 GB into RAM), output
    matching the dry-run estimate to within safetensors header padding
    (+35,528 bytes; actual 56,580,816,384 vs estimated 56,580,780,856 --
    see the corrections in the current-state entry above),
    `scheme_digest c8ad351ab048cc46...`). `conversion.provenance.json` +
    `conversion.manifest.json` + full `local_model_manifest` all written
    into the output dir itself.
  - **Loading smoke test, real and passed**: 4-bit NF4 load on the RTX
    5060 Ti (~10 min, 1107 weight tensors vs 851 dense), `audit_moe_
    architecture` confirmed 64/64 layers converted (0 dense layers left),
    `num_experts=16`/`num_experts_per_tok=16` (top_k=E, exactness-preserving
    as designed — **no compute savings yet, that's router healing's job**),
    real `model.generate()` produced coherent, factually correct text.
    **Genuinely tight resource moment during this load**: commit headroom
    dropped to 0.9 GiB free system-wide mid-load (process RAM peaked
    ~47 GB, well above the dense tournament loads' ~13-14 GB — the extra
    router/expert/shared-expert tensors add real overhead). User explicitly
    chose to let it ride rather than kill it; it completed without
    crashing, headroom recovered to 83+ GiB the moment the process exited.
    **If this load is repeated, expect the same tight window — the gate's
    80 GiB gate is necessary but was not sufficient headroom margin here.**
  - `peak_gpu_mib: 41991.3` (~41 GiB) is NOT a real VRAM measurement — the
    RTX 5060 Ti only has 16.3 GiB. Same already-documented
    Windows-driver VRAM-to-system-RAM paging fallback from a prior
    session's Memory Fabric work (`torch.cuda.max_memory_allocated()`
    reports inflated numbers under this fallback rather than raising a
    clean OOM on this machine) — not a new anomaly, not evidence the
    conversion is wrong.
  - **Not yet done, deliberately out of scope today**: a full-scale
    numerical-exactness comparison (dense parent A vs the E=16 conversion,
    both unquantized, on real inputs) — `conversion_exactness.py`'s
    bit-exactness proof so far only covers a **tiny random composite**
    fixture, never the real 27B checkpoint. The plan doc flags this
    explicitly as "the verification is the deliverable, not the math."
    Real next step if picked up.

- **Phase 4 census: reduced-scope checkpoint (subset40) COMPLETE this
  session, full-scale run still pending.** Do not confuse the two — this
  HANDOFF's 2026-09-08 entry below describes the FULL 1162-passage
  two-half census as "armed, fires when GPU frees"; that full run has
  still never executed. What DID run and complete (2026-09-09/10): the
  reduced 40-passage checkpoint (`REDUCED SCOPE` logged explicitly, "not
  the full-fidelity run"), via `Temp/run_phase4_census.py` +
  `Temp/supervise_phase4_census.py`, output at
  `Chowder-Protected/runs/phase4-census-parent-a-subset40/`. **Negative
  result, honestly recorded**: for E in {8, 16, 32}, every clustering
  strategy tested (contiguous, frequency-stratified, sketch-cluster,
  sketch-cluster+contribution) showed `ratio_adv` ~x1.0-1.2 vs random
  partitioning, all flagged "no signal." Verdict line: "NO material
  advantage over random partitioning; record the negative result and
  retain the mechanical converter." This is WHY today's E=16 conversion
  used the default contiguous `PartitionScheme` rather than an
  activation-derived grouping — the census gave no evidence to justify
  anything else, on this reduced subset. Whether the FULL 1162-passage
  census would reverse this is open; see Next steps.

**Next steps, in rough priority order:**
1. **Merge PR #154** (tokenizer-gate fix) before D (or any future parent)
   re-enters a freeze packet.
2. **Router healing** (Phase 8) -- SUPERSEDED, see the current-state entry:
   measurement shows frozen-expert healing cannot work on a
   complementary-expert partition. Originally described as the
   real unlock — train only `gate.weight` + `shared_expert_gate` on the
   E=16 checkpoint with expert weights frozen, bounded budget, success
   gated on the existing hard regression gate against the protected
   suite. This is what actually reduces `num_experts_per_tok` below 16
   and produces a real compute saving; nothing today reduced compute yet.
3. **Full-scale conversion-exactness check** against the real E=16
   output (not just the tiny fixture) — measure forward-pass numerical
   agreement between dense parent A and the converted model on real
   inputs, unquantized if resources allow.
4. **D's evidence gap remains open and reopenable**: if D's driver-script
   bug gets fixed and a real v3 run completes, redo the A/B/C/D packet
   (now with PR #154 merged) and check whether D changes the calibration
   picture or anything else before treating A's selection as final.
5. **Full 1162-passage Phase 4 census** (not just the 40-passage
   reduced checkpoint) remains unrun — would need to confirm whether the
   negative result holds at full scale before fully retiring the idea of
   activation-derived expert grouping for future conversions (E=8/32 or a
   second model).
6. GPU is free as of this writing — nothing is currently running.

---

## Prior state (2026-09-08, early) — protocol-v2 tournament, C/D acquisition

- `main` = `cf94a30` (everything below plus #139 commit-headroom gate +
  native-crash retry, #140 parent-eval protocol v2: thinking-aware
  final-answer extraction, 256-token budget, protocol-version digest,
  #142 four-parent freeze pipeline + C/D acquisition tooling, #143
  Kaggle-as-qualified-parallel-evaluation-backend for C/D, #144 docs:
  completed protocol-v2 A/B tournament, #145-#147 C/D local decision /
  orchestrators / four-parent consolidation, #148 sparse-research
  foundation, #149 D-acquire crash docs, #150 Phase 4 census prep,
  #151 census batching + 4-bit-dequant fixes, #152 protocol v3 (canonical
  rendering + behavioral tokenizer gate; 3 CI fixups: sys.modules
  transformers fake, integrity-first gate ordering, symmetric
  digest-additive fingerprints) — all post-merge CI green)
- **Track A (real A/B parent tournament) is COMPLETE under protocol v2
  (retry7).** Both parents ran the full 9-dimension / 54-item protected
  suite; see "A/B result" below.
- **C/D execution decision (2026-09-08): the Kaggle T4 preflight was run
  for real and REFUSED** — parent A's measured 16004 MiB peak × 1.10
  margin = 17.19 GiB required vs the 14.8 GiB usable T4 ceiling
  (KAGGLE_T4_USABLE_VRAM_GIB). A single T4 cannot hold this workload
  under the frozen protocol; the tooling refuses by design rather than
  diverge (no budget/quantization/precision changes). Per explicit user
  decision, **parents C and D now run LOCALLY on F:** (336 GiB free —
  both ~52 GiB + the future converted checkpoint fit). See "C/D
  acquisition and evaluation" below.
- **Track E (full recursive-repair acceptance through Unsloth) done for
  real, PR #135**: a real isolated Unsloth environment was provisioned for
  the first time this session (`chowder setup unsloth --root
  C:\Users\nikma\Chowder-Protected\unsloth-real-smoke` — keep reusing this
  location, it's outside any worktree so it survives worktree/branch
  churn; `chowder doctor unsloth` reports every check OK including a real
  4-bit `bitsandbytes.nn.Linear4bit` CUDA forward pass on the RTX 5060 Ti).
  `tests/test_project_runner_repair_unsloth.py` then ran the exact same
  `run_project()` recursive-repair path already proven for Transformers in
  `test_project_runner_repair.py`, changing only `backend.engine='unsloth'`
  in the project config — **no new orchestration code was needed**, because
  `backend_selection.py`'s `create_training_executor` and
  `repair_candidates.py`'s `build_repair_candidate` were already
  engine-neutral (built as part of Tracks B/C/D's own field additions:
  `backend.parent_adapter`, `backend.replay`, `text_field`). Passed for
  real in 125s: baseline trained and evaluated, the initial candidate was
  deterministically rejected (an impossible `minimum_promotion_gain: 2.0`
  gate), a real failure was harvested and clustered, a real repair dataset
  passed contamination audit against the holdout, and a real second
  Unsloth training hop ran — with real evidence
  (`continued_from_parent_adapter: True`, `parent_adapter_sha256` present)
  that it continued from the rejected candidate's *exact* hashed adapter
  weights rather than a fresh-initialized one. Gated behind
  `CHOWDER_REAL_UNSLOTH_SMOKE=1` plus a new optional
  `CHOWDER_REAL_UNSLOTH_ENV_ROOT` env var (points the test at the
  persistent env above instead of pytest's throwaway `tmp_path`, which is
  what made `test_unsloth_peft_real.py`'s equivalent real-smoke test
  impractical to actually run before now — that env var is the fix, kept
  local to the new test rather than touching the older file). Contamination
  coverage under Unsloth needed no separate work either: the audit runs on
  repair-dataset content before `build_repair_candidate` ever branches on
  engine, so it was already backend-neutral.
- **Squash-merge branch-history gotcha, hit twice this session (#131→#132,
  and again for Track D): a feature branch built by `git checkout -b` from
  another *unmerged* feature branch, after that parent branch later gets
  squash-merged, phantom-conflicts against `main` and — worse — its PR's
  CI silently never triggers at all (observed for real, `gh pr checks`
  reports "no checks reported" indefinitely).** Symptom: `gh pr view
  <n> --json mergeable` shows `"CONFLICTING"` even though the real file
  content is compatible. Fix: `git branch -f <name>-v2 origin/main &&
  git checkout <name>-v2 && git cherry-pick <original-commit-sha>` — a
  clean cherry-pick onto current main, verified to trigger CI immediately.
  Close the broken PR, delete its branch, open a fresh PR from the `-v2`
  branch. **Always start a new Unsloth-track branch from a fresh
  `git checkout -b <name> origin/main` (never from another in-flight
  feature branch) to avoid this entirely.**
- **Real A/B parent tournament: still not complete — now blocked on a
  real, confirmed-reproducible system resource constraint, not a code
  defect.** Three real defects found and fixed along the way (each with
  its own regression test where the defect was a real code bug):
  1. `precision="bfloat16"` default (`BaseTextEvalSpec` only accepts
     `{"auto","bf16","fp16","fp32"}`) — fixed, PR #129 (merged).
  2. The driver script's `sys.path.insert()` (controller-process-only)
     didn't propagate to the worker subprocess, which fell back to
     whatever `chowder` `.venv-repro` was editable-installed from (the
     **stale main checkout**, `C:\Users\nikma\Chowder\src`, which predates
     the local-model-source fix and crashed calling
     `try_to_load_from_cache` on a raw filesystem path) — fixed by setting
     `PYTHONPATH` as a real env var in `/tmp/run_tournament_ab.py` (not a
     chowder source bug; this was a scratch-script gotcha, no PR).
  3. **`OSError: The paging file is too small for this operation to
     complete. (os error 1455)`, raised inside `safetensors`' `safe_open`
     while `AutoModelForCausalLM.from_pretrained` loads parent A's first
     shard.** Reproduced identically on **two separate real attempts**
     (retry2 and retry3), each after ~25-30 real minutes of the
     integrity-hashing phase completing successfully first — this is not
     transient noise, it is a real, repeatable failure at the same step.
     Diagnosis (real numbers, not guessed): the page file itself is
     already substantial (`Win32_PageFileUsage.AllocatedBaseSize` ≈
     65,439 MiB ≈ 64 GiB) so "just increase the page file" is not
     obviously the fix; `\Memory\Commit Limit` is ≈127.8 GiB and
     `\Memory\Committed Bytes` was measured at 83.6 GiB, then 89.8 GiB,
     then 97.1 GiB across three checks over roughly an hour — a real,
     **growing** trend, not a one-off spike, on a machine running **571
     processes** at last count (many concurrent Claude/agent sessions and
     Hermes services, confirmed via `Get-CimInstance Win32_Process`). Safe
     mmap'ing an 18-shard/51.75 GiB checkpoint via `safe_open` needs a
     real chunk of committed virtual-memory headroom that this
     increasingly-loaded shared machine may simply not have free at the
     moment of the attempt. **This is a genuine system-resource
     constraint, not something further Chowder code changes can fix** —
     modifying the page file size or killing other processes are both
     system-setting/user-owned actions outside what an agent session
     should do unilaterally. Do not keep blindly retrying without either
     (a) confirming real free commit headroom is meaningfully higher than
     the ~30-40 GiB observed at each failure, or (b) the user's own
     action. Retry script (still valid, just bump the `retryN` output dir
     name to avoid the `run_dir.mkdir(..., exist_ok=False)` collision):
     `/tmp/run_tournament_ab.py`, registry
     `C:\Users\nikma\Chowder-Protected\tournament-ab.registry.db`, log
     `C:\Users\nikma\AppData\Local\Temp\tournament_ab.log`.
  4. **retry4/retry5: a NEW, now CONFIRMED-REPRODUCIBLE crash, distinct
     from #3, and the GPU-contention theory below is REFUTED — read this
     whole item before touching the tournament again.** retry4 got past
     the integrity-hashing phase into real weight loading
     (`Loading weights: 0%|...`) before the worker crashed with
     `exit 3221225477` (`0xC0000005` = `STATUS_ACCESS_VIOLATION`). At the
     time, `nvidia-smi` showed the RTX 5060 Ti at 11.2/16.3 GiB VRAM used
     (Ollama's `llama-server.exe` was an active compute process), so GPU
     contention looked like the explanation and retry4 was **not**
     initially treated as a reproducible Chowder defect. **That theory is
     now refuted**: retry5 was relaunched only after confirming, for
     real, that the GPU was clear (`nvidia-smi`: 527 MiB used / 15.5 GiB
     free, Ollama no longer listed as a compute process) and commit
     charge was healthy (83.7/127.8 GiB, ~44 GiB headroom) — and it
     crashed **identically**, same exit code, same exact point
     (`Loading weights: 0%|          | 0/851 [00:00<?, ?it/s]`), zero
     bytes of additional stderr either time (`worker-stderr.log` in each
     run's parent-a subdirectory has exactly those two lines and nothing
     else — this is a silent native crash, no Python traceback, no CUDA
     error text). A **third, direct, non-tournament reproduction**
     (bypassing `_run_worker`'s subprocess wrapper entirely, with
     `CUDA_LAUNCH_BLOCKING=1` for synchronous CUDA errors, script saved
     durably at
     `C:\Users\nikma\Chowder-Protected\repro_parent_a_load_4bit_segfault.py`
     — rerun with `PYTHONPATH=<worktree>/src python
     repro_parent_a_load_4bit_segfault.py`) reproduced it a third time, again at
     the identical point, confirmed via Bash as a real `Segmentation
     fault` (exit 139) — so this is **not** wrapper-related and **not**
     resource-contention-related; it reproduces 3/3 under materially
     different system conditions. Environment at reproduction: `torch
     2.11.0+cu128`, `transformers 5.16.1`, parent A
     (`F:\Local Models\HuggingFace\Qwen\Qwen3.8-27B`), `quantization_config=
     BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
     bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)`,
     `device_map={"": 0}` — i.e. it crashes inside
     `AutoModelForCausalLM.from_pretrained` right as NF4 4-bit weight
     quantization/loading begins on this exact 18-shard/51.75 GiB
     checkpoint. No corresponding entry appears in the Windows
     Application (WER, event ID 1000) or System (nvlddmkm/display/TDR)
     event logs at either crash timestamp — ruled out a driver-level GPU
     reset. **Leading real hypothesis, not yet confirmed**: a
     bitsandbytes/torch/CUDA version incompatibility specific to 4-bit
     NF4 quantization of a checkpoint this large on this exact hardware
     (RTX 5060 Ti, Blackwell, cu128 torch build) — not something narrowed
     down further yet. **Next steps for a future session, in order**:
     (a) try `quantization="none"` with plain bf16 (no bitsandbytes at
     all) as a differential diagnostic — if that loads cleanly, the fault
     is bitsandbytes-specific, not a generic OOM/driver issue; (b) check
     for a newer/older `bitsandbytes` release with known Blackwell fixes;
     (c) only after (a)/(b) narrow it down, decide whether this becomes a
     real Chowder-level workaround (e.g. an alternate quantization path)
     or stays an upstream-dependency bug to track. Do not blindly retry
     retry6 expecting a different result — this is now a confirmed,
     reproducible defect, not transient contention.
  5. **RESOLVED (2026-09-07): root cause found and gated — it was Windows
     commit exhaustion, not a bitsandbytes/torch/CUDA version defect.**
     Full differential evidence (scripts + JSON results) preserved in
     `C:/Users/nikma/Chowder-Protected/diagnostics/`. The decisive facts:
     - The checkpoint and the bnb kernels are innocent: 8/8 representative
       tensors read cleanly via safetensors alone, and every real
       parent-A tensor shape x {NF4, FP4} x {double-quant on/off} plus a
       size ladder to 680M elements passed 35/35 direct-kernel
       quantize+dequantize round-trips on the RTX 5060 Ti.
     - The load drives Windows commit charge up ~68-71 GiB above its
       launch baseline (instrumented run: 28.2 -> 96.6 GiB of 119 limit).
       When launch headroom is below that requirement, the process dies
       as the silent `STATUS_ACCESS_VIOLATION` — inside a native
       allocation path, so no Python OOM/traceback is ever raised.
     - Crash/success now fully correlates with headroom: 5/5 crashes
       under low headroom (retry4/retry5 at 44 GiB -> early death; two
       instrumented runs at intermediate headroom -> mid-load death at
       conversion ~#186; a controlled stress test holding 45 GiB of
       commit in a side process, leaving 49.4 GiB -> crash at 414 s),
       and 5/5 completions at ~90 GiB headroom — including a bare,
       un-hooked load (675 s, SUCCESS). The earlier per-conversion
       sync/empty-cache "rescues" were confounded by the same
       time-correlated headroom change; no hook is needed.
     - The apparent "crash at 0%" positions in retry4/retry5 were an
       artifact of tqdm's `\r` updates sitting in block-buffered stderr
       when the process died; worker stderr is now launched with
       PYTHONUNBUFFERED=1 so future crash positions are real.
     - Fix (in `parent_tournament.py`): a commit-headroom preflight gate
       before every worker launch (default 80 GiB, env
       `CHOWDER_MIN_COMMIT_HEADROOM_GIB`, measured numbers in the error),
       bounded retry (2) on native-crash exit codes with the gate
       re-checked before each relaunch, and the measured headroom +
       attempt count recorded in each run's evidence. The speculative
       per-conversion hygiene hook was NOT shipped — its mechanism does
       not address the proven cause.
    6. **retry6 result: the load fix worked; the run was then discarded for
       a protocol defect — protocol v2 replaces it.** With the headroom
       gate in place, both parents' integrity verification passed and
       **parent A loaded all 851 weight tensors in 10:51 on the first
       attempt** — the phase that killed retries 1-5. Evaluation ran to
       completion, but every item scored 0.0. The predictions show why:
       Qwen3.8 emits visible chain-of-thought, closes it with `</think>`,
       and then answers correctly (e.g. the tungsten item ends
       `...</think>\n\nW` against expected `w`) — but `max_new_tokens`
       was 64 and `_score` matched the *whole* raw generation against the
       expected value, so every thinking-model item failed. This is a
       protocol defect, not a model result: **all retry6 rows are invalid
       as parent evidence** and are retained on disk only as the negative
       evidence that motivated protocol v2. Fixes (protocol v2, applied
       identically to every parent): (a) worker `_score` now extracts the
       final answer after the last `</think>` (no marker → whole
       prediction; unclosed `<think>` → empty → honest miss), with the
       real retry6 item pinned as a regression test; (b)
       `ParentSuiteSpec.max_new_tokens` default 64 → 256 (observed
       thinking ~40-150 tokens + answer headroom); (c)
       `ParentEvalSpec.protocol_version = "v2"` participates in the
       protocol digest, so v1 and v2 rows can never be compared as
       commensurable. Tournament relaunched as retry7 under v2.

- **C/D acquisition and evaluation (2026-09-08): acquisitions COMPLETE;
  both tournaments RAN and FAILED CLOSED at the tokenizer gate — the
  fail-closed design worked; the protocol is untouched.** Both parents
  were acquired locally to F: with the exact A/B standard (pinned
  revision, full-mode manifest, Phase 11 parameter accounting,
  tokenizer gate vs parent A):
  - C = `OBLITERATUS/Qwen3.8-27B-OBLITERATED` @
    `a58c3b53b3ce71551eafde2ed5ec8df48e0f4ff8` →
    `F:\Local Models\HuggingFace\OBLITERATUS\Qwen3.8-27B-OBLITERATED`
    (70 files; download launched 2026-09-08 ~08:10).
  - D = `DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-    Uncensored-NM-DAU` @ `81c73940f94023f7d64e3ae6abcc653fc837d415` →
    `F:\Local Models\HuggingFace\DavidAU\...` (26 files; download
    launched 2026-09-08 ~08:14). The GGUF variant is not the training
    parent; the pinned Safetensors/Transformers checkpoint is used.
  - The frozen protocol-v2 sequence is applied exactly as retry7:
    integrity verification -> tokenizer gate (A vs C, A vs D) ->
    4-bit/bf16 load -> 9-suite protected evaluation (seed 20260907,
    256-token budget, digest `c5e964df...`). Orchestration scripts are
    one-off Temp files (established pattern): `acquire_parent_c.py`,
    `acquire_parent_d.py`, `run_parent_c.py`, `run_parent_d.py`; both
    parents share `Chowder-Protected\tournament-cd.registry.db` and
    `runs\cd-20260908\` so the #142 four-parent freeze pipeline can
    consume all four parents' evidence from one registry.
  - Kaggle remains qualified-but-unused: the #143 toolkit is tested and
    merged, and the preflight refusal above is the honest, recorded
    first real result from it. A P100-class accelerator or a
    deliberately-justified margin change are the only honest routes to
    re-enable it; neither is needed while C/D run locally.
  - **C's real size is 103.5 GiB, not ~52** (48 root-level safetensors
    shards in two series -- an 18-shard and a 28-shard set -- plus 8
    GGUF files that are NOT downloaded; D is the standard 51.8 GiB /
    12 shards). F: holds both plus the existing A/B and the future
    converted checkpoint (~321 GiB free at acquisition start).
  - **Transport finding (real, diagnosed, fixed): long-lived HF download
    sessions wedge on this network** -- xet stalled at 0 MB/s twice,
    plain HTTP and hf_transfer too, each after ~20-40 min, always
    without erroring (process alive, zero bytes); fresh short-lived
    connections verified healthy throughout. Fix:
    `robust_fetch.py` downloads every file in 512 MiB chunks, each
    chunk a fresh curl connection (`--max-time 900`, per-chunk retry
    with backoff, 6-way parallel, atomic per-shard assembly into the
    target path). Stable for hours at ~4-8 MB/s combined where every
    hub transport died. Corollary guard: both acquire scripts now pass
    `ignore_patterns=["*.gguf", "*.GGUF"]` so snapshot_download can
    never start pulling C's GGUF variants at the acquire stage.
  - **Four-parent consolidation ready and validated (partial packet
    already produced):** `Temp/consolidate_four_parent.py` idempotently
    adopts A/B's experiment + evaluation_run rows verbatim from
    `tournament-retry7.registry.db` into `tournament-cd.registry.db`
    (`_insert_immutable` treats identical replays as no-ops -- registry
    verified clean, 2 experiments + 2 evaluation_runs, no duplicates),
    then runs `parent_freeze.build_selection_packet` over roles
    A/B/C/D with per-role tokenizer evidence measured from the local
    model dirs. First real run: A/B gates all PASS (dimension coverage,
    revision pins, protocol digest `c5e964df...`, suite content,
    tokenizer identity), C/D recorded as `missing_roles`, dimension
    comparisons match the retry7 A/B result (7 ties + calibration
    clear-difference). Packet persisted at
    `Chowder-Protected/runs/four-parent-selection-packet.json`. Both
    tournament runners now refresh the packet automatically after
    their evaluation completes, so whichever of C/D finishes last
    produces the full four-parent packet; only `all_roles_present`
    still fails, and the freeze decision waits for it.
  - **Orchestrators (detached, survive agent-session restarts):**
    `orchestrate_c.py` chains C fetch-complete -> acquire -> protocol-v2
    tournament; `orchestrate_d.py` chains D fetch-complete -> acquire
    (CPU, overlaps C's GPU work) -> **tournament queued behind C's
    chain** (proceeds only when orchestrate_c.log records parent C
    complete, C's chain reports a failure, or no C-chain process is
    alive). Logs: `Temp\orchestrate_{c,d}.log` (+ per-stage .out/.err),
    `Temp\fetch_{c,d}.log`. First-shard assembly verified on both
    parents before the long haul.
  - **Tournament outcome (2026-09-08 ~16:50 C, ~17:14 D): both refused at
    `ensure_parent_tokenizer_compatible` before any GPU work (fail closed,
    pre-load, no scores produced). C: identity `a0f5fa9cdb67f0d1` vs A's
    `3b0d63376fde773a` (same Qwen2Tokenizer class, same vocab 248077). D:
    identity `79ee8b68c2354772`, class `TokenizersBackend` vs A's
    `Qwen2Tokenizer` (same vocab 248077). Registry
    `tournament-cd.registry.db` records both acquisitions + refused runs;
    no prediction files were produced (correct).**
  - **Tokenizer diagnosis (2026-09-09, real behavioral probe on
    public-domain Phase 4 corpus text — never tournament content): C and
    D tokenize IDENTICALLY to A** (17,232-token probe interleaved across
    all ten Gutenberg books, exact ID-sequence match for both). Asset
    diff vs A: C differs only in `tokenizer_config.json` (re-serialized;
    A carries the vision-aware chat template, C a text-only chat
    template); C's `tokenizer.json`/`vocab.json`/`merges.txt` are
    byte-identical to A's. D additionally has a different
    `tokenizer.json` (D-specific serialization) and the
    `TokenizersBackend` class identity. The refusal is **serialization
    hygiene, not behavioral divergence** on this probe.
  - **Why the gate still refuses correctly: the frozen protocol renders
    every suite prompt through each model's OWN chat template
    (`use_chat_template=True`, parent_suite_content.py, part of the
    protocol fingerprint). C's chat template genuinely differs from A's,
    so prompt rendering — not base tokenization — would differ between
    parents; A-vs-C scores would not be protocol-comparable. Resolving
    this honestly requires a protocol v3 (pin ONE canonical chat-template
    rendering for all parents, re-run everyone) — suite-v3 territory, not
    a silent edit; per protocol discipline nothing was changed.**
  - **Decision required (user): (a) freeze protocol v2 as the A/B-only
    result and record C/D as excluded-on-evidence (honest, no protocol
    change), (b) author protocol v3 with a canonical rendering template
    and re-run all four parents (retry7 preserved as v2 evidence), or
    (c) treat C/D as research-only references outside the tournament. No
    unilateral choice was made.**
  - **Decision made (2026-09-09, user): protocol v3.** Authored and
    shipped (#152): `canonical_chat_template.py` embeds the official
    Qwen/Qwen3.8-27B chat template (8,952 chars, digest-pinned
    `c3cf9e34abf4...`, base64-embedded, verify-at-load, fail-closed);
    `ParentEvalSpec` gains digest-additive `canonical_rendering` +
    `canonical_template_sha256` (v2 canonical JSON unchanged — retry7's
    digest `c5e964df...` reproduces byte-for-byte, verified against the
    frozen suites root; v3 digest = `6a18a4e4f03df8ca...`); v3 tokenizer
    gate is BEHAVIORAL (`ensure_parent_tokenizer_behavior_compatible`):
    parents must produce identical token-ID sequences on a pinned probe
    (16 passages from the hashed public-domain Phase 4 corpus
    `a05451e9...`, never tournament content) — serialization may differ,
    behavior may not. Worker renders v3 prompts through the canonical
    template only (`render_canonical`); the per-parent own-template path
    stays for v2 replay. `run_tournament(protocol_version=...)` selects
    the generation; default stays v2. Also fixed: `ParentEvalSpec.from_dict`
    dropped `protocol_version` on round-trip.
  - **Phase 4 census launch evidence (2026-09-09): the reduced-scope
    parent-A census run (subset40) crashed with the documented signature —
    silent native access violation, rc=3221225477, during 4-bit weight
    staging at 70.4 GiB measured commit headroom** (inside the ~68-71 GiB
    staging band from the PR #139 evidence; the runner had skipped the
    tournament's preflight). Fix + hardening: the runner now runs the
    commit-headroom preflight (correct 64-byte MEMORYSTATUSEX struct —
    the first probe used a 56-byte struct and silently returned nan) and
    a supervisor (`Temp/supervise_phase4_census.py`) relaunches it with
    bounded native-crash retries and the SAME 80 GiB headroom gate as the
    tournament (no override; nan = fail closed). The supervisor waits
    (5-min polls) until commit headroom clears 80 GiB, then fires the
    census automatically. Current machine commit usage (~48 GiB held by
    other apps) keeps headroom at ~71 GiB, so the run is legitimately
    gated until apps are closed or the pagefile grows — recorded, not
    bypassed.

- **A/B result (retry7, protocol v2) — both parents complete and persisted.
  `C:\Users\nikma\Chowder-Protected\tournament-retry7.registry.db`
  (fresh DB from the retry7 resume path; A adopted in, B run to completion),
  prediction/evidence per parent under
  `runs/ab-20260907-retry7/parent-{a,b}-*`. Protocol digest IDENTICAL for
  both parents (`c5e964df3a2a7917...`); identical protocol (4-bit / bf16,
  256-token budget, same suite content).**

  | dimension | A | B | verdict |
  |---|---|---|---|
  | reasoning | 6/6 | 6/6 | tie |
  | coding | 5/6 | 5/6 | tie |
  | knowledge | 5/6 | 5/6 | tie |
  | instruction-following | 6/6 | 6/6 | tie |
  | self-correction | 4/6 | 4/6 | tie |
  | agentic | 6/6 | 6/6 | tie |
  | thinking-efficiency | 5/6 | 5/6 | tie |
  | calibration | 5/6 | 3/6 | **CLEAR (A)** |
  | behavior/refusal | 0/6 | 0/6 | artifact* |

  - Capability aggregate (8 dims, 48 items): **A 42/48 (0.875), B 40/48
    (0.833)**. Only calibration differs by ≥2 items — the sole clear
    difference, in the official parent's favor. Every other capability
    dimension is an exact tie.
  - *behavior/refusal 0/6 on both is a **labeling protocol artifact, not a
    model result**: the expected values are the class labels `comply` /
    `refuse`, but Qwen3.8 answers with the actual poem / refusal text, so
    exact-match scoring against the literal label can never succeed. The
    model's behavior itself is correct on all 6 items for both parents.
    The scorer is deliberately unchanged mid-tournament per the frozen
    protocol; this affects both parents identically, so the A/B comparison
    stays valid. A post-tournament scoring fix (answer-class mapping) would
    require a protocol bump and a full re-run to be usable as evidence.
  - **No parent is selected from A/B alone** (program policy: final
    selection is the four-parent tournament). OrcaRouter (B) shows no
    capability advantage over the official control (A) on this suite and
    is measurably worse on calibration.
  - Peak VRAM: B `peak_gpu_mib_sampled` = 15969 MiB. A's peak (16004 MiB,
    measured during the load, recorded in the Kaggle-C/D doc) was not
    carried into the fresh registry during adoption (registry shows 0) —
    the 16 GiB-class headroom finding stands and drives the Kaggle C/D
    VRAM preflight.
- **Track B (Unsloth chat-format parity) done, merged, PR #130**: the
  isolated Unsloth worker previously supported text-format datasets only.
  Now `unsloth_peft.py` (controller-side) pre-renders every chat row
  through the exact shared contract `transformers_worker.py` uses
  (`training_data._validate_chat_messages`/`_build_chat_example`) into a
  content-addressed, pretokenized JSONL handoff file *before* the
  isolated worker ever starts — the worker's chat path is just "load
  three already-tokenized columns," with zero chat-template/masking
  logic of its own, so there is no code path where Unsloth's semantics
  could drift from Transformers'. 15 tests, including the exact
  regression cases the Qwen3.8 program directive named (multi-turn,
  system prompt, multiple assistant turns, empty-assistant-content — a
  real finding: still produces real turn-marker labels, not "nothing to
  train on" — Unicode, truncation before/inside the assistant response,
  malformed role, no assistant turn, long conversation).
- **Track C (Unsloth parent-adapter continuation) done, merged, PR #132**:
  `spec.parent_adapter` loads via plain PEFT's `PeftModel.from_pretrained`
  directly onto the Unsloth-loaded base model (an Unsloth model is a real
  transformers-compatible model underneath) instead of a fresh
  `get_peft_model` adapter — mirrors `transformers_worker.py`'s identical
  continuation path. `parent_adapter_sha256` is a real bound-input (resume
  against a different parent adapter fails closed). A real bug this PR's
  own parity test caught before it ever reached hardware: the isolated
  worker's local `sha256_directory` mirror (it cannot import
  `chowder.provenance` in the isolated env) was missing a trailing
  `digest.update(b"\0")` separator the real implementation has — would
  have made the worker's own adapter re-verification silently disagree
  with the controller's on every real run.
- **Track D (Unsloth replay/rehearsal) done, merged, PR #133** (chat + text
  format both — chat merges replay in the controller before
  tokenization inside `_materialize_pretokenized_chat_dataset`; text
  merges it inside the isolated worker via a new pure
  `_load_text_dataset_with_replay(dataset, spec)` helper). **A real,
  CI-only test failure and its fix are worth reading before touching this
  area again**: an earlier version of the text-format test tried to
  monkeypatch `transformers.Trainer` to drive `unsloth_worker.train()` end
  to end. It passed in an isolated single-file local run but failed for
  real in the full CI suite. Root cause, confirmed for real (not
  guessed): `transformers`' top-level package is a `_LazyModule` whose
  `__getattr__` caches each name's *first* real resolution directly into
  the module's own `__dict__`. Once any *other* real-ML test in the same
  process had already touched `transformers.Trainer` first (many do,
  across 1200+ tests), a later `from transformers import Trainer` found
  that cached real class in `__dict__` directly and never called
  `__getattr__` again — so patching `transformers.trainer.Trainer`, and
  even directly overwriting `transformers.__dict__['Trainer']`, both
  confirmed ineffective once that caching had already happened. Which
  behavior you observed depended on unrelated test execution order — not
  a foundation to build a test on. Fix: extracted the real row-mixing
  logic into `_load_text_dataset_with_replay`, a pure function over real
  `datasets` objects with zero `torch`/`unsloth`/`transformers`/`Trainer`
  involvement, and test that directly. **If you ever need to fake a
  `transformers` class again, do not trust that patching it once in
  isolation means it will hold in a full suite run — verify with
  `CHOWDER_REAL_ML_SMOKE=1 pytest tests/ -q` (the whole suite, not just
  your file) before considering it done.**
- **Track F (Qwen3.8 campaign manifest) implemented, PR open**:
  `src/chowder/qwen38_campaign.py` adds `Qwen38CampaignManifest` — binds
  the primary parent (orcarouter, pinned), native control (official Qwen),
  both comparison parents, the frozen protected-suite version (`v1`,
  digest `7946d8c9…`, cross-checked in a real machine-local test against
  the actual file at `C:\Users\nikma\Chowder-Protected\suites\v1\manifest.json`),
  lineage policy (native-required, distillation hard-rejected), the sparse
  target range, training engine, a real `RecursiveRepairPolicy` (reusing
  the existing type, not reinventing it — `max_depth`,
  `min_score_improvement`, `max_failure_signature_occurrences`,
  `replay_ratio` already covered every knob the program directive named),
  and promotion rules, into one object with a real `manifest_sha256()`
  content hash. `__post_init__` fails closed on exactly the failure mode
  the directive warned about: a manifest with an empty repair corpus or
  empty repair-variant list, or `require_protocol_match=False`, is
  rejected outright — a campaign literally cannot be constructed in a
  shape that would silently degrade into train→evaluate→stop.
  `default_qwen38_campaign_manifest()` is the real, concrete factory
  (exact pinned revisions and suite digest, not placeholders) but takes
  `repair_corpus_files`/`repair_variant_names`/`gpu_hour_budget` as
  required keyword arguments — there is no safe default for those, by
  design. 25 tests, all passing for real (`pytest
  tests/test_qwen38_campaign.py`), including a hash-changes-with-every-real-
  input-change sweep and every fail-closed rule. This module does not
  touch contamination auditing (already backend/campaign-neutral — see
  Track E) or replay (already a real `RecursiveRepairPolicy` field); it
  only binds identities that had no home before. The real A/B parent
  tournament (Track A, still blocked on the page-file constraint above)
  can and should proceed independently once real headroom is available —
  its infrastructure has been merged since PR #128 and needs no Track
  E/F work first.
- Recent merges, prior session: #109 (`training_engine` evidence field),
  #110 (censored-outcome view `censored_outcomes.py`), #111 (Teacher
  Fabric Slice A: `teacher_fabric.py` + `docs/TEACHER_FABRIC.md`),
  #112 (dataset identity/scale + accelerator context in
  `intervention_outcomes.py` — closed the Priority-6 context-gap item),
  #113–#128 (Qwen3.8 program retarget, parent A/B acquisition, protected
  nine-dimension suite content, campaign scoreboard + Fable reference,
  Phase 6 dense→MoE converter, Phase 11 parameter accounting, the parent
  tournament orchestrator itself — see `docs/ROADMAP.md`'s "Qwen3.8
  Native Sparse Program" entry for the full, current, evidence-cited
  state; this doc does not restate it).
- **Model program retarget (2026-09-06):** the primary model target is
  now the Qwen3.8 Native Sparse Program
  (`docs/QWEN38_SPARSE_PROGRAM.md`). Read that doc before any Qwen
  work: it pins all four parent revisions (A control `1d4bf0f2...`,
  B primary `404ea47a...`, C comparison `a58c3b53...`, D comparison
  `81c73940...`, D resolved from the DavidAU GGUF card's base_model),
  records the real architecture audit (all dense `qwen3_5`-family
  Qwen3_5ForConditionalGeneration, 64L/5120h, 15 MTP tensors + 333
  vision tensors preserved in every readable parent), and the blockers:
  **B (orcarouter) is gated — 401 without authenticated access; no
  weights are cached (~55 GB each, ~220 GB for all four vs ~239 GB
  fragmented free); the protected 9-dimension evaluation suite does not
  exist yet.** D's tokenizer class differs (TokenizersBackend vs
  Qwen2Tokenizer) — compare tokenizer identity hashes, never class
  names, and token-aligned signals against D fail closed. Milestone-1
  checklist in the doc is the honest gate for "underway" claims.
- Priority 6 evidence foundation: the production incident-persistence
  caller is DONE (`ExperimentCycleRunner._persist_executor_analysis`
  writes every non-cancelled crash's analysis to `execution_incidents`
  after the failure settles; persistence failures become diagnostics).
  Remaining, per ROADMAP's own list: the chronological backtest validator
  vs UCB1 (zero-hard-gate-violations check required), the EI/GPU-hour-
  aware policy itself, and the cross-model transfer mechanism.
- Teacher Fabric: Slices A–B done; Slices C–J not started. Slice B
  (`src/chowder/teacher_signal_store.py`, `tests/test_teacher_signal_store.py`,
  34 tests) implemented the decisions the orientation had locked in:
  migration 4 (`teacher_signals` append-only ledger, `database.py` now at
  `CURRENT_SCHEMA_VERSION = 4`), content-addressed payloads, atomic writes
  + interrupted-write recovery, verified-or-absent reads, dedup over
  `(request_digest, payload_file_sha256)`, required no-default
  `local_cache_max_bytes` with measured `disk_bytes()`. Two refinements
  the tests forced beyond the orientation decisions: (1) the ledger
  append is *evidence-idempotent* — re-acquiring identical evidence after
  cache eviction replays idempotently (`stored_at` stays first-acquisition;
  genuine divergence raises `RegistryInvariantError`); (2) the store's
  payload-file hash is named `payload_file_sha256`, deliberately distinct
  from Slice A's `TeacherSignalArtifact.payload_content_sha256`
  (signal-payload digest) — different identities must not share a name.
  `canonical_payload()` now carries `peak_vram_gb_by_accelerator` so
  GPU-backed artifacts round-trip losslessly (no compatibility surface:
  nothing persisted artifacts before Slice B). Cache-lookups skip corrupt
  entries; `load` fails loudly. Eviction is explicit `discard` only —
  no silent policy (that is Slice D's decision). Next slice: C
  (`teacher_blackbox.py`, Regression Surgeon integration) per the file
  plan; §16.1's hit-rate experiment remains unruns until real queries
  exist.
- The expected-improvement selector (626-line module + 38 tests, honest
  "alternative selector, not shown to beat UCB1" status) is rescued on
  pushed branch `claude/expected-improvement-rescue` (`50170bd`, based on
  main). Merging it is a separate decision from the rescue; its backtest
  harness (`backtest_selectors`) is the natural base for the roadmap's
  held-out validator.

## Sparse-architecture research track (TurboSparse/PowerInfer) — 2026-09-08

Research directive: investigate activation-informed sparse architecture
WITHOUT touching the live parent tournament or frozen protocol. Landed
on this branch (all tested, full suite green, ruff clean):

1. **Research note** `docs/TURBOSPARSE_POWERINFER_RESEARCH.md` — grounded
   in both papers (read directly, not summaries): dReLU needs ~150B-token
   continued pretraining (nothing transfers free at conversion time);
   PowerInfer's CPU-direct-beats-PCIe-below-batch-32 law makes offloading
   a measured decision, not an assumption; transfer/inference-only/
   retraining/conflict/speculative classifications recorded.
2. **Phase 3 census** `src/chowder/activation_census.py` — forward-hook
   only, never modifies the model; dReLU-counterfactual activity
   `(gate>0)&(up>0)` from captured pre-activations (the initial
   `gated != 0` inference was a real bug — SwiGLU output is never exactly
   zero — fixed and regression-pinned); frequency/magnitude/contribution,
   Gini, exact hot-set co-occurrence + 256-dim JL sketch, mark_split
   held-out halves, concurrent-census guard, atomic profile artifacts.
3. **Phase 4 structure evaluation** `src/chowder/activation_experiments.py`
   — contiguous/random/frequency/sketch-cluster/sketch+contribution
   groupings; held-out half-B metrics; 3-part verdict (absolute held-out
   ratio >= 2.0 AND split-half stability >= 0.90 AND >= 1.10x random
   null). Planted-structure fixture is deliberately NON-contiguous
   (round-robin) so the mechanical baseline cannot trivially equal it.
   The Phase 4 checkpoint (real parent) has NOT run yet — GPU belongs to
   the tournament first.
4. **Phase 9 hierarchy** `src/chowder/sparse_accounting.py` —
   total = always-on + dense/shared + routed-active (top_k/E) ->
   neuron-active = routed-active x (1 - MEASURED sparsity), fail-closed
   on census evidence (digest + per-layer sparsities all-or-nothing),
   explicit definition ids for cross-paper normalization.
5. **Base-module defect fixed** `parameter_accounting.py`: the Phase 11
   active formula `total - routed - router` excluded the routed top-k
   share that IS computed every token and subtracted the always-on
   router. Now `total - routed x (1 - top_k/num_experts)` (exact integer
   division with a divisibility gate). Pinned test updated; every sparse
   A-label in future conversions is corrected by this.

**Next checkpoints:** (a) C/D tournaments complete -> four-parent packet
(consolidation machinery already validated, auto-refresh wired);
(b) Phase 4 real-parent census run when the GPU frees — calibration
corpus must NOT be protected tournament content.

---

## C/D acquisition: crash + resume (2026-09-08 13:38)

- **D fetch COMPLETE** (chunked-curl transport, `ALL FILES PRESENT` 12:48).
  **C fetch in progress** (22/62 files, second shard series; ~2-4h left).
- **D acquire crashed** at the tokenizer-gate step: my acquire scripts
  passed `tokenizer_evidence` directly as `measure_tokenizer_fn`, but
  `acquire_parent` invokes that callable with a bare destination `Path`
  while `tokenizer_evidence` expects a `LocalParent`
  (`AttributeError: 'WindowsPath' object has no attribute 'local_path'`).
  Fix: adapter lambda constructing `LocalParent(label, revision,
  local_path, manifest_path)` from pin + destination. Patched BOTH
  acquire scripts (C would have crashed identically at its gate step).
- **Manifest + verification of D completed BEFORE the crash point** (the
  crash was post-manifest), so the retry took the fast
  `check_already_acquired` path -- re-hashing 52 GiB was avoided.
- The crash also killed the D orchestrator; replacement
  `orchestrate_d_resume.py` waits for `ACQUIRE DONE` from the running
  acquire retry, then queues D's tournament behind C on the GPU
  (identical c_holds_gpu fallback logic), then runs `run_parent_d.py`.
- **Research track merged**: PR #148 (`3fc077f`) -- census, structure
  evaluation, hierarchical accounting, Phase 11 active-formula fix, all
  6/6 CI green. Live tournament untouched.

---

## Phase 4 census prepared + D acquired (2026-09-08 14:20)

**Phase 4 real-parent census (armed, fires when the GPU frees):**
- **Calibration corpus built + hashed**: 10 public-domain Gutenberg books
  (narrative/gothic/detective/science/philosophy/economics/political/
  translated/nonsense/dialect), 1162 interleaved passages (~628k tokens),
  round-robin so the midpoint split keeps every domain in both halves.
  Digest `a05451e901d819a5...`; manifest with per-source sha256 at
  `Chowder-Protected/calibration/phase4-parent-a/corpus_manifest.json`.
  NOT protected tournament content.
- **Census scaled to real parents** (I=17408 x 64 layers would have been
  ~91 GB inline): sketch accumulators moved to torch float64 tensors
  (bit-identical round-5 output, vectorized), per-token sets now a
  50k-row reservoir (10% sampling), co-occurrence capped to top-256
  hottest neurons (bounded tables), sketches >4096 intermediate written
  as verified float64 `.npy` sidecars; numpy clustering path (farthest-
  point + argmax, sidecar digest-verified) for large-I layers.
- **Runner** `Temp/run_phase4_census.py`: corpus hash verified BEFORE any
  GPU work; parent A loaded tournament-identical (4-bit NF4 double-quant,
  bf16 compute, device_map cuda:0); census consumes half A ->
  mark_split -> half B; artifacts (profile + sidecars + verdicts at
  E in {8,16,32}) to `Chowder-Protected/runs/phase4-census-parent-a/`.
- **Watcher** `Temp/watch_gpu_phase4.py` armed (PID detached): fires the
  census only when C AND D chains are complete AND no chain process is
  alive, with an immediate pre-launch re-check. Refuses on corpus digest
  mismatch. Log: `Temp/watch_gpu_phase4.log`.

**Parent D acquisition COMPLETE (with two material findings):**
- Manifest `dcfa6c63d407dc9f...` (51.75 GiB) verified; acquisition
  standard identical to A/B (full-mode manifest, accounting, gate).
- **Finding 1 -- D is DENSE**: 27,781,427,952 total = active parameters,
  no routed experts, no top-k. "TURBO" refers to inference optimization,
  not sparsity; D contributes no MoE-sparsity comparison to the
  tournament and its effective-active number is simply its total.
- **Finding 2 -- D's tokenizer gate FAILS (by design)**: class
  `TokenizersBackend` vs A's `Qwen2Tokenizer` (same vocab 248077 but
  different serialized-asset identity `79ee8b68...` vs `3b0d6337...`).
  `ensure_parent_tokenizer_compatible` fail-closes: direct A-vs-D score
  comparison is refused by the machinery, exactly as the protocol
  requires. D's tournament (queued behind C) will fail fast at the gate,
  pre-GPU; its evidence rows record the refusal as the result.
- Bug fixed in passing: acquire scripts logged
  `TokenizerGateResult.passed` (attr is `.compatible`); the crash was
  post-acquisition, so D needed no re-hash. C's script fixed too.

---

## Environment facts (not written anywhere else in the repo)

- Primary working directory:
  `C:\Users\nikma\Chowder\.claude\worktrees\claude-moe-instrument` (a git
  worktree; keep all work here).
- Test runner: `../../../.venv-repro/Scripts/python.exe -m pytest` — never
  bare `pytest`. CLI invocations need `PYTHONPATH=src` (this worktree's)
  because sys.path insertion does not propagate to subprocess workers.
- Tooling gotcha: tools that address files cannot reach paths under
  `.claude\worktrees\` (dot-directory). `read_files`/`str_replace` fail
  there; use `write_file` with the full path, or terminal reads
  (`sed -n`), or a python heredoc for in-place multi-edit with assertions.
- Real-hardware tests: `CHOWDER_REAL_ML_SMOKE=1` etc.; torch imported
  lazily (CI base jobs install only `chowder[dev]`). Local GPU: shared
  RTX 5060 Ti — check free VRAM before real-CUDA runs.
- CI: 5 fast jobs + a ~9-minute "real transformers peft cpu smoke" job.
  Watch with `gh pr checks <n> --watch`; merge only when all green, via
  `gh pr merge <n> --squash --delete-branch` — never `--admin`, never
  bypass branch protection. Confirm post-merge CI on `main` before
  reporting a session done.
- Editing files under this worktree: `read_files`/`str_replace` cannot
  address dot-directory paths; use `write_file` with the full path, or a
  temp edit script (`_slice_b_*.py` pattern: assert every anchor, run,
  delete) for in-place multi-edits. Bash heredocs get CRLF-mangled in
  transit here — prefer the temp-script route for anything multiline.
- Test count after the dense→MoE conversion slice: 1145 passed / 76
  skipped on this worktree's `main` (was 1124/71 after Phase 11
  accounting, 1082/71 after the parent-eval harness slice, 1016/71
  after incident persistence). The +5 gated skips are
  `tests/test_conversion_exactness.py` (run locally with
  `CHOWDER_REAL_ML_SMOKE=1`; all 5 pass on this box, including the
  real-parent-A stage-2 test).
- Parent A acquisition (**complete and verified**, 2026-09-06):
  `Qwen/Qwen3.8-27B` @ pin `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
  fully downloaded to `F:\Local Models\HuggingFace\Qwen\Qwen3.8-27B`
  and verified by a **full-mode manifest**
  (`d382d54f159f7b6c6b03afac88f7f445d03385684f77ae159fa5db07d7533f59`,
  18/18 shards hashed, 51.75 GiB, verification clean; signed manifest
  JSON at `F:\Local Models\HuggingFace\Qwen\Qwen3.8-27B.manifest.json`).
  Reference it as a local model source by that directory path
  (LOCAL_MODELS.md policy). If the directory ever changes, the manifest
  verifier will catch it — re-verify before trusting a new experiment.
  Parents B/C/D are still not acquired.
- Phase 11 accounting (PR #123): `src/chowder/parameter_accounting.py`
  measures model directories from safetensors headers (stdlib-only).
  `chowder moe account-parameters --model <dir> --output <json>` runs it
  from the CLI (writes hash-recorded evidence, prints a JSON summary;
  dense models honestly report the absent a-label with the module's
  reason). Run it on any converted checkpoint before claiming
  active-parameter numbers; evidence JSON for parent A lives beside the
  model
  (`Qwen3.8-27B.accounting.json`). Measured parent A truth: 27.78B
  total; dense floor 9.78B active/token (10.21B with MTP).
- Phase 6 conversion implemented (stages 1–2 of the plan's validation
  ladder): `src/chowder/dense_to_moe.py` (stdlib-only byte-surgery
  converter; multi-shard MLP triples — parent A layer 15 straddles
  shards, 63/64 co-locate) and `src/chowder/conversion_exactness.py`
  (torch-gated harness). Two plan claims were disproven by
  implementation and are recorded as errata in
  docs/PHASE6_CONVERSION_PLAN.md: top-1 rungs are NOT init-exact
  (top-k = E is required with the stock router), and only down_proj may
  carry the ×E scaling (silu is not positively homogeneous). Measured
  init-forward deviation: f32 max_abs 1.341e-07, bf16 1.953e-03, with
  bitwise dense recovery and exactly uniform routers in both. The full
  parent-A conversion (stage 3/4, ~52 GiB output) has NOT run — that is
  a disk-acquisition decision.
- Program state (2026-09-06): the protected nine-dimension suite
  content **exists** — `src/chowder/parent_suite_content.py` (54
  original hand-authored items, 6 per dimension, deterministic
  materialization to datasets + hash-only fingerprint indexes via the
  canonical `parent_eval.build_protected_suite_dir` path, root-free
  byte-identical manifest; verified the Contamination Guard catches a
  verbatim protected prompt). Parent-eval harness:
  `src/chowder/parent_eval.py`. Parent B
  (`orcarouter/Qwen3.8-27B-Uncensored`, pin `404ea47a`) downloaded and
  **fully verified** at `G:\Local Models\HuggingFace\orcarouter\
  Qwen3.8-27B-Uncensored`: all 18 shards' sha256 match the Hub's LFS
  digests at the pin, all small files byte-compare, zero divergence
  (`Qwen3.8-27B-Uncensored.verification.json` beside the dir; full-mode
  manifest sha `fab432f1…`, 18/18 shards hashed). Phase 11 accounting
  measured from its real headers: **27,781,427,952 total parameters /
  1199 tensors — byte-identical census to parent A** (same architecture,
  different weights), honest dense no-a-label. The protected suite v1 is
  frozen at `C:\Users\nikma\Chowder-Protected\suites\v1` (manifest
  sha `7946d8c9…`; never mutate — a content change is suite v2). The
  public-benchmark campaign scoreboard (`src/chowder/campaign_scoreboard.py`,
  historical targets MMLU>0.90 / GSM8K>0.90 / HumanEval>0.60 / MATH>0.40,
  signed digests, Fable standing reference with parity gated on full
  measurement) is implemented with tests. Next executable steps: the
  Phase-4 A/B tournament run (suite materialized, both parents trusted
  locally), then C/D acquisition.
- Commit messages end `Co-Authored-By: Claude Sonnet 5
  <noreply@anthropic.com>`; PR descriptions end with the Claude Code
  attribution line; branch naming `claude/<slug>`; one focused PR per
  slice.

## Known hazards / local-only state

- The **main clone** (`C:\Users\nikma\Chowder`, not the worktree) has a
  stale working tree from the stride-fix investigation:
  `src/chowder/backends/transformers_worker.py` is a pre-#96 copy, and
  untracked `activation_offload_hooks.py` +
  `docs/ACTIVATION_OFFLOAD_STRIDE_FIX.md` mirror merged work (PR #92).
  Unreconciled on purpose — multiple agent sessions share that checkout;
  fixing it needs the user's go-ahead.
- Three agent worktrees under `.claude/worktrees/` besides the primary
  one; `agent-ae537f2ed08ff8828` (on `feature/intervention-outcomes`)
  has unaudited dirty state — same rescue-before-prune caution as the EI
  selector needed.
- `agent-a66d39fd3e0ec3607`'s untracked EI files are now safely on the
  pushed rescue branch, so that worktree is prunable.

## Update rule

When you finish a session that changed `main`: update the Current state
section (replace, do not append — old state is in git history), keep
hazard list accurate, and land the update through the same PR/CI
discipline as code. If you did not merge anything, still update this doc
if environment facts or hazards changed.
