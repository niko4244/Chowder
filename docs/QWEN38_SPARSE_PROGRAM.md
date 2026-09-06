# Chowder Qwen3.8 Native Sparse Program

**Status: program defined (this document); all four revisions pinned; parent B's gate cleared and full architecture audit recorded; protected nine-dimension evaluation harness implemented (`src/chowder/parent_eval.py`); parent A (control) cached at its pin with a verified full-mode content manifest and **Phase 11 accounting measured from its real tensor headers: 27,781,427,952 total parameters (27.78B), dense floor 9.78B active/token**; Phase 6 conversion plan generated (`docs/PHASE6_CONVERSION_PLAN.md`, PR #120); Phase 11 accounting machinery implemented (`src/chowder/parameter_accounting.py`); no protected suite content authored; no evaluation run; no transformation executed. Nothing here may be read as "the sparse-model project is underway" — see the milestone checklist at the end.**

This document retargets Chowder's primary model research from the prior
Qwen3.6-35B-A3B commissioning branch to a **native-Qwen3.8-derived
sparse/MoE program**. The prior 8B QLoRA campaign and the
`docs/MOE_DOWNSIZING.md` Qwen3.6 program remain in the repository as
historical evidence — nothing in them is deleted or rewritten; they
proved Unsloth setup, 4-bit QLoRA, checkpoint/resume, independent
evaluation, real CUDA operation, and the MoE audit/instrumentation
machinery this program will reuse.

## Program statement

Chowder's primary model research target is a directly
native-Qwen3.8-derived uncensored sparse language model. Development
begins from `orcarouter/Qwen3.8-27B-Uncensored`, with official
Qwen3.8-27B as the untouched control and OBLITERATUS/DavidAU variants as
comparison parents. The long-term target is approximately 3–4B **active
parameters per token** without distilling Qwen3.8 into another
architecture, while preserving as much reasoning, coding, knowledge,
calibration, agentic performance and self-correction capability as
empirical evidence allows. Every architecture and training intervention
remains subject to Chowder's independent evaluation, provenance,
regression and promotion gates.

Shorthand: `Chowder-Qwen3.8-A4B`. **A3B/A4B always means active
parameters per token, never total stored parameters.** Both are tracked
separately (Phase 11 accounting); a model is not labeled "A4B" unless
measured routing geometry supports the claim.

## Phase 11 accounting of the cached control (measured 2026-09-06)

`src/chowder/parameter_accounting.py` (with tests) accounts model
directories from real safetensors headers — stdlib-only, no torch, no
safetensors import, cross-checked against the shard index, failing closed
on unknown dtypes, duplicate tensors, index/shard mismatch, and (for
sparse models) missing `num_experts_per_tok`. The Phase 11 rule is
mechanical there: `a_label()` refuses to exist without measured routing
geometry. Parent A's measured split (evidence JSON beside the model
directory):

| Category | Parameters | Tensors | GiB |
|---|---|---|---|
| Total | **27,781,427,952** | 1199 | 51.75 |
| embedding | 2,542,796,800 | 2 | 4.74 |
| attention + GatedDeltaNet | 7,239,780,864 | 528 | 13.49 |
| dense FFN | 17,112,760,320 | 192 | 31.88 |
| MTP | 424,699,392 | 15 | 0.79 |
| vision | 460,730,096 | 333 | 0.86 |
| layernorms | 660,480 | 129 | 0.00 |

Dense model, so active/token = total (the definition string is carried on
the accounting object). No A-label exists for A — correctly, because
there is no routing geometry to measure. **Correction this measurement
forces on the Phase 6 plan's estimates:** the true dense floor is
**9.78B active/token** (attention+DeltaNet+embeddings+norms), not the
~10.55B the plan estimated by hand — and 10.21B with the MTP head. The
routed share of the FFN at conversion remains 17,112,760,320 parameters.

## Target definition

```yaml
program: chowder-qwen3.8-native-sparse
primary_parent: orcarouter/Qwen3.8-27B-Uncensored   # GATED — see blockers
native_control: Qwen/Qwen3.8-27B
comparison_parents:
  - OBLITERATUS/Qwen3.8-27B-OBLITERATED
  - DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU
lineage_policy:
  native_qwen3_8_required: true
  distillation_parent_allowed: false
target:
  architecture: sparse_moe
  desired_active_parameters_b: "3-4"
```

`lineage_policy` is a hard rule, not a preference: the primary lineage is
native Qwen3.8 weights → pruning / expert partitioning / routing /
sparse upcycling / channel reduction / structured weight surgery /
low-rank factorization / layer removal / continued training / SFT /
preference-repair. A distillation construct (teacher → unrelated/smaller
student initialization → imitation) may never be the model's primary
lineage. Teacher-generated signal (Teacher Fabric, Slices A–B already
landed) may repair or improve the native descendant later; it never
redefines what the model descends from.

## Parent manifest — pinned revisions (Phase 2)

Resolved from the Hugging Face Hub API on 2026-09-06. Never run any
tournament step against moving `main` revisions; every command that
touches a parent pins the revision recorded here.

| Branch | Repo | Pinned revision (sha) | Role |
|---|---|---|---|
| A | `Qwen/Qwen3.8-27B` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | untouched native control |
| B | `orcarouter/Qwen3.8-27B-Uncensored` | `404ea47aaa5d8a8b00049c9e9750089aca011ab2` | **PRIMARY development parent** |
| C | `OBLITERATUS/Qwen3.8-27B-OBLITERATED` | `a58c3b53b3ce71551eafde2ed5ec8df48e0f4ff8` | aggressive-abliteration comparison |
| D | `DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU` | `81c73940f94023f7d64e3ae6abcc653fc837d415` | heavily-optimized comparison |

**Comparison C resolution record (rule: "the GGUF is not the training
parent").** The user-named repo
`DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NEO-CODER-MAX-MTP-GGUF`
(sha `6408ab122688c54ba5b7cea19084307ef153410f`) contains only GGUF
quantizations. Its model card's `base_model` metadata names the
full-precision source this program pins as branch D above (13
safetensors shards, config + tokenizer present, zero GGUF files at that
revision). Resolution trail: GGUF card → `base_model` →
Transformers/Safetensors repo, verified by direct file listing.

## Architecture audit (Phase 3 evidence so far)

From real `config.json` + `safetensors.index.json` reads at the pinned
revisions (full file reads for A/C/D; authenticated file reads for B
after the auto-gate accepted the account's terms — B's revision
`404ea47a…` verified byte-identical to the pin):

| Evidence | A (control) | C (OBLITERATUS) | D (DavidAU parent) | B (primary) |
|---|---|---|---|---|
| architecture | `Qwen3_5ForConditionalGeneration` | same | same | `Qwen3_5ForConditionalGeneration` |
| model_type | `qwen3_5` | same | same | `qwen3_5` (text: `qwen3_5_text`) |
| nested text_config | yes | yes | yes | yes |
| layers / hidden / FFN | 64 / 5120 / 17408 | same | same | 64 / 5120 / 17408 |
| dense vs MoE | dense (no expert keys) | dense | dense | dense (no expert keys) |
| total tensors | 1199 | 1199 | 1199 | 1199-equivalent footprint (392 tensors in shard 1 + 16 in shard 18 censused; full index verified) |
| **MTP tensors** | **15** (`mtp.fc.*`, `mtp.layers.*`) | **15** | **15** (plus an `mtp`-named file) | **15** (`mtp.fc.*`, `mtp.layers.*` — last shard) |
| **vision tensors** | **333** (`model.visual.*`) | **333** | **333** | **333** (`model.visual.*` — first shard) |
| tokenizer class | `Qwen2Tokenizer` | `Qwen2Tokenizer` | **`TokenizersBackend`** ⚠ | `Qwen2Tokenizer` |
| license (card) | apache-2.0 | apache-2.0 | apache-2.0 | apache-2.0 (gate mode: `auto`) |

Parent B checkpoint form (authenticated `files_metadata` at the pin):
**18 safetensors shards, 51.7 GiB total, zero GGUF files** — a true
Transformers/Safetensors checkpoint, exactly what the lineage policy
requires. Companion files present: `chat_template.jinja`,
`preprocessor_config.json`, `video_preprocessor_config.json` (the
multimodal stack), plus the full tokenizer trio.

Reading of the evidence:

- **The family label is `qwen3_5`, not `qwen3_8`.** The Qwen3.8-27B
  checkpoints are served under the existing Qwen3.5
  `ForConditionalGeneration` architecture (multimodal-capable wrapper
  with nested `text_config`). This matches Chowder's existing
  `moe_instrumentation.py` experience: Qwen family members are loaded
  through shared architecture classes, and the honest identification is
  by config + tensor evidence, not by the marketing name.
- **All three readable parents are dense** — no expert keys anywhere.
  The dense→MoE conversion premise holds; none of them is already sparse.
- **MTP is present in all three** (15 tensors, config keys
  `mtp_num_hidden_layers` / `mtp_use_dedicated_embeddings`): Phase 17's
  preserve-by-default policy applies from the first transformation, with
  hashes recorded before/after.
- **Vision tower is present in all three** (333 `model.visual.*`
  tensors): Phase 18's boundary rule applies — the first sparse stage
  operates on the language tower only, with the vision tower and
  projector frozen/preserved and that boundary documented in every run.
- **⚠ D's tokenizer class differs** (`TokenizersBackend` vs
  `Qwen2Tokenizer`). Before any cross-parent comparison or Teacher
  Fabric usage involving D, tokenizer *identity hashes* must be
  compared, not class names. If D's vocab/merges diverge from A/B/C,
  token-aligned teacher signals against D fail closed (already
  enforced by `teacher_fabric.ensure_tokenizer_compatible`), and any
  repair-data exchange with D must route through digest-only, not
  token-text, comparison. This is exactly the structural divergence
  Phase 3 says to fail closed on.

## Blockers (honest, unresolved)

1. **~~Branch B (primary parent) is gated~~ RESOLVED (2026-09-06).**
   The repo is auto-gated (`gated: auto`); the account's stored HF token
   accepted the terms and authenticated file access was verified at the
   pinned revision `404ea47a…` (config, tokenizer assets, shard
   metadata, and safetensors headers all read successfully). Full
   evidence is in the manifest above. Operational note: the token is
   stored only in this machine's HF token store, never in the
   repository — and because it was once shared in plaintext, rotate it
   when convenient.
2. **Weights: parent A cached and verified; B/C/D not acquired.**
   Parent A (control) completed its pinned-revision download (61
   minutes) into the established local-models home
   `F:\Local Models\HuggingFace\Qwen\Qwen3.8-27B` and is **verified**:
   full-mode manifest `d382d54f159f7b6c6b03afac88f7f445d03385684f77ae159fa5db07d7533f59`
   (18/18 weight shards hashed, 55,563,006,776 bytes = 51.75 GiB; 10/12
   semantic files hashed — `special_tokens_map.json` and
   `added_tokens.json` are recorded-absent, which matches the repo's
   file list), verification **clean**, zero divergences. The signed
   manifest lives at `F:\Local Models\HuggingFace\Qwen\Qwen3.8-27B.manifest.json`
   (sha256 `401f8e7a…`). Per `LOCAL_MODELS.md` policy, use the local
   directory path as the model source — no second copy, no silent
   re-download. Parents B/C/D (51.7 / ~52 / ~52 GiB) still have no
   bytes on disk; G: (80 GB free, least-full volume) is the next
   candidate target, or F: again (59 GB free). Clearing space (I: and
   F: are >96% full) remains a user decision.
3. **Protected evaluation suite content does not exist yet — the
   harness does.** `src/chowder/parent_eval.py` (with tests) implements
   the nine-dimension suite schema with complete-coverage validation, the
   protocol fingerprint over suite definitions (candidate identity
   excluded), capability/behavior aggregation kept separate by
   construction, a fail-closed tokenizer-identity gate, hash-only
   protected-suite fingerprint indexes, the Phase-13 contamination audit
   hook, and FK-anchored persistence into `evaluation_runs`. What still
   does not exist is the protected *content*: the real prompts/answers
   per dimension that the suite specs reference. Authoring and curating
   that content is evidence work, not code, and it gates Phase 4. The prior
   campaign's evaluation protocol covered the 8B model's task suite; the
   Phase 4 tournament requires the nine-dimension suite (reasoning,
   coding, knowledge, calibration, self-correction, instruction
   following, agentic, thinking-efficiency, behavior) under one protocol
   for all four parents, with capability and behavior scored separately.

## Phase plan mapped onto Chowder's real machinery

| Phase (mission) | Chowder reality |
|---|---|
| 0 rescan | done this pass (main, PRs, CI, docs, code inventory) |
| 1 explicit target | this document + ROADMAP Priority 0 pointer |
| 2 pin revisions | manifest above (B blocked by gate); extend `local_model` manifests with shard hashes at cache time |
| 3 trainability audit | architecture table above; per-parent `AutoConfig`/`AutoModel` load checks and PEFT target-module audit at cache time, fail-closed (`MoeArchitectureAuditError` pattern) on surprises |
| 4 parent tournament | new protected evaluation suite (see blockers); identical protocol across parents; multi-objective decision recorded with evidence |
| 5 preserve baselines | every parent's tournament results become the frozen comparison set for all later architecture gates |
| 6 first dense→MoE | weight-preserving FFN-partition conversion prototype; attention/DeltaNet/core untouched; success criterion is *working conversion machinery*, not size |
| 7 instrumentation | `moe_instrumentation.py` (landed, #106) — router probabilities, activation frequency, entropy, dead/overloaded experts, persisted as evidence |
| 8 router healing | bounded QLoRA/Unsloth recovery stage (`UNSLOTH.md` isolated env); expert weights stay native; isolates router problem from expert-weight problem |
| 9 progressive sparsification | one compression change per experiment cycle; Chowder's cycle/gate machinery already enforces this |
| 10 dense-floor attack | structured channel/layer/low-rank experiments, each with its own hypothesis + regression suite; never magnitude-only pruning |
| 11 active-param accounting | measured routing geometry per output (total/active/shared/routed/MTP/vision/embedding split); a label unsupported by measurement is a false claim |
| 12 A4B frontier | Pareto-preserving; A4B is unproven until measured, and A5–A9 results are preserved if they win |
| 13–14 no gaming / Regression Surgeon | existing machinery (`contamination.py`, `repair_orchestrator.py`, recursive repair) applies unchanged |
| 15 training engine | `UNSLOTH.md` isolated env for QLoRA recovery; Transformers/PyTorch stack for surgery; a transformed checkpoint must load independently before any backend is trusted with it |
| 16 Memory Fabric | opt-in only; interaction tested explicitly first (per `MEMORY_FABRIC_ACCEPTANCE.md` evidence discipline) |
| 17 MTP preservation | 15 tensors + config keys audited per transformation, hashes before/after |
| 18 multimodal preservation | vision tower + projector frozen/preserved during language-tower stages; boundary documented per run |
| 19 local-first | `LOCAL_MODELS.md` order already implemented; parent dirs cached once, `offline: true` for runs |
| 20 Flash-Next note | research note below; not part of the current program |
| 21 fractal control | reassess after every major experiment; a bottleneck invalidates the plan, not the plan the evidence |

## Flash-Next research note (Phase 20 — future, not current)

Qwen3.8-Flash-Next already ships the sparse geometry this program is
trying to reach (large expert pools, high top-k). A future comparison
route could profile its experts → prune the expert pool → reduce top-k →
heal the router toward ~A4B. The blocker is its enormous stored
footprint (hundreds of GB across experts), which this workstation cannot
hold alongside the four 27B parents. Revisit only if the dense-27B path
stalls at an active-parameter floor above target.

## Acquisition order (once blockers clear)

1. A `Qwen/Qwen3.8-27B` @ pinned sha — control, readable, apache-2.0.
2. D DavidAU parent @ pinned sha — readable, carries the tokenizer
   caveat; cache and hash its tokenizer assets immediately.
3. C OBLITERATUS @ pinned sha — readable.
4. B orcarouter @ pinned sha — only after authenticated access exists.

Cache location per volume free space at download time; each cache entry
recorded as a local-model manifest (repo, pinned sha, shard names/sizes,
config/tokenizer hashes) so runs use `LOCAL_MODELS.md` local paths with
full provenance. Local paths are the run-facing `base_model` values from
then on.

## Milestone 1 checklist (the honest gate)

Do not call the sparse-model project underway merely because this
document exists. Milestone 1 completes when:

- [x] OrcaRouter exact revision pinned **and accessible** (`404ea47a…`; auto-gate accepted, authenticated file reads verified)
- [x] Official Qwen exact revision pinned (`1d4bf0f2…`)
- [x] OBLITERATUS exact revision pinned (`a58c3b53…`)
- [x] DavidAU trainable parent resolved and pinned (`81c73940…`, resolved from the GGUF card)
- [x] all four architecture manifests recorded (A/C/D full file reads; B authenticated reads at the pinned revision)
- [ ] protected parent-evaluation suite established (harness implemented in `src/chowder/parent_eval.py`; protected suite *content* per dimension still to be authored)
- [ ] all four evaluated under identical protocol
- [ ] results persisted
- [ ] parent-selection decision recorded with evidence
- [ ] selected parent cached locally (parent A *control* is cached and manifest-verified at its pin; selection itself awaits the Phase-4 tournament, so this box stays unchecked regardless of A's status)
- [x] first dense→MoE transformation plan generated (PR #120: docs/PHASE6_CONVERSION_PLAN.md — partition-conversion design against the audited qwen3_5/qwen3_5_moe module shapes, with the exactness harness and validation ladder specified; the transformation itself remains unexecuted)
- [x] no distillation involved (lineage policy fixed above)

Four checkboxes are pre-checked because this document and its
successors closed them with verified evidence; the rest require real
protected content, real weight downloads, and real evaluation runs.
