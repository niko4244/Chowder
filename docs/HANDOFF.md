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

## Current state (updated 2026-09-07, later same day)

- `main` = `e2a144b` (#129 precision fix, #130 chat parity, #132 parent-adapter
  continuation, #133 replay/rehearsal all merged — Tracks B, C, and D are
  now all done on `main`). PR #133's CI re-run (after the `_LazyModule`
  test-fragility fix below) came back green including the previously
  failing `real transformers peft cpu smoke` job, confirming the fix was
  real; merged (squash, branch deleted).
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
- **Not started yet** (per the Qwen3.8 program directive's own PR
  ordering): Track E (full recursive-repair acceptance through Unsloth —
  the real end-to-end loop: baseline → train → evaluate → fail → harvest
  → cluster → repair → contamination-audit → replay → continue → retrain
  → evaluate → gate → promote, all through the Unsloth engine, on real
  small-model hardware before ever touching the 27B parent), Track F
  (Qwen3.8 campaign manifest/config). Track E is a large, multi-subsystem
  real-hardware integration task (`autonomous_repair.py`,
  `checkpoint_bisect.py`, `contamination.py`, `replay_history.py`,
  `failures.py`, all composed through Unsloth for the first time) — do
  not attempt it without enough real session/hardware time budgeted to
  see a real run through to a real promote-or-reject outcome.
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
