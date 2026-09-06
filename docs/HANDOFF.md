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

## Current state (updated 2026-09-06)

- `main` = `ee9df2a`, CI green post-merge, zero open PRs.
- Recent merges, this session: #109 (`training_engine` evidence field),
  #110 (censored-outcome view `censored_outcomes.py`), #111 (Teacher
  Fabric Slice A: `teacher_fabric.py` + `docs/TEACHER_FABRIC.md`),
  #112 (dataset identity/scale + accelerator context in
  `intervention_outcomes.py` — closed the Priority-6 context-gap item).
- Priority 6 evidence foundation remaining, per ROADMAP's own list:
  production caller that persists executor-failure incidents
  (`record_execution_incident` exists, only tests call it), the
  chronological backtest validator vs UCB1 (zero-hard-gate-violations
  check required), the EI/GPU-hour-aware policy itself, and the
  cross-model transfer mechanism.
- Teacher Fabric: Slice A done; Slices B–J not started. Slice B (teacher
  signal store) orientation already established these decisions, from
  `docs/TEACHER_FABRIC.md` §8/§9/§16 and the brief's regression rules:
  registry migration 4 adds a `teacher_signals` ledger table (append-only
  via `registry._insert_immutable`, mirroring `_migration_2_execution_
  incidents`; `database.py` currently at `CURRENT_SCHEMA_VERSION = 3`);
  payload files are content-addressed by digest with atomic writes and
  interrupted-write recovery; dedup key = digest over
  (`request_digest`, payload content hash); **verified-or-absent** (rule
  #8: corruption is never served, recovery is explicit); `local_cache_
  max_bytes` is a required argument (no default budget — §16.1 stays
  genuinely open); streamed shard *transport* belongs to later slices —
  Slice B implements complete-or-absent integrity only. New module home:
  `src/chowder/teacher_signal_store.py` per the doc's file plan.
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
