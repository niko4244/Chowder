# Qwen3.8 parent-freeze pipeline and C/D acquisition tooling

Handoff for this isolated branch/worktree. Written for whoever picks this
up next (human or agent) with zero other context. This document describes
only what this branch actually did — nothing about retry7's live progress
is asserted here beyond what was true and read-only-observed at the time
this branch was created.

## Mission and isolation

This work was done in parallel with the live `retry7` A/B/C/D parent
tournament (running under `Chowder-Protected/`), which was explicitly
**frozen and off-limits**: no modification of `parent_eval.py`,
`parent_tournament.py`, `evaluators/base_text_worker.py`, protected suite
content, scorer semantics, generation budget, tokenizer handling,
quantization config, or the CUDA/bitsandbytes/torch/transformers
environment; no GPU-heavy tests, no model loads, no restarting services,
no touching `Chowder-Protected`'s output directory or registry.

- Branch: `feat/qwen38-parent-freeze-and-cd-acquisition`
- Base: `origin/main` at `2892c231a96d0517b8db0ee01ae00f8f5abefd15`
  (PR #139 commit-headroom gate, #140 protocol v2, #141 HANDOFF refresh —
  confirmed via `git log` at branch creation time).
- Created as a fresh `git worktree` (`../Chowder-qwen38-freeze`), fully
  separate from the primary checkout (which had unrelated uncommitted
  work on `docs/roadmap-sync-priority6`) and from `Chowder-Protected`.
- Verified zero interaction with the live tournament: `Chowder-Protected`
  was only ever read (a copy of its sqlite registry and a listing of its
  real retry7 run directory, both read-only, to learn the exact evidence
  schema this work needed to consume) — never written to.
- At the time this branch was created, retry7 had only parent A
  (`parent-a-qwen38-27b-official`) completed; B/C/D were still pending.
  Nothing here required or assumed B/C/D's completion.

## What was implemented (both deliverables complete, tested, committed)

### 1. Deterministic four-parent evidence-reporting and freeze layer

**File:** [`src/chowder/parent_freeze.py`](../src/chowder/parent_freeze.py)
**Tests:** [`tests/test_parent_freeze.py`](../tests/test_parent_freeze.py) (16 tests, all passing)

A pure, offline consumer of already-persisted `evaluation_runs` rows
(the exact rows `parent_tournament.record_parent_tournament_result`
writes via `parent_eval.aggregate_parent_result`). Produces:

- `ParentSelectionPacket` — always buildable, tolerant of partial/missing
  role evidence, every gate's pass/fail recorded as data rather than
  raised. Carries parent identity + pinned revision (and whether it
  matches), manifest hash, per-suite protected-content digest
  (`holdout_fingerprints_sha256`), the full 9-dimension score table with
  capability/behavior kept separate (never blended), worker
  runtime/versions, wall-clock and peak-VRAM where recorded, a
  best-effort item-level failure inventory (read, hash-verified, from
  `predictions-*.jsonl` under the run's recorded artifact directory —
  flagging likely-truncation vs genuine misses), and an explicit
  `evidence_gaps` list naming exactly which mission-requested fields
  (`worker_attempts`, `commit_headroom_gib_at_launch`, tokenizer
  identity) are **not currently persisted anywhere durable by the
  existing tournament code** — confirmed by reading both the real
  registry evidence shape and the real retry7 artifacts on disk. This is
  a genuine upstream gap in `parent_tournament.py`/`parent_eval.py`
  (computed values that are never copied into the persisted evidence
  dict), which this branch is not permitted to fix.
- `ParentFreezeRecord` — the strict artifact. Raises the first violated
  fail-closed gate (`MissingParentEvidenceError`,
  `DuplicateParentEvidenceError`, `ProtocolMismatchError`,
  `SuiteContentMismatchError`, `TokenizerComparabilityError`,
  `RevisionMismatchError`, `IncompleteDimensionCoverageError`,
  `MalformedEvidenceError`) unless every gate passes for all four bound
  roles. `default_decision_rule` recommends parent B (the documented
  primary development parent) only when no dimension shows a
  clear-difference (≥2-item) regression against native control A;
  otherwise it looks for a single clean alternative among B/C/D, and
  returns "no automatic selection" — never a guessed pick, never a
  blended capability+behavior scalar — when none is unique.
- Deterministic sha256 digest over canonical JSON for both artifacts
  (changes whenever any material evidence changes; stable otherwise) —
  tested directly.

**How to wire this up for real** once B/C/D finish:

1. Bind roles: `RoleBinding(role="A", label="parent-a-qwen38-27b-official", expected_revision=<A's pin>)`,
   similarly for B (`parent-b-orcarouter-uncensored`), and for C/D once
   acquired (see below for their `LocalParent.label` convention — this
   branch did not invent C/D's tournament labels since that's
   `parent_tournament.py`'s decision, off-limits to add to here).
2. Compute `expected_protocol_sha256` from the **live** frozen suite root:
   `parent_suite_content.build_tournament_spec(r"C:\Users\nikma\Chowder-Protected\suites\v1").digest()`
   — never hardcode this string; the whole point of the gate is that it's
   recomputed from the real, current frozen content.
3. Measure tokenizer evidence for each parent via the existing,
   unmodified `parent_tournament.tokenizer_evidence(LocalParent)` (this
   branch deliberately does not re-implement or re-measure it) and pass
   the four results as `tokenizer_evidence={"A": ..., "B": ..., ...}`.
4. Open the real registry (`RunRegistry(r"C:\Users\nikma\Chowder-Protected\tournament-ab.registry.db")`, **read-only usage** — nothing in `parent_freeze.py` writes to a registry) and call
   `build_selection_packet(...)` then `freeze_selected_parent(...)`.

### 2. Fail-closed acquisition/preflight tooling for parents C and D

**File:** [`src/chowder/qwen38_acquisition.py`](../src/chowder/qwen38_acquisition.py)
**Tests:** [`tests/test_qwen38_acquisition.py`](../tests/test_qwen38_acquisition.py) (12 tests, all passing)

Prepares — **does not perform** — acquisition of
`OBLITERATUS/Qwen3.8-27B-OBLITERATED` (`PARENT_C_PIN`) and
`DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU`
(`PARENT_D_PIN`) at the exact revisions pinned in
`docs/QWEN38_SPARSE_PROGRAM.md` / `qwen38_campaign.py`. No network call,
no download, no CUDA, no `torch` import happened anywhere in this task or
its tests — every Hub- and disk-touching step
(`list_files_fn`, `snapshot_download_fn`, `candidate_roots`) is an
injected callable, exercised only against synthetic fixtures.

Reuses rather than reinvents: `qwen38_campaign.ParentPin` for pin
binding (refuses a branch name or `main`/`latest` at construction),
`local_model_manifest.py` for the identical hash/verify provenance
standard parents A/B already have, `parameter_accounting.account_parameters`
for stdlib-only shard/parameter/config metadata, and
`parent_tournament.tokenizer_evidence` /
`parent_eval.ensure_parent_tokenizer_compatible` for tokenizer identity
and the comparability gate.

Fail-closed: refuses a GGUF-only revision (mirrors the exact "GGUF is not
the training parent" resolution already recorded for Comparison C in
`docs/QWEN38_SPARSE_PROGRAM.md`), refuses when no candidate destination
volume has ≥1.15× the remote total free, detects a partial download
(`verify_expected_files_present`) and a diverged manifest
(`check_already_acquired` returns `None`, forcing re-acquisition rather
than silently trusting stale bytes), and is restart-safe/idempotent (a
clean, verified destination short-circuits without calling
`snapshot_download_fn` again).

**How to run this for real** once A/B's tournament finishes and someone
decides to acquire C/D: wire `list_files_fn` to
`huggingface_hub.HfApi().model_info(repo_id, revision=revision, files_metadata=True).siblings`
(mapping each `RepoSibling` to a `RemoteFileInfo`), wire
`snapshot_download_fn` to `huggingface_hub.snapshot_download`, measure
`candidate_roots` via `shutil.disk_usage` on the real candidate volumes
(the program doc mentions G: and F: as the current least-full
candidates — verify current free space before use, it changes), and call
`acquire_parent(PARENT_C_PIN, destination, ...)` /
`acquire_parent(PARENT_D_PIN, destination, ...)`. This is a real,
multi-hour, ~52 GiB-per-parent network operation — deliberately never
triggered by this branch or its tests.

## What was audited but explicitly NOT implemented

Per the mission's own ordering ("only implement this third area if it
remains isolated... after those are complete"), the optional third area —
wiring `successive_halving.run_successive_halving()` /
`candidate_selection.prioritize_candidates()` into `project_runner.py` —
was **audited, not implemented**:

- Confirmed still true by grep: neither function is called from
  `project_runner.py` (or anywhere under `src/chowder/` outside their own
  modules, `intervention_outcomes.py`, and `expected_improvement.py`) on
  this branch's base commit. `docs/ROADMAP.md`'s "Wiring them into
  `project_runner.py` remains open" statement is accurate as of
  `2892c231a9...`.
- **Found a live, unrelated worktree** at
  `.claude/worktrees/agent-a66d39fd3e0ec3607` on branch
  `feature/expected-improvement-selector` (HEAD `23d6548`), which already
  touches candidate-selection-adjacent code (`expected_improvement.py`
  calls `prioritize_candidates`). Implementing project_runner.py wiring
  in *this* branch risked duplicating or conflicting with whatever that
  concurrent effort is doing to the same seam. Given deliverables 1 and 2
  were the explicit priorities and this third area was conditional,
  the safer choice was to stop here and hand off the audit finding rather
  than guess at another session's in-flight design.
- **Recommendation for whoever picks this up:** check the state of
  `feature/expected-improvement-selector` first; if it already covers the
  `project_runner.py` wiring (or a superseding design), that work should
  take precedence over a fresh UCB1/successive-halving integration here.

## Verification run in this branch

```
python -m pytest tests/test_parent_freeze.py tests/test_qwen38_acquisition.py -q
# 16 + 12 = 28 passed

python -m ruff check src/chowder/parent_freeze.py src/chowder/qwen38_acquisition.py \
    tests/test_parent_freeze.py tests/test_qwen38_acquisition.py
# All checks passed!

python -m pytest tests/test_parent_eval.py tests/test_parent_tournament.py tests/test_registry.py \
    tests/test_parameter_accounting.py tests/test_qwen38_campaign.py tests/test_local_model_manifest.py -q
# 71 + 44 = 115 passed (no regressions in every module this work reuses)
```

A full-suite run (`python -m pytest -q`) confirmed zero regressions
repo-wide: **1284 passed, 77 skipped (GPU-gated), 0 failed** in 64.75s,
CPU-only, on this branch's HEAD.

## Not done / explicitly out of scope

- C/D were not downloaded, verified, or evaluated. Nothing in this branch
  claims otherwise.
- The four-parent tournament is not complete (only parent A had run at
  branch-creation time) and this branch makes no claim that it is.
- No PR was opened and nothing was pushed to `origin` from this branch —
  that decision (and CI) is left to whoever reviews this handoff.
