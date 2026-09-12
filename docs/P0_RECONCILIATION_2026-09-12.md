# P0 — Reconciliation inventory, execution-source decision, and preservation record

**Date:** 2026-09-12 (all timestamps local CDT unless marked Z).
**Authority:** [2026-09-12 reconciled plan](../../../Chowder-handoff-update/docs/superpowers/plans/2026-09-12-chowder-training-first-reconciled.md), task P0. This file is the "new reconciliation inventory beside the next protected engineering run" that task P0 requires. It records state; it authorizes nothing.
**Method:** read-only inspection (`git status/rev-list/diff/show`, file hashes, `sqlite3` in `mode=ro`, SQLite online backup API, process and GPU enumeration). No source file, dataset, registry row, checkpoint, PR setting, process, or automation was modified. The only files created by this task are this inventory and the two verified registry backups under `backups/`.

---

## 1. Execution-source decision

**Selected execution worktree: `C:/Users/nikma/Chowder-router-healing`, branch `feat/hot-core-upcycling`, HEAD `75705ac2f266c92ee06b14590ecbf7a293463657` (clean).**

Rationale: it is the PR158 head branch, all six required CI checks pass at this revision (run 34721322736), and P1–P3 of the plan are already merged into that branch as commits `f538f65`, `5f87398`, `75705ac`. `origin/main` = `41a2913` (19 commits ahead of local `main`, which is stale); PR158 remains **OPEN, MERGEABLE, unreviewed — no merge authorized**.

Worktree sync state (verified after `git fetch origin`, 2026-09-12):

| Worktree | Branch | HEAD | vs origin | Dirty |
|---|---|---|---|---|
| `Chowder-router-healing` | `feat/hot-core-upcycling` | `75705ac` | 0/0 | 0 files |
| `Chowder-handoff-update` | `docs/engineering-run-results` | `c254c0c` | 0/0 | 0 files |
| `Chowder-kaggle-cd` | `feat/kaggle-cd-acquisition-and-equivalence` | `a62ddc5` | 0/0 | 0 files |
| `Chowder-qwen38-freeze` | `feat/qwen38-parent-freeze-and-cd-acquisition` | `ddf58bc` | 0/0 | 0 files |
| `Chowder-v3tournament` | `protocol-v4-behavior-scoring` | `5c1f388` | 0/0 | **4 files** |
| `Chowder-freeze-fix` | `fix/freeze-v3-behavioral-tokenizer-gate` | `432be82` | local **behind origin by 4** (origin `7a116e6`), 0 ahead | **2 untracked** |
| `Chowder` (primary) | `docs/roadmap-sync-priority6` | `e8b8696` | 0/0 | **6 modified + 7 untracked** |

Registered worktrees (from `git worktree list --porcelain`, 12 total) include five inside `C:/Users/nikma/Chowder/.claude/worktrees/` and one inside `C:/Users/nikma/.claude/worktrees/`; see §4 for their classification.

**Runtime identities pinned for subsequent runs** (recorded on the host that executed P1–P3 verification): Python 3.11.9 (`C:/Users/nikma/AppData/Local/Programs/Python/Python311/python.exe`), torch 2.11.0+cu128, transformers 5.16.1, pytest 8.4.2. Child/worker processes must record and match these per plan P4.

**Import-shadowing hazard (P0 finding, load-bearing):** a bare `import chowder` from the host Python resolves to `C:/Users/nikma/Chowder/src/chowder/__init__.py` — a *different* worktree with in-flight edits — because that checkout is on `sys.path` via its `.venv`/installation, while `pip show chowder` returns nothing. All direct-python verification in the selected worktree MUST pin `PYTHONPATH=src` (pytest is already correct via `pythonpath = ["src"]` in `pyproject.toml`). This is now codified in the plan's execution protocol.

## 2. Active worker / editor check

- No process whose command line references `Chowder|chowder` (PowerShell `Win32_Process` enumeration, all python.exe inspected; several uv-spawned helper processes carry no Chowder path).
- No Chowder training/evaluation/heartbeat job running. The existing heartbeat remains PAUSED per the audit; nothing here restarted it.
- GPU ownership: GPU0 16 GB shows 636 MiB used by desktop processes (explorer/ShellHost/StartMenu, plus a permission-restricted PID 2112 — *not* attributable and left untouched); GPU1 6 GB at 0 MiB. No compute job present. Per plan, this is a point-in-time observation, not a reservation.
- No `git stash` entries exist in `frontier-lowram-autoresearch` (checked; no stashes were created or dropped).

## 3. Protected frontier modifications — verified unchanged

All five paths exist as working-tree modifications in `C:/Users/nikma/frontier-lowram-autoresearch` (the frontier repo, NOT the Chowder checkouts) and were **not touched** by P0 or by the earlier P1–P3 batch. Content hashes at inventory time (SHA-256, first 20 hex):

| File | sha256 (first 20) |
|---|---|
| `evals/mmlu_pro_eval.py` | `013252a6046457ea7647` |
| `kaggle/cycle4_math_kernel/run_cycle4_math_train.py` | `ee61853cc89c0b480797` |
| `tests/test_mmlu_pro.py` | `1ba57e5fe7fc13cbabcb` |
| `training/adapter_delta.py` | `bc7ae6f6e95a1fd51aa9` |
| `training/train_dpo_trl.py` | `4dfd5c4da865193709ea` |

`git status` in frontier shows exactly these five as modified and nothing else. Downstream tasks must re-hash against this table before and after implementation.

## 4. Pending / local artifact classification

Classes per plan P0: **[UPSTREAM]** already integrated; **[RETAIN]** retained local work with a decision pending; **[DEFER]** deferred unrelated work, do not integrate with Chowder changes; **[SUPERSEDED]** superseded with preserved evidence; **[ACTIVE]** someone else's in-flight work — leave alone; **[EMPTY]** inert leftover.

### 4.1 Primary checkout `C:/Users/nikma/Chowder` (`docs/roadmap-sync-priority6`, synced with origin)

| Artifact | Class | Evidence / disposition |
|---|---|---|
| M `src/chowder/backends/activation_offload_worker.py`, M `tests/test_activation_offload.py`, ?? `src/chowder/backends/activation_offload_hooks.py`, ?? `docs/ACTIVATION_OFFLOAD_STRIDE_FIX.md` | **[ACTIVE]** | One workstream: stride-preserving activation offload hooks (dated 2026-09-04). Mirrored byte-identically (verified by diff) in agent worktree `agent-af43e28fc0d45b5bf` (`codex/activation-offload-layout`, dirty: 5). Owner unknown (codex-origin branch name); **do not commit, revert, or edit**. Integration is a separate, unmade decision. |
| M `src/chowder/backends/transformers_worker.py` (37 lines) | **[ACTIVE]** | Same workstream: wires `offload_pack/offload_unpack` into the trainer's save-for-backward path. **Overlap hazard:** this is the same file P1 (`75705ac`) fixed in `Chowder-router-healing`; the two edits are in *different regions* (hooks wiring vs preset table) and no conflict is claimed, but any future merge of either must re-read the other. NOT a duplicate of P1 — verified by content. |
| M `src/chowder/evaluators/base_text_worker.py` (+82), M `src/chowder/evaluators/transformers_text_worker.py` (+29), M `src/chowder/evaluators/transformers_text.py` | **[ACTIVE]** | Unicode-punct normalization in scoring paths (dated ≤2026-09-10). **Overlap hazard:** `base_text_worker.py`/`transformers_text_worker.py` carry the *same filenames* as Protocol-V4 scoring edits in `Chowder-v3tournament` (see §4.4). Two parallel evaluator workstreams; P4/P9 in the plan touch the same seams. Single-owner rule applies. |
| ?? `.claude/`, `.venv-repro/`, `campaign/` | **[ACTIVE]/[RETAIN]** | Untracked working dirs of the primary checkout (sizes not enumerated — `du` timed out on `.venv-repro`; assumed large). Not evidence; not part of any Chowder change. Owner processes may use them. |
| ?? `chowder-project.json` (2,682 B) | **[RETAIN]** | Saved project spec "Qwen3.8-9B Abliterated - Phase A Compatibility Pilot" (schema v1, seed 17, 2.0 GPU-h goal). Untracked; a candidate input for a future normal-interface run. Not this plan's artifact; do not delete. |
| ?? `docs_brief_staging_teacher_fabric.txt` (17,878 B) | **[DEFER]** | Prompt/brief for "Priority 8 — Teacher Fabric / Remote Intelligence Distillation" (frontier-scale teacher distillation). Unrelated to the router milestone; per plan §8 this class of work is deferred. Do not integrate into Chowder commits. |
| Worktrees `agent-a66d39fd3e0ec3607` (`feature/expected-improvement-selector`, dirty 2: `src/chowder/expected_improvement.py`, `tests/test_expected_improvement.py`, dated 2026-09-04), `agent-ae537f2ed08ff8828` (`feature/intervention-outcomes`, clean @ `06ff222`), `claude-moe-instrument` (**on `main` @ `beffd23`, clean**) | **[ACTIVE]/[RETAIN]** | Agent worktrees under `Chowder/.claude/worktrees/`. The `main`-checkout one is a hazard class of its own (bare imports resolve here — see §1); flagged, not touched. |

### 4.2 `C:/Users/nikma/Chowder-freeze-fix` (`fix/freeze-v3-behavioral-tokenizer-gate`)

| Artifact | Class | Evidence / disposition |
|---|---|---|
| ?? `src/chowder/router_healing.py`, ?? `tests/test_router_healing.py` | **[SUPERSEDED]** | Pre-merge drafts of the router-healing pilot, dated 2026-09-10, sitting untracked on the freeze-fix branch. Both **differ** from the versions that were actually merged via PR156 (`Chowder-router-healing` has tracked, tested versions): 95 / 64 changed lines respectively by diff count. Evidence preserved in place; superseded by PR156 content. Do not copy forward; do not delete without owner review (they are the only record of the draft variants). |
| Branch state: local behind origin by 4 commits (`origin/...` = `7a116e6`), 0 ahead | **[RETAIN]** | The branch's own work is upstream (`4491b06` merge ancestry); the local checkout is simply stale. Refresh (pull) is a safe future action for its owner; not performed here. |

### 4.3 `C:/Users/nikma/Chowder-v3tournament` (`protocol-v4-behavior-scoring`, synced with origin)

| Artifact | Class | Evidence / disposition |
|---|---|---|
| M `docs/PROTOCOL_V4_BEHAVIOR_SCORING.md`, M `src/chowder/evaluators/base_text_worker.py`, M `src/chowder/evaluators/transformers_text_worker.py`, M `tests/test_protocol_v4.py` (dated 2026-09-10) | **[ACTIVE]** | Protocol-V4 behavior-scoring workstream (own branch, in progress, 4 dirty files). Shares evaluator filenames with §4.1's scoring edits — see hazard register §8. Untouched. |

### 4.4 Clean worktrees and non-git leftovers

| Artifact | Class | Disposition |
|---|---|---|
| `Chowder-kaggle-cd` @ `a62ddc5`, clean, synced | **[UPSTREAM]** | Kaggle CD acquisition/equivalence branch; pushed and green. No local-only content. |
| `Chowder-qwen38-freeze` @ `ddf58bc`, clean, synced | **[UPSTREAM]** | Parent-freeze/CD branch; pushed. No local-only content. |
| `Chowder-handoff-update` @ `c254c0c`, clean, synced | **[UPSTREAM]** | Carries the 2026-09-12 plan (this inventory's authority) and the execution protocol. |
| `.wt-g00`, `Chowder-baseline`, `Chowder-pr40` | **[EMPTY]** | Empty directories (verified `ls` empty). Inert; cleanup is the owner's call. |

## 5. Run evidence — verified against the audit

Re-verified 2026-09-12 by direct hash and file inventory (all match [AUDIT_2026-09-12.md](AUDIT_2026-09-12.md) §2–§3; audit itself sha256 `94b0036eedafbbfd64db…`, untouched):

| Evidence | Verification |
|---|---|
| Rerun final report `F:/llm-models/_a4b/realtrain-gsm8k-2/realtrain-report.json` | sha256 `8f19701dc0450f1b8703…` = audit value. Intact. |
| Fraction sweep `F:/llm-models/_a4b/prune-fraction-generation-sweep.json` | sha256 `555832fb2eba890d6031ae32…` = audit value. Intact. |
| Rerun adapter `…/runs/realtrain-unsloth-a9e4dbb91fa5/adapter/` | 7 files (incl. `adapter_model.safetensors`, `README.md`, `progress.json`); **no `trainer/` directory** — corroborates "no resumable training state saved". My directory-digest scheme (sorted relpath+content sha256) computes `589528d281a02c84…`, which is *not* comparable to the registry's `64309720…` (different canonicalization; registry value remains the authority and is unchanged). |
| Failed run `F:/llm-models/_a4b/realtrain-gsm8k/` | `.chowder/runs/realtrain-unsloth-162930115b1d/` present with `stdout.log`/`stderr.log`; baseline eval artifacts present. Unchanged. |
| Preregistrations | `9f66960` (original, adds `docs/PRUNED_9B_REAL_TRAINING_PREREG.md` + drivers) and `5de632d` (pre-launch addendum, adds `docs/PRUNED_9B_REAL_TRAINING_PREREG_ADDENDUM.md`, lr-scheduler honouring + tests) both resolve in the selected worktree's history. Intact. |
| Eval artifacts | Both runs carry `evals/*/eval-{spec,result}.json`, `predictions-gsm8k.jsonl`, `holdout-fingerprints-gsm8k.jsonl`, `stdout/stderr.log`. Untouched. |
| **CSV backup** | **Not found** in `F:/llm-models/_a4b/` or the protected run dir within searched depth — the plan's "CSV backup" item could not be located and is recorded as UNVERIFIED rather than assumed present. If it exists elsewhere, its owner should register it here. |

## 6. Registry preservation (consistent backups, verified)

Both run registries had `level2.db-wal` = 0 bytes (no pending WAL frames) but were backed up with the SQLite **online backup API** from a `mode=ro` source connection anyway, per plan. Backups written under `backups/` beside this file; integrity-checked; source rows unchanged.

| Source | Backup | integrity | row counts (notable) | backup sha256 (first 12) |
|---|---|---|---|---|
| run1 `.chowder/level2.db` | `backups/level2-run1-failed.db` | ok | 1 execution incident, 1 eval, 1 result, 6 run_events, 0 failure_records | `d31dfb077442` |
| run2 `.chowder/level2.db` | `backups/level2-run2-rerun.db` | ok | 1 training run, 2 evals, 2 results, 44 failure_records, 1 repair_plan, 9 run_events, 0 incidents | `fdb37f9b9b02` |

Read-only row inspection corroborates the audit: run2 `experiments` holds `baseline` (status **`planned`** despite completed evaluation results — the known lifecycle defect, left as-is per "do not edit old rows") and `realtrain-unsloth` (status `passed`). Corrections stay linked as new audit evidence; no migration was run on these files.

## 7. Plan-doc and PR state (for continuity)

- Plan document: `Chowder-handoff-update/docs/superpowers/plans/2026-09-12-chowder-training-first-reconciled.md`, commit `c254c0c` on `docs/engineering-run-results` (pushed; sha256 of current content `fe8f585c2991fc882edc…`). Carries the P1–P3 completion record and execution protocol added earlier today.
- PR158: OPEN at `75705ac`, 6/6 checks green (run 34721322736), MERGEABLE, no reviews, **no merge authorized**. Execution-log comments: `issuecomment-5648951860`, `issuecomment-5648999281`.

## 8. Coordination hazard register (forwarded to P4+)

1. **Import shadowing** (§1): direct-python invocations off the selected worktree can silently import another checkout. Mitigation pinned: `PYTHONPATH=src`; P4's parent/child identity check will make this a hard failure instead of a hazard.
2. **Same-file workstreams** (§4.1/§4.3): evaluator worker files are concurrently drafted in `Chowder/` (unicode-punct scoring) and `Chowder-v3tournament` (protocol V4); `transformers_worker.py` carries active offload work while P1's preset fix lives on the PR branch. Any of these merging into the PR branch requires re-reading and re-running the affected suites.
3. **Stale local branches** (§4.2): `Chowder-freeze-fix` is 4 behind its origin; refresh before any further use.
4. **`main`-checked-out worktree** (`claude-moe-instrument`): a worktree holding `main` blocks main-branch operations; flagged for owner cleanup.

## 9. Disposition decisions recorded by P0

- Execution worktree: `Chowder-router-healing` @ `75705ac` (see §1). Implementation slices P4 onward execute there under the plan's single-owner rule.
- Nothing was integrated: no [ACTIVE] item was committed, stashed, reset, or edited; no [SUPERSEDED] item was copied forward or deleted; no [EMPTY] directory was removed.
- "Already upstream" classification confirmed for `kaggle-cd`, `qwen38-freeze`, `handoff-update`, and the freeze-fix branch's own content; no local-only commit on any synced branch awaits integration.
- CSV backup: recorded UNVERIFIED (not found) rather than fabricated.

## 10. Verification of this inventory

Re-run paths for the next auditor: `git worktree list --porcelain` from any Chowder worktree; the hash commands of §3/§5; `PRAGMA integrity_check` on the two backups; and the process/GPU enumeration of §2. Any mismatch against this file after the recorded time is new evidence, not grounds for rewriting history here.
