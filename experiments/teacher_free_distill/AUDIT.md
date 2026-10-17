# Phase 1 audit — repository and PR #201

Recorded before any modification.

- Repository commit at audit start: `3d5c0101e40406e01defa7591e025f28178a7be9`
  (`docs: define isolated teacher-free distillation pilot and explicit limitations`)
- PR #201: OPEN, base `main`, head `experiment/teacher-free-distillation-pilot` (this branch), 5 files.
- Main checkout (`~/Chowder`) has unrelated uncommitted campaign work on
  `docs/roadmap-sync-priority6`; this experiment runs in an isolated worktree
  (`~/Brainz/workspaces/Chowder-tfd-pilot`) and does not touch it.
- Tests verified by execution, not by trusting prior claims:
  `tests/test_teacher_free_prepare.py` → **6 passed** (fresh run at audit time).
  Full suite collects 2693 tests; focused runs used for this pilot.

## PR #201 file review

| File | Verdict |
|---|---|
| `experiments/teacher_free_distill/sources.json` | Fail-closed catalog, all `approved: false`. Correct posture. MoT/SWE-smith revisions were placeholder/partial — fixed in this work. |
| `experiments/teacher_free_distill/stream_hf.py` | Gated, seeded reservoir export, bounded scan. No retry/backoff/resume (acceptable at pilot scale; recorded as a limitation). |
| `experiments/teacher_free_distill/prepare.py` | Provenance gate, license gate, chat normalization, trace digest, holdout exclusion, dedup, deterministic dev split. Sound and well-tested. |
| `tests/test_teacher_free_prepare.py` | Covers the important fail-closed paths; passes when actually run. |
| `experiments/teacher_free_distill/README.md` | Honest scope statement ("does not run a teacher or student, claim knowledge transfer"). |

## Gaps found (driving the implementation plan)

1. No SWE-smith replay/verification system — Phase 3 is entirely missing.
2. No quality filtering beyond dedup (near-dup, malformed-length policy, quarantine reporting).
3. No student selection/memory planning wiring (Phase 4).
4. No training preflight or training execution binding (Phase 5).
5. No evaluation harness for this workflow (Phase 6).
6. No interface surface for the workflow (Phase 7).
7. No repair-task-level holdout wiring for real repair workloads (Phase 6 leakage control).

## Production infrastructure to reuse (no separate framework)

- Training: `chowder.backends.transformers_peft` (`TransformersPeftRunSpec`, `TransformersPeftExecutor`) — digest-pinned, resumable, cancellation-aware.
- Loss masking: `chowder.backends.training_data._build_chat_example` — prefix-consistent completion-only assistant masking with explicit failure on inconsistent templates.
- Dataset contract: `messages` chat rows (validated by `_validate_chat_messages`).
- Memory planning: `chowder.memory.plan_memory` + `chowder.hardware.HardwareSnapshot`.
- Evaluation: `chowder.evaluators.*` (text generation workers, VRAM tracking).
- Interface: Textual TUI (`tui.py`, `tui_growth.py` `Screen` pattern) — new isolated screen reusing real service state.
- Sandbox execution: Podman 5.8.2 present on host (Docker absent).

## Environment facts

- GPUs: RTX 5060 Ti 16GB, RTX 2060 6GB (CUDA available in torch 2.11.0+cu128). Any GPU training run requires explicit operator authorization; not requested by this audit.
- Software stack: Python 3.11.9, torch 2.11.0+cu128, transformers 5.16.1, peft 0.20.0, datasets 4.3.0 — matches `train` extras constraints.
- Disk: 36 GB free on C: — pilot-scale data only.

## Implementation plan (phases 2–8, this branch)

1. **Phase 2** — source verification results into `sources.json` (OT3 approved with pinned revision; MoT stays blocked — no dataset license; SWE-smith-trajectories MIT with underlying-repo caveat). Add near-dup filter, quarantine bucket, quality report to `prepare.py`; add `fetch_smith.py` bounded trajectory fetcher.
2. **Phase 3** — `replay_smith.py`: Podman-isolated replay runner producing digest-evidenced trace records consumable by `prepare.py` repair path. Fails closed without real execution evidence.
3. **Phase 4** — `student.py`: hardware snapshot + memory plan + student selection record (Qwen3-1.7B primary; control optional), no model download without operator go-ahead.
4. **Phase 5** — `preflight.py` (CPU masking/adapter/gradient/checkpoint-roundtrip smoke test) + `recipes/` documenting the three conditions (SFT, repair-continue, preference-if-pairs) bound to the existing PEFT backend.
5. **Phase 6** — `evaluate.py`: baseline-vs-student protocol hooks reusing `chowder.evals`, split-by-repository discipline, leakage checks.
6. **Phase 7** — `TeacherFreeDistillScreen` in the TUI: stage list with real state from the manifest/manifest-producing commands; no decorative progress.
7. **Phase 8** — focused tests for every new module, honest IMPLEMENTED/TESTED/… report in `REPORT.md`, branch commits; PR #201 updated only if the operator asks.
