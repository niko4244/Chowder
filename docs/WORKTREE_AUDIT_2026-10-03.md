# Worktree & Clone Audit — 2026-10-03

Scope: every registered worktree of the Chowder repository (31 total), the
main clone's stale state, and the branch landscape. **Read-only audit** —
nothing was deleted, stashed, or reset. Every recommended action preserves
work: rescue-branch first, prune second.

## 1. The main clone (`C:\Users\nikma\Chowder`) — still stale, worse than documented

The HANDOFF hazard documented a stale working tree from the stride-fix
investigation. Current measurement: **94 modified/untracked paths**, HEAD at
`245f48a` on branch `docs/roadmap-sync-priority6` ([behind 1] of its
upstream). Notable dirty content beyond the previously documented files:

- modified: `docs/HANDOFF.md`, `docs/SPARK_GSM8K_CAMPAIGN.md`,
  `pyproject.toml`, `src/chowder/adapter_guard.py`,
  `src/chowder/backends/transformers_peft.py`,
  `src/chowder/backends/transformers_worker.py`,
  `src/chowder/config_validation.py`, `src/chowder/evaluators/base_text.py`,
  `src/chowder/evaluators/base_text_worker.py`, `chowder_batch/runtime_loop.py`
  and more;
- untracked: a dozen `tests/test_exp_*.py` / `test_kaggle_qat_lane.py` /
  `test_repair_only_peft.py` / `test_reward_training_cpu.py` etc. — a
  body of experiment work that exists nowhere else if lost.

This clone IS the dev tip (`245f48a` = `main`'s tip), so the *checkout* is
current; the *working tree* on top of it is the problem.

### Recommended plan (in order, each step reversible)

1. **Inventory, don't guess (no risk).** From the main clone:
   `git status --porcelain=v1 > ../main-clone-dirty-inventory.txt`,
   `git diff > ../main-clone-dirty.patch`, and copy untracked files aside:
   `git ls-files --others --exclude-standard | tar -T - -cf ../main-clone-untracked.tar`.
2. **Rescue untracked tests to a branch (low risk).** The untracked
   `tests/test_exp_*` family looks like real experiment work. Create a
   rescue branch, commit only the untracked files, push:
   `git checkout -b rescue/main-clone-untracked-2026-10-03`, add ONLY the
   untracked paths (`git ls-files --others --exclude-standard`), commit, push.
3. **Rescue the modified files to a second branch (low risk).** Same pattern
   for the ~60 modified paths: `git checkout -b rescue/main-clone-modified-2026-10-03`,
   `git add -u`, commit with a message naming the suspected origins
   (stride-fix leftovers, PEFT work), push. Nothing is deleted; both branches
   can be diffed against `main` later at leisure.
4. **Only after 1–3 are pushed:** decide per-file what belongs on `main`
   (some changes may already exist in merged PRs — compare against the rescue
   branches), then either restore the working tree
   (`git checkout -- .` + delete rescued strays) or cherry-pick.
   Never `git reset --hard` before 2 and 3 exist on the remote.

## 2. Worktrees (30 besides the main clone)

| Group | Worktrees | State | Action |
|---|---|---|---|
| **This lane** | `F:/chowder-worktrees/scientist` | `a7c1c1d`, clean, all pushed | keep (active) |
| HANDOFF-documented dirty | `C:/Users/nikma/Chowder/.claude/worktrees/agent-ae537f2ed08ff8828` (`feature/intervention-outcomes`) | **unaudited dirty state** per HANDOFF | audit first: `git -C <path> status --short`; rescue-branch anything uncommitted, then prune |
| HANDOFF-documented prunable | `.../agent-a66d39fd3e0ec3607` (`feature/expected-improvement-selector`) | files rescued to pushed branch (per HANDOFF) | safe to `git worktree remove` now; keep the branch |
| Also worth checking | `.../agent-af43e28fc0d45b5bf` (`codex/activation-offload-layout`), `claude-moe-instrument` (`main` @ `71af763` — stale main), `eval-7445759`, `eval-d98edb5` (detached) | unknown | run `git status --short` in each; rescue → prune pattern |
| Brainz workspaces | `Chowder-p7-evidence-fixes` (`closeout/final-pass`, behind 36), `Chowder-p7-verification` (detached), `Chowder-router-review` (`codex/router-resume-review`), `Chowder-tfd-pilot` | branches exist; some behind | behind-branches: verify merged (§3) then prune worktree; detached: confirm nothing uncommitted first |
| `C:` root task worktrees | `Chowder-freeze-fix`, `Chowder-handoff-update`, `Chowder-kaggle-cd`, `Chowder-qwen38-freeze`, `Chowder-router-healing`, `Chowder-v3tournament`, `.claude/worktrees/chowder-lora-preset-*`, `kaggle-a3-lane`, `tfd-revised` | one branch each, presumably merged | verify merged into origin/main (§3), then `git worktree remove` |
| `F:/chowder-worktrees/*` gen/rung lanes | `gen0`, `gen1`, `gen1-rebase`, `growth`, `growth-trainfn`, `frontier-seed`, `memfab-qual`, `rung4`, `rung5` | feature branches | same: verify merged → prune; keep `scientist` (active) |

## 3. How to decide "prunable" safely (the rule)

A worktree is prunable when (a) `git -C <path> status --short` is empty (or
its dirt has been rescue-branched), and (b) its branch is merged:

    git branch --merged origin/main | grep <branch-shortname>

- **Merged + clean** → `git worktree remove <path>` (add `--force` only if
  git complains about ignored files; NEVER before checking status).
- **Merged + dirty** → rescue-branch the dirt first (steps 1–3 pattern), then
  remove.
- **Unmerged** → that is real work: push the branch if unpushed
  (`git push -u origin <branch>`), leave the worktree until someone decides
  the branch's fate.
- **Detached HEAD worktrees** (`eval-7445759`, `eval-d98edb5`,
  `Chowder-p7-verification`): check for uncommitted work first; a detached
  checkout has no branch to lose but dirty files can still be unique.

## 4. The stale `main` checkout inside a side worktree

`C:/Users/nikma/Chowder/.claude/worktrees/claude-moe-instrument` sits on
`main` at `71af763` — **the stale main** that once confused a session (the
scientist worktree had to `git reset --hard 245f48a` to recover).
Recommendation: after confirming clean status, remove this worktree; local
`main` should only ever be updated by fetch/pull in the main clone, never
held hostage in a side worktree.

## 5. What this audit deliberately did NOT do

No `git worktree remove`, no `git reset`, no `git stash`, no branch deletion,
no file deletion anywhere. Multiple agent sessions share these checkouts; the
rescue-before-prune rule (the same one that saved the EI selector files)
applies to every path above. Executing the plan needs the operator's
go-ahead — each numbered step is safe to run independently and reversibly.

## 6. Execution record — 2026-10-03 (steps 1–3 executed with operator go-ahead)

**Inventory (step 1)** — no risk, done first:
`C:\Users\nikma\main-clone-dirty-inventory.txt`, `main-clone-dirty.patch`
(103 KB), `main-clone-untracked.tar` (181 MB, includes everything below).

**Untracked rescue (step 2)** — `rescue/main-clone-untracked-2026-10-03`
@ `cb92214`, **pushed**. 82 files, +23,232 lines: the chowder_batch
experiment series, new src modules, experiment docs, the test_exp_* family,
evidence/, examples/, the QAT lane, pr199-fix.patch. Deliberately NOT
staged: `.venv-repro/` (a virtualenv, not work), `.claude/worktrees/`
(nested registered worktrees), `.scratch_exp_d_smoke*.json` (scratch) —
all preserved in the tar.

**Modified rescue (step 3)** — `rescue/main-clone-modified-2026-10-03`
@ `1d08bcb` (built on cb92214), **pushed**. The 16 paths with REAL content
diffs (`--ignore-cr-at-eol`), +1431/−31. CRLF-only churn deliberately left
out. The main clone now sits on this branch; residue fell 94 → 5 paths (only
the excluded junk). Original branch `docs/roadmap-sync-priority6` untouched.

**Prunability sweep (section-3 rule: clean AND merged into origin/main):**

| Verdict | Worktrees |
|---|---|
| **prunable now** | `claude-moe-instrument` (stale-main hostage, clean, main merged — §4), `Chowder-p7-verification` (detached, clean) |
| **rescue-then-prune (merged, real tracked diffs)** | `Chowder-p7-evidence-fixes` (5f +438/−64), `Chowder-router-review` (8f +495/−91), `agent-af43e28fc0d45b5bf` (3f +193/−28), `Chowder-v3tournament` (4f +195/−11), `gen1-rebase` (1f +19/−8) |
| **rescue-then-prune (merged, untracked strays only)** | `agent-a66d39fd3e0ec3607` (2 — HANDOFF's "prunable" claim is stale), `eval-7445759` (2), `eval-d98edb5` (3), `Chowder-freeze-fix` (2), `rung4` (4), `tfd-revised` (20) |
| **keep: unmerged branch (all pushed except 2)** | `agent-ae537f2ed08ff8828` (now clean), `Chowder-tfd-pilot`, `Chowder-handoff-update`, `Chowder-kaggle-cd`, `Chowder-qwen38-freeze`, `Chowder-router-healing`, `kaggle-a3-lane`, `lora-preset-a8f123dc`, all nine `F:/chowder-worktrees` feature lanes, `scientist` (active) |

Unpushed branches flagged (safe in shared refs regardless — removal of a
worktree never deletes its branch; push is machine-loss hardening):
`lora-preset-qwen3-5`, `feat/candidate-search`. Nothing was removed, reset,
or stashed in this sweep; removal remains the operator's call per worktree.
