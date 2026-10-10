# 0.5 architecture preparation — current-state audit (2026-10-04)

Scope: everything that stood between Chowder and a scientifically
meaningful 0.5 autonomous model-improvement campaign, audited before any
new tuning. Worktree `F:\chowder-worktrees\autonomy05`, branch
`feature/autonomous-05-architecture`, base = origin/feat/candidate-search
@ 2aef4e6 (PR #204's head), which sits on the #203 lineage, not on main
(2369d37).

## 1. PR landscape (Phase 0)

| PR | Branch | CI state | Files | Verdict |
| --- | --- | --- | --- | --- |
| #199 roadmap-sync-priority6 | Spark 2.5 + lifecycle hardening | green except (then-)drifted cpu-smoke | 83 | merge first; large but self-consistent |
| #200 kaggle-dispatch | push/poll/pull + smoke | all green incl. cpu-smoke | 3 | safe, isolated; production bar in KAGGLE_BACKEND_REQUIREMENTS.md |
| #201 teacher-free-distillation-pilot | experiment | **all CI failing, untriaged**; body says do-not-merge | — | needs triage before anything touches it |
| #202 teacher-free-revised-plan | base = #201's branch | no CI checks reported; conflicts with main | — | blocked on #201 triage |
| #203 fold/main-clone-rescue | light jobs green, cpu-smoke = drift | — | merge after #199 |
| #204 feat/candidate-search | light jobs green, cpu-smoke = drift | 2,645 passed / 77 skipped locally | merge after #199 |

**Merge-order recommendation:** #199 → (#203, #204) → #200 → triage #201 →
decide #202. #199 and #203/#204 overlap in **9 files**
(`transformers_peft`/worker, evaluators, project.py, config_validation,
pyproject), so #199 must land first and the others rebase; #203∩#204 = 0;
#201∩#199 = 0. Branch protection is ON; PRs are the only path.

## 2. CI root cause (proven by experiment)

The cpu-smoke failures across PRs #199/#203/#204 are **not** code. Re-run
of PR #199's CI at the same SHA (5e862c57) that passed on 2026-09-26
failed on 2026-10-04: same code, different day ⇒ dependency drift. Fresh
pip resolution now installs transformers 5.18.0, which no longer populates
`model.config._commit_hash`; `resolved_model_commit` therefore became
`None` at three sites (evaluators/base_text_worker.py,
evaluators/transformers_text_worker.py ~L172, backends/transformers_worker.py
~L1012), failing `tests/test_real_ml_training.py:366`
(`assert baseline_revision` → `assert None`).

**Fix shipped as PR #205** (`fix/provenance-commit-resolution`, @ 4a5946a):
`hf_resilience.resolve_model_commit(repo_id, revision, *, config_commit,
...)` — config attr → local HF cache (refs/snapshot resolution) → Hub API
(retried, never fatal), with 10 new tests. **All PR #205 checks pass
including the previously-drifted cpu-smoke (14m05s)** — the drift is
root-caused and closed, and every open PR should rebase onto it.

## 3. PR #204 audit (the candidate-search code 0.5 would build on)

Invariants verified against the implementation:

- declared-search schema enforced; unknown `candidate_search` keys refuse;
- the screen is training-side-only (`_ADVANCE_FIELDS` =
  status/candidate_succeeded/artifact_ref) — no benchmark score can advance
  a candidate;
- pre-compute projection with per-recipe and per-campaign ceilings;
- plan-schedule identity check between `plan_search` and `run_search`.

**One pre-existing hole (found, documented, not yet fixed here):** attempts
whose settlement is REFUSED (ACTUAL_EXCEEDS_PROJECTION) still carry
`candidate_succeeded=True` and can advance and promote. The campaign
fixtures' economics are fake (runner reported gpu_hours=0.05 against a
0.0049 projection), which is why no test caught it. The attempt-failure
taxonomy (`growth/attempt_failure.py`) now classifies settlement refusals
as BUDGET_EXHAUSTED with **no evidence record**, and R3 of the Kaggle bar
requires remote spend through the same settlement — but the *runner's*
advance rule itself still needs the settlement-refusal check wired in.
That is the single most important open code change on the road to 0.5.

**Resolved (the one next action, now wired):** `compute_cost.settlement_refusal`
is the one owner of the refusal vocabulary; `run_search` ends the lineage of
a settlement-refused attempt, `advanced` and `cycle.select_candidate` refuse
it, and `GrowthCycle.decide_promotion`/`decide_promotion_from_runs` apply
the preregistered retention profile (fail-closed) and the tier wall. As
predicted, the fake fixture economics were hiding the hole: with the gate
in place the harness's "clean" run refused at settlement, so the fixtures
now report settleable costs and the deliberate-overrun scenarios assert the
honest refusal path. Tests: `tests/test_growth_runner_gates.py`.

## 4. Architectural state found (gaps that motivated this branch)

| Gap (mandate phase) | State found |
| --- | --- |
| true checkpoint-resuming progressive halving (2) | EvolutionEngine stack had real resume; campaign search retrained from scratch each round |
| recipe-field classification + search-axis contract (3) | nothing; a declared knob nothing reads would silently do nothing |
| intervention families + maturity classes (4/5) | nothing |
| experimental memory (6) | nothing durable; Run 4's falsification lived in a chat log |
| 3-tier eval separation (7) | eval_tiers existed (cost tiers only); no evidence-trust wall |
| retention as first-class constraint (8) | gate tests existed; no named, preregistered, per-campaign profiles |
| paired causal deltas (9) | aggregate means only |
| adaptive budget ladder (10) | fixed HalvingSchedule; UCB1/EI machinery existed unused at campaign level |
| failure-driven generation (11) | all failures were one bucket |
| Kaggle backend bar (12) | dispatch plumbing only (PR #200) |
| campaign design (14) | none; costs unmeasured |

## 5. What this branch changed (summary)

Five commits on `feature/autonomous-05-architecture` (details and
verification in AUTONOMY_05_IMPLEMENTATION_REPORT.md):

1. b997c5c — checkpoint-resuming progressive halving, SearchProgress resume,
   lineage stops, delta pricing;
2. 4b797d1 — recipe-field classification against real readers, axis-contract
   tests including a live peft spec parse;
3. cc497c0 — intervention families + maturity gate, hash-chained evidence
   store, prior derivation, hypothesis-first generation;
4. 798cdb7 + 54705f7 — the evidence-trust wall, retention profiles, paired
   deltas, and the adaptive budget ladder;
5. 49a73a2 + 07bc00a — attempt-failure classification (after reconciling the
   failure_taxonomy name collision with origin/main), the Kaggle production
   bar, the campaign design from measured costs, and the six mutation
   checks as executable tests.

## 6. Environment note

The ambient editable install resolves `import chowder` to
`C:\Users\nikma\Chowder` (stale main). Every check in this report was run
from the worktree cwd (tests pin `src` themselves) or with
`sys.path.insert(0, "F:/chowder-worktrees/autonomy05/src")`. Two name
collisions were caught and fixed because of this: eval_tiers (pre-existing
cost-tier planner) and failure_taxonomy (origin/main's text classifier).
