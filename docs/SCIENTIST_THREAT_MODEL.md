# Scientist mode: threat model and security boundary

Principle (non-negotiable): **the scientist proposes; Chowder admits; Chowder
executes; Chowder measures; the scientist interprets; Chowder verifies.**

The scientist layer — whether the deterministic test provider or the AI
Scientist v2 sidecar — is treated as **untrusted input generation**. It has
no more authority over Chowder than a config file, and considerably less than
a policy document.

## Trust boundaries

```
┌──────────────────────────────────────────────────────────────┐
│ TRUSTED: Chowder process                                     │
│  - GrowthLoop, LoopPolicy (closed keys), GrowthState         │
│  - RunRegistry (immutable evidence), certification,          │
│    promotion gate, frozen judge                              │
│  - ResearchDirector, ResearchTree, ResearchMemory            │
│  - ExperimentCompiler (admission + compilation)              │
└──────────────△───────────────────────────────△───────────────┘
               │ typed objects only            │ files only
┌──────────────┴───────────────┐   ┌───────────┴───────────────┐
│ ScientistProvider            │   │ Sidecar runtime (untrusted)│
│ (Hypotheses/Proposals/       │   │  - AI Scientist v2 code    │
│  Findings: DATA, never code) │   │  - LLM-written code        │
└──────────────────────────────┘   │  - its own LLM API keys    │
                                   │  - NO Chowder imports      │
                                   │  - NO trusted paths        │
                                   └────────────────────────────┘
```

- The **provider contract** returns typed data objects
  (`Hypothesis`, `ExperimentProposal`, `ResearchFinding`, `export_state()`
  dicts). A malicious provider cannot express "change the promotion
  threshold" in that type system — there is no such field, and unknown
  fields are refused at the dataclass boundary.
- The **sidecar runtime** executes LLM-written code *by design* (upstream
  interpreter). It therefore gets the least possible authority: a workspace
  directory, file-drop/extract protocol, its own model keys, and nothing
  else. It never receives trusted paths, registry files, policy documents,
  or credentials beyond its own model API configuration.

## Threats and controls

| # | Threat | Control | Pinned by |
|---|---|---|---|
| T1 | Provider output mutates trusted policy (thresholds, protected set, budgets) | The provider protocol has no policy mutation surface; `LoopPolicy` loads via closed-key `from_mapping` and is frozen; provider objects are dataclasses with closed fields | `test_scoped_policy_isolation_pinned_*` (director tests) |
| T2 | Proposal requests more compute than the mission has | `ResearchDirector.submit` runs budget admission **before** compilation; over-budget proposals are refused and journaled | `test_over_budget_proposal_refused_before_compute` |
| T3 | Malformed proposal | Strict dataclasses with `validate()`; admission re-validates; refusals recorded | `test_invalid_proposal_refused` |
| T4 | Architecture-mutating experiment proposed | `ExperimentConstraint.allowed_types` allowlist; unallowed type refused pre-compile; architecture types must be mission-allowed explicitly | `test_unsupported_experiment_type_refused` |
| T5 | Sidecar executes LLM code with Chowder trust | Sidecar is a separate runtime (docker/wsl/local-config); file-only protocol; no Chowder imports in the adapter; no sidecar code in the Chowder process | `test_local_runtime_rejects_missing_home`, sidecar tests |
| T6 | Provider fabricates measurement results | Observations enter only via `record_observation(run_id, metrics)`, which fails closed unless the run exists and is complete in the run registry | `test_fabricated_observation_refused` |
| T7 | Evidence attributed to the wrong candidate/run | Observations carry the exact `run_id`; findings reference `run_id`s that must resolve | `test_observation_bound_to_exact_run`, `test_memory_points_at_real_runs` |
| T8 | Carried/parent evidence presented as fresh | Provider context marks carried evidence; fresh observations are produced only by `record_observation` against new runs | `test_carried_evidence_flagged` |
| T9 | Scientist declares promotion / success | No promotion surface exists in the scientist layer; promotion remains certification + `evaluate_promotion` + frozen judge; findings carry claim **status**, never verdicts | director/tree tests; absent-by-construction |
| T10 | Research memory points at nothing (drift) | `ResearchMemory` resolves every evidence `run_id` at query time; dangling references fail loudly | `test_memory_points_at_real_runs` |
| T11 | Sidecar modifies trusted artifacts | Adapter-constructed workspace paths only; the sidecar config never contains trusted paths; isolation is the operator's documented requirement | `SCIENTIST_MODE.md` § Security; adapter code review |
| T12 | Provider down → fake results | No silent fallback. The fake provider is import-named for tests; the sidecar adapter raises `SidecarError` when its runtime is missing | `test_sidecar_missing_home_fails_loudly` |
| T13 | Restart loses research state or forks it | `ResearchTree.save()`/`load()` and `ResearchMemory` JSONL are durable; resume reconstructs identically | `test_resume_reconstructs_tree` |
| T14 | Two budget ledgers interfere | Mission budget decremented only by the director; growth envelope only by the loop, from measured outcomes; neither reads the other's ledger | `test_two_ledgers_do_not_mix` |
| T15 | Human review bypassed by re-requesting | `scientist_policy.state_treatments` is read from the policy document (human-owned) each diversion; provider cannot flip gates; default `human_review` | `test_growth_integration_default_unchanged`, `test_scientist_then_review_gates` |
| T16 | Untrusted code runs in the Chowder process | The adapter never `exec`s/`eval`s/imports sidecar code; the only subprocess is the sidecar runtime entry point inside its runtime | adapter source; `test_adapter_imports_nothing_from_sidecar` |

## What the scientist may read (exported context)

`ResearchDirector.export_provider_context()` produces the sanitized view the
provider may see: the mission, current `SkillProfile` estimates (with
UNKNOWN ≠ zero semantics), open failure categories and counts, previously
tried mechanisms with outcomes, remaining budget, **and explicitly no**:
candidate evaluation item content, protected evaluation content, registry
internals, other candidates' full results, or policy documents. Carried
(evidence-derived) facts are marked `carried: true`.

## Data-flow rules (restated for reviewers)

1. Every experiment exists because a `Hypothesis` exists (director refuses
   orphans).
2. Every executed experiment is a compiled, admitted `ExperimentProposal` —
   admission order: schema → policy/types → budget → compile.
3. Every `ExperimentObservation` cites the immutable run that produced it.
4. Every `ResearchFinding` cites observations; status transitions
   (`provisional → replicated` / `rejected`) require the replication policy's
   evidence counts.
5. Promotion is decided only by the existing growth promotion path. Scientist
   mode can, at most, hand a candidate to it.

## Residual risks (honest)

- **Sidecar escape**: if the operator configures `runtime: local` and the
  local environment is not actually isolated, LLM-written code runs with that
  environment's user privileges. Chowder cannot force isolation; it refuses
  missing runtimes and documents the requirement. Docker/WSL are the
  recommended runtimes.
- **Prompt injection through research content**: future corpora summaries
  exported to the provider could carry text that manipulates the provider's
  LLM. The typed-boundary design contains the damage (worst case: bad
  hypotheses, which then fail admission or measurement), but content-level
  sanitization of exported context is future work.
- **Cost accounting**: sidecar LLM spend is not metered by Chowder
  (`max_cost_usd` is enforced against Chowder-side estimates only).
