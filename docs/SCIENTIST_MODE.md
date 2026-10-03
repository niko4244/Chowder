# Scientist mode: audit, architecture and phased plan

Status: **Phases 0–5 implemented and tested; Phase 6 (TUI) deliberately deferred
(see the end of this document for what exists and what does not).** Scientist
mode is **off by default**; nothing in the default growth path changed.

This document is the Phase-0 deliverable the scientist-mode work was required to
start with: an audit of both systems, the resolved architecture decision, the
license position, the threat model, the exact module map, and the phased plan.
It is written to be updated rather than rewritten; each phase records what is
proven (tests, production wiring) versus merely scaffolded.

---

## 1. What the audit found

### 1.1 Chowder (the system we must not break)

Audited at tip `245f48a` ("Evaluate corrected batch-006 repair data", 28
commits ahead of `origin/main` at audit time; the branch was rebased onto this
tip). Everything below was read in source, not from docs alone.

**Production-wired (do not duplicate, do not weaken):**

| Capability | Where | Wiring evidence |
|---|---|---|
| Experiment DAG | `graph.py` | `ExperimentGraph`, `GraphInvariantError` |
| Hypothesis objects (kernel) | `models.py` (`Hypothesis`: observation / suspected_cause / intervention) | README design rule 2 ("Every change has a hypothesis") |
| Pluggable hypothesis generation seam | `hypothesis_generation.py` | `HypothesisGenerator` Protocol + `RuleBasedGenerator`; used by the repair-investigation machinery (`investigation.py`), **execution-failure repair, not capability research** |
| Growth control plane | `growth/growth_loop.py` (1,447 ln) | `GrowthLoop.plan_next/_one_generation`, `LoopDecision` closed terminal set, plateau detection, `LoopBudget.admits()` |
| One production service | `growth/service.py` (`AutonomousGrowthService`) | CLI + TUI + loop + tests are all thin clients; no decision in CLI/TUI |
| Target selection | `growth/target_selection.py` (1,049 ln) | `SkillProfile` (evidence-weighted, UNKNOWN≠zero), `NextTargetSelector.propose()`, `TargetScoreFactors` product formula |
| Intervention classification | same | `classify_intervention()` closed outcome set; `AUTONOMOUS_TREATMENTS` vs `REVIEW_TREATMENTS` |
| Durable growth state | `target_selection.GrowthState` | append-only JSONL under one root: failure-bank / intervention-history / target-history / capability-history / stopping-state |
| Failure bank | `growth/failure_bank.py` | `FailureRecord` taxonomy, recurrence, repair state, `from_records` restore |
| Campaign freeze | `growth/next_campaign.py` (`NextCampaignBuilder`, `LoopPolicy` closed `_KEYS`, `FrozenCampaign`) | preregistration: recipe ids frozen before spend; policy is read-only to the loop |
| Prepare + readiness | `growth/campaign_prepare.py`, `check_campaign_readiness` | corpus dispatch + quality gate before spend (#198) |
| Training binding | `growth/training_binding.py` (1,167 ln) | `SubprocessTrainingFn` (isolated worker subprocess, cost pre-check, material check) |
| Evaluation binding | `growth/evaluation_binding.py` (922 ln) | suite materialization, protocol freeze |
| Certification | `growth/certification.py` (723 ln) | `MeasuredArm` identity checks, protocol match, `Certification` |
| Promotion gate | `growth/promotion.py` | `evaluate_promotion(PromotionInput) -> PromotionDecision`, pre-registered thresholds |
| Frozen judge / settlement | `growth/campaign.py` (`settle_campaign`), README "frozen judge" | `PROMOTED` and judge may not disagree (invariant 8) |
| Successive halving | `successive_halving.py` | **library-only** — `run_successive_halving` has no production caller; doc says wiring is its own pass |
| Research knowledge base | `growth/research_kb.py` | **library-only**: append-only `ResearchEntry`s with scale applicability screening (`applicability()`), never consulted in production decisions |
| Contamination firewall | `growth/contamination.py`, `contamination.py` | 6-layer detection stack; UNKNOWN ≠ CLEAN; protected examples never trainable |
| Data policy | `growth/data_providers.py` + registry | `TrainingDataProvider` Protocol (`supports`/`produce`), `assess_corpus`/`assert_corpus_quality` refuse thin/unverified/contaminated corpora |
| Evidence persistence | `registry.py` (`RunRegistry`, SQLite/WAL, immutable, schema-versioned) | every artifact/eval/result |
| CLI | `growth/cli.py` (995 ln), registered into `cli.py:472` via `register_growth_subcommands` | `chowder growth ...` |
| TUI | `tui.py` (868 ln), `tui_growth.py` (482 ln) | `ChowderTUI` pushes `AutonomousGrowthScreen` (`tui.py:664`) |
| Run events / executor investigator | `run_events.py`, `executor_investigator.py`, `incident.py` | structured failure capture, fingerprint routing |

**Library-only or incomplete (do not reimplement; note, don't wire blindly):**

- `successive_halving.run_successive_halving` — qualified implementation, no
  production caller (drives `ExperimentCycleRunner` rounds, not the growth
  bindings; `run_project` has no `search` section). Wiring it is its own pass;
  scientist mode **documents** it as the designated future resource allocator
  inside a branch rather than duplicating it.
- `growth/research_kb.py` — append-only ResearchKB with honest scale-
  applicability screening. Scientist-mode research memory **extends** this
  pattern (JSONL, research_kb-style entries) rather than replacing it.
- `successive_halving`'s LoRA rank/alpha mapping — "proposed but unmapped"
  per the growth-loop doc.

**The 15 invariants** (docs/AUTONOMOUS_GROWTH_LOOP.md) are the contract the
scientist layer must honor. The four the mission calls out map directly:

| Invariant | Scientist-mode guard |
|---|---|
| 1. Candidate results cannot alter frozen thresholds | provider output is data-only; policy paths are read-only to the provider contract |
| 3. Parent evidence cannot masquerade as candidate evidence | observations always carry the exact `run_id` of the run that produced them |
| 4. Carried evidence cannot masquerade as fresh | provider context exports mark carried evidence as `carried: true`; observations are fresh-by-construction (produced by `record_observation`) |
| 6. All compute after campaign start is durably accounted | research branch budget charges come from measured `CampaignOutcome` costs, not provider estimates |
| 7. Promotion cannot occur unless production certification passes | provider sees promotion as an opaque status; decision made solely by `growth/promotion.py` |

**The config idiom:** JSON documents with a **closed key set** validated at
construction (`LoopPolicy._KEYS` + `from_mapping` refuses unknown keys
fail-closed). Scientist mode reuses this idiom exactly.

**The extension boundary:** `REVIEW_TREATMENTS` = {evaluation_needed,
architecture_research, untrainable_with_current_path} → `REQUIRES_HUMAN_REVIEW`
decision at `growth_loop.py:738`. That is the single, tested diversion point.

### 1.2 AI Scientist v2 (read in source, not from README)

Audited from a shallow clone (`F:\audit\AI-Scientist-v2`, depth 50, checked
against the MIT-licensed Chowder tree).

| Component | What it does | Relevance |
|---|---|---|
| `ai_scientist/ideas/i_cant_believe_its_not_better.py` (413 ln) | The "I Can't Believe It's Not Better" workshop pipeline: given a goal/eval, LLM ideates improvement ideas over the listed datasets/models | ideation format |
| `ai_scientist/ideas/i_cant_believe_its_not_better.json` | The idea schema: Name / Title / Short Hypothesis / Related Work / Abstract / (optional) Description, Experiments, Risks & Limitations | the ideation JSON the adapter must parse |
| `ai_scientist/treesearch/parallel_agent.py` (2,368 ln) + `agent_manager.py` (1,221 ln) | **BFTS**: 4 stages (draft → debug → improve → (report)); `num_workers` parallel agents; `search.max_debug_depth`/`debug_prob`/`num_drafts`; per-node LLM code writing + execution; `multi_seed_eval.num_seeds` | progressive tree search; the provider runs this internally |
| `ai_scientist/treesearch/journal.py` (612 ln) | `Node` (DataClassJsonMixin; step, parent, code, metric, `to_dict`/`from_dict` roundtrip), `Journal` (append, draft/buggy/good node views, metric history) | tree persistence: `journal.json` export is the adapter's read channel |

A design point the audit forced, recorded here because it shaped the whole
adapter: **Chowder owns the tree; the provider suggests.** `ResearchTree` is
Chowder-native (`chowder/scientist/research_tree.py`). AI Scientist v2's BFTS
operates **inside the provider** on sanitized material; its tree state is a
*resource*, read via the provider's `export_state()`. Chowder's tree is the
authority for what exists, what is running, what was spent, and what the
evidence licenses. A sidecar crash loses nothing durable: Chowder's tree,
memory and budget state are the authority, and the provider is restartable
from its own `export_state()` blob (the adapter round-trips through the
sidecar's `journal.json` export — proven by
`test_resume_reconstructs_tree_from_export_state`).
| `bfts_config.yaml` / `bfts_utils.edit_bfts_config_file()` | per-run config: `desc_file`, `workspace_dir`, `data_dir`, `log_dir`, `exec.timeout`, `agent.{type,num_workers,stages,steps,search,...}`, `agent.code.model`, `agent.feedback.model`, `report.model` | the adapter writes one per mission |
| `ai_scientist/treesearch/interpreter.py` (313 ln) | executes LLM-written code **in the agent workspace** (`exec.{timeout, agent_file_name: runfile.py}`) | the security boundary — must stay on the sidecar |
| `ai_scientist/treesearch/backend/backend_openai.py`, `backend_anthropic.py` (+ Bedrock clients in llm.py) | model backends incl. OpenAI-compatible and AWS Bedrock | model-provider configurability |
| `launch_scientist_bfts.py` (369 ln) | end-to-end driver: load ideas JSON → `idea_to_markdown` → per-idea `bfts_config.yaml` → BFTS → writeup/review | what the sidecar runtime ultimately invokes (we use only its ideation + BFTS entry points, never the writeup pipeline) |
| `perform_writeup.py` / `perform_llm_review.py` / `perform_vlm_review.py` | manuscript generation + LLM/VLM review | **explicitly out of scope** (paper generation is not the goal) |

**Sidecar protocol established by the audit (no upstream changes required):**
(1) adapter writes `idea.json` in the upstream ideas schema;
(2) adapter writes a `bfts_config.yaml` with Chowder-desired knobs
(`agent.code.model`, `num_workers`, stage iters, `exec.timeout`) and a
`desc_file`/`workspace_dir`/`log_dir` under the mission workspace;
(3) the sidecar runtime runs the upstream BFTS entry point
(`perform_experiments_bfts`) against that config — upstream code, unmodified;
(4) the adapter reads `journal.json` (upstream `Journal.to_dict()`) and
`tree_data.json` for the tree structure and best node; (5) writeup/review
stages are never invoked. Everything (config, idea, journal, config-yaml)
crosses the boundary as **files** — no RPC, no imports.

**License position (Section 2 below):** the sidecar runs upstream code
unmodified under its own license in an isolated runtime; Chowder ships **zero
AI Scientist source**; all translation is clean-room against file formats
(JSON/YAML schemas) and documented behavior.

### 1.3 What is NOT production-wired in Chowder (audit conclusion)

1. Successive halving (see above) — documented as the future in-branch
   allocator; **not wired in this branch** (honest: scaffolding-level).
2. ResearchKB — production decisions never read it; scientist-mode memory is
   JSONL in the same spirit, deliberately separate from (and recorded against)
   the `RunRegistry` evidence that grounds every claim.
3. Gen-2 campaign: readiness READY, no candidate trained — scientist mode
   integrates *alongside*, never modifies, this plan.

---

## 2. License assessment

**Chowder: MIT** (`LICENSE`, "MIT License / Copyright (c) 2026 niko4244").

**AI Scientist v2: The AI Scientist Source Code License v1.0** (Dec 2025), a
derivative of the Responsible AI Source Code License v1.1 — **source-available,
not OSI open source**. Key points from the license text:

- §2: copyright license to reproduce/prepare derivative works/distribute —
  subject to §3 restrictions;
- §3.1: distributions must include a complete copy of the license;
- §3.2: use restrictions (surveillance, synthetic media, health care, criminal,
  and the "AI Scientist" clause: manuscripts must disclose machine generation);
- §3.3: the §3.2 restrictions must be passed on in agreements covering
  derivative works.

**Compatibility analysis:**

- Copying AI Scientist v2 **source** into MIT-licensed Chowder would create a
  derivative work of a non-MIT codebase inside an MIT project. The §3.3
  flow-down obligation and the risk of mislicensing make this unacceptable.
  Chowder's MIT license would also effectively misrepresent the origin and
  restrictions of that code.
- **Conclusion: zero AI Scientist source in Chowder. Not vendored, not copied,
  not translated line-by-line.** The adapter is clean-room: it consumes and
  produces documented file formats (the upstream ideas JSON schema, the
  upstream `bfts_config.yaml` keys, the upstream `Journal.to_dict()` shape)
  and treats the sidecar as a black box invoked through its documented entry
  point.
- Upstream runs **unmodified** in its own runtime under its own license; the
  operator obtains it (`git clone` upstream) and points Chowder at it.
  Chowder's docs will carry the notice that the sidecar is external software
  under The AI Scientist Source Code License, and that its output (if it were
  ever to include manuscript material) carries the license's disclosure
  obligations — irrelevant to scientist mode, which uses no writeup pipeline.

**Deliverable:** `docs/SCIENTIST_LICENSE.md` (written alongside this document)
carries the full assessment and the operator obligations.

---

## 3. Architecture decision: A, B, or C

The mission asked for an explicit resolution among:

- **A.** Scientist mode sits above GrowthLoop.
- **B.** Scientist mode lives inside GrowthLoop only when research is needed.
- **C.** A peer service coordinated by a higher `ModelResearchService`.

**Decision: C, with a specific composition contract.** Reasons from the audit:

1. **State machines differ.** GrowthLoop's state machine is generation-
   campaign-settle: one campaign at a time, promotion-gated, with the loop's
   15 invariants. A research tree holds *branch* state (hypotheses, proposals,
   competing branches, replication stages) that has no meaning in the loop's
   model. Cramming tree state into `GrowthState` would either fork the
   evidence store (forbidden) or overload the loop's stopping semantics.
2. **Budgets differ.** The loop's envelope is measured wall-GPU-hours per
   campaign with `LoopBudget.admits()` re-checked before each generation. The
   research tree spends across branches with successive-halving-style
   elimination and replication stages. Two ledgers, one authority per ledger.
3. **But they must compose.** The growth loop's `REVIEW_TREATMENTS` diversion
   (`evaluation_needed` / `architecture_research` /
   `untrainable_with_current_path` → currently always human review) is the
   one place a research mission should be creatable *from* growth. The policy
   that controls that diversion must be a growth-policy concern
   (`scientist_policy`), not an internal of a standalone service — which is
   exactly B's insight. C gives the composition without the coupling.

**Composition contract (implemented):**

```
ModelResearchService (new, chowder/scientist/research_service.py)
 ├── AutonomousGrowthService (existing, unchanged)      ← owns growth state
 └── ResearchDirector (new)                             ← owns research state
       ScientistProvider (Protocol; FakeDeterministicScientistProvider for
       tests; AIScientistV2Provider sidecar adapter)
```

- `ModelResearchService` composes both, routes by state machine, and owns the
  two ledger rule: growth campaigns charge the growth envelope; research
  experiments charge the mission budget; neither service can spend from the
  other's ledger.
- Growth-loop integration (Phase 5) is a **policy-gated, default-off
  diversion**: `LoopPolicy.scientist_policy` (new optional key) says, per
  research-heavy treatment, one of `human_review` (default), `scientist_allowed`,
  `scientist_then_review`. When the loop's `_select_target` hits a
  `REVIEW_TREATMENTS` treatment and the policy says scientist_allowed, the
  loop creates a research mission from the proposal and stops the growth
  session with a terminal `RESEARCH_MISSION_CREATED` decision. **The loop
  never runs the research tree itself** — the mission is executed by
  `ModelResearchService` (CLI/TUI), exactly as the mission's option C
  describes: related state machines, separate services, one coordinator.
- Option B was rejected because inside-GrowthLoop research would give the
  loop two state machines and break invariant 15's clean terminal-set.
  Option A was rejected because a layer "above" GrowthLoop cannot reach the
  policy-gated diversion seam without duplicating the loop's decision logic.

---

## 4. Security / execution boundary (threat model)

Principle: **the scientist proposes; Chowder admits; Chowder executes;
Chowder measures; the scientist interprets; Chowder verifies.** The detailed
threat model lives in `docs/SCIENTIST_THREAT_MODEL.md`; the summary:

| # | Threat | Control (implemented) |
|---|---|---|
| T1 | Provider output mutates trusted policy | Provider methods return data-only objects; the provider protocol has **no write access to policy paths** — policies load through `LoopPolicy.from_mapping` closed-key validation, which the provider never touches. A test pins that a provider cannot widen a policy. |
| T2 | Proposal exceeds budget → compute spent | `ResearchDirector.submit` → `admit_proposal()` checks the mission budget **before** compilation; over-budget → `refusals.jsonl` + `REJECTED` status, zero compute. Pinned by test. |
| T3 | Invalid proposal schema | Strict typed dataclasses + `validate()` at construction; admission re-validates. Pinned. |
| T4 | Unsupported experiment type (architecture mutation) | `ExperimentConstraint` allowlist: `admit_proposal` refuses unallowed types before compile. Architecture-mutating types require the mission's `allowed_experiment_types` to name them explicitly. Pinned. |
| T5 | Sidecar runs LLM-written code with Chowder trust | Sidecar runs in an isolated runtime (docker/wsl/local-config); the adapter communicates **only via files** in the mission workspace; no Chowder imports, no Chowder process code execution. Pinned: the adapter module imports nothing from the sidecar and executes nothing from it in-process. |
| T6 | Provider fabricates observations | Observations enter only via `record_observation(run_id=..., metrics=...)`, which **verifies the run exists and is complete** in the run registry (fail-closed). Fabricated metrics without a real run are refused. Pinned. |
| T7 | Evidence attribution to wrong candidate | Observation records the exact `run_id`; the lab bridge binds observations to the compiled campaign's immutable run root. Pinned. |
| T8 | Scientist declares promotion | Provider/director have no promotion API. Promotion remains `growth/promotion.py` + certification + frozen judge. A provider could *claim* anything in its own files; nothing in Chowder reads claims as verdicts. |
| T9 | Research memory points at nothing | Every finding's `evidence` entries are `run_id`s that must exist in the run registry at `ResearchMemory` read time; dangling refs fail the query. Pinned. |
| T10 | Sidecar modifies trusted artifacts | Sidecar runtime gets a workspace dir; trusted state (growth state, registry, policies) lives outside it; the adapter never passes trusted paths into the config. Documented operator boundary + adapter-constructed paths only. |
| T11 | Provider unavailable → silent fake results | `FakeDeterministicScientistProvider` exists **for tests only** and is explicitly named as such; the sidecar adapter fails loudly (`SidecarError`) when the runtime/skills are missing. No silent fallback exists. |
| T12 | Untrusted code in Chowder's process | The adapter never `exec`s, never imports sidecar modules, never shells out to sidecar *code* — it shells out to the sidecar runtime binary (documented path) inside its runtime. |
| T13 | Restart loses research state | `ResearchTree`/`ResearchMemory` are durable JSONL; `ModelResearchService.run_mission(resume=True)` reconstructs the tree and skips completed experiments (tree-state roundtrip pinned by test). |
| T14 | Budget ledger confusion | Two-ledger rule: mission budget decremented only by the director from compiled-experiment costs; growth envelope only by the loop from measured campaign costs. Documented; neither service reads the other's ledger. |
| T15 | Human-review bypass via repeated requests | The `scientist_policy` gates are read from the policy document each time; a provider cannot flip its own gate. Default is `human_review` for every treatment. |

**Non-negotiables honored:** no LLM promotion judge; no candidate-eval-as-
training-data; no second evidence store (research memory records carry
`run_id`s into the existing `RunRegistry`); no UI-only orchestration; no
fail-open; no human-review bypass.

---

## 5. Module map (delivered)

```
src/chowder/scientist/
  __init__.py                 re-exports the public contract
  provider.py                 ScientistProvider Protocol + ProviderUnavailability
  mission.py                  ResearchMission (priorities, protected caps,
                              budgets, autonomy, stop conditions)
  hypothesis.py               Hypothesis, ResearchQuestion
  proposal.py                 ExperimentProposal + ExperimentConstraint
  observation.py              ExperimentObservation
  findings.py                 Claim / ClaimEvidence / ResearchFinding
  research_tree.py            ResearchBranch / ResearchNode / ResearchTree
  research_decision.py        ResearchDecision (expand/reject/replicate/
                              promote_candidate/investigate)
  research_memory.py          ResearchMemory (JSONL, run-registry-backed)
  research_director.py        ResearchDirector (portfolio, admission, review)
  research_service.py         ModelResearchService (C-composition)
  lab_bridge.py               ExperimentCompiler (proposal → campaign spec)
  providers/
    __init__.py
    fake.py                   FakeDeterministicScientistProvider (tests)
    ai_scientist_v2.py        AIScientistV2Provider (sidecar adapter)
docs/
  SCIENTIST_MODE.md           this document
  SCIENTIST_LICENSE.md        license assessment + operator obligations
  SCIENTIST_THREAT_MODEL.md   full threat model
  SCIENTIST_CONFIG_EXAMPLE.json  end-to-end config example
tests/
  test_scientist_contracts.py       Phase 1
  test_scientist_director.py        Phase 1
  test_scientist_research_tree.py   Phase 1+4
  test_scientist_memory.py          Phase 1+4
  test_scientist_sidecar.py         Phase 2
  test_scientist_lab_bridge.py      Phase 3
  test_scientist_growth_integration.py  Phase 5
```

Config example (the `scientist_policy` growth-policy key):

```json
{
  "scientist_policy": {
    "provider": "fake_deterministic",
    "provider_config": {
      "runtime": "local",
      "home": "F:/audit/AI-Scientist-v2",
      "model": "bedrock/anthew-claude-v2",
      "max_tree_nodes": 40,
      "max_parallel_branches": 3
    },
    "state_treatments": {
      "evaluation_needed": "scientist_allowed",
      "architecture_research": "human_review",
        "untrainable_with_current_path": "scientist_then_review"
    },
    "max_tree_nodes": 40,
    "max_parallel_branches": 3,
    "max_cost_usd": 50.0,
    "max_gpu_hours": 20.0,
    "sandbox_required": true
  }
}
```

The exact provider names and knobs are documented in the example file.

---

## 5b. Compute providers: where experiments run (2026-10-03 addendum)

`chowder/scientist/compute.py` adds the scheduler seam the Phase-3 compiler
needed: a `ComputeProvider` protocol (name/hardware_class, `available()`,
`quota()`, `estimate_cost`, `submit`, `poll`), a `LocalCudaProvider` (detected
from the real `HardwareSnapshot`), a `KaggleProvider` (T4×2, 12 h sessions,
declined weekly quota — opportunistic capacity, never load-bearing), and an
`ExperimentScheduler` with fail-closed routing: explicit provider list (no
default), preference order, a screening lane (`experiment_class=screening`
prefers a provider that declares `screening = True`), pinned requests that
never reroute, and quota-out fallback along the declared order. The
hardware-context evidence rule is enforced in the memory layer: **quality
claims may cite cross-hardware runs (protocol-scoped); efficiency claims
(`hardware_dependent: true`) can reach `replicated` only when every cited run
shares one hardware class** — and the director's capability-delta feed takes
quality measurements only, so a cross-hardware efficiency number can never
masquerade as a capability gain. Full design + honest status:
docs/COMPUTE_PROVIDERS.md.

**2026-10-03 update (same section):** the seam now carries three providers —
`LocalCudaProvider`, `KaggleProvider`, `RunPodProvider` — and the Kaggle path
is real: `submit` performs an actual `KaggleApi.kernels_push` (pinned 40-hex
commit + operator-supplied `kernel_command` required, or it refuses), `poll`
maps the real `KernelWorkerStatus` lifecycle and fetches output artifacts,
and `sync_quota_from_api` reconciles the declared weekly budget with the
operator's actual Kaggle GPU quota. RunPod goes through REST v2
(`POST /v2/pods` → `GET /v2/pods/{id}`) behind an injectable transport, and
an EXITED pod never reports complete without a confirmed artifact. The
screening lane's allocator is now wired successive halving
(`scientist/screening_halving.py`): budget-driven elimination rounds
(grow the per-candidate budget, halve the candidates by the tree's
deterministic score, eliminate by gate vs cutoff, graduate only the final
round's survivors to substantial runs), mirroring the growth library's
`run_successive_halving` semantics. The wiring is now service-level and
resumable: a policy `compute` section (closed `provider_from_config` set:
local_cuda | kaggle | runpod) builds the scheduler, `advance_screening`
advances ONE durable step per call (`scientist screen`; `scientist run`
advances it automatically when the policy authorizes compute), candidates
come from admitted proposals in durable memory, and the session survives
restarts (`screening-session.json`). The first real end-to-end Kaggle run
through this path is recorded with verbatim artifacts in
docs/KAGGLE_PROVIDER_ACCEPTANCE.md (pinned-commit install verified in-kernel,
2× Tesla T4, real quota reconciliation, one measured fp16 workload,
0.0155 device GPU-hours of the operator's real 30 h weekly quota). The
kernel template now measures each run's real wall clock and attached
accelerator count, `chowder_result.json` carries measured
`device_gpu_hours`, and the provider settles its weekly budget with the
measured cost (the estimate stands only when a result omits the metering
fields). Graduated survivors hand off to the growth loop through
`graduate_survivors_to_campaign_drafts` — the loop's own
`NextCampaignBuilder` composes real campaign drafts from them (never
freezing, executing, or promoting), with the per-surface pinned-benchmark
mapping an explicit operator input and every hand-off journaled.

## 6. Phases — what is proven, what is scaffolded

### Phase 1 — native scientist contract (PROVEN)

`chowder/scientist` with the full typed contract, deterministic fake provider,
`ResearchDirector`, durable JSONL research memory grounded in the run
registry, and the Chowder-native research tree with a static scoring function
(the mission's prefer/penalize factor list). Tests pin the 20 mission
requirements that are Phase-1-testable (policy isolation, budget refusal,
schema refusal, type refusal, fabricated observation refusal, memory→run
evidence binding, tree resume, attribution, carried-vs-fresh, repeated-failure
reallocation, replication requests, transfer-vs-generalized distinction).

### Phase 2 — AI Scientist v2 sidecar adapter (IMPLEMENTED; sidecar execution NOT yet run against the real upstream)

`AIScientistV2Provider` translates between the upstream file formats and the
Chowder contract: builds upstream ideas-JSON from a `ResearchMission`,
writes `bfts_config.yaml`, parses upstream `journal.json` (`Journal.to_dict()`
shape) into `Hypothesis`/`ExperimentProposal`/`ExperimentObservation` objects.
Two runtime modes: `local` (executes the upstream BFTS entry point as a
subprocess in the configured home — **requires the upstream repo + deps**),
and `wsl`/`docker` (plumbed, operator-configured command). The adapter tests
run the provider **against fixture journal files** (upstream shape, checked
into tests/fixtures), not against a live LLM-backed BFTS run: no test in this
branch has executed the real upstream pipeline, because that requires
sidecar-provisioned LLM keys and a Linux/CUDA runtime. **That end-to-end
sidecar run is explicitly future work.**

### Phase 3 — lab bridge (PROVEN for the growth-campaign path)

`ExperimentCompiler` compiles admitted proposals into growth-campaign-shaped
experiment specs (recipe deltas, data strategy, requested evaluations) and
`ModelResearchService.record_observation` grounds every observation in the
run registry. The compiler targets the growth campaign spec shape; a real
`campaign_prepare`-equivalent execution is Phase-3-future: **today the
compiler output is a spec, not a submitted campaign**, and the end-to-end
"compile → train → observe" demo has run only against the simulator-backed
fake provider, not against real GPU training.

### Phase 4 — progressive research tree (PROVEN)

`ResearchTree` with static scoring (prefer: capability gain, novelty,
information gain, transferability, reproducibility, compute efficiency,
uncertainty reduction; penalize: regression risk, repeated failed mechanism,
contamination uncertainty, non-transferable gains, cost, redundancy),
branch expansion/pruning, replication requests, plateau/stop rules. The
successive-halving integration is **documented, not wired**: the tree's
in-branch resource allocation between sibling candidates names
`successive_halving.run_successive_halving` as the designated allocator once
it gains a production seam (it drives `ExperimentCycleRunner`, the tree
drives compiled campaign specs — the bridge is future work).

### Phase 5 — growth integration (PROVEN, default-off)

`LoopPolicy.scientist_policy` (optional, closed-key validated) with per-state
treatment policy (`human_review` default / `scientist_allowed` /
`scientist_then_review`); `GrowthLoop._select_target` diverts to
`research_mission_required` in the decision when the policy allows; the
growth session ends with a terminal `RESEARCH_MISSION_CREATED` decision and
the mission is executed by `ModelResearchService`. Existing growth behavior
is unchanged when the key is absent (pinned by test), and all existing
boundaries (protected set, promotion, thresholds) are untouched.

### Phase 6 — CLI delivered; TUI research workspace (DEFERRED, deliberately)

The headless CLI is delivered: `chowder scientist status|plan|run|observe|findings|tree`,
all thin clients of `ModelResearchService` (verified end-to-end with the fake
provider: portfolio → admission with journaled refusals → compiled experiment
durable state → closed-vocabulary decision). The TUI Research workspace is
**not** implemented in this branch: `AutonomousGrowthScreen` was the model to
follow, but the scientist surface (tree visualization, findings views,
capability map) deserves its own pass and its own review, and shipping a
shallow TUI ahead of the sidecar's real end-to-end run would be UI-first in
exactly the way the mission forbids. CLI and TUI share one service contract —
the TUI, when built, will be a thin client of the same `ModelResearchService`.

---

## 7. What still prevents Chowder from being called a truly autonomous
model-research system (honest list)

1. **The sidecar has not run end-to-end against real upstream AI Scientist
   v2** (LLM keys + Linux/CUDA runtime required). The adapter is proven
   against fixture journals in the upstream file format.
2. **The lab bridge compiles to campaign specs, not yet to executed
   campaigns** — "compile → real GPU training → independent eval" is
   demonstrated only through the growth loop's own production path, which
   scientist mode diverts to, not through the research tree driving real
   training.
3. **No real-model research mission has been run.** The Definition of
   Success (§ Definition of Success in the mission) — baseline → hypotheses
   → admitted experiments → trained → replicated → transfer/protected evals
   → promotion verdict → persisted findings used as priors — has not been
   demonstrated on real hardware. The machinery is test-proven; the
   demonstration is not.
4. **Successive halving remains unwired** (pre-existing; documented).
5. **TUI workspace not built** (Phase 6 deferred).
6. **`ResearchKB` is still not consulted by production decisions**;
   scientist-mode findings will cite run evidence directly.
7. **Cost model is GPU-hours only** — `max_cost_usd` is tracked but the
   sidecar LLM spend is not metered by Chowder.

Until 1–3 land, the honest description of this branch is: **a
test-proven native scientist contract and research control layer, with a
file-protocol sidecar adapter and default-off growth integration — not yet an
end-to-end autonomous research system.**
