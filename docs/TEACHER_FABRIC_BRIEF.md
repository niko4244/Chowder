You are working on the GitHub repository `niko4244/Chowder`.
Your mission is to design, implement, audit, test, and document a new major Chowder capability:
Priority 8 — Teacher Fabric / Remote Intelligence Distillation
Core objective
Enable Chowder to learn from frontier-scale teacher models that are much too large to store or execute on the user's local hardware.
The architecture MUST NOT assume that teacher model weights are local.
The fundamental abstraction is:
A teacher is a source of training signal, not necessarily a locally executable model.
Chowder's local hardware should primarily run the student, evaluation, local teachers/proxies, and lightweight training work.
Very large open-weight models may instead live on:

* a hosted inference provider,
* a remote API,
* a rented GPU machine,
* a multi-node remote inference cluster,
* an ephemeral cloud job,
* or another machine controlled by the user.

The teacher's useful signal should move to Chowder; the teacher weights generally should not.
Existing repository context
Before making changes:

1. Inspect the complete current Chowder repository.
2. Read `docs/ROADMAP.md`.
3. Read the current executor, project, registry, evaluation, repair, candidate-selection, hardware/resource-accounting and training code.
4. Inspect PR #104, "Local-first model sources and DS4-derived MoE downsizing program."
5. Do not overwrite, duplicate or regress the local-first model changes or Elastic MoE work from PR #104.
6. If PR #104 is not merged, design your branch so that it can cleanly compose with it or base your work on that branch where appropriate.
7. Preserve Chowder's existing hard regression gate as the final authority for promotion.

No claimed capability is allowed unless backed by real code and tests.
Do not add stubs that return plausible-looking success dictionaries.
Do not mark experimental research as proven.
Multi-agent review structure
Act as the orchestrating lead and assign independent specialist roles.
At minimum use these specialist perspectives:
1. Systems Architect
Own the Teacher Fabric boundaries, protocols, dependency direction and integration with existing Chowder architecture.
2. Distillation Researcher
Audit contemporary methods including:

* on-policy distillation,
* multi-teacher on-policy distillation,
* sampled-token teacher scoring,
* sparse/top-K logit distillation,
* black-box knowledge distillation,
* critique/revision distillation,
* preference training,
* verifier/reward distillation,
* remote white-box distillation,
* tokenizer compatibility problems.

Pay particular attention to:

* NVIDIA NeMo-RL MOPD,
* MOPD,
* Open-MOPD,
* DistillKit,
* GrayKD,
and newer directly relevant literature discovered during the audit.

Separate experimentally supported ideas from hypotheses.
3. Distributed Infrastructure Engineer
Design remote teacher execution without requiring teacher residency on Chowder's machine.
Cover:

* synchronous API teachers,
* asynchronous batch teachers,
* remote open-weight inference,
* ephemeral GPU workers,
* multi-node teacher workers,
* remote distillation microjobs,
* retry/resume/idempotency,
* worker disappearance,
* partial results,
* job manifests.

4. Data / Storage Engineer
Design teacher-signal storage for a user with severe local disk constraints.
The design must:

* never duplicate teacher weights locally by default,
* support streamed signal shards,
* support a strict configurable local cache ceiling,
* content-address artifacts,
* deduplicate repeated teacher requests,
* support compression,
* evict consumed/cold shards,
* preserve required provenance.

5. Cost / Meta-controller Researcher
Design deterministic teacher-query selection first, with a path to learned selection later.
Teacher queries must have measurable:

* monetary cost,
* GPU-hour cost where applicable,
* token cost,
* latency,
* expected value,
* measured downstream student improvement.

Do not query an expensive frontier teacher if a local/cheap teacher is sufficient.
6. Reliability / Regression Engineer
Assume:

* providers fail,
* network connections fail,
* teachers change revisions,
* remote jobs disappear,
* results arrive late,
* partial shards are corrupt,
* APIs lack claimed capabilities,
* teachers sometimes give bad answers.

Every failure mode must fail closed without corrupting Chowder's experiment history.
7. Security / Provenance Reviewer
Audit:

* API secret handling,
* accidental prompt/data leakage,
* untrusted remote results,
* manifest verification,
* content hashing,
* teacher identity/revision,
* tokenizer identity,
* licensing/provenance metadata.

8. Integration / Test Engineer
Own the end-to-end acceptance matrix and ensure the feature integrates with Chowder's existing project runner, experiment graph, registry, resource accounting, regression surgeon and hard gate.
Require disagreement between specialists to be explicitly resolved before final architecture decisions.
Architecture requirement
Create a provider-neutral Teacher Fabric.
Do NOT make one provider or API the Teacher Fabric.
The system should have approximately these conceptual components; adjust names if repository conventions justify something better:

* `TeacherRegistry`
* `TeacherCapabilities`
* `TeacherRequest`
* `TeacherSignal`
* `TeacherSignalArtifact`
* `TeacherProvider` protocol
* `TeacherBroker`
* `TeacherQueryController`
* `TeacherSignalStore`
* teacher cost/resource accounting
* provider adapters
* tokenizer compatibility validation
* remote-job manifests
* teacher-effectiveness history

A provider declares capabilities.
Example capabilities:

```python
TeacherCapabilities(
    generate=True,
    critique=True,
    revise=True,
    rank=True,
    scalar_reward=True,
    selected_token_logprobs=False,
    topk_logprobs=False,
    hidden_projection=False,
    remote_training=False,
)

```

Do not infer capabilities from provider names.
Teacher signal hierarchy
Support an extensible signal taxonomy.
At minimum design for:

1. scalar reward / pass-fail
2. generated answer
3. critique
4. corrected/revised answer
5. candidate ranking/preferences
6. sampled/student-token log probabilities
7. sparse top-K token distributions
8. full logits as an optional research-only mode
9. projected hidden representations as experimental
10. remote resulting adapter/checkpoint artifact

Every signal artifact must carry enough metadata to reproduce or invalidate its use.
At minimum:

* teacher stable id
* provider type
* model/revision
* tokenizer identity/hash when relevant
* signal schema version
* request digest
* prompt/input digest
* student trajectory digest
* generation/scoring parameters
* timestamp
* monetary/token/GPU cost
* latency
* payload content hash
* parent teacher-job manifest where relevant
* provenance/licensing metadata where known

Critical research path: sampled-token remote scoring
Implement the architecture needed for a teacher capability conceptually equivalent to:

```python
score_selected_tokens(
    prompt,
    student_trajectory,
) -> SelectedTokenLogprobs

```

For compatible-tokenizer models, this permits the teacher to return only its log-probability for the action/token actually selected by the student instead of a full vocabulary distribution.
The training objective should be designed around the supported MOPD/OPD formulation, with teacher-versus-student sampled-token log-probability gap available as a dense token-level learning signal.
Do not claim this works across arbitrary tokenizers.
Tokenization compatibility must be explicit.
If teacher and student tokenizers are not provably compatible:

* reject token-level alignment,
* downgrade to text/ranking/reward/critique supervision,
* or use a separately researched and validated alignment method.

Never silently approximate token correspondence.
Black-box teacher path
Build a first-class path for teachers that expose only text generation.
At minimum support conceptual operations for:

* generate solution,
* critique student solution,
* propose correction,
* compare/rank candidates,
* produce counterexamples,
* produce independent repair material.

Integrate these with Chowder's existing Regression Surgeon instead of building an unrelated repair system.
Protected evaluation/holdout material must remain excluded from teacher-generated training data according to Chowder's existing contamination rules.
Teacher Signal Store
Teacher outputs must be reusable across experiments.
Implement content-addressed deduplication.
Equivalent teacher requests should not be paid for or executed twice unless explicitly forced.
Design a configurable local cache budget such as:

```yaml
teacher_fabric:
  local_cache_max_gb: 1.0

```

The exact default must be justified.
The local cache may contain teacher signals.
It must NOT silently stage or cache giant teacher weights.
Support:

* atomic writes,
* interrupted-write recovery,
* integrity hashes,
* LRU or evidence-based eviction,
* consumed-shard eviction,
* optional external/object-store backing,
* streaming signal shards into training.

Disk preflight should reason about teacher signal/artifact space, not assume teacher-weight download space when the teacher is remote.
Query escalation controller
Implement a deterministic first-generation query policy.
Example hierarchy:

```text
student passes + high confidence
    -> no teacher

student disagreement / uncertainty
    -> local or cheap teacher

repeated failure / regression cluster
    -> stronger teacher

high-value unresolved failure
    -> frontier teacher

```

The exact scoring rule should be evidence-driven, versioned and testable.
Track at least:

* teacher cost,
* latency,
* failure rate,
* downstream candidate improvement,
* hard-gate success/failure,
* domain/task class,
* signal type.

Do not implement a learned policy yet unless the evidence requirements are already satisfied.
Instead, produce durable data suitable for the existing Priority 6 meta-controller research.
Teacher tournament
Treat teachers and teacher-signal strategies as experiment arms.
Measure improvement-per-cost, not reputation.
The system should eventually be able to conclude things such as:

* Teacher A is most cost-effective for debugging.
* Teacher B's selected-token scores improve tool use.
* Teacher C is expensive and adds no measurable value for this domain.
* No teacher is required for easy tasks.

Integrate this evidence with Chowder's existing intervention-history infrastructure where semantically appropriate.
Do not bypass the hard regression gate.
Multi-teacher training
Do not start with naive teacher averaging.
The design must explicitly account for the failure modes identified by recent multi-teacher OPD research:

* domain sequence-length imbalance,
* unequal token-share/optimization budgets,
* non-uniform convergence,
* asynchronous teacher/student staleness,
* capability starvation.

Design for:

* per-domain/token budget accounting,
* teacher gap tracking,
* signal freshness,
* configurable balancing,
* refresh/re-score policies.

Keep this behind an experimental flag until verified.
Remote open-weight / white-box path
Design a remote worker protocol where a large open-weight teacher runs entirely outside the Chowder host.
Conceptual flow:

```text
Chowder creates immutable job manifest
        ->
remote worker obtains teacher weights itself
        ->
remote worker processes requested inputs
        ->
remote worker emits signed/hashed teacher-signal shards
        ->
Chowder verifies and stores/streams results
        ->
remote worker can be destroyed

```

The local machine must not need the teacher checkpoint.
Support idempotent retries and partial batch recovery.
Do not require a specific cloud vendor.
Provider-specific remote execution belongs behind adapters.
Remote distillation microjobs
Design a stronger mode where training occurs beside the remote teacher.
Example:

```text
local:
  Chowder adapter N
  training manifest
        |
        v
remote:
  huge teacher
  student base
  Chowder adapter N
  distillation job
        |
        v
local:
  Chowder adapter N+1

```

Only small Chowder state should need to cross the network where possible.
The returned adapter/checkpoint must be independently evaluated locally by Chowder before promotion.
Remote success is not evidence of model quality.
The existing hard gate remains authoritative.
Teacher proxy research
Design, but do not prematurely implement as proven, a smaller teacher-proxy/critic.
Goal:
Train a locally affordable model to approximate useful teacher judgments, not necessarily reproduce the frontier teacher's entire generative capability.
Possible proxy outputs:

* candidate ranking,
* pass/fail,
* failure class,
* critique labels,
* teacher-query-needed probability.

The frontier teacher periodically refreshes the proxy.
Measure whether this actually reduces frontier queries without reducing downstream hard-gate success.
Relationship to Elastic MoE / PR #104
The Teacher Fabric and Elastic MoE programs must compose.
Eventually support experiments such as:

```text
full Chowder local teacher
        +
remote frontier capability teacher
        ->
pruned Chowder student
        ->
recovery / distillation
        ->
hard regression gate

```

The first Teacher Fabric PR must NOT perform Qwen expert surgery.
Keep architecture-change work behind the Elastic MoE gates.
Suggested implementation sequence
Do not attempt everything in one giant PR.
Use regression-proof slices.
Slice A — Teacher protocol
Schemas, capability negotiation, provider protocol, artifact format, fake provider, tests.
Slice B — Teacher signal store
Content addressing, dedupe, cache quota, atomicity, corruption tests.
Slice C — Black-box teacher integration
Generation/critique/rank/correction operations integrated with the existing repair system.
Slice D — Teacher cost accounting + deterministic query controller
Budget enforcement, escalation policy, persisted effectiveness data.
Slice E — Selected-token scorer protocol
Tokenizer validation, score artifacts, fake/local scorer and objective-level tests.
Slice F — Real remote scorer commissioning
At least one real non-colocated teacher path tested end to end.
Slice G — Remote job protocol
Ephemeral open-weight worker manifest, retry/resume, verified signal return.
Slice H — Remote distillation microjob
Return adapter, independently evaluate locally, hard-gate promotion.
Slice I — Multi-teacher experiments
Budget balancing, staleness, per-domain accounting.
Slice J — Teacher-selection/meta-controller research
Only after sufficient historical outcomes exist.
For every slice:

* write tests,
* run existing regression suite,
* update documentation,
* update roadmap truthfully,
* do not mark later stages complete.

Non-negotiable regression rules

1. Existing local-only Chowder projects must behave exactly as before when Teacher Fabric is disabled.
2. No remote call occurs unless explicitly enabled/configured.
3. No teacher model download occurs locally as a hidden side effect of remote-teacher use.
4. A provider failure cannot corrupt experiment history.
5. A teacher response cannot directly promote a candidate.
6. All trained outputs must pass existing independent evaluation and hard gating.
7. Teacher cost must enter budget/resource accounting.
8. Duplicate teacher requests should reuse verified cached signal where policy allows.
9. Cross-tokenizer token-level distillation must fail closed.
10. Secrets must never appear in registry artifacts, logs, manifests or committed config.
11. Network tests must not be required for ordinary CI.
12. Real remote commissioning tests must be separately gated.
13. Every claimed remote result must have provenance and integrity evidence.
14. Local disk constraints must be honored and measured.
15. Do not introduce a provider lock-in at the architectural layer.

Evaluation program
Create a benchmark comparing:
A. no teacher
B. local teacher
C. black-box frontier critique
D. black-box ranking/preference
E. selected-token teacher scoring
F. sparse top-K teacher signal where available
G. remote distillation microjob
Measure:

* target benchmark improvement,
* regression count,
* hard-gate pass rate,
* teacher tokens,
* monetary cost,
* GPU-hours,
* wall time,
* network bytes,
* local disk consumed,
* cached teacher-signal reuse,
* retries/failures,
* improvement per teacher dollar,
* improvement per teacher token.

Do not conclude a strategy is better from one example.
Required final deliverables
Return:

1. architecture audit of current Chowder,
2. gap analysis,
3. Teacher Fabric architecture,
4. exact proposed data contracts,
5. file/module plan,
6. integration map into current Chowder modules,
7. phased implementation roadmap,
8. threat/failure model,
9. storage/cost model,
10. test matrix,
11. benchmark protocol,
12. implemented first safe slice,
13. tests and test results,
14. documentation,
15. truthful ROADMAP.md update,
16. remaining unresolved research questions,
17. pull request containing the regression-safe changes.

Where research disagrees, preserve the disagreement and design an experiment instead of choosing a preferred answer without evidence.
Continue auditing and correcting your work until the implemented slice passes its tests and the new feature does not regress existing Chowder behavior.