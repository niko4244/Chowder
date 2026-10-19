# Goal lifecycle contract

The canonical `run_project()` path creates or resumes one frozen `GoalLifecycle` for the project's `objective_version`. It reuses the project's existing `Goal` and `MetricTarget` values; it does not define a second goal model.

`ProjectRunOutcome.succeeded` is true only when the lifecycle records `STOP_GOALS_MET`. A successful training process, evaluated candidate, promotion, or clean return is not sufficient. `STOP_GENERATION_LIMIT`, `STOP_BUDGET`, `STOP_PLATEAU`, `STOP_UNCERTAIN`, `STOP_OPERATOR`, `REQUIRES_HUMAN_REVIEW`, cancellation, refusal, crashes, and incomplete evidence are non-success outcomes.

The lifecycle freezes objective version, goal digest, benchmark digest, evaluation-protocol digest, constitution digest, and the deterministic `protocol_contract_digest` in the run registry. A later invocation resumes only when all identity fields match. A terminal objective is returned from persisted evidence without launching another candidate. An existing caller may inject a `GoalLifecycle` explicitly for controlled orchestration; the default project path constructs the lifecycle and cannot silently fall back to candidate execution success.

## Legacy protocol-contract migration

An objective created before `protocol_contract_digest` existed is not resumed automatically. `run_project()` refuses it with `migration required` and never mutates the legacy row.

An operator may explicitly call the narrow protocol-contract migration operation with:

- the legacy project;
- a distinct new objective version;
- a non-empty approval ID, approver, timestamp, and reason.

The operation recomputes the current contract from the project configuration, verifies that the goal and benchmark are unchanged, creates a new frozen objective, and appends a migration record containing source/target identities, contract digest, approval, and provenance. The legacy objective remains immutable. Only the returned project with the new objective version may resume; changing its configuration afterward is refused. No approval, reused objective version, goal change, benchmark change, or missing provenance is accepted.

## Legacy unbounded mode (`goal_lifecycle.mode: "legacy_unbounded"`)

The canonical path freezes an evaluation-protocol digest and refuses to record evidence under a changed protocol. A project may opt out ONLY with an explicit config key (`config.goal_lifecycle.mode = "legacy_unbounded"`) AND a goal whose every metric has no `minimum`/`maximum` bound; either alone is inert. When the measured protocol differs from the frozen one under this mode, the lifecycle records the assessment against the frozen identity for continuity instead of refusing.

This mode exists for callers whose objectives intentionally have no bounded goals (generation-limit-only budgets). It must never be combined with bounded goal metrics: a project that declares a pass/fail bar cannot silently keep measuring after its evaluation contract changed. Note the semantics trade-off: an unbounded metric assesses as MET whenever a finite measurement exists, so under this mode a legacy objective can complete with `STOP_GOALS_MET` exactly as it did before protocol freezing — the double gate (explicit opt-in plus all metrics unbounded) is the only guard against accidental use.
