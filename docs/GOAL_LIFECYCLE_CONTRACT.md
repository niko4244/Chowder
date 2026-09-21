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
