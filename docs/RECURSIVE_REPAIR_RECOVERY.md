# Recursive Repair Recovery

Chowder does not blindly resume an interrupted recursive repair session.

The recovery layer first performs a read-only reconciliation between the durable recursive checkpoint and the `RunRegistry`. Only a checkpoint whose state can be proven from persisted evidence is classified as resumable.

## Why reconciliation comes before resume

A process can fail in several places:

1. before a repair population is proposed;
2. after experiments are persisted but before training;
3. after training/evaluation records are persisted;
4. after a complete single-hop generation but before the recursive hop checkpoint commits;
5. after the hop checkpoint commits but before the controller writes its terminal state.

Restarting training without distinguishing these states can duplicate compute, create sibling repair experiments from stale state, or double-count a repair that already ran.

## Engine snapshot

New recursive sessions persist the exact engine semantics required for deterministic recovery:

- full `Goal` metric targets and directions;
- GPU-hour budget;
- parallel-candidate limit;
- promotion threshold and protocol-match policy;
- full baseline `ExperimentResult`, including metrics and evaluation evidence.

Legacy sessions without this snapshot are reported as `MISSING_ENGINE_SNAPSHOT`; Chowder does not infer missing gate semantics.

## Recovery dispositions

- `RESUMABLE` — checkpoint and registry evidence agree and no unrecorded descendants exist.
- `ALREADY_TERMINAL` — session already completed or failed.
- `TERMINAL_PENDING` — checkpoint itself proves the loop should terminate, such as promotion, max depth, or zero remaining budget.
- `SESSION_NOT_FOUND` — requested session does not exist.
- `CHECKPOINT_INCONSISTENT` — hop count, current candidates, signature counts, score history, or promotion state disagree.
- `MISSING_ENGINE_SNAPSHOT` — deterministic goal/baseline semantics were not persisted.
- `INVALID_ENGINE_SNAPSHOT` — the persisted goal/baseline snapshot is structurally invalid or inconsistent with checkpoint budget.
- `MISSING_REGISTRY_EVIDENCE` — required experiment/training/evaluation/result/repair evidence is absent.
- `AMBIGUOUS_REGISTRY_EVIDENCE` — multiple training or evaluation records make the intended candidate state ambiguous.
- `REGISTRY_EVIDENCE_MISMATCH` — training, evaluation, result, protocol, status, artifact, or compute records disagree.
- `ORPHANED_PROGRESS` — child repair experiments exist beyond the committed recursive checkpoint.

`ORPHANED_PROGRESS` is particularly important: it represents the crash window where a repair hop may already have created or executed children but the recursive trace did not advance. Chowder refuses to retrain from that checkpoint.

## Evidence reconciliation

For every current checkpoint candidate, recovery verifies:

- experiment exists and is durably rejected;
- exactly one training artifact exists;
- exactly one evaluation exists;
- one result exists;
- evaluation references the same artifact produced by training;
- result references that same artifact;
- result training/evaluation run IDs match persisted runs;
- result GPU-hours equal training + evaluation GPU-hours;
- embedded compute evidence matches those persisted values;
- evaluation protocol hash matches result protocol evidence;
- harvested failure records and repair plans are recoverable when diagnostics exist.

It also checks that no experiment has been persisted as a child of a current checkpoint candidate. Such descendants are treated as uncommitted/orphaned progress rather than silently re-run.

## Current scope

This layer provides discovery and conservative recovery classification:

```python
list_interrupted_recursive_sessions(registry_path)
analyze_recursive_repair_session(registry, session_id)
```

It is deliberately read-only. Actual resume/reconstruction is a separate step and may only consume a `RESUMABLE` report. This separation keeps evidence reconciliation testable and prevents a failed integrity check from having training side effects.
