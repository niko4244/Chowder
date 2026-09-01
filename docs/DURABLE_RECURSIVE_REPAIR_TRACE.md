# Durable Recursive Repair Trace

Bounded recursive repair now records controller state in the same SQLite database used by `RunRegistry` whenever a registry is attached to the runner.

The trace is stored in separate tables so the mature experiment/result schema remains unchanged.

## Session record

Each invocation receives a unique session ID and records:

- policy;
- provider identity;
- repair variant metadata;
- baseline experiment ID;
- initial candidate IDs;
- current checkpoint state;
- terminal status and stop reason.

Checkpoint state contains:

- completed repair depth;
- current candidate IDs;
- stable failure-signature occurrence counts;
- previous selected target score;
- remaining GPU-hour budget;
- promoted experiment ID, when present.

## Hop event

Each completed repair hop records:

- depth;
- target experiment ID;
- stable failure signature;
- target score;
- score improvement from the prior selected target;
- remaining budget after the hop;
- produced candidate IDs;
- promoted experiment ID, when present.

The hop insert and session checkpoint update occur in the same SQLite transaction.

## Failure semantics

Expected controller stop conditions finish the session with the corresponding bounded-repair stop reason.

Unexpected exceptions are not converted into successful stop conditions. The session is marked:

```text
status = failed
stop_reason = error
```

and the original exception is re-raised.

## Crash model

A process interruption after a committed hop leaves a durable checkpoint showing exactly which hop completed and what controller state followed it. A future resume/reconciliation layer can therefore restart from durable evidence instead of reconstructing signature counts or depth from guesswork.

This PR provides the durable audit/checkpoint substrate; automatic resume is intentionally a separate feature because it must reconcile already-persisted experiments, artifacts, evaluation evidence, and compute accounting before re-entering training.
