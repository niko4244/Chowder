# Registered observer calibration chain-of-custody

The REAP observer previously read calibration rows once at startup but
computed `observation.json.calibration_sha256` by reading that pathname again
after the multi-layer model run. A concurrent edit, replacement or truncation
could leave the observed hidden states based on one set of inputs while
attesting the SHA-256 of a different dataset. This is a provenance defect,
not evidence that the historical observation was actually attacked.

The observer now reads the calibration bytes into a single immutable
snapshot, parses the rows from those bytes and records their SHA-256 as the
input witness. It checks the saved pathname against that witness before
publishing each layer's observation stats and before publishing
`observation.json`. If bytes differ, it refuses; partial layer artifacts
must not be scored or adopted into a completed frozen observation.

The output directory must not already exist or overlap either the original
model, weight directory or calibration input. This prevents an attempted
rerun from overwriting prior experiment artifacts. The change does not
silently rescore, retokenize, change thresholds, claim a completed D0,
authenticate tens of gigabytes of weight payloads, or start any GPU work.
Like all at-rest hash checks, it cannot prevent a malicious source from being
replaced and restored entirely between checks; immutable storage or an
external trusted digest remains a separate provenance requirement.

CPU-only regression command:

```powershell
$env:PYTHONPATH='src;tests'
$env:CHOWDER_NO_GPU='1'
python -B -m pytest -q tests/test_reap_observe_calibration_snapshot.py
```
