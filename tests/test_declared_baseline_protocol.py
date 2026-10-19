"""A fixed (declared) baseline carries the protocol identity it was declared
under, so the goal lifecycle can assess it instead of stopping the project
before its first experiment (Gen-2 run 9, 2026-10-19)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from chowder.models import ExperimentResult
from chowder.project_runner import _declared_baseline, _measured_protocol_digest
from chowder.protocol import protocol_fingerprint


class _Project:
    def __init__(self, baseline: ExperimentResult, evaluation: dict) -> None:
        self.baseline = baseline
        self.config = {"evaluation": evaluation}


EVAL = {"suites": [{"name": "math500", "dataset": "slice.json", "scoring": "normalized_exact_match"}]}


def _baseline(evidence: dict | None = None) -> ExperimentResult:
    return ExperimentResult(experiment_id="baseline", metrics={"math500": 0.0}, gpu_hours=0.0,
                            artifact_ref=None, evidence=evidence or {})


def test_a_declared_baseline_without_protocol_evidence_carries_the_config_fingerprint():
    stamped = _declared_baseline(_Project(_baseline(), EVAL))
    assert _measured_protocol_digest(stamped) == protocol_fingerprint({"evaluation": EVAL})
    assert "declared fixed baseline" in stamped.evidence["evaluation_protocol_source"]
    assert stamped.metrics == {"math500": 0.0}


def test_a_baseline_with_measured_protocol_evidence_is_untouched():
    measured = _baseline({"evaluation_protocol_sha256": "a" * 64})
    assert _declared_baseline(_Project(measured, EVAL)) is measured
