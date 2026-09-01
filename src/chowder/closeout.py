from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from .investigation import (
    HypothesisTrial,
    Investigation,
    InvestigationStatus,
    RemediationOutcome,
    RemediationRecord,
)


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TrialSummary:
    """One trial's full evidence, typed -- not a free-text note.

    Captures enough that a future investigation into the same or a related
    fingerprint can see not just what worked, but what was tried and
    rejected along the way, and why (via the probe evidence each trial
    accumulated before its remediation was attempted).
    """

    hypothesis_observation: str
    hypothesis_suspected_cause: str
    hypothesis_intervention: str
    config_patch: Mapping[str, Any]
    rank: float | None
    probe_observations: tuple[Mapping[str, Any], ...]
    outcome: RemediationOutcome | None
    remediation_id: str | None


def _summarize_trial(trial: HypothesisTrial) -> TrialSummary:
    return TrialSummary(
        hypothesis_observation=trial.hypothesis.observation,
        hypothesis_suspected_cause=trial.hypothesis.suspected_cause,
        hypothesis_intervention=trial.hypothesis.intervention,
        config_patch=trial.config_patch,
        rank=trial.rank,
        probe_observations=tuple(r.observation for r in trial.probe_results),
        outcome=trial.remediation.outcome if trial.remediation else None,
        remediation_id=trial.remediation.remediation_id if trial.remediation else None,
    )


def _trial_summary_digest_payload(summary: TrialSummary) -> Mapping[str, Any]:
    return {
        "hypothesis_observation": summary.hypothesis_observation,
        "hypothesis_suspected_cause": summary.hypothesis_suspected_cause,
        "hypothesis_intervention": summary.hypothesis_intervention,
        "config_patch": dict(summary.config_patch),
        "rank": summary.rank,
        "probe_observations": [dict(o) for o in summary.probe_observations],
        "outcome": summary.outcome.value if summary.outcome else None,
        "remediation_id": summary.remediation_id,
    }


@dataclass(frozen=True)
class AuditTrail:
    """The full, ordered record of every trial an investigation tried --
    not just the one that won.

    This is what makes "is the final environment reproducible" a mechanical
    question rather than a claim: a future attempt can diff its own state
    against exactly what was tried, in what order, with what evidence, not
    just read the final answer in isolation.
    """

    investigation_id: str
    fingerprint_sha256: str
    trials: tuple[TrialSummary, ...]
    trail_sha256: str


def finalize_investigation(investigation: Investigation) -> tuple[RemediationRecord, AuditTrail]:
    """Close out a resolved investigation into a registry-ready record plus
    its full audit trail.

    Requires the investigation to actually be RESOLVED -- finalizing an
    abandoned or still-open investigation would produce a record with
    nothing to register, a caller bug this catches rather than papers over.
    If (not expected given the current API, but not structurally
    impossible) more than one trial somehow carries a RESOLVED remediation,
    the first one in trial order is used, deterministically.
    """
    if investigation.status is not InvestigationStatus.RESOLVED:
        raise ValueError(
            f"cannot finalize investigation {investigation.investigation_id!r} "
            f"with status {investigation.status.value!r} -- only a RESOLVED "
            "investigation has a remediation to register"
        )

    resolving_trial = next(
        (
            t
            for t in investigation.trials
            if t.remediation is not None and t.remediation.outcome is RemediationOutcome.RESOLVED
        ),
        None,
    )
    if resolving_trial is None or resolving_trial.remediation is None:
        raise ValueError(
            f"investigation {investigation.investigation_id!r} is RESOLVED but "
            "no trial carries a RESOLVED remediation -- this should not happen "
            "and indicates a bug in how the investigation reached this status"
        )

    summaries = tuple(_summarize_trial(t) for t in investigation.trials)
    trail_payload = {
        "investigation_id": investigation.investigation_id,
        "fingerprint_sha256": investigation.fingerprint.fingerprint_sha256,
        "trials": [_trial_summary_digest_payload(s) for s in summaries],
    }
    trail = AuditTrail(
        investigation_id=investigation.investigation_id,
        fingerprint_sha256=investigation.fingerprint.fingerprint_sha256,
        trials=summaries,
        trail_sha256=_canonical_digest(trail_payload),
    )
    return resolving_trial.remediation, trail
