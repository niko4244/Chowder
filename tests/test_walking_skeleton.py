from dataclasses import replace

from chowder.closeout import finalize_investigation
from chowder.hypothesis_generation import RuleBasedGenerator
from chowder.incident import SignatureKind, compute_fingerprint
from chowder.investigation import (
    DiagnosticProbeResult,
    HypothesisTrial,
    Investigation,
    InvestigationStatus,
    RemediationOutcome,
    RemediationRegistry,
    config_patch_digest,
    route_failure,
)
from chowder.models import Hypothesis
from chowder.probes import HardwareCompatibilityProbe, InstalledPackageProbe, ProbeContext
from chowder.ranking import rank_trials
from chowder.remediation_runner import RemediationExperiment
from chowder.replay import ReplayExecutor, ReplayGroundTruth

from fixtures_incidents import PEFT_KBIT_PREP_OOM, QWEN3_5_CONV1D_NO_ENGINE


def test_walking_skeleton_resolves_known_incident_end_to_end():
    """One real dev-set incident, driven through the entire loop the plan's
    architecture diagram claims: route_failure -> Investigation -> probes ->
    hypothesis -> rank_trials -> RemediationExperiment -> resolve ->
    finalize_investigation -> RemediationRegistry.with_record -> and back
    around through route_failure again, now hitting the known-remediation
    fast path instead of opening a second investigation."""
    capture = QWEN3_5_CONV1D_NO_ENGINE
    fingerprint = compute_fingerprint(capture)
    registry = RemediationRegistry()

    routed = route_failure(
        capture, fingerprint, registry, gpu_hour_budget=1.0, investigation_id="inv-skeleton-1"
    )
    assert isinstance(routed, Investigation)
    investigation = routed

    context = ProbeContext(capture=capture, fingerprint=fingerprint, registry=registry)
    probe_results = (
        HardwareCompatibilityProbe().run(context),
        InstalledPackageProbe().run(context),
    )
    assert probe_results[0].observation["compatible"] is True  # T4/sm_75, not the wrong-GPU case

    candidates = RuleBasedGenerator().generate(fingerprint)
    assert len(candidates) == 1
    candidate = candidates[0]

    trial = investigation.add_hypothesis(
        candidate.hypothesis, config_patch=candidate.config_patch, registry=registry
    )
    for result in probe_results:
        investigation.record_probe(trial, result)

    ranked = rank_trials(investigation.trials)
    assert ranked == (trial,)
    assert trial.rank == 2.0  # two probes recorded against this trial

    truth = ReplayGroundTruth(
        fingerprint_sha256=fingerprint.fingerprint_sha256,
        outcomes={config_patch_digest(candidate.config_patch): RemediationOutcome.RESOLVED},
    )
    experiment = RemediationExperiment(executor=ReplayExecutor(truth))
    record = experiment.run(investigation, trial, environment=capture.environment)
    assert record.outcome is RemediationOutcome.RESOLVED

    investigation.resolve(trial, record)
    assert investigation.status is InvestigationStatus.RESOLVED

    remediation_record, audit_trail = finalize_investigation(investigation)
    assert len(audit_trail.trials) == 1
    assert audit_trail.trials[0].outcome is RemediationOutcome.RESOLVED

    fresh_registry = RemediationRegistry().with_record(remediation_record)
    assert fresh_registry.lookup(fingerprint) is remediation_record

    routed_again = route_failure(
        capture, fingerprint, fresh_registry, gpu_hour_budget=1.0, investigation_id="inv-skeleton-2"
    )
    assert routed_again is remediation_record


def test_generator_returns_nothing_for_unmodeled_signature():
    """UNKNOWN incidents get no fabricated hypothesis -- the generator must
    fail safe rather than guess a plausible-looking fix for a class no rule
    recognizes. (As of Task 6 the table covers all 9 signature kinds the
    real dev fixtures classify into, so this test needs a genuinely
    unclassifiable message rather than reusing a fixture that's now
    covered.)"""
    unclassified = replace(
        PEFT_KBIT_PREP_OOM,
        exception_type="RuntimeError",
        exception_message="an entirely novel failure mode no rule recognizes",
        traceback_text="RuntimeError: an entirely novel failure mode no rule recognizes",
    )
    fingerprint = compute_fingerprint(unclassified)
    assert fingerprint.signature_kind is SignatureKind.UNKNOWN
    assert RuleBasedGenerator().generate(fingerprint) == ()


def test_rank_trials_orders_by_probe_corroboration_count_descending():
    def _trial(probe_count: int) -> HypothesisTrial:
        trial = HypothesisTrial(
            hypothesis=Hypothesis(observation="o", suspected_cause="c", intervention="i"),
            config_patch={},
        )
        trial.probe_results = tuple(
            DiagnosticProbeResult(probe_id=f"p{i}", description="d", observation={})
            for i in range(probe_count)
        )
        return trial

    low, high = _trial(1), _trial(3)
    ranked = rank_trials([low, high])
    assert ranked == (high, low)
    assert high.rank == 3.0
    assert low.rank == 1.0
