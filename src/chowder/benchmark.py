from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .closeout import finalize_investigation
from .hypothesis_generation import HypothesisGenerator
from .incident import FailureCapture, SignatureKind, compute_fingerprint
from .investigation import (
    HypothesisTrial,
    Investigation,
    InvestigationStatus,
    RemediationOutcome,
    RemediationRegistry,
    route_failure,
)
from .probes import HardwareCompatibilityProbe, InstalledPackageProbe, ProbeContext
from .ranking import rank_trials
from .remediation_runner import RemediationExperiment
from .replay import ReplayExecutor, ReplayGroundTruth


@dataclass(frozen=True)
class BenchmarkCase:
    """One incident to run through the investigation loop and score: the
    raw evidence, what a ``ReplayExecutor`` should say about every patch a
    generator might plausibly propose, and the classification a human
    reviewer expects (pre-registered, for the hidden set, in
    docs/HIDDEN_SET_FREEZE.md -- this field is where that prediction gets
    checked against the classifier's actual output).
    """

    capture: FailureCapture
    ground_truth: ReplayGroundTruth
    expected_signature_kind: SignatureKind


def run_investigation(
    capture: FailureCapture,
    ground_truth: ReplayGroundTruth,
    generator: HypothesisGenerator,
    registry: RemediationRegistry,
    *,
    gpu_hour_budget: float = 1.0,
    investigation_id: str,
) -> tuple[Investigation, RemediationRegistry]:
    """route -> probe -> generate -> rank -> attempt ranked candidates in
    order until resolved or the budget runs out -> finalize if resolved,
    else mark abandoned explicitly.

    The one orchestration loop every dev-fixture run (Task 6), hidden-set
    run, and benchmark score (Task 8) drives an incident through -- kept
    here, in source, rather than re-implemented per test file, since Task 8
    needs it for real and Task 6 already proved its shape works.
    """
    fingerprint = compute_fingerprint(capture)
    routed = route_failure(
        capture, fingerprint, registry, gpu_hour_budget=gpu_hour_budget, investigation_id=investigation_id
    )
    if not isinstance(routed, Investigation):
        raise ValueError(
            f"incident {capture.incident_id!r} already has a resolved remediation in "
            "the registry before this run started -- benchmark/dev-set cases must be "
            "first-time incidents against a registry that has never seen them"
        )
    investigation = routed

    context = ProbeContext(capture=capture, fingerprint=fingerprint, registry=registry)
    probe_results = (
        HardwareCompatibilityProbe().run(context),
        InstalledPackageProbe().run(context),
    )

    trials: list[HypothesisTrial] = []
    for candidate in generator.generate(fingerprint):
        trial = investigation.add_hypothesis(
            candidate.hypothesis,
            config_patch=candidate.config_patch,
            estimated_gpu_hours=candidate.estimated_gpu_hours,
            registry=registry,
        )
        for result in probe_results:
            investigation.record_probe(trial, result)
        trials.append(trial)

    for trial in rank_trials(trials):
        if investigation.remaining_budget() <= 0:
            break
        experiment = RemediationExperiment(
            executor=ReplayExecutor(ground_truth), gpu_hours_per_attempt=trial.estimated_gpu_hours
        )
        record = experiment.run(investigation, trial, environment=capture.environment)
        if record.outcome is RemediationOutcome.RESOLVED:
            investigation.resolve(trial, record)
            break
        investigation.record_failed_trial(trial, record)

    if investigation.status is InvestigationStatus.RESOLVED:
        remediation_record, _trail = finalize_investigation(investigation)
        registry = registry.with_record(remediation_record)
    elif investigation.status is not InvestigationStatus.ABANDONED:
        # Either no candidate existed at all (generator has nothing for
        # this signature_kind -- still InvestigationStatus.OPEN) or every
        # candidate was tried and none resolved with budget left over
        # (still HYPOTHESIS_TESTING). Both are genuinely abandoned; neither
        # should be left in a non-terminal status just because budget
        # exhaustion specifically didn't happen to be what stopped it.
        investigation.abandon()

    return investigation, registry


# --- false-blame: structured namespace check, not free-text keyword match --
#
# Fixed here, not decided per-case: a config_patch key's *namespace* --
# dotted prefix if it has one, else the bare key -- classifies as touching
# "infrastructure" (driver/library/hardware/network/kernel-provisioning) or
# "model_data" (dataset/hyperparameters). For a case whose classified
# signature_kind is infrastructure-rooted (every kind this project defines
# except UNKNOWN -- this whole taxonomy is infra-failure-focused, no
# "model quality" kind exists), a patch that touches a model_data-namespace
# key is false blame: treating an infra problem as if it needed a
# model/data-side fix. This checks the *structured* config_patch a
# hypothesis actually proposes, not free text the same generator being
# graded also wrote.
_NAMESPACE_BY_KEY_OR_PREFIX: Mapping[str, str] = {
    # dotted-prefix namespaces, matching the plan's own examples verbatim
    "driver": "infrastructure",
    "library": "infrastructure",
    "hardware": "infrastructure",
    "network": "infrastructure",
    "kernel_metadata": "infrastructure",
    "dataset": "model_data",
    "hyperparameters": "model_data",
    # flat keys this project's generator table (Tasks 3-7) actually uses,
    # predating this namespace scheme -- mapped explicitly here rather than
    # retrofitting every prior config_patch to a dotted key.
    "cudnn_enabled": "infrastructure",
    "transformers_version": "infrastructure",
    "device_map": "infrastructure",
    "allocator_conf": "infrastructure",
    "resume_download": "infrastructure",
    "resync_kernel_dataset": "infrastructure",
    "dataset_path_resolution": "infrastructure",
    "max_length": "model_data",  # a training-shape knob, not infra -- see
    # CaseScore.avoided_false_blame's docstring for what this means for the
    # two dev/hidden OOM cases that resolve via this key.
}


def _namespace_of(key: str) -> str | None:
    prefix = key.split(".", 1)[0]
    return _NAMESPACE_BY_KEY_OR_PREFIX.get(prefix) or _NAMESPACE_BY_KEY_OR_PREFIX.get(key)


def _namespace_verdict(config_patch: Mapping[str, Any]) -> bool | None:
    """True: every key in this patch is infrastructure-namespaced. False:
    at least one key is model_data-namespaced (false blame). None: a key's
    namespace isn't classifiable at all -- undetermined, not a guess."""
    namespaces = {_namespace_of(key) for key in config_patch}
    if not namespaces or None in namespaces:
        return None
    return namespaces == {"infrastructure"}


_INFRASTRUCTURE_ROOT_CAUSE_KINDS = frozenset(SignatureKind) - {SignatureKind.UNKNOWN}


@dataclass(frozen=True)
class CaseScore:
    """The 9 original benchmark dimensions for one incident.

    ``trials_to_resolution`` is a proxy for "time to root cause": this
    benchmark replays pre-recorded outcomes (``ReplayExecutor``) with no
    real wall-clock cost, so there is no genuine duration to measure --
    counting attempts is the closest honest substitute, not a stand-in
    presented as if it were real elapsed time.

    ``artifact_preserved`` is ``None`` whenever the incident's
    ``partial_artifact_ref`` is unset, which is every dev and hidden
    fixture as of this benchmark -- none currently models a live
    partial-download scenario, so this dimension has no real signal yet
    across the full case set. Recorded honestly rather than hidden.

    ``avoided_false_blame`` is ``None`` when no trial was ever attempted
    (nothing to check), otherwise the structured namespace verdict on the
    resolving trial's patch (or the last attempted trial's, if abandoned).
    """

    incident_id: str
    expected_signature_kind: SignatureKind
    actual_signature_kind: SignatureKind
    correct_classification: bool
    final_status: InvestigationStatus
    recovery_success: bool
    gpu_hours_spent: float
    gpu_hours_wasted: float
    unnecessary_retries: int
    trials_to_resolution: int | None
    avoided_repeated_interventions: bool
    reproducible: bool | None
    artifact_preserved: bool | None
    avoided_false_blame: bool | None
    resolving_config_patch: Mapping[str, Any] | None


def score_case(case: BenchmarkCase, investigation: Investigation) -> CaseScore:
    actual_kind = investigation.fingerprint.signature_kind
    attempted = [t for t in investigation.trials if t.remediation is not None]
    resolving = next(
        (t for t in attempted if t.remediation is not None and t.remediation.outcome is RemediationOutcome.RESOLVED),
        None,
    )

    gpu_hours_wasted = sum(
        t.remediation.gpu_hours_spent
        for t in attempted
        if t.remediation is not None and t.remediation.outcome is not RemediationOutcome.RESOLVED
    )

    patch_signatures = [tuple(sorted(t.config_patch.items())) for t in investigation.trials]
    avoided_repeated_interventions = len(patch_signatures) == len(set(patch_signatures))

    resolving_patch: Mapping[str, Any] | None = None
    reproducible: bool | None = None
    avoided_false_blame: bool | None = None
    trials_to_resolution: int | None = None

    if resolving is not None and resolving.remediation is not None:
        resolving_patch = resolving.config_patch
        trials_to_resolution = len(attempted)
        fresh_registry = RemediationRegistry().with_record(resolving.remediation)
        reproducible = fresh_registry.lookup(investigation.fingerprint) is resolving.remediation
        if actual_kind in _INFRASTRUCTURE_ROOT_CAUSE_KINDS:
            avoided_false_blame = _namespace_verdict(resolving_patch)
    elif attempted:
        last_attempt = attempted[-1]
        if actual_kind in _INFRASTRUCTURE_ROOT_CAUSE_KINDS:
            avoided_false_blame = _namespace_verdict(last_attempt.config_patch)

    ref = case.capture.partial_artifact_ref
    artifact_preserved = None if ref is None else Path(ref).is_file()

    return CaseScore(
        incident_id=case.capture.incident_id,
        expected_signature_kind=case.expected_signature_kind,
        actual_signature_kind=actual_kind,
        correct_classification=actual_kind == case.expected_signature_kind,
        final_status=investigation.status,
        recovery_success=investigation.status is InvestigationStatus.RESOLVED,
        gpu_hours_spent=investigation.gpu_hours_spent,
        gpu_hours_wasted=gpu_hours_wasted,
        unnecessary_retries=max(0, len(attempted) - 1),
        trials_to_resolution=trials_to_resolution,
        avoided_repeated_interventions=avoided_repeated_interventions,
        reproducible=reproducible,
        artifact_preserved=artifact_preserved,
        avoided_false_blame=avoided_false_blame,
        resolving_config_patch=resolving_patch,
    )


@dataclass(frozen=True)
class BenchmarkReport:
    """Per-case results, dev and hidden kept separate and never combined
    into one aggregate number -- especially not the hidden set, which at 8
    cases is far too small to support a generalization claim. Read each
    hidden case individually; a mismatch between ``expected_signature_kind``
    and ``actual_signature_kind`` there is a finding to write up, not a
    score to average away.
    """

    dev_scores: tuple[CaseScore, ...]
    hidden_scores: tuple[CaseScore, ...]

    def render_table(self) -> str:
        header = (
            f"{'set':6} {'incident_id':52} {'expected':24} {'actual':24} "
            f"{'status':16} {'trials':7} {'wasted':7} {'blame':12}"
        )
        lines = [header]
        for label, scores in (("dev", self.dev_scores), ("hidden", self.hidden_scores)):
            for s in scores:
                blame = (
                    "n/a"
                    if s.avoided_false_blame is None
                    else ("ok" if s.avoided_false_blame else "FALSE-BLAME")
                )
                trials = "-" if s.trials_to_resolution is None else str(s.trials_to_resolution)
                lines.append(
                    f"{label:6} {s.incident_id:52} {s.expected_signature_kind.value:24} "
                    f"{s.actual_signature_kind.value:24} {s.final_status.value:16} {trials:7} "
                    f"{s.gpu_hours_wasted:7.2f} {blame:12}"
                )
        return "\n".join(lines)


def run_benchmark(
    dev_cases: Sequence[BenchmarkCase],
    hidden_cases: Sequence[BenchmarkCase],
    generator: HypothesisGenerator,
) -> BenchmarkReport:
    """Run the dev set, then the hidden set, each against its own fresh
    registry -- no dev-set resolution history leaks into the hidden pass,
    keeping it a genuine held-out evaluation rather than a warmed-up one.
    """
    dev_scores = []
    registry = RemediationRegistry()
    for index, case in enumerate(dev_cases):
        investigation, registry = run_investigation(
            case.capture, case.ground_truth, generator, registry, investigation_id=f"bench-dev-{index}"
        )
        dev_scores.append(score_case(case, investigation))

    hidden_scores = []
    registry = RemediationRegistry()
    for index, case in enumerate(hidden_cases):
        investigation, registry = run_investigation(
            case.capture, case.ground_truth, generator, registry, investigation_id=f"bench-hidden-{index}"
        )
        hidden_scores.append(score_case(case, investigation))

    return BenchmarkReport(dev_scores=tuple(dev_scores), hidden_scores=tuple(hidden_scores))
