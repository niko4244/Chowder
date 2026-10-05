"""The judged evidence set, and branch protection against the trusted ancestor.

``write_certification_evidence`` writes the arms a promotion is judged from and
binds each to the bytes it measured. ``certify_before_lineage`` runs the same
production certification the frozen judge runs, *before* the lineage record is
written, so a candidate the judge would refuse cannot be recorded as promoted
and then audited.

Extracted from ``campaign_runner``, which keeps the orchestration and the
two injection seams. Nothing here re-implements a decision another module owns.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import json
from pathlib import Path
from chowder.evals.result import BenchmarkRun, EvalReport
from ..candidate_evaluation import CANDIDATE_EVALUATION_NOT_PRODUCED, CandidateEvaluation, CandidateEvaluationRefusal, CandidateEvaluator, EvaluationRequest, coerce_evaluation, evaluation_detail, validate_candidate_report, validate_evaluation_cost, write_candidate_evaluation
from ..campaign import PROMOTION_POLICY_VERSION, CampaignManifest, CampaignManifestError, settle_campaign, settle_campaign_projection, stops_on_admission_refusal, stops_on_campaign_overrun
from ..certification import FAIL, PASS, UNKNOWN, Certification, CertificationRow, ProtectionPolicy, certify_run_root, digest_of
from ..training_binding import STATUS_SUCCEEDED, GrowthEnvelope, SubprocessTrainingFn, default_runner, directory_digest
from ..campaign_controllers.contracts import CampaignRunRefusal, FIELD_ENFORCEMENT, NON_BEHAVIORAL_FIELDS, assert_every_field_enforced
from ..campaign_controllers.declared import DECLARED_INPUT_REQUIREMENTS, _require_path, require_declared_inputs, undeclared_inputs

from ..contamination import ContaminationFirewall


CERTIFICATION_EVIDENCE: Mapping[str, str] = {
    "candidate": "candidate_evaluation.json",
    "parent": "parent_evaluation.json",
    "ancestor": "baseline_evaluation.json",
    "selection": "chosen_candidate.json",
    "contamination": "gen2_contamination_manifest.json",
}


_EVIDENCE_ARM_SOURCES: Mapping[str, str] = {
    "parent": "parent_eval_report_path",
    "ancestor": "baseline_eval_report_path",
}


def write_certification_evidence(
    manifest: CampaignManifest,
    *,
    root: Path,
    selected: Mapping[str, Any] | None,
    candidate: CandidateEvaluation | None,
) -> tuple[str, ...]:
    """Materialise the judged artifacts from the run's own evidence.

    Every file here is derived from something the run actually has: the
    candidate arm this run measured over the artifact it selected, the declared
    arms it reads (copied verbatim, so a row's ``measurement_origin`` is the
    evaluator's declaration and not the runner's election), the contamination
    manifest the firewall bound, and the adapter artifact the winning attempt
    produced -- digested over its bytes.

    An input the manifest does not declare produces **no** file rather than a
    placeholder: the judge then reports that gate UNKNOWN, which is the honest
    answer. Nothing here can invent a measurement the run never took, and no
    gate is loosened by writing these: they are the same numbers the run
    already adjudicated with, at the paths the frozen judge reads.

    Every declared input is read and validated *before* the first file is
    written, so a refusal (a declared report that does not exist, or one whose
    rows cannot be parsed) never leaves a half-materialised evidence set for the
    judge to read.
    """
    arms: list[tuple[str, Path, EvalReport]] = []
    for arm, field_name in _EVIDENCE_ARM_SOURCES.items():
        declared = str(getattr(manifest, field_name))
        if not declared:
            continue
        source = _require_path(
            declared, field_name, purpose="the judged evidence set is built from it"
        )
        # An unreadable report must refuse here rather than reach the judge as an
        # arm whose rows cannot be parsed.
        arms.append((arm, source, EvalReport.load(source)))
    if candidate is not None:
        # Produced by this run, so it is written rather than read: nothing here
        # can substitute a report the campaign declared.
        arms.append(("candidate", None, candidate.report))
    artifacts = _measurement_artifacts(arms, root=root)
    contamination: Path | None = None
    if manifest.contamination_manifest_path:
        contamination = _require_path(
            manifest.contamination_manifest_path,
            "contamination_manifest_path",
            purpose="the run's contamination evidence is copied for the judge",
        )
        if not isinstance(json.loads(contamination.read_text(encoding="utf-8")), Mapping):
            raise CampaignRunRefusal(
                f"contamination manifest {contamination} is not a JSON object, so "
                "it cannot be the contamination evidence the judge reads"
            )

    written: list[str] = []
    for arm, source, _report in arms:
        destination = root / CERTIFICATION_EVIDENCE[arm]
        if source is None:
            # The run's own validated evaluation object rather than a rebuilt
            # one: it is the object whose measured cost the ledger charged.
            assert candidate is not None  # the only source-less arm
            write_candidate_evaluation(
                candidate, root=root, name=CERTIFICATION_EVIDENCE[arm]
            )
        else:
            destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        written.append(destination.name)

    # The measurements those rows declare, so the run root carries the bytes
    # each row's digest is computed over.
    for origin, relative in artifacts:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(origin.read_bytes())
        written.append(str(relative))

    if contamination is not None:
        destination = root / CERTIFICATION_EVIDENCE["contamination"]
        destination.write_text(contamination.read_text(encoding="utf-8"), encoding="utf-8")
        written.append(destination.name)

    chosen = _chosen_candidate_document(selected)
    if chosen is not None:
        destination = root / CERTIFICATION_EVIDENCE["selection"]
        destination.write_text(
            json.dumps(chosen, indent=2, sort_keys=True), encoding="utf-8"
        )
        written.append(destination.name)
    return tuple(written)


def _measurement_artifacts(
    arms: Sequence[tuple[str, Path | None, EvalReport]],
    *,
    root: Path,
) -> list[tuple[Path, Path]]:
    """Every measurement artifact the arms' rows name, as (origin, relative) pairs.

    A row that names an artifact is bound to it: the judge recomputes the digest
    the row declares over those bytes, so a run root that lacks them carries a
    measurement nobody can verify. A reference that resolves to nothing refuses
    the run -- evidence is not assembled around a missing artifact -- and two arms
    that name the same relative path with different content refuse too, because
    one run root cannot be both.
    """
    artifacts: list[tuple[Path, Path]] = []
    seen: dict[Path, Path] = {}
    for arm, source, report in arms:
        for run in report.runs:
            reference = str(run.raw_artifact_ref or "")
            if not reference:
                continue
            relative = Path(reference)
            if relative.is_absolute():
                if source is None:
                    # The candidate arm is produced by this run, so it has no
                    # excuse for pointing outside it: a certified verdict that
                    # depends on an external directory can be destroyed by
                    # deleting that directory.
                    raise CampaignRunRefusal(
                        f"the candidate arm's {run.benchmark_qualified_id} row "
                        f"names absolute artifact {reference!r}; the run's own "
                        "measurement must live inside the run root, so the "
                        "evidence a verdict rests on cannot be moved out from "
                        "under it"
                    )
                # A declared arm may name an absolute artifact; the judge
                # hashes it there, so the dependence is recorded below.
                continue
            if ".." in relative.parts:
                raise CampaignRunRefusal(
                    f"the {arm} arm's {run.benchmark_qualified_id} row names raw "
                    f"artifact {reference!r}, which points outside the run root; the "
                    "judged evidence set cannot follow it"
                )
            if source is None:
                # The candidate arm's relative refs are resolved against the run
                # root: the evaluator wrote its measurements into the run it was
                # asked to measure for, so there is nothing to carry.
                origin = root / relative
                if not origin.is_file():
                    raise CampaignRunRefusal(
                        f"the candidate arm's {run.benchmark_qualified_id} row names "
                        f"raw artifact {reference!r}, which the evaluation did not "
                        "write into the run root, so the measurement it declares "
                        "cannot be verified"
                    )
                previous = seen.get(relative)
                if previous is not None and previous.read_bytes() != origin.read_bytes():
                    raise CampaignRunRefusal(
                        f"two arms name {reference!r} with different content, so one "
                        "run root cannot carry both measurements as verifiable evidence"
                    )
                seen[relative] = origin
                continue
            origin = source.parent / relative
            if not origin.is_file():
                raise CampaignRunRefusal(
                    f"the {arm} arm's {run.benchmark_qualified_id} row names raw "
                    f"artifact {reference!r}, which does not exist next to {source}, "
                    "so the measurement it declares cannot be verified"
                )
            previous = seen.get(relative)
            if previous is not None and previous.read_bytes() != origin.read_bytes():
                raise CampaignRunRefusal(
                    f"two arms name {reference!r} with different content, so one run "
                    "root cannot carry both measurements as verifiable evidence"
                )
            seen[relative] = origin
            artifacts.append((origin, relative))
    return artifacts


def _chosen_candidate_document(
    selected: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """The winner's identity, or None when the run produced no artifact.

    The digest is the one the binding already measured over the artifact's
    bytes, recomputed here only when the attempt recorded none. A candidate
    with no artifact gets no record rather than a formatted placeholder the
    judge would have to reject.
    """
    if not selected:
        return None
    artifact_ref = selected.get("artifact_ref")
    if not isinstance(artifact_ref, str) or not artifact_ref.strip():
        return None
    artifact = Path(artifact_ref)
    if not artifact.exists():
        return None
    digest = selected.get("artifact_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        try:
            digest, _entries = directory_digest(artifact)
        except OSError:
            return None
    return {
        "recipe_id": str(selected.get("recipe_id", "")),
        "artifact_ref": str(artifact),
        "artifact_sha256": digest,
        "attempt": selected.get("attempt"),
    }


#: Placed here rather than in ``training``: what it preflights is this
#: module's judged evidence set -- the declared parent and ancestor arms --
#: not the executor. The call graph put it in ``training`` first, and the
#: undefined ``_EVIDENCE_ARM_SOURCES`` it produced is what corrected that.
def _preflight_arms(manifest: CampaignManifest) -> str:
    """Load the declared evidence arms before any compute, or say what is wrong.

    Everything here is knowable without touching a GPU: whether the parent and
    trusted-ancestor reports exist and parse, and whether the contamination
    evidence the binder already loaded is a real declaration. A run that would
    refuse on them should refuse before it trains, not after.
    """
    checked: list[str] = []
    for arm, field_name in _EVIDENCE_ARM_SOURCES.items():
        declared = str(getattr(manifest, field_name))
        if not declared:
            raise CampaignRunRefusal(
                f"the campaign declares no {field_name}, so the {arm} arm of "
                "adjudication and branch protection cannot be read; the run "
                "refuses before compute rather than after it"
            )
        source = _require_path(
            declared,
            field_name,
            purpose=f"the {arm} arm is read before compute",
        )
        try:
            report = EvalReport.load(source)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise CampaignRunRefusal(
                f"the declared {arm} arm {source} is unreadable: {error}; a run "
                "cannot adjudicate against evidence it cannot parse"
            ) from error
        if not report.runs:
            raise CampaignRunRefusal(
                f"the declared {arm} arm {source} carries no measured rows, so it "
                "cannot be the evidence this run adjudicates against"
            )
        checked.append(f"{arm}={report.generation_version or '(unlabelled)'}/{len(report.runs)} rows")
    return ", ".join(checked)


def certify_before_lineage(
    manifest: CampaignManifest,
    *,
    root: Path,
    selected: Mapping[str, Any] | None,
) -> Certification:
    """Certify the run's own evidence, before the lineage record is written.

    ``_finalize`` records a generation only for a PROMOTED decision, so the
    certification verdict is applied to the decision *first*: a candidate that
    the frozen judge would refuse at audit time -- because the trusted-ancestor
    arm is missing, because a protected slice was never really measured, or
    because the evaluation names bytes the campaign did not select -- cannot be
    recorded as promoted and then audited. The mechanism is the production one
    (:mod:`chowder.growth.certification`), the same code the frozen judge runs,
    so the run and the audit cannot disagree about what the evidence says.

    Arms are read from the judged evidence set this run wrote, never from a
    report handed to the campaign: ``candidate_evaluation.json`` is bound to the
    digest of the artifact the run selected, the parent arm to the frozen parent
    adapter, and the ancestor arm to the frozen base model.
    """
    protection = manifest.protection
    if (
        not protection.trusted_ancestor_version
        or protection.slice_regression_max is None
        or protection.protocol is None
    ):
        return Certification(
            status=UNKNOWN,
            rows=(
                CertificationRow(
                    requirement="a declared branch-protection policy",
                    status=UNKNOWN,
                    detail=(
                        "the campaign declares no protection policy "
                        "(trusted_ancestor_version + slice_regression_max + "
                        "protocol), so it cannot certify that the branch is "
                        "protected"
                    ),
                ),
            ),
            reasons=(
                "the campaign declares no protection policy, so no generation may "
                "be recorded as promoted",
            ),
        )
    root = Path(root)
    arms: dict[str, Path | None] = {
        "candidate": root / CERTIFICATION_EVIDENCE["candidate"],
        "parent": (
            root / CERTIFICATION_EVIDENCE["parent"]
            if (root / CERTIFICATION_EVIDENCE["parent"]).is_file()
            else None
        ),
        "ancestor": (
            root / CERTIFICATION_EVIDENCE["ancestor"]
            if (root / CERTIFICATION_EVIDENCE["ancestor"]).is_file()
            else None
        ),
    }
    ancestor_path = str(manifest.baseline_eval_report_path or "")
    if not ancestor_path:
        arms["ancestor"] = None
    adapter_digests: dict[str, str] = {}
    selected_digest = str((selected or {}).get("artifact_sha256", ""))
    if selected_digest:
        adapter_digests["candidate"] = selected_digest
    if manifest.has_parent_adapter() and manifest.parent_adapter_digest:
        adapter_digests["parent"] = manifest.parent_adapter_digest
    base_digests = {"ancestor": manifest.base_model_digest} if manifest.base_model_digest else {}
    return certify_run_root(
        policy=ProtectionPolicy(
            required_protected=tuple(manifest.protected_benchmarks),
            slice_regression_max=float(protection.slice_regression_max),
            trusted_ancestor_version=protection.trusted_ancestor_version,
            protocol=protection.protocol,
            candidate_version=manifest.resolved_candidate_version(),
            parent_version=manifest.parent_version,
        ),
        run_root=root,
        arms=arms,
        expected_adapter_digests=adapter_digests,
        expected_base_digests=base_digests,
    )
