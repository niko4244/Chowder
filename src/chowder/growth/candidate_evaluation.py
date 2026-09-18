"""Candidate evaluation is a run output, never a campaign input.

The candidate arm of the judged evidence set is the one measurement a growth
cycle must *produce*: the evaluation of the artifact this run trained and
selected, taken after selection. A report prepared before the run is not the
same evidence -- whoever prepared it could not have measured the adapter this
run just made -- so it is not accepted as the candidate side of promotion. The
manifest declares its evaluator's *inputs* (the benchmark sets and the
protected mini-slice protocol); the evaluation itself is an output.

Two things make that more than a naming convention:

* **Provenance.** Every row must declare
  ``measurement_origin = MEASURED_THIS_GENERATION``. A parent row, a carried
  reference or an unmeasured row is refused by name rather than being rebound
  as candidate evidence.
* **Identity.** The report must name the bytes it measured (``model_identity``),
  and those must be the bytes the run selected: ``adapter_digest`` equals the
  artifact digest the binding measured over the winning attempt, and
  ``base_model_digest`` equals the declared dense base. A report that cannot say
  what it measured is undecided evidence, not evidence.

Coverage is enforced too: every declared benchmark must have a row (an
unevaluated declaration is a refusal, not an omission), and a benchmark measured
twice is refused because two measurements under one name are ambiguous. A row
for a benchmark no declared set names is *reported* in the run record rather
than refused here: the frozen judge owns "no undeclared row substitutes for a
required slice", and a second copy of that policy is a rule that can drift from
it.

This module owns the contract, not the measurement: the :class:`CandidateEvaluator`
seam is what an evaluator implements, and the production wiring for it is a
separate build (see ``campaign_runner.build_evaluator``). What this module
guarantees is that whatever that evaluator returns is bound to the artifact the
run selected before any verdict can read it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from chowder.evals.result import (
    CARRIED_REFERENCE,
    MEASURED_PARENT,
    MEASURED_THIS_GENERATION,
    SUPPORTED,
    UNMEASURED,
    BenchmarkRun,
    EvalReport,
)

from .compute_cost import ComputeCost

#: The run had no way to measure the artifact it selected.
CANDIDATE_EVALUATION_NOT_PRODUCED = "CANDIDATE_EVALUATION_NOT_PRODUCED"
#: The evaluator produced nothing at all for a declared benchmark set.
CANDIDATE_EVALUATION_EMPTY = "CANDIDATE_EVALUATION_EMPTY"
#: A declared benchmark has no row in the evaluation.
CANDIDATE_EVALUATION_INCOMPLETE = "CANDIDATE_EVALUATION_INCOMPLETE"
#: A row's provenance is not candidate-measured.
CANDIDATE_EVALUATION_NOT_CANDIDATE_MEASURED = "CANDIDATE_EVALUATION_NOT_CANDIDATE_MEASURED"
#: The report does not name the bytes the run selected.
CANDIDATE_EVALUATION_IDENTITY_UNBOUND = "CANDIDATE_EVALUATION_IDENTITY_UNBOUND"
#: The report's declared generation is not this run's candidate.
CANDIDATE_EVALUATION_WRONG_GENERATION = "CANDIDATE_EVALUATION_WRONG_GENERATION"
#: One benchmark measured more than once in one evaluation.
CANDIDATE_EVALUATION_DUPLICATE_BENCHMARK = "CANDIDATE_EVALUATION_DUPLICATE_BENCHMARK"
#: The evaluation reported no cost, so its compute cannot be charged.
CANDIDATE_EVALUATION_COST_UNREPORTED = "CANDIDATE_EVALUATION_COST_UNREPORTED"
#: A zero cost that names no measurement method is an unreported cost wearing a
#: zero: only an explicit, stated zero may be charged as zero.
CANDIDATE_EVALUATION_COST_UNMEASURED = "CANDIDATE_EVALUATION_COST_UNMEASURED"


class CandidateEvaluationRefusal(RuntimeError):
    """The run could not produce candidate-measured evidence for its artifact.

    Raised rather than defaulted: a campaign that cannot evaluate the candidate
    it trained has nothing to adjudicate, and inventing an arm (or reusing a
    pre-existing report) is exactly the substitution the promotion path refuses.
    """


@dataclass(frozen=True)
class EvaluationRequest:
    """What the evaluator is asked to measure, and for which artifact.

    The artifact and base digests are part of the request rather than appended
    afterwards, so an evaluator cannot be asked to measure one artifact and
    return a report about another: whatever it returns is checked against these.
    """

    cycle_id: str
    candidate_version: str
    base_model_path: str
    base_model_digest: str
    artifact_ref: str
    artifact_sha256: str
    recipe_id: str
    attempt: str
    target_benchmarks: tuple[str, ...] = ()
    protected_benchmarks: tuple[str, ...] = ()
    broad_benchmarks: tuple[str, ...] = ()
    calibration_benchmarks: tuple[str, ...] = ()
    reliability_benchmarks: tuple[str, ...] = ()
    #: The declared mini-slice protocol, verbatim from the campaign's protection
    #: declaration: the evaluator measures under the protocol the certification
    #: will check the rows against, not one of its own choosing.
    protocol: Mapping[str, Any] = field(default_factory=dict)
    output_root: str = ""

    def __post_init__(self) -> None:
        for label, value in (
            ("cycle_id", self.cycle_id),
            ("candidate_version", self.candidate_version),
            ("artifact_ref", self.artifact_ref),
            ("recipe_id", self.recipe_id),
        ):
            if not isinstance(value, str) or not value.strip():
                raise CandidateEvaluationRefusal(
                    f"{label} is required to evaluate a candidate: an evaluation "
                    "that cannot name what it measured is not evidence"
                )
        for label, value in (
            ("artifact_sha256", self.artifact_sha256),
            ("base_model_digest", self.base_model_digest),
        ):
            if not _is_sha256(value):
                raise CandidateEvaluationRefusal(
                    f"{label} must be a sha256 hex digest, got {value!r}; the "
                    "candidate arm is bound to the artifact the run selected, so "
                    "both digests are required before the evaluator runs"
                )
        if not self.sets:
            raise CandidateEvaluationRefusal(
                "the campaign declares no benchmark set to evaluate the candidate "
                "on, so there is nothing the candidate arm could attest"
            )

    @property
    def sets(self) -> Mapping[str, tuple[str, ...]]:
        """The declared benchmark sets, by role."""
        return {
            "target": tuple(self.target_benchmarks),
            "protected": tuple(self.protected_benchmarks),
            "broad": tuple(self.broad_benchmarks),
            "calibration": tuple(self.calibration_benchmarks),
            "reliability": tuple(self.reliability_benchmarks),
        }

    @property
    def declared_benchmarks(self) -> tuple[str, ...]:
        """Every declared benchmark id, deduplicated, in declaration order.

        A benchmark that appears in two sets (gen2 declares the same mini-slices
        protected and broad) is one measurement, not two.
        """
        ordered: list[str] = []
        for role in ("target", "protected", "broad", "calibration", "reliability"):
            for qualified_id in self.sets[role]:
                if qualified_id not in ordered:
                    ordered.append(qualified_id)
        return tuple(ordered)

    def roles_of(self, qualified_id: str) -> tuple[str, ...]:
        return tuple(
            role for role in ("target", "protected", "broad", "calibration", "reliability")
            if qualified_id in self.sets[role]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "candidate_version": self.candidate_version,
            "base_model_path": self.base_model_path,
            "base_model_digest": self.base_model_digest,
            "artifact_ref": self.artifact_ref,
            "artifact_sha256": self.artifact_sha256,
            "recipe_id": self.recipe_id,
            "attempt": self.attempt,
            "benchmarks": {
                role: list(ids) for role, ids in self.sets.items()
            },
            "protocol": dict(self.protocol),
            "output_root": self.output_root,
        }


@dataclass(frozen=True)
class CandidateEvaluation:
    """What an evaluator returns: the measured rows, and what they cost.

    ``cost`` is the compute this evaluation actually consumed. It is charged to
    the cycle's ledger before settlement, so a campaign cannot spend compute on
    evaluation without that spend counting against its own declared ceilings.

    It is therefore **not** optional at the boundary: an evaluation that reports
    no cost is charged as zero, and a zero charge is exactly how a real
    evaluation leg disappears from the campaign's accounting. A zero is legal
    only as an explicit, measured zero (see :func:`validate_evaluation_cost`).
    """

    report: EvalReport
    cost: ComputeCost | None = None


#: The seam. An evaluator takes the request and returns a
#: :class:`CandidateEvaluation` -- the measurement *and* the compute it cost.
#: A bare report is refused rather than defaulted to zero cost.
CandidateEvaluator = Callable[[EvaluationRequest], Any]


def coerce_evaluation(value: Any) -> CandidateEvaluation:
    """Accept the seam's one shape, refusing everything else.

    A bare :class:`~chowder.evals.result.EvalReport` used to be accepted and
    silently became a zero-cost evaluation. It is refused now: the campaign
    charges the evaluation leg against its own ceilings, and an evaluator that
    does not say what it spent cannot be charged.
    """
    if isinstance(value, CandidateEvaluation):
        return value
    if isinstance(value, EvalReport):
        raise CandidateEvaluationRefusal(
            f"{CANDIDATE_EVALUATION_COST_UNREPORTED}: the evaluator returned a "
            "bare EvalReport, which reports no compute; a report without a cost "
            "would be charged as zero, so it is refused. Return a "
            "CandidateEvaluation carrying the measured ComputeCost, and use an "
            "explicit measured zero for a leg that really spent nothing"
        )
    raise CandidateEvaluationRefusal(
        f"the evaluator returned {type(value).__name__}, which is not a "
        "CandidateEvaluation; the run will not guess at measured evidence"
    )


def validate_evaluation_cost(evaluation: CandidateEvaluation) -> ComputeCost:
    """The compute this evaluation must be charged for, or a refusal.

    Fail-closed in the two ways that let evaluation spend vanish: a cost that
    was never reported, and a zero that names no measurement method -- which is
    indistinguishable from "not reported" and would make every free-looking
    evaluation leg disappear from the campaign's accounting.
    """
    cost = evaluation.cost
    if cost is None:
        raise CandidateEvaluationRefusal(
            f"{CANDIDATE_EVALUATION_COST_UNREPORTED}: the evaluation reports no "
            "cost, so its compute cannot be charged; measuring the candidate is "
            "spend like any other leg and an unreported cost would settle as zero"
        )
    if cost.wall_gpu_hours == 0.0 and not cost.measurement_method.strip():
        raise CandidateEvaluationRefusal(
            f"{CANDIDATE_EVALUATION_COST_UNMEASURED}: the evaluation reports zero "
            "wall GPU-hours without naming how that was measured, so the zero is "
            "indistinguishable from an unreported cost; a measured zero must say "
            "what was measured (ComputeCost.zero(measurement_method=...))"
        )
    return cost


def validate_candidate_report(
    evaluation: CandidateEvaluation, request: EvaluationRequest
) -> CandidateEvaluation:
    """Bind a returned report to the artifact the run selected.

    Every failure mode here is named, so a refused run says which rule held
    rather than only that it failed.
    """
    report = evaluation.report
    rows = tuple(report.runs)
    if not rows:
        declared = ", ".join(request.declared_benchmarks)
        raise CandidateEvaluationRefusal(
            f"{CANDIDATE_EVALUATION_EMPTY}: the evaluator measured nothing for "
            f"{declared}"
        )
    # The cost is validated here, at the boundary, so every consumer downstream
    # may treat it as present and charged. Before this rule a report without a
    # cost settled as zero and the campaign could stay inside a budget it had
    # actually spent.
    validate_evaluation_cost(evaluation)
    if report.generation_version != request.candidate_version:
        raise CandidateEvaluationRefusal(
            f"{CANDIDATE_EVALUATION_WRONG_GENERATION}: the evaluation is labelled "
            f"{report.generation_version!r} but this run produces "
            f"{request.candidate_version!r}; evidence for another generation is "
            "not this candidate's evidence"
        )

    identity = {str(key): str(value) for key, value in dict(report.model_identity).items()}
    measured_adapter = identity.get("adapter_digest", "")
    if measured_adapter != request.artifact_sha256:
        raise CandidateEvaluationRefusal(
            f"{CANDIDATE_EVALUATION_IDENTITY_UNBOUND}: the evaluation names "
            f"adapter_digest {measured_adapter or '(none)'} but the run selected "
            f"{request.artifact_sha256}; a report that cannot name the bytes it "
            "measured cannot be the candidate arm"
        )
    measured_base = identity.get("base_model_digest", "")
    if measured_base != request.base_model_digest:
        raise CandidateEvaluationRefusal(
            f"{CANDIDATE_EVALUATION_IDENTITY_UNBOUND}: the evaluation names "
            f"base_model_digest {measured_base or '(none)'} but the campaign "
            f"declares {request.base_model_digest}; the candidate is measured "
            "over the declared dense base or not at all"
        )

    seen: dict[str, BenchmarkRun] = {}
    for run in rows:
        qualified_id = run.benchmark_qualified_id
        # A row outside the declared sets is *reported*, not dropped and not
        # refused here: the frozen judge is the owner of "no undeclared row
        # substitutes for a required slice" (its T11), and duplicating that policy
        # in the runner would be a second rule that can drift from it. The run
        # record names any such row, so the divergence is visible before it is
        # adjudicated rather than silent.
        if run.measurement_origin in (MEASURED_PARENT, CARRIED_REFERENCE):
            # Substitution, not absence: a parent row or a carried reference is
            # somebody else's measurement wearing this generation's label.
            raise CandidateEvaluationRefusal(
                f"{CANDIDATE_EVALUATION_NOT_CANDIDATE_MEASURED}: {qualified_id} "
                f"carries measurement_origin {run.measurement_origin!r}; the "
                "candidate arm is measured by the candidate or it is not "
                "candidate evidence"
            )
        if run.measurement_origin == UNMEASURED:
            # An honest non-measurement is not a provenance violation: the
            # evaluator was asked and could not measure. It covers the declared
            # benchmark (the question was put to it) and stays unmeasured all the
            # way through, which is how a gate refuses to read it as a pass.
            # It is still one row per benchmark: two unmeasured rows under one
            # id are ambiguous evidence, exactly like two measured ones.
            if qualified_id in seen:
                raise CandidateEvaluationRefusal(
                    f"{CANDIDATE_EVALUATION_DUPLICATE_BENCHMARK}: {qualified_id} has "
                    "more than one row (at least one of them unmeasured); one "
                    "benchmark carries one row, measured or not"
                )
            seen[qualified_id] = run
            continue
        if run.generation_version != request.candidate_version:
            raise CandidateEvaluationRefusal(
                f"{CANDIDATE_EVALUATION_WRONG_GENERATION}: the {qualified_id} row "
                f"is labelled {run.generation_version!r} but this run produces "
                f"{request.candidate_version!r}"
            )
        if qualified_id in seen:
            raise CandidateEvaluationRefusal(
                f"{CANDIDATE_EVALUATION_DUPLICATE_BENCHMARK}: {qualified_id} is "
                "measured twice in one evaluation; two measurements under one "
                "benchmark id cannot both be the candidate arm"
            )
        seen[qualified_id] = run

    missing = [
        qualified_id
        for qualified_id in request.declared_benchmarks
        if qualified_id not in seen
    ]
    if missing:
        raise CandidateEvaluationRefusal(
            f"{CANDIDATE_EVALUATION_INCOMPLETE}: the campaign declared "
            f"{missing} as measured sets but the evaluation has no row for them; "
            "an unevaluated declaration is a missing measurement, not an "
            "omission"
        )

    measured = tuple(
        run
        for run in rows
        if run.measurement_origin == MEASURED_THIS_GENERATION
        and run.support == SUPPORTED
        and run.score is not None
    )
    if not measured:
        raise CandidateEvaluationRefusal(
            f"{CANDIDATE_EVALUATION_EMPTY}: the evaluator returned no "
            "candidate-measured, scored row, so this run has no measurement of "
            "the artifact it selected to adjudicate on"
        )
    return evaluation


def write_candidate_evaluation(
    evaluation: CandidateEvaluation, *, root: Path | str, name: str
) -> Path:
    """Write the validated report where the frozen judge reads it.

    The file carries the report's own ``model_identity`` and each row's own
    ``measurement_origin``; the writer adds nothing and reconciles nothing, so
    what the judge reads is what the evaluator measured.
    """
    destination = Path(root) / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(evaluation.report.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return destination


def undeclared_rows(
    evaluation: CandidateEvaluation, request: EvaluationRequest
) -> tuple[str, ...]:
    """Rows for benchmarks no declared set names, in evaluation order."""
    return tuple(
        run.benchmark_qualified_id
        for run in evaluation.report.runs
        if not request.roles_of(run.benchmark_qualified_id)
    )


def evaluation_detail(evaluation: CandidateEvaluation, request: EvaluationRequest) -> str:
    """A one-line record of what the candidate arm attests.

    Undeclared rows are named here rather than judged here: the frozen judge
    decides whether one substitutes for a required slice, and the run's own
    record should show the evidence it will be asked about.
    """
    measured = ", ".join(
        run.benchmark_qualified_id for run in evaluation.report.runs
    )
    undeclared = undeclared_rows(evaluation, request)
    return (
        f"{len(evaluation.report.runs)} candidate-measured row(s) for "
        f"{request.candidate_version} over adapter "
        f"{request.artifact_sha256[:12]} ({measured})"
        + (f"; outside the declared sets: {', '.join(undeclared)}" if undeclared else "")
    )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "CANDIDATE_EVALUATION_COST_UNMEASURED",
    "CANDIDATE_EVALUATION_COST_UNREPORTED",
    "CANDIDATE_EVALUATION_DUPLICATE_BENCHMARK",
    "CANDIDATE_EVALUATION_EMPTY",
    "CANDIDATE_EVALUATION_IDENTITY_UNBOUND",
    "CANDIDATE_EVALUATION_INCOMPLETE",
    "CANDIDATE_EVALUATION_NOT_CANDIDATE_MEASURED",
    "CANDIDATE_EVALUATION_NOT_PRODUCED",
    "CANDIDATE_EVALUATION_WRONG_GENERATION",
    "CandidateEvaluation",
    "CandidateEvaluationRefusal",
    "CandidateEvaluator",
    "EvaluationRequest",
    "coerce_evaluation",
    "evaluation_detail",
    "undeclared_rows",
    "validate_candidate_report",
    "validate_evaluation_cost",
    "write_candidate_evaluation",
]
