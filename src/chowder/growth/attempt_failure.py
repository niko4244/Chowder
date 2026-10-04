"""Attempt-failure classification: every failed attempt teaches the next
generation.

This module classifies why a *candidate attempt* failed (infrastructure,
budget, underpowered measurement, clean falsification, ...) -- the
model-failure text taxonomy lives in ``failure_taxonomy.py``, which the
failure bank and the curriculum engine read.

A failure is not one thing. An infrastructure crash teaches nothing about the
mechanism; an underpowered measurement teaches nothing at all; a clean
measured negative falsifies. Treating these alike -- one bucket of "failed",
one prior penalty, one retry policy -- is how a search budget gets spent
retrying untested hypotheses or re-testing falsified ones.

This module classifies a failed attempt into the taxonomy, maps each class
to the evidence state that *should* be recorded (or to no record at all,
when nothing was tested), and names the next action the generation loop
should take. It also enforces the anti-repeat rule: a candidate whose exact
intervention was already measured to failure in scope cannot be generated
again.

The precedence is explicit because classification order *is* the policy:

1. settlement/budget refusals and a reported non-resume are budget or
   infrastructure classes -- the mechanism was never trained, so no evidence
   record is written;
2. a failed training run is infrastructure;
3. incomplete measurements are inconclusive -- re-measure, do not conclude;
4. a paired interval excluding zero is a measured effect: negative means
   falsified (a good, recordable outcome), positive means this was not a
   failure;
5. anything else escalates rather than guessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from .compute_cost import settlement_refusal

if TYPE_CHECKING:  # pragma: no cover - import cycle guard for type checkers only
    from .evidence import EvidenceStore

__all__ = [
    "FailureClass",
    "FailureClassification",
    "NextAction",
    "classify_failure",
    "next_action",
    "record_refuses_disproven_repeat",
]

#: Refusal code for everything this module refuses.
ATTEMPT_FAILURE_SCHEMA = "ATTEMPT_FAILURE"


class FailureClass(str, Enum):
    """Why an attempt did not produce a measured improvement."""

    #: The environment, provider or executor failed; the mechanism is untested.
    INFRASTRUCTURE = "infrastructure"
    #: The measurement could not decide (underpowered, incomplete, noisy).
    EVAL_NOISE = "eval-noise"
    #: The mechanism may be right; the tested parameters were wrong.
    PARAMETER = "parameter"
    #: A clean, measured negative: the mechanism is wrong in this scope.
    MECHANISM_FALSIFIED = "mechanism-falsified"
    #: The intervention cannot apply to this architecture at all.
    ARCHITECTURE_INCOMPATIBLE = "architecture-incompatible"
    #: The data or curriculum itself was the problem.
    DATA_OR_CURRICULUM = "data-or-curriculum"
    #: Compute ran out before the attempt could be tested at all.
    BUDGET_EXHAUSTED = "budget-exhausted"


class NextAction(str, Enum):
    """What the generation loop should do with a classified failure."""

    #: The hypothesis was never tested; run it again once the cause clears.
    RETRY_UNCHANGED = "retry-unchanged"
    #: Re-measure with more evidence; do not update the family prior.
    REPLICATE_WITH_MORE_EVIDENCE = "replicate-with-more-evidence"
    #: Same mechanism, narrowed parameter range; record as neutral.
    NARROW_PARAMETERS = "narrow-parameters"
    #: Record the falsification and stop proposing this exact intervention.
    RECORD_FALSIFIED_AND_SKIP = "record-falsified-and-skip"
    #: Record the architectural exclusion for this scope.
    EXCLUDE_ARCHITECTURE = "exclude-architecture"
    #: A human must look: the data, or the classification itself, is suspect.
    ESCALATE_TO_OPERATOR = "escalate-to-operator"


#: The evidence state each class records. ``None`` means *no record*: writing
#: a FAILED row for an attempt that never trained would be fabricated
#: evidence, and fabricating evidence is the one unforgivable failure mode.
_EVIDENCE_STATE_BY_CLASS: dict[FailureClass, str | None] = {
    FailureClass.INFRASTRUCTURE: None,
    FailureClass.BUDGET_EXHAUSTED: None,
    FailureClass.EVAL_NOISE: "inconclusive",
    FailureClass.PARAMETER: "neutral",
    FailureClass.MECHANISM_FALSIFIED: "failed",
    FailureClass.ARCHITECTURE_INCOMPATIBLE: "architecture-incompatible",
    FailureClass.DATA_OR_CURRICULUM: None,
}

_ACTION_BY_CLASS: dict[FailureClass, NextAction] = {
    FailureClass.INFRASTRUCTURE: NextAction.RETRY_UNCHANGED,
    FailureClass.BUDGET_EXHAUSTED: NextAction.RETRY_UNCHANGED,
    FailureClass.EVAL_NOISE: NextAction.REPLICATE_WITH_MORE_EVIDENCE,
    FailureClass.PARAMETER: NextAction.NARROW_PARAMETERS,
    FailureClass.MECHANISM_FALSIFIED: NextAction.RECORD_FALSIFIED_AND_SKIP,
    FailureClass.ARCHITECTURE_INCOMPATIBLE: NextAction.EXCLUDE_ARCHITECTURE,
    FailureClass.DATA_OR_CURRICULUM: NextAction.ESCALATE_TO_OPERATOR,
}


@dataclass(frozen=True)
class FailureClassification:
    """The classified failure, the record it earns, and the reason why."""

    failure_class: FailureClass
    #: The EvidenceState *value* to record, or None for no record at all.
    evidence_state: str | None
    action: NextAction
    reason: str


def classify_failure(attempt: Mapping[str, Any]) -> FailureClassification:
    """Classify one failed attempt from the fields its own record carries.

    Reads only training-side and settlement facts -- status, refusal and
    settlement codes, resume state, the paired interval and its sample size,
    and whether every declared retention constraint was measured. No
    benchmark score is needed to classify a failure, so classification never
    becomes a way to read tier-3 evidence.
    """
    resume_state = attempt.get("resume_state")
    if resume_state == "not-a-resume":
        return FailureClassification(
            failure_class=FailureClass.INFRASTRUCTURE,
            evidence_state=None,
            action=NextAction.RETRY_UNCHANGED,
            reason=(
                "the executor reported a restart instead of the declared "
                "resume: the continuation was never actually trained"
            ),
        )

    # One predicate owns the refusal vocabulary, so classification and the
    # runner's advance rule can never disagree about what was refused: the
    # same settlement fact the gate reads is the one classified here.
    settlement_reason = settlement_refusal(attempt)
    if settlement_reason:
        return FailureClassification(
            failure_class=FailureClass.BUDGET_EXHAUSTED,
            evidence_state=None,
            action=NextAction.RETRY_UNCHANGED,
            reason=(
                f"settlement refused ({settlement_reason}): the attempt cost more "
                "than its declared projection, so the mechanism's evidence is "
                "unpriced and unrecorded"
            ),
        )

    if attempt.get("candidate_succeeded") is not True:
        status = attempt.get("status", "<missing>")
        return FailureClassification(
            failure_class=FailureClass.INFRASTRUCTURE,
            evidence_state=None,
            action=NextAction.RETRY_UNCHANGED,
            reason=(
                f"training did not produce a candidate (status {status!r}): "
                "no mechanism claim can be made from a run that never ran"
            ),
        )

    if attempt.get("measurements_complete") is False:
        return FailureClassification(
            failure_class=FailureClass.EVAL_NOISE,
            evidence_state="inconclusive",
            action=NextAction.REPLICATE_WITH_MORE_EVIDENCE,
            reason=(
                "not every declared measurement was taken; an incomplete "
                "evaluation is not a verdict"
            ),
        )

    ci = attempt.get("paired_delta_ci")
    if ci is not None:
        low, high = float(ci[0]), float(ci[1])
        if low > 0.0:
            return FailureClassification(
                failure_class=FailureClass.INFRASTRUCTURE,
                evidence_state=None,
                action=NextAction.ESCALATE_TO_OPERATOR,
                reason=(
                    "the paired interval excludes zero on the positive side: "
                    "this attempt did not fail, and classifying it as a "
                    "failure is an operator-level inconsistency"
                ),
            )
        if high < 0.0:
            sample_size = int(attempt.get("sample_size") or 0)
            return FailureClassification(
                failure_class=FailureClass.MECHANISM_FALSIFIED,
                evidence_state="failed",
                action=NextAction.RECORD_FALSIFIED_AND_SKIP,
                reason=(
                    f"the paired interval [{low:g}, {high:g}] excludes zero on "
                    f"the negative side over {sample_size} paired task(s): a "
                    "clean measured negative, which is a recordable result"
                ),
            )
        return FailureClassification(
            failure_class=FailureClass.EVAL_NOISE,
            evidence_state="inconclusive",
            action=NextAction.REPLICATE_WITH_MORE_EVIDENCE,
            reason=(
                f"the paired interval [{low:g}, {high:g}] includes zero: the "
                "measurement cannot distinguish effect from noise"
            ),
        )

    # No paired interval at all: with an adequate sample a negative mean is a
    # measured negative; without one it is only a hint.
    delta = attempt.get("mean_delta")
    sample_size = int(attempt.get("sample_size") or 0)
    if delta is not None:
        delta = float(delta)
        if delta < 0.0 and sample_size >= 3:
            return FailureClassification(
                failure_class=FailureClass.MECHANISM_FALSIFIED,
                evidence_state="failed",
                action=NextAction.RECORD_FALSIFIED_AND_SKIP,
                reason=(
                    f"negative mean delta {delta:g} over {sample_size} sample(s) "
                    "without a paired interval: measured negative, unpaired"
                ),
            )
        if delta < 0.0:
            return FailureClassification(
                failure_class=FailureClass.EVAL_NOISE,
                evidence_state="inconclusive",
                action=NextAction.REPLICATE_WITH_MORE_EVIDENCE,
                reason=(
                    f"negative mean delta {delta:g} over only {sample_size} "
                    "sample(s): too thin to conclude, paired evidence required"
                ),
            )
        if delta > 0.0:
            return FailureClassification(
                failure_class=FailureClass.INFRASTRUCTURE,
                evidence_state=None,
                action=NextAction.ESCALATE_TO_OPERATOR,
                reason=(
                    "a positive delta was submitted to the failure classifier: "
                    "the campaign wiring, not the mechanism, is suspect"
                ),
            )

    # The architecture check is an explicit attempt field because the family
    # registry, not the attempt, owns valid_architectures.
    if attempt.get("architecture_compatible") is False:
        return FailureClassification(
            failure_class=FailureClass.ARCHITECTURE_INCOMPATIBLE,
            evidence_state="architecture-incompatible",
            action=NextAction.EXCLUDE_ARCHITECTURE,
            reason=(
                "the family's valid_architectures excludes this model's "
                "architecture; the intervention cannot apply here"
            ),
        )

    return FailureClassification(
        failure_class=FailureClass.DATA_OR_CURRICULUM,
        evidence_state=None,
        action=NextAction.ESCALATE_TO_OPERATOR,
        reason=(
            "the attempt failed with no signal the taxonomy can read "
            "(no settle refusal, no resume state, no paired interval, no "
            "delta); refusing to guess is the classification"
        ),
    )


def next_action(classification: FailureClassification) -> NextAction:
    """The action the generation loop takes; explicit for testability."""
    return classification.action


def record_refuses_disproven_repeat(
    store: "EvidenceStore",
    *,
    family_id: str,
    model_family: str,
    architecture: str,
    intervention_parameters: Mapping[str, Any],
    note: str = "",
) -> None:
    """Refuse a candidate whose exact intervention already failed in scope.

    "Repeated exploration of already-disproven interventions" is the waste
    the architecture exists to prevent. A repeat here means: same family,
    same model family, same architecture, and the *same* intervention
    parameters, where the recorded outcome was a measured failure. A new
    record is not demanded when the prior failure was inconclusive (that is
    the replicate action's job, with more evidence, not a refusal).

    Raises ``ValueError`` naming the disproving record when the candidate is
    a repeat; returns silently when the record may be written.
    """
    from .evidence import EvidenceState

    for record in store.records_for_family(family_id):
        if record.model_family != model_family:
            continue
        if record.architecture and record.architecture != architecture:
            continue
        if record.state is not EvidenceState.FAILED:
            continue
        if dict(record.intervention_parameters) != dict(intervention_parameters):
            continue
        raise ValueError(
            f"{ATTEMPT_FAILURE_SCHEMA}: {record.record_id} already measured this "
            f"exact intervention failed for {family_id!r} on "
            f"{model_family!r}/{architecture!r}{' -- ' + note if note else ''}; "
            "the next generation must differ where the evidence says so "
            "(narrow the parameters, change the mechanism, or record new "
            "inconclusive evidence -- not repeat a measured negative)"
        )
    return None
