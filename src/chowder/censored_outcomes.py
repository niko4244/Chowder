"""A normalized censored-outcome view over the run registry.

Roadmap Priority 6 requires that a policy trained over the intervention/outcome
view either accounts for censored outcomes or explicitly constrains its claim:
"define how censored failures/cancellations enter the dataset (or constrain
the claim)". This module is the censored half of that dataset. It is the
deliberate parallel of `intervention_outcomes.InterventionOutcome`:

  `InterventionOutcome`   -- experiments that ran to a scored
                             `ExperimentResult`. The reward signal.
  `CensoredOutcome`       -- experiments that never produced one. The
                             survival signal.

An experiment can end without a scored result in exactly two stored ways:

  REJECTED (no result row)
      The candidate was refused before any real GPU work -- either by
      preflight (config invalid, dependency missing, disk/VRAM/dataset
      preflight) or cancelled before start. The registry records the
      experiment row with `status='rejected'` and typically zero real
      GPU-hours; `results` has no row for it. The specific sub-cause is
      NOT stored: the `experiments` table has no error/message column,
      so none exists to read and none is invented here.

  FAILED
      The candidate started real work and crashed -- the training or
      evaluation executor raised. The registry records
      `status='failed'` and, when the Executor Investigator path ran and
      its analysis was persisted, a real `execution_incidents` row with
      a structured `signature_kind` (CUDA OOM, dependency incompatibility,
      ...), a stable `fingerprint_sha256`, and the real measured
      `gpu_hours_spent` up to the crash.

Why this matters for any per-arm reward model: `InterventionOutcome` rows
are, by construction, experiments that survived long enough to be scored.
A per-arm failure rate computed without this view is a survivor-biased
estimate -- "the best arm" computed over scored rows alone can simply be
"the arm whose failures nobody recorded". This module makes the censored
half of that dataset first-class instead of invisible.

No score is invented here -- the whole point of "censored" is that the
outcome the reward model needs was never observed. A censored row has
exactly the evidence that exists: what the intervention was, what it
cost, how it ended, and (for failures with a persisted incident) the
structured failure classification. What a policy does with censoring is
a modelling decision belonging to the policy layer; this module's own
documented, minimal position is only:

  1. The per-arm *censoring rate* itself is a first-class signal. An arm
     with high preflight rejection or crash rates is telling you
     something real about the arm, before any reward is considered.
  2. Crash-cost evidence is real where it was measured: a censored row
     with a recorded incident carries the capture-time
     `gpu_hours_spent`, and any budget-aware policy must account for
     compute spent on experiments that produced no score.
  3. Nothing here imputes a counterfactual score, estimates what the
     experiment "would have" scored, or backfills missing sub-cause
     detail the registry never stored.

Honesty rule (the same one `intervention_outcomes.py` states)
-------------------------------------------------------------
Every field is read from real stored evidence, or is `None`. The fields
that are frequently `None`, and exactly why:

  `signature_kind`, `fingerprint_sha256`, `incident_id`
      Read from the joined `execution_incidents` row, present only when
      an incident was actually recorded for this experiment. The cycle
      runner now persists every non-cancelled crash's analysis (the
      production caller), so FAILED rows produced by current runs carry a
      real classification; absence still happens -- runs before this
      caller existed, runs with no registry attached, deliberate
      cancellations (which construct no analysis), and crashes whose
      persistence itself failed -- and that absence is reported as
      `None`, never approximated from the status alone.

  `executor_name`
      The executor that crashed, from the incident row when joined;
      `None` when no incident was recorded. The experiments table does
      not store which executor an experiment was bound to.

  `gpu_hours`
      The real measured `gpu_hours_spent` from the joined incident row --
      a genuine capture-time measurement when a crash was recorded.
      `None` when no incident was recorded, which is the honest state:
      the experiments table stores the *estimated* reservation (exposed
      here as `estimated_gpu_hours`) but not the settled actual charge.
      A REJECTED-before-
      start experiment spent no real compute by construction, but that
      is a semantic inference from the status, not a stored measurement,
      so it is still `None` here -- a policy may treat it as zero-cost
      by its own documented rule, not by this view guessing.

  `error_message`
      Not stored for censored experiments at all. The `experiments`
      table has no error column; incident rows store the structured
      classification, not the raw message, and transcribing traceback
      text into this view is deliberately out of scope. This field is
      always `None` and exists only to make the absence explicit in the
      row shape. (If a future registry schema records failure text,
      this field is where it belongs -- that change must come with the
      schema, not with a guess here.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .candidate_selection import dotted_paths
from .models import ExperimentStatus
from .registry import RunRegistry


@dataclass(frozen=True)
class CensoredOutcome:
    """One historical experiment that ended without a scored result.

    See this module's docstring for the honesty rule and for which fields
    are frequently `None` and why. Every `| None` below means "this
    evidence is genuinely not present for this run", never "unknown, so
    we picked a default". `gpu_hours` is the real capture-time measurement
    from the joined incident when one exists, and `None` when it does not
    (see the module docstring for why nothing else honest is available).
    """

    # Identity / lineage -- always present (schema columns).
    experiment_id: str
    parent_id: str | None

    # How it ended. `REJECTED` without a result row = refused before any
    # real work (preflight rejection or cancellation before start);
    # `FAILED` = started real work and crashed. Both are stored statuses;
    # this view never emits a row for any other status.
    status: ExperimentStatus

    # The intervention.
    config_patch: Mapping[str, Any]
    arm: frozenset[str]
    intervention: str

    # What it cost (see the `gpu_hours` field comment in the module
    # docstring for why this is not the settled ledger charge and is
    # `None` without a recorded incident).
    estimated_gpu_hours: float
    gpu_hours: float | None

    # Structured crash classification -- joined from
    # `execution_incidents` when an incident was recorded, else `None`
    # (see module docstring: absence is real, not imputed).
    incident_id: str | None
    signature_kind: str | None
    fingerprint_sha256: str | None
    executor_name: str | None

    # Always None today: the registry stores no error text for censored
    # experiments (see module docstring). Present in the row shape so the
    # absence is visible rather than silent.
    error_message: None = None


def build_censored_outcomes(
    registry: RunRegistry,
) -> tuple[CensoredOutcome, ...]:
    """Assemble the censored-outcome view from *registry*.

    Rows come back in the order the experiments were recorded
    (`RunRegistry.list_experiments()` is `ORDER BY rowid`), so this is
    deterministic for a given database. An experiment with a persisted
    `ExperimentResult` is never a censored row -- it belongs in
    `intervention_outcomes.build_intervention_outcomes` even if its gate
    verdict was rejection, because there the *outcome* was observed.
    "Censored" means the outcome itself was never measured, not that it
    was bad.
    """
    results = {result.experiment_id for result in registry.list_results()}
    incidents: dict[str, Mapping[str, object]] = {}
    for row in registry.list_execution_incidents():
        experiment_id = row.get("experiment_id")
        if isinstance(experiment_id, str) and experiment_id not in incidents:
            # Incidents are append-only (`_insert_immutable`) and keyed by
            # incident_id, but an experiment could in principle have more
            # than one recorded incident (e.g. retried investigations).
            # list_execution_incidents returns them in rowid order; the
            # first recorded is kept -- "earliest observed classification"
            # is the deterministic, stored-evidence-only choice, and no
            # merge/override is invented.
            incidents[experiment_id] = row

    rows: list[CensoredOutcome] = []
    for experiment in registry.list_experiments():
        if experiment.experiment_id in results:
            # A scored outcome exists -- not censored, whichever gate
            # verdict it later received.
            continue
        if experiment.status not in (ExperimentStatus.REJECTED, ExperimentStatus.FAILED):
            # PLANNED/RUNNING experiments are not outcomes yet, censored
            # or otherwise; representing them would fabricate an ending.
            continue

        incident = incidents.get(experiment.experiment_id)

        rows.append(
            CensoredOutcome(
                experiment_id=experiment.experiment_id,
                parent_id=experiment.parent_id,
                status=experiment.status,
                config_patch=dict(experiment.config_patch),
                arm=dotted_paths(experiment.config_patch),
                intervention=experiment.hypothesis.intervention,
                estimated_gpu_hours=experiment.estimated_gpu_hours,
                gpu_hours=_incident_gpu_hours(incident),
                incident_id=(
                    incident.get("incident_id")
                    if incident is not None and isinstance(incident.get("incident_id"), str)
                    else None
                ),
                signature_kind=(
                    incident.get("signature_kind")
                    if incident is not None and isinstance(incident.get("signature_kind"), str)
                    else None
                ),
                fingerprint_sha256=(
                    incident.get("fingerprint_sha256")
                    if incident is not None and isinstance(incident.get("fingerprint_sha256"), str)
                    else None
                ),
                executor_name=(
                    incident.get("executor_name")
                    if incident is not None and isinstance(incident.get("executor_name"), str)
                    else None
                ),
            )
        )
    return tuple(rows)


def _incident_gpu_hours(incident: Mapping[str, object] | None) -> float | None:
    """The incident's measured `gpu_hours_spent`, or None without one.

    See the `gpu_hours` field comment: this reads the capture-time
    measurement when an incident row exists and is `None` when it does
    not -- the experiments table stores no settled actual charge, so
    there is nothing else honest to return.
    """
    if incident is None:
        return None
    value = incident.get("gpu_hours_spent")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def censoring_rate_by_arm(
    censored: Sequence[CensoredOutcome],
) -> dict[frozenset[str], float]:
    """Per-arm fraction of censored rows that are REJECTED-no-work.

    The denominator is the number of censored rows in the arm, so this
    is a shape-of-censoring summary, not a failure rate over all
    attempts: arms whose experiments all ran and passed contribute no
    censored rows here and are absent from the result. Joining this with
    `intervention_outcomes.group_by_arm`'s scored-row counts is the
    policy layer's job -- deliberately not done here, because the right
    combined statistic is exactly the modelling decision Priority 6
    gates on.

    A REJECTED censored row means the experiment was refused before real
    work (preflight or cancel-before-start); a FAILED row means real
    work happened and crashed. The split is the honest per-arm signal
    the registry actually stores: "how often does attempting this arm
    not even start" versus "how often does it start and die".
    """
    counts: dict[frozenset[str], tuple[int, int]] = {}
    for row in censored:
        rejected, total = counts.get(row.arm, (0, 0))
        counts[row.arm] = (
            rejected + (1 if row.status is ExperimentStatus.REJECTED else 0),
            total + 1,
        )
    return {arm: rejected / total for arm, (rejected, total) in counts.items()}


def filter_censored(
    censored: Sequence[CensoredOutcome],
    *,
    status: ExperimentStatus | None = None,
    signature_kind: str | None = None,
    touches_key_path: str | None = None,
) -> tuple[CensoredOutcome, ...]:
    """Filter *censored* rows; every criterion given is ANDed, order preserved.

    `signature_kind` matches only rows with a recorded incident of that
    exact kind -- a row with no incident (`signature_kind is None`) is
    excluded by either value. That is the same "not on record is not
    evidence" rule `filter_outcomes` applies to `gate_accepted`, and it
    matters here for the same reason: absence of a persisted incident
    says nothing about what kind of failure it was.
    """
    selected = tuple(censored)
    if status is not None:
        selected = tuple(row for row in selected if row.status is status)
    if signature_kind is not None:
        selected = tuple(row for row in selected if row.signature_kind == signature_kind)
    if touches_key_path is not None:
        selected = tuple(row for row in selected if touches_key_path in row.arm)
    return selected


def group_censored_by_arm(
    censored: Sequence[CensoredOutcome],
) -> dict[frozenset[str], tuple[CensoredOutcome, ...]]:
    """Group *censored* rows by intervention arm, preserving input order.

    The arm is the same frozenset of dotted `config_patch` key-paths
    `candidate_selection.dotted_paths` produces everywhere else, so an
    arm here is the same arm `intervention_outcomes.group_by_arm` groups
    and the UCB1 selector bandits over.
    """
    grouped: dict[frozenset[str], list[CensoredOutcome]] = {}
    for row in censored:
        grouped.setdefault(row.arm, []).append(row)
    return {arm: tuple(rows) for arm, rows in grouped.items()}
