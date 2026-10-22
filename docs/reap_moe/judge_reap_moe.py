#!/usr/bin/env python3
"""Frozen promotion judge for the REAP-MoE lineage — ``reap-moe-judge-v1``.

Frozen with ``docs/quals/REAP_MOE_JUDGE_V1_PREREG_2026-10-19.md``, which is the
contract this file implements: every gating row below has an id, a rule and an
evidence source there, and none of them may move after a candidate's results
are visible. The lineage charter is ``docs/REAP_MOE_LINEAGE.md`` (owner decision
N3): the promotion judge is a frozen local code judge, and quality evidence is
execution-verified only.

What a verdict means
--------------------
``PROMOTED``   every gating row PASSES on the run's own bytes.
``REJECTED``   at least one gating row FAILS — the candidate is refused.
``INCONCLUSIVE`` no row fails, but at least one cannot be decided (missing or
               unreadable evidence, an undeclared threshold). Strict epistemics:
               an artifact that is not there is UNKNOWN, never an assumed pass.

Exit codes: ``0`` PROMOTED, ``1`` REJECTED, ``2`` INCONCLUSIVE (and ``2`` for a
usage error). That mapping is the one difference from the gen2 judge, whose
"not certified" and "refused" share an exit code; a promotion panel needs to
tell "this candidate lost" from "this panel cannot answer".

Evidence (read-only; the run root is never written)
---------------------------------------------------
  candidate_evaluation.json    the measured candidate  (MEASURED_THIS_GENERATION)
  parent_evaluation.json       the measured parent     (MEASURED_PARENT)
  chosen_candidate.json        the selected artifact: ref + sha256
  cycle_compute_accounting.json  the campaign's cost ledger
  campaign-run.json            what the run itself decided (CampaignRun.to_dict)
  <raw>.jsonl                  the per-item predictions each arm names, pinned
                               by ``metadata.artifact_sha256``
  reports/*.json               descriptive, **non-gating** (never read by a row)

Each arm's target and protected rows must name their raw predictions file and
pin its sha256; the judge recomputes that digest, re-reads the items with
production's own parser (``evaluation_binding._read_items``) and re-scores them
with production's declared readout (``evaluators.scoring.score``). A score the
run reports and the judge cannot reproduce is a FAIL, not an argument.

The campaign declaration
------------------------
``reap_moe_campaign.json`` beside this file. It is the campaign's frozen
pre-registration (sets, ceilings, retention profile, and the target-side
``target_threshold`` that says what "improved" means); the judge must not carry
a threshold of its own. The file is placed here when a campaign names this
judge version — it is not shipped with placeholder values, because a TODO is
not a threshold. Absent or unparseable, every row that needs it is UNKNOWN and
the verdict is INCONCLUSIVE.

Usage:
    python docs/reap_moe/judge_reap_moe.py <run_root>
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "docs" / "quals"))
sys.path.insert(0, str(REPO / "src"))

from quals_harness import (  # noqa: E402  (paths inserted above, by design)
    FAIL,
    PASS,
    UNKNOWN,
    Verdict,
)
from chowder.evals.result import (  # noqa: E402
    MEASURED_PARENT,
    MEASURED_THIS_GENERATION,
    BenchmarkRun,
)
from chowder.evaluators.scoring import score, tool_schema_for  # noqa: E402
from chowder.growth import certification as _certification  # noqa: E402
from chowder.growth.campaign import CampaignManifest, settle_campaign  # noqa: E402
from chowder.growth.candidate_evaluation import (  # noqa: E402
    CandidateEvaluationRefusal,
)
from chowder.growth.certification import ArmError, MeasuredArm  # noqa: E402
from chowder.growth.compute_cost import checked_ledger_total, ledger_digest  # noqa: E402
from chowder.growth.cycle import retention_values  # noqa: E402
from chowder.growth.evaluation_binding import (  # noqa: E402
    EvaluationMaterial,
    _read_items,
)
from chowder.growth.generation_diagnostics import (  # noqa: E402
    GenerationDiagnostics,
    _has_repeated_lines,
    _trigram_ratio,
)
from chowder.growth.promotion import BenchmarkResult  # noqa: E402
from chowder.growth.retention import evaluate_retention  # noqa: E402
from chowder.growth.statistics import compare  # noqa: E402
from chowder.provenance import sha256_file  # noqa: E402

# ---------------------------------------------------------------------------
# Frozen policy. Do not edit after a candidate's results are visible.
# ---------------------------------------------------------------------------
VERSION = "reap-moe-judge-v1"
PREREG = "docs/quals/REAP_MOE_JUDGE_V1_PREREG_2026-10-19.md"

#: The frozen campaign declaration this judge audits against. Placed beside the
#: script when a campaign names this judge version; never written by the run.
CAMPAIGN_MANIFEST = HERE / "reap_moe_campaign.json"

CANDIDATE_REPORT = "candidate_evaluation.json"
PARENT_REPORT = "parent_evaluation.json"
CHOSEN = "chosen_candidate.json"
ACCOUNTING = "cycle_compute_accounting.json"
RUN_RECORD = "campaign-run.json"
DESCRIPTIVE_DIR = "reports"

#: Gate 0: the target and protected generations must not lose more than two
#: points of termination or gain more than two points of degeneracy against the
#: parent. The bar is a difference, never an absolute rate: a parent that is
#: already weak must not make a weaker candidate look acceptable.
GATE0_TOLERANCE = 0.02
#: One item is degenerate when production's loop detector fires, when its
#: distinct-trigram ratio falls under this floor, or when it ran into the cap
#: (a capped generation did not finish). The first two are production's own
#: predicates; the third is the ``cap_hit`` fact the diagnostics already derive.
DEGENERACY_TRIGRAM_MIN = 0.900

#: The target instrument's declared readout. A campaign that declares any other
#: readout for this benchmark is measuring a different instrument.
TOOL_USE_SCORING = "tool_call_exact"

#: A reported score must be the readout's own answer. One part in a million is
#: float noise; anything larger is a different number.
SCORE_TOLERANCE = 1e-6

ARTIFACT_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

#: Named reasons, so a refusal says which verification failed.
RUN_RECORD_ABSENT = "RUN_RECORD_ABSENT"
RUN_RECORD_WRONG_CYCLE = "RUN_RECORD_WRONG_CYCLE"
RUN_DECISION_ABSENT = "RUN_DECISION_ABSENT"
DECLARED_GATE_REJECTED_RUN = "DECLARED_GATE_REJECTED_RUN"
RETENTION_RECOMPUTATION_DISAGREES = "RETENTION_RECOMPUTATION_DISAGREES"
ACCOUNTING_ARTIFACT_MOVED = "ACCOUNTING_ARTIFACT_MOVED"
ACCOUNTING_UNPINNED = "ACCOUNTING_UNPINNED"
ACCOUNTING_UNSETTLED_BY_RUN = "ACCOUNTING_UNSETTLED_BY_RUN"
SETTLEMENT_DISAGREES_WITH_RECORD = "SETTLEMENT_DISAGREES_WITH_RECORD"

#: Production's measurement-verification vocabulary, re-exported so a refusal
#: here is named identically to the one the runner would raise.
MEASUREMENT_ARTIFACT_MISSING = _certification.MEASUREMENT_ARTIFACT_MISSING
MEASUREMENT_DIGEST_ABSENT = _certification.MEASUREMENT_DIGEST_ABSENT
MEASUREMENT_DIGEST_MISMATCH = _certification.MEASUREMENT_DIGEST_MISMATCH
ARM_ADAPTER_DIGEST_MISSING = _certification.ARM_ADAPTER_DIGEST_MISSING
ARM_ADAPTER_DIGEST_MISMATCH = _certification.ARM_ADAPTER_DIGEST_MISMATCH

_measurement_artifact = _certification.measurement_artifact
_digest_of = _certification.digest_of


# ---------------------------------------------------------------------------
# reading helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _open_arm(
    path: Path,
    *,
    origin: str,
    generation: str,
    label: str,
) -> tuple[MeasuredArm | None, str]:
    """One arm, or the reason it is not usable evidence."""
    try:
        return (
            MeasuredArm.load(
                path,
                expected_origin=origin,
                expected_generation=generation,
                label=label,
            ),
            "",
        )
    except ArmError as error:
        return None, str(error)


def _load_campaign() -> tuple[CampaignManifest | None, str]:
    """The frozen declaration, or the reason there is none.

    Absence is not a pass and not a crash: every row that needs a declared set,
    ceiling or threshold reports UNKNOWN against it.
    """
    if not CAMPAIGN_MANIFEST.is_file():
        return None, f"campaign declaration {CAMPAIGN_MANIFEST} is not present"
    try:
        return CampaignManifest.from_file(CAMPAIGN_MANIFEST), ""
    except Exception as error:  # noqa: BLE001 - unusable declaration is UNKNOWN
        return None, f"campaign declaration {CAMPAIGN_MANIFEST} is not usable: {error}"


def _item_key(item: Mapping[str, Any]) -> str | None:
    for field in ("uuid", "prompt_id", "prompt"):
        value = item.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _expected_text(item: Mapping[str, Any], scoring: str) -> str:
    """The expected answer as the two-argument readout wants it.

    ``tool_call_exact`` reads the whole expected call as JSON and needs the
    tool's ``parameters`` schema beside it; the committed item carries the
    schema in ``tools``, so it is folded in here exactly as the evaluator's own
    binding does. Every other readout compares the expected answer's text.
    """
    expected = item.get("expected")
    if isinstance(expected, Mapping):
        document = dict(expected)
        if scoring == TOOL_USE_SCORING and "parameters" not in document:
            schema = tool_schema_for(item.get("tools"), str(document.get("name", "")))
            document["parameters"] = dict(schema or {})
        return json.dumps(document, sort_keys=True)
    return "" if expected is None else str(expected)


def _read_evidence(
    run_root: Path, run: BenchmarkRun, *, label: str
) -> tuple[list[dict[str, Any]] | None, str, str]:
    """The item rows a measurement names, after verifying its pin.

    Returns ``(items, status, detail)``. A missing file, an absent digest or an
    unparseable row is UNKNOWN — the evidence is not there to decide anything.
    A digest that disagrees with the bytes is FAIL: the row and its file are
    two claims about the same artifact and they contradict each other.
    """
    reference = str(run.raw_artifact_ref or "")
    if not reference:
        return None, UNKNOWN, f"{label}: the row names no raw artifact, so its items cannot be read"
    artifact, reason = _measurement_artifact(run_root, reference)
    if artifact is None:
        return None, UNKNOWN, f"{label}: {reason}"
    declared = (run.metadata or {}).get("artifact_sha256")
    if not isinstance(declared, str) or not ARTIFACT_SHA256_PATTERN.fullmatch(declared):
        return (
            None,
            UNKNOWN,
            f"{label}: {MEASUREMENT_DIGEST_ABSENT} (got {declared!r}), so the row is "
            "not bound to the bytes it names",
        )
    try:
        actual = sha256_file(artifact)
    except OSError as error:
        return None, UNKNOWN, f"{label}: {reference!r} cannot be hashed ({error})"
    if actual != declared:
        return (
            None,
            FAIL,
            f"{label}: {MEASUREMENT_DIGEST_MISMATCH} — {reference!r} hashes to "
            f"{actual[:12]}, the row declares {declared[:12]}",
        )
    try:
        items = _read_items(artifact, suite_name=reference)
    except CandidateEvaluationRefusal as error:
        return None, UNKNOWN, f"{label}: {error}"
    return items, PASS, ""


def _reconcile(
    items: Sequence[Mapping[str, Any]],
    run: BenchmarkRun,
    *,
    scoring: str,
    label: str,
) -> tuple[str, str]:
    """The reported score must be the declared readout's own answer.

    Every item is re-scored from the prediction it recorded, so a run cannot
    report an aggregate its own rows do not produce — the defect a score alone
    can never reveal. Item count and mean are checked against the row too.
    """
    if len(items) != run.n_samples:
        return (
            FAIL,
            f"{label}: {len(items)} item(s) on disk against n_samples={run.n_samples}",
        )
    if not items:
        return UNKNOWN, f"{label}: the evidence holds no scored item"
    recomputed: list[float] = []
    disagreements = 0
    first_disagreement = ""
    for item in items:
        value = score(
            str(item.get("prediction", "")),
            _expected_text(item, scoring),
            scoring,
        )
        reported = item.get("score")
        if not isinstance(reported, (int, float)) or isinstance(reported, bool):
            disagreements += 1
            first_disagreement = first_disagreement or "an item carries no numeric score"
        elif abs(float(reported) - value) > SCORE_TOLERANCE:
            disagreements += 1
            if not first_disagreement:
                first_disagreement = (
                    f"item {_item_key(item)!r} reports {float(reported):.4f}, "
                    f"{scoring} gives {value:.4f}"
                )
        recomputed.append(value)
    mean = sum(recomputed) / len(recomputed)
    if disagreements:
        return (
            FAIL,
            f"{label}: {disagreements}/{len(items)} item(s) report a score the declared "
            f"readout does not produce ({first_disagreement})",
        )
    if isinstance(run.score, bool) or not isinstance(run.score, (int, float)):
        return UNKNOWN, f"{label}: the row records no numeric score"
    if abs(mean - float(run.score)) > SCORE_TOLERANCE:
        return (
            FAIL,
            f"{label}: reported {float(run.score):.6f} != recomputed {mean:.6f} "
            f"over {len(items)} item(s) under {scoring}",
        )
    return PASS, f"{label}: {len(items)} item(s) re-score to {mean:.6f} == reported, under {scoring}"


# ---------------------------------------------------------------------------
# R1 — candidate artifact identity and adapter binding
# ---------------------------------------------------------------------------


def _identity_gate(
    verdict: Verdict,
    run_root: Path,
    candidate: MeasuredArm | None,
    candidate_error: str,
) -> None:
    requirement = "candidate artifact identity and adapter binding"
    chosen = _load_json(run_root / CHOSEN)
    if not isinstance(chosen, Mapping):
        verdict.add("R1", requirement, UNKNOWN, f"{CHOSEN} missing or unreadable")
        return
    reference = chosen.get("artifact_ref")
    if not isinstance(reference, str) or not reference.strip():
        verdict.add("R1", requirement, FAIL, "chosen_candidate.json declares no artifact_ref")
        return
    artifact = Path(reference)
    if not artifact.is_absolute():
        artifact = run_root / artifact
    if not artifact.exists():
        verdict.add("R1", requirement, FAIL, f"artifact_ref {reference!r} does not exist")
        return
    try:
        recomputed = _digest_of(artifact)
    except OSError as error:
        verdict.add("R1", requirement, FAIL, f"hashing {reference!r} failed: {error}")
        return
    recorded = chosen.get("artifact_sha256")
    if not isinstance(recorded, str) or not ARTIFACT_SHA256_PATTERN.fullmatch(recorded):
        verdict.add(
            "R1", requirement, FAIL,
            f"recorded artifact_sha256 is not a sha256 hex string: {recorded!r}",
        )
        return
    if recomputed != recorded:
        verdict.add(
            "R1", requirement, FAIL,
            f"the selected artifact hashes to {recomputed[:12]}, the run recorded "
            f"{recorded[:12]}; the bytes named are not the bytes present",
        )
        return
    if candidate is None:
        verdict.add(
            "R1", requirement, UNKNOWN,
            f"the artifact is the recorded one ({recorded[:12]}), but the candidate "
            f"arm is unusable: {candidate_error}",
        )
        return
    problems = candidate.identity_problems(adapter_digest=recorded)
    verdict.add(
        "R1",
        requirement,
        PASS if not problems else FAIL,
        "; ".join(problems)
        or f"the candidate report names the selected artifact ({recorded[:12]}) and its "
        f"generation {candidate.report.generation_version!r}",
    )


# ---------------------------------------------------------------------------
# R2, R3 — settlement, and the artifact the run settled
# ---------------------------------------------------------------------------


def _ledger_total(document: Mapping[str, Any], cycle_id: str):
    """Reconcile a complete accounting artifact against its paid entries.

    Old totals-only artifacts contain too little evidence for certification.
    Distinguish that absence (UNKNOWN) from a contradictory modern ledger
    (FAIL) without changing the frozen settlement rule.
    """
    totals = document.get("totals")
    if (
        not isinstance(totals, Mapping)
        or not isinstance(totals.get("incremental"), Mapping)
        or not isinstance(document.get("entries"), list)
        or not isinstance(document.get("digest_sha256"), str)
        or not document.get("cycle_id")
    ):
        raise LookupError("incremental totals, entry ledger, cycle identity or embedded digest missing")
    return checked_ledger_total(document, cycle_id=cycle_id)


def _settlement_gate(
    verdict: Verdict, run_root: Path, campaign: CampaignManifest | None
) -> None:
    requirement = "campaign cost settles within the declared ceilings"
    document = _load_json(run_root / ACCOUNTING)
    if not isinstance(document, Mapping):
        verdict.add("R2", requirement, UNKNOWN, f"{ACCOUNTING} missing or unreadable")
        return
    declared_totals = document.get("totals")
    totals = declared_totals.get("incremental") if isinstance(declared_totals, Mapping) else None
    if not isinstance(totals, Mapping):
        verdict.add(
            "R2", requirement, UNKNOWN,
            "the accounting artifact declares no incremental totals",
        )
        return
    if campaign is None:
        verdict.add(
            "R2", requirement, UNKNOWN,
            "the campaign declaration is unavailable, so no ceiling can be settled against",
        )
        return
    try:
        total = _ledger_total(document, campaign.cycle_id)
    except LookupError as error:
        verdict.add(
            "R2", requirement, UNKNOWN,
            f"accounting evidence is incomplete: {error}",
        )
        return
    except (KeyError, TypeError, ValueError) as error:
        verdict.add(
            "R2", requirement, FAIL,
            f"accounting entries, flags or totals are inconsistent: {error}",
        )
        return
    settlement = settle_campaign(campaign, total=total)
    verdict.add(
        "R2",
        requirement,
        PASS if settlement.compliant else FAIL,
        (
            f"device {total.device_gpu_hours:.4f} "
            f"({'measured' if total.device_measured else 'unmeasured'}) / wall "
            f"{total.wall_gpu_hours:.4f} against the declared envelope"
            + ("" if settlement.compliant else "; " + "; ".join(settlement.failure_reasons))
        ),
    )


def _accounting_digest_gate(
    verdict: Verdict, run_root: Path, campaign: CampaignManifest | None
) -> None:
    """The artifact must be the one the run settled, and settle the same way.

    Same shape as the gen2 judge's T23: the record pins ``cost.accounting_digest``
    when it writes the ledger, so an edit to paid entries or totals moves a
    number the record already carries. R2 independently checks internal ledger
    consistency; R3 also detects a replaced, internally consistent artifact.
    """
    requirement = "the run's recorded settlement is the settlement of the artifact it pinned"
    record = _load_json(run_root / RUN_RECORD)
    if not isinstance(record, Mapping):
        verdict.add(
            "R3", requirement, UNKNOWN,
            f"{RUN_RECORD_ABSENT}: {RUN_RECORD} is absent or unreadable, so the "
            "artifact it settled cannot be identified",
        )
        return
    document = _load_json(run_root / ACCOUNTING)
    if not isinstance(document, Mapping):
        verdict.add(
            "R3", requirement, UNKNOWN, f"{ACCOUNTING} missing or unreadable"
        )
        return
    record_cost = record.get("cost")
    pinned = str(record_cost.get("accounting_digest") or "") if isinstance(record_cost, Mapping) else ""
    if not pinned:
        verdict.add(
            "R3", requirement, UNKNOWN,
            f"{ACCOUNTING_UNPINNED}: the run recorded no accounting digest, so no "
            "artifact can be settled here",
        )
        return
    recomputed = ledger_digest(document)
    if recomputed != pinned:
        verdict.add(
            "R3", requirement, FAIL,
            f"{ACCOUNTING_ARTIFACT_MOVED}: the accounting artifact hashes to "
            f"{recomputed[:12]}, the run pinned {pinned[:12]}; these are not the "
            "bytes the run settled",
        )
        return
    recorded = record.get("settlement")
    if not isinstance(recorded, Mapping) or "budget_compliant" not in recorded:
        verdict.add(
            "R3", requirement, UNKNOWN,
            f"{ACCOUNTING_UNSETTLED_BY_RUN}: the run pinned this artifact but "
            "recorded no settlement verdict for it",
        )
        return
    if campaign is None:
        verdict.add(
            "R3", requirement, UNKNOWN,
            "the campaign declaration is unavailable, so the recorded settlement "
            "cannot be recomputed",
        )
        return
    try:
        total = _ledger_total(document, campaign.cycle_id)
    except LookupError as error:
        verdict.add(
            "R3", requirement, UNKNOWN,
            f"accounting evidence is incomplete: {error}",
        )
        return
    except (KeyError, TypeError, ValueError) as error:
        verdict.add(
            "R3", requirement, FAIL,
            f"accounting entries, flags or totals are inconsistent: {error}",
        )
        return
    settlement = settle_campaign(campaign, total=total)
    if not isinstance(recorded["budget_compliant"], bool):
        verdict.add(
            "R3", requirement, FAIL,
            "recorded budget_compliant must be a JSON boolean",
        )
        return
    if recorded["budget_compliant"] != settlement.compliant:
        verdict.add(
            "R3", requirement, FAIL,
            f"{SETTLEMENT_DISAGREES_WITH_RECORD}: the run recorded "
            f"budget_compliant={recorded['budget_compliant']!r} while the artifact it "
            "pinned settles "
            + (
                "compliant"
                if settlement.compliant
                else "non-compliant; " + "; ".join(settlement.failure_reasons)
            ),
        )
        return
    verdict.add(
        "R3", requirement, PASS,
        f"the artifact is the one the run pinned ({pinned[:12]}) and settles as the "
        f"record says ({'compliant' if settlement.compliant else 'non-compliant'})",
    )


# ---------------------------------------------------------------------------
# R4, R5 — the declared retention profile, recomputed and coupled to the run
# ---------------------------------------------------------------------------


def _recomputed_retention(
    campaign: CampaignManifest | None,
    candidate: MeasuredArm | None,
    parent: MeasuredArm | None,
) -> tuple[set[tuple[str, str]] | None, str]:
    """Production's own answer for the declared profile, or why it is unknown."""
    profile = campaign.retention_profile if campaign is not None else None
    if campaign is None:
        return None, "the campaign declaration is unavailable"
    if profile is None:
        return None, "the campaign declares no retention profile"
    if candidate is None or parent is None:
        return None, "candidate and parent arms are both needed to recompute retention"

    def results(arm: MeasuredArm) -> dict[str, BenchmarkResult]:
        rows: dict[str, BenchmarkResult] = {}
        for constraint in profile.constraints:
            run = arm.run_for(constraint.benchmark)
            if run is None or run.score is None:
                continue
            rows[constraint.benchmark] = BenchmarkResult(
                benchmark_qualified_id=constraint.benchmark,
                score=float(run.score),
                samples=tuple(float(value) for value in run.per_sample_scores),
                measurement_origin=str(run.measurement_origin),
            )
        return rows

    candidate_rows = results(candidate)
    parent_rows = results(parent)
    missing = sorted(
        {
            constraint.benchmark
            for constraint in profile.constraints
            if constraint.benchmark not in candidate_rows
            or constraint.benchmark not in parent_rows
        }
    )
    if missing:
        return None, f"no usable run for {missing} in both arms"
    violations = evaluate_retention(
        profile,
        parent_values=retention_values(profile, parent_rows, candidate_side=False),
        candidate_values=retention_values(profile, candidate_rows, candidate_side=True),
    )
    return {(violation.code, violation.reason) for violation in violations}, ""


def _retention_gate(
    verdict: Verdict,
    campaign: CampaignManifest | None,
    candidate: MeasuredArm | None,
    parent: MeasuredArm | None,
) -> set[tuple[str, str]] | None:
    requirement = "declared retention profile holds on the measured arms"
    recomputed, why = _recomputed_retention(campaign, candidate, parent)
    if recomputed is None:
        verdict.add("R4", requirement, UNKNOWN, why)
        return None
    breaches = sorted(reason for _code, reason in recomputed)
    verdict.add(
        "R4",
        requirement,
        PASS if not breaches else FAIL,
        (
            f"no violation of {campaign.retention_profile.profile_id}"
            if not breaches
            else "; ".join(breaches)
        ),
    )
    return recomputed


def _decision_gate(
    verdict: Verdict,
    run_root: Path,
    campaign: CampaignManifest | None,
    recomputed: set[tuple[str, str]] | None,
) -> None:
    """The run's recorded decision and this judge must not disagree.

    The run enforces the declared profile before it records anything; a judge
    that never opened ``campaign-run.json`` could certify a candidate the branch
    had already refused. Two directions are checked: the breaches each side
    found must be the same set, and a run that refused itself may not be
    certified here — this judge audits no rule the run could have applied
    besides the ones above.
    """
    requirement = "the run's recorded decision agrees with this judge"
    record = _load_json(run_root / RUN_RECORD)
    if record is None:
        verdict.add(
            "R5", requirement, UNKNOWN,
            f"{RUN_RECORD_ABSENT}: {RUN_RECORD} is absent or unreadable, so the "
            "run's own decision cannot be audited",
        )
        return
    if campaign is not None and str(record.get("cycle_id", "")) != str(campaign.cycle_id):
        verdict.add(
            "R5", requirement, UNKNOWN,
            f"{RUN_RECORD_WRONG_CYCLE}: the record belongs to cycle "
            f"{record.get('cycle_id')!r}, not {campaign.cycle_id!r}",
        )
        return
    decision = (record.get("promotion") or {}).get("decision") or {}
    if not isinstance(decision, Mapping) or not decision:
        verdict.add(
            "R5", requirement, UNKNOWN,
            f"{RUN_DECISION_ABSENT}: the run record carries no promotion decision",
        )
        return
    if recomputed is None:
        verdict.add(
            "R5", requirement, UNKNOWN,
            "this judge could not recompute the retention profile, so there is "
            "nothing to compare the recorded decision with",
        )
        return

    reasons = [str(reason) for reason in decision.get("reasons", ()) or ()]
    recorded = {
        reason.split(":", 1)[0].strip()
        for reason in reasons
        if reason.startswith("RETENTION_")
    }
    recorded_codes = sorted(recorded)
    recomputed_codes = sorted({code for code, _reason in recomputed})
    run_verdict = str(record.get("verdict", ""))
    breach_list = sorted(reason for _code, reason in recomputed)

    if recorded_codes != recomputed_codes:
        verdict.add(
            "R5", requirement, FAIL,
            f"{RETENTION_RECOMPUTATION_DISAGREES}: the run recorded "
            f"{recorded_codes or 'no breach'} while production's evaluator, re-run "
            f"here over the same arms, gives {recomputed_codes or 'no breach'}; the "
            "two authorities disagree about the same candidate",
        )
        return
    if run_verdict == "PROMOTED" and recomputed_codes:
        verdict.add(
            "R5", requirement, FAIL,
            f"the run recorded PROMOTED together with declared-gate breaches "
            f"{breach_list}",
        )
        return
    if run_verdict in {"REJECTED", "TAINTED"}:
        verdict.add(
            "R5", requirement, FAIL,
            f"{DECLARED_GATE_REJECTED_RUN}: the run refused this candidate "
            f"({'; '.join(reasons) or 'no reasons recorded'}); this judge audits no "
            "declared rule beyond its own rows and cannot overturn that refusal",
        )
        return
    verdict.add(
        "R5", requirement, PASS,
        f"run verdict {run_verdict or 'unrecorded'}; recorded and recomputed "
        f"breaches agree ({recomputed_codes or 'none'})",
    )


# ---------------------------------------------------------------------------
# R6, R7 — the tool-use target: re-score, then the declared delta
# ---------------------------------------------------------------------------


def _target_benchmark(campaign: CampaignManifest | None) -> tuple[str | None, str]:
    if campaign is None:
        return None, "the campaign declaration is unavailable"
    targets = tuple(campaign.target_benchmarks)
    if len(targets) != 1:
        return None, (
            f"this judge re-scores one declared target instrument; the campaign "
            f"declares {list(targets)}"
        )
    return targets[0], ""


def _suite_for(
    campaign: CampaignManifest | None, benchmark: str
) -> tuple[Any | None, str]:
    """The declared suite (dataset + readout) for one benchmark."""
    if campaign is None:
        return None, "the campaign declaration is unavailable"
    declared = str(campaign.evaluation_material_path or "").strip()
    if not declared:
        return None, (
            "the campaign declares no evaluation_material_path, so the readout for "
            f"{benchmark} is undeclared"
        )
    path = Path(declared)
    if not path.is_absolute():
        path = REPO / path
    try:
        material = EvaluationMaterial.load(path)
    except CandidateEvaluationRefusal as error:
        return None, str(error)
    for suite in material.suites:
        if suite.benchmark_qualified_id == benchmark:
            return suite, ""
    return None, f"the evaluation material declares no suite for {benchmark}"


def _tool_use_gate(
    verdict: Verdict,
    run_root: Path,
    campaign: CampaignManifest | None,
    candidate: MeasuredArm | None,
    parent: MeasuredArm | None,
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]] | None:
    """R6: both arms' reported target score must reproduce from their items."""
    requirement = "tool-use target re-scored with the declared tool_call_exact readout"
    target, why = _target_benchmark(campaign)
    if target is None:
        verdict.add("R6", requirement, UNKNOWN, why)
        return None
    suite, why = _suite_for(campaign, target)
    if suite is None:
        verdict.add("R6", requirement, UNKNOWN, why)
        return None
    if suite.scoring != TOOL_USE_SCORING:
        verdict.add(
            "R6", requirement, FAIL,
            f"the campaign declares {suite.scoring!r} for {target}; the target "
            f"instrument is scored {TOOL_USE_SCORING!r}, so this is a different "
            "instrument's number",
        )
        return None

    flags: dict[str, dict[str, int]] = {}
    for role, arm in (("candidate", candidate), ("parent", parent)):
        label = f"{role} {target}"
        if arm is None:
            verdict.add(
                "R6", requirement, UNKNOWN,
                f"{label}: the {role} arm is unusable",
            )
            return None
        run = arm.run_for(target)
        if run is None:
            verdict.add(
                "R6", requirement, UNKNOWN,
                f"{label}: no usable run for the declared target benchmark",
            )
            return None
        items, status, detail = _read_evidence(run_root, run, label=label)
        if items is None:
            verdict.add("R6", requirement, status, detail)
            return None
        status, detail = _reconcile(items, run, scoring=suite.scoring, label=label)
        verdict.add("R6", requirement, status, detail)
        if status != PASS:
            return None
        item_flags: dict[str, int] = {}
        for item in items:
            key = _item_key(item)
            if key is None or key in item_flags:
                verdict.add(
                    "R6", requirement, UNKNOWN,
                    f"{label}: an item has no identity or repeats one ({key!r}), so "
                    "the arms cannot be paired",
                )
                return None
            item_flags[key] = int(
                score(str(item.get("prediction", "")), _expected_text(item, suite.scoring), suite.scoring)
                >= 0.5
            )
        flags[role] = item_flags
    return flags.get("candidate"), flags.get("parent")


def _paired_delta_gate(
    verdict: Verdict,
    campaign: CampaignManifest | None,
    candidate_flags: Mapping[str, int] | None,
    parent_flags: Mapping[str, int] | None,
) -> None:
    """R7: the improvement must clear the bar the *campaign* declared."""
    requirement = "target improvement clears the declared min_delta"
    if campaign is None:
        verdict.add("R7", requirement, UNKNOWN, "the campaign declaration is unavailable")
        return
    threshold = campaign.target_threshold
    if threshold is None:
        verdict.add(
            "R7", requirement, UNKNOWN,
            "the campaign declares no target_threshold, so this judge has no bar "
            "to compare against — and it may not invent one",
        )
        return
    target = campaign.target_benchmarks[0] if len(campaign.target_benchmarks) == 1 else ""
    if threshold.benchmark != target:
        verdict.add(
            "R7", requirement, UNKNOWN,
            f"the declared bar is about {threshold.benchmark!r}, not the audited "
            f"target {target!r}",
        )
        return
    if candidate_flags is None or parent_flags is None:
        verdict.add(
            "R7", requirement, UNKNOWN,
            "the paired per-item evidence needed for the delta is unavailable",
        )
        return
    if not parent_flags or set(parent_flags) != set(candidate_flags):
        verdict.add(
            "R7", requirement, UNKNOWN,
            "the two arms were not measured on the same item identities, so no "
            "paired comparison exists",
        )
        return
    keys = sorted(parent_flags)
    before = [float(parent_flags[key]) for key in keys]
    after = [float(candidate_flags[key]) for key in keys]
    comparison = compare(before, after, min_effect=float(threshold.min_delta))
    verdict.add(
        "R7",
        requirement,
        PASS if comparison.verdict == "improved" else FAIL,
        f"paired delta {comparison.delta:+.4f} vs declared min_delta "
        f"{float(threshold.min_delta):.4f} over {len(keys)} item(s); production's "
        f"compare() answers {comparison.verdict!r} "
        f"(parent rate {sum(before) / len(before):.3f}, candidate rate "
        f"{sum(after) / len(after):.3f})",
    )


# ---------------------------------------------------------------------------
# R8 — protected math, re-scored with the declared suite readout
# ---------------------------------------------------------------------------


def _protected_gate(
    verdict: Verdict,
    run_root: Path,
    campaign: CampaignManifest | None,
    candidate: MeasuredArm | None,
    parent: MeasuredArm | None,
) -> None:
    requirement = "protected math re-scores with its declared readout"
    if campaign is None:
        verdict.add("R8", requirement, UNKNOWN, "the campaign declaration is unavailable")
        return
    protected = tuple(campaign.protected_benchmarks)
    if not protected:
        verdict.add(
            "R8", requirement, UNKNOWN, "the campaign declares no protected benchmarks"
        )
        return
    for benchmark in protected:
        suite, why = _suite_for(campaign, benchmark)
        if suite is None:
            verdict.add("R8", f"{benchmark}: {requirement}", UNKNOWN, why)
            continue
        for role, arm in (("candidate", candidate), ("parent", parent)):
            label = f"{role} {benchmark}"
            if arm is None:
                verdict.add(
                    "R8", f"{benchmark}: {requirement}", UNKNOWN,
                    f"{label}: the {role} arm is unusable",
                )
                break
            run = arm.run_for(benchmark)
            if run is None:
                verdict.add(
                    "R8", f"{benchmark}: {requirement}", UNKNOWN,
                    f"{label}: no usable run for the protected benchmark",
                )
                break
            items, status, detail = _read_evidence(run_root, run, label=label)
            if items is None:
                verdict.add("R8", f"{benchmark}: {requirement}", status, detail)
                break
            status, detail = _reconcile(items, run, scoring=suite.scoring, label=label)
            verdict.add("R8", f"{benchmark}: {requirement}", status, detail)
            if status != PASS:
                break


# ---------------------------------------------------------------------------
# R9, R10 — Gate 0: termination and degeneracy
# ---------------------------------------------------------------------------


def _gate0_rates(
    items: Sequence[Mapping[str, Any]],
    *,
    max_new_tokens: int,
    label: str,
) -> tuple[float, float, int]:
    """(termination rate, degeneracy rate, n) from production's own diagnostics."""
    diagnostics = GenerationDiagnostics.from_items(
        items, max_new_tokens=max_new_tokens, seed=0, source=label
    )
    degenerate = sum(
        1
        for entry in diagnostics.per_prompt
        if entry["cap_hit"]
        or _has_repeated_lines(str(entry["completion"]))
        or _trigram_ratio(str(entry["completion"])) < DEGENERACY_TRIGRAM_MIN
    )
    return (
        float(diagnostics.eos_termination_rate),
        degenerate / diagnostics.n_prompts,
        diagnostics.n_prompts,
    )


def _gate0(
    verdict: Verdict,
    campaign: CampaignManifest | None,
    candidate_items: Sequence[Mapping[str, Any]] | None,
    parent_items: Sequence[Mapping[str, Any]] | None,
) -> None:
    requirement_termination = "Gate 0: termination is no worse than the parent by more than 2 points"
    requirement_degeneracy = (
        "Gate 0: degeneracy is no worse than the parent by more than 2 points"
    )
    if campaign is None:
        verdict.add(
            "R9", requirement_termination, UNKNOWN,
            "the campaign declaration is unavailable",
        )
        verdict.add(
            "R10", requirement_degeneracy, UNKNOWN,
            "the campaign declaration is unavailable",
        )
        return
    decoding = (campaign.protection.protocol.decoding if campaign.protection.protocol else {}) or {}
    max_new_tokens = decoding.get("max_new_tokens")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
        verdict.add(
            "R9", requirement_termination, UNKNOWN,
            "the declared protection protocol names no positive max_new_tokens, so "
            "a cap hit cannot be decided",
        )
        verdict.add("R10", requirement_degeneracy, UNKNOWN, "the same undeclared token budget")
        return
    if candidate_items is None or parent_items is None:
        verdict.add(
            "R9", requirement_termination, UNKNOWN,
            "the target generations needed for Gate 0 are unavailable",
        )
        verdict.add("R10", requirement_degeneracy, UNKNOWN, "the same missing generations")
        return
    try:
        candidate_eos, candidate_degenerate, n = _gate0_rates(
            candidate_items, max_new_tokens=max_new_tokens, label="candidate"
        )
        parent_eos, parent_degenerate, parent_n = _gate0_rates(
            parent_items, max_new_tokens=max_new_tokens, label="parent"
        )
    except CandidateEvaluationRefusal as error:
        verdict.add("R9", requirement_termination, UNKNOWN, str(error))
        verdict.add("R10", requirement_degeneracy, UNKNOWN, str(error))
        return
    verdict.add(
        "R9",
        requirement_termination,
        PASS if candidate_eos >= parent_eos - GATE0_TOLERANCE else FAIL,
        f"candidate {candidate_eos:.3f} vs parent {parent_eos:.3f} over "
        f"{n}/{parent_n} generation(s); the bar is parent - {GATE0_TOLERANCE:.2f}",
    )
    verdict.add(
        "R10",
        requirement_degeneracy,
        PASS if candidate_degenerate <= parent_degenerate + GATE0_TOLERANCE else FAIL,
        f"candidate {candidate_degenerate:.3f} vs parent {parent_degenerate:.3f} over "
        f"{n}/{parent_n} generation(s); the bar is parent + {GATE0_TOLERANCE:.2f}",
    )


# ---------------------------------------------------------------------------
# the judgement
# ---------------------------------------------------------------------------


def branch_verdict(verdict: Verdict) -> str:
    """PROMOTED / REJECTED / INCONCLUSIVE, from the gating rows only."""
    rows = verdict.thresholds()
    if not rows:
        return "INCONCLUSIVE"
    statuses = {row[2] for row in rows}
    if FAIL in statuses:
        return "REJECTED"
    if UNKNOWN in statuses:
        return "INCONCLUSIVE"
    return "PROMOTED"


def judge(run_root: Path) -> int:
    verdict = Verdict()
    campaign, campaign_why = _load_campaign()

    candidate_version = campaign.resolved_candidate_version() if campaign is not None else ""
    parent_version = campaign.parent_version if campaign is not None else ""
    candidate, candidate_why = _open_arm(
        run_root / CANDIDATE_REPORT,
        origin=MEASURED_THIS_GENERATION,
        generation=candidate_version,
        label="candidate",
    )
    parent, parent_why = _open_arm(
        run_root / PARENT_REPORT,
        origin=MEASURED_PARENT,
        generation=parent_version,
        label="parent",
    )

    _identity_gate(verdict, run_root, candidate, candidate_why)
    _settlement_gate(verdict, run_root, campaign)
    _accounting_digest_gate(verdict, run_root, campaign)
    recomputed = _retention_gate(verdict, campaign, candidate, parent)
    _decision_gate(verdict, run_root, campaign, recomputed)
    candidate_flags, parent_flags = _tool_use_gate(
        verdict, run_root, campaign, candidate, parent
    ) or (None, None)
    _paired_delta_gate(verdict, campaign, candidate_flags, parent_flags)
    _protected_gate(verdict, run_root, campaign, candidate, parent)

    # Gate 0 is read from the same target generations R6 verified: the
    # termination and degeneracy facts live in the worker's per-item rows, not
    # in the aggregate the report carries.
    target, _why = _target_benchmark(campaign)
    candidate_items = parent_items = None
    if target is not None and candidate is not None and parent is not None:
        candidate_run = candidate.run_for(target)
        parent_run = parent.run_for(target)
        if candidate_run is not None:
            loaded, _status, _detail = _read_evidence(
                run_root, candidate_run, label=f"candidate {target}"
            )
            candidate_items = loaded
        if parent_run is not None:
            loaded, _status, _detail = _read_evidence(
                run_root, parent_run, label=f"parent {target}"
            )
            parent_items = loaded
    _gate0(verdict, campaign, candidate_items, parent_items)

    descriptive: list[str] = []
    reports_dir = run_root / DESCRIPTIVE_DIR
    if reports_dir.is_dir():
        descriptive = sorted(path.name for path in reports_dir.glob("*.json"))

    final = branch_verdict(verdict)
    print(f"judge: {VERSION}")
    print(f"prereg: {PREREG}")
    print(f"campaign declaration: {CAMPAIGN_MANIFEST}" + (f" ({campaign_why})" if campaign is None else ""))
    print(f"run root: {run_root}")
    print()
    print(verdict.render())
    print()
    print(
        "descriptive reports (non-gating, never read by a row above): "
        + (", ".join(descriptive) if descriptive else "none")
    )
    print(f"VERDICT: {final}")
    print(f"DETAIL: {verdict.finalize_status()}")
    return {"PROMOTED": 0, "REJECTED": 1}.get(final, 2)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    run_root = Path(argv[1])
    if not run_root.is_dir():
        print(f"run root does not exist: {run_root}")
        return 2
    return judge(run_root)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
