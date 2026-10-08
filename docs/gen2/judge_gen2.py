#!/usr/bin/env python3
"""Frozen mechanical judge for the gen2 response-surface-compliance cycle.

Frozen with ``docs/quals/GEN2_PREREG_2026-09-17.md`` and its amendments
``GEN2_PREREG_AMENDMENT1/2/3_2026-09-18.md``; amended by
``GEN2_PREREG_AMENDMENT15_2026-10-04.md`` (the declared-gate coupling, T21/T22)
and ``GEN2_PREREG_AMENDMENT16_2026-10-04.md`` (the settlement artifact pin, T23);
amended by ``GEN2_PREREG_AMENDMENT17_2026-10-08.md`` (the answer readout disclosure, T24);
amended by ``GEN2_PREREG_AMENDMENT18_2026-10-08.md`` (the row-label audit: F2-F7 applied, no
decision moved).
Thresholds may not change after candidate results are visible. Reads the run's
durable artifacts read-only and emits one verdict table over the branch rules.

Evidence is verified, never assumed. Two rules follow from that, and both are
fail-closed:

* the contamination evidence is the artifact the *campaign pinned* --
  ``contamination_manifest_path``. The judge does not read a file that merely
  sits in the run root, and it will not accept one that disagrees with the pin:
  an unpinned artifact cannot certify anything (T12/T18);
* every protected measurement is bound to bytes that exist. A slice row must
  name its raw artifact *and* the artifact's sha256, and the judge recomputes
  that digest from the file (T11). A row whose samples do not add up to its own
  aggregate, or that names a file which is not there, is refused rather than
  certified.

The judge owns the *frozen policy*: the benchmark set, the thresholds, the
branch rules, and how they compose. It deliberately owns no second
implementation of anything the production engine already decides:

* directory/file hashing -> ``chowder.growth.training_binding.directory_digest``
  (the same canonical digest the gen1 re-adjudication re-verifies artifacts
  with) and ``chowder.provenance.sha256_file`` for single files;
* resource settlement -> ``chowder.growth.campaign.settle_campaign`` over the
  campaign manifest, so the judge and ``chowder growth campaign settle``
  cannot disagree about the same accounting artifact;
* measurement provenance -> the origin constants in
  ``chowder.evals.result`` (a row is candidate evidence only when it says so);
* contamination interpretation -> ``chowder.growth.metric_binding.MetricBinder``
  (absence binds UNKNOWN, never clean);
* statistics -> ``chowder.growth.statistics.compare``;
* report parsing -> ``chowder.evals.result.EvalReport``.

Frozen arm artifacts, all ``EvalReport`` JSON, one per measured generation:

  candidate_evaluation.json   the gen2 candidate        (MEASURED_THIS_GENERATION)
  parent_evaluation.json      the gen1 parent adapter   (MEASURED_PARENT)
  baseline_evaluation.json    the trusted ancestor gen0 (MEASURED_PARENT)

Each arm carries one ``generation-diagnostics@gen2-response-surface-v1`` run
whose ``metadata.per_prompt`` holds the raw completions the judge scores,
plus one run per protected mini-slice carrying its protocol metadata. The
judge never reads a parent score out of the candidate's own file: the parent
arm is its own provenance-bound artifact.

With amendment 15 the judge reads one more artifact from the same directory:
``campaign-run.json``, the run's own record of what it decided
(``CampaignRun.to_dict``). It is read, never written, and it is what lets the
judge stop certifying a candidate the run already refused: T21 audits the
recorded decision and T22 recomputes the declared ``retention_profile`` through
production's own evaluator (``evaluate_retention`` plus the promotion path's
``retention_values``) and compares the two answers. Neither row re-derives the
policy -- the declaration owns the constraint, production owns the comparison,
and this judge owns only the agreement between the two authorities. A run root
with no such record is UNKNOWN, never a pass.

Amendment 16 adds T23, over the same record and the accounting artifact T13
already settles. The record pins the ledger digest of the artifact it settled
(``CampaignRun.cost['accounting_digest']``), so T23 recomputes that digest with
production's ``ledger_digest`` and requires the artifact's own settlement to
agree with the recorded one. Without it a run the branch refused on its frozen
envelope could be certified PROMOTED by editing one file in the run root: T13
would settle the edited bytes and find them compliant, and nothing compared
that answer with the refusal the run had recorded.

Amendment 17 adds T24, over the same ``metadata["per_prompt"]`` rows T5 counts.
T5's row said "answer correctness" while the quantity it computes is *presence*:
the expected string appears somewhere on the answer surface. Presence is what
the prereg declares for that row ("answer-correct (expected string present after
reasoning)"), and it is not the readout production owns -- ``evaluators.scoring``
declares the scoring mode, extracts the answer after the last ``</think>``, and
refuses an unclosed reasoning budget. A completion that mentions the expected
string without answering it ("There are 17 continents." against "7") is present
and is 0.0 under every readout production declares. T24 reads those rows, runs
both declared production readouts over each completion through production's own
``score``, and refuses only the irreducible disagreement -- an item presence
accepts that every declared readout refuses. The row is labelled by the quantity
it measures, and the gap between the two readouts is measured on the evidence
the run already wrote instead of being argued about. No threshold moves.

Usage:
    python docs/gen2/judge_gen2.py <run_root>

where ``<run_root>`` holds the three arm artifacts, plus
``cycle_compute_accounting.json`` and ``chosen_candidate.json``, and the
contamination manifest the campaign pinned (``contamination_manifest_path``),
copied in verbatim by the run.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE.parent / "quals"))
sys.path.insert(0, str(REPO / "src"))

from quals_harness import (  # noqa: E402  (paths inserted above, by design)
    FAIL,
    INFO,
    PASS,
    UNKNOWN,
    Verdict,
)

from chowder.evals.result import (  # noqa: E402
    MEASURED_PARENT,
    MEASURED_THIS_GENERATION,
    BenchmarkRun,
    EvalReport,
)
from chowder.growth.campaign import CampaignManifest, settle_campaign  # noqa: E402
from chowder.growth.cycle import retention_values  # noqa: E402
from chowder.growth.promotion import BenchmarkResult  # noqa: E402
from chowder.growth.retention import (  # noqa: E402
    RetentionConstraint,
    RetentionViolation,
    evaluate_retention,
)
from chowder.growth import certification as _certification  # noqa: E402
from chowder.growth.certification import ArmError, MeasuredArm  # noqa: E402
from chowder.growth.catalog import default_registry  # noqa: E402
from chowder.growth.compute_cost import ComputeCost, ledger_digest  # noqa: E402
from chowder.growth.metric_binding import MetricBinder  # noqa: E402
from chowder.growth.statistics import compare  # noqa: E402
from chowder.growth.training_binding import directory_digest  # noqa: E402
from chowder.provenance import sha256_file  # noqa: E402

# ---------------------------------------------------------------------------
# Frozen policy (prereg section 2 + amendment 1). Do not edit after results
# are visible.
# ---------------------------------------------------------------------------
CAMPAIGN_MANIFEST = HERE / "gen2_campaign.json"

INSTRUMENT_ID = "generation-diagnostics@gen2-response-surface-v1"
REQUIRED_PROTECTED = ("math500@2024-04", "mgsm@2022-11")
TRUSTED_ANCESTOR_VERSION = "gen0"

PROTECTED_N_SAMPLES = 16
PROTECTED_SAMPLE_INDICES = tuple(range(16))
PROTECTED_SEED = 1234
PROTECTED_SHUFFLE = False
PROTECTED_DECODING = {"temperature": 0.0, "do_sample": False, "max_new_tokens": 512}
PROTECTED_PROMPT_POLICY = "chat_template"

#: The shape a measurement's declared artifact digest must have before the
#: judge will recompute it, and the tolerance for "the aggregate *is* the mean
#: of the per-sample values".
ARTIFACT_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
SAMPLE_MEAN_TOLERANCE = 1e-6

#: Named reasons, so a refusal says which verification failed rather than only
#: that something did.
CONTAMINATION_PIN_ABSENT = "CONTAMINATION_PIN_ABSENT"
CONTAMINATION_PIN_MISSING = "CONTAMINATION_PIN_MISSING"
CONTAMINATION_EVIDENCE_NOT_IN_RUN_ROOT = "CONTAMINATION_EVIDENCE_NOT_IN_RUN_ROOT"
CONTAMINATION_EVIDENCE_NOT_PINNED = "CONTAMINATION_EVIDENCE_NOT_PINNED"
#: Why the run's own promotion decision could not be audited (GEN2 prereg
#: amendment 15). Absence is UNKNOWN, never an assumed pass -- the same rule
#: every other gate here follows.
RUN_RECORD_ABSENT = "RUN_RECORD_ABSENT"
RUN_RECORD_WRONG_CYCLE = "RUN_RECORD_WRONG_CYCLE"
RUN_DECISION_ABSENT = "RUN_DECISION_ABSENT"
#: A candidate the run refused on a declared gate cannot be certified here: this
#: judge audits no declared gate, so its own table cannot overturn that refusal.
DECLARED_GATE_REJECTED_RUN = "DECLARED_GATE_REJECTED_RUN"
#: A run record that contradicts itself, or a gate the declaration does not name.
UNDECLARED_GATE_IN_RUN = "UNDECLARED_GATE_IN_RUN"
#: The run's recorded breaches and this judge's own recomputation, through the
#: production evaluator, do not match: the two authorities disagree about the
#: same candidate, and neither may be assumed right.
RETENTION_RECOMPUTATION_DISAGREES = "RETENTION_RECOMPUTATION_DISAGREES"
#: The measurement reason codes are production's (see the verification block
#: below): the judge re-exports them so a refusal is named identically wherever
#: it is raised.

#: Paired-decision constants for the target gates: the minimum effect on the
#: rate delta, and the strict-improvement count the absolute path requires.
TARGET_MIN_EFFECT = 0.25
TARGET_STRICT_PROMPT_MIN = 12

TARGET_DUPLICATION_MAX = 0.125  # parent 0.688
TARGET_ECHO_MAX = 0.062  # parent 0.438
TARGET_FORMAT_MIN = 8  # of the 8 declared constrained instrument prompts
PROTECTED_ANSWER_CORRECT_MIN = 15  # of 16
PROTECTED_READOUT_DISAGREEMENT_MAX = 0  # of the candidate's per-prompt rows (amendment 17)

#: The readouts production declares for text scoring (``evaluators.scoring``).
#: T24 compares presence against both and counts a disagreement only when *every*
#: one of them refuses the item. ``exact_match`` is deliberately absent: it is
#: strictly stricter than ``normalized_exact_match``, so it cannot accept anything
#: these two refuse, and listing it would only restate one of them.
PRODUCTION_READOUT_MODES = ("normalized_exact_match", "final_number_match")
PROTECTED_EOS_MIN = 0.900
PROTECTED_CAP_MAX = 0.100
PROTECTED_UNCLOSED_THINK_MAX = 0.250
PROTECTED_LOOP_MAX = 0
PROTECTED_TRIGRAM_MIN = 0.900
SLICE_REGRESSION_MAX = 0.0625

REQUIRED_RECIPES_MIN = 2
CONTAMINATED = {"KNOWN_CONTAMINATION", "POSSIBLE"}

#: The durable record the run writes beside the evidence it certifies:
#: ``CampaignRun.to_dict`` at ``<state_root>/campaign-run.json``, the same
#: directory this judge reads the arms from.
RUN_RECORD_NAME = "campaign-run.json"

#: Why the run's settlement could not be audited against the artifact it names
#: (T23). All of these are one class of finding -- the record and the
#: accounting artifact disagree, or one of the two is not there to compare --
#: and each fails closed rather than certifying the half that happens to be
#: readable.
ACCOUNTING_ARTIFACT_MOVED = "ACCOUNTING_ARTIFACT_MOVED"
ACCOUNTING_UNPINNED = "ACCOUNTING_UNPINNED"
ACCOUNTING_UNSETTLED_BY_RUN = "ACCOUNTING_UNSETTLED_BY_RUN"
SETTLEMENT_DISAGREES_WITH_RECORD = "SETTLEMENT_DISAGREES_WITH_RECORD"

#: Why T5's presence readout could not be reconciled with production's declared
#: scoring (T24, amendment 17). Presence is what the prereg declares for T5; T24
#: asks only that a count labelled as an answer be reachable as an answer when
#: production reads the same completion.
READOUT_DISAGREES_WITH_PRODUCTION = "READOUT_DISAGREES_WITH_PRODUCTION"
READOUT_UNMEASURED = "READOUT_UNMEASURED"

#: Production's declared-gate vocabulary, read off its owner rather than restated
#: here. ``RetentionViolation.code`` answers NaN -> ``RETENTION_UNMEASURED`` and
#: otherwise selects by constraint kind, so one violation per reachable shape is
#: the whole vocabulary;
#: ``test_growth_gen2_judge.py`` asserts this set equals the codes the promotion
#: gate actually emits, so a new shape cannot slip past a prefix match here.
RETENTION_CODES: frozenset[str] = frozenset(
    RetentionViolation(
        dimension="code-probe",
        constraint=RetentionConstraint(
            dimension="code-probe",
            kind=kind,
            value=0.0,
            benchmark="code-probe@2026-01",
        ),
        measured=measured,
        detail="",
    ).code
    for kind, measured in (
        ("max-regression", 0.0),           # RETENTION_REGRESSION
        ("absolute-floor", 0.0),           # RETENTION_FLOOR
        ("max-regression", float("nan")),  # RETENTION_UNMEASURED
    )
)
RETENTION_REASON_PREFIX = "RETENTION_"

#: The frozen 16-prompt instrument, in order, with the expected answer the
#: correctness check looks for on the answer surface (the same pairs the gen1
#: driver froze; ``tests/test_growth_gen2_judge.py`` cross-checks this list
#: against that source so the two cannot drift).
INSTRUMENT_PROMPTS: tuple[tuple[str, str], ...] = (
    ("Reply with exactly: ping", "ping"),
    ("What is 17 * 23? Answer with the number only.", "391"),
    ("Name the capital of Australia in one word.", "Canberra"),
    ("Write one sentence describing rain.", "rain"),
    ("Count from 1 to 5, digits only.", "5"),
    ("What is the boiling point of water in Celsius?", "100"),
    ("Translate 'good morning' into French.", "bonjour"),
    ("Complete: The opposite of hot is", "cold"),
    ("List the first three prime numbers.", "2"),
    ("Who wrote Romeo and Juliet?", "Shakespeare"),
    ("What is 100 divided by 4?", "25"),
    ("Say 'done' and nothing else.", "done"),
    ("Give one synonym for 'happy'.", "joyful"),
    ("How many continents are there?", "7"),
    ("What color is a banana?", "yellow"),
    ("Answer with a single word: 2 + 2 =", "4"),
)

#: The 8 constrained prompts whose format compliance is a hard target gate.
CONSTRAINED_PROMPTS = frozenset(
    {
        "What is 17 * 23? Answer with the number only.",
        "Name the capital of Australia in one word.",
        "Count from 1 to 5, digits only.",
        "What is the boiling point of water in Celsius?",
        "Say 'done' and nothing else.",
        "How many continents are there?",
        "Answer with a single word: 2 + 2 =",
        "Reply with exactly: ping",
    }
)


# ---------------------------------------------------------------------------
# instrument scoring (frozen, applied identically to every arm)
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _answer_surface(completion: str) -> str:
    if "</think>" in completion:
        return completion.split("</think>")[-1].strip()
    return completion.strip()


def _answer_present(entry: Mapping[str, Any]) -> bool:
    """T5's frozen rule, in one place: the expected string is on the answer surface.

    Shared by T5 and T24 on purpose -- the row that counts and the row that audits
    the count cannot be allowed to become two readings of the same evidence.
    """
    expected = str(entry.get("expected") or "").lower()
    return expected in _answer_surface(str(entry.get("completion", ""))).lower()


def _answer_duplicated(completion: str) -> bool:
    """The post-think answer also appears inside the reasoning block."""
    if "</think>" not in completion:
        return False
    before = completion.split("</think>")[0]
    after = _answer_surface(completion)
    if not after:
        return False
    first_line = after.splitlines()[0].strip()
    return bool(first_line) and first_line.lower() in before.lower()


def _template_echo(completion: str) -> bool:
    """The continuation opens by echoing the prompt tail + 'assistant'."""
    head = completion.split("<think>")[0]
    if "assistant" not in head:
        return False
    stripped = head.strip()
    return stripped.startswith(("assistant", "with ", "describing", "Answer", "\n"))


def _prompt_key(entry: Mapping[str, Any]) -> str | None:
    """Stable prompt identity: an explicit id, else the frozen prompt text.

    Alignment is never by list position alone -- a reordered but
    identity-equivalent arm must still pair correctly, and a duplicated
    identity must refuse to pair at all.
    """
    for field in ("prompt_id", "prompt"):
        value = entry.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


class _ArmError(Exception):  # noqa: D101 - retained for older call sites/tests
    """The arm artifact is missing, unreadable, or not usable as evidence."""


class Arm(MeasuredArm):
    """One measured generation's durable evidence, parsed and gated by production.

    Production owns what an arm *is*: its provenance, its generation and the
    bytes its ``model_identity`` names. The judge adds only what is specific to
    scoring this cycle's instrument (the per-prompt completions).
    """

    @classmethod
    def open(
        cls,
        path: Path,
        *,
        expected_origin: str,
        expected_generation: str,
        label: str,
    ) -> "Arm":
        loaded = MeasuredArm.load(
            path,
            expected_origin=expected_origin,
            expected_generation=expected_generation,
            label=label,
        )
        return cls(
            label=loaded.label,
            origin=loaded.origin,
            generation=loaded.generation,
            report=loaded.report,
            path=loaded.path,
        )

    def per_prompt(self) -> list[Mapping[str, Any]]:
        run = self.run_for(INSTRUMENT_ID)
        if run is None:
            return []
        entries = run.metadata.get("per_prompt")
        if not isinstance(entries, list):
            return []
        return [entry for entry in entries if isinstance(entry, Mapping)]

    def flags(self, scorer) -> dict[str, int] | None:
        """``{prompt identity: 0/1}`` for one scorer, or None when unusable.

        A duplicated or unidentified prompt identity refuses: pairing two
        arms on ambiguous identities would manufacture a comparison.
        """
        flags: dict[str, int] = {}
        for entry in self.per_prompt():
            key = _prompt_key(entry)
            if key is None or key in flags:
                return None
            flags[key] = 1 if scorer(str(entry.get("completion", ""))) else 0
        return flags


def _duplication_flag(completion: str) -> bool:
    return _answer_duplicated(completion)


def _echo_flag(completion: str) -> bool:
    return _template_echo(completion)


# ---------------------------------------------------------------------------
# frozen gate helpers
# ---------------------------------------------------------------------------


def _aligned(
    parent: Mapping[str, int] | None, candidate: Mapping[str, int] | None
) -> tuple[list[float], list[float]] | None:
    """Two aligned per-prompt vectors, or None when identity does not match."""
    if parent is None or candidate is None:
        return None
    if not parent or set(parent) != set(candidate):
        return None
    keys = sorted(parent)
    return (
        [float(parent[key]) for key in keys],
        [float(candidate[key]) for key in keys],
    )


def _target_gate(
    parent: Sequence[float],
    candidate: Sequence[float],
    *,
    lower_is_better: bool,
    absolute_threshold: float,
) -> tuple[str, str]:
    """The prereg's frozen target rule, mechanically.

    A target gate is met only when the paired per-prompt comparison is
    ``improved`` at the declared minimum effect **or** the frozen absolute
    threshold is crossed with a strictly better rate on at least
    :data:`TARGET_STRICT_PROMPT_MIN` of the prompts. A tie fails.
    """
    comparison = compare(list(parent), list(candidate), min_effect=TARGET_MIN_EFFECT)
    paired_improved = (
        comparison.verdict == "regressed" if lower_is_better else comparison.verdict == "improved"
    )
    candidate_rate = sum(candidate) / len(candidate)
    parent_rate = sum(parent) / len(parent)
    absolute_met = (
        candidate_rate <= absolute_threshold
        if lower_is_better
        else candidate_rate >= absolute_threshold
    )
    if lower_is_better:
        strict = sum(1 for p, c in zip(parent, candidate) if c < p)
    else:
        strict = sum(1 for p, c in zip(parent, candidate) if c > p)
    strict_ok = strict >= TARGET_STRICT_PROMPT_MIN
    # These rates are lower-is-better and ``compare`` answers in a
    # higher-is-better vocabulary, so production's word is kept -- a reader can
    # trace it to the function that produced it -- and its polarity is named
    # beside it. An unlabelled "improved" on a candidate that made the rate worse
    # was amendment 18's finding: the detail said the opposite of the row.
    detail = (
        f"paired {'better' if paired_improved else 'not better'} "
        f"(production's compare() answers {comparison.verdict!r} in a "
        f"higher-is-better vocabulary; delta {comparison.delta:+.4f}, "
        f"min_effect {TARGET_MIN_EFFECT}); candidate rate {candidate_rate:.3f} vs "
        f"parent {parent_rate:.3f}; absolute {'met' if absolute_met else 'not met'}; "
        f"strictly better on {strict}/{len(candidate)} (need {TARGET_STRICT_PROMPT_MIN})"
    )
    if paired_improved:
        return PASS, f"paired improvement — {detail}"
    if absolute_met and strict_ok:
        return PASS, f"absolute threshold + strict count — {detail}"
    return FAIL, detail


# ---------------------------------------------------------------------------
# measurement verification and slice protocol: production owns the mechanism
# ---------------------------------------------------------------------------

#: The verification the judge applies is the production one (PR: "the campaign
#: cannot promote before certification says it may"), so the runner that certifies
#: a run and the judge that audits it cannot disagree about the same row. Only the
#: policy values below -- which benchmarks, which protocol, which tolerance -- are
#: frozen here.
_measurement_artifact = _certification.measurement_artifact
_evidence_problems = _certification.evidence_problems
_protocol_problems = _certification.protocol_problems


def _slice_status(
    arm: MeasuredArm | None,
    qualified_id: str,
    *,
    label: str,
    run_root: Path,
    protocol: Any,
) -> tuple[str, Any, str]:
    """Production's slice verdict, as the (status, run, detail) the table reads."""
    check = _certification.slice_status(
        arm, qualified_id, label=label, run_root=run_root, protocol=protocol
    )
    return check.status, check.run, check.detail
ARTIFACT_SHA256_PATTERN = _certification.ARTIFACT_SHA256_PATTERN
SAMPLE_MEAN_TOLERANCE = _certification.SAMPLE_MEAN_TOLERANCE
MEASUREMENT_ARTIFACT_MISSING = _certification.MEASUREMENT_ARTIFACT_MISSING
MEASUREMENT_ARTIFACT_ESCAPES_RUN_ROOT = _certification.MEASUREMENT_ARTIFACT_ESCAPES_RUN_ROOT
MEASUREMENT_DIGEST_ABSENT = _certification.MEASUREMENT_DIGEST_ABSENT
MEASUREMENT_DIGEST_MISMATCH = _certification.MEASUREMENT_DIGEST_MISMATCH
MEASUREMENT_SAMPLES_INCONSISTENT = _certification.MEASUREMENT_SAMPLES_INCONSISTENT
ARM_GENERATION_MISMATCH = _certification.ARM_GENERATION_MISMATCH
ARM_ADAPTER_DIGEST_MISSING = _certification.ARM_ADAPTER_DIGEST_MISSING
ARM_ADAPTER_DIGEST_MISMATCH = _certification.ARM_ADAPTER_DIGEST_MISMATCH
ARM_BASE_DIGEST_MISSING = _certification.ARM_BASE_DIGEST_MISSING
ARM_BASE_DIGEST_MISMATCH = _certification.ARM_BASE_DIGEST_MISMATCH

#: The declared protocol, as this judge's own frozen constants above (they are
#: the policy; the mechanism that applies them is production's).
JUDGE_PROTOCOL = _certification.ProtocolSpec(
    n_samples=PROTECTED_N_SAMPLES,
    seed=PROTECTED_SEED,
    decoding=dict(PROTECTED_DECODING),
    prompt_policy=PROTECTED_PROMPT_POLICY,
    shuffle=PROTECTED_SHUFFLE,
)


def _digest_of(path: Path) -> str:
    """The repo's canonical digest: directory tree, or single-file sha256."""
    if path.is_dir():
        digest, _entries = directory_digest(path)
        return digest
    return sha256_file(path)


# ---------------------------------------------------------------------------
# the judgement
# ---------------------------------------------------------------------------


def judge(run_root: Path) -> int:
    verdict = Verdict()
    campaign = _load_campaign()

    # Each arm's generation is pinned: the candidate must be the version this
    # campaign is judged under, the parent its declared parent, and the trusted
    # ancestor the frozen one. A protocol-correct row of the wrong generation is
    # not the measurement it claims.
    candidate_version = (
        campaign.resolved_candidate_version() if campaign is not None else TRUSTED_ANCESTOR_VERSION
    )
    parent_version = campaign.parent_version if campaign is not None else ""
    arms: dict[str, Arm | None] = {}
    for key, filename, origin, label, generation in (
        ("candidate", "candidate_evaluation.json", MEASURED_THIS_GENERATION, "candidate", candidate_version),
        ("parent", "parent_evaluation.json", MEASURED_PARENT, "parent (gen1)", parent_version),
        ("ancestor", "baseline_evaluation.json", MEASURED_PARENT, "trusted ancestor (gen0)", TRUSTED_ANCESTOR_VERSION),
    ):
        try:
            arms[key] = Arm.open(
                run_root / filename,
                expected_origin=origin,
                expected_generation=generation,
                label=label,
            )
        except ArmError as error:
            arms[key] = None
            if key == "candidate":
                verdict.add("T1", "candidate instrument provenance + row identity", UNKNOWN, str(error))
    candidate = arms["candidate"]

    _instrument_gates(verdict, candidate, arms["parent"])
    _protected_gates(verdict, arms, campaign, run_root=run_root)
    _evidence_identity_gate(verdict, run_root, arms, campaign)
    _protection_agreement_gate(verdict, campaign)
    _declared_gate_agreement(verdict, arms, campaign, run_root=run_root)
    _contamination_gate(verdict, run_root, campaign)
    _settlement_gates(verdict, run_root, campaign)
    _settlement_artifact_gate(verdict, run_root, campaign)
    _identity_gate(verdict, run_root)

    # Context that does not gate certification.
    verdict.add(
        INFO,
        "parent evidence state",
        INFO,
        "gen1 effective verdict INCONCLUSIVE (target_repair_validated=true)",
    )
    verdict.add(
        INFO,
        "frozen policy",
        INFO,
        "docs/quals/GEN2_PREREG_2026-09-17.md + GEN2_PREREG_AMENDMENT1/2/3/4_2026-09-18.md "
        "+ GEN2_PREREG_AMENDMENT15_2026-10-04.md "
        "+ GEN2_PREREG_AMENDMENT16_2026-10-04.md "
        "+ GEN2_PREREG_AMENDMENT17_2026-10-08.md "
        "+ GEN2_PREREG_AMENDMENT18_2026-10-08.md",
    )

    final = branch_verdict(verdict)
    print(f"run root: {run_root}")
    print()
    print(verdict.render())
    print()
    print(f"VERDICT: {final}")
    print(f"DETAIL: {verdict.finalize_status()}")
    print(f"TARGET_REPAIR_VALIDATED: {str(_target_repair_validated(verdict, final)).lower()}")
    statuses = {row[2] for row in verdict.thresholds()}
    return 0 if statuses <= {PASS} else 1


def _target_repair_validated(verdict: Verdict, final: str) -> bool:
    """Scoped repair: the target gates hold, wider evidence is incomplete.

    Deliberately not a verdict class: it says the repair itself is validated
    from candidate-measured evidence while the candidate stays unpromoted.
    """
    targets = [row for row in verdict.thresholds() if row[0] in {"T2", "T3", "T4"}]
    return final == "INCONCLUSIVE" and bool(targets) and all(row[2] == PASS for row in targets)


def _load_campaign() -> CampaignManifest | None:
    """The frozen campaign declaration, or None when it cannot be read.

    Returning None is not a pass: the gates that need the declared sets or
    ceilings report UNKNOWN rather than settling against an invented envelope.
    """
    try:
        return CampaignManifest.from_file(CAMPAIGN_MANIFEST)
    except Exception:  # noqa: BLE001 - an unusable manifest is UNKNOWN, not a crash
        return None


def branch_verdict(verdict: Verdict) -> str:
    """The promotion-language verdict: PROMOTED / REJECTED / INCONCLUSIVE / TAINTED.

    Scoped repair (target met, wider evidence incomplete) is deliberately not a
    separate verdict: it is INCONCLUSIVE here, the same refusal to certify the
    production rule applies.
    """
    rows = verdict.thresholds()
    statuses = {row[2] for row in rows}
    if FAIL in statuses:
        if any(row[0] == "T12" and row[2] == FAIL for row in rows):
            return "TAINTED"
        return "REJECTED"
    if UNKNOWN in statuses:
        return "INCONCLUSIVE"
    return "PROMOTED"


def _instrument_gates(verdict: Verdict, candidate: Arm | None, parent: Arm | None) -> None:
    if candidate is None:
        for threshold, name in (
            ("T1", "candidate instrument provenance + row identity"),
            ("T2", "answer-duplication target"),
            ("T3", "template-echo target"),
            ("T4", "constrained-prompt format"),
            ("T5", "answer presence"),
            ("T6", "EOS termination"),
            ("T7", "max-token-cap rate"),
            ("T8", "obvious loops"),
            ("T9", "distinct-trigram ratio mean"),
            ("T10", "unclosed think rate"),
        ):
            verdict.add(threshold, name, UNKNOWN, "candidate evaluation artifact unavailable")
        # T24 names its own reason in every branch: a disclosure row that could not
        # be computed is UNKNOWN with the code, never a silent roster label.
        _readout_disclosure_gate(verdict, None)
        return

    duplicates = candidate.duplicate_ids()
    if duplicates:
        verdict.add(
            "T1",
            "candidate instrument provenance + row identity",
            FAIL,
            f"candidate arm duplicates rows for {list(duplicates)}",
        )
    instrument_run = candidate.run_for(INSTRUMENT_ID)
    if instrument_run is None:
        verdict.add(
            "T1",
            "candidate instrument provenance + row identity",
            UNKNOWN,
            f"no single {INSTRUMENT_ID} run carrying {MEASURED_THIS_GENERATION}",
        )
    else:
        verdict.add(
            "T1",
            "candidate instrument provenance + row identity",
            PASS,
            f"measurement_origin={instrument_run.measurement_origin}",
        )

    per_prompt = candidate.per_prompt()
    if instrument_run is None or not per_prompt:
        for threshold, name in (
            ("T2", "answer-duplication target"),
            ("T3", "template-echo target"),
            ("T4", "constrained-prompt format"),
            ("T5", "answer presence"),
        ):
            verdict.add(threshold, name, UNKNOWN, "no candidate per-prompt evidence")
        _readout_disclosure_gate(verdict, candidate)
    else:
        dup_flags = candidate.flags(_duplication_flag)
        echo_flags = candidate.flags(_echo_flag)
        parent_dup = parent.flags(_duplication_flag) if parent is not None else None
        parent_echo = parent.flags(_echo_flag) if parent is not None else None

        _paired_target_gate(
            verdict,
            "T2",
            "answer-duplication",
            parent_flags=parent_dup,
            candidate_flags=dup_flags,
            lower_is_better=True,
            absolute_threshold=TARGET_DUPLICATION_MAX,
        )
        _paired_target_gate(
            verdict,
            "T3",
            "template-echo",
            parent_flags=parent_echo,
            candidate_flags=echo_flags,
            lower_is_better=True,
            absolute_threshold=TARGET_ECHO_MAX,
        )

        constrained = [
            entry
            for entry in per_prompt
            if str(entry.get("prompt", "")).strip() in CONSTRAINED_PROMPTS
        ]
        if len(constrained) != len(CONSTRAINED_PROMPTS):
            verdict.add(
                "T4",
                f"constrained-prompt compliance == {TARGET_FORMAT_MIN}/8",
                UNKNOWN,
                f"located {len(constrained)} of {len(CONSTRAINED_PROMPTS)} declared "
                "constrained prompts",
            )
        else:
            ok = 0
            for entry in constrained:
                surface = _answer_surface(str(entry.get("completion", "")))
                if surface and len(surface) <= 40 and surface.count("\n") <= 1:
                    ok += 1
            verdict.add(
                "T4",
                f"constrained-prompt compliance == {TARGET_FORMAT_MIN}/8",
                PASS if ok >= TARGET_FORMAT_MIN else FAIL,
                f"compliant {ok}/{len(constrained)} constrained prompts",
            )

        correct = sum(1 for entry in per_prompt if _answer_present(entry))
        verdict.add(
            "T5",
            f"answer presence >= {PROTECTED_ANSWER_CORRECT_MIN}/16",
            PASS if correct >= PROTECTED_ANSWER_CORRECT_MIN else FAIL,
            f"measured {correct}/{len(per_prompt)} present; parent 16/16 present "
            "(readout: answer-surface presence; disclosure: T24)",
        )
        # Amendment 17: the count T5 just made is audited against production's
        # declared readouts over the same rows, and the row above is now named by
        # the quantity it measures rather than one it does not.
        _readout_disclosure_gate(verdict, candidate)

    metadata = (instrument_run.metadata or {}) if instrument_run is not None else {}
    diagnostics = (
        ("T6", f"EOS termination >= {PROTECTED_EOS_MIN}", "eos_termination_rate",
         lambda v: v >= PROTECTED_EOS_MIN, "1.000"),
        ("T7", f"cap-hit < {PROTECTED_CAP_MAX}", "max_token_cap_rate",
         lambda v: v < PROTECTED_CAP_MAX, "0.000"),
        ("T8", f"obvious loops <= {PROTECTED_LOOP_MAX}", "obvious_loop_count",
         lambda v: v <= PROTECTED_LOOP_MAX, "0"),
        ("T9", f"distinct-trigram ratio mean >= {PROTECTED_TRIGRAM_MIN}", "distinct_trigram_ratio_mean",
         lambda v: v >= PROTECTED_TRIGRAM_MIN, "0.973"),
        ("T10", f"unclosed think <= {PROTECTED_UNCLOSED_THINK_MAX}", "unclosed_think_rate",
         lambda v: v <= PROTECTED_UNCLOSED_THINK_MAX, "0.000"),
    )
    for threshold, name, key, predicate, parent_value in diagnostics:
        value = metadata.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            verdict.add(threshold, name, UNKNOWN, f"{key} missing from the instrument run")
        else:
            verdict.add(
                threshold,
                name,
                PASS if predicate(value) else FAIL,
                f"measured {value}; parent {parent_value}",
            )


def _production_scoring() -> Any:
    """Production's declared scoring module, imported where T24 needs it.

    Reached through this one function so an unavailable readout is T24's UNKNOWN
    rather than a judge that cannot run, and so a test can pin that branch.
    """
    from chowder.evaluators import scoring

    return scoring


def _readout_disclosure_gate(verdict: Verdict, candidate: Arm | None) -> None:
    """T24 (amendment 17): T5's presence readout, reconciled with production's.

    T5 counts an item as an answer when the expected string appears anywhere on
    the answer surface -- the presence the prereg declares for that row. The label
    used to call it "answer correctness", and correctness is a readout production
    owns: ``evaluators.scoring`` declares the mode, extracts the answer after the
    last ``</think>`` and refuses an unclosed reasoning budget. The two readouts
    disagree on the same completion whenever a mention carries the expected
    string: "There are 17 continents." against "7" is present, and 0.0 under
    every readout production declares.

    This gate re-scores nothing and moves no threshold. It reads the rows T5
    already reads, runs both declared production readouts over each completion
    through production's own ``score``, and refuses only the irreducible case --
    an item presence accepts that *every* declared readout refuses. The reverse
    direction (a presence miss production accepts) can only refuse more, so it is
    disclosed in the detail and does not gate. A candidate whose completions
    agree, and a candidate arm with no per-prompt evidence, keep the verdict they
    had before.
    """
    name = f"answer-readout disagreements <= {PROTECTED_READOUT_DISAGREEMENT_MAX}"
    if candidate is None:
        verdict.add("T24", name, UNKNOWN, f"{READOUT_UNMEASURED}: no candidate arm")
        return
    per_prompt = candidate.per_prompt()
    if not per_prompt:
        verdict.add(
            "T24",
            name,
            UNKNOWN,
            f"{READOUT_UNMEASURED}: no candidate per-prompt evidence",
        )
        return
    try:
        scoring = _production_scoring()
    except Exception as exc:  # noqa: BLE001 - an unreadable readout is UNKNOWN, not a crash
        verdict.add(
            "T24",
            name,
            UNKNOWN,
            f"{READOUT_UNMEASURED}: production readouts unavailable ({type(exc).__name__})",
        )
        return

    disagreements: list[str] = []
    disclosed_misses = 0
    for entry in per_prompt:
        expected = str(entry.get("expected") or "")
        completion = str(entry.get("completion", ""))
        accepted = [
            mode
            for mode in PRODUCTION_READOUT_MODES
            if scoring.score(completion, expected, mode) > 0
        ]
        if not accepted:
            if _answer_present(entry):
                disagreements.append(_prompt_key(entry) or "(unnamed item)")
            continue
        if not _answer_present(entry):
            disclosed_misses += 1

    if disagreements:
        verdict.add(
            "T24",
            name,
            FAIL,
            f"{READOUT_DISAGREES_WITH_PRODUCTION}: {len(disagreements)} item(s) pass "
            f"presence while {' and '.join(PRODUCTION_READOUT_MODES)} refuse them: "
            f"{disagreements}; {disclosed_misses} presence miss(es) production "
            "accepts (disclosed)",
        )
        return
    verdict.add(
        "T24",
        name,
        PASS,
        f"presence agrees with every declared production readout on "
        f"{len(per_prompt)} item(s); {disclosed_misses} presence miss(es) production "
        "accepts (disclosed, not gated)",
    )


def _paired_target_gate(
    verdict: Verdict,
    threshold: str,
    label: str,
    *,
    parent_flags: Mapping[str, int] | None,
    candidate_flags: Mapping[str, int] | None,
    lower_is_better: bool,
    absolute_threshold: float,
) -> None:
    name = f"{label} target"
    if candidate_flags is None:
        verdict.add(threshold, name, UNKNOWN, "candidate prompt identities are ambiguous")
        return
    aligned = _aligned(parent_flags, candidate_flags)
    if aligned is None:
        verdict.add(
            threshold,
            name,
            UNKNOWN,
            "parent per-prompt evidence is missing or its prompt identities do not "
            "align with the candidate's, so the frozen paired rule cannot be applied",
        )
        return
    status, detail = _target_gate(
        *aligned, lower_is_better=lower_is_better, absolute_threshold=absolute_threshold
    )
    verdict.add(threshold, name, status, detail)


def _protected_gates(
    verdict: Verdict,
    arms: Mapping[str, Arm | None],
    campaign: CampaignManifest | None,
    *,
    run_root: Path,
) -> None:
    required = tuple(campaign.protected_benchmarks) if campaign else REQUIRED_PROTECTED
    candidate = arms["candidate"]
    parent = arms["parent"]
    ancestor = arms["ancestor"]

    if candidate is None:
        verdict.add("T11", "protected mini-slices", UNKNOWN, "candidate arm unavailable")
    else:
        # The candidate side is the gate: the required slices must exist, once
        # each, candidate-measured, and protocol-exact. The parent and ancestor
        # sides are compared separately below (T16/T17) so that an unavailable
        # parent arm is a specific, adjudicable state rather than a blanket
        # refusal of every protected row.
        for qualified_id in required:
            status, run, detail = _slice_status(
                candidate, qualified_id, label="candidate", run_root=run_root,
                protocol=JUDGE_PROTOCOL
            )
            verdict.add("T11", f"candidate {qualified_id} measured + protocol-exact", status, detail)
            if status != PASS or parent is None:
                continue
            parent_status, parent_run, parent_detail = _slice_status(
                parent, qualified_id, label="parent", run_root=run_root,
                protocol=JUDGE_PROTOCOL
            )
            if parent_status != PASS:
                continue
            delta = run.score - parent_run.score
            verdict.add(
                "T11",
                f"{qualified_id} candidate-vs-parent regression <= {SLICE_REGRESSION_MAX}",
                PASS if delta >= -SLICE_REGRESSION_MAX else FAIL,
                f"candidate {run.score:.4f} vs parent {parent_run.score:.4f} (delta {delta:+.4f})",
            )

    if campaign is not None:
        # Declared *anywhere* in the campaign: a target or broad row is a declared
        # measurement, not a substitute for a required protected slice. What must
        # not happen is an undeclared benchmark standing in for one.
        declared_elsewhere = {
            qualified_id
            for declared_set in (
                campaign.target_benchmarks,
                campaign.protected_benchmarks,
                campaign.broad_benchmarks,
                campaign.calibration_benchmarks,
                campaign.reliability_benchmarks,
            )
            for qualified_id in declared_set
        }
        undeclared = sorted(
            {
                run.benchmark_qualified_id
                for run in (candidate.report.runs if candidate else ())
                if run.benchmark_qualified_id != INSTRUMENT_ID
                and run.benchmark_qualified_id not in declared_elsewhere
            }
        )
        verdict.add(
            "T11",
            "no undeclared protected row substitutes for a required slice",
            PASS if not undeclared else FAIL,
            f"undeclared protected rows: {undeclared}" if undeclared else "none",
        )

    _ancestor_gates(
        verdict, required, candidate=candidate, parent=parent, ancestor=ancestor,
        run_root=run_root,
    )


def _ancestor_gates(
    verdict: Verdict,
    required: Sequence[str],
    *,
    candidate: Arm | None,
    parent: Arm | None,
    ancestor: Arm | None,
    run_root: Path,
) -> None:
    """Protection against the last *trusted* generation, not just the parent.

    An INCONCLUSIVE ancestor must not become a trusted protection baseline: the
    trusted arm is the frozen gen0, and a candidate that only matches an
    already-regressed parent fails here.
    """
    if candidate is None:
        verdict.add(
            "T16", f"trusted-ancestor protection (vs {TRUSTED_ANCESTOR_VERSION})",
            UNKNOWN, "candidate arm unavailable",
        )
        verdict.add("T17", "immediate-parent regression", UNKNOWN, "candidate arm unavailable")
        return
    if ancestor is None:
        verdict.add(
            "T16", f"trusted-ancestor protection (vs {TRUSTED_ANCESTOR_VERSION})",
            UNKNOWN,
            "baseline_evaluation.json (trusted ancestor arm) is missing, so branch "
            "protection cannot be established",
        )
    else:
        details: list[str] = []
        failed = False
        unresolved = False
        for qualified_id in required:
            candidate_status, candidate_run, candidate_detail = _slice_status(
                candidate, qualified_id, label="candidate", run_root=run_root,
                protocol=JUDGE_PROTOCOL
            )
            ancestor_status, ancestor_run, ancestor_detail = _slice_status(
                ancestor, qualified_id, label="ancestor", run_root=run_root,
                protocol=JUDGE_PROTOCOL
            )
            if candidate_status != PASS or ancestor_status != PASS:
                unresolved = True
                details.append(f"{qualified_id}: {candidate_detail}; {ancestor_detail}")
                continue
            delta = candidate_run.score - ancestor_run.score
            details.append(
                f"{qualified_id}: gen2 {candidate_run.score:.4f} vs gen0 "
                f"{ancestor_run.score:.4f} (delta {delta:+.4f})"
            )
            if delta < -SLICE_REGRESSION_MAX:
                failed = True
        if failed:
            verdict.add(
                "T16", f"trusted-ancestor protection (vs {TRUSTED_ANCESTOR_VERSION})",
                FAIL, "; ".join(details),
            )
        elif unresolved:
            verdict.add(
                "T16", f"trusted-ancestor protection (vs {TRUSTED_ANCESTOR_VERSION})",
                UNKNOWN, "; ".join(details),
            )
        else:
            verdict.add(
                "T16", f"trusted-ancestor protection (vs {TRUSTED_ANCESTOR_VERSION})",
                PASS, "; ".join(details),
            )

    trusted_rows = [row for row in verdict.thresholds() if row[0] == "T16"]
    trusted_ok = bool(trusted_rows) and trusted_rows[0][2] == PASS
    parent_slice_evidence = parent is not None and any(
        _slice_status(parent, qualified_id, label="parent", run_root=run_root,
                protocol=JUDGE_PROTOCOL)[0] == PASS
        for qualified_id in required
    )
    if not parent_slice_evidence:
        # The immediate-parent protected evidence is unavailable (arm absent,
        # or it carries no usable slices). Promotion stays possible only when
        # the trusted-ancestor arm independently resolved the risk -- an
        # unresolved ancestor must not become a trusted baseline by default.
        verdict.add(
            "T17",
            "immediate-parent (gen1) protected regression",
            PASS if trusted_ok else UNKNOWN,
            (
                "gen1 protected evidence unavailable; resolved by the trusted-ancestor "
                "arm, which found no regression against gen0"
                if trusted_ok
                else "gen1 protected evidence unavailable and the trusted-ancestor arm "
                "did not resolve it"
            ),
        )
        if parent is not None:
            verdict.add(
                INFO,
                "gen1 protected evidence",
                INFO,
                "the gen1 arm carries no usable protected mini-slice measurement",
            )
        return

    details = []
    failed = False
    unresolved = False
    parent_branch_broken = False
    for qualified_id in required:
        candidate_status, candidate_run, candidate_detail = _slice_status(
            candidate, qualified_id, label="candidate", run_root=run_root,
                protocol=JUDGE_PROTOCOL
        )
        parent_status, parent_run, parent_detail = _slice_status(
            parent, qualified_id, label="parent", run_root=run_root,
                protocol=JUDGE_PROTOCOL
        )
        if candidate_status != PASS or parent_status != PASS:
            unresolved = True
            details.append(f"{qualified_id}: {candidate_detail}; {parent_detail}")
            continue
        delta = candidate_run.score - parent_run.score
        details.append(
            f"{qualified_id}: gen2 {candidate_run.score:.4f} vs gen1 "
            f"{parent_run.score:.4f} (delta {delta:+.4f})"
        )
        if delta < -SLICE_REGRESSION_MAX:
            failed = True
        ancestor_run = ancestor.run_for(qualified_id) if ancestor is not None else None
        if ancestor_run is not None and parent_run.score < ancestor_run.score - SLICE_REGRESSION_MAX:
            parent_branch_broken = True
    verdict.add(
        "T17",
        "immediate-parent (gen1) protected regression",
        FAIL if failed else (UNKNOWN if unresolved else PASS),
        "; ".join(details),
    )
    if parent_branch_broken:
        verdict.add(
            INFO,
            "gen1 branch state",
            INFO,
            "gen1 itself regressed against the trusted ancestor on at least one slice; "
            "the gen2 verdict is adjudicated against gen0 for that reason",
        )


def _contamination_gate(
    verdict: Verdict, run_root: Path, campaign: CampaignManifest | None
) -> None:
    """Contamination evidence is the artifact the campaign *pinned*.

    The judged evidence is ``contamination_manifest_path``, not whatever file
    happens to sit in the run root. A manifest that declines to pin one, a pin
    that is not there, and a run-root copy that disagrees with the pin are all
    refusals (T12/T18): an unpinned artifact cannot certify a release, and a
    campaign that pins known contamination must not be certified by a clean file
    that merely shares the run root's naming.
    """
    threshold = "T12"
    name = "contamination CLEAN on the frozen evaluated set"
    pin_row = "judged contamination evidence is the pinned artifact"
    declared = str(campaign.contamination_manifest_path) if campaign is not None else ""
    if not declared:
        detail = (
            f"{CONTAMINATION_PIN_ABSENT}: the campaign declaration pins no "
            "contamination_manifest_path, so the judged contamination evidence "
            "cannot be identified"
        )
        verdict.add(threshold, name, UNKNOWN, detail)
        verdict.add("T18", pin_row, UNKNOWN, detail)
        verdict.add(
            threshold, "training-source contamination CLEAN", UNKNOWN,
            "no pinned contamination manifest to read, so no curriculum was proven non-leaking",
        )
        return
    pin = Path(declared)
    if not pin.is_absolute():
        pin = run_root / pin
    if not pin.is_file():
        detail = (
            f"{CONTAMINATION_PIN_MISSING}: the campaign pins "
            f"{str(pin)!r}, which does not exist, so the contamination evidence it "
            "declares cannot be verified"
        )
        verdict.add(threshold, name, UNKNOWN, detail)
        verdict.add("T18", pin_row, UNKNOWN, detail)
        verdict.add(
            threshold, "training-source contamination CLEAN", UNKNOWN,
            "the pinned contamination manifest is missing, so no curriculum was proven non-leaking",
        )
        return
    document = _load_json(pin)
    if not isinstance(document, Mapping):
        detail = (
            f"{CONTAMINATION_PIN_MISSING}: the pinned artifact {str(pin)!r} is not a "
            "JSON object, so it carries no contamination verdicts"
        )
        verdict.add(threshold, name, UNKNOWN, detail)
        verdict.add("T18", pin_row, UNKNOWN, detail)
        verdict.add(
            threshold, "training-source contamination CLEAN", UNKNOWN,
            "the pinned contamination manifest is unreadable, so no curriculum was proven non-leaking",
        )
        return

    # The run root must carry that very evidence, not a different file with the
    # same name: this is what makes the judged root and the declared evidence the
    # same object.
    carried = run_root / "gen2_contamination_manifest.json"
    if not carried.is_file():
        detail = (
            f"{CONTAMINATION_EVIDENCE_NOT_IN_RUN_ROOT}: the run root carries no "
            "gen2_contamination_manifest.json, so the judged evidence set is "
            "incomplete"
        )
        verdict.add(threshold, name, UNKNOWN, detail)
        verdict.add("T18", pin_row, UNKNOWN, detail)
        verdict.add(
            threshold, "training-source contamination CLEAN", UNKNOWN,
            "the pinned contamination evidence is absent from the run root",
        )
        return
    if sha256_file(carried) != sha256_file(pin):
        detail = (
            f"{CONTAMINATION_EVIDENCE_NOT_PINNED}: the run root's "
            f"gen2_contamination_manifest.json is not the artifact the campaign "
            f"pinned ({str(pin)!r}): the two documents differ"
        )
        verdict.add("T18", pin_row, FAIL, detail)
        verdict.add(threshold, name, FAIL, detail)
        verdict.add(
            threshold, "training-source contamination CLEAN", FAIL,
            "the judged contamination evidence is not the pinned evidence",
        )
        return
    verdict.add(
        "T18", pin_row, PASS,
        f"the judged contamination evidence is {str(pin)!r}, byte-identical to the pin",
    )

    section = document.get("benchmarks", {})
    # Production owns the interpretation of absence: an unlisted benchmark
    # binds UNKNOWN, never clean.
    binder = MetricBinder(default_registry(), contamination=section if isinstance(section, Mapping) else {})
    required = [INSTRUMENT_ID, *REQUIRED_PROTECTED]
    if campaign is not None:
        required = list(dict.fromkeys([*campaign.target_benchmarks, *campaign.protected_benchmarks]))
    statuses = {qid: binder.contamination_status(qid) for qid in required}
    flagged = {qid: status for qid, status in statuses.items() if status in CONTAMINATED}
    unknown = {qid: status for qid, status in statuses.items() if status not in CONTAMINATED and status != "CLEAN"}
    if flagged:
        verdict.add(
            "T12", "contamination CLEAN on the frozen evaluated set", FAIL, f"flagged: {flagged}"
        )
    elif unknown:
        verdict.add(
            "T12", "contamination CLEAN on the frozen evaluated set", UNKNOWN,
            f"not CLEAN for: {unknown}",
        )
    else:
        verdict.add(
            "T12", "contamination CLEAN on the frozen evaluated set", PASS,
            f"all required benchmarks CLEAN: {sorted(statuses)}",
        )

    # Training sources are a declared hard prerequisite (prereg section 3): the
    # curriculum must be proven non-leaking, so an empty section is UNKNOWN.
    sources = document.get("training_sources", {})
    if not isinstance(sources, Mapping) or not sources:
        verdict.add(
            "T12", "training-source contamination CLEAN", UNKNOWN,
            "the contamination manifest declares no training sources, so no "
            "curriculum was proven non-leaking",
        )
        return
    bad = {
        str(source): (entry.get("status") if isinstance(entry, Mapping) else None)
        for source, entry in sources.items()
        if not isinstance(entry, Mapping) or str(entry.get("status", "")).upper() != "CLEAN"
    }
    verdict.add(
        "T12",
        "training-source contamination CLEAN",
        FAIL if bad else PASS,
        f"non-clean training sources: {bad}" if bad else f"{len(sources)} sources CLEAN",
    )


def _settlement_gates(
    verdict: Verdict, run_root: Path, campaign: CampaignManifest | None
) -> None:
    path = run_root / "cycle_compute_accounting.json"
    document = _load_json(path)
    if not isinstance(document, Mapping):
        verdict.add(
            "T13", "cost settles within the declared ceilings", UNKNOWN,
            "cycle_compute_accounting.json missing or unreadable",
        )
        verdict.add("T14", "all recipes accounted", UNKNOWN, "accounting artifact unavailable")
        return
    totals = (document.get("totals") or {}).get("incremental")
    if not isinstance(totals, Mapping):
        verdict.add(
            "T13", "cost settles within the declared ceilings", UNKNOWN,
            "accounting artifact declares no incremental totals",
        )
    elif campaign is None:
        verdict.add(
            "T13", "cost settles within the declared ceilings", UNKNOWN,
            "campaign manifest unavailable, so no ceilings can be settled against",
        )
    else:
        try:
            total = ComputeCost.from_dict(totals)
        except (KeyError, TypeError, ValueError) as error:
            verdict.add(
                "T13", "cost settles within the declared ceilings", UNKNOWN,
                f"incremental totals are not a readable ComputeCost: {error}",
            )
        else:
            # The production settlement, verbatim: the judge and
            # `chowder growth campaign settle` answer this the same way.
            settlement = settle_campaign(campaign, total=total)
            verdict.add(
                "T13",
                "cost settles within the declared ceilings",
                PASS if settlement.compliant else FAIL,
                (
                    f"device {total.device_gpu_hours:.4f} "
                    f"({'measured' if total.device_measured else 'unmeasured'}) / "
                    f"wall {total.wall_gpu_hours:.4f} against the campaign envelope"
                    + (
                        ""
                        if settlement.compliant
                        else "; " + "; ".join(settlement.failure_reasons)
                    )
                ),
            )

    entries = document.get("entries") or []
    recipe_ids = {
        entry.get("recipe_id")
        for entry in entries
        if isinstance(entry, Mapping)
        and entry.get("kind") in {"training", "evaluation", "failed_attempt"}
        and entry.get("recipe_id")
    }
    # The declared recipe set, not merely "two recipes": a campaign that accounted
    # for a different set than it declared has not accounted for its own run.
    declared = tuple(campaign.recipe_ids) if campaign is not None else ()
    if not declared:
        verdict.add(
            "T14",
            "all recipes accounted",
            PASS if len(recipe_ids) >= REQUIRED_RECIPES_MIN else FAIL,
            f"recipe ids in accounting: {sorted(recipe_ids)} (no declared set to compare)",
        )
        return
    missing = sorted(set(declared) - recipe_ids)
    extra = sorted(recipe_ids - set(declared))
    verdict.add(
        "T14",
        "all recipes accounted",
        PASS if not missing and not extra else FAIL,
        (
            f"accounted {sorted(recipe_ids)} against the declared {sorted(declared)}"
            if not missing and not extra
            else f"declared but unaccounted: {missing}; accounted but undeclared: {extra}"
        ),
    )


def _settlement_artifact_gate(
    verdict: Verdict, run_root: Path, campaign: CampaignManifest | None
) -> None:
    """The run's recorded settlement must be the settlement of the artifact it pinned.

    T13 settles the accounting artifact against the declared ceilings, from
    whatever bytes are in the run root. This gate closes the other half: that
    the artifact *is* the one the run settled. ``CycleCostLedger.write``
    (``chowder.growth.compute_cost``) stamps a digest over the document it
    writes and ``CampaignRun.to_dict`` pins that digest in
    ``cost.accounting_digest``, so an edit to the artifact -- a total, an entry,
    a measurement flag -- moves a number the record already carries. Without
    this gate a run the branch refused on its frozen envelope could be
    certified PROMOTED by editing one file: the refusal lives in the record,
    and T13 would recompute the settlement from the edited bytes and find them
    compliant.

    Fail-closed in every branch: an absent record, an absent pin, an absent
    recorded settlement, an unreadable artifact or an unreadable campaign is
    UNKNOWN (INCONCLUSIVE), and any disagreement between the record and the
    artifact's own settlement is a FAIL. Nothing is reimplemented: the digest
    is production's ``ledger_digest`` and the settlement is production's
    ``settle_campaign`` -- the same two functions the run itself called.
    """
    requirement = (
        "the run's recorded settlement is the settlement of the artifact it pinned"
    )
    record = _load_json(run_root / RUN_RECORD_NAME)
    if record is None:
        verdict.add(
            "T23", requirement, UNKNOWN,
            f"{RUN_RECORD_ABSENT}: {run_root / RUN_RECORD_NAME} is absent or "
            "unreadable, so the artifact it settled cannot be identified",
        )
        return
    document = _load_json(run_root / "cycle_compute_accounting.json")
    if not isinstance(document, Mapping):
        verdict.add(
            "T23", requirement, UNKNOWN,
            "cycle_compute_accounting.json missing or unreadable, so there is no "
            "artifact to settle",
        )
        return
    pinned = str((record.get("cost") or {}).get("accounting_digest") or "")
    if not pinned:
        verdict.add(
            "T23", requirement, UNKNOWN,
            f"{ACCOUNTING_UNPINNED}: the run recorded no accounting digest at all "
            "(a run refused before settlement pins none), so no artifact can be "
            "settled here",
        )
        return
    recomputed = ledger_digest(document)
    if recomputed != pinned:
        verdict.add(
            "T23", requirement, FAIL,
            f"{ACCOUNTING_ARTIFACT_MOVED}: the accounting artifact hashes to "
            f"{recomputed[:12]}, but the run pinned {pinned[:12]}; these are not the "
            "bytes the run settled, so no settlement may be read out of them",
        )
        return
    recorded = record.get("settlement")
    if not isinstance(recorded, Mapping) or "budget_compliant" not in recorded:
        verdict.add(
            "T23", requirement, UNKNOWN,
            f"{ACCOUNTING_UNSETTLED_BY_RUN}: the run pinned this artifact but "
            "recorded no settlement verdict for it",
        )
        return
    if campaign is None:
        verdict.add(
            "T23", requirement, UNKNOWN,
            "the campaign manifest is unreadable, so the recorded settlement cannot "
            "be recomputed",
        )
        return
    totals = (document.get("totals") or {}).get("incremental")
    try:
        total = ComputeCost.from_dict(totals)
    except (KeyError, TypeError, ValueError) as error:
        verdict.add(
            "T23", requirement, UNKNOWN,
            f"incremental totals are not a readable ComputeCost: {error}",
        )
        return
    settlement = settle_campaign(campaign, total=total)
    if bool(recorded["budget_compliant"]) != settlement.compliant:
        verdict.add(
            "T23", requirement, FAIL,
            f"{SETTLEMENT_DISAGREES_WITH_RECORD}: the run recorded "
            f"budget_compliant={recorded['budget_compliant']!r} while the artifact "
            "it pinned settles "
            + (
                "compliant"
                if settlement.compliant
                else "non-compliant; " + "; ".join(settlement.failure_reasons)
            ),
        )
        return
    verdict.add(
        "T23", requirement, PASS,
        f"the artifact is the one the run pinned ({pinned[:12]}) and it settles as "
        f"the record says ({'compliant' if settlement.compliant else 'non-compliant'})",
    )


def _protection_agreement_gate(verdict: Verdict, campaign: CampaignManifest | None) -> None:
    """The campaign declaration and the frozen judge must state the same policy.

    The runner certifies a run with the manifest's declared protection (ancestor,
    tolerance, protocol); the judge audits with its frozen constants. If the two
    ever disagree, one of them is enforcing a policy the other does not know
    about -- so the disagreement is a hard failure here rather than a silent
    difference of opinion between the run and its audit.
    """
    requirement = "the campaign declares the policy this judge enforces"
    if campaign is None:
        verdict.add(
            "T20", requirement, UNKNOWN,
            "the campaign manifest is unreadable, so its declared policy cannot be "
            "compared with the frozen one",
        )
        return
    protection = campaign.protection
    if (
        protection.protocol is None
        or not protection.trusted_ancestor_version
        or protection.slice_regression_max is None
    ):
        verdict.add(
            "T20", requirement, UNKNOWN,
            "the campaign declares no protection policy (trusted ancestor, "
            "tolerance, protocol), so the run cannot certify what this judge audits",
        )
        return
    declared = (
        protection.trusted_ancestor_version,
        float(protection.slice_regression_max),
        protection.protocol.to_dict(),
    )
    frozen = (
        TRUSTED_ANCESTOR_VERSION,
        float(SLICE_REGRESSION_MAX),
        JUDGE_PROTOCOL.to_dict(),
    )
    verdict.add(
        "T20",
        requirement,
        PASS if declared == frozen else FAIL,
        (
            f"declared {declared} == frozen {frozen}"
            if declared == frozen
            else f"the campaign declares {declared}, this judge enforces {frozen}"
        ),
    )


def _declared_gate_agreement(
    verdict: Verdict,
    arms: Mapping[str, "Arm | None"],
    campaign: CampaignManifest | None,
    *,
    run_root: Path,
) -> None:
    """The run's recorded decision and the declared retention profile must agree.

    The run enforces ``retention_profile`` on both promotion paths
    (``GrowthCycle._apply_promotion_gates``), and records what it decided in
    ``campaign-run.json`` beside the arms this judge reads. Before amendment 15
    this judge never opened that record and never read a declared profile, so a
    candidate the run had already refused could reach a table in which every
    audited gate passed -- a certification of something the branch rejected.
    Two rows close that, and both fail closed:

    * the *recorded* decision: a run refused on a declared gate is FAIL here,
      because this judge audits no declared gate and its own table cannot
      overturn that refusal; a record that is absent, belongs to another cycle,
      or carries no decision is UNKNOWN;
    * the *recomputed* decision: the declared profile is evaluated again here,
      through production's own evaluator and production's own provenance filter
      (``evaluate_retention`` + ``retention_values``, the same two functions the
      promotion path calls), on the arms this judge already audited. A
      disagreement with the run's recorded breaches is FAIL. Nothing is
      reimplemented: the policy, the threshold and the rows that count all come
      from production, and this judge only compares the two answers.
    """
    requirement = "run decision on the declared retention profile"
    profile = campaign.retention_profile if campaign is not None else None
    record = _load_json(run_root / RUN_RECORD_NAME)
    if record is None:
        verdict.add(
            "T21", requirement, UNKNOWN,
            f"{RUN_RECORD_ABSENT}: {run_root / RUN_RECORD_NAME} is absent or "
            "unreadable, so the run's own decision cannot be audited here",
        )
        return
    if campaign is not None and str(record.get("cycle_id", "")) != str(campaign.cycle_id):
        verdict.add(
            "T21", requirement, UNKNOWN,
            f"{RUN_RECORD_WRONG_CYCLE}: the record in this run root belongs to "
            f"cycle {record.get('cycle_id')!r}, not {campaign.cycle_id!r}",
        )
        return
    decision = (record.get("promotion") or {}).get("decision") or {}
    if not decision:
        verdict.add(
            "T21", requirement, UNKNOWN,
            f"{RUN_DECISION_ABSENT}: the run record carries no promotion decision "
            "(a run refused before adjudication records none)",
        )
        return

    run_verdict = str(record.get("verdict", ""))
    reasons = [str(reason) for reason in decision.get("reasons", ()) or ()]
    recorded = {
        (reason.split(":", 1)[0].strip(), reason)
        for reason in reasons
        if reason.startswith(RETENTION_REASON_PREFIX)
    }
    breach_list = [reason for _code, reason in sorted(recorded)]
    recomputed = _recomputed_retention(arms, profile)

    # T21: the recorded decision. Decided once, recorded once, and never as a
    # silent skip -- every branch below states what it found.
    if recorded and profile is None:
        verdict.add(
            "T21", requirement, FAIL,
            f"{UNDECLARED_GATE_IN_RUN}: the run recorded declared-gate breaches "
            f"{breach_list} while the declaration names no retention profile",
        )
    elif run_verdict == "PROMOTED" and recorded:
        verdict.add(
            "T21", requirement, FAIL,
            f"{UNDECLARED_GATE_IN_RUN}: the run recorded PROMOTED together with "
            f"declared-gate breaches {breach_list}",
        )
    elif run_verdict in {"REJECTED", "TAINTED"} and recorded:
        verdict.add(
            "T21", requirement, FAIL,
            f"{DECLARED_GATE_REJECTED_RUN}: the run refused this candidate on the "
            f"declared gate(s) {breach_list}; this judge audits no declared gate, "
            "so its table cannot overturn that refusal",
        )
    elif run_verdict in {"REJECTED", "TAINTED"}:
        # Refused on the predeclared rule alone: T11/T16/T17 audit that rule
        # against the arms, and the run's own reasons belong in the record here
        # so a reader sees which refusal is being certified.
        verdict.add(
            "T21", requirement, PASS,
            "the run refused this candidate without a declared-gate breach: "
            f"{'; '.join(reasons) or 'no reasons recorded'}",
        )
    else:
        verdict.add(
            "T21", requirement, PASS,
            f"run verdict {run_verdict or 'unrecorded'}; declared-gate breaches "
            f"{breach_list or 'none'}",
        )

    # T22: the recomputation, always reported -- whether the run was refused or
    # not, a reader needs to know whether the two authorities agreed.
    recomputed_codes = sorted({code for code, _reason in recomputed})
    recorded_codes = sorted({code for code, _reason in recorded})
    if recomputed_codes == recorded_codes:
        verdict.add(
            "T22", "the judge recomputes the declared profile as the run did",
            PASS,
            f"recorded {recorded_codes or 'no breach'} == recomputed through "
            "production's evaluator "
            f"({profile.profile_id if profile is not None else 'no profile'})",
        )
        return
    verdict.add(
        "T22", "the judge recomputes the declared profile as the run did",
        FAIL,
        f"{RETENTION_RECOMPUTATION_DISAGREES}: the run recorded "
        f"{recorded_codes or 'no breach'} and this judge's recomputation through "
        f"production's evaluator gives {recomputed_codes or 'no breach'}; the two "
        "authorities disagree about the same candidate and neither may be assumed right",
    )


def _recomputed_retention(
    arms: Mapping[str, "Arm | None"],
    profile: Any,
) -> set[tuple[str, str]]:
    """``{(code, reason)}`` this judge computes for the declared profile.

    The arms are read through ``Arm.run_for``, so a row only counts when
    production has already accepted its provenance and generation, and the
    provenance filter that decides which rows may anchor a constraint is the
    promotion path's own ``retention_values``. With no declared profile the
    recomputation is empty, which is not a pass -- T21 refuses that case above.
    """
    if profile is None:
        return set()
    candidate = arms.get("candidate")
    parent = arms.get("parent")

    def results(arm: "Arm | None") -> dict[str, BenchmarkResult]:
        rows: dict[str, BenchmarkResult] = {}
        if arm is None:
            return rows
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

    violations = evaluate_retention(
        profile,
        parent_values=retention_values(
            profile, results(parent), candidate_side=False
        ),
        candidate_values=retention_values(
            profile, results(candidate), candidate_side=True
        ),
    )
    return {(violation.code, violation.reason) for violation in violations}


def _evidence_identity_gate(
    verdict: Verdict,
    run_root: Path,
    arms: Mapping[str, Arm | None],
    campaign: CampaignManifest | None,
) -> None:
    """Each arm's report must name the bytes it measured -- and those bytes the
    campaign's.

    This is what makes ``candidate_evaluation.json`` and ``chosen_candidate.json``
    one truth: the candidate arm must name the digest of the artifact the campaign
    selected, the parent arm the declared parent adapter, and the ancestor arm the
    declared dense base. A report that names nothing is UNDECIDED; a report that
    names a different model is a hard failure, because it is evidence about some
    other model wearing this generation's label.
    """
    # Report-level generation identity first, decided by the same production
    # mechanism the runner's own pre-ledger certification applies
    # (``MeasuredArm.identity_problems``). A protocol-correct row set inside a
    # report labelled for another generation is not this arm: the ancestor arm
    # answers "did the branch regress against the trusted ancestor", and reading a
    # gen1 report as gen0 would move that comparison to the immediate parent.
    for role in ("candidate", "parent", "ancestor"):
        arm = arms.get(role)
        if arm is None:
            continue
        generation_problems = tuple(
            problem
            for problem in arm.identity_problems()
            if ARM_GENERATION_MISMATCH in problem
        )
        verdict.add(
            "T19",
            f"{role} report is labelled for the generation it claims",
            PASS if not generation_problems else FAIL,
            "; ".join(generation_problems)
            or (
                f"{role} report carries generation "
                f"{arm.report.generation_version!r}"
            ),
        )

    chosen = _load_json(run_root / "chosen_candidate.json")
    selected = str(chosen.get("artifact_sha256", "")) if isinstance(chosen, Mapping) else ""
    expectations: list[tuple[str, str, str, str]] = []
    if selected:
        expectations.append(("candidate", selected, "adapter_digest", "the selected candidate artifact"))
    if campaign is not None and campaign.has_parent_adapter() and campaign.parent_adapter_digest:
        expectations.append(
            ("parent", campaign.parent_adapter_digest, "adapter_digest", "the declared parent adapter")
        )
    if campaign is not None and campaign.base_model_digest:
        expectations.append(
            ("ancestor", campaign.base_model_digest, "base_model_digest", "the declared dense base")
        )
    if not expectations:
        verdict.add(
            "T19", "evaluation evidence names the bytes it measured", UNKNOWN,
            "no campaign declaration names the artifacts the arms must be bound to",
        )
        return
    for role, expected, key, what in expectations:
        requirement = f"{role} evidence names {what}"
        arm = arms.get(role)
        if arm is None:
            verdict.add("T19", requirement, UNKNOWN, f"{role} arm unavailable")
            continue
        identity = arm.report.model_identity or {}
        declared_digest = str(identity.get(key, "")) if isinstance(identity, Mapping) else ""
        if not declared_digest:
            verdict.add(
                "T19", requirement, UNKNOWN,
                (
                    f"{ARM_ADAPTER_DIGEST_MISSING if key == 'adapter_digest' else ARM_BASE_DIGEST_MISSING}: "
                    f"the {role} report declares no {key}, so it is not bound to the "
                    "bytes it measured"
                ),
            )
        elif declared_digest != expected:
            verdict.add(
                "T19", requirement, FAIL,
                (
                    f"{ARM_ADAPTER_DIGEST_MISMATCH if key == 'adapter_digest' else ARM_BASE_DIGEST_MISMATCH}: "
                    f"the {role} report measured {declared_digest}, the campaign "
                    f"declares {expected} for {what}"
                ),
            )
        else:
            verdict.add(
                "T19", requirement, PASS,
                f"{role} measured {what} (digest {declared_digest[:12]})",
            )


def _identity_gate(verdict: Verdict, run_root: Path) -> None:
    chosen = _load_json(run_root / "chosen_candidate.json")
    if not isinstance(chosen, Mapping):
        verdict.add(
            "T15", "candidate artifact identity", UNKNOWN,
            "chosen_candidate.json missing or unreadable",
        )
        return
    ref = chosen.get("artifact_ref")
    recorded = chosen.get("artifact_sha256")
    if not isinstance(ref, str) or not ref.strip():
        verdict.add("T15", "candidate artifact identity", FAIL, "artifact_ref missing")
        return
    artifact = Path(ref)
    if not artifact.is_absolute():
        artifact = run_root / artifact
    if not artifact.exists():
        verdict.add(
            "T15", "candidate artifact identity", FAIL, f"artifact_ref {ref!r} does not exist"
        )
        return
    try:
        recomputed = _digest_of(artifact)
    except OSError as error:
        verdict.add(
            "T15", "candidate artifact identity", FAIL, f"hashing {ref!r} failed: {error}"
        )
        return
    if not isinstance(recorded, str) or len(recorded) != 64:
        verdict.add(
            "T15", "candidate artifact identity", FAIL,
            f"recorded digest is not a sha256 hex string: {recorded!r}",
        )
        return
    verdict.add(
        "T15",
        "candidate artifact identity (digest recomputed)",
        PASS if recomputed == recorded else FAIL,
        f"recorded {recorded[:12]} vs recomputed {recomputed[:12]}",
    )


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
