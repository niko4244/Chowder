#!/usr/bin/env python3
"""Frozen mechanical judge for the gen2 response-surface-compliance cycle.

Frozen with ``docs/quals/GEN2_PREREG_2026-09-17.md`` and its amendment
``docs/quals/GEN2_PREREG_AMENDMENT1_2026-09-18.md``. Thresholds may not
change after candidate results are visible. Reads the run's durable
artifacts read-only and emits one verdict table over the branch rules.

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

Usage:
    python docs/gen2/judge_gen2.py <run_root>

where ``<run_root>`` holds the three arm artifacts, plus
``cycle_compute_accounting.json``, ``gen2_contamination_manifest.json`` and
``chosen_candidate.json``.
"""

from __future__ import annotations

import json
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
from chowder.growth.catalog import default_registry  # noqa: E402
from chowder.growth.compute_cost import ComputeCost  # noqa: E402
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

#: Paired-decision constants for the target gates: the minimum effect on the
#: rate delta, and the strict-improvement count the absolute path requires.
TARGET_MIN_EFFECT = 0.25
TARGET_STRICT_PROMPT_MIN = 12

TARGET_DUPLICATION_MAX = 0.125  # parent 0.688
TARGET_ECHO_MAX = 0.062  # parent 0.438
TARGET_FORMAT_MIN = 8  # of the 8 declared constrained instrument prompts
PROTECTED_ANSWER_CORRECT_MIN = 15  # of 16
PROTECTED_EOS_MIN = 0.900
PROTECTED_CAP_MAX = 0.100
PROTECTED_UNCLOSED_THINK_MAX = 0.250
PROTECTED_LOOP_MAX = 0
PROTECTED_TRIGRAM_MIN = 0.900
SLICE_REGRESSION_MAX = 0.0625

REQUIRED_RECIPES_MIN = 2
CONTAMINATED = {"KNOWN_CONTAMINATION", "POSSIBLE"}

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


class _ArmError(Exception):
    """The arm artifact is missing, unreadable, or not usable as evidence."""


class Arm:
    """One measured generation's durable evidence, parsed by production."""

    def __init__(self, path: Path, *, expected_origin: str, label: str) -> None:
        self.label = label
        self.path = path
        if not path.is_file():
            raise _ArmError(f"{label} artifact {path.name} is missing")
        try:
            self.report = EvalReport.load(path)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise _ArmError(f"{label} artifact {path.name} is unreadable: {error}") from error
        self.expected_origin = expected_origin

    def run_for(self, qualified_id: str) -> BenchmarkRun | None:
        """The single run for a benchmark, or None when absent/duplicated."""
        matches = [
            run for run in self.report.runs if run.benchmark_qualified_id == qualified_id
        ]
        if len(matches) != 1:
            return None
        run = matches[0]
        if run.measurement_origin != self.expected_origin:
            return None
        return run

    def duplicate_ids(self) -> tuple[str, ...]:
        counts: dict[str, int] = {}
        for run in self.report.runs:
            counts[run.benchmark_qualified_id] = counts.get(run.benchmark_qualified_id, 0) + 1
        return tuple(sorted(qid for qid, n in counts.items() if n > 1))

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
    detail = (
        f"paired={comparison.verdict} (delta {comparison.delta:+.4f}, "
        f"min_effect {TARGET_MIN_EFFECT}); candidate rate {candidate_rate:.3f} vs "
        f"parent {parent_rate:.3f}; absolute {'met' if absolute_met else 'not met'}; "
        f"strictly better on {strict}/{len(candidate)} (need {TARGET_STRICT_PROMPT_MIN})"
    )
    if paired_improved:
        return PASS, f"paired improvement — {detail}"
    if absolute_met and strict_ok:
        return PASS, f"absolute threshold + strict count — {detail}"
    return FAIL, detail


def _protocol_problems(
    run: BenchmarkRun, *, field: str = "metadata"
) -> tuple[str, ...]:
    """Every way a protected slice fails to match the frozen protocol."""
    metadata = run.metadata or {}
    problems: list[str] = []
    if run.n_samples != PROTECTED_N_SAMPLES:
        problems.append(f"n_samples={run.n_samples} (expected {PROTECTED_N_SAMPLES})")
    indices = metadata.get("sample_indices")
    if not isinstance(indices, list) or tuple(indices) != PROTECTED_SAMPLE_INDICES:
        problems.append(f"sample_indices={indices!r} (expected 0..{PROTECTED_N_SAMPLES - 1})")
    if metadata.get("seed") != PROTECTED_SEED:
        problems.append(f"seed={metadata.get('seed')!r} (expected {PROTECTED_SEED})")
    if metadata.get("shuffle") is not PROTECTED_SHUFFLE:
        problems.append(f"shuffle={metadata.get('shuffle')!r} (expected {PROTECTED_SHUFFLE})")
    decoding = metadata.get("decoding")
    if not isinstance(decoding, Mapping):
        problems.append("decoding metadata missing")
    else:
        for key, expected in PROTECTED_DECODING.items():
            if decoding.get(key) != expected:
                problems.append(f"decoding.{key}={decoding.get(key)!r} (expected {expected!r})")
    if metadata.get("prompt_policy") != PROTECTED_PROMPT_POLICY:
        problems.append(
            f"prompt_policy={metadata.get('prompt_policy')!r} "
            f"(expected {PROTECTED_PROMPT_POLICY!r})"
        )
    if not isinstance(run.score, (int, float)) or isinstance(run.score, bool):
        problems.append("score is not a measured number")
    if not run.raw_artifact_ref:
        problems.append("raw_artifact_ref missing (no underlying evidence named)")
    return tuple(problems)


def _slice_status(
    arm: Arm | None,
    qualified_id: str,
    *,
    label: str,
) -> tuple[str, Any, str]:
    """Locate one required slice in one arm, with its provenance and protocol."""
    if arm is None:
        return UNKNOWN, None, f"{label} arm unavailable"
    duplicates = arm.duplicate_ids()
    if qualified_id in duplicates:
        return FAIL, None, f"{label} arm carries {qualified_id} more than once"
    run = arm.run_for(qualified_id)
    if run is None:
        present = arm.report.runs and any(
            r.benchmark_qualified_id == qualified_id for r in arm.report.runs
        )
        if present:
            return FAIL, None, (
                f"{label} arm carries {qualified_id} with provenance "
                f"{arm.expected_origin} refused"
            )
        return UNKNOWN, None, f"{label} arm has no {qualified_id} measurement"
    problems = _protocol_problems(run)
    if problems:
        return FAIL, run, f"{label} protocol mismatch: " + "; ".join(problems)
    return PASS, run, f"{label} {qualified_id} = {run.score:.4f} ({run.n_samples} items)"


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

    arms: dict[str, Arm | None] = {}
    for key, filename, origin, label in (
        ("candidate", "candidate_evaluation.json", MEASURED_THIS_GENERATION, "candidate"),
        ("parent", "parent_evaluation.json", MEASURED_PARENT, "parent (gen1)"),
        ("ancestor", "baseline_evaluation.json", MEASURED_PARENT, "trusted ancestor (gen0)"),
    ):
        try:
            arms[key] = Arm(run_root / filename, expected_origin=origin, label=label)
        except _ArmError as error:
            arms[key] = None
            if key == "candidate":
                verdict.add("T1", "candidate measured evidence", UNKNOWN, str(error))
    candidate = arms["candidate"]

    _instrument_gates(verdict, candidate, arms["parent"])
    _protected_gates(verdict, arms, campaign)
    _contamination_gate(verdict, run_root, campaign)
    _settlement_gates(verdict, run_root, campaign)
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
        "docs/quals/GEN2_PREREG_2026-09-17.md + GEN2_PREREG_AMENDMENT1_2026-09-18.md",
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
            ("T1", "candidate instrument provenance"),
            ("T2", "answer-duplication target"),
            ("T3", "template-echo target"),
            ("T4", "constrained-prompt format"),
            ("T5", "answer correctness"),
            ("T6", "EOS termination"),
            ("T7", "max-token-cap rate"),
            ("T8", "obvious loops"),
            ("T9", "distinct-trigram ratio"),
            ("T10", "unclosed think rate"),
        ):
            verdict.add(threshold, name, UNKNOWN, "candidate evaluation artifact unavailable")
        return

    duplicates = candidate.duplicate_ids()
    if duplicates:
        verdict.add(
            "T1",
            "candidate instrument provenance",
            FAIL,
            f"candidate arm duplicates rows for {list(duplicates)}",
        )
    instrument_run = candidate.run_for(INSTRUMENT_ID)
    if instrument_run is None:
        verdict.add(
            "T1",
            "candidate instrument provenance",
            UNKNOWN,
            f"no single {INSTRUMENT_ID} run carrying {MEASURED_THIS_GENERATION}",
        )
    else:
        verdict.add(
            "T1",
            "candidate instrument provenance",
            PASS,
            f"measurement_origin={instrument_run.measurement_origin}",
        )

    per_prompt = candidate.per_prompt()
    if instrument_run is None or not per_prompt:
        for threshold, name in (
            ("T2", "answer-duplication target"),
            ("T3", "template-echo target"),
            ("T4", "constrained-prompt format"),
            ("T5", "answer correctness"),
        ):
            verdict.add(threshold, name, UNKNOWN, "no candidate per-prompt evidence")
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

        correct = sum(
            1
            for entry in per_prompt
            if str(entry.get("expected") or "").lower()
            in _answer_surface(str(entry.get("completion", ""))).lower()
        )
        verdict.add(
            "T5",
            f"answer correctness >= {PROTECTED_ANSWER_CORRECT_MIN}/16",
            PASS if correct >= PROTECTED_ANSWER_CORRECT_MIN else FAIL,
            f"measured {correct}/{len(per_prompt)}; parent 16/16",
        )

    metadata = (instrument_run.metadata or {}) if instrument_run is not None else {}
    diagnostics = (
        ("T6", f"EOS termination >= {PROTECTED_EOS_MIN}", "eos_termination_rate",
         lambda v: v >= PROTECTED_EOS_MIN, "1.000"),
        ("T7", f"cap-hit < {PROTECTED_CAP_MAX}", "max_token_cap_rate",
         lambda v: v < PROTECTED_CAP_MAX, "0.000"),
        ("T8", f"obvious loops <= {PROTECTED_LOOP_MAX}", "obvious_loop_count",
         lambda v: v <= PROTECTED_LOOP_MAX, "0"),
        ("T9", f"distinct-trigram >= {PROTECTED_TRIGRAM_MIN}", "distinct_trigram_ratio_mean",
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
    verdict: Verdict, arms: Mapping[str, Arm | None], campaign: CampaignManifest | None
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
            status, run, detail = _slice_status(candidate, qualified_id, label="candidate")
            verdict.add("T11", f"candidate {qualified_id} measured + protocol-exact", status, detail)
            if status != PASS or parent is None:
                continue
            parent_status, parent_run, parent_detail = _slice_status(
                parent, qualified_id, label="parent"
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
        undeclared = sorted(
            {
                run.benchmark_qualified_id
                for run in (candidate.report.runs if candidate else ())
                if run.benchmark_qualified_id != INSTRUMENT_ID
                and run.benchmark_qualified_id not in required
            }
        )
        verdict.add(
            "T11",
            "no undeclared protected row substitutes for a required slice",
            PASS if not undeclared else FAIL,
            f"undeclared protected rows: {undeclared}" if undeclared else "none",
        )

    _ancestor_gates(verdict, required, candidate=candidate, parent=parent, ancestor=ancestor)


def _ancestor_gates(
    verdict: Verdict,
    required: Sequence[str],
    *,
    candidate: Arm | None,
    parent: Arm | None,
    ancestor: Arm | None,
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
                candidate, qualified_id, label="candidate"
            )
            ancestor_status, ancestor_run, ancestor_detail = _slice_status(
                ancestor, qualified_id, label="ancestor"
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
        _slice_status(parent, qualified_id, label="parent")[0] == PASS
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
            candidate, qualified_id, label="candidate"
        )
        parent_status, parent_run, parent_detail = _slice_status(
            parent, qualified_id, label="parent"
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
    path = run_root / "gen2_contamination_manifest.json"
    document = _load_json(path)
    if not isinstance(document, Mapping):
        verdict.add(
            "T12", "contamination CLEAN on the frozen evaluated set", UNKNOWN,
            "gen2_contamination_manifest.json missing or unreadable",
        )
        return

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
            "T13", "actual cost settled within the declared ceilings", UNKNOWN,
            "cycle_compute_accounting.json missing or unreadable",
        )
        verdict.add("T14", "all recipes accounted", UNKNOWN, "accounting artifact unavailable")
        return
    totals = (document.get("totals") or {}).get("incremental")
    if not isinstance(totals, Mapping):
        verdict.add(
            "T13", "actual cost settled within the declared ceilings", UNKNOWN,
            "accounting artifact declares no incremental totals",
        )
    elif campaign is None:
        verdict.add(
            "T13", "actual cost settled within the declared ceilings", UNKNOWN,
            "campaign manifest unavailable, so no ceilings can be settled against",
        )
    else:
        try:
            total = ComputeCost.from_dict(totals)
        except (KeyError, TypeError, ValueError) as error:
            verdict.add(
                "T13", "actual cost settled within the declared ceilings", UNKNOWN,
                f"incremental totals are not a readable ComputeCost: {error}",
            )
        else:
            # The production settlement, verbatim: the judge and
            # `chowder growth campaign settle` answer this the same way.
            settlement = settle_campaign(campaign, total=total)
            verdict.add(
                "T13",
                "actual cost settled within the declared ceilings",
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
    verdict.add(
        "T14",
        "all recipes accounted",
        PASS if len(recipe_ids) >= REQUIRED_RECIPES_MIN else FAIL,
        f"recipe ids in accounting: {sorted(recipe_ids)}",
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
