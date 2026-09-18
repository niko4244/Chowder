#!/usr/bin/env python3
"""Frozen mechanical judge for the gen2 response-surface-compliance cycle.

Frozen with ``docs/quals/GEN2_PREREG_2026-09-17.md`` — thresholds may not
change after candidate results are visible. Reads the run's durable
artifacts read-only and emits one verdict table; exit 0 only when every
gating threshold is PASS (UNKNOWN refuses to certify, per the quals
harness's epistemics).

The judge consumes candidate-measured evidence only: any row whose
``measurement_origin`` is not ``MEASURED_THIS_GENERATION`` is UNKNOWN for a
gate, never a pass. That is the integrity rule this judge inherits from the
2016-09-17 re-adjudication.

Usage:
    python docs/gen2/judge_gen2.py <run_root>

where ``<run_root>`` holds:
  candidate_evaluation.json        instrument + mini-slice measurements
  cycle_compute_accounting.json    settlement artifact
  gen2_contamination_manifest.json firewall verdicts
  chosen_candidate.json            selected recipe + artifact digest
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "quals"))

from quals_harness import (  # noqa: E402  (path inserted above, by design)
    FAIL,
    INFO,
    PASS,
    UNKNOWN,
    Verdict,
)

# ---------------------------------------------------------------------------
# Frozen thresholds (prereg section 2). Do not edit after results are visible.
# ---------------------------------------------------------------------------
TARGET_DUPLICATION_MAX = 0.125
TARGET_ECHO_MAX = 0.062
TARGET_FORMAT_MIN = 8  # of the 8 declared constrained instrument prompts
PROTECTED_ANSWER_CORRECT_MIN = 15  # of 16
PROTECTED_EOS_MIN = 0.900
PROTECTED_CAP_MAX = 0.100
PROTECTED_UNCLOSED_THINK_MAX = 0.250
PROTECTED_LOOP_MAX = 0
PROTECTED_TRIGRAM_MIN = 0.900
SLICE_REGRESSION_MAX = 0.0625
CANDIDATE_MEASURED = "MEASURED_THIS_GENERATION"
CONTAMINATED = {"KNOWN_CONTAMINATION", "POSSIBLE"}

# The 16 instrument prompts, in frozen order, with their answer-surface
# expectations used by the duplication/echo/format checks.
CONSTRAINED_PROMPTS = {
    "What is 17 * 23? Answer with the number only.",
    "Name the capital of Australia in one word.",
    "Count from 1 to 5, digits only.",
    "What is the boiling point of water in Celsius?",
    "Say 'done' and nothing else.",
    "How many continents are there?",
    "Answer with a single word: 2 + 2 =",
    "Reply with exactly: ping",
}


def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _rate(numerator: int, denominator: int) -> float | None:
    return (numerator / denominator) if denominator else None


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


def judge(run_root: Path) -> int:
    verdict = Verdict()
    candidate_eval = _load(run_root / "candidate_evaluation.json")
    accounting = _load(run_root / "cycle_compute_accounting.json")
    contamination = _load(run_root / "gen2_contamination_manifest.json")
    chosen = _load(run_root / "chosen_candidate.json")

    if candidate_eval is None:
        verdict.add("T1", "candidate instrument evidence", UNKNOWN, "candidate_evaluation.json missing or unreadable")
        print(verdict.render())
        print()
        print(f"VERDICT: {verdict.finalize_status()}")
        return 1

    row_origin = candidate_eval.get("measurement_origin")
    diagnostics = candidate_eval.get("diagnostics") or {}
    # Provenance must be DECLARED by the artifact. An artifact that carries no
    # origin map is UNKNOWN -- assuming candidate provenance would be the very
    # fail-open default this judge exists to prevent.
    if not isinstance(row_origin, dict) or "diagnostics" not in row_origin:
        instrument_origin = None
    else:
        instrument_origin = row_origin["diagnostics"]
    instrument_measured = instrument_origin == CANDIDATE_MEASURED
    if instrument_origin is None:
        verdict.add("T1", "instrument measured on candidate", UNKNOWN,
                    "candidate_evaluation.json declares no measurement_origin")
    else:
        verdict.add(
            "T1",
            "instrument measured on candidate",
            PASS if instrument_measured else FAIL,
            f"measurement_origin={instrument_origin}",
        )

    per_prompt = diagnostics.get("per_prompt") or []
    n = len(per_prompt)
    if not instrument_measured or n == 0:
        for name in ("T2", "T3", "T4", "T5", "T6", "T7"):
            verdict.add(name, "instrument metric", UNKNOWN, "no candidate-measured per-prompt evidence")
    else:
        dup = sum(1 for p in per_prompt if _answer_duplicated(p.get("completion", "")))
        echo = sum(1 for p in per_prompt if _template_echo(p.get("completion", "")))
        dup_rate, echo_rate = _rate(dup, n), _rate(echo, n)

        verdict.add(
            "T2",
            f"answer-duplication rate <= {TARGET_DUPLICATION_MAX}",
            PASS if dup_rate is not None and dup_rate <= TARGET_DUPLICATION_MAX else FAIL,
            f"measured {dup_rate:.3f} ({dup}/{n}); parent 0.688",
        )
        verdict.add(
            "T3",
            f"template-echo rate <= {TARGET_ECHO_MAX}",
            PASS if echo_rate is not None and echo_rate <= TARGET_ECHO_MAX else FAIL,
            f"measured {echo_rate:.3f} ({echo}/{n}); parent 0.438",
        )

        constrained = [
            p for p in per_prompt if p.get("prompt", "").strip() in CONSTRAINED_PROMPTS
        ]
        if len(constrained) < 8:
            verdict.add("T4", "constrained-prompt format compliance", UNKNOWN,
                        f"only {len(constrained)} constrained prompts located (expected 8)")
        else:
            # A constrained prompt complies when the answer surface is a single
            # short line (no enumeration) under 40 characters. All 8 declared
            # constrained prompts must comply (prereg section 2, T4).
            ok = 0
            for p in constrained:
                surface = _answer_surface(p.get("completion", ""))
                if surface and len(surface) <= 40 and surface.count("\n") <= 1:
                    ok += 1
            verdict.add(
                "T4",
                f"constrained-prompt compliance == {TARGET_FORMAT_MIN}/8",
                PASS if ok >= TARGET_FORMAT_MIN else FAIL,
                f"compliant {ok}/{len(constrained)} constrained prompts",
            )

        surface_correct = sum(
            1
            for p in per_prompt
            if (p.get("expected") or "").lower() in _answer_surface(p.get("completion", "")).lower()
        )
        verdict.add(
            "T5",
            f"answer correctness >= {PROTECTED_ANSWER_CORRECT_MIN}/16",
            PASS if surface_correct >= PROTECTED_ANSWER_CORRECT_MIN else FAIL,
            f"measured {surface_correct}/{n}; parent 16/16",
        )

        eos = diagnostics.get("eos_termination_rate")
        eos_ok = isinstance(eos, (int, float)) and eos >= PROTECTED_EOS_MIN
        verdict.add("T6", f"EOS termination >= {PROTECTED_EOS_MIN}", PASS if eos_ok else FAIL,
                    f"measured {eos}; parent 1.000")

        cap = diagnostics.get("max_token_cap_rate")
        loops = diagnostics.get("obvious_loop_count")
        trigram = diagnostics.get("distinct_trigram_ratio_mean")
        unclosed = diagnostics.get("unclosed_think_rate")
        verdict.add("T7", f"cap-hit < {PROTECTED_CAP_MAX}", PASS if isinstance(cap, (int, float)) and cap < PROTECTED_CAP_MAX else FAIL, f"measured {cap}")
        verdict.add("T8", f"obvious loops <= {PROTECTED_LOOP_MAX}", PASS if loops == 0 else FAIL, f"measured {loops}")
        verdict.add("T9", f"distinct-trigram >= {PROTECTED_TRIGRAM_MIN}", PASS if isinstance(trigram, (int, float)) and trigram >= PROTECTED_TRIGRAM_MIN else FAIL, f"measured {trigram}")
        verdict.add("T10", f"unclosed think <= {PROTECTED_UNCLOSED_THINK_MAX}", PASS if isinstance(unclosed, (int, float)) and unclosed <= PROTECTED_UNCLOSED_THINK_MAX else FAIL, f"measured {unclosed}")

    # T11/T12: mini-slice protected evidence, candidate-measured only.
    slices = candidate_eval.get("protected_slices") or []
    if not slices:
        verdict.add("T11", "candidate-measured protected slices", UNKNOWN,
                    "no protected_slices in the candidate evaluation artifact")
    for row in slices:
        qid = row.get("benchmark_qualified_id", "?")
        origin = row.get("measurement_origin")
        parent_score = row.get("parent_score")
        score = row.get("score")
        if origin != CANDIDATE_MEASURED:
            verdict.add("T11", f"slice {qid} candidate-measured", FAIL,
                        f"measurement_origin={origin} (carried/parent evidence cannot gate)")
            continue
        if not isinstance(score, (int, float)) or not isinstance(parent_score, (int, float)):
            verdict.add("T11", f"slice {qid} score", UNKNOWN, "score/parent_score missing")
            continue
        delta = score - parent_score
        verdict.add("T11", f"slice {qid} regression <= {SLICE_REGRESSION_MAX}", PASS if delta >= -SLICE_REGRESSION_MAX else FAIL,
                    f"candidate {score:.4f} vs parent {parent_score:.4f} (delta {delta:+.4f})")

    # T12: contamination — every evaluated benchmark CLEAN.
    if contamination is None:
        verdict.add("T12", "contamination CLEAN on evaluated benchmarks", UNKNOWN,
                    "gen2_contamination_manifest.json missing")
    else:
        benchmarks = contamination.get("benchmarks") or {}
        flagged = {k: v.get("status") for k, v in benchmarks.items() if isinstance(v, dict) and v.get("status") in CONTAMINATED}
        unchecked = [k for k, v in benchmarks.items() if not isinstance(v, dict) or not v.get("status")]
        if flagged:
            verdict.add("T12", "contamination CLEAN", FAIL, f"flagged: {flagged}")
        elif unchecked:
            verdict.add("T12", "contamination CLEAN", UNKNOWN, f"unchecked: {unchecked}")
        else:
            verdict.add("T12", "contamination CLEAN", PASS, f"{len(benchmarks)} benchmarks, all CLEAN")

    # T13: settlement over the complete accounting artifact.
    if accounting is None:
        verdict.add("T13", "actual cost settled within ceilings", UNKNOWN,
                    "cycle_compute_accounting.json missing")
    else:
        totals = ((accounting.get("totals") or {}).get("incremental") or {})
        wall = totals.get("wall_gpu_hours")
        ceiling = 1.50  # campaign wall ceiling (prereg section 5)
        if not isinstance(wall, (int, float)):
            verdict.add("T13", "actual cost settled within ceilings", UNKNOWN, "wall_gpu_hours missing")
        else:
            verdict.add("T13", f"actual wall cost <= {ceiling}", PASS if wall <= ceiling else FAIL,
                        f"measured {wall:.4f}")
        # Losing-recipe accounting must be present (winner-only sums are the gen1 defect).
        recipe_ids = {
            entry.get("recipe_id")
            for entry in (accounting.get("entries") or [])
            if entry.get("kind") in {"training", "evaluation"} and entry.get("recipe_id")
        }
        verdict.add("T14", "all recipes accounted", PASS if len(recipe_ids) >= 2 else FAIL,
                    f"recipe ids in accounting: {sorted(recipe_ids)}")

    # T15: chosen candidate identity is recorded and digest-verified.
    if chosen is None:
        verdict.add("T15", "candidate artifact identity", UNKNOWN, "chosen_candidate.json missing")
    else:
        digest = chosen.get("artifact_sha256")
        recorded = bool(chosen.get("artifact_ref")) and isinstance(digest, str) and len(digest) == 64
        verdict.add("T15", "candidate artifact digest recorded", PASS if recorded else FAIL,
                    f"artifact_ref={chosen.get('artifact_ref')} digest={str(digest)[:12]}")

    # INFO rows: context that does not gate certification.
    verdict.add(INFO, "parent evidence state",
                INFO if True else INFO,
                "gen1 effective verdict INCONCLUSIVE (target_repair_validated=true)")
    verdict.add(INFO, "prereg", INFO, "docs/quals/GEN2_PREREG_2026-09-17.md (frozen)")

    print(f"run root: {run_root}")
    print()
    print(verdict.render())
    print()
    print(f"VERDICT: {verdict.finalize_status()}")
    statuses = {row[2] for row in verdict.thresholds()}
    return 0 if statuses <= {PASS} else 1


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
