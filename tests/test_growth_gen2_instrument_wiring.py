"""The instrument the run output must carry for T1-T10 to decide.

The gap this closes, stated as a measurement rather than a plan: the frozen
judge's T1-T10 read one row of the run's candidate arm -- the generation
diagnostics instrument -- and every key they read must be in that row's
``metadata``. Nothing previously tied the two sides together, so the
end-to-end judge-agreement measurement had to record T1-T10 as ``UNKNOWN``
"because this fixture's synthetic candidate arm leaves the instrument gates
undecided". Production *does* emit the metadata
(:class:`chowder.growth.generation_diagnostics.GenerationDiagnostics`), and the
committed declaration *does* declare the instrument as its target set; what was
missing was a proof that those two facts make the gates decide rather than
shrug.

So this asserts, through the judge's own reader:

* the declaration's target set is the judge's ``INSTRUMENT_ID`` -- otherwise T1
  is ``UNKNOWN`` no matter what the run measures;
* a candidate arm carrying exactly what ``to_metadata()`` emits drives all ten
  thresholds to a *decided* state (PASS or FAIL, never ``UNKNOWN``), which is
  what "a real arm can certify" means;
* the parent arm's per-prompt identities align with the candidate's, because
  T2/T3 are paired comparisons and an unalignable parent makes them
  ``UNKNOWN``;
* every prompt identity is unambiguous, because :meth:`Arm.flags` refuses a
  duplicate rather than pairing an ambiguous arm.

A rename or reshape of any key here breaks this file, which is the point: the
judge is frozen, so the *producer* is what has to keep matching it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

from chowder.evals.result import MEASURED_PARENT, MEASURED_THIS_GENERATION, BenchmarkRun, EvalReport
from chowder.growth.generation_diagnostics import (
    INSTRUMENT_PROMPTS,
    GenerationDiagnostics,
)

GEN2 = Path(__file__).resolve().parent.parent / "docs" / "gen2"
DECLARATION = GEN2 / "gen2_campaign.json"

_spec = importlib.util.spec_from_file_location("judge_gen2_instrument", GEN2 / "judge_gen2.py")
judge_gen2 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(judge_gen2)

#: The declared generation the judge and the campaign agree on.
CANDIDATE_VERSION = "gen2"
PARENT_VERSION = "gen1"


def _generations(completion_for) -> list[dict[str, Any]]:
    """The worker's own per-item rows, one per frozen instrument prompt.

    These are the facts only the generating worker can observe -- how many
    tokens it produced and whether it stopped on EOS -- so a fixture states
    them rather than letting them be inferred from the text.
    """
    rows = []
    for index, (prompt, expected) in enumerate(INSTRUMENT_PROMPTS):
        rows.append(
            {
                "prompt": prompt,
                "expected": expected,
                "prediction": completion_for(prompt, expected),
                "generated_tokens": 12,
                "eos_terminated": True,
            }
        )
    return rows


def _instrument_row(
    *,
    generation: str,
    origin: str,
    completion_for,
) -> BenchmarkRun:
    """The run's instrument row, exactly as the production evaluator emits it."""
    diagnostics = GenerationDiagnostics.from_items(
        _generations(completion_for),
        max_new_tokens=512,
        seed=judge_gen2.PROTECTED_SEED,
        source=f"fixture:{generation}",
    )
    items = _generations(completion_for)
    return BenchmarkRun(
        benchmark_qualified_id=judge_gen2.INSTRUMENT_ID,
        adapter="transformers_text",
        generation_version=generation,
        score=sum(1.0 if item["prediction"] == item["expected"] else 0.0
                  for item in items) / len(items),
        n_samples=diagnostics.n_prompts,
        # The declared protocol certifies against these, and the diagnostics
        # alone cannot stand in for them: the per-item scores are what make
        # ``n_samples`` and the aggregate checkable.
        per_sample_scores=tuple(1.0 if item["prediction"] == item["expected"] else 0.0
                                for item in items),
        metric="eos_termination_rate",
        measurement_origin=origin,
        metadata=diagnostics.to_metadata(),
    )


def _write_arm(path: Path, run: BenchmarkRun) -> None:
    EvalReport(generation_version=run.generation_version, runs=(run,)).save(path)


def _candidates(root: Path) -> tuple[object, object]:
    """The candidate and parent arms as the judge opens them."""
    candidate = _instrument_row(
        generation=CANDIDATE_VERSION,
        origin=MEASURED_THIS_GENERATION,
        completion_for=lambda prompt, expected: expected,
    )
    parent = _instrument_row(
        generation=PARENT_VERSION,
        origin=MEASURED_PARENT,
        completion_for=lambda prompt, expected: (
            expected if prompt not in judge_gen2.CONSTRAINED_PROMPTS else f"  {expected}"
        ),
    )
    _write_arm(root / "candidate_evaluation.json", candidate)
    _write_arm(root / "parent_evaluation.json", parent)
    return (
        judge_gen2.Arm.open(
            root / "candidate_evaluation.json",
            expected_origin=MEASURED_THIS_GENERATION,
            expected_generation=CANDIDATE_VERSION,
            label="candidate",
        ),
        judge_gen2.Arm.open(
            root / "parent_evaluation.json",
            expected_origin=MEASURED_PARENT,
            expected_generation=PARENT_VERSION,
            label="parent",
        ),
    )


def _instrument_statuses(root: Path) -> dict[str, str]:
    """T1-T10 as the judge's own instrument gate decides them."""
    candidate, parent = _candidates(root)
    verdict = judge_gen2.Verdict()
    judge_gen2._instrument_gates(verdict, candidate, parent)
    return {row[0]: row[2] for row in verdict.thresholds()}


def test_the_committed_declaration_measures_the_instrument_the_judge_reads() -> None:
    """T1 is a lookup by qualified id, so a different target id is a dead gate.

    ``generation-diagnostics@gen2-response-surface-v1`` is the id the frozen
    judge hardcodes. If the shipped declaration named any other target set,
    every real run would leave T1-T10 ``UNKNOWN`` while the run itself looked
    healthy -- the exact failure mode this file exists to rule out.
    """
    import json

    declaration = json.loads(DECLARATION.read_text(encoding="utf-8"))
    assert judge_gen2.INSTRUMENT_ID in declaration["target_benchmarks"], declaration[
        "target_benchmarks"
    ]


def test_the_production_instrument_prompts_are_the_ones_the_judge_scores() -> None:
    """The dataset and the gate must agree on which prompts exist.

    ``INSTRUMENT_PROMPTS`` is the production owner of the instrument's items and
    ``CONSTRAINED_PROMPTS`` is the judge's subset of them; T4 locates 8 of 8 by
    prompt text, so a prompt that drifted between the two owners would silently
    make T4 ``UNKNOWN``.
    """
    prompts = {prompt for prompt, _expected in INSTRUMENT_PROMPTS}
    assert len(prompts) == 16, len(prompts)
    assert judge_gen2.CONSTRAINED_PROMPTS <= prompts, sorted(
        judge_gen2.CONSTRAINED_PROMPTS - prompts
    )
    assert judge_gen2.PROTECTED_N_SAMPLES == len(INSTRUMENT_PROMPTS)


def test_a_run_output_carrying_the_production_instrument_decides_all_ten_gates(
    tmp_path: Path,
) -> None:
    """The measurement the gap was about: decided, not ``UNKNOWN``.

    A row carrying ``to_metadata()`` must give the judge something to decide
    on. ``UNKNOWN`` is the failure this file exists to rule out: it is what an
    audit table looks like when the run's output never carried the instrument.

    ``PASS`` is expected of the gates a well-behaved candidate earns -- it
    answers every prompt, terminates on EOS and hits no cap -- while T2/T3 are
    *paired* comparisons against this fixture's deliberately duplication-prone
    parent, and the improvement is too small to cross the frozen minimum
    effect. Those two FAIL, which is a decision, and is exactly what a FAIL is
    for.
    """
    statuses = _instrument_statuses(tmp_path)
    assert set(statuses) == {f"T{index}" for index in range(1, 11)}, sorted(statuses)
    undecided = {key: value for key, value in statuses.items() if value == "UNKNOWN"}
    assert not undecided, f"T1-T10 cannot decide on the production metadata: {undecided}"
    earned = {key for key in ("T1", "T4", "T5", "T6", "T7", "T8", "T9", "T10")}
    assert {key for key in earned if statuses[key] == "PASS"} == earned, statuses
    assert {key for key, value in statuses.items() if value == "FAIL"} <= {"T2", "T3"}, statuses


def test_every_per_prompt_identity_is_unambiguous_and_aligns_across_arms(
    tmp_path: Path,
) -> None:
    """T2/T3 are *paired* rules, so ambiguous identity is ``UNKNOWN``, not a pass.

    ``Arm.flags`` refuses a duplicated or missing identity rather than pairing
    two arms on identities it cannot trust; this pins that the production
    metadata yields a clean alignment on both sides.
    """
    candidate, parent = _candidates(tmp_path)
    for arm in (candidate, parent):
        entries = arm.per_prompt()
        assert len(entries) == 16, len(entries)
        flags = arm.flags(judge_gen2._duplication_flag)
        assert flags is not None, f"{arm.label} prompt identities are ambiguous"
        assert len(flags) == 16, len(flags)
    assert set(candidate.flags(judge_gen2._duplication_flag)) == set(
        parent.flags(judge_gen2._duplication_flag)
    )


@pytest.mark.parametrize(
    ("key", "gate"),
    [
        ("eos_termination_rate", "T6"),
        ("max_token_cap_rate", "T7"),
        ("obvious_loop_count", "T8"),
        ("distinct_trigram_ratio_mean", "T9"),
        ("unclosed_think_rate", "T10"),
    ],
)
def test_each_diagnostics_key_is_read_by_the_gate_that_names_it(key: str, gate: str) -> None:
    """One row per key: the judge's gate list and the producer's keys must match.

    Read out of the frozen source rather than hardcoded here, so a new gate
    without a producer key (or a producer key no gate reads) fails here rather
    than as an ``UNKNOWN`` in a table someone reads later.
    """
    source = (GEN2 / "judge_gen2.py").read_text(encoding="utf-8")
    start = source.index("def _instrument_gates(")
    body = source[start : start + source.index("\ndef _paired_target_gate(", start)]
    assert f'"{gate}"' in body, gate
    assert f'"{key}"' in body, key
    metadata = GenerationDiagnostics.from_items(
        _generations(lambda prompt, expected: expected),
        max_new_tokens=512,
        seed=judge_gen2.PROTECTED_SEED,
        source="fixture",
    ).to_metadata()
    assert isinstance(metadata.get(key), (int, float)), key
    assert not isinstance(metadata.get(key), bool), key


def test_the_instrument_row_is_also_a_certifiable_protected_slice() -> None:
    """The same row must satisfy the *declared protocol* too, not just T1-T10.

    T1-T10 read the diagnostics; ``certification.protocol_problems`` reads the
    protocol fields. Both live in one ``metadata`` mapping, so a producer that
    emitted one and not the other would produce an arm that scores and cannot
    certify. ``evaluation_binding`` writes both from the same call, which is
    what this asserts by construction.
    """
    from chowder.growth.certification import ProtocolSpec

    run = _instrument_row(
        generation=CANDIDATE_VERSION,
        origin=MEASURED_THIS_GENERATION,
        completion_for=lambda prompt, expected: expected,
    )
    metadata = dict(run.metadata)
    metadata.update(
        {
            "sample_indices": list(range(len(INSTRUMENT_PROMPTS))),
            "seed": judge_gen2.PROTECTED_SEED,
            "shuffle": judge_gen2.PROTECTED_SHUFFLE,
            "decoding": dict(judge_gen2.PROTECTED_DECODING),
            "prompt_policy": judge_gen2.PROTECTED_PROMPT_POLICY,
        }
    )
    protocol_run = BenchmarkRun(
        benchmark_qualified_id=run.benchmark_qualified_id,
        adapter=run.adapter,
        generation_version=run.generation_version,
        score=run.score,
        n_samples=run.n_samples,
        per_sample_scores=run.per_sample_scores,
        metric=run.metric,
        measurement_origin=run.measurement_origin,
        raw_artifact_ref=run.raw_artifact_ref,
        metadata=metadata,
    )
    spec = ProtocolSpec.from_mapping(
        {
            "n_samples": judge_gen2.PROTECTED_N_SAMPLES,
            "seed": judge_gen2.PROTECTED_SEED,
            "shuffle": judge_gen2.PROTECTED_SHUFFLE,
            "decoding": dict(judge_gen2.PROTECTED_DECODING),
            "prompt_policy": judge_gen2.PROTECTED_PROMPT_POLICY,
        }
    )
    from chowder.growth.certification import protocol_problems

    # Only the raw-artifact evidence is outstanding here: this fixture states no
    # bytes on disk, so those three problems are expected and the protocol
    # fields themselves must be clean.
    problems = protocol_problems(protocol_run, run_root=Path("."), protocol=spec)
    assert not [p for p in problems if "n_samples=" in p or "sample_indices=" in p
                or "seed=" in p or "shuffle=" in p or "decoding." in p
                or "prompt_policy=" in p], problems