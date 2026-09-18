"""The frozen gen2 judge, tested before the run it will judge.

A judge that always refuses is as useless as one that always certifies, so
every gate below is pinned in both directions with synthetic fixtures, plus
the epistemics rule inherited from the re-adjudication: undeclared provenance,
missing protocol metadata and unaligned prompt identities are UNKNOWN, never a
silent pass.

The judge must also not drift from the production engine it claims to read:
the settlement verdict is the production :func:`settle_campaign` answer, the
artifact digest is the production :func:`directory_digest`, and the frozen
instrument prompt list is cross-checked against the gen1 driver's source so
the two cannot diverge.
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pytest

from chowder.evals.result import (
    MEASURED_PARENT,
    MEASURED_THIS_GENERATION,
    BenchmarkRun,
    EvalReport,
)
from chowder.growth.campaign import CampaignManifest, settle_campaign
from chowder.growth.compute_cost import ComputeCost
from chowder.growth.statistics import compare
from chowder.growth.training_binding import directory_digest

GEN2 = Path(__file__).resolve().parent.parent / "docs" / "gen2"
REPO = GEN2.parent.parent
_spec = importlib.util.spec_from_file_location("judge_gen2", GEN2 / "judge_gen2.py")
judge_gen2 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(judge_gen2)

INSTRUMENT = judge_gen2.INSTRUMENT_ID
MATH = "math500@2024-04"
MGSM = "mgsm@2022-11"
DECODING = dict(judge_gen2.PROTECTED_DECODING)


# --------------------------------------------------------------------------
# instrument fixtures
# --------------------------------------------------------------------------


def _completion(*, duplicate: bool = False, echo: bool = False, answer: str = "ping") -> str:
    head = " with exactly: ping\nassistant\n" if echo else ""
    think_body = f"I should answer directly.\n{answer}\n" if duplicate else "I should answer directly.\n"
    return f"{head}<think>\n{think_body}</think>\n\n{answer}"


def _prompt_entries(*, dup: int, echo: int, correct: bool = True) -> list[dict]:
    entries = []
    for index, (prompt, expected) in enumerate(judge_gen2.INSTRUMENT_PROMPTS):
        entries.append(
            {
                "prompt_id": f"gen1-diag-{index:02d}",
                "prompt": prompt,
                "expected": expected,
                "completion": _completion(
                    duplicate=index < dup,
                    echo=index < echo,
                    answer=expected if correct else "definitely-wrong",
                ),
            }
        )
    return entries


def _instrument_run(
    version: str,
    origin: str,
    *,
    dup: int = 0,
    echo: int = 0,
    correct: bool = True,
    metadata_override: dict | None = None,
) -> BenchmarkRun:
    entries = _prompt_entries(dup=dup, echo=echo, correct=correct)
    metadata = {
        "per_prompt": entries,
        "eos_termination_rate": 1.0,
        "max_token_cap_rate": 0.0,
        "unclosed_think_rate": 0.0,
        "obvious_loop_count": 0,
        "distinct_trigram_ratio_mean": 0.97,
    }
    if metadata_override:
        metadata.update(metadata_override)
    return BenchmarkRun(
        benchmark_qualified_id=INSTRUMENT,
        adapter="chowder_custom",
        generation_version=version,
        score=0.0,
        n_samples=len(entries),
        metric="response_surface_compliance",
        measurement_origin=origin,
        raw_artifact_ref=f"raw/{version}-instrument.json",
        metadata=metadata,
    )


def _slice_run(
    qualified_id: str,
    version: str,
    origin: str,
    score: float,
    *,
    n_samples: int = 16,
    indices: list[int] | None = None,
    seed: int = 1234,
    shuffle: bool = False,
    decoding: dict | None = None,
    prompt_policy: str = "chat_template",
    artifact_ref: str = "raw/slice.json",
) -> BenchmarkRun:
    metadata = {
        "sample_indices": list(range(16)) if indices is None else indices,
        "seed": seed,
        "shuffle": shuffle,
        "decoding": DECODING if decoding is None else decoding,
        "prompt_policy": prompt_policy,
    }
    return BenchmarkRun(
        benchmark_qualified_id=qualified_id,
        adapter="lm_eval",
        generation_version=version,
        score=score,
        n_samples=n_samples,
        per_sample_scores=tuple([score] * n_samples),
        metric="exact_match",
        measurement_origin=origin,
        raw_artifact_ref=artifact_ref,
        metadata=metadata,
    )


def _write_arm(
    path: Path,
    version: str,
    origin: str,
    *,
    dup: int = 0,
    echo: int = 0,
    correct: bool = True,
    slices: tuple[BenchmarkRun, ...] = (),
    metadata_override: dict | None = None,
    with_instrument: bool = True,
) -> Path:
    runs = []
    if with_instrument:
        runs.append(
            _instrument_run(
                version, origin, dup=dup, echo=echo, correct=correct,
                metadata_override=metadata_override,
            )
        )
    runs.extend(slices)
    EvalReport(generation_version=version, runs=tuple(runs)).save(path)
    return path


# --------------------------------------------------------------------------
# run-root fixture: three provenance-bound arms + accounting + artifacts
# --------------------------------------------------------------------------


ARTIFACT_FILE = "adapter_model.safetensors"


def _build_artifact(root: Path) -> tuple[Path, str]:
    artifact = root / "candidate-adapter"
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact / ARTIFACT_FILE).write_text("weights-for-gen2", encoding="utf-8")
    digest, _entries = directory_digest(artifact)
    return artifact, digest


def _accounting(wall: float = 1.1, *, device: float = 0.40, device_measured: bool = False) -> dict:
    return {
        "totals": {
            "incremental": {
                "device_gpu_hours": device,
                "wall_gpu_hours": wall,
                "device_measured": device_measured,
                "source": "total",
            }
        },
        "entries": [
            {"kind": "training", "recipe_id": "gen2-recipe-a"},
            {"kind": "training", "recipe_id": "gen2-recipe-b"},
            {"kind": "evaluation", "recipe_id": "gen2-recipe-b"},
        ],
    }


def _contamination(*, benchmarks: dict | None = None, training_sources: dict | None = None) -> dict:
    return {
        "benchmarks": (
            {INSTRUMENT: {"status": "CLEAN"}, MATH: {"status": "CLEAN"}, MGSM: {"status": "CLEAN"}}
            if benchmarks is None
            else benchmarks
        ),
        "training_sources": (
            {"src-1": {"status": "CLEAN"}} if training_sources is None else training_sources
        ),
    }


def _run_root(
    tmp_path: Path,
    *,
    candidate_dup: int = 0,
    candidate_echo: int = 0,
    candidate_correct: bool = True,
    candidate_instrument_metadata: dict | None = None,
    parent_dup: int = 11,
    parent_echo: int = 7,
    candidate_slices: tuple[BenchmarkRun, ...] | None = None,
    parent_slices: tuple[BenchmarkRun, ...] | None = None,
    ancestor_slices: tuple[BenchmarkRun, ...] | None = None,
    parent_arm: bool = True,
    ancestor_arm: bool = True,
    contamination: dict | None = None,
    accounting: dict | None = None,
    artifact_digest: str | None = None,
    artifact_ref: str | None = None,
    chosen: bool = True,
) -> Path:
    root = tmp_path / "run"
    root.mkdir(parents=True, exist_ok=True)

    if candidate_slices is None:
        candidate_slices = (
            _slice_run(MATH, "gen2", MEASURED_THIS_GENERATION, 0.0),
            _slice_run(MGSM, "gen2", MEASURED_THIS_GENERATION, 0.0625),
        )
    if parent_slices is None:
        parent_slices = (
            _slice_run(MATH, "gen1", MEASURED_PARENT, 0.0),
            _slice_run(MGSM, "gen1", MEASURED_PARENT, 0.0),
        )
    if ancestor_slices is None:
        ancestor_slices = (
            _slice_run(MATH, "gen0", MEASURED_PARENT, 0.0),
            _slice_run(MGSM, "gen0", MEASURED_PARENT, 0.0),
        )

    _write_arm(
        root / "candidate_evaluation.json", "gen2", MEASURED_THIS_GENERATION,
        dup=candidate_dup, echo=candidate_echo, correct=candidate_correct,
        slices=candidate_slices, metadata_override=candidate_instrument_metadata,
    )
    if parent_arm:
        _write_arm(
            root / "parent_evaluation.json", "gen1", MEASURED_PARENT,
            dup=parent_dup, echo=parent_echo, slices=parent_slices,
        )
    if ancestor_arm:
        _write_arm(
            root / "baseline_evaluation.json", "gen0", MEASURED_PARENT,
            slices=ancestor_slices,
        )

    (root / "cycle_compute_accounting.json").write_text(
        json.dumps(_accounting() if accounting is None else accounting), encoding="utf-8"
    )
    (root / "gen2_contamination_manifest.json").write_text(
        json.dumps(_contamination() if contamination is None else contamination), encoding="utf-8"
    )

    if chosen:
        artifact, digest = _build_artifact(root)
        reference = artifact_ref if artifact_ref is not None else str(artifact)
        (root / "chosen_candidate.json").write_text(
            json.dumps(
                {
                    "recipe_id": "gen2-recipe-b",
                    "artifact_ref": reference,
                    "artifact_sha256": artifact_digest if artifact_digest is not None else digest,
                }
            ),
            encoding="utf-8",
        )
    return root


# --------------------------------------------------------------------------
# the clean run certifies; the gen1-shaped defect refuses
# --------------------------------------------------------------------------


def test_judge_certifies_a_clean_run(tmp_path: Path) -> None:
    root = _run_root(tmp_path)
    assert judge_gen2.judge(root) == 0


def test_judge_refuses_gen1_shaped_defects(tmp_path: Path) -> None:
    """The measured gen1 defect rates (dup 0.688, echo 0.438) must refuse."""
    root = _run_root(tmp_path, candidate_dup=11, candidate_echo=7)
    assert judge_gen2.judge(root) == 1


def test_judge_refuses_when_the_candidate_arm_is_missing(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    root.mkdir()
    assert judge_gen2.judge(root) == 1


# --------------------------------------------------------------------------
# contamination coverage is exact and fail-closed
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "benchmarks",
    (
        {},
        {MATH: {"status": "CLEAN"}},
        {MATH: {"status": "CLEAN"}, MGSM: {"status": "CLEAN"}},
        {INSTRUMENT: {"status": "CLEAN"}, MATH: {"status": "CLEAN"}},
        {INSTRUMENT: {"status": "CLEAN"}, MATH: {"reason": "not checked"}},
    ),
)
def test_contamination_coverage_must_be_exact(tmp_path: Path, benchmarks: dict) -> None:
    root = _run_root(tmp_path, contamination=_contamination(benchmarks=benchmarks))
    assert judge_gen2.judge(root) == 1


def test_contamination_flag_is_a_hard_failure(tmp_path: Path) -> None:
    benchmarks = {
        INSTRUMENT: {"status": "CLEAN"},
        MATH: {"status": "POSSIBLE"},
        MGSM: {"status": "KNOWN_CONTAMINATION"},
    }
    root = _run_root(tmp_path, contamination=_contamination(benchmarks=benchmarks))
    assert judge_gen2.judge(root) == 1


def test_an_empty_training_source_section_is_not_clean(tmp_path: Path) -> None:
    root = _run_root(tmp_path, contamination=_contamination(training_sources={}))
    assert judge_gen2.judge(root) == 1


def test_benchmark_coverage_alone_is_not_enough(tmp_path: Path) -> None:
    """Extra CLEAN rows never compensate for a missing required one."""
    benchmarks = {
        INSTRUMENT: {"status": "CLEAN"},
        MATH: {"status": "CLEAN"},
        "extra@v1": {"status": "CLEAN"},
    }
    root = _run_root(tmp_path, contamination=_contamination(benchmarks=benchmarks))
    assert judge_gen2.judge(root) == 1


# --------------------------------------------------------------------------
# protected mini-battery: exact identity + protocol
# --------------------------------------------------------------------------


def test_both_required_slices_present_and_protocol_exact(tmp_path: Path) -> None:
    assert judge_gen2.judge(_run_root(tmp_path)) == 0


@pytest.mark.parametrize(
    "slices",
    (
        # math only: mgsm missing
        (_slice_run(MATH, "gen2", MEASURED_THIS_GENERATION, 0.0),),
        # mgsm only: math missing
        (_slice_run(MGSM, "gen2", MEASURED_THIS_GENERATION, 0.0),),
        # duplicate math rows
        (
            _slice_run(MATH, "gen2", MEASURED_THIS_GENERATION, 0.0),
            _slice_run(MATH, "gen2", MEASURED_THIS_GENERATION, 0.0625),
            _slice_run(MGSM, "gen2", MEASURED_THIS_GENERATION, 0.0),
        ),
        # wrong sample count
        (
            _slice_run(MATH, "gen2", MEASURED_THIS_GENERATION, 0.0, n_samples=8, indices=list(range(8))),
            _slice_run(MGSM, "gen2", MEASURED_THIS_GENERATION, 0.0),
        ),
        # wrong indices (offset window)
        (
            _slice_run(MATH, "gen2", MEASURED_THIS_GENERATION, 0.0, indices=list(range(1, 17))),
            _slice_run(MGSM, "gen2", MEASURED_THIS_GENERATION, 0.0),
        ),
        # wrong seed
        (
            _slice_run(MATH, "gen2", MEASURED_THIS_GENERATION, 0.0, seed=4321),
            _slice_run(MGSM, "gen2", MEASURED_THIS_GENERATION, 0.0),
        ),
        # shuffled
        (
            _slice_run(MATH, "gen2", MEASURED_THIS_GENERATION, 0.0, shuffle=True),
            _slice_run(MGSM, "gen2", MEASURED_THIS_GENERATION, 0.0),
        ),
        # wrong decoding
        (
            _slice_run(
                MATH,
                "gen2",
                MEASURED_THIS_GENERATION,
                0.0,
                decoding={"temperature": 0.7, "do_sample": True, "max_new_tokens": 512},
            ),
            _slice_run(MGSM, "gen2", MEASURED_THIS_GENERATION, 0.0),
        ),
        # wrong prompt policy
        (
            _slice_run(MATH, "gen2", MEASURED_THIS_GENERATION, 0.0, prompt_policy="raw_completion"),
            _slice_run(MGSM, "gen2", MEASURED_THIS_GENERATION, 0.0),
        ),
        # carried candidate origin
        (
            _slice_run(MATH, "gen2", "CARRIED_REFERENCE", 0.0),
            _slice_run(MGSM, "gen2", MEASURED_THIS_GENERATION, 0.0),
        ),
        # a slice that regressed past tolerance against the parent arm
        (
            _slice_run(MATH, "gen2", MEASURED_THIS_GENERATION, 0.0),
            _slice_run(MGSM, "gen2", MEASURED_THIS_GENERATION, -0.2),
        ),
    ),
)
def test_malformed_protected_evidence_refuses(tmp_path: Path, slices: tuple) -> None:
    root = _run_root(tmp_path, candidate_slices=slices)
    assert judge_gen2.judge(root) == 1


def test_an_undeclared_protected_row_cannot_substitute_for_a_required_slice(
    tmp_path: Path,
) -> None:
    slices = (
        _slice_run(MATH, "gen2", MEASURED_THIS_GENERATION, 0.0),
        _slice_run(MGSM, "gen2", MEASURED_THIS_GENERATION, 0.0),
        _slice_run("mmlu_pro@v2", "gen2", MEASURED_THIS_GENERATION, 0.9),
    )
    assert judge_gen2.judge(_run_root(tmp_path, candidate_slices=slices)) == 1


# --------------------------------------------------------------------------
# parent evidence is its own artifact
# --------------------------------------------------------------------------


def test_the_parent_arm_must_be_independently_measured(tmp_path: Path) -> None:
    """A missing parent arm cannot be replaced by anything the candidate says."""
    assert judge_gen2.judge(_run_root(tmp_path, parent_arm=False)) == 1


def test_a_parent_row_relabelled_as_candidate_evidence_refuses(tmp_path: Path) -> None:
    """Same score, wrong provenance: the candidate arm must not borrow it."""
    slices = (
        _slice_run(MATH, "gen2", MEASURED_PARENT, 0.0),
        _slice_run(MGSM, "gen2", MEASURED_THIS_GENERATION, 0.0),
    )
    assert judge_gen2.judge(_run_root(tmp_path, candidate_slices=slices)) == 1


def test_the_parent_arm_must_match_the_frozen_protocol_too(tmp_path: Path) -> None:
    parent_slices = (
        _slice_run(MATH, "gen1", MEASURED_PARENT, 0.0, seed=4321),
        _slice_run(MGSM, "gen1", MEASURED_PARENT, 0.0),
    )
    assert judge_gen2.judge(_run_root(tmp_path, parent_slices=parent_slices)) == 1


# --------------------------------------------------------------------------
# trusted-ancestor protection
# --------------------------------------------------------------------------


def test_gen2_cannot_promote_by_inheriting_a_gen1_regression(tmp_path: Path) -> None:
    """Case A: gen0 good, gen1 regressed badly, gen2 matches gen1."""
    ancestor = (
        _slice_run(MATH, "gen0", MEASURED_PARENT, 0.5),
        _slice_run(MGSM, "gen0", MEASURED_PARENT, 0.5),
    )
    parent = (
        _slice_run(MATH, "gen1", MEASURED_PARENT, 0.0),
        _slice_run(MGSM, "gen1", MEASURED_PARENT, 0.0),
    )
    candidate = (
        _slice_run(MATH, "gen2", MEASURED_THIS_GENERATION, 0.0),
        _slice_run(MGSM, "gen2", MEASURED_THIS_GENERATION, 0.0),
    )
    root = _run_root(
        tmp_path, ancestor_slices=ancestor, parent_slices=parent, candidate_slices=candidate
    )
    assert judge_gen2.judge(root) == 1


def test_gen2_promotes_when_the_whole_branch_holds(tmp_path: Path) -> None:
    """Case B: gen0 good, gen1 holds, gen2 holds."""
    ancestor = (
        _slice_run(MATH, "gen0", MEASURED_PARENT, 0.5),
        _slice_run(MGSM, "gen0", MEASURED_PARENT, 0.5),
    )
    parent = (
        _slice_run(MATH, "gen1", MEASURED_PARENT, 0.5),
        _slice_run(MGSM, "gen1", MEASURED_PARENT, 0.5),
    )
    candidate = (
        _slice_run(MATH, "gen2", MEASURED_THIS_GENERATION, 0.5),
        _slice_run(MGSM, "gen2", MEASURED_THIS_GENERATION, 0.5),
    )
    root = _run_root(
        tmp_path, ancestor_slices=ancestor, parent_slices=parent, candidate_slices=candidate
    )
    assert judge_gen2.judge(root) == 0


def test_an_absent_parent_arm_stays_inconclusive_unless_the_ancestor_resolves_it(
    tmp_path: Path,
) -> None:
    """Case C: gen1 unresolved, with and without independent gen0 resolution."""
    good_ancestor = (
        _slice_run(MATH, "gen0", MEASURED_PARENT, 0.5),
        _slice_run(MGSM, "gen0", MEASURED_PARENT, 0.5),
    )
    holding = (
        _slice_run(MATH, "gen2", MEASURED_THIS_GENERATION, 0.5),
        _slice_run(MGSM, "gen2", MEASURED_THIS_GENERATION, 0.5),
    )
    # Gen1's protected measurement is unresolved: the arm carries the target
    # instrument (so the paired rule is applicable) but no mini-slice rows.
    # Resolved by the trusted ancestor: promotion stays possible.
    resolved = _run_root(
        tmp_path / "resolved",
        parent_slices=(),
        ancestor_slices=good_ancestor,
        candidate_slices=holding,
    )
    assert judge_gen2.judge(resolved) == 0

    # Not resolved: the candidate regressed against gen0 -> hard refusal.
    regressed = (
        _slice_run(MATH, "gen2", MEASURED_THIS_GENERATION, 0.0),
        _slice_run(MGSM, "gen2", MEASURED_THIS_GENERATION, 0.0),
    )
    unresolved = _run_root(
        tmp_path / "unresolved",
        parent_slices=(),
        ancestor_slices=good_ancestor,
        candidate_slices=regressed,
    )
    assert judge_gen2.judge(unresolved) == 1


def test_a_missing_ancestor_arm_blocks_branch_protection(tmp_path: Path) -> None:
    assert judge_gen2.judge(_run_root(tmp_path, ancestor_arm=False)) == 1


# --------------------------------------------------------------------------
# settlement is the production settlement
# --------------------------------------------------------------------------


def test_the_judge_and_production_settlement_agree(tmp_path: Path) -> None:
    campaign = CampaignManifest.from_file(judge_gen2.CAMPAIGN_MANIFEST)
    for wall in (0.2, 1.1, 1.4, 2.5):
        total = ComputeCost(
            device_gpu_hours=0.4,
            wall_gpu_hours=wall,
            source="test",
            device_measured=False,
        )
        production = settle_campaign(campaign, total=total)
        root = _run_root(tmp_path / f"wall-{wall}", accounting=_accounting(wall=wall))
        judge_ok = judge_gen2.judge(root) == 0
        assert judge_ok == production.compliant, (
            f"wall {wall}: judge certified={judge_ok}, production compliant="
            f"{production.compliant} ({production.failure_reasons})"
        )


def test_a_wall_overrun_is_a_hard_failure_not_an_unknown(tmp_path: Path) -> None:
    root = _run_root(tmp_path, accounting=_accounting(wall=2.5))
    assert judge_gen2.judge(root) == 1


def test_a_device_settlement_ceiling_cannot_be_satisfied_by_an_unmeasured_device(
    tmp_path: Path,
) -> None:
    """The policy amendment says device is admission-only *because*
    device_time_measured=false. Declaring it true restores the hard gate, and
    an unmeasured device figure must then refuse rather than pass."""
    campaign = CampaignManifest.from_file(judge_gen2.CAMPAIGN_MANIFEST)
    assert campaign.budget.device_time_measured is False
    import dataclasses

    hard = dataclasses.replace(
        campaign, budget=dataclasses.replace(campaign.budget, device_time_measured=True)
    )
    unmeasured = ComputeCost.from_wall_only(0.2, source="wall-only ledger")
    verdict = settle_campaign(hard, total=unmeasured)
    assert verdict.compliant is False
    assert any("ACTUAL_DEVICE_GPU_HOURS_UNMEASURED" in reason for reason in verdict.failure_reasons)

    measured = ComputeCost.measured(device_gpu_hours=0.1, wall_gpu_hours=0.2, source="probe")
    assert settle_campaign(hard, total=measured).compliant is True


def test_losing_recipe_accounting_is_required(tmp_path: Path) -> None:
    accounting = _accounting()
    accounting["entries"] = [{"kind": "training", "recipe_id": "gen2-recipe-a"}]
    assert judge_gen2.judge(_run_root(tmp_path, accounting=accounting)) == 1


def test_a_broken_manifest_campaign_is_inconclusive_not_a_crash(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(judge_gen2, "CAMPAIGN_MANIFEST", tmp_path / "absent.json")
    assert judge_gen2.judge(_run_root(tmp_path)) == 1


# --------------------------------------------------------------------------
# artifact identity is recomputed
# --------------------------------------------------------------------------


def test_artifact_digest_is_recomputed(tmp_path: Path) -> None:
    assert judge_gen2.judge(_run_root(tmp_path)) == 0


def test_a_fabricated_digest_refuses(tmp_path: Path) -> None:
    root = _run_root(tmp_path, artifact_digest="a" * 64)
    assert judge_gen2.judge(root) == 1


def test_an_artifact_mutated_after_recording_refuses(tmp_path: Path) -> None:
    root = _run_root(tmp_path)
    chosen = json.loads((root / "chosen_candidate.json").read_text(encoding="utf-8"))
    Path(chosen["artifact_ref"], ARTIFACT_FILE).write_text("tampered", encoding="utf-8")
    assert judge_gen2.judge(root) == 1


def test_a_missing_artifact_refuses(tmp_path: Path) -> None:
    root = _run_root(tmp_path)
    chosen = json.loads((root / "chosen_candidate.json").read_text(encoding="utf-8"))
    import shutil

    shutil.rmtree(chosen["artifact_ref"])
    assert judge_gen2.judge(root) == 1


# --------------------------------------------------------------------------
# the frozen paired/absolute target rule
# --------------------------------------------------------------------------


def test_paired_improvement_passes_the_frozen_rule() -> None:
    status, detail = judge_gen2._target_gate(
        [1.0] * 11 + [0.0] * 5,
        [0.0] * 16,
        lower_is_better=True,
        absolute_threshold=0.125,
    )
    assert status == judge_gen2.PASS
    assert "paired improvement" in detail


def test_aggregate_threshold_alone_does_not_pass_the_frozen_rule() -> None:
    """Rate at the threshold, no paired signal, too few strict improvements."""
    status, detail = judge_gen2._target_gate(
        [1.0, 1.0] + [0.0] * 14,
        [0.0] * 14 + [1.0, 1.0],
        lower_is_better=True,
        absolute_threshold=0.125,
    )
    assert status == judge_gen2.FAIL
    assert "absolute met" in detail
    assert "strictly better on 2/16 (need 12)" in detail


def test_a_tie_fails() -> None:
    status, _detail = judge_gen2._target_gate(
        [1.0] * 5 + [0.0] * 11,
        [1.0] * 5 + [0.0] * 11,
        lower_is_better=True,
        absolute_threshold=0.125,
    )
    assert status == judge_gen2.FAIL


def test_the_absolute_threshold_path_can_pass_without_a_paired_signal() -> None:
    """The prereg's second path: absolute threshold *and* the strict count.

    With tiny per-prompt deltas the paired test has no power (its effect is
    below the declared minimum), so the frozen rule's absolute branch is the
    only thing that can pass -- which is exactly why it is written down.
    """
    parent = [0.9, 0.1] * 8
    candidate = [0.8, 0.0] * 8
    assert compare(parent, candidate, min_effect=judge_gen2.TARGET_MIN_EFFECT).verdict == "flat"
    status, detail = judge_gen2._target_gate(
        parent, candidate, lower_is_better=True, absolute_threshold=0.5
    )
    assert status == judge_gen2.PASS
    assert "absolute threshold + strict count" in detail


# --------------------------------------------------------------------------
# prompt identity alignment
# --------------------------------------------------------------------------


def _reorder_and_relabel(arm_path: Path, *, ids: dict | None = None) -> None:
    report = json.loads(arm_path.read_text(encoding="utf-8"))
    for run in report["runs"]:
        per_prompt = (run.get("metadata") or {}).get("per_prompt")
        if not per_prompt:
            continue
        if ids is not None:
            for entry in per_prompt:
                entry["prompt_id"] = ids[entry["prompt_id"]]
        run["metadata"]["per_prompt"] = list(reversed(per_prompt))
    arm_path.write_text(json.dumps(report), encoding="utf-8")


def test_reordered_but_identity_equivalent_prompts_still_pair(tmp_path: Path) -> None:
    root = _run_root(tmp_path)
    _reorder_and_relabel(root / "parent_evaluation.json")
    assert judge_gen2.judge(root) == 0


def test_a_duplicated_prompt_identity_refuses_to_pair(tmp_path: Path) -> None:
    root = _run_root(tmp_path)
    _reorder_and_relabel(
        root / "parent_evaluation.json",
        ids={f"gen1-diag-{i:02d}": "gen1-diag-00" for i in range(16)},
    )
    assert judge_gen2.judge(root) == 1


def test_a_missing_parent_prompt_refuses_to_pair(tmp_path: Path) -> None:
    root = _run_root(tmp_path)
    report = json.loads((root / "parent_evaluation.json").read_text(encoding="utf-8"))
    for run in report["runs"]:
        per_prompt = (run.get("metadata") or {}).get("per_prompt")
        if per_prompt:
            run["metadata"]["per_prompt"] = per_prompt[:-1]
    (root / "parent_evaluation.json").write_text(json.dumps(report), encoding="utf-8")
    assert judge_gen2.judge(root) == 1


def test_a_claimed_aggregate_cannot_override_the_per_prompt_evidence(tmp_path: Path) -> None:
    """The artifact can claim anything; the judge scores the completions."""
    root = _run_root(
        tmp_path,
        candidate_dup=0,
        candidate_echo=0,
        candidate_instrument_metadata={"declared_duplication_rate": 0.0, "declared_echo_rate": 0.0},
    )
    # Now make the completions the gen1-shaped defect while the claims stay clean.
    report = json.loads((root / "candidate_evaluation.json").read_text(encoding="utf-8"))
    for run in report["runs"]:
        per_prompt = (run.get("metadata") or {}).get("per_prompt")
        if not per_prompt:
            continue
        for entry in per_prompt:
            entry["completion"] = _completion(
                duplicate=True, echo=True, answer=str(entry.get("expected"))
            )
    (root / "candidate_evaluation.json").write_text(json.dumps(report), encoding="utf-8")
    assert judge_gen2.judge(root) == 1


# --------------------------------------------------------------------------
# the frozen instrument list cannot drift from the gen1 driver
# --------------------------------------------------------------------------


def test_the_frozen_instrument_matches_the_gen1_driver_source() -> None:
    """The judge freezes its own copy of the 16 prompts; that copy must equal
    the driver's literals, or the instrument quietly changes."""
    source = (REPO / "docs" / "gen1" / "run_gen1_cycle.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    literals: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    literals[target.id] = ast.literal_eval(node.value)
                except (ValueError, SyntaxError):
                    continue
    prompts = list(literals["DIAG_PROMPTS"])
    expected = list(literals["_DIAG_EXPECTED"])
    assert list(judge_gen2.INSTRUMENT_PROMPTS) == list(zip(prompts, expected))
    assert len(judge_gen2.INSTRUMENT_PROMPTS) == 16
    assert judge_gen2.CONSTRAINED_PROMPTS <= set(prompts)
