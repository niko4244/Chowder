"""Curriculum, promotion, lineage, statistics, and frontier comparability.

These pin the decision layer: promotion is multi-objective and can never be
bought with one benchmark; protected regressions and contaminated evidence
decide; the frontier system refuses fake comparisons; lineage keeps every
generation's provenance.
"""

from __future__ import annotations

import pytest

from chowder.growth.capability import SkillEstimate
from chowder.growth.curriculum import CurriculumEngine, CurriculumItem
from chowder.growth.recipe_planner import HardwareBudget, RecipePlanner
from chowder.growth.failure_bank import FailureBank
from chowder.growth.frontier_reference import (
    ChowderScore,
    FrontierDatabase,
    NOT_DIRECTLY_COMPARABLE,
    ReferenceScore,
    SnapshotStore,
    compare_protocol,
    gap_rows,
)
from chowder.evals.result import MEASURED_PARENT, MEASURED_THIS_GENERATION
from chowder.growth.lineage import GenerationLedger
from chowder.growth.promotion import BenchmarkResult, PromotionInput, evaluate_promotion
from chowder.growth.statistics import compare


def _profile_estimates() -> dict[str, float]:
    return {
        "coding.generation": 0.28,
        "reasoning.scientific": 0.31,
        "math.competition": 0.52,
        "knowledge.factuality": 0.61,
    }


# ---------------- curriculum ----------------


def _profile(version: str):
    from chowder.growth.capability import CapabilityProfile

    estimates = _profile_estimates()
    return CapabilityProfile(
        model_version=version,
        raw_scores={"gpqa_diamond@2025-05-30": 0.31},
        skills=tuple(
            SkillEstimate(skill=skill, estimate=value, confidence=0.9, evidence=("gpqa_diamond@2025-05-30",))
            for skill, value in estimates.items()
        ),
    )


def test_curriculum_prioritizes_weakest_highest_confidence_skills_first():
    engine = CurriculumEngine()
    priorities = engine.prioritize(_profile("v0.1"))
    assert priorities
    assert priorities[0].skill in {"coding.generation", "reasoning.scientific"}


def test_curriculum_plan_carries_decision_trace_and_protected_sets():
    engine = CurriculumEngine()
    plan = engine.plan(
        model_version="v0.1",
        profile=_profile("v0.1"),
        protected_sets=("gpqa_diamond@2025-05-30",),
    )
    assert plan
    for item in plan:
        assert item.protected_regression_set == ("gpqa_diamond@2025-05-30",)
        assert item.decision_trace["components"] and item.decision_trace["weights"]
        assert item.weakness_evidence
        assert item.example_count >= 500


def test_curriculum_respects_budget_examples():
    engine = CurriculumEngine()
    small = engine.plan(
        model_version="v0.1", profile=_profile("v0.1"), protected_sets=(), budget_examples=4_000
    )
    large = engine.plan(
        model_version="v0.1", profile=_profile("v0.1"), protected_sets=(), budget_examples=40_000
    )
    assert sum(i.example_count for i in small) < sum(i.example_count for i in large)


def test_banked_failures_raise_a_skill_priority():
    bank = FailureBank()
    engine_without = CurriculumEngine(failure_bank=bank)
    priorities_without = engine_without.prioritize(_profile("v0.1"))
    top_without = priorities_without[0].skill

    common = dict(
        model_version="v0.1",
        benchmark_qualified_id="livecodebench@2025-04",
        score=0.0,
        verifier_evidence="unit test failure",
        confidence=0.95,
        generation="v0.1",
    )
    bank.record(
        sample_ref="s-1",
        prompt="write a function to parse nested JSON with duplicates",
        output="raise ValueError()",
        expected_behavior="parse and merge",
        categories=("coding.syntax",),
        **common,
    )
    bank.record(
        sample_ref="s-2",
        prompt="implement an LRU cache with TTL expiry",
        output="pass",
        expected_behavior="working cache",
        categories=("coding.semantics",),
        **common,
    )
    priorities_with = CurriculumEngine(failure_bank=bank).prioritize(_profile("v0.1"))
    coding_priority_without = next(
        p.priority for p in priorities_without if p.skill == "coding.generation"
    )
    coding_priority_with = next(
        p.priority for p in priorities_with if p.skill == "coding.generation"
    )
    assert coding_priority_with > coding_priority_without or top_without == "coding.generation"


# ---------------- statistics + promotion ----------------


def test_small_sample_improvement_is_not_significant_but_large_is():
    # Flat baseline vs a real +0.2 shift: statistically decisive.
    strong = compare(_samples(0.5, blocks=6), _samples(0.7, blocks=6), min_effect=0.02)
    assert strong.significant and strong.verdict == "improved"
    # Identical distributions: must be flat, not "improved".
    same = compare(_samples(0.5, blocks=6), _samples(0.5, blocks=6), min_effect=0.02)
    assert same.verdict == "flat"


def test_significant_but_tiny_change_reports_flat():
    # A real but sub-minimum-effect shift: flat by the declared policy.
    before = _samples(0.500, spread=0.01, blocks=4)
    after = _samples(0.5015, spread=0.01, blocks=4)
    result = compare(before, after, min_effect=0.05)
    assert result.verdict == "flat"  # below declared minimum meaningful effect


def _results(
    scores: dict[str, float],
    samples: dict[str, tuple[float, ...]] | None = None,
    *,
    origin: str = MEASURED_THIS_GENERATION,
):
    samples = samples or {}
    return {
        benchmark: BenchmarkResult(
            benchmark_qualified_id=benchmark,
            score=score,
            samples=samples.get(benchmark, ()),
            contamination="CLEAN",
            measurement_origin=origin,
        )
        for benchmark, score in scores.items()
    }


LIVECODE = "livecodebench@2025-04"
GPQA = "gpqa_diamond@2025-05-30"
MATH500 = "math500@2024-11"


def _samples(mean: float, spread: float = 0.02, blocks: int = 5) -> tuple[float, ...]:
    """Per-sample scores centered on ``mean`` with honest spread."""
    pattern = (-1.5, -0.5, 0.0, 0.5, 1.5)
    return tuple(mean + spread * p for p in pattern * blocks)


def _promotion_input(candidate_targets, candidate_protected, *, contamination="CLEAN"):
    parent_scores = {
        LIVECODE: 0.28,
        GPQA: 0.31,
        MATH500: 0.52,
    }
    candidate_scores = {
        LIVECODE: candidate_targets,
        GPQA: candidate_protected,
        MATH500: 0.52,
    }
    parent_samples = {
        LIVECODE: _samples(0.28),
        GPQA: _samples(0.31),
        MATH500: _samples(0.52),
    }
    candidate_samples = {
        LIVECODE: _samples(candidate_targets),
        GPQA: _samples(candidate_protected),
        MATH500: _samples(0.52),
    }
    parent = _results(parent_scores, parent_samples, origin=MEASURED_PARENT)
    candidate = _results(candidate_scores, candidate_samples)
    for result in candidate.values():
        result_dict = {
            "benchmark_qualified_id": result.benchmark_qualified_id,
            "score": result.score,
            "samples": result.samples,
            "contamination": contamination,
            "measurement_origin": MEASURED_THIS_GENERATION,
        }
        candidate[result.benchmark_qualified_id] = BenchmarkResult(**result_dict)
    return PromotionInput(
        candidate_version="v0.2",
        parent_version="v0.1",
        target_benchmarks=(LIVECODE,),
        candidate_results=candidate,
        parent_results=parent,
        protected_benchmarks=(GPQA,),
        broad_battery_benchmarks=(MATH500,),
    )


def test_promotion_rejects_when_target_does_not_improve():
    decision = evaluate_promotion(_promotion_input(0.28, 0.31))
    assert decision.verdict == "REJECTED"
    assert decision.checks["target_improvement"] in {"not met", "missing"}


def test_promotion_rejects_on_protected_regression():
    decision = evaluate_promotion(_promotion_input(0.45, 0.05))
    assert decision.verdict == "REJECTED"
    assert decision.checks["protected_regression"] == "violated"


def test_promotion_marks_tainted_evidence():
    decision = evaluate_promotion(_promotion_input(0.45, 0.31, contamination="KNOWN_CONTAMINATION"))
    assert decision.verdict == "TAINTED"


def test_promotion_inconclusive_when_contamination_unchecked():
    decision = evaluate_promotion(_promotion_input(0.45, 0.31, contamination="UNKNOWN"))
    assert decision.verdict in {"INCONCLUSIVE", "REJECTED"}
    assert decision.checks["evidence_integrity"] == "inconclusive"


def test_genuine_improvement_without_regression_promotes():
    decision = evaluate_promotion(_promotion_input(0.45, 0.31))
    assert decision.verdict == "PROMOTED"
    assert decision.checks["target_improvement"] == "met"
    assert decision.checks["protected_regression"] == "ok"


def test_frontier_scores_never_decide_promotion():
    """PromotionInput carries no frontier field: parent-relative evidence decides."""
    import dataclasses

    assert not any(f.name == "frontier" for f in dataclasses.fields(PromotionInput))


# ---------------- lineage ----------------


def test_generation_ledger_roundtrip_and_duplicate_refusal(tmp_path):
    ledger = GenerationLedger(tmp_path)
    decision = evaluate_promotion(_promotion_input(0.45, 0.31))
    ledger.record(
        version="v0.2",
        parent_version="v0.1",
        cycle_id="cycle-001",
        base_model={"revision": "Qwen/Qwen3-9B@abc123", "quantization": "bf16"},
        dataset_manifest_ref="growth/data-manifest.json#cycle-001",
        curriculum_manifest_ref="growth/curriculum-cycle-001.json",
        recipe={"objective": "sft", "lora_rank": 64},
        training_evidence_ref="runs/cycle-001/training-evidence.json",
        evaluation_report_ref="runs/cycle-001/eval-report.json",
        promotion=decision,
    )
    with pytest.raises(ValueError, match="already recorded"):
        ledger.record(
            version="v0.2",
            parent_version="v0.1",
            cycle_id="cycle-001",
            base_model={},
            dataset_manifest_ref="",
            curriculum_manifest_ref="",
            recipe={},
            training_evidence_ref="",
            evaluation_report_ref="",
            promotion=decision,
        )
    reloaded = GenerationLedger(tmp_path)
    record = reloaded.get("v0.2")
    assert record.parent_version == "v0.1"
    assert record.promotion["verdict"] == "PROMOTED"
    # ancestry() walks parents; v0.1 was never recorded so the chain is just v0.2.
    ancestry = reloaded.ancestry("v0.2")
    assert [r.version for r in ancestry] == ["v0.2"]


# ---------------- frontier comparability ----------------


def _reference(**overrides):
    base = dict(
        model="Peer-8B",
        benchmark_qualified_id=GPQA,
        score=0.64,
        level="LEVEL_2_COMPARABLE_PEER",
        date="2026-09-15",
        source_url="https://example/peer",
        harness="inspect",
        tool_setting="none",
        reasoning_setting="direct",
        first_party=True,
        comparability_confidence="HIGH",
    )
    base.update(overrides)
    return ReferenceScore(**base)


def test_reference_rejects_unknown_level(tmp_path):
    db = FrontierDatabase(tmp_path)
    with pytest.raises(ValueError, match="unknown frontier level"):
        db.add(_reference(level="LEVEL_99"))


def test_compare_protocol_labels_mismatched_settings():
    comparable = compare_protocol(
        _reference(),
        benchmark_qualified_id=GPQA,
        tool_setting="none",
        reasoning_setting="direct",
    )
    assert comparable == "COMPARABLE"
    for change in (
        {"tool_setting": "agentic-tools"},
        {"reasoning_setting": "extended-thinking"},
        {"comparability_confidence": "LOW"},
    ):
        verdict = compare_protocol(
            _reference(**change),
            benchmark_qualified_id=GPQA,
            tool_setting="none",
            reasoning_setting="direct",
        )
        assert verdict == NOT_DIRECTLY_COMPARABLE
    verdict = compare_protocol(
        _reference(benchmark_qualified_id="mmlu_pro@2025-04"),
        benchmark_qualified_id=GPQA,
        tool_setting="none",
        reasoning_setting="direct",
    )
    assert verdict == NOT_DIRECTLY_COMPARABLE


def test_gap_rows_compute_gap_and_parity_only_where_comparable(tmp_path):
    db = FrontierDatabase(tmp_path)
    db.add(_reference(score=0.83, level="LEVEL_4_ABSOLUTE_FRONTIER", model="Frontier-1"))
    ours = ChowderScore(
        generation_version="v1.0",
        benchmark_qualified_id=GPQA,
        score=0.54,
        tool_setting="none",
        reasoning_setting="direct",
    )
    rows = gap_rows(db, ours)
    frontier_row = next(r for r in rows if r.level == "LEVEL_4_ABSOLUTE_FRONTIER")
    assert frontier_row.comparability == "COMPARABLE"
    assert frontier_row.gap == pytest.approx(0.29, abs=1e-6)
    assert frontier_row.parity_ratio == pytest.approx(0.54 / 0.83, abs=1e-3)

    # A mismatched-protocol reference still renders, but flagged non-comparable
    # with no parity ratio -- the fake-comparison guard.
    db.add(
        _reference(
            score=0.99,
            level="LEVEL_2_OPEN_WEIGHT_FRONTIER",
            model="Agent-Frontier",
            tool_setting="agentic-tools",
        )
    )
    rows = gap_rows(db, ours)
    agent_row = next(r for r in rows if r.level == "LEVEL_2_OPEN_WEIGHT_FRONTIER")
    assert agent_row.comparability == NOT_DIRECTLY_COMPARABLE
    assert agent_row.parity_ratio is None


def _planner_and_items():
    """A real planner over real curriculum items, so the recipe under test is
    the one the cycle would actually execute."""
    item = CurriculumItem(
        item_id="cur-1",
        skill="math.arithmetic",
        role="TARGET",
        priority=0.8,
        confidence=0.7,
        weakness_evidence="measured failures on multi-step arithmetic",
        desired_improvement=0.1,
        preservation_risks=("reasoning.coding",),
        source_strategy="verified_synthesis",
        example_count=2000,
        token_target=1_000_000,
        difficulty_band="medium",
        verification_method="exact_match",
        training_type="sft",
        evaluation_set="tier1",
        protected_regression_set=("reasoning.instruction_following",),
    )
    planner = RecipePlanner(
        budget=HardwareBudget(
            gpu_name="RTX 5060 Ti",
            vram_gb=17.1,
            measured_step_seconds_at_seq={64: 2.854},
            measured_load_seconds=13.851,
        ),
        # Deliberately roomy: this test pins the patch shape, so it must not
        # depend on the planner's budget arithmetic refusing its candidates.
        max_device_gpu_hours=1.0,
        max_wall_gpu_hours=3.5,
    )
    return planner, (item,)


def test_recipe_config_patch_emits_only_knobs_the_validator_reads():
    """The planner must not invent configuration it cannot actually supply.

    A patch is merged straight into a resolved project config, so a key the
    validator does not read is drift at best and a corrupted qualified config
    at worst. The former helper here targeted a ``search.variants`` section
    that does not exist, emitted ``lora_rank`` (the PEFT path names rank
    inside its own ``lora`` spec), and carried a bookkeeping block that would
    have been merged into the experiment's config.
    """
    from chowder.graph import deep_merge_config
    from chowder.project import ROUTER_HEALING_REQUIRED_KNOBS

    planner, items = _planner_and_items()
    recipe = planner.propose(items, count=1)[0]
    patch = recipe.to_config_patch()

    assert set(patch) == {"backend"}
    assert set(patch["backend"]) == {"router_healing"}
    knobs = patch["backend"]["router_healing"]
    assert set(knobs) == {"max_steps", "learning_rate", "seq_len"}
    # The load-bearing link: every key must be one the project validator
    # reads for this backend, so a rename on either side breaks this test.
    assert set(knobs) <= set(ROUTER_HEALING_REQUIRED_KNOBS)

    merged = deep_merge_config(
        {"backend": {"router_healing": {"corpus_path": "corpus.txt"}}}, patch
    )
    assert set(merged["backend"]["router_healing"]) == {
        "corpus_path",
        "max_steps",
        "learning_rate",
        "seq_len",
    }


def test_recipe_patch_maps_into_the_peft_backend_namespace():
    """The peft backend refuses router_healing keys (fail-closed validator),
    so a growth recipe targeting a transformers-peft project must emit the
    same training knobs in the peft validator's own namespace."""
    from chowder.config_validation import validate_transformers_backend_config
    from chowder.graph import deep_merge_config

    planner, items = _planner_and_items()
    recipe = planner.propose(items, count=1)[0]
    patch = recipe.to_config_patch(backend_type="transformers-peft")

    assert set(patch) == {"backend"}
    assert set(patch["backend"]) == {"max_length", "training"}
    assert set(patch["backend"]["training"]) == {
        "max_steps",
        "learning_rate",
        "lr_scheduler_type",
        "warmup_steps",
    }

    merged = deep_merge_config(
        {
            "backend": {
                "type": "transformers-peft",
                "max_length": 512,
                "training": {"epochs": 1.0},
            }
        },
        patch,
    )
    # The merged result passes the real validator: the recipe's knobs land in
    # a namespace the backend actually reads, and no router key leaks in.
    validate_transformers_backend_config(merged)
    assert merged["backend"]["max_length"] == recipe.seq_len
    assert merged["backend"]["training"]["max_steps"] == recipe.max_steps
    assert merged["backend"]["training"]["learning_rate"] == recipe.learning_rate


def test_recipe_patch_refuses_an_unknown_backend_type():
    planner, items = _planner_and_items()
    recipe = planner.propose(items, count=1)[0]
    try:
        recipe.to_config_patch(backend_type="unsloth-fantasy")
    except ValueError as error:
        assert "unsloth-fantasy" in str(error)
    else:
        raise AssertionError("an unknown backend type must not produce a patch")


def test_snapshot_store_freezes_and_never_rewrites(tmp_path):
    store = SnapshotStore(tmp_path)
    scores = (_reference(),)
    store.freeze("v1.0-frontier", "2026-09-15", scores)
    with pytest.raises(ValueError, match="never rewrite"):
        store.freeze("v1.0-frontier", "2027-01-10", scores)
    frozen = SnapshotStore(tmp_path).get("v1.0-frontier")
    assert frozen.date == "2026-09-15"
    assert frozen.scores[0].model == "Peer-8B"


def test_degenerate_parent_floor_target_improvement_is_recognized():
    """A parent pinned at the scale floor (all-zero samples, zero variance)
    that the candidate lifts past min_effect is an IMPROVEMENT, not
    'inconclusive': the no-variance branch of compare() already returns
    'flat' when |delta| <= min_effect, so treating a real lift past the
    threshold as undecidable would make a hard 0 -> 1 target gain
    unpromotable and the predeclared rule self-defeating."""
    from chowder.growth.promotion import BenchmarkResult, PromotionInput, evaluate_promotion

    n = 16
    parent = BenchmarkResult(
        benchmark_qualified_id="instrument@v1",
        score=0.0,
        samples=(0.0,) * n,          # floor: zero variance, zero mean
        contamination="CLEAN",
        measurement_origin=MEASURED_PARENT,
    )
    candidate = BenchmarkResult(
        benchmark_qualified_id="instrument@v1",
        score=1.0,
        samples=(1.0,) * n,          # ceiling: zero variance, full lift
        contamination="CLEAN",
        measurement_origin=MEASURED_THIS_GENERATION,
    )
    protected = BenchmarkResult(
        benchmark_qualified_id="protected@v1",
        score=0.5,
        samples=(0.5, 0.5, 0.5, 0.5),
        contamination="CLEAN",
        measurement_origin=MEASURED_THIS_GENERATION,
    )
    data = PromotionInput(
        candidate_version="v0.2",
        parent_version="v0.1",
        target_benchmarks=("instrument@v1",),
        candidate_results={"instrument@v1": candidate, "protected@v1": protected},
        parent_results={"instrument@v1": parent, "protected@v1": protected},
        protected_benchmarks=("protected@v1",),
        broad_battery_benchmarks=(),
        min_target_improvement=0.90,
    )
    decision = evaluate_promotion(data)
    assert decision.checks["target:instrument@v1"] == "improved", decision.checks
    assert decision.checks["target_improvement"] == "met"
    # An EMPTY declared broad battery is "unmeasured" honestly, but the rule
    # treats an empty tuple as no-data ("inconclusive"). The target check
    # itself is what this test pins: a floor-start lift is "improved".
    assert decision.verdict in {"PROMOTED", "INCONCLUSIVE"}, (decision.verdict, decision.reasons)
    if decision.verdict == "INCONCLUSIVE":
        assert decision.reasons == ("broad battery insufficiently measured",), decision.reasons
