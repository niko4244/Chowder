"""Tests for the 0.5 intervention architecture's three data layers:

families + maturity gating (isolation), the evidence store (scoped,
tamper-evident memory with priors), and hypothesis generation (candidates
must be experiments). All offline and deterministic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chowder.growth.evidence import (
    EvidenceRecord,
    EvidenceState,
    EvidenceStore,
    prior_for_family,
)
from chowder.growth.hypotheses import (
    HypothesisRefusal,
    Observation,
    generate_hypotheses,
    hypothesis_candidate_brief,
)
from chowder.growth.interventions import (
    InterventionFamily,
    InterventionFamilyRefusal,
    Maturity,
    families_for_campaign,
    family_from_id,
    family_registry,
    register_family,
)


def _record(**overrides) -> EvidenceRecord:
    fields: dict[str, object] = {
        "record_id": "rec-001",
        "state": EvidenceState.FAILED,
        "family_id": "training.sft-curriculum",
        "model_family": "qwen3.8",
        "checkpoint_identity": "gen2@abc123",
        "architecture": "dense",
        "eval_suite": "gen2-screening@v1",
        "intervention_parameters": {"learning_rate": 2e-4},
        "software_runtime": {"transformers": "5.18.0", "torch": "2.14.1"},
        "hardware_class": "kaggle_2x_t4_16gb",
        "sample_size": 3,
        "evidence_quality": "paired-seeds",
        "measured_effect": {"target_metric": 0.08},
        "date": "2026-10-03",
        "source_run": "kernel-20261003-203018",
    }
    fields.update(overrides)
    return EvidenceRecord(**fields)


# --------------------------------------------------------------------------
# families and maturity isolation (Phases 4 + 13)
# --------------------------------------------------------------------------


def test_the_registry_declares_the_contract_per_family() -> None:
    for family in family_registry():
        assert family.target_failure_class
        assert family.eval_dimensions, f"{family.family_id} declares no eval dimensions"
        assert family.parameters, f"{family.family_id} declares no parameter space"
        if family.maturity is Maturity.RESEARCH:
            assert family.basis, f"{family.family_id}: research without a basis"


def test_an_unknown_family_refuses_rather_than_being_invented() -> None:
    with pytest.raises(InterventionFamilyRefusal, match="no intervention family"):
        family_from_id("architecture.does-not-exist")


def test_a_rejected_family_requires_a_recorded_basis() -> None:
    with pytest.raises(InterventionFamilyRefusal, match="without a recorded basis"):
        InterventionFamily(
            family_id="compression.bad-idea",
            name="Bad idea",
            target_failure_class="vram-footprint",
            parameters={"x": {"type": "int", "range": [0, 1]}},
            maturity=Maturity.REJECTED,
        )


def test_default_campaign_policy_permits_only_production_families() -> None:
    permitted = {f.family_id for f in families_for_campaign({})}
    assert "training.sft-curriculum" in permitted
    assert "architecture.conditional-ffn" not in permitted
    assert "compression.ptq" not in permitted


def test_experimental_families_need_the_campaign_to_say_so() -> None:
    policy = {"experimental_interventions": True}
    permitted = {f.family_id for f in families_for_campaign(policy)}
    # research still refuses without an explicitly research campaign
    assert "compression.ptq" not in permitted
    research_policy = {"experimental_interventions": True, "research_campaign": True}
    permitted = {f.family_id for f in families_for_campaign(research_policy)}
    assert "compression.ptq" in permitted


def test_a_rejected_family_only_returns_through_an_explicit_reopen(tmp_path) -> None:
    rejected = InterventionFamily(
        family_id="compression.test-rejected",
        name="Measured and rejected",
        target_failure_class="vram-footprint",
        parameters={"rank": {"type": "int", "range": [1, 2]}},
        maturity=Maturity.REJECTED,
        basis=("a measured rejection lives here",),
    )
    try:
        register_family(rejected)
        assert all(
            f.family_id != "compression.test-rejected"
            for f in families_for_campaign({})
        )
        reopened = {
            f.family_id for f in families_for_campaign({"reopen": {"compression.test-rejected": "hyp-999-new-cause"}})
        }
        assert "compression.test-rejected" in reopened
        # a reopen without a hypothesis id is not a reopen
        empty = {
            f.family_id
            for f in families_for_campaign({"reopen": {"compression.test-rejected": ""}})
        }
        assert "compression.test-rejected" not in empty
    finally:
        from chowder.growth import interventions as interventions_module

        interventions_module._EXTRA_FAMILIES.clear()


# --------------------------------------------------------------------------
# the evidence store (Phase 5)
# --------------------------------------------------------------------------


def test_records_are_scoped_and_priors_are_scope_bound(tmp_path) -> None:
    store = EvidenceStore(path=tmp_path / "evidence.jsonl")
    store.record(_record(state=EvidenceState.FAILED, model_family="qwen3.8"))
    store.record(_record(record_id="rec-002", state=EvidenceState.PROMISING, model_family="spark-2.5"))

    in_scope = prior_for_family(
        store,
        family_id="training.sft-curriculum",
        model_family="qwen3.8",
        architecture="dense",
    )
    other_model_own_history = prior_for_family(
        store,
        family_id="training.sft-curriculum",
        model_family="spark-2.5",
        architecture="dense",
    )
    untested = prior_for_family(
        store,
        family_id="training.adapter-continuation",
        model_family="qwen3.8",
        architecture="dense",
    )
    assert in_scope.multiplier < 1.0, "a measured failure in scope must shrink the prior"
    # One model's failure is not another model's truth: spark's own promising
    # history boosts IT, and qwen's failure does not touch spark's prior.
    assert other_model_own_history.multiplier > 1.0
    assert untested.multiplier == 1.0, "no in-scope history means an exploration candidate"
    assert any("exploration" in reason for reason in untested.reasons)


def test_a_fresh_store_marks_everything_an_exploration_candidate(tmp_path) -> None:
    store = EvidenceStore(path=tmp_path / "evidence.jsonl")
    prior = prior_for_family(
        store,
        family_id="inference.speculative",
        model_family="qwen3.8",
        architecture="dense",
    )
    assert prior.multiplier == 1.0
    assert not prior.excluded


def test_architecture_incompatibility_excludes_the_family_in_scope(tmp_path) -> None:
    store = EvidenceStore(path=tmp_path / "evidence.jsonl")
    store.record(
        _record(
            state=EvidenceState.ARCHITECTURE_INCOMPATIBLE,
            family_id="architecture.conditional-ffn",
            architecture="dense",
            measured_effect={},
        )
    )
    prior = prior_for_family(
        store,
        family_id="architecture.conditional-ffn",
        model_family="qwen3.8",
        architecture="dense",
    )
    assert prior.excluded, "an incompatible family cannot apply here at all"
    # ...but a different architecture in the same model family is untested, not excluded
    other = prior_for_family(
        store,
        family_id="architecture.conditional-ffn",
        model_family="qwen3.8",
        architecture="moe",
    )
    assert not other.excluded


def test_repeated_failures_shrink_the_prior_harder_than_one(tmp_path) -> None:
    store = EvidenceStore(path=tmp_path / "evidence.jsonl")
    for index in range(3):
        store.record(
            _record(
                record_id=f"rec-fail-{index}",
                state=EvidenceState.FAILED,
                model_family="qwen3.8",
            )
        )
    prior = prior_for_family(
        store,
        family_id="training.sft-curriculum",
        model_family="qwen3.8",
        architecture="dense",
    )
    single = EvidenceStore(path=tmp_path / "one.jsonl")
    single.record(_record(model_family="qwen3.8"))
    one = prior_for_family(
        single,
        family_id="training.sft-curriculum",
        model_family="qwen3.8",
        architecture="dense",
    )
    assert prior.multiplier < one.multiplier < 1.0


def test_a_promising_record_earns_a_modest_boost_and_never_automatic_promotion(
    tmp_path,
) -> None:
    store = EvidenceStore(path=tmp_path / "evidence.jsonl")
    store.record(_record(state=EvidenceState.PROMISING, measured_effect={"target": 0.2}))
    prior = prior_for_family(
        store,
        family_id="training.sft-curriculum",
        model_family="qwen3.8",
        architecture="dense",
    )
    assert 1.0 < prior.multiplier <= 1.5, "a prior boosts exploration, never the gate"


def test_mutating_a_recorded_outcome_breaks_the_audit(tmp_path) -> None:
    """The tamper test: a recorded outcome cannot be quietly rewritten."""
    path = tmp_path / "evidence.jsonl"
    store = EvidenceStore(path=path)
    store.record(_record(state=EvidenceState.FAILED, measured_effect={"target_metric": 0.08}))
    store.record(_record(record_id="rec-002", state=EvidenceState.PROMISING))

    lines = path.read_text(encoding="utf-8").splitlines()
    mutated = json.loads(lines[0])
    mutated["measured_effect"]["target_metric"] = -0.99  # rewrite history
    mutated["state"] = EvidenceState.PROMISING.value
    lines[0] = json.dumps(mutated, sort_keys=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash chain"):
        EvidenceStore(path=path)
    fresh = EvidenceStore(path=tmp_path / "other.jsonl")
    with pytest.raises(ValueError, match="hash chain"):
        fresh.audit if False else EvidenceStore(path=path)


def test_removing_a_recorded_outcome_breaks_the_audit(tmp_path) -> None:
    path = tmp_path / "evidence.jsonl"
    store = EvidenceStore(path=path)
    store.record(_record())
    store.record(_record(record_id="rec-002"))
    lines = path.read_text(encoding="utf-8").splitlines()
    del lines[0]  # remove one attempt from the memory
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash chain"):
        EvidenceStore(path=path)


def test_duplicate_record_ids_are_refused_append_only(tmp_path) -> None:
    store = EvidenceStore(path=tmp_path / "evidence.jsonl")
    store.record(_record())
    with pytest.raises(ValueError, match="append-only"):
        store.record(_record())


# --------------------------------------------------------------------------
# hypothesis generation (Phase 6)
# --------------------------------------------------------------------------


def _observation(**overrides) -> Observation:
    fields: dict[str, object] = {
        "metric": "arithmetic",
        "value": 0.42,
        "threshold": 0.60,
        "evidence_ref": "gen2-capability-profile@v3",
        "direction": "max",  # accuracy-shaped: weakness means value < threshold
    }
    fields.update(overrides)
    return Observation(**fields)


def test_a_candidate_must_cite_a_measured_weakness(tmp_path) -> None:
    store = EvidenceStore(path=tmp_path / "evidence.jsonl")
    # A healthy observation generates nothing: no weakness, no hypothesis.
    healthy = [_observation(value=0.95)]
    assert generate_hypotheses(
        healthy,
        evidence_store=store,
        model_family="qwen3.8",
        architecture="dense",
        campaign_policy={},
        family_ids=("training.sft-curriculum",),
    ) == ()


def test_every_hypothesis_names_observation_cause_prediction_and_falsifier(
    tmp_path,
) -> None:
    store = EvidenceStore(path=tmp_path / "evidence.jsonl")
    hypotheses = generate_hypotheses(
        [_observation()],
        evidence_store=store,
        model_family="qwen3.8",
        architecture="dense",
        campaign_policy={},
        family_ids=("training.sft-curriculum",),
    )
    assert len(hypotheses) == 1
    hypothesis = hypotheses[0]
    assert hypothesis.observation.metric == "arithmetic"
    assert hypothesis.observation.evidence_ref
    assert hypothesis.suspected_cause
    assert hypothesis.predicted_improvement
    assert hypothesis.predicted_risks
    assert hypothesis.required_measurements
    assert "falsif" in hypothesis.falsification_criterion.lower() or "regress" in hypothesis.falsification_criterion.lower()
    # The candidate brief carries the preregistration into the campaign.
    family = family_from_id(hypothesis.family_id)
    brief = hypothesis_candidate_brief(hypothesis, family=family)
    assert brief["hypothesis_id"] == hypothesis.hypothesis_id
    assert brief["falsification_criterion"] == hypothesis.falsification_criterion


def test_an_excluded_family_generates_no_hypothesis(tmp_path) -> None:
    store = EvidenceStore(path=tmp_path / "evidence.jsonl")
    store.record(
        _record(
            state=EvidenceState.ARCHITECTURE_INCOMPATIBLE,
            family_id="architecture.conditional-ffn",
            architecture="dense",
        )
    )
    store.record(
        _record(
            record_id="rec-002",
            state=EvidenceState.FAILED,
            family_id="training.replay-balanced",
            model_family="qwen3.8",
        )
    )
    policy = {"experimental_interventions": True, "research_campaign": True}
    hypotheses = generate_hypotheses(
        [_observation()],
        evidence_store=store,
        model_family="qwen3.8",
        architecture="dense",
        campaign_policy=policy,
        family_ids=(
            "architecture.conditional-ffn",
            "training.replay-balanced",
            "training.sft-curriculum",
        ),
    )
    proposed = {h.family_id for h in hypotheses}
    assert "architecture.conditional-ffn" not in proposed, (
        "an architecture-incompatible family is excluded, not merely discouraged"
    )
    assert "training.sft-curriculum" in proposed


def test_a_research_family_needs_a_research_campaign_policy(tmp_path) -> None:
    store = EvidenceStore(path=tmp_path / "evidence.jsonl")
    # A VRAM observation is the one a compression family is declared for
    # (headroom is higher-is-better: 0.4 GB against a 1.0 GB line is weak).
    vram_observation = _observation(metric="vram_headroom_gb", value=0.4, threshold=1.0, direction="max")
    with_policy = generate_hypotheses(
        [vram_observation],
        evidence_store=store,
        model_family="qwen3.8",
        architecture="dense",
        campaign_policy={"experimental_interventions": True, "research_campaign": True},
        family_ids=("compression.ptq",),
    )
    without_policy = generate_hypotheses(
        [vram_observation],
        evidence_store=store,
        model_family="qwen3.8",
        architecture="dense",
        campaign_policy={},
        family_ids=("compression.ptq",),
    )
    assert len(with_policy) == 1
    assert without_policy == (), "research isolation is the default, not the exception"


def test_a_brief_refuses_a_family_the_hypothesis_is_not_about(tmp_path) -> None:
    store = EvidenceStore(path=tmp_path / "evidence.jsonl")
    hypotheses = generate_hypotheses(
        [_observation()],
        evidence_store=store,
        model_family="qwen3.8",
        architecture="dense",
        campaign_policy={},
        family_ids=("training.sft-curriculum",),
    )
    with pytest.raises(HypothesisRefusal, match="proposes"):
        hypothesis_candidate_brief(
            hypotheses[0], family=family_from_id("training.replay-balanced")
        )
