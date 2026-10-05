"""The runnability gate: no smoke record, no proposal.

``tests/test_growth_family_smoke_matrix.py`` records that every registered
family's cheapest declared mechanism was invoked at least once. This file
makes that record load-bearing: ``families_for_campaign`` and
``generate_hypotheses`` refuse a family whose row is missing, skipped, stale,
or points at an artifact the family no longer declares -- so runnability gates
proposals instead of documenting them.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from chowder.growth.evidence import EvidenceStore
from chowder.growth.hypotheses import Observation, generate_hypotheses
from chowder.growth.interventions import (
    FAMILY_SMOKE_RECORD_PATH,
    FamilySmokeRecord,
    InterventionFamily,
    InterventionFamilyRefusal,
    Maturity,
    assert_smoke_permits,
    families_for_campaign,
    family_from_id,
    family_registry,
    family_smoke_declaration_digest,
    load_family_smoke_records,
    register_family,
)

RESEARCH_POLICY = {"experimental_interventions": True, "research_campaign": True}


def _fresh_record(family_id: str) -> FamilySmokeRecord:
    family = family_from_id(family_id)
    return FamilySmokeRecord(
        family_id=family_id,
        artifact=family.implementation[0],
        mechanism="test smoke",
        status="ran",
        outcome="invoked",
        declaration_digest=family_smoke_declaration_digest(family),
    )


def _observation() -> Observation:
    return Observation(
        metric="arithmetic",
        value=0.42,
        threshold=0.60,
        evidence_ref="gen2-capability-profile@v3",
        direction="max",
    )


def _hypotheses_for(
    store: EvidenceStore, family_id: str, records: dict[str, FamilySmokeRecord]
):
    return generate_hypotheses(
        [_observation()],
        evidence_store=store,
        model_family="qwen3.8",
        architecture="dense",
        campaign_policy={},
        family_ids=(family_id,),
        smoke_records=records,
    )


def _extra_family(family_id: str) -> InterventionFamily:
    return InterventionFamily(
        family_id=family_id,
        name="Operator-declared test family",
        target_failure_class="target-capability-weakness",
        parameters={"step": {"type": "int", "range": [1, 2]}},
        implementation=("src/chowder/growth/interventions.py",),
        maturity=Maturity.PRODUCTION,
        basis=("declared by the runnability-gate tests",),
    )


def _clear_extras() -> None:
    from chowder.growth import interventions as interventions_module

    interventions_module._EXTRA_FAMILIES.clear()
    interventions_module._EXTRA_SMOKE.clear()


def test_a_missing_smoke_record_refuses_every_family() -> None:
    assert families_for_campaign({}, smoke_records={}) == ()
    assert families_for_campaign(RESEARCH_POLICY, smoke_records={}) == ()


def test_a_missing_smoke_record_generates_no_hypothesis(tmp_path) -> None:
    store = EvidenceStore(path=tmp_path / "evidence.jsonl")
    assert _hypotheses_for(store, "training.sft-curriculum", {}) == ()


def test_a_skipped_smoke_row_refuses_even_with_a_binding_digest() -> None:
    record = replace(
        _fresh_record("training.sft-curriculum"),
        status="skipped",
        outcome="torch was absent",
    )
    assert (
        families_for_campaign(
            {}, smoke_records={"training.sft-curriculum": record}
        )
        == ()
    )


def test_a_stale_declaration_digest_refuses() -> None:
    record = replace(
        _fresh_record("training.sft-curriculum"), declaration_digest="0" * 64
    )
    assert (
        families_for_campaign(
            {}, smoke_records={"training.sft-curriculum": record}
        )
        == ()
    )


def test_a_row_pointing_at_an_undeclared_artifact_refuses() -> None:
    record = replace(
        _fresh_record("training.sft-curriculum"),
        artifact="src/chowder/does-not-exist.py",
    )
    assert (
        families_for_campaign(
            {}, smoke_records={"training.sft-curriculum": record}
        )
        == ()
    )


def test_a_fresh_row_permits_a_production_family_and_its_hypothesis(tmp_path) -> None:
    records = {"training.sft-curriculum": _fresh_record("training.sft-curriculum")}
    permitted = {f.family_id for f in families_for_campaign({}, smoke_records=records)}
    assert permitted == {"training.sft-curriculum"}
    store = EvidenceStore(path=tmp_path / "evidence.jsonl")
    hypotheses = _hypotheses_for(store, "training.sft-curriculum", records)
    assert [h.family_id for h in hypotheses] == ["training.sft-curriculum"]


def test_the_missing_row_is_named_in_the_refusal() -> None:
    with pytest.raises(InterventionFamilyRefusal, match="has no smoke record"):
        assert_smoke_permits(
            family_from_id("training.sft-curriculum"), smoke_records={}
        )


def test_a_registered_family_without_a_smoke_record_is_never_proposed() -> None:
    family = _extra_family("compression.test-unproven")
    try:
        register_family(family)
        assert all(
            f.family_id != family.family_id for f in families_for_campaign({})
        ), "a family with no runnability proof must not be proposed"
    finally:
        _clear_extras()


def test_a_registered_family_carries_its_smoke_record() -> None:
    family = _extra_family("compression.test-proven")
    record = FamilySmokeRecord(
        family_id=family.family_id,
        artifact=family.implementation[0],
        mechanism="registration smoke",
        status="ran",
        outcome="invoked",
        declaration_digest=family_smoke_declaration_digest(family),
    )
    try:
        register_family(family, smoke_record=record)
        assert family.family_id in {
            f.family_id for f in families_for_campaign({})
        }
    finally:
        _clear_extras()


def test_registering_a_family_with_an_unusable_smoke_record_refuses() -> None:
    family = _extra_family("compression.test-bad-proof")
    bad = FamilySmokeRecord(
        family_id=family.family_id,
        artifact=family.implementation[0],
        mechanism="stale",
        status="skipped",
        declaration_digest=family_smoke_declaration_digest(family),
    )
    with pytest.raises(InterventionFamilyRefusal, match="unusable smoke record"):
        register_family(family, smoke_record=bad)
    assert all(f.family_id != family.family_id for f in family_registry()), (
        "a refused registration must not leave the family behind"
    )


def test_a_missing_file_is_no_record_not_an_error(tmp_path) -> None:
    assert load_family_smoke_records(tmp_path / "absent.json") == {}


def test_an_unreadable_record_refuses_rather_than_emptying(tmp_path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(InterventionFamilyRefusal, match="cannot be read"):
        load_family_smoke_records(path)


def test_duplicate_rows_refuse(tmp_path) -> None:
    path = tmp_path / "dupes.json"
    row = {
        "family_id": "training.sft-curriculum",
        "artifact": "a",
        "mechanism": "m",
        "status": "ran",
    }
    path.write_text(json.dumps({"rows": [row, row]}), encoding="utf-8")
    with pytest.raises(InterventionFamilyRefusal, match="two rows"):
        load_family_smoke_records(path)


def test_the_committed_record_binds_every_shipped_family() -> None:
    assert FAMILY_SMOKE_RECORD_PATH.is_file(), (
        "the committed smoke record is the proposal gate; without it nothing "
        "is runnable"
    )
    records = load_family_smoke_records()
    registry = {family.family_id: family for family in family_registry()}
    assert set(records) == set(registry), (
        "the committed record and the shipped registry must cover each other"
    )
    for family_id, family in registry.items():
        record = records[family_id]
        assert record.status == "ran", f"{family_id}: the committed row is not runnable"
        assert record.artifact in family.implementation, family_id
        assert record.declaration_digest == family_smoke_declaration_digest(family), (
            f"{family_id}: the committed smoke row is stale for the live declaration"
        )


def test_the_gate_reads_the_committed_record_by_default() -> None:
    permitted = {f.family_id for f in families_for_campaign({})}
    assert "training.sft-curriculum" in permitted
    assert "architecture.conditional-ffn" not in permitted  # research, not production
