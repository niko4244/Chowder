"""Fail-closed decoding of persisted GPU cost flags and legacy defaults."""

import pytest

from chowder.growth.compute_cost import (
    ComputeCost, CycleCostLedger, checked_ledger_total, ledger_digest, settle_cost,
)


def _record(**overrides):
    return {
        "device_gpu_hours": 0.0,
        "wall_gpu_hours": 0.25,
        "source": "receipt:verified",
        "incremental": True,
        "device_measured": False,
        **overrides,
    }


@pytest.mark.parametrize("flag", ["false", "true", "0", "1", 0, 1, None, []])
@pytest.mark.parametrize("field", ["incremental", "device_measured"])
def test_persisted_cost_flags_reject_nonboolean_values(field, flag):
    with pytest.raises(ValueError, match="JSON booleans"):
        ComputeCost.from_dict(_record(**{field: flag}))


@pytest.mark.parametrize("field", ["wall_gpu_hours", "device_gpu_hours"])
def test_gpu_hours_reject_bool_as_numeric(field):
    with pytest.raises(ValueError, match="numeric, not booleans"):
        ComputeCost.from_dict(_record(**{field: True}))


@pytest.mark.parametrize("field", ["wall_gpu_hours", "device_gpu_hours"])
@pytest.mark.parametrize("value", ["0.25", "0", "1e-3", None, [], {}])
def test_persisted_gpu_hours_reject_coercible_nonnumeric_json_values(field, value):
    # A typed serialized charge is part of the payment/receipt proof.  A
    # self-pinned ledger cannot redefine its numeric cost as a JSON string.
    with pytest.raises(ValueError, match="GPU-hour amounts must be JSON numbers"):
        ComputeCost.from_dict(_record(**{field: value}))


@pytest.mark.parametrize("field", ["wall_gpu_hours", "device_gpu_hours"])
def test_persisted_gpu_hours_allow_finite_numeric_ints(field):
    assert getattr(ComputeCost.from_dict(_record(**{field: 1})), field) == 1.0


def test_checked_ledger_rejects_cost_string_with_recomputed_digest():
    ledger = CycleCostLedger("cycle-1")
    ledger.add("paid", "training", ComputeCost.from_wall_only(0.25, source="fixture"))
    doc = ledger.render()
    doc["entries"][0]["cost"]["wall_gpu_hours"] = "0.25"
    doc["digest_sha256"] = ledger_digest(doc)
    with pytest.raises(ValueError, match="GPU-hour amounts must be JSON numbers"):
        checked_ledger_total(doc, cycle_id="cycle-1")


def test_legacy_absent_flags_are_unmeasured_and_incremental():
    decoded = ComputeCost.from_dict({"wall_gpu_hours": 0.25, "device_gpu_hours": 0.0})
    assert decoded.incremental is True
    assert decoded.device_measured is False
    verdict = settle_cost(actual=decoded, projected=None, device_ceiling=1.0, wall_ceiling=1.0)
    assert verdict.compliant is False
    assert "UNMEASURED" in " ".join(verdict.failure_reasons)


def test_actual_json_flags_round_trip_without_changing_device_measurement():
    original = ComputeCost.measured(device_gpu_hours=0.2, wall_gpu_hours=0.25, source="verified")
    assert ComputeCost.from_dict(original.to_dict()) == original


def test_direct_constructor_rejects_nonboolean_incremental():
    with pytest.raises(ValueError, match="incremental must be a bool"):
        ComputeCost(0.0, 0.1, source="fixture", incremental="false")


def test_checked_ledger_settlement_recomputes_independent_entries():
    ledger = CycleCostLedger("cycle-1")
    ledger.add("first", "training", ComputeCost.measured(
        device_gpu_hours=0.1, wall_gpu_hours=0.2, source="fixture"))
    ledger.add_reference("baseline", "parent")
    original = ledger.render()
    assert checked_ledger_total(original, cycle_id="cycle-1") == ledger.total()

    changed = {**original, "totals": {"incremental": {**original["totals"]["incremental"],
                                                     "wall_gpu_hours": 0.0},
                                          "per_recipe": original["totals"]["per_recipe"]}}
    changed["digest_sha256"] = ledger_digest(changed)
    with pytest.raises(ValueError, match="totals or entry fields"):
        checked_ledger_total(changed, cycle_id="cycle-1")


def test_checked_ledger_requires_digest_and_cycle_identity():
    ledger = CycleCostLedger("cycle-1")
    doc = ledger.render()
    with pytest.raises(ValueError, match="cycle identity"):
        checked_ledger_total(doc, cycle_id="other")
    with pytest.raises(ValueError, match="digest"):
        checked_ledger_total({**doc, "digest_sha256": "0" * 64}, cycle_id="cycle-1")


def test_checked_ledger_refuses_forged_device_measured_string():
    ledger = CycleCostLedger("cycle-1")
    ledger.add("first", "training", ComputeCost.from_wall_only(0.2, source="fixture"))
    doc = ledger.render()
    doc["entries"][0]["cost"]["device_measured"] = "false"
    doc["digest_sha256"] = ledger_digest(doc)
    with pytest.raises(ValueError, match="JSON booleans"):
        checked_ledger_total(doc, cycle_id="cycle-1")


def test_checked_ledger_refuses_suppressed_paid_training_spend():
    ledger = CycleCostLedger("cycle-1")
    ledger.add("paid run", "training", ComputeCost.from_wall_only(0.2, source="fixture"))
    doc = ledger.render()
    doc["entries"][0]["cost"]["incremental"] = False
    doc["totals"]["incremental"]["wall_gpu_hours"] = 0.0
    doc["digest_sha256"] = ledger_digest(doc)
    with pytest.raises(ValueError, match="paid accounting entry"):
        checked_ledger_total(doc, cycle_id="cycle-1")
