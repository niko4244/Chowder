"""Compute cost accounting: explicit units, admission, settlement.

Chowder has twice confused wall GPU-hours with device GPU-hours (the
attempt-07/08 refusals documented in GEN1_PREREG_AMENDMENT3). The growth
layer therefore never carries a bare ``gpu_hours`` float across a boundary
where the unit matters: it carries a :class:`ComputeCost` with separate
``device_gpu_hours`` and ``wall_gpu_hours`` fields, validated finite and
non-negative.

Two distinct controls use those numbers, and keeping them distinct is the
point of this module:

- **Admission control** (before compute): projected costs decide whether a
  job may start. Enforced by the training binding's preflight.
- **Settlement control** (after compute): actual costs decide whether the
  completed run *stayed inside* the frozen envelope. A successful training
  process does not imply a budget-compliant experiment; settlement is where
  an overrun becomes a mechanical refusal instead of a note.

The ledger (:class:`CycleCostLedger`) accumulates every cost a cycle
actually consumed -- winning recipes, losing recipes, failed attempts,
final candidate evaluation -- plus explicit zero-cost *references* to
historical measurements (a carried baseline costs nothing new but is
recorded, so an auditor can see the measurement was reused rather than
paid for). Its rendering is deterministic: identical entries produce an
identical JSON document and digest.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping

#: Machine-readable settlement failure identifiers. A promotion judge can
#: branch on these without parsing prose.
RESOURCE_OVERRUN = "RESOURCE_OVERRUN"
ACTUAL_DEVICE_GPU_HOURS_EXCEEDED = "ACTUAL_DEVICE_GPU_HOURS_EXCEEDED"
ACTUAL_WALL_GPU_HOURS_EXCEEDED = "ACTUAL_WALL_GPU_HOURS_EXCEEDED"
ACTUAL_EXCEEDS_PROJECTION = "ACTUAL_EXCEEDS_PROJECTION"
#: A declared device ceiling was asked to be settled against a device figure
#: that was never measured. Zero is not evidence of zero: settling it would
#: certify compliance the run did not demonstrate.
ACTUAL_DEVICE_GPU_HOURS_UNMEASURED = "ACTUAL_DEVICE_GPU_HOURS_UNMEASURED"
#: A campaign's *planned* spend was refused against its declared campaign
#: ceilings before compute. A plan needs no measurement claim, so these are
#: deliberately separate codes from the actual-vs-ceiling ones above: admission
#: and settlement are different controls and are never substituted for one
#: another.
PROJECTED_DEVICE_GPU_HOURS_EXCEEDED = "PROJECTED_DEVICE_GPU_HOURS_EXCEEDED"
PROJECTED_WALL_GPU_HOURS_EXCEEDED = "PROJECTED_WALL_GPU_HOURS_EXCEEDED"


def _checked(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite, got {value!r}")
    if value < 0.0:
        raise ValueError(f"{label} must be non-negative, got {value!r}")
    return value


@dataclass(frozen=True)
class ComputeCost:
    """A GPU-hour cost with explicit units.

    ``device_gpu_hours`` is time-intensity-on-hardware (what a device
    ceiling constrains); ``wall_gpu_hours`` is elapsed-clock cost including
    streaming/load overhead (what a wall envelope constrains). The two are
    never interchangeable, so every field is named at every boundary.
    """

    device_gpu_hours: float
    wall_gpu_hours: float
    source: str = "measured"
    incremental: bool = True
    measurement_method: str = ""
    #: Whether ``device_gpu_hours`` is a *measurement* or just the absence of
    #: one. Defaults to ``False``: a caller that never separated device time
    #: has an unmeasured zero, and settlement must not read that as "device
    #: free". A device ceiling may only be settled against a cost whose device
    #: figure was measured -- zero measured is fine, zero unmeasured refuses.
    device_measured: bool = False

    def __post_init__(self) -> None:
        _checked(self.device_gpu_hours, "device_gpu_hours")
        _checked(self.wall_gpu_hours, "wall_gpu_hours")
        if not isinstance(self.device_measured, bool):
            raise ValueError(
                f"device_measured must be a bool, got {self.device_measured!r}"
            )
        if not self.source.strip():
            raise ValueError("ComputeCost.source must name where the cost came from")

    @classmethod
    def zero(cls, *, source: str, incremental: bool = False, measurement_method: str = "") -> "ComputeCost":
        return cls(0.0, 0.0, source=source, incremental=incremental, measurement_method=measurement_method)

    @classmethod
    def measured(
        cls,
        *,
        device_gpu_hours: float,
        wall_gpu_hours: float,
        source: str,
        measurement_method: str = "",
    ) -> "ComputeCost":
        """A cost whose device figure really was measured (zero included)."""
        return cls(
            device_gpu_hours,
            wall_gpu_hours,
            source=source,
            measurement_method=measurement_method,
            device_measured=True,
        )

    @classmethod
    def from_wall_only(cls, wall_gpu_hours: float, *, source: str) -> "ComputeCost":
        """Wrap a wall-charged summary number (the trainer reports wall).

        The trainer's ``gpu_hours`` summary field is wall-charged (it counts
        elapsed process time including load and streaming). Device time is
        unknown at that boundary and recorded as 0.0 rather than guessed;
        ``device_measured`` stays ``False`` so no consumer can mistake the
        placeholder for "device free" -- :func:`settle_cost` refuses to settle
        a device ceiling against it.
        """
        return cls(
            0.0,
            _checked(wall_gpu_hours, "wall_gpu_hours"),
            source=source,
            measurement_method="wall-charged trainer summary; device time not separated",
            device_measured=False,
        )

    def plus(self, other: "ComputeCost") -> "ComputeCost":
        return ComputeCost(
            self.device_gpu_hours + other.device_gpu_hours,
            self.wall_gpu_hours + other.wall_gpu_hours,
            source=f"{self.source}+{other.source}",
            measurement_method=self.measurement_method or other.measurement_method,
            # A sum carries a device measurement only if every part it is made
            # of was measured: one unseparated contributor makes the total an
            # estimate, not a measurement.
            device_measured=self.device_measured and other.device_measured,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_gpu_hours": self.device_gpu_hours,
            "wall_gpu_hours": self.wall_gpu_hours,
            "source": self.source,
            "incremental": self.incremental,
            "measurement_method": self.measurement_method,
            "device_measured": self.device_measured,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ComputeCost":
        return cls(
            device_gpu_hours=float(data["device_gpu_hours"]),
            wall_gpu_hours=float(data["wall_gpu_hours"]),
            source=str(data.get("source", "unspecified")),
            incremental=bool(data.get("incremental", True)),
            measurement_method=str(data.get("measurement_method", "")),
            # Absent in a record written before the field existed, and absent
            # means unmeasured -- a stored row does not become a device
            # measurement by being read back.
            device_measured=bool(data.get("device_measured", False)),
        )


@dataclass(frozen=True)
class SettlementVerdict:
    """The mechanical result of comparing actual cost against a frozen envelope."""

    compliant: bool
    failure_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"budget_compliant": self.compliant, "budget_failure_reasons": list(self.failure_reasons)}


def settle_cost(
    *,
    actual: ComputeCost,
    projected: ComputeCost | None,
    device_ceiling: float | None,
    wall_ceiling: float | None,
    project_budget_wall_gpu_hours: float | None = None,
    projection_tolerance: float = 0.25,
) -> SettlementVerdict:
    """Post-run budget settlement. Actual cost is authoritative.

    Checks, in order: hard device ceiling, hard wall ceiling, declared
    project budget (wall units -- the project engine charges wall), and a
    projection-vs-actual comparison at the declared tolerance. Every breach
    is reported with its machine-readable identifier; none is downgraded to
    a warning.

    A device ceiling is settled only against a *measured* device figure.
    Wall-charged summaries leave device time unseparated (``0.0``), and that
    placeholder must never be read as "device free": settling it would
    certify compliance the run never demonstrated, so a declared device
    ceiling with an unmeasured device figure fails closed with
    :data:`ACTUAL_DEVICE_GPU_HOURS_UNMEASURED`. Zero is still a legitimate
    measurement -- ``ComputeCost.measured(device_gpu_hours=0.0, ...)`` passes
    a ceiling, because that zero was observed.
    """
    if projection_tolerance < 0.0:
        raise ValueError("projection_tolerance must be non-negative")
    reasons: list[str] = []
    if device_ceiling is not None and not actual.device_measured:
        reasons.append(
            f"{ACTUAL_DEVICE_GPU_HOURS_UNMEASURED}: device ceiling "
            f"{device_ceiling:.6f} cannot be settled against device "
            f"{actual.device_gpu_hours:.6f}, which was not measured "
            f"(source {actual.source!r}); unmeasured is not compliance"
        )
    elif device_ceiling is not None and actual.device_gpu_hours > device_ceiling + 1e-12:
        reasons.append(
            f"{ACTUAL_DEVICE_GPU_HOURS_EXCEEDED}: actual device "
            f"{actual.device_gpu_hours:.6f} > ceiling {device_ceiling:.6f}"
        )
    if wall_ceiling is not None and actual.wall_gpu_hours > wall_ceiling + 1e-12:
        reasons.append(
            f"{ACTUAL_WALL_GPU_HOURS_EXCEEDED}: actual wall "
            f"{actual.wall_gpu_hours:.6f} > ceiling {wall_ceiling:.6f}"
        )
    if (
        project_budget_wall_gpu_hours is not None
        and actual.wall_gpu_hours > project_budget_wall_gpu_hours + 1e-12
    ):
        reasons.append(
            f"{RESOURCE_OVERRUN}: actual wall {actual.wall_gpu_hours:.6f} > "
            f"project gpu_hour_budget {project_budget_wall_gpu_hours:.6f} (wall units)"
        )
    if projected is not None and projected.wall_gpu_hours > 0.0:
        if actual.wall_gpu_hours > projected.wall_gpu_hours * (1.0 + projection_tolerance) + 1e-12:
            reasons.append(
                f"{ACTUAL_EXCEEDS_PROJECTION}: actual wall {actual.wall_gpu_hours:.6f} "
                f"exceeds projection {projected.wall_gpu_hours:.6f} by more than the "
                f"declared tolerance {projection_tolerance:.2f}"
            )
    return SettlementVerdict(compliant=not reasons, failure_reasons=tuple(reasons))


@dataclass
class _LedgerEntry:
    label: str
    kind: str  # training | evaluation | baseline_reference | failed_attempt | load_probe
    cost: ComputeCost
    recipe_id: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "kind": self.kind,
            "recipe_id": self.recipe_id,
            "cost": self.cost.to_dict(),
            "notes": self.notes,
        }


@dataclass
class CycleCostLedger:
    """Every GPU-hour a cycle consumed, plus what it referenced for free."""

    cycle_id: str
    entries: list[_LedgerEntry] = field(default_factory=list)

    def add(
        self,
        label: str,
        kind: str,
        cost: ComputeCost,
        *,
        recipe_id: str = "",
        notes: str = "",
    ) -> ComputeCost:
        if kind not in {
            "training",
            "evaluation",
            "baseline_reference",
            "failed_attempt",
            "load_probe",
        }:
            raise ValueError(f"unknown ledger entry kind {kind!r}")
        self.entries.append(_LedgerEntry(label, kind, cost, recipe_id=recipe_id, notes=notes))
        return cost

    def add_reference(self, label: str, referenced_artifact: str) -> ComputeCost:
        """A historical measurement reused at zero incremental cost."""
        return self.add(
            label,
            "baseline_reference",
            ComputeCost.zero(source=f"reference:{referenced_artifact}"),
            notes=f"referenced historical measurement: {referenced_artifact}",
        )

    def total(self, *, incremental_only: bool = True) -> ComputeCost:
        total = ComputeCost.zero(source="total")
        device_measured = True
        for entry in self.entries:
            cost = entry.cost
            if incremental_only and not cost.incremental:
                continue
            total = total.plus(cost)
            device_measured = device_measured and cost.device_measured
        return ComputeCost(
            total.device_gpu_hours,
            total.wall_gpu_hours,
            source="total",
            measurement_method="sum of ledger entries",
            device_measured=device_measured,
        )

    def total_for_recipe(self, recipe_id: str) -> ComputeCost:
        total = ComputeCost.zero(source=f"recipe:{recipe_id}")
        device_measured = True
        for entry in self.entries:
            if entry.recipe_id == recipe_id:
                total = total.plus(entry.cost)
                device_measured = device_measured and entry.cost.device_measured
        return ComputeCost(
            total.device_gpu_hours,
            total.wall_gpu_hours,
            source=f"recipe:{recipe_id}",
            measurement_method="sum of recipe entries",
            device_measured=device_measured,
        )

    def render(self) -> dict[str, Any]:
        totals = self.total()
        per_recipe: dict[str, dict[str, float]] = {}
        for entry in self.entries:
            if not entry.recipe_id:
                continue
            slot = per_recipe.setdefault(entry.recipe_id, {"device_gpu_hours": 0.0, "wall_gpu_hours": 0.0})
            slot["device_gpu_hours"] += entry.cost.device_gpu_hours
            slot["wall_gpu_hours"] += entry.cost.wall_gpu_hours
        document = {
            "cycle_id": self.cycle_id,
            "entries": [entry.to_dict() for entry in self.entries],
            "totals": {
                "incremental": totals.to_dict(),
                "per_recipe": per_recipe,
            },
        }
        digest = hashlib.sha256(
            json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        document["digest_sha256"] = digest
        return document

    def write(self, path) -> str:
        document = self.render()
        from pathlib import Path as _P

        path = _P(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
        return document["digest_sha256"]
