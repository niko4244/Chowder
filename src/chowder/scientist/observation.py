"""ExperimentObservation: what a compiled experiment measured.

An observation always names the immutable run that produced it. There is no
such thing as an observation without a run: the director refuses fabricated
results (T6 in the threat model), and a run that does not exist in the run
registry cannot back an observation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class Measurement:
    """One measured number, scoped to the capability surface it measured."""

    surface: str          # skill/capability name (e.g. "reasoning")
    benchmark: str        # the sensor used (e.g. "math500@2024-04")
    value: float
    paired: bool = False  # paired with the parent's arm under the same protocol
    n_items: int = 0

    def __post_init__(self) -> None:
        if not self.surface.strip() or not self.benchmark.strip():
            raise ValueError("a measurement names its surface and its benchmark sensor")
        n = float(self.value)
        if n != n:  # NaN
            raise ValueError("a measurement cannot be NaN")

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface": self.surface,
            "benchmark": self.benchmark,
            "value": self.value,
            "paired": self.paired,
            "n_items": self.n_items,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Measurement":
        unknown = sorted(set(data) - {"surface", "benchmark", "value", "paired", "n_items"})
        if unknown:
            raise ValueError(f"unknown measurement keys: {unknown}")
        return cls(
            surface=str(data["surface"]),
            benchmark=str(data["benchmark"]),
            value=float(data["value"]),
            paired=bool(data.get("paired", False)),
            n_items=int(data.get("n_items", 0)),
        )


@dataclass(frozen=True)
class ExperimentObservation:
    """Grounded in exactly one immutable run."""

    observation_id: str
    run_id: str                 # the RunRegistry run root / run id
    experiment_ref: str         # the compiled experiment's id
    proposal_id: str
    hypothesis_id: str
    measurements: tuple[Measurement, ...]
    status: str = "complete"    # complete | failed | refused | cancelled
    wall_gpu_hours: float = 0.0
    notes: str = ""
    provider: str = ""          # recorded for provenance; never an authority
    mission_id: str = ""
    carried: bool = False       # True only for explicitly imported prior evidence
    #: The hardware class this measurement came from (e.g. "kaggle_2x_t4_16gb",
    #: "local_cuda_..."), stamped by the compute scheduler. Quality findings
    #: may compare across hardware classes when the protocol is identical;
    #: efficiency findings never leave theirs (docs/COMPUTE_PROVIDERS.md §3).
    hardware_class: str = ""

    _KNOWN = (
        "observation_id", "run_id", "experiment_ref", "proposal_id",
        "hypothesis_id", "measurements", "status", "wall_gpu_hours", "notes",
        "provider", "mission_id", "carried", "hardware_class",
    )

    def __post_init__(self) -> None:
        if not self.observation_id:
            raise ValueError("observation_id is required")
        if not self.run_id:
            raise ValueError("an observation must cite the run that produced it")
        if not self.measurements and self.status == "complete":
            raise ValueError("a complete observation without measurements is not evidence")

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "run_id": self.run_id,
            "experiment_ref": self.experiment_ref,
            "proposal_id": self.proposal_id,
            "hypothesis_id": self.hypothesis_id,
            "measurements": [m.to_dict() for m in self.measurements],
            "status": self.status,
            "wall_gpu_hours": self.wall_gpu_hours,
            "notes": self.notes,
            "provider": self.provider,
            "mission_id": self.mission_id,
            "carried": self.carried,
            "hardware_class": self.hardware_class,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentObservation":
        unknown = sorted(set(data) - set(cls._KNOWN))
        if unknown:
            raise ValueError(f"unknown observation keys (fail-closed): {unknown}")
        missing = [k for k in ("observation_id", "run_id", "experiment_ref",
                               "proposal_id", "hypothesis_id") if k not in data]
        if missing:
            raise ValueError(f"observation is missing required keys: {missing}")
        return cls(
            observation_id=str(data["observation_id"]),
            run_id=str(data["run_id"]),
            experiment_ref=str(data["experiment_ref"]),
            proposal_id=str(data["proposal_id"]),
            hypothesis_id=str(data["hypothesis_id"]),
            measurements=tuple(Measurement.from_dict(m)
                               for m in data.get("measurements", ())),
            status=str(data.get("status", "complete")),
            wall_gpu_hours=float(data.get("wall_gpu_hours", 0.0)),
            notes=str(data.get("notes", "")),
            provider=str(data.get("provider", "")),
            mission_id=str(data.get("mission_id", "")),
            carried=bool(data.get("carried", False)),
            hardware_class=str(data.get("hardware_class", "")),
        )
