"""Scientist-handoff consumption: the growth loop adopts a graduated survivor.

The screening lane graduates a survivor; the growth loop must be able to make
it the next generation's target WITHOUT a manual copy step. The hand-off is a
durable file — ``scientist-handoff.json`` in the loop's session state — written
by ``ModelResearchService.graduate_survivors_to_campaign_drafts`` (the journal
records the same fact; the file is what the loop consumes). One survivor per
generation: the loop adopts it at most once, then the normal target-selection
gates resume for the generation after.

The adoption point is the loop's own gate chain (``_proposal_or_decision``):
a survivor becomes the target proposal INSTEAD OF the selector's pick, then
flows through every existing gate unchanged — treatment allowlist, budget,
draft composition, production preparation, freeze. The loop never freezes,
executes, or promotes anything on the scientist's behalf; it just chooses the
target the evidence already earned.

Honest limits (all refuse by name):

- ``HANDOFF_NO_BENCHMARK_MAP``: a survivor names capability surfaces; the loop
  cannot know which pinned ``name@version`` benchmark measures a surface. That
  mapping is an operator decision (the same rule the bridge enforces) and is
  carried in the hand-off file, not invented here.
- ``HANDOFF_FILE_UNREADABLE``: the hand-off exists but is not valid JSON /
  missing required fields — durable state that cannot be read is refused, not
  skipped.
- A survivor whose target skill is in the policy's PROTECTED set still refuses
  downstream in the builder (protected sets are gates) — adoption does not
  bypass any gate, it only nominates the target.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .next_campaign import NextCampaignRefusal
from .target_selection import TargetProposal, TargetScoreFactors

HANDOFF_FILENAME = "scientist-handoff.json"


@dataclass(frozen=True)
class ScreeningGraduate:
    """One graduated survivor, projected onto the loop's target shape.

    Fields mirror what ``ModelResearchService`` journaled; ``benchmark_map``
    is the operator's per-surface pinned-benchmark mapping (the same input the
    bridge required — the loop refuses without it rather than inventing an
    instrument).
    """

    experiment_id: str
    proposal_id: str
    intervention: str
    falsification_rule: str
    target_skill: str
    suggested_training_type: str
    priority: float
    expected_cost_gpu_hours: float
    survivor_score: float
    benchmark_map: Mapping[str, str]
    screening_mission_id: str

    def to_target_proposal(self, *, parent_version: str) -> TargetProposal:
        """The loop-native TargetProposal for the survivor's next generation.

        The audit fields carry the screening session's own provenance (the
        survivor's score in ``factors.total``, the intervention and
        falsification rule as the treatment reason) — never invented numbers.
        """
        pinned = self.benchmark_map.get(self.target_skill, "")
        if not pinned:
            raise NextCampaignRefusal(
                "HANDOFF_NO_BENCHMARK_MAP: the graduated survivor targets "
                f"{self.target_skill!r} but the hand-off carries no pinned "
                "benchmark for that surface; naming the eval instrument is an "
                "operator decision, not an invention"
            )
        note = (
            f"graduated from scientist screening: {self.intervention} "
            f"(proposal {self.proposal_id}, falsification: "
            f"{self.falsification_rule}, survivor score {self.survivor_score:.4f})"
        )
        return TargetProposal(
            parent_version=parent_version,
            target_skill=self.target_skill,
            target_benchmarks=(pinned,),
            weakness_evidence=(note,),
            priority=self.priority,
            confidence=0.5,
            expected_trainability=0.5,
            regression_risks=(),
            suggested_training_type=self.suggested_training_type,
            expected_cost_gpu_hours=self.expected_cost_gpu_hours,
            why_not_other_targets={},
            factors=TargetScoreFactors(
                weakness=0.0,
                confidence=0.5,
                importance=0.0,
                recurrence=0.0,
                frontier_gap=0.0,
                trainability=0.5,
                novelty=0.0,
                efficiency=0.0,
                regression_risk=0.0,
                repeat_penalty=0.0,
                uncertainty_penalty=0.0,
                total=float(self.survivor_score),
            ),
            treatment_reason=note,
        )


def write_handoff_file(
    state_root: Path,
    *,
    graduate: ScreeningGraduate,
) -> Path:
    """Write the durable hand-off the loop consumes. Called by the service's
    graduation path (same facts it journals); one file per session state."""
    path = Path(state_root) / HANDOFF_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "experiment_id": graduate.experiment_id,
                "proposal_id": graduate.proposal_id,
                "intervention": graduate.intervention,
                "falsification_rule": graduate.falsification_rule,
                "target_skill": graduate.target_skill,
                "suggested_training_type": graduate.suggested_training_type,
                "priority": graduate.priority,
                "expected_cost_gpu_hours": graduate.expected_cost_gpu_hours,
                "survivor_score": graduate.survivor_score,
                "benchmark_map": dict(graduate.benchmark_map),
                "screening_mission_id": graduate.screening_mission_id,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def load_graduated_survivor(
    state_root: Path,
) -> tuple[ScreeningGraduate, Path] | None:
    """The pending hand-off for this session, or None. A file that exists but
    cannot be parsed raises (durable state that cannot be read is refused,
    never skipped)."""
    path = Path(state_root) / HANDOFF_FILENAME
    if not path.exists():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise NextCampaignRefusal(
            f"HANDOFF_FILE_UNREADABLE: {path} is not valid JSON: {error}"
        ) from error
    required = (
        "experiment_id",
        "proposal_id",
        "intervention",
        "falsification_rule",
        "target_skill",
        "suggested_training_type",
        "priority",
        "expected_cost_gpu_hours",
        "survivor_score",
        "benchmark_map",
        "screening_mission_id",
    )
    missing = [key for key in required if key not in document]
    if missing:
        raise NextCampaignRefusal(
            f"HANDOFF_FILE_UNREADABLE: {path} is missing required fields: {missing}"
        )
    graduate = ScreeningGraduate(
        experiment_id=str(document["experiment_id"]),
        proposal_id=str(document["proposal_id"]),
        intervention=str(document["intervention"]),
        falsification_rule=str(document["falsification_rule"]),
        target_skill=str(document["target_skill"]),
        suggested_training_type=str(document["suggested_training_type"]),
        priority=float(document["priority"]),
        expected_cost_gpu_hours=float(document["expected_cost_gpu_hours"]),
        survivor_score=float(document["survivor_score"]),
        benchmark_map={str(k): str(v) for k, v in dict(document["benchmark_map"]).items()},
        screening_mission_id=str(document["screening_mission_id"]),
    )
    return graduate, path


def clear_consumed_handoff(state_root: Path, *, target_skill: str) -> bool:
    """Clear the hand-off after the loop durably records its target. Returns
    True when a hand-off for this target was consumed. Refuses to clear a
    file naming a DIFFERENT target — a mismatch means the durable file and
    the loop disagree about what was adopted, and that is a reviewable event,
    not a silent delete."""
    path = Path(state_root) / HANDOFF_FILENAME
    if not path.exists():
        return False
    loaded = load_graduated_survivor(state_root)
    if loaded is None:
        return False
    graduate, _ = loaded
    if graduate.target_skill != str(target_skill):
        raise NextCampaignRefusal(
            f"HANDOFF_TARGET_MISMATCH: the durable hand-off targets "
            f"{graduate.target_skill!r} but the recorded target is "
            f"{str(target_skill)!r}; refusing to clear"
        )
    path.unlink()
    return True
