"""ResearchTree: the Chowder-native progressive research tree.

Borrowed conceptually from AI Scientist v2's BFTS, but native: a node
represents an experimentally meaningful research state, branches compete for
budget through an explicit static scoring function, and the tree — not the
provider — is the authority on what exists and what was spent. Scoring is
deterministic and recomputable; an LLM never ranks branches into existence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

NODE_STATUSES = (
    "proposed",     # node created from an admitted proposal
    "running",
    "observed",     # observations recorded
    "expanded",
    "pruned",
    "terminal",
)


@dataclass
class ResearchNode:
    node_id: str
    parent_id: str | None
    hypothesis_id: str
    proposal_id: str
    experiment_ref: str = ""
    status: str = "proposed"
    observation_ids: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()          # run ids
    measured_cost_gpu_hours: float = 0.0
    information_gain: float = 0.0
    capability_delta: float | None = None
    transfer_delta: float | None = None
    regression_risk: float = 0.0
    replication_confidence: float = 0.0
    next_questions: tuple[str, ...] = ()
    depth: int = 0

    _KNOWN = (
        "node_id", "parent_id", "hypothesis_id", "proposal_id", "experiment_ref",
        "status", "observation_ids", "evidence_refs", "measured_cost_gpu_hours",
        "information_gain", "capability_delta", "transfer_delta",
        "regression_risk", "replication_confidence", "next_questions", "depth",
    )

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("node_id is required")
        if not self.hypothesis_id or not self.proposal_id:
            raise ValueError("a research node exists because of a hypothesis and a proposal")
        if self.status not in NODE_STATUSES:
            raise ValueError(f"unknown node status: {self.status}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "hypothesis_id": self.hypothesis_id,
            "proposal_id": self.proposal_id,
            "experiment_ref": self.experiment_ref,
            "status": self.status,
            "observation_ids": list(self.observation_ids),
            "evidence_refs": list(self.evidence_refs),
            "measured_cost_gpu_hours": self.measured_cost_gpu_hours,
            "information_gain": self.information_gain,
            "capability_delta": self.capability_delta,
            "transfer_delta": self.transfer_delta,
            "regression_risk": self.regression_risk,
            "replication_confidence": self.replication_confidence,
            "next_questions": list(self.next_questions),
            "depth": self.depth,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResearchNode":
        unknown = sorted(set(data) - set(cls._KNOWN))
        if unknown:
            raise ValueError(f"unknown node keys (fail-closed): {unknown}")
        return cls(
            node_id=str(data["node_id"]),
            parent_id=(None if data.get("parent_id") is None else str(data["parent_id"])),
            hypothesis_id=str(data["hypothesis_id"]),
            proposal_id=str(data["proposal_id"]),
            experiment_ref=str(data.get("experiment_ref", "")),
            status=str(data.get("status", "proposed")),
            observation_ids=tuple(str(o) for o in data.get("observation_ids", ())),
            evidence_refs=tuple(str(e) for e in data.get("evidence_refs", ())),
            measured_cost_gpu_hours=float(data.get("measured_cost_gpu_hours", 0.0)),
            information_gain=float(data.get("information_gain", 0.0)),
            capability_delta=(None if data.get("capability_delta") is None
                              else float(data["capability_delta"])),
            transfer_delta=(None if data.get("transfer_delta") is None
                            else float(data["transfer_delta"])),
            regression_risk=float(data.get("regression_risk", 0.0)),
            replication_confidence=float(data.get("replication_confidence", 0.0)),
            next_questions=tuple(str(q) for q in data.get("next_questions", ())),
            depth=int(data.get("depth", 0)),
        )


@dataclass
class ResearchBranch:
    branch_id: str
    hypothesis_id: str
    root_node_id: str
    status: str = "active"       # active | leading | exhausted | pruned | promoted
    allocated_gpu_hours: float = 0.0
    spent_gpu_hours: float = 0.0
    failed_mechanism_repeats: int = 0
    _KNOWN = ("branch_id", "hypothesis_id", "root_node_id", "status",
              "allocated_gpu_hours", "spent_gpu_hours", "failed_mechanism_repeats")

    def __post_init__(self) -> None:
        if not self.branch_id or not self.hypothesis_id or not self.root_node_id:
            raise ValueError("a branch names its id, hypothesis and root node")
        if self.status not in ("active", "leading", "exhausted", "pruned", "promoted"):
            raise ValueError(f"unknown branch status: {self.status}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "hypothesis_id": self.hypothesis_id,
            "root_node_id": self.root_node_id,
            "status": self.status,
            "allocated_gpu_hours": self.allocated_gpu_hours,
            "spent_gpu_hours": self.spent_gpu_hours,
            "failed_mechanism_repeats": self.failed_mechanism_repeats,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResearchBranch":
        unknown = sorted(set(data) - set(cls._KNOWN))
        if unknown:
            raise ValueError(f"unknown branch keys (fail-closed): {unknown}")
        return cls(
            branch_id=str(data["branch_id"]),
            hypothesis_id=str(data["hypothesis_id"]),
            root_node_id=str(data["root_node_id"]),
            status=str(data.get("status", "active")),
            allocated_gpu_hours=float(data.get("allocated_gpu_hours", 0.0)),
            spent_gpu_hours=float(data.get("spent_gpu_hours", 0.0)),
            failed_mechanism_repeats=int(data.get("failed_mechanism_repeats", 0)),
        )


@dataclass(frozen=True)
class TreeScoreWeights:
    """The static scoring weights; prefer/penalize factors from the mission.

    Defaults encode the mission's list; they are data, not an LLM's opinion."""

    capability_gain: float = 1.0
    novelty: float = 0.5
    information_gain: float = 0.8
    transferability: float = 0.9
    reproducibility: float = 0.6
    compute_efficiency: float = 0.5
    uncertainty_reduction: float = 0.4
    regression_risk_penalty: float = 1.2
    repeated_failure_penalty: float = 1.0
    contamination_uncertainty_penalty: float = 0.8
    non_transfer_penalty: float = 1.0
    cost_penalty: float = 0.6

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TreeScoreWeights":
        valid = {f for f in cls().__dict__ if not f.startswith("_")
                 and not callable(getattr(cls, f, None))}
        unknown = sorted(set(data) - valid)
        if unknown:
            raise ValueError(f"unknown tree-score weights: {unknown}")
        return cls(**{k: float(v) for k, v in data.items()})


class ResearchTree:
    """Durable, Chowder-owned research tree. Save/load is lossless."""

    def __init__(self, mission_id: str) -> None:
        if not mission_id:
            raise ValueError("a tree belongs to a mission")
        self.mission_id = mission_id
        self._nodes: dict[str, ResearchNode] = {}
        self._branches: dict[str, ResearchBranch] = {}
        self._order: list[str] = []

    # -- construction -------------------------------------------------------

    def add_branch(self, branch: ResearchBranch) -> ResearchBranch:
        if branch.branch_id in self._branches:
            raise ValueError(f"branch already exists: {branch.branch_id}")
        self._branches[branch.branch_id] = branch
        return branch

    def add_node(self, node: ResearchNode) -> ResearchNode:
        if node.node_id in self._nodes:
            raise ValueError(f"node already exists: {node.node_id}")
        if node.parent_id is not None:
            parent = self._nodes.get(node.parent_id)
            if parent is None:
                raise ValueError(f"node parent does not exist: {node.parent_id}")
            node.depth = parent.depth + 1
        self._nodes[node.node_id] = node
        self._order.append(node.node_id)
        return node

    # -- queries -------------------------------------------------------------

    def node(self, node_id: str) -> ResearchNode:
        node = self._nodes.get(node_id)
        if node is None:
            raise KeyError(node_id)
        return node

    def branch(self, branch_id: str) -> ResearchBranch:
        branch = self._branches.get(branch_id)
        if branch is None:
            raise KeyError(branch_id)
        return branch

    def branches(self) -> tuple[ResearchBranch, ...]:
        return tuple(self._branches.values())

    def nodes(self) -> tuple[ResearchNode, ...]:
        return tuple(self._nodes[n] for n in self._order)

    def active_branches(self) -> tuple[ResearchBranch, ...]:
        return tuple(b for b in self._branches.values() if b.status in ("active", "leading"))

    def max_depth(self) -> int:
        return max((n.depth for n in self._nodes.values()), default=0)

    # -- scoring -------------------------------------------------------------

    def score_node(
        self,
        node: ResearchNode,
        *,
        weights: TreeScoreWeights | None = None,
        novel_mechanism: bool = True,
        contamination_uncertain: bool = False,
        mechanism_failed_before: bool = False,
    ) -> float:
        """Deterministic branch-competition score. Named factors, no LLM."""
        w = weights or TreeScoreWeights()
        cap = node.capability_delta if node.capability_delta is not None else 0.0
        transfer = node.transfer_delta if node.transfer_delta is not None else 0.0
        # A transfer delta that was never measured is UNKNOWN, not a failure
        # (the growth control plane's own invariant): the non-transfer penalty
        # fires only when transfer was measured and contradicts the gain.
        transfer_measured = node.transfer_delta is not None
        non_transferable = (
            cap > 0 and transfer_measured and node.transfer_delta is not None
            and node.transfer_delta <= 0
        )
        score = (
            w.capability_gain * max(0.0, cap)
            + w.novelty * (1.0 if novel_mechanism else 0.0)
            + w.information_gain * max(0.0, node.information_gain)
            + w.transferability * max(0.0, transfer)
            + w.reproducibility * node.replication_confidence
            + w.uncertainty_reduction * (1.0 if node.next_questions else 0.0)
            - w.regression_risk_penalty * max(0.0, node.regression_risk)
            - w.repeated_failure_penalty * (1.0 if mechanism_failed_before else 0.0)
            - w.contamination_uncertainty_penalty * (1.0 if contamination_uncertain else 0.0)
            - w.non_transfer_penalty * (1.0 if non_transferable else 0.0)
        )
        if node.measured_cost_gpu_hours > 0:
            score -= w.cost_penalty * min(
                1.0, node.measured_cost_gpu_hours / max(0.1, cap + 0.1)
            )
        return score

    def rank_active_branches(self, **score_kwargs: Any) -> list[tuple[float, ResearchBranch]]:
        """Active branches ordered best-first by their best observed node."""
        scored: list[tuple[float, ResearchBranch]] = []
        for branch in self.active_branches():
            branch_nodes = [n for n in self._nodes.values()
                            if n.hypothesis_id == branch.hypothesis_id]
            best = max(
                (self.score_node(n, **score_kwargs) for n in branch_nodes),
                default=-1e9,
            )
            scored.append((best, branch))
        return sorted(scored, key=lambda pair: pair[0], reverse=True)

    def prune_repeated_failures(self, *, max_repeats: int = 2) -> tuple[str, ...]:
        """A branch that keeps failing the same way loses allocation (pinned:
        a repeatedly failing branch loses allocation)."""
        pruned: list[str] = []
        for branch in self._branches.values():
            if branch.failed_mechanism_repeats >= max_repeats and branch.status == "active":
                branch.status = "pruned"
                pruned.append(branch.branch_id)
        return tuple(pruned)

    def budget_survivors(self, *, keep_top: int) -> tuple[ResearchBranch, ...]:
        """Successive-halving-style elimination between sibling branches:
        keep the top `keep_top` by score, prune the rest of the active ones."""
        ranked = self.rank_active_branches()
        survivors = [b for _, b in ranked[:keep_top]]
        pruned: list[str] = []
        for _, branch in ranked[keep_top:]:
            branch.status = "pruned"
            pruned.append(branch.branch_id)
        for branch in survivors:
            branch.status = "leading"
        return tuple(survivors)

    def plateaued(self, *, window: int = 3, epsilon: float = 0.0) -> bool:
        """No capability improvement across the last `window` observed nodes."""
        observed = [n for n in self._nodes.values()
                    if n.status in ("observed", "expanded")
                    and n.capability_delta is not None]
        if len(observed) < window:
            return False
        recent = observed[-window:]
        return all((n.capability_delta or 0.0) <= epsilon for n in recent)

    # -- durability ------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "branches": [b.to_dict() for b in self._branches.values()],
            "nodes": [self._nodes[n].to_dict() for n in self._order],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResearchTree":
        tree = cls(mission_id=str(data["mission_id"]))
        for b in data.get("branches", ()):
            tree.add_branch(ResearchBranch.from_dict(b))
        for n in data.get("nodes", ()):
            tree.add_node(ResearchNode.from_dict(n))
        return tree

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ResearchTree":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
