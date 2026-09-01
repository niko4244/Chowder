from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .models import Experiment


def deep_merge_config(base: Mapping[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge an experiment patch without mutating its ancestors."""
    merged: dict[str, Any] = deepcopy(dict(base))
    for key, value in patch.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            merged[key] = deep_merge_config(current, value)
        else:
            merged[key] = deepcopy(value)
    return merged


class GraphInvariantError(ValueError):
    pass


@dataclass
class ExperimentGraph:
    nodes: dict[str, Experiment] = field(default_factory=dict)
    children: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    def validate_additions(self, experiments: Iterable[Experiment]) -> tuple[Experiment, ...]:
        """Validate an ordered batch without mutating the graph.

        Parents may be existing nodes or earlier members of the same batch. This
        preserves the historical parent-then-child proposal behavior while making
        a later invariant failure atomic instead of leaving earlier nodes behind.
        """

        rows = tuple(experiments)
        staged_ids = set(self.nodes)
        for experiment in rows:
            if experiment.experiment_id in staged_ids:
                raise GraphInvariantError(f"duplicate experiment id: {experiment.experiment_id}")
            if experiment.parent_id is not None and experiment.parent_id not in staged_ids:
                raise GraphInvariantError(f"unknown parent: {experiment.parent_id}")
            staged_ids.add(experiment.experiment_id)
        return rows

    def add_many(self, experiments: Iterable[Experiment]) -> tuple[Experiment, ...]:
        rows = self.validate_additions(experiments)
        for experiment in rows:
            self.nodes[experiment.experiment_id] = experiment
            if experiment.parent_id is not None:
                self.children[experiment.parent_id].add(experiment.experiment_id)
        return rows

    def add(self, experiment: Experiment) -> None:
        self.add_many((experiment,))

    def remove_many(self, experiment_ids: Iterable[str]) -> tuple[Experiment, ...]:
        """Atomically remove a batch when it has no children outside the batch."""

        ids = tuple(experiment_ids)
        if len(ids) != len(set(ids)):
            raise GraphInvariantError("duplicate experiment id in removal batch")
        remove_set = set(ids)
        for experiment_id in ids:
            if experiment_id not in self.nodes:
                raise GraphInvariantError(f"unknown experiment id: {experiment_id}")
            external_children = self.children.get(experiment_id, set()) - remove_set
            if external_children:
                raise GraphInvariantError(
                    f"cannot remove experiment {experiment_id}: external children exist"
                )

        removed = tuple(self.nodes[experiment_id] for experiment_id in ids)
        for experiment_id in ids:
            experiment = self.nodes[experiment_id]
            if experiment.parent_id is not None:
                siblings = self.children.get(experiment.parent_id)
                if siblings is not None:
                    siblings.discard(experiment_id)
                    if not siblings:
                        self.children.pop(experiment.parent_id, None)
        for experiment_id in ids:
            self.children.pop(experiment_id, None)
            self.nodes.pop(experiment_id)
        return removed

    def ancestors(self, experiment_id: str) -> tuple[str, ...]:
        if experiment_id not in self.nodes:
            raise KeyError(experiment_id)
        out: list[str] = []
        current = self.nodes[experiment_id]
        while current.parent_id is not None:
            out.append(current.parent_id)
            current = self.nodes[current.parent_id]
        return tuple(out)

    def resolve_config(
        self, experiment_id: str, base_config: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Resolve root→child config patches into the immutable execution config."""
        if experiment_id not in self.nodes:
            raise KeyError(experiment_id)
        lineage = tuple(reversed(self.ancestors(experiment_id))) + (experiment_id,)
        resolved: dict[str, Any] = deepcopy(dict(base_config or {}))
        for node_id in lineage:
            resolved = deep_merge_config(resolved, self.nodes[node_id].config_patch)
        return resolved

    def topological(self) -> tuple[Experiment, ...]:
        indegree = {node_id: 0 for node_id in self.nodes}
        for parent, kids in self.children.items():
            if parent not in self.nodes:
                raise GraphInvariantError(f"dangling parent: {parent}")
            for child in kids:
                indegree[child] += 1

        queue = deque(sorted(k for k, d in indegree.items() if d == 0))
        order: list[Experiment] = []
        while queue:
            node_id = queue.popleft()
            order.append(self.nodes[node_id])
            for child in sorted(self.children.get(node_id, ())):
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)

        if len(order) != len(self.nodes):
            raise GraphInvariantError("experiment graph contains a cycle")
        return tuple(order)
