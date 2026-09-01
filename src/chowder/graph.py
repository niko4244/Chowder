from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

from .models import Experiment


class GraphInvariantError(ValueError):
    pass


@dataclass
class ExperimentGraph:
    nodes: dict[str, Experiment] = field(default_factory=dict)
    children: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    def add(self, experiment: Experiment) -> None:
        if experiment.experiment_id in self.nodes:
            raise GraphInvariantError(f"duplicate experiment id: {experiment.experiment_id}")
        if experiment.parent_id is not None and experiment.parent_id not in self.nodes:
            raise GraphInvariantError(f"unknown parent: {experiment.parent_id}")
        self.nodes[experiment.experiment_id] = experiment
        if experiment.parent_id is not None:
            self.children[experiment.parent_id].add(experiment.experiment_id)

    def ancestors(self, experiment_id: str) -> tuple[str, ...]:
        if experiment_id not in self.nodes:
            raise KeyError(experiment_id)
        out: list[str] = []
        current = self.nodes[experiment_id]
        while current.parent_id is not None:
            out.append(current.parent_id)
            current = self.nodes[current.parent_id]
        return tuple(out)

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
