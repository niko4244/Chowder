"""Cluster suspicious training examples into coherent candidate regression
causes -- distinct from the already-shipped eval-failure clustering in
`failures.py`, which clusters independent EVALUATION failures by exact-
match metadata (evaluator/suite/protocol_sha256/...), not training
examples. There is no equivalent exact-match key for two different
training prompts, so this clusters by real TEXT SIMILARITY among the
examples `dataset_influence.py` already flagged as suspicious (a
positive, real, measured `influence_score` -- the example's own loss got
measurably worse between the last-good and first-regressing checkpoint).

Deliberately a simple, deterministic, dependency-free similarity metric
(token-set Jaccard) rather than an embedding model: this operates entirely
on `TrainingExampleInfluence` records `dataset_influence.py` already
computed for real, so clustering itself needs no additional real-hardware
measurement, no GPU, and no new dependency. Keeps distinct clusters
separate rather than prematurely collapsing them -- a real, honest
similarity threshold, not a claim that clustered examples provably share
one root cause.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import Sequence

from .dataset_influence import TrainingExampleInfluence

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_DEFAULT_SIMILARITY_THRESHOLD = 0.3


def _tokenize(text: str) -> frozenset[str]:
    return frozenset(_TOKEN_PATTERN.findall(text.lower()))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


@dataclass(frozen=True)
class TrainingSampleCluster:
    """One candidate regression cause: a group of suspicious training
    examples whose text is similar enough to plausibly share a root cause.
    Not a causal claim -- see this module's own docstring and
    `dataset_influence.TrainingExampleInfluence.confidence` for why.
    """

    cluster_id: str
    member_row_indices: tuple[int, ...]
    representative_excerpt: str
    mean_influence_score: float
    max_influence_score: float
    member_confidences: tuple[str, ...]


def cluster_suspicious_training_examples(
    records: Sequence[TrainingExampleInfluence],
    *,
    similarity_threshold: float = _DEFAULT_SIMILARITY_THRESHOLD,
    min_influence_score: float = 0.0,
) -> tuple[TrainingSampleCluster, ...]:
    """Greedy single-link clustering by token-set Jaccard similarity among
    records whose `influence_score` exceeds `min_influence_score` (the
    "suspicious," real-measured-worsened examples `dataset_influence.py`
    already ranked -- a non-suspicious example, one that got *better*
    between the two checkpoints, is never a candidate regression cause and
    is excluded before clustering even starts).

    Single-link (a record joins the first existing cluster any one of its
    members is similar enough to) rather than requiring similarity to
    every member: cheap, deterministic, and -- since inputs here are
    already pre-filtered to a small, real "suspicious" set, not an entire
    raw dataset -- transitive chaining within one real regression's
    worth of examples is an acceptable, honest tradeoff for a first slice.

    Clusters are returned sorted by `max_influence_score` descending: the
    cluster containing the single most-worsened example first.
    """
    suspicious = sorted(
        (r for r in records if r.influence_score > min_influence_score),
        key=lambda r: r.row_index,
    )
    tokens_by_index = {r.row_index: _tokenize(r.prompt_excerpt) for r in suspicious}

    clusters: list[list[TrainingExampleInfluence]] = []
    for record in suspicious:
        record_tokens = tokens_by_index[record.row_index]
        placed_cluster = next(
            (
                cluster
                for cluster in clusters
                if any(
                    _jaccard(record_tokens, tokens_by_index[member.row_index]) >= similarity_threshold
                    for member in cluster
                )
            ),
            None,
        )
        if placed_cluster is not None:
            placed_cluster.append(record)
        else:
            clusters.append([record])

    results = tuple(
        TrainingSampleCluster(
            cluster_id=f"cluster-{index}",
            member_row_indices=tuple(sorted(member.row_index for member in cluster)),
            representative_excerpt=cluster[0].prompt_excerpt,
            mean_influence_score=statistics.mean(member.influence_score for member in cluster),
            max_influence_score=max(member.influence_score for member in cluster),
            member_confidences=tuple(member.confidence for member in cluster),
        )
        for index, cluster in enumerate(clusters)
    )
    return tuple(sorted(results, key=lambda cluster: cluster.max_influence_score, reverse=True))
