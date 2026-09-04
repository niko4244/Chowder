from __future__ import annotations

from chowder.dataset_influence import TrainingExampleInfluence
from chowder.training_sample_clusters import (
    TrainingSampleCluster,
    cluster_suspicious_training_examples,
)


def _record(row_index, prompt_excerpt, influence_score, confidence="low"):
    return TrainingExampleInfluence(
        row_index=row_index,
        prompt_excerpt=prompt_excerpt,
        good_checkpoint_loss=1.0,
        bad_checkpoint_loss=1.0 + influence_score,
        influence_score=influence_score,
        confidence=confidence,
        supporting_evidence={},
        checkpoint_interval=("good-checkpoint", "bad-checkpoint"),
        provenance={},
    )


def test_similar_examples_are_grouped_into_one_cluster():
    records = [
        _record(0, "the quick brown fox jumps", 0.5, "high"),
        _record(1, "the quick brown fox leaps", 0.4, "medium"),
    ]
    clusters = cluster_suspicious_training_examples(records)
    assert len(clusters) == 1
    assert clusters[0].member_row_indices == (0, 1)


def test_dissimilar_examples_stay_in_separate_clusters():
    records = [
        _record(0, "the quick brown fox jumps", 0.5),
        _record(1, "unrelated totally different sentence here", 0.3),
    ]
    clusters = cluster_suspicious_training_examples(records)
    assert len(clusters) == 2
    assert {c.member_row_indices for c in clusters} == {(0,), (1,)}


def test_non_suspicious_examples_are_excluded_before_clustering():
    """An example that got BETTER between the two checkpoints (negative or
    zero influence_score) is never a candidate regression cause."""
    records = [
        _record(0, "the quick brown fox jumps", 0.5),
        _record(1, "the quick brown fox leaps", -0.2),
        _record(2, "the quick brown fox runs", 0.0),
    ]
    clusters = cluster_suspicious_training_examples(records)
    all_members = {idx for c in clusters for idx in c.member_row_indices}
    assert 1 not in all_members
    assert 2 not in all_members
    assert 0 in all_members


def test_custom_min_influence_score_raises_the_suspicion_bar():
    records = [
        _record(0, "topic alpha", 0.5),
        _record(1, "topic beta", 0.1),
    ]
    clusters = cluster_suspicious_training_examples(records, min_influence_score=0.3)
    all_members = {idx for c in clusters for idx in c.member_row_indices}
    assert all_members == {0}


def test_clusters_sorted_by_max_influence_score_descending():
    records = [
        _record(0, "alpha topic one", 0.2),
        _record(1, "beta topic two", 0.9),
        _record(2, "gamma topic three", 0.5),
    ]
    clusters = cluster_suspicious_training_examples(records)
    scores = [c.max_influence_score for c in clusters]
    assert scores == sorted(scores, reverse=True)
    assert clusters[0].member_row_indices == (1,)


def test_cluster_carries_real_aggregate_statistics():
    records = [
        _record(0, "the quick brown fox jumps", 0.6, "high"),
        _record(1, "the quick brown fox leaps", 0.2, "low"),
    ]
    clusters = cluster_suspicious_training_examples(records)
    assert len(clusters) == 1
    cluster = clusters[0]
    assert cluster.mean_influence_score == 0.4
    assert cluster.max_influence_score == 0.6
    assert cluster.member_confidences == ("high", "low")
    assert cluster.representative_excerpt == "the quick brown fox jumps"


def test_empty_input_returns_empty_tuple():
    assert cluster_suspicious_training_examples([]) == ()


def test_single_link_chains_a_bridging_example_into_one_cluster():
    """A single-link design: an example similar to ANY existing cluster
    member joins that cluster, even if it isn't directly similar to every
    other member -- a deliberate, documented tradeoff for a first slice."""
    records = [
        _record(0, "alpha beta gamma delta", 0.5),
        _record(1, "gamma delta epsilon zeta", 0.4),  # bridges to 0 via gamma/delta
        _record(2, "epsilon zeta eta theta", 0.3),  # bridges to 1 via epsilon/zeta, not directly to 0
    ]
    clusters = cluster_suspicious_training_examples(records, similarity_threshold=0.3)
    assert len(clusters) == 1
    assert clusters[0].member_row_indices == (0, 1, 2)


def test_similarity_threshold_is_configurable():
    records = [
        _record(0, "alpha beta gamma", 0.5),
        _record(1, "alpha delta epsilon", 0.4),  # shares only "alpha" -- low overlap
    ]
    loose = cluster_suspicious_training_examples(records, similarity_threshold=0.1)
    strict = cluster_suspicious_training_examples(records, similarity_threshold=0.9)
    assert len(loose) == 1
    assert len(strict) == 2


def test_result_type_is_training_sample_cluster():
    records = [_record(0, "some text", 0.5)]
    clusters = cluster_suspicious_training_examples(records)
    assert all(isinstance(c, TrainingSampleCluster) for c in clusters)
