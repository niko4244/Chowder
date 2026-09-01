import json

import pytest

from chowder.contamination import write_holdout_fingerprint_index
from chowder.failures import FailureCluster, FailureSourceRole, RepairPlan
from chowder.repair_requests import (
    RepairSourceProposal,
    RepairStrategy,
    build_repair_request,
    materialize_repair_proposal,
    request_repair_sources,
)
from chowder.repair_sources import RepairSource, SourcedRepairExample


def _cluster(kind="answer_mismatch"):
    return FailureCluster(
        cluster_id="c" * 64,
        evaluator="transformers-text",
        suite="reasoning",
        protocol_sha256="a" * 64,
        source_role=FailureSourceRole.GATE_HOLDOUT,
        failure_kind=kind,
        failure_ids=("f" * 64, "e" * 64),
    )


def _plan(*, malicious_prose=False):
    secret = "HOLDOUT RAW: secret prompt => secret answer" if malicious_prose else "generic"
    return RepairPlan(
        plan_id="p" * 64,
        cluster_id="c" * 64,
        observation=secret,
        suspected_cause=secret,
        intervention=secret,
        source_failure_ids=("f" * 64, "e" * 64),
        direct_training_allowed=False,
        requires_independent_source=True,
    )


def test_provider_payload_excludes_raw_and_freeform_failure_content():
    request = build_repair_request(plan=_plan(malicious_prose=True), cluster=_cluster())
    payload = request.to_provider_payload()
    serialized = json.dumps(payload, sort_keys=True)
    assert request.strategy is RepairStrategy.NEAR_NEIGHBOR_REASONING
    assert "secret prompt" not in serialized
    assert "secret answer" not in serialized
    forbidden_keys = {
        "prompt",
        "expected",
        "prediction",
        "row_index",
        "observation",
        "suspected_cause",
        "intervention",
    }
    assert forbidden_keys.isdisjoint(payload)


def test_failure_kind_maps_to_controlled_strategy():
    assert build_repair_request(plan=_plan(), cluster=_cluster("empty_prediction")).strategy is RepairStrategy.CONCISE_ANSWER
    assert build_repair_request(plan=_plan(), cluster=_cluster("refusal_or_unknown")).strategy is RepairStrategy.CALIBRATED_ANSWERING
    assert build_repair_request(plan=_plan(), cluster=_cluster("overlong_mismatch")).strategy is RepairStrategy.FORMAT_CONTROL


def test_request_refuses_plan_with_different_failure_lineage():
    plan = RepairPlan(
        plan_id="p" * 64,
        cluster_id="c" * 64,
        observation="x",
        suspected_cause="x",
        intervention="x",
        source_failure_ids=("1" * 64,),
        direct_training_allowed=False,
        requires_independent_source=True,
    )
    with pytest.raises(ValueError, match="failure lineage"):
        build_repair_request(plan=plan, cluster=_cluster())


class Provider:
    name = "independent-corpus-provider"
    version = "1"

    def __init__(self, *, contaminated=False, wrong_identity=False):
        self.contaminated = contaminated
        self.wrong_identity = wrong_identity
        self.seen_payload = None

    def propose(self, request):
        self.seen_payload = request.to_provider_payload()
        prompt, expected = (
            ("2+2?", "4") if self.contaminated else ("3+3?", "6")
        )
        return RepairSourceProposal(
            request_id=request.request_id,
            provider_name=("wrong" if self.wrong_identity else self.name),
            provider_version=self.version,
            sources=(RepairSource("src-1", "corpus://math", "b" * 64),),
            examples=(SourcedRepairExample("ex-1", "src-1", prompt, expected),),
        )


def test_provider_invocation_validates_identity_and_uses_request_boundary():
    request = build_repair_request(plan=_plan(malicious_prose=True), cluster=_cluster())
    provider = Provider()
    proposal = request_repair_sources(provider=provider, request=request)
    assert proposal.provider_name == provider.name
    serialized = json.dumps(provider.seen_payload, sort_keys=True)
    assert "secret prompt" not in serialized

    with pytest.raises(ValueError, match="provider identity"):
        request_repair_sources(provider=Provider(wrong_identity=True), request=request)


def test_materialization_runs_contamination_and_source_provenance_checks(tmp_path):
    holdout = tmp_path / "holdout.jsonl"
    write_holdout_fingerprint_index([("2+2?", "4")], holdout)
    request = build_repair_request(plan=_plan(), cluster=_cluster())
    provider = Provider()
    proposal = request_repair_sources(provider=provider, request=request)
    materialized = materialize_repair_proposal(
        request=request,
        proposal=proposal,
        provider=provider,
        holdout_fingerprint_files=(holdout,),
        dataset_path=tmp_path / "repair.jsonl",
        source_manifest_path=tmp_path / "sources.json",
    )
    assert materialized.contamination_audit.clean
    assert len(materialized.dataset_sha256) == 64
    assert len(materialized.source_manifest_sha256) == 64


def test_provider_cannot_launder_holdout_example_through_proposal(tmp_path):
    holdout = tmp_path / "holdout.jsonl"
    write_holdout_fingerprint_index([("2+2?", "4")], holdout)
    request = build_repair_request(plan=_plan(), cluster=_cluster())
    provider = Provider(contaminated=True)
    proposal = request_repair_sources(provider=provider, request=request)
    with pytest.raises(ValueError, match="overlaps holdout"):
        materialize_repair_proposal(
            request=request,
            proposal=proposal,
            provider=provider,
            holdout_fingerprint_files=(holdout,),
            dataset_path=tmp_path / "blocked.jsonl",
            source_manifest_path=tmp_path / "blocked-sources.json",
        )
