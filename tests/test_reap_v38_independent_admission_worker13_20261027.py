"""Independent fail-closed review of research HumanEval source admission."""

from __future__ import annotations

import pytest

from chowder_batch import reap_humaneval_v38_actual_execution_20261027 as v38


def test_original_generated_code_stays_unapproved():
    from chowder_batch import reap_humaneval_v2_attestation as v2
    assert v2.APPROVED_SANDBOX_CAPABILITY_SHA256 is None
    assert v2.APPROVED_INDEPENDENT_WITNESS_SHA256 is None
    assert v38.APPROVED_V38_RESEARCH_LOADER_SHA256 == v38._sha(v38.V38_LOADER.encode())
    assert v38.APPROVED_V38_RESEARCH_HOST_OBSERVER_SHA256 == v38._sha(
        v38.V38_VM_SOURCE_OBSERVER.encode())


def test_host_observer_cannot_be_approved_on_current_network_evidence():
    source = v38.V38_VM_SOURCE_OBSERVER
    assert "net/dev" in source
    assert "net/route" in source
    assert "net/ipv6_route" in source
    assert "mountinfo" in source
    assert "st_dev" in source and "st_ino" in source
    assert "O_NOFOLLOW" in source
    assert v38.APPROVED_V38_RESEARCH_HOST_OBSERVER_SHA256 == v38._sha(source.encode())


def test_bogus_kernel_observation_must_never_admit_staged_source(monkeypatch):
    class FakeResult:
        returncode = 0
        stdout = '{"staged_program_sha256":"' + 'a' * 64 + '","net_interfaces":["lo"]}'
        stderr = ""

    monkeypatch.setattr(v38, "_bounded_exec", lambda *_, **__: FakeResult())
    with pytest.raises(RuntimeError, match="witness failed"):
        v38._source_stage_host_observe(123, "a" * 64, 3)


def test_candidate_release_fails_before_source_access_or_output_creation(monkeypatch, tmp_path):
    monkeypatch.setattr(v38, "APPROVED_V38_RESEARCH_LOADER_SHA256", None)
    monkeypatch.setattr(v38, "original_program_manifest", lambda *_: pytest.fail("read original candidate"))
    private = tmp_path / "reap-v2-human164-v38-private-not-admitted"
    with pytest.raises(RuntimeError, match="REFUSED unreviewed"):
        v38.execute_original_research_164("q4", private_output=private)
    assert not private.exists()
