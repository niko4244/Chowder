"""Native-scored observations remain descriptive and source-bound."""

import json
from pathlib import Path

import pytest

from chowder_batch import reap_q4_original600_session_efficiency_20261027 as audit


def test_real_windows_and_original_benchmark_crosscheck():
    if not audit.NATIVE_RESOURCE.is_file() or not audit.ORIGINAL_GUARD_V3.is_dir():
        pytest.skip("private genuine F: research evidence unavailable")
    result = audit.independent_efficiency()
    assert result["original_completed_scored_tasks"] >= 1
    assert result["actual_original_model_calls"] > 0
    assert result["original_output_tokens"] > 0
    assert result["native_allocation"]["native_cuda_offloaded_layers"] == 41
    assert result["native_allocation"]["native_cpu_expert_offload"] is False
    assert result["registered_dense_same_session_first200_ratio"] is None
    assert result["qualified_frozen_speed_comparison"] is False
    assert result["rate_is_not_original_llama_cpp_native_decode_tokens_per_second"] is True
    assert result["release_approved"] is False and result["chosen"] is None


def test_real_original_resource_identity_and_summary_consistency():
    if not audit.NATIVE_RESOURCE.is_file():
        pytest.skip("private genuine original telemetry unavailable")
    result = audit.verify_real_native_resource()
    assert result["source_sha256"] == audit.NATIVE_RESOURCE_SHA
    assert result["derived_sample_extrema"]["min_host_gpu_free_mib"] > 0
    assert result["telemetry"]["source_owner_sha256"] == audit.OWNER_SHA


def test_original_resource_mutation_refused_without_write(tmp_path, monkeypatch):
    if not audit.NATIVE_RESOURCE.is_file():
        pytest.skip("private genuine original telemetry unavailable")
    forged = tmp_path / "modified.json"
    payload = json.loads(audit.NATIVE_RESOURCE.read_bytes())
    payload["min_host_gpu_free_mib"] = 12000
    forged.write_text(json.dumps(payload))
    monkeypatch.setattr(audit, "NATIVE_RESOURCE", forged)
    with pytest.raises(ValueError, match="mutated"):
        audit.verify_real_native_resource(forged)


def test_forged_telemetry_summary_cannot_be_presented_as_measured(tmp_path, monkeypatch):
    if not audit.NATIVE_RESOURCE.is_file():
        pytest.skip("private genuine original telemetry unavailable")
    payload = json.loads(audit.NATIVE_RESOURCE.read_bytes())
    payload["min_host_gpu_free_mib"] = 12000
    forged = tmp_path / "modified-with-forged-matching-digest.json"
    forged.write_text(json.dumps(payload))
    import hashlib

    monkeypatch.setattr(audit, "NATIVE_RESOURCE", forged)
    monkeypatch.setattr(audit, "NATIVE_RESOURCE_SHA",
                        hashlib.sha256(forged.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="summary differs"):
        audit.verify_real_native_resource(forged)


def test_tampered_cuda_offload_cannot_pass_native_acceptance(tmp_path, monkeypatch):
    if not audit.NATIVE_RESOURCE.is_file():
        pytest.skip("private genuine original telemetry unavailable")
    payload = json.loads(audit.NATIVE_RESOURCE.read_bytes())
    payload["source_bound_native_allocator"]["native_cpu_expert_offload"] = True
    forged = tmp_path / "fake-offload.json"
    forged.write_text(json.dumps(payload))
    import hashlib

    monkeypatch.setattr(audit, "NATIVE_RESOURCE", forged)
    monkeypatch.setattr(audit, "NATIVE_RESOURCE_SHA",
                        hashlib.sha256(forged.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="CUDA allocation"):
        audit.verify_real_native_resource(forged)


def test_foreign_original_experiment_owner_refused(tmp_path):
    with pytest.raises(ValueError, match="not original"):
        audit.independent_efficiency(tmp_path)


def test_old_native_research_snapshot_must_not_be_a_current_candidate(tmp_path):
    with pytest.raises(ValueError, match="missing/foreign"):
        audit.verify_real_native_resource(Path(tmp_path / "fake.json"))
