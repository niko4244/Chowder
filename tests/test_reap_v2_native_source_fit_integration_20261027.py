"""End-to-end original native source-fit gate; no model or GPU launched."""

from pathlib import Path

import pytest

from chowder_batch import reap_v2_owner_bound_runtime_supervisor_20261027 as owner

ORIGINAL_LOG = Path(
    "F:/reap-v2-q4-windows-full1319-phase34-original-20261027-0152/"
    "reap128_v2_q4/server.log"
)
MODEL = Path("G:/reap128-v2-q4_k_m-20261025.gguf")


def observation(**updated):
    result = {
        "physical_available_bytes": 9 * owner.GIB,
        "commit_headroom_bytes": 11 * owner.GIB,
        "disk_free_bytes": 12 * owner.GIB,
        "gpu": {"uuid": "GPU-12345678", "total_mib": 65536,
                "used_mib": 46000, "free_mib": 19300,
                "reserved_mib": 236, "utilization_pct": 20},
        "root_owner_valid": True, "job_processes_valid": True,
        "foreign_ports": [], "foreign_processes": [],
        "owned_listener": True, "native_started": True,
        "native_residency": {"layers_offloaded": 41, "layers_total": 41,
                             "context": 16384, "kv_total_mib": 320.0,
                             "cpu_expert_offload": 0},
        "startup_seconds": 30, "elapsed_seconds": 60,
        "model_log_bytes": 21000, "job_process_count": 3,
    }
    result.update(updated)
    return result


def make_original_session_log(tmp_path, data: bytes) -> Path:
    arm_dir = tmp_path / "reap128_v2_q4"
    arm_dir.mkdir()
    (arm_dir / "server.log").write_bytes(data)
    return tmp_path


def original_prefix() -> bytes:
    if not ORIGINAL_LOG.is_file():
        pytest.skip("Original immutable local Windows Q4 full1319 evidence not mounted")
    with ORIGINAL_LOG.open("rb") as src:
        return src.read(owner.MAX_LOG_PREFIX_BYTES)


def test_real_original_full1319_native_fit_is_source_admissible(tmp_path):
    content = original_prefix()
    output = make_original_session_log(tmp_path, content)
    r = owner.verify_owned_original_native_fit(
        observation(), output=output, arm="reap128_v2_q4", model=MODEL)
    assert r["status"] == "NATIVE_SOURCE_FIT_VERIFIED_RUNTIME_ONLY"
    assert len(r["source_sha256"]) == 64
    assert not r["reasons"]


def test_seemingly_full_cuda_and_kv_but_fit_line_missing_must_never_pass(tmp_path):
    content = original_prefix().replace(b"successfully fit params", b"could not fit params")
    output = make_original_session_log(tmp_path, content)
    assert owner.validate_sample(observation()) == []  # reproduced old false-pass path
    r = owner.verify_owned_original_native_fit(
        observation(), output=output, arm="reap128_v2_q4", model=MODEL)
    assert r["status"] in {"PENDING_NATIVE_SOURCE_FIT_UNPROVEN", "REFUSED"}
    assert r["status"] != "NATIVE_SOURCE_FIT_VERIFIED_RUNTIME_ONLY"


def test_original_model_identity_swapped_is_hard_refusal(tmp_path):
    content = original_prefix().replace(
        b"G:\\reap128-v2-q4_k_m-20261025.gguf", b"C:\\foreign-model-q4.gguf"
    )
    output = make_original_session_log(tmp_path, content)
    r = owner.verify_owned_original_native_fit(
        observation(), output=output, arm="reap128_v2_q4", model=MODEL)
    assert r["status"] == "REFUSED"
    assert "NATIVE_MODEL_IDENTITY" in r["reasons"]


def test_native_cpu_expert_spill_is_hard_refusal(tmp_path):
    content = original_prefix().replace(
        b"n_expert_used         = 8",
        b"n_expert_used         = 8\nllama_context: n_cpu_moe = 3",
    )
    output = make_original_session_log(tmp_path, content)
    r = owner.verify_owned_original_native_fit(
        observation(), output=output, arm="reap128_v2_q4", model=MODEL)
    assert r["status"] == "REFUSED"
    assert "NATIVE_CPU_EXPERT_OFFLOAD" in r["reasons"]


def test_no_owned_listener_or_no_log_is_pending_not_pass(tmp_path):
    output = tmp_path / "not-created"
    r = owner.verify_owned_original_native_fit(
        observation(owned_listener=False), output=output, arm="reap128_v2_q4", model=MODEL)
    assert r["status"].startswith("PENDING")
    assert r["source_sha256"] is None


def test_foreign_symlinked_server_log_is_never_trusted(tmp_path):
    session = tmp_path / "reap128_v2_q4"
    session.mkdir()
    try:
        (session / "server.log").symlink_to(ORIGINAL_LOG)
    except (OSError, NotImplementedError):
        pytest.skip("Windows symlink permission unavailable")
    r = owner.verify_owned_original_native_fit(
        observation(), output=tmp_path, arm="reap128_v2_q4", model=MODEL)
    assert r["status"] == "REFUSED"
    assert "OWNED_NATIVE_SOURCE_LINKED" in r["reasons"]
