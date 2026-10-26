"""Synthetic, CPU-only safety regressions; no private examples or GPU work."""

from __future__ import annotations

import hashlib
from decimal import Decimal

import pytest

from chowder_batch import reap_hle_q4_source_preflight as pre

SAMPLE_BLOB = "a" * 64
PREDICTION_BYTES = b"synthetic pinned public prediction script"
JUDGE_BYTES = b"synthetic pinned public judge script"


def _head(_url: str):
    return {
        "status": 302, "revision": pre.HLE_COMMIT,
        "blob_sha256": SAMPLE_BLOB, "size": 123456,
    }


def _kaggle(args: list[str]) -> str:
    if args[1] == "quota":
        return "resource used remaining total refreshAt\nGPU 13.53h 16.47h 30.00h 2026-10-31T00:00:00"
    return 'kernel has status "KernelWorkerStatus.COMPLETE"'


def _public(_url: str) -> bytes:
    return PREDICTION_BYTES if _url.endswith(pre.PREDICTION_SCRIPT) else JUDGE_BYTES


@pytest.fixture
def mock_public_shas(monkeypatch):
    monkeypatch.setattr(pre, "PREDICTION_SCRIPT_SHA", hashlib.sha256(PREDICTION_BYTES).hexdigest())
    monkeypatch.setattr(pre, "JUDGE_SCRIPT_SHA", hashlib.sha256(JUDGE_BYTES).hexdigest())


def test_manifest_pins_exact_q4_binary_original_hle_source_and_no_tools():
    manifest = pre.frozen_manifest()
    source = manifest["protocol"]
    assert source["dataset_revision"] == "5a81a4c7271a2a2a312b9a690f0c2fde837e4c29"
    assert source["model_sha256"] == pre.Q4_GGUF_SHA
    assert source["llama_server_binary_sha256"] == pre.PINNED_LLAMA_SERVER_SHA
    assert source["serving_flags"] == ["--fit", "on", "-np", "1", "-c", "16384", "--jinja"]
    assert source["max_completion_tokens"] == 8192
    assert source["temperature"] == 0
    assert source["judge_model"] == "o3-mini-2025-01-31"
    assert source["tools"] == "none"
    assert source["original_scientific_gate_eligible"] is False
    assert manifest == pre.frozen_manifest()


def test_text_only_selection_rejects_image_questions_and_preserves_source_order():
    items = [
        {"id": "a", "question": "question a", "answer": "one", "image": None},
        {"id": "b", "question": "question b", "answer": "two", "image": "base64-imagedata"},
        {"id": "c", "question": "question c", "answer": "three", "image": ""},
    ]
    assert pre.text_only_ids(items) == ["a", "c"]


@pytest.mark.parametrize("malformed", [
    [{"id": "a", "question": "q", "answer": "a"}, {"id": "a", "question": "q", "answer": "a"}],
    [{"id": "x", "question": "", "answer": "answer"}],
    [{"id": "x", "question": "q", "answer": ""}],
    [{"id": "x", "question": "q", "answer": "a", "image": {}}],
])
def test_text_only_rejects_ambiguous_or_invalid_rows(malformed):
    with pytest.raises(ValueError):
        pre.text_only_ids(malformed)


def test_access_denied_is_auth_failure_even_without_pinned_blob():
    assert pre.verified_hle_head(lambda _: {"status": 403}, "") == (
        False, "HLE_GATED_ACCESS_DENIED"
    )


def test_access_requires_independent_blob_digest_and_exact_revision():
    assert pre.verified_hle_head(_head, "") == (False, "HLE_BLOB_SHA_NOT_FROZEN")
    assert pre.verified_hle_head(_head, "b" * 64) == (
        False, "HLE_DATA_BLOB_IDENTITY_MISMATCH"
    )
    assert pre.verified_hle_head(
        lambda _: {**_head(""), "revision": "0" * 40}, SAMPLE_BLOB
    ) == (False, "HLE_SOURCE_REVISION_MISMATCH")
    assert pre.verified_hle_head(_head, SAMPLE_BLOB) == (
        True, "HLE_AUTHORIZED_EXACT_BLOB_HEAD"
    )


@pytest.mark.parametrize("status", [200, 302, 307])
def test_metadata_only_authorized_statuses(status):
    assert pre.verified_hle_head(lambda _: {**_head(""), "status": status}, SAMPLE_BLOB)[0]


@pytest.mark.parametrize("status", [201, 301, 404, 429, 500, None])
def test_non_authorized_responses_refused(status):
    assert not pre.verified_hle_head(lambda _: {**_head(""), "status": status}, SAMPLE_BLOB)[0]


def test_quota_rejects_missing_nan_and_inconsistent_snapshots():
    assert pre.parsed_quota("GPU 13.53h 16.47h 30.00h") == Decimal("16.47")
    assert pre.parsed_quota("GPU nanh 10.00h 30.00h") is None
    assert pre.parsed_quota("GPU 40.00h 10.00h 30.00h") is None
    assert pre.parsed_quota("TPU 0.00h 10.00h 20.00h") is None


def test_protected_job_running_refuses_one_item():
    def active(args):
        if args[1] == "quota":
            return _kaggle(args)
        if pre.PROTECTED_ACTIVE_KERNEL in args:
            return 'status "KernelWorkerStatus.RUNNING"'
        return 'status "KernelWorkerStatus.COMPLETE"'

    blockers, state = pre.check_kaggle(active)
    assert "PROTECTED_KAGGLE_JOB_ACTIVE_OR_UNVERIFIED" in blockers
    assert state[pre.PROTECTED_ACTIVE_KERNEL] == "RUNNING"
    assert "FULL_GPU_JOB_INVENTORY_NOT_ATTESTED" in blockers


def test_protected_status_malformed_or_cli_failure_refused():
    bad, _ = pre.check_kaggle(lambda _args: "unknown")
    assert "PROTECTED_KAGGLE_JOB_ACTIVE_OR_UNVERIFIED" in bad
    assert "GPU_QUOTA_UNVERIFIED_OR_INSUFFICIENT" in bad
    bad, _ = pre.check_kaggle(lambda _args: (_ for _ in ()).throw(RuntimeError("unavailable")))
    assert "GPU_QUOTA_UNAVAILABLE" in bad


def test_public_source_immutable_sha_rejects_mutated_script(mock_public_shas):
    assert not pre.check_public_source_bytes(_public)
    assert "PUBLIC_EVALUATOR_SHA_MISMATCH" in pre.check_public_source_bytes(
        lambda _: b"unexpected replacement"
    )


def test_even_fully_authenticated_metadata_is_not_gpu_or_judge_approval(mock_public_shas):
    response = pre.source_preflight(
        expected_blob=SAMPLE_BLOB, head=_head, kaggle_read=_kaggle, public_fetch=_public
    )
    assert response["status"] == "REFUSED"
    assert response["hle_access_result"] == "HLE_AUTHORIZED_EXACT_BLOB_HEAD"
    assert response["runtime_one_item_approved"] is False
    assert response["gpu_job_launched"] is False
    assert response["generated_programs_executed"] == 0
    assert response["dataset_rows_read"] == 0
    assert response["chosen"] is None
    assert response["scientific_selection"] == "BLOCKED"
    assert response["blockers"] == sorted(set(response["blockers"]))
    assert "RUNTIME_GPU_MOUNT_AND_MEMORY_NOT_ATTESTED" in response["blockers"]


def test_denied_hle_and_active_protected_job_block_independently(mock_public_shas):
    def active(args):
        if args[1] == "quota":
            return _kaggle(args)
        if pre.PROTECTED_ACTIVE_KERNEL in args:
            return 'KernelWorkerStatus.RUNNING'
        return 'KernelWorkerStatus.COMPLETE'

    response = pre.source_preflight(
        expected_blob="", head=lambda _: {"status": 403},
        kaggle_read=active, public_fetch=_public,
    )
    assert "HLE_GATED_ACCESS_DENIED" in response["blockers"]
    assert "PROTECTED_KAGGLE_JOB_ACTIVE_OR_UNVERIFIED" in response["blockers"]
    assert response["runtime_one_item_approved"] is False
