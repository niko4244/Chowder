"""Only synthetic local candidate files; never run them."""

import io
import json
import tarfile

import pytest

from chowder_batch import reap_programbench_original_receipt_audit_worker14_20261027 as a


def source(monkeypatch):
    monkeypatch.setattr(a, "audit", lambda _: {
        "source_commit": a.SOURCE_COMMIT, "original_task_manifest_sha256": "a" * 64,
        "source_bound_task_inventory": [{"instance_id": "sample__task.1234567",
             "original_task_yaml_sha256": "b" * 64,
             "original_tests_json_sha256": "c" * 64}]})


def test_empty_original_model_population_is_unscored(tmp_path, monkeypatch):
    source(monkeypatch)
    outcome = a.inspect(tmp_path, tmp_path)
    assert outcome["candidate_archives_present"] == 0
    assert outcome["official_scored_rows"] == 0
    assert outcome["official_score"] is None


def test_claimed_q4_and_forged_score_cannot_become_official(tmp_path, monkeypatch):
    source(monkeypatch)
    folder = tmp_path / "sample__task.1234567"
    folder.mkdir()
    with tarfile.open(folder / "submission.tar.gz", "w:gz") as out:
        for name in ("compile.sh", "main.py"):
            data = b"echo test"
            info = tarfile.TarInfo(name)
            info.size = len(data)
            out.addfile(info, io.BytesIO(data))
    (folder / "inference_receipt.json").write_text(json.dumps({
        "model_sha256": a.REGISTERED_Q4_GGUF_SHA256,
        "generated_archive_sha256": a.sha256(folder / "submission.tar.gz"),
        "official_score": 1, "network_isolation_verified": True,
        "independent_admission_verified": True}))
    result = a.inspect(tmp_path, tmp_path)
    assert result["candidate_archives_present"] == 1
    assert result["candidates"][0]["independent_model_origin_verified"] is False
    assert result["candidates"][0]["original_official_score"] is None
    assert result["official_score"] is None


def test_unknown_registered_task_refused(tmp_path, monkeypatch):
    source(monkeypatch)
    (tmp_path / "unregistered").mkdir()
    with pytest.raises(ValueError, match="unregistered"):
        a.inspect(tmp_path, tmp_path)


def test_linked_task_refused(tmp_path, monkeypatch):
    source(monkeypatch)
    folder = tmp_path / "outside"
    folder.mkdir()
    link = tmp_path / "sample__task.1234567"
    try:
        link.symlink_to(folder, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink privilege unavailable")
    with pytest.raises(ValueError, match="linked"):
        a.inspect(tmp_path, tmp_path)
