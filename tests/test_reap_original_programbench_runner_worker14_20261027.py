"""Fail-closed local admission tests; these are not model benchmark scores."""

import io
import json
import tarfile

import pytest

from chowder_batch import reap_original_programbench_runner_worker14_20261027 as runner

checked_archive = runner.checked_archive


def archive(tmp_path, entries):
    path = tmp_path / "submission.tar.gz"
    with tarfile.open(path, "w:gz") as out:
        for name, data, kind in entries:
            info = tarfile.TarInfo(name)
            if kind == "file":
                raw = data.encode()
                info.size = len(raw)
                out.addfile(info, io.BytesIO(raw))
            else:
                info.type = kind
                info.linkname = "../../host"
                out.addfile(info)
    return path


def test_valid_program_archive_is_only_staged_not_scored(tmp_path):
    path = archive(tmp_path, [("compile.sh", "#!/bin/sh\n", "file"),
                              ("main.py", "print(42)\n", "file")])
    r = checked_archive(path)
    assert r["archive_files"] == 2
    assert len(r["submission_sha256"]) == 64


@pytest.mark.parametrize("unsafe", ["../escape", "/absolute", "lib/../../escape",
                                        "C:/host", "lib\\evil"])
def test_refuse_archive_path_escape(tmp_path, unsafe):
    path = archive(tmp_path, [("compile.sh", "x", "file"),
                              (unsafe, "bad", "file")])
    with pytest.raises(ValueError, match="unsafe"):
        checked_archive(path)


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE,
                                   tarfile.CHRTYPE, tarfile.FIFOTYPE])
def test_refuse_links_devices_and_pipes(tmp_path, kind):
    path = archive(tmp_path, [("compile.sh", "x", "file"),
                              ("evil", "", kind)])
    with pytest.raises(ValueError, match="unsafe"):
        checked_archive(path)


def test_refuse_duplicate_member(tmp_path):
    path = archive(tmp_path, [("compile.sh", "x", "file"),
                              ("compile.sh", "y", "file")])
    with pytest.raises(ValueError, match="unsafe"):
        checked_archive(path)


def test_refuse_missing_compilation_contract(tmp_path):
    path = archive(tmp_path, [("main.py", "print(3)", "file")])
    with pytest.raises(ValueError, match="compile.sh"):
        checked_archive(path)


def test_actual_submission_layout_and_sha_binding(tmp_path, monkeypatch):
    task = "ffmpeg__ffmpeg.360a402"
    task_row = {"instance_id": task, "scorable_behavioral_tests": 10,
                "original_task_yaml_sha256": "a" * 64,
                "original_tests_json_sha256": "b" * 64,
                "original_repository": "ffmpeg/ffmpeg", "original_commit": "c" * 40}
    monkeypatch.setattr(runner, "audit", lambda _: {
        "source_commit": runner.SOURCE_COMMIT,
        "original_task_manifest_sha256": "b" * 64,
        "source_bound_task_inventory": [task_row],
    })
    source = archive(tmp_path, [("compile.sh", "#!/bin/sh", "file"),
                                ("main.py", "print(42)", "file")])
    receipt = tmp_path / "inference.json"
    record = {"model_sha256": runner.REGISTERED_Q4_GGUF_SHA256, "quantization": "Q4_K_M",
              "prompt_sha256": runner.task_prompt_identity(runner.SOURCE_COMMIT, "b" * 64, task_row),
              "response_sha256": "3" * 64, "source_commit": runner.SOURCE_COMMIT,
              "population_sha256": "b" * 64, "instance_id": task,
              "task_yaml_sha256": "a" * 64, "tests_json_sha256": "b" * 64,
              "task_cleanroom_image_digest": "sha256:" + "4" * 64,
              "network_isolation_verified": True,
              "generated_archive_sha256": runner.sha256(source)}
    receipt.write_text(json.dumps(record))
    output = tmp_path / "original-evidence"
    result = runner.stage(tmp_path, task, source, output, receipt)
    assert result["scored"] is False
    assert result["official_score"] is None
    assert result["cleanroom_independently_admitted"] is False
    assert result["model_origin_independently_verified"] is False
    assert result["independent_external_witness_verified"] is False
    assert result["status"].startswith("UNVERIFIED_")
    assert result["independent_admission_required"] is True
    assert result["self_asserted_network_isolation"] is True
    assert result["self_asserted_image_digest"] == "sha256:" + "4" * 64
    assert result["official_evaluator_command"][0:2] == ["programbench", "eval"]
    assert (output / task / "submission.tar.gz").read_bytes() == source.read_bytes()
    record["generated_archive_sha256"] = "5" * 64
    receipt.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="does not match"):
        runner.stage(tmp_path, task, source, tmp_path / "second", receipt)


@pytest.mark.parametrize("changed,invalid,reason", [
    ("model_sha256", "1" * 64, "unregistered original Q4"),
    ("prompt_sha256", "2" * 64, "immutable official"),
    ("source_commit", "f" * 40, "immutable official"),
    ("population_sha256", "e" * 64, "immutable official"),
    ("instance_id", "wrong-task", "immutable official"),
    ("task_yaml_sha256", "d" * 64, "immutable official"),
    ("tests_json_sha256", "c" * 64, "immutable official"),
])
def test_refuse_spoofed_model_source_or_prompt(tmp_path, monkeypatch, changed, invalid, reason):
    task = "ffmpeg__ffmpeg.360a402"
    row = {"instance_id": task, "original_task_yaml_sha256": "a" * 64,
           "original_tests_json_sha256": "b" * 64,
           "original_repository": "ffmpeg/ffmpeg", "original_commit": "c" * 40}
    monkeypatch.setattr(runner, "audit", lambda _: {
        "source_commit": runner.SOURCE_COMMIT,
        "original_task_manifest_sha256": "b" * 64,
        "source_bound_task_inventory": [row]})
    source = archive(tmp_path, [("compile.sh", "#!/bin/sh", "file"),
                                ("main.py", "print(1)", "file")])
    receipt = {"model_sha256": runner.REGISTERED_Q4_GGUF_SHA256, "quantization": "Q4_K_M",
               "prompt_sha256": runner.task_prompt_identity(runner.SOURCE_COMMIT, "b" * 64, row),
               "response_sha256": "3" * 64, "source_commit": runner.SOURCE_COMMIT,
               "population_sha256": "b" * 64, "instance_id": task,
               "task_yaml_sha256": "a" * 64, "tests_json_sha256": "b" * 64,
               "task_cleanroom_image_digest": "sha256:" + "4" * 64,
               "network_isolation_verified": True,
               "generated_archive_sha256": runner.sha256(source)}
    receipt[changed] = invalid
    evidence = tmp_path / "claims.json"
    evidence.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match=reason):
        runner.stage(tmp_path, task, source, tmp_path / "staged", evidence)
    assert not (tmp_path / "staged").exists()


def test_self_asserted_independent_witness_cannot_promote_admission(tmp_path, monkeypatch):
    task = "ffmpeg__ffmpeg.360a402"
    row = {"instance_id": task, "original_task_yaml_sha256": "a" * 64,
           "original_tests_json_sha256": "b" * 64,
           "original_repository": "ffmpeg/ffmpeg", "original_commit": "c" * 40}
    monkeypatch.setattr(runner, "audit", lambda _: {
        "source_commit": runner.SOURCE_COMMIT,
        "original_task_manifest_sha256": "b" * 64,
        "source_bound_task_inventory": [row]})
    source = archive(tmp_path, [("compile.sh", "#!/bin/sh", "file"),
                                ("main.py", "print(1)", "file")])
    claim = {"model_sha256": runner.REGISTERED_Q4_GGUF_SHA256, "quantization": "Q4_K_M",
             "prompt_sha256": runner.task_prompt_identity(runner.SOURCE_COMMIT, "b" * 64, row),
             "response_sha256": "3" * 64, "source_commit": runner.SOURCE_COMMIT,
             "population_sha256": "b" * 64, "instance_id": task,
             "task_yaml_sha256": "a" * 64, "tests_json_sha256": "b" * 64,
             "task_cleanroom_image_digest": "sha256:" + "f" * 64,
             "network_isolation_verified": True,
             "independent_external_witness_verified": True,
             "cleanroom_independently_admitted": True,
             "official_score": 1.0,
             "generated_archive_sha256": runner.sha256(source)}
    evidence = tmp_path / "claim.json"
    evidence.write_text(json.dumps(claim))
    staged = runner.stage(tmp_path, task, source, tmp_path / "staged", evidence)
    assert staged["independent_external_witness_verified"] is False
    assert staged["cleanroom_independently_admitted"] is False
    assert staged["official_score"] is None
    assert staged["scored"] is False
    assert staged["self_asserted_image_digest"] == "sha256:" + "f" * 64
