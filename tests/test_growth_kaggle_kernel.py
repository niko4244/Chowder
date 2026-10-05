"""The kernel-side job runner, tested on a laptop.

``run_kernel_job`` is the code the Kaggle entry script calls after installing
the pinned commit. Every guarantee the dispatcher later verifies -- the
commit echo, the input digests, the artifact manifest, the environment, the
resume vocabulary -- is exercised here with injected seams, so a change to
them is caught without a GPU or a Kaggle session.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from chowder.growth.kaggle_kernel import (
    JOB_RECORD_NAME,
    RESUME_STATE_NAME,
    install_spec,
    run_kernel_job,
    sha256_file,
)

COMMIT = "a" * 40
OTHER_COMMIT = "b" * 40


def _declared_input(tmp_path: Path) -> tuple[Path, dict]:
    material = tmp_path / "training-material.json"
    material.write_text('{"examples": 3}\n', encoding="utf-8")
    sha256, size = sha256_file(material)
    return material, {"name": "training_material", "sha256": sha256, "bytes": size}


def _document(
    tmp_path: Path,
    *,
    input_path: Path | None = None,
    input_entry: dict | None = None,
    command: list[str] | None = None,
    resume_from: str | None = None,
    model_commit: str | None = "c" * 40,
    output_dir: Path | None = None,
) -> dict:
    output = output_dir or (tmp_path / "working")
    entry = input_entry
    if entry is None:
        material, entry = _declared_input(tmp_path)
        input_path = input_path or material
    input_paths = {entry["name"]: str(input_path)} if input_path is not None else {}
    return {
        "spec": {
            "spec_id": "gen2:recipe-01:attempt-01",
            "source": {
                "repository": "https://github.com/niko4244/Chowder",
                "commit_sha": COMMIT,
                "chowder_version": "0.5.0-dev",
                "cycle_id": "gen2",
                "recipe_id": "recipe-01",
                "attempt_id": "attempt-01",
            },
            "inputs": [entry],
            "mounts": ["owner/dataset"],
            "resume_from": resume_from,
        },
        "kernel": {
            "kernel_ref": "nikma/chowder-gen2-recipe-01-abcd1234",
            "install_spec": install_spec("https://github.com/niko4244/Chowder", COMMIT),
            "command": command or [],
            "input_paths": input_paths,
            "output_dir": str(output),
            "model_commit": model_commit,
            "skip_install": True,
        },
    }


def _run(document, output_dir: Path, *, installed: str | None = COMMIT, versions=None):
    return run_kernel_job(
        document,
        installed_commit_fn=lambda: installed,
        package_versions_fn=lambda: dict(versions or {"torch": "2.4.0"}),
        command_runner=lambda command, cwd: subprocess.run(
            list(command), cwd=str(cwd), capture_output=True, text=True
        ),
    )


def test_install_spec_pins_the_repository_and_exact_commit() -> None:
    spec = install_spec("https://github.com/niko4244/Chowder", COMMIT, extras=("train", "qlora"))
    assert spec == (
        f"chowder-ai[train,qlora] @ git+https://github.com/niko4244/Chowder.git@{COMMIT}"
    )
    with pytest.raises(ValueError, match="40-character"):
        install_spec("https://github.com/niko4244/Chowder", "abc123")
    with pytest.raises(ValueError, match="git URL"):
        install_spec("no-scheme-repo", COMMIT)


def test_a_complete_job_records_inputs_artifacts_and_environment(tmp_path: Path) -> None:
    output = tmp_path / "working"
    document = _document(
        tmp_path,
        command=[
            sys.executable,
            "-c",
            "from pathlib import Path; Path('adapter').mkdir(); "
            "Path('adapter/model.bin').write_bytes(b'weights')",
        ],
    )
    record = _run(document, output)
    assert record["state"] == "complete", record["error"]
    assert record["source_commit_sha"] == COMMIT
    entry = document["spec"]["inputs"][0]
    assert record["input_verification"] == [[entry["name"], entry["sha256"]]]
    artifacts = {entry["path"]: entry for entry in record["artifact_manifest"]}
    assert "adapter/model.bin" in artifacts
    assert artifacts["adapter/model.bin"]["bytes"] == len(b"weights")
    assert JOB_RECORD_NAME not in artifacts, "a record cannot hash itself into its own manifest"
    assert record["environment"]["python_version"]
    assert record["environment"]["packages"] == {"torch": "2.4.0"}
    assert record["environment"]["model_commit"] == "c" * 40
    written = json.loads((output / JOB_RECORD_NAME).read_text(encoding="utf-8"))
    assert written == record


def test_a_moved_input_is_an_error_record(tmp_path: Path) -> None:
    document = _document(tmp_path)
    document["spec"]["inputs"][0]["sha256"] = "f" * 64  # declared bytes that are not there
    record = _run(document, tmp_path / "working")
    assert record["state"] == "error"
    assert "moved" in record["error"]
    assert record["input_verification"] == []


def test_a_conflicting_mount_is_resolved_by_digest_not_by_position(
    tmp_path: Path,
) -> None:
    """Kaggle mounts a dataset under the bare slug, or the owner-qualified
    path when the slug is taken; the declared bytes decide which one is ours."""
    material, entry = _declared_input(tmp_path)
    output = tmp_path / "working"
    conflicting = tmp_path / "someone-elses-material.json"
    conflicting.write_bytes(b"another dataset with the same slug\n")
    document = _document(
        tmp_path,
        input_path=material,
        input_entry=entry,
    )
    document["kernel"]["input_paths"] = {
        entry["name"]: [str(conflicting), str(material)]
    }
    record = _run(document, output)
    assert record["state"] == "complete", record["error"]
    assert record["input_locations"] == {entry["name"]: str(material)}


def test_no_candidate_carrying_the_declared_bytes_is_an_error_record(
    tmp_path: Path,
) -> None:
    material, entry = _declared_input(tmp_path)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_bytes(b"wrong bytes, first mount\n")
    second.write_bytes(b"wrong bytes, second mount\n")
    document = _document(tmp_path, input_path=material, input_entry=entry)
    document["kernel"]["input_paths"] = {
        entry["name"]: [str(first), str(second)]
    }
    record = _run(document, tmp_path / "working")
    assert record["state"] == "error"
    assert "moved" in record["error"]
    assert str(first) in record["error"] and str(second) in record["error"]
    assert record["input_verification"] == []


def test_an_input_without_a_kernel_path_is_an_error_record(tmp_path: Path) -> None:
    document = _document(tmp_path)
    document["kernel"]["input_paths"] = {}
    record = _run(document, tmp_path / "working")
    assert record["state"] == "error"
    assert "no kernel path" in record["error"]


def test_an_installed_commit_that_differs_is_an_error_record(tmp_path: Path) -> None:
    document = _document(tmp_path)
    record = _run(document, tmp_path / "working", installed=OTHER_COMMIT)
    assert record["state"] == "error"
    assert "does not equal the declared" in record["error"]
    assert record["source_commit_sha"] == OTHER_COMMIT


def test_an_uninstalled_source_is_an_error_record(tmp_path: Path) -> None:
    document = _document(tmp_path)
    record = _run(document, tmp_path / "working", installed=None)
    assert record["state"] == "error"
    assert "not installed" in record["error"]


def test_a_failing_command_is_an_error_record_with_its_output(tmp_path: Path) -> None:
    document = _document(
        tmp_path,
        command=[
            sys.executable,
            "-c",
            "import sys; sys.stderr.write('exploded'); sys.exit(3)",
        ],
    )
    record = _run(document, tmp_path / "working")
    assert record["state"] == "error"
    assert "exited 3" in record["error"] and "exploded" in record["error"]
    assert record["command"]["returncode"] == 3


def test_a_declared_resume_reports_the_search_vocabulary(tmp_path: Path) -> None:
    output = tmp_path / "working"
    document = _document(tmp_path, resume_from="ckpt-0007")
    record = _run(document, output)
    assert record["resume_state"] == "not-a-resume", (
        "without proof the checkpoint loaded, the honest answer is not-a-resume"
    )
    (output / RESUME_STATE_NAME).write_text(
        json.dumps({"resume_state": "resumed"}), encoding="utf-8"
    )
    record = _run(document, output)
    assert record["resume_state"] == "resumed"


def test_no_declared_resume_records_no_resume_state(tmp_path: Path) -> None:
    record = _run(_document(tmp_path), tmp_path / "working")
    assert record["resume_state"] is None


def test_bytecode_caches_are_not_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "working"
    (output / "__pycache__").mkdir(parents=True)
    (output / "__pycache__" / "module.cpython-312.pyc").write_bytes(b"cache")
    record = _run(_document(tmp_path), output)
    assert record["artifact_manifest"] == []
