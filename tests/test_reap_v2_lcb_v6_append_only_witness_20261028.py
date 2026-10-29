"""Adversarial CPU-only tests; never run generated LiveCodeBench code."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from chowder_batch import reap_v2_lcb_v6_append_only_witness_20261028 as witness

CANONICAL = [("a", "test.jsonl"), ("b", "test2.jsonl"),
             ("c", "test3.jsonl")]
FAILURE = (
    "URLError: <urlopen error [WinError 10061] "
    "No connection could be made because the target machine actively refused it>"
)


def row(qid: str, source: str, *, error: str | None = None,
        code: str = "print(1)") -> dict:
    result = {
        "question_id": qid,
        "source_file": source,
        "task": "LCB_generation",
        "prompt_style": "OpenAIChat",
        "max_tokens": 4096,
        "temperature": 0.2,
        "top_p": 0.95,
        "raw": "" if error else f"candidate confidential text\n```python\n{code}\n```",
        "code": "" if error else code,
        "error": error,
        "elapsed_s": 1.0,
    }
    if error is None:
        result.update(finish_reason="stop", prompt_tokens=10, completion_tokens=20)
    return result


def encode(*rows: dict) -> bytes:
    return b"".join((json.dumps(item) + "\n").encode() for item in rows)


def make(initial: bytes, later: bytes | None = None) -> dict:
    return witness.build_witness(
        CANONICAL, initial, initial if later is None else later,
        provenance={"source_git_commit": "pinned"},
    )


def test_successful_live_append_is_accepted_without_pooling() -> None:
    first = encode(row("a", "test.jsonl"), row("b", "test2.jsonl", error=FAILURE))
    later = first + encode(row("c", "test3.jsonl", error=FAILURE))
    result = make(first, later)
    assert result["prefix_rows"] == 2
    assert result["subsequent_bytes_beyond_prefix"] > 0
    assert result["transport_classification"]["connection_refused_rows"] == 1
    assert result["transport_classification"]["unresolved_source_cases"] == 2
    assert result["official_score"] is None
    assert result["frozen_gen0"] == {
        "status": "BLOCKED", "chosen": None, "release_approved": False,
    }
    assert not result["cross_session_pooling_allowed"]
    assert "candidate confidential text" not in json.dumps(result)


def test_inflight_partial_append_excluded_and_accounted() -> None:
    first = encode(row("a", "test.jsonl")) + b'{"question_id": "b"'
    result = make(first, first + b"}\n")
    assert result["prefix_rows"] == 1
    assert result["initial_inflight_tail_bytes_excluded"] == 19
    assert result["subsequent_bytes_beyond_prefix"] == 21


def test_empty_code_without_transport_error_is_still_response() -> None:
    result = make(encode(row("a", "test.jsonl", code="")))
    assert result["transport_classification"]["successful_transport_rows"] == 1
    assert result["transport_classification"]["model_responses_with_empty_code"] == 1
    assert result["transport_classification"]["failed_transport_rows_not_scored"] == 0


def test_missing_explicit_error_key_cannot_be_witnessed_as_success() -> None:
    item = row("a", "test.jsonl")
    del item["error"]
    with pytest.raises(ValueError, match="missing original transport status"):
        make(encode(item))


def test_forged_extracted_code_from_empty_content_cannot_be_witnessed() -> None:
    item = row("a", "test.jsonl")
    item["raw"] = ""
    with pytest.raises(ValueError, match="disagrees with original extraction"):
        make(encode(item))


@pytest.mark.parametrize("initial,later", [
    (encode(row("a", "test.jsonl")), b""),
    (encode(row("a", "test.jsonl")), encode(row("b", "test2.jsonl"))),
    (encode(row("a", "test.jsonl")), encode(row("a", "test.jsonl"))[:-1]),
    (encode(row("a", "test.jsonl")), b"corrupt" + encode(row("a", "test.jsonl"))),
])
def test_truncation_or_rewrite_refused(initial: bytes, later: bytes) -> None:
    with pytest.raises(ValueError, match="truncated or rewritten"):
        make(initial, later)


@pytest.mark.parametrize("raw", [
    b"",
    b'{"question_id":"a"}',
    b"\n",
    b'{"question_id":"a"}\n\n',
    b'{"question_id":"a", "question_id":"b"}\n',
    b'{"question_id":"a", "nested":{"x":1,"x":2}}\n',
    b'{"question_id":"a", "v":NaN}\n',
    b"\xff\n",
])
def test_empty_malformed_ambiguous_or_nonfinite_refused(raw: bytes) -> None:
    with pytest.raises(ValueError):
        make(raw)


def test_overlarge_unterminated_tail_refused() -> None:
    with pytest.raises(ValueError, match="excessive"):
        make(encode(row("a", "test.jsonl")) + b"x" * (witness.MAX_INFLIGHT_TAIL + 1))


@pytest.mark.parametrize("change", [
    {"question_id": "b"},
    {"source_file": "test2.jsonl"},
    {"task": "LCB_other"},
    {"prompt_style": "Other"},
    {"temperature": 0.0},
    {"top_p": 1.0},
    {"max_tokens": 2048},
    {"code": None},
    {"raw": None},
    {"error": "failure", "raw": "candidate"},
])
def test_source_prompt_transport_tampering_refused(change: dict) -> None:
    item = row("a", "test.jsonl")
    item.update(change)
    with pytest.raises(ValueError):
        make(encode(item))


def test_completed_rows_cannot_exceed_canonical_population() -> None:
    raw = encode(
        row("a", "test.jsonl"),
        row("b", "test2.jsonl"),
        row("c", "test3.jsonl"),
        row("d", "test4.jsonl"),
    )
    with pytest.raises(ValueError, match="excessive"):
        make(raw)


def test_partial_tail_does_not_allow_invalid_completed_row() -> None:
    raw = encode(row("a", "test.jsonl")) + b'{"question_id": "a"}\npartial'
    with pytest.raises(ValueError):
        make(raw)


def test_verify_append_only_requires_newline_terminated_prefix() -> None:
    with pytest.raises(ValueError, match="complete original prefix"):
        witness.verify_append_only(b"partial", b"partial-more")


def test_exact_official_python_runtime_copy(tmp_path: Path) -> None:
    source = tmp_path / "source"
    runtime = tmp_path / "runtime"
    for root in (source, runtime):
        folder = root / "lcb_runner"
        folder.mkdir(parents=True)
        (folder / "__init__.py").write_text("", encoding="utf-8")
        (folder / "eval.py").write_text("def score(): return 1\n", encoding="utf-8")
    assert witness.verify_runtime_copy(source, runtime) == 2
    extra = runtime / "injected.py"
    extra.write_text("print('unapproved')\n", encoding="utf-8")
    with pytest.raises(ValueError, match="file set drift"):
        witness.verify_runtime_copy(source, runtime)
    extra.unlink()
    (runtime / "lcb_runner" / "eval.py").write_text(
        "def score(): return 0\n", encoding="utf-8",
    )
    with pytest.raises(ValueError, match="SHA drift"):
        witness.verify_runtime_copy(source, runtime)
    (runtime / "lcb_runner" / "eval.py").unlink()
    with pytest.raises(ValueError, match="file set drift"):
        witness.verify_runtime_copy(source, runtime)


@pytest.mark.parametrize("marked", ["runtime", "runtime/lcb_runner", "runtime/lcb_runner/nested"])
def test_live_witness_runtime_copy_refuses_ntfs_junction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, marked: str,
) -> None:
    source = tmp_path / "source"
    runtime = tmp_path / "runtime"
    for tree in (source, runtime):
        package = tree / "lcb_runner"
        (package / "nested").mkdir(parents=True)
        (package / "nested" / "__init__.py").write_text("# original\n", encoding="utf-8")
    original_lstat = witness.lcb.os.lstat

    def reparse_lstat(path, *args, **kwargs):
        if str(path) == str(tmp_path / marked):
            return SimpleNamespace(st_file_attributes=0x400)
        return original_lstat(path, *args, **kwargs)

    monkeypatch.setattr(witness.lcb.os, "lstat", reparse_lstat)
    with pytest.raises(ValueError, match="reparse"):
        witness.verify_runtime_copy(source, runtime)


@pytest.mark.parametrize("source_attack", ["ignored", "assume-unchanged", "skip-worktree"])
def test_live_source_witness_refuses_unpinned_python_even_with_clean_git_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_attack: str,
) -> None:
    """A research witness must not call source/runtime parity Git authentication."""
    source = tmp_path / "source"
    runtime = tmp_path / "runtime"
    for tree in (source, runtime):
        package = tree / "lcb_runner"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("# pinned source\n", encoding="utf-8")
    (source / ".gitignore").write_text("/lcb_runner/ignored.py\n", encoding="utf-8")

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(source), *args], stderr=subprocess.PIPE, timeout=15,
        )

    git("init", "-q")
    git("add", ".gitignore", "lcb_runner/__init__.py")
    git("-c", "user.name=Qualification Test", "-c", "user.email=test@example.invalid",
        "commit", "-qm", "Pinned original")
    if source_attack == "ignored":
        for tree in (source, runtime):
            (tree / "lcb_runner" / "ignored.py").write_text(
                "# unreviewed executable module\n", encoding="utf-8",
            )
    else:
        git("update-index", "--" + source_attack, "--", "lcb_runner/__init__.py")
        for tree in (source, runtime):
            (tree / "lcb_runner" / "__init__.py").write_text(
                "# hidden tracked source modification\n", encoding="utf-8",
            )
    assert not git("status", "--porcelain").strip()
    generator = tmp_path / "generator.py"
    evaluator = tmp_path / "evaluator.py"
    generator.write_text("# generator\n", encoding="utf-8")
    evaluator.write_text("# evaluator\n", encoding="utf-8")
    monkeypatch.setattr(witness.lcb, "SOURCE", source)
    monkeypatch.setattr(witness.lcb, "RUNTIME", runtime)
    monkeypatch.setattr(witness.lcb, "SOURCE_COMMIT", git("rev-parse", "HEAD").decode().strip())
    monkeypatch.setattr(witness.lcb, "GENERATOR", generator)
    monkeypatch.setattr(witness.lcb, "GENERATOR_SHA", witness.lcb.sha(generator))
    monkeypatch.setattr(witness.lcb, "EVALUATOR", evaluator)
    monkeypatch.setattr(witness.lcb, "EVALUATOR_SHA", witness.lcb.sha(evaluator))
    monkeypatch.setattr(witness.lcb, "DATA", tmp_path / "missing-dataset")
    with pytest.raises(ValueError, match="(pinned Git|Git blob)"):
        witness.source_provenance()


def test_source_bound_canonical_single_pass_matches_exact_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "test.jsonl"
    raw = b'{"question_id":"a","input":"first"}\n{"question_id":"b","input":"second"}\n'
    source.write_bytes(raw)
    expected = hashlib.sha256(raw).hexdigest()

    def reject_second_read(_path: Path) -> list[dict]:
        raise AssertionError("source must never be parsed by a second read")

    monkeypatch.setattr(witness.lcb, "read_jsonl", reject_second_read)
    entries, digest = witness.read_source_bound_canonical(source, "test.jsonl", expected)
    assert entries == [("a", "test.jsonl"), ("b", "test.jsonl")]
    assert digest == expected
    source.write_bytes(raw.replace(b"first", b"other"))
    with pytest.raises(ValueError, match="SHA drift"):
        witness.read_source_bound_canonical(source, "test.jsonl", expected)


@pytest.mark.parametrize("raw", [
    b'{"question_id":"a","question_id":"b"}\n',
    b'{"question_id":"a","nested":{"x":1,"x":2}}\n',
    b'{"question_id":"a","input":NaN}\n',
    b'{"question_id":"a"}',
    b'{"question_id":"a"}\n\n',
    b'{"question_id":""}\n',
    b'\xff\n',
])
def test_source_bound_canonical_refuses_ambiguous_or_partial_rows(
    tmp_path: Path, raw: bytes,
) -> None:
    source = tmp_path / "test.jsonl"
    source.write_bytes(raw)
    with pytest.raises(ValueError):
        witness.read_source_bound_canonical(
            source, "test.jsonl", hashlib.sha256(raw).hexdigest(),
        )


def test_unverified_bytecode_is_disclosed_not_execution_attested(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    cache = runtime / "lcb_runner" / "__pycache__"
    cache.mkdir(parents=True)
    bytecode = cache / "evaluator.cpython-313.pyc"
    bytecode.write_bytes(b"unverified executable bytecode")
    archive = runtime / "optional.zip"
    archive.write_bytes(b"unverified import archive")
    report = witness.inventory_unverified_runtime_artifacts(runtime)
    assert report["unverified_importable_non_source_artifact_count"] == 2
    assert report["runtime_execution_provenance_independently_attested"] is False
    assert report["unverified_importable_non_source_artifact_sha256"] == {
        "lcb_runner/__pycache__/evaluator.cpython-313.pyc": witness._sha_bytes(
            bytecode.read_bytes()),
        "optional.zip": witness._sha_bytes(archive.read_bytes()),
    }
    assert "unverified executable bytecode" not in json.dumps(report)


def test_empty_non_source_inventory_is_still_not_execution_attestation(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "module.py").write_text("pass\n", encoding="utf-8")
    report = witness.inventory_unverified_runtime_artifacts(runtime)
    assert report["unverified_importable_non_source_artifact_count"] == 0
    assert report["runtime_execution_provenance_independently_attested"] is False


def test_final_seal_verifies_file_identity_and_prefix(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    prefix = encode(row("a", "test.jsonl"))
    path.write_bytes(prefix)
    info = path.stat()
    receipt = {
        "prefix_bytes": len(prefix),
        "prefix_sha256": witness._sha_bytes(prefix),
        "journal_file_identity": {"device": info.st_dev, "inode": info.st_ino},
    }
    witness._verify_prefix_still_present(path, receipt)
    with path.open("ab") as stream:
        stream.write(encode(row("b", "test2.jsonl")))
    witness._verify_prefix_still_present(path, receipt)
    receipt["journal_file_identity"]["inode"] += 1
    with pytest.raises(ValueError, match="identity changed"):
        witness._verify_prefix_still_present(path, receipt)


def test_final_seal_refuses_rewritten_prefix(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    prefix = encode(row("a", "test.jsonl"))
    path.write_bytes(prefix)
    info = path.stat()
    receipt = {
        "prefix_bytes": len(prefix),
        "prefix_sha256": witness._sha_bytes(prefix),
        "journal_file_identity": {"device": info.st_dev, "inode": info.st_ino},
    }
    path.write_bytes(prefix.replace(b"candidate", b"malicious"))
    with pytest.raises(ValueError, match="prefix changed"):
        witness._verify_prefix_still_present(path, receipt)


def test_exclusive_snapshot_publishes_only_verified_prefix(tmp_path: Path) -> None:
    journal = tmp_path / "journal.jsonl"
    prefix = encode(row("a", "test.jsonl"))
    journal.write_bytes(prefix)
    info = journal.stat()
    receipt = {
        "prefix_bytes": len(prefix),
        "prefix_sha256": witness._sha_bytes(prefix),
        "journal_file_identity": {"device": info.st_dev, "inode": info.st_ino},
    }
    target = tmp_path / "sealed.json"
    witness._seal_exclusive_snapshot(target, journal, receipt)
    assert json.loads(target.read_text()) == receipt
    assert not list(tmp_path.glob("*.pending-*"))
    with pytest.raises(FileExistsError):
        witness._seal_exclusive_snapshot(target, journal, receipt)
    assert json.loads(target.read_text()) == receipt


def test_exclusive_snapshot_does_not_publish_if_final_check_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal = tmp_path / "journal.jsonl"
    prefix = encode(row("a", "test.jsonl"))
    journal.write_bytes(prefix)
    info = journal.stat()
    receipt = {
        "prefix_bytes": len(prefix),
        "prefix_sha256": witness._sha_bytes(prefix),
        "journal_file_identity": {"device": info.st_dev, "inode": info.st_ino},
    }
    original = witness._verify_prefix_still_present
    checks = 0

    def reject_second(path: Path, data: dict) -> None:
        nonlocal checks
        checks += 1
        original(path, data)
        if checks == 2:
            raise ValueError("prefix changed during final verification")

    monkeypatch.setattr(witness, "_verify_prefix_still_present", reject_second)
    target = tmp_path / "unsealed.json"
    with pytest.raises(ValueError, match="final verification"):
        witness._seal_exclusive_snapshot(target, journal, receipt)
    assert not target.exists()
    assert not list(tmp_path.glob("*.pending-*"))
