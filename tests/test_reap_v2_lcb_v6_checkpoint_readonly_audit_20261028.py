"""Adversarial accounting tests; never execute generated Python programs."""

from __future__ import annotations

import json
import hashlib
import subprocess
from pathlib import Path

import pytest

from chowder_batch import reap_v2_lcb_v6_checkpoint_readonly_audit_20261028 as audit


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def evidence(tmp_path):
    data = tmp_path / "raw"
    data.mkdir()
    for index, name in enumerate(audit.FILES):
        _write(data / name, [{"question_id": f"id{index}"}])
    generated = tmp_path / "generated.jsonl"
    _write(generated, [
        {"question_id": f"id{i}", "source_file": audit.FILES[i],
         "task": "LCB_generation", "prompt_style": "OpenAIChat",
         "max_tokens": 4096, "temperature": 0.2, "top_p": 0.95,
         "code": "print(1)" if i != 2 else "", "error": None}
        for i in range(4)
    ])
    metrics = tmp_path / "metrics.json"
    report = {
        "release": "release_v6",
        "dataset": "livecodebench/code_generation_lite",
        "problems_total": 6,
        "scored_population": 3,
        "missing": 3,
        "empty_code": 1,
        "transport_errors": 0,
        "metrics": {"pass@1": 1 / 3, "detail": {
            "pass@1": {"0": 1.0, "1": 0.0, "2": 0.0},
        }},
        "detail": [
            {"question_id": "id0", "score": [True, True]},
            {"question_id": "id1", "score": [False]},
            {"question_id": "id2", "score": [-4]},
        ],
    }
    metrics.write_text(json.dumps(report), encoding="utf-8")
    return data, generated, metrics


def _mutate_json(path, mutator):
    value = json.loads(path.read_text(encoding="utf-8"))
    mutator(value)
    path.write_text(json.dumps(value), encoding="utf-8")


def _mutate_jsonl(path, mutator):
    rows = audit.read_jsonl(path)
    mutator(rows)
    _write(path, rows)


def test_genuine_prefix_internal_consistency_is_not_release(evidence):
    result = audit.checkpoint_consistency(*evidence)
    assert (result["source_population"], result["generated_prefix"],
            result["scored_prefix"], result["correct"]) == (6, 4, 3, 1)
    assert result["pass_at_1_scored_prefix"] == 1 / 3
    assert result["full_official_score"] is None
    assert result["cpu_code_isolation_independently_attested"] is False
    assert result["original_serving_session_independently_attested"] is False
    assert result["frozen_gen0_status"] == "BLOCKED"
    assert result["chosen"] is None and result["release_approved"] is False


@pytest.mark.parametrize("change", [
    lambda x: x[1].update(question_id="id9"),
    lambda x: x[1].update(question_id="id0"),
    lambda x: x[0].update(source_file="test6.jsonl"),
    lambda x: x[0].update(temperature=0.7),
    lambda x: x[0].update(prompt_style="custom"),
    lambda x: x[0].update(max_tokens=1024),
])
def test_rejects_generated_source_and_prompt_drift(evidence, change):
    _mutate_jsonl(evidence[1], change)
    with pytest.raises(ValueError):
        audit.checkpoint_consistency(*evidence)


@pytest.mark.parametrize("change", [
    lambda x: x.update(missing=2),
    lambda x: x.update(empty_code=0),
    lambda x: x.update(scored_population=4),
    lambda x: x["metrics"].update(**{"pass@1": 0.8}),
    lambda x: x["metrics"]["detail"]["pass@1"].update(**{"1": 1.0}),
    lambda x: x["detail"][1].update(score=[True]),
    lambda x: x["detail"][2].update(score=[-2]),
    lambda x: x["detail"][2].update(question_id="id5"),
])
def test_rejects_checkpoint_tampering(evidence, change):
    _mutate_json(evidence[2], change)
    with pytest.raises(ValueError):
        audit.checkpoint_consistency(*evidence)


def test_rejects_duplicate_canonical_source_id(evidence):
    _mutate_jsonl(evidence[0] / audit.FILES[1],
                  lambda rows: rows[0].update(question_id="id0"))
    with pytest.raises(ValueError, match="duplicate"):
        audit.checkpoint_consistency(*evidence)


def test_rejects_truncated_generation_line(evidence):
    with evidence[1].open("a", encoding="utf-8") as stream:
        stream.write('{"question_id":')
    with pytest.raises(json.JSONDecodeError):
        audit.checkpoint_consistency(*evidence)


def test_rejects_missing_source_file(evidence):
    (evidence[0] / audit.FILES[3]).unlink()
    with pytest.raises(ValueError, match="missing"):
        audit.checkpoint_consistency(*evidence)


def test_checkpoint_hash_and_metrics_parse_use_identical_bytes(evidence, monkeypatch):
    """A forged second metrics read must not pair with the original file hash."""
    metrics = evidence[2]
    original_digest = hashlib.sha256(metrics.read_bytes()).hexdigest()
    forged = json.loads(metrics.read_text())
    forged["metrics"]["pass@1"] = 1.0
    forged["metrics"]["detail"]["pass@1"] = {str(i): 1.0 for i in range(3)}
    forged["detail"] = [
        {"question_id": f"id{i}", "score": [True]} for i in range(3)
    ]
    original_read_text = Path.read_text

    def swapped_read_text(self, *args, **kwargs):
        if self == metrics:
            return json.dumps(forged)
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", swapped_read_text)
    result = audit.checkpoint_consistency(*evidence)
    assert result["correct"] == 1
    assert result["checkpoint_sha256"] == original_digest


def test_checkpoint_canonical_source_never_uses_unhashed_second_read(
    evidence, monkeypatch,
):
    """Unscored source IDs cannot be replaced after a different source SHA."""
    original_read_jsonl = audit.read_jsonl

    def forged_second_read(path):
        if path == evidence[0] / audit.FILES[-1]:
            return [{"question_id": "fabricated-unscored-task"}]
        return original_read_jsonl(path)

    monkeypatch.setattr(audit, "read_jsonl", forged_second_read)
    result = audit.checkpoint_consistency(*evidence, include_canonical=True)
    assert result["_canonical_source"][-1] == ("id5", audit.FILES[-1])
    assert result["source_population"] == 6


def test_checkpoint_rejects_duplicate_canonical_json_keys(evidence):
    path = evidence[0] / audit.FILES[-1]
    path.write_bytes(b'{"question_id":"id5","question_id":"forged"}\n')
    with pytest.raises(ValueError, match="duplicate"):
        audit.checkpoint_consistency(*evidence)


def test_runtime_python_inventory_must_be_symmetric_and_source_bound(tmp_path):
    source = tmp_path / "source"
    runtime = tmp_path / "runtime"
    original = source / "lcb_runner"
    copied = runtime / "lcb_runner"
    original.mkdir(parents=True)
    copied.mkdir(parents=True)
    (original / "__init__.py").write_text("# pinned original\n", encoding="utf-8")
    (copied / "__init__.py").write_text("# pinned original\n", encoding="utf-8")
    audit.verify_runtime_python_sources(source, runtime)

    # An extra importable runtime module previously escaped the one-way
    # source inventory comparison despite having no source-controlled peer.
    extra = copied / "injected.py"
    extra.write_text("# unreviewed runtime code\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory drift"):
        audit.verify_runtime_python_sources(source, runtime)
    extra.unlink()

    # A forged timestamp-valid .pyc can execute despite matching .py sources.
    cache = copied / "__pycache__"
    cache.mkdir()
    (cache / "__init__.cpython-311.pyc").write_bytes(b"unreviewed bytecode")
    with pytest.raises(ValueError, match="unreviewed importable bytecode"):
        audit.verify_runtime_python_sources(source, runtime)
    (cache / "__init__.cpython-311.pyc").unlink()
    cache.rmdir()

    # An empty/sparse source tree must not make the source loop vacuously pass.
    (original / "__init__.py").unlink()
    with pytest.raises(ValueError, match="inventory drift"):
        audit.verify_runtime_python_sources(source, runtime)

    (original / "__init__.py").write_text("# pinned original\n", encoding="utf-8")
    (copied / "__init__.py").write_text("# altered runtime\n", encoding="utf-8")
    with pytest.raises(ValueError, match="runtime source drift"):
        audit.verify_runtime_python_sources(source, runtime)


@pytest.mark.parametrize("relative", [
    "sitecustomize.py",
    "usercustomize.py",
    "unreviewed.py",
    "injected.pyw",
    "startup.pth",
    "imports.zip",
    "imports.pyz",
    "imports.egg",
    "imports.whl",
    "legacy.pyo",
    "lcb_runner/startup.pth",
    "lcb_runner/imports.zip",
])
def test_runtime_refuses_unreviewed_executable_import_paths(tmp_path, relative):
    source = tmp_path / "source"
    runtime = tmp_path / "runtime"
    for tree in (source, runtime):
        package = tree / "lcb_runner"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("# original\n", encoding="utf-8")
    injected = runtime / relative
    injected.parent.mkdir(parents=True, exist_ok=True)
    injected.write_bytes(b"unreviewed executable import path")
    with pytest.raises(ValueError, match="unreviewed.*(source|import|bytecode)"):
        audit.verify_runtime_python_sources(source, runtime)


@pytest.mark.parametrize("relative", [
    "lcb_runner/ignored.py",
    "lcb_runner/nested/ignored.py",
])
def test_git_pinned_runtime_refuses_ignored_source_python_even_if_copied(
    tmp_path, relative,
):
    """A clean Git status and matching runtime bytes cannot authorize ignored code."""
    source = tmp_path / "source"
    runtime = tmp_path / "runtime"
    for tree in (source, runtime):
        package = tree / "lcb_runner"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("# original\n", encoding="utf-8")
    (source / ".gitignore").write_text(f"/{relative}\n", encoding="utf-8")

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(source), *args], stderr=subprocess.PIPE,
            timeout=15,
        )

    git("init", "-q")
    git("add", ".gitignore", "lcb_runner/__init__.py")
    git("-c", "user.name=Qualification Test", "-c", "user.email=test@example.invalid",
        "commit", "-qm", "Pinned original")
    assert audit.verify_source_python_git_inventory(source) == 1

    for tree in (source, runtime):
        injected = tree / relative
        injected.parent.mkdir(parents=True, exist_ok=True)
        injected.write_text("# executable ignored source\n", encoding="utf-8")
    assert not git("status", "--porcelain").strip()
    audit.verify_runtime_python_sources(source, runtime)
    with pytest.raises(ValueError, match="pinned Git inventory"):
        audit.verify_source_python_git_inventory(source)
