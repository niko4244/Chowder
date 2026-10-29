"""Adversarial, read-only tests for the immutable historical v3 scorer."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import types

import pytest

from chowder_batch import reap_original600_guardv3_git_pinned_reader_20261027 as reader


def test_historical_git_source_survives_later_runner_commit() -> None:
    sources = reader.source_bytes()
    for name, digest in reader.PINNED.items():
        assert reader.sha(sources[name]) == digest
    current = reader.REPO / "chowder_batch/reap_q4_automationbench_full600_resumable_20261027.py"
    assert reader.sha(current.read_bytes()) != reader.PINNED[
        "chowder_batch/reap_q4_automationbench_full600_resumable_20261027.py"
    ]


def test_git_source_tamper_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    actual = reader.git_bytes

    def corrupt(path: str) -> bytes:
        raw = actual(path)
        return raw + b"\n# attacker drift" if path in reader.PINNED else raw

    monkeypatch.setattr(reader, "git_bytes", corrupt)
    with pytest.raises(ValueError, match="historical Git source SHA mismatch"):
        reader.source_bytes()


def test_auxiliary_dependency_drift_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    actual = reader.git_bytes

    def corrupt(path: str) -> bytes:
        raw = actual(path)
        return raw + b"\n# drift" if path in reader.IMPORTED_AUXILIARY else raw

    monkeypatch.setattr(reader, "git_bytes", corrupt)
    with pytest.raises(ValueError, match="auxiliary dependency drift"):
        reader.source_bytes()


def test_transitive_context_budget_source_drift_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Historical ctx64k imports this executable helper transitively."""
    path = "chowder_batch/reap_context_budget_20261026.py"
    actual = reader.git_bytes

    def corrupt(name: str) -> bytes:
        raw = actual(name)
        return raw + b"\n# modified context budget\n" if name == path else raw

    monkeypatch.setattr(reader, "git_bytes", corrupt)
    with pytest.raises(ValueError, match="auxiliary dependency drift"):
        reader.source_bytes()


def test_coordinated_git_replacement_and_auxiliary_worktree_drift_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Matching forged Git/worktree bytes must not redefine the 84ff970 source."""
    target = "chowder_batch/reap_context_budget_20261026.py"
    actual_git = reader.git_bytes
    actual_read = Path.read_bytes
    forged = actual_git(target) + b"\n# attacker-controlled original dependency\n"

    def forged_git(path: str) -> bytes:
        return forged if path == target else actual_git(path)

    def forged_read(path: Path) -> bytes:
        return forged if path == reader.REPO / target else actual_read(path)

    monkeypatch.setattr(reader, "git_bytes", forged_git)
    monkeypatch.setattr(Path, "read_bytes", forged_read)
    with pytest.raises(ValueError, match="auxiliary dependency drift"):
        reader.source_bytes()


def test_historical_git_bytes_ignore_replacement_refs(monkeypatch: pytest.MonkeyPatch) -> None:
    actual = reader.subprocess.check_output
    seen = []

    def checked(*args, **kwargs):
        seen.append(kwargs.get("env", {}).get("GIT_NO_REPLACE_OBJECTS"))
        return actual(*args, **kwargs)

    monkeypatch.setattr(reader.subprocess, "check_output", checked)
    path = "chowder_batch/reap_context_budget_20261026.py"
    assert reader.sha(reader.git_bytes(path)) == reader.AUXILIARY_SHA256[path]
    assert seen == ["1"]


def test_later_worktree_helper_edits_do_not_change_git_pinned_historical_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = "chowder_batch/reap_context_budget_20261026.py"
    original_read = Path.read_bytes

    def forged_read(self: Path) -> bytes:
        if self == reader.REPO / path:
            return original_read(self) + b"\n# legitimate later helper edit\n"
        return original_read(self)

    monkeypatch.setattr(Path, "read_bytes", forged_read)
    sources = reader.source_bytes()
    assert reader.sha(sources[path]) == reader.AUXILIARY_SHA256[path]


@pytest.mark.parametrize("auxiliary", [
    "reap_context_budget_20261026", "reap_q8_automationbench_kaggle_20261026",
])
def test_historical_loader_refuses_preimported_auxiliary(auxiliary: str) -> None:
    """A preloaded executable module cannot bypass pinned source bytes."""
    code = (
        "import importlib, sys\n"
        "importlib.import_module('chowder_batch.' + sys.argv[1])\n"
        "from chowder_batch import reap_original600_guardv3_git_pinned_reader_20261027 as r\n"
        "try:\n"
        "    r.historical_modules()\n"
        "except ValueError as exc:\n"
        "    if 'fresh interpreter required' in str(exc):\n"
        "        sys.exit(0)\n"
        "sys.exit(3)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, auxiliary], cwd=reader.REPO,
        capture_output=True, text=True, timeout=25,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_historical_loader_refuses_shadow_namespace_path() -> None:
    code = (
        "import sys, chowder_batch\n"
        "from chowder_batch import reap_original600_guardv3_git_pinned_reader_20261027 as r\n"
        "chowder_batch.__path__ = [str(r.REPO / 'chowder_batch'), 'C:/untrusted-shadow']\n"
        "try:\n"
        "    r.historical_modules()\n"
        "except ValueError as exc:\n"
        "    if 'namespace source path' in str(exc):\n"
        "        sys.exit(0)\n"
        "sys.exit(3)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=reader.REPO,
        capture_output=True, text=True, timeout=25,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_historical_loader_refuses_preimported_module(monkeypatch: pytest.MonkeyPatch) -> None:
    name = "chowder_batch.reap_original600_live_independent_audit_20261027"
    monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    with pytest.raises(ValueError, match="fresh interpreter required"):
        reader._load(name, b"pass\n")


def test_non_original_root_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="only the original ordinary"):
        reader.inspect(tmp_path)


def test_original_owner_drift_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(reader, "ROOT", tmp_path)
    (tmp_path / "owner.json").write_text('{"schema":"forged"}', encoding="utf-8")
    with pytest.raises(ValueError, match="owner SHA drift"):
        reader.inspect(tmp_path)


def test_original_owner_symlink_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(reader, "ROOT", tmp_path)
    target = tmp_path / "actual.json"
    target.write_text("{}", encoding="utf-8")
    try:
        (tmp_path / "owner.json").symlink_to(target)
    except OSError:
        pytest.skip("native symlink privilege unavailable")
    with pytest.raises(ValueError, match="owner SHA drift"):
        reader.inspect(tmp_path)


def test_actual_historical_reader_is_read_only_and_fail_closed() -> None:
    owner = reader.ROOT / "owner.json"
    before = reader.sha(owner.read_bytes())
    result = subprocess.run(
        [sys.executable, "-m", "chowder_batch.reap_original600_guardv3_git_pinned_reader_20261027"],
        cwd=reader.REPO, capture_output=True, text=True, timeout=120, check=True,
    )
    audit = json.loads(result.stdout)
    assert audit["completed_scored_tasks"] >= 23
    assert audit["official_complete_600_claim_allowed"] is False
    assert audit["native_sessions_pooled_for_frozen_comparison"] is False
    assert audit["checkpoint_may_span_distinct_native_sessions"] is True
    assert audit["frozen_gen0_status"] == "BLOCKED"
    assert audit["chosen"] is None and audit["release_approved"] is False
    assert reader.sha(owner.read_bytes()) == before


def test_existing_or_non_f_snapshot_refused(tmp_path: Path) -> None:
    existing = tmp_path / "existing.json"
    existing.write_text("do not overwrite", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "chowder_batch.reap_original600_guardv3_git_pinned_reader_20261027",
         "--snapshot", str(existing)],
        cwd=reader.REPO, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 2
    assert json.loads(result.stdout)["status"] == "REFUSED_HISTORICAL_SOURCE_OR_EVIDENCE"
    assert existing.read_text(encoding="utf-8") == "do not overwrite"
