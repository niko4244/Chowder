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
