"""Adversarial CPU-only verification of original guard.v3 stable evidence reads."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys

import pytest

from chowder_batch import reap_original600_guardv3_stable_snapshot_20261029 as snap


pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows NTFS share-mode contract")


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "original"
    root.mkdir()
    (root / "owner.json").write_bytes(b"immutable owner\n")
    task = root / "task-000"
    task.mkdir()
    (task / "receipt.json").write_bytes(b'{"strict":false}\n')
    return root, task / "receipt.json"


def _attempt_write(path: Path) -> subprocess.CompletedProcess[str]:
    code = (
        "from pathlib import Path; import sys\n"
        "try:\n"
        "    Path(sys.argv[1]).write_bytes(b'attacker replacement')\n"
        "except PermissionError:\n"
        "    sys.exit(0)\n"
        "sys.exit(4)\n"
    )
    return subprocess.run([sys.executable, "-c", code, str(path)],
                          capture_output=True, text=True, timeout=15)


def test_windows_source_handle_denies_competing_writer(tmp_path: Path) -> None:
    root, receipt = _fixture(tmp_path)
    before = receipt.read_bytes()
    with snap._locked_read(receipt, directory=False) as stream:
        assert stream.read() == before
        attempted = _attempt_write(receipt)
        assert attempted.returncode == 0, attempted.stderr
    assert receipt.read_bytes() == before
    assert root.is_dir()


def test_snapshot_is_byte_identical_and_unwritable_during_scoring(tmp_path: Path) -> None:
    root, receipt = _fixture(tmp_path)
    original_sha = hashlib.sha256(receipt.read_bytes()).hexdigest()
    with snap.stable_snapshot(root, tmp_path) as stable:
        copied = stable / "task-000" / "receipt.json"
        assert copied != receipt
        assert hashlib.sha256(copied.read_bytes()).hexdigest() == original_sha
        attempted = _attempt_write(copied)
        assert attempted.returncode == 0, attempted.stderr
        assert copied.read_bytes() == receipt.read_bytes()
    assert receipt.exists() and hashlib.sha256(receipt.read_bytes()).hexdigest() == original_sha
    assert not list(tmp_path.glob(".reap-guardv3-readonly-*"))


def test_original_bytes_changed_during_scoring_fail_closed(tmp_path: Path) -> None:
    root, receipt = _fixture(tmp_path)
    with pytest.raises(ValueError, match="evidence bytes changed during scoring"):
        with snap.stable_snapshot(root, tmp_path) as stable:
            assert (stable / "task-000" / "receipt.json").read_bytes() == receipt.read_bytes()
            receipt.write_bytes(b"substituted while original scorer is running")
    assert not list(tmp_path.glob(".reap-guardv3-readonly-*"))


def test_original_inventory_added_during_scoring_fail_closed(tmp_path: Path) -> None:
    root, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="inventory changed during scoring"):
        with snap.stable_snapshot(root, tmp_path):
            (root / "new-unscored-receipt.json").write_bytes(b"not part of snapshot")
    assert not list(tmp_path.glob(".reap-guardv3-readonly-*"))


def test_empty_original_active_task_directory_is_preserved(tmp_path: Path) -> None:
    root, _ = _fixture(tmp_path)
    (root / "task-001").mkdir()
    with snap.stable_snapshot(root, tmp_path) as stable:
        assert (stable / "task-001").is_dir()
        assert list((stable / "task-001").iterdir()) == []


def test_original_source_symlink_refused(tmp_path: Path) -> None:
    root, receipt = _fixture(tmp_path)
    link = root / "redirect.json"
    try:
        link.symlink_to(receipt)
    except OSError:
        pytest.skip("Windows symlink privilege unavailable")
    with pytest.raises(ValueError, match="reparse point"):
        with snap.stable_snapshot(root, tmp_path):
            pytest.fail("linked evidence was admitted")


def test_hardlinked_source_alias_refused(tmp_path: Path) -> None:
    root, receipt = _fixture(tmp_path)
    alias = root / "alias.json"
    try:
        os.link(receipt, alias)
    except OSError:
        pytest.skip("hardlinks unavailable on this volume")
    with pytest.raises(ValueError, match="hardlink aliases"):
        with snap.stable_snapshot(root, tmp_path):
            pytest.fail("hardlinked evidence was admitted")


def test_bounded_original_evidence_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, _ = _fixture(tmp_path)
    monkeypatch.setattr(snap, "_MAX_TOTAL", 4)
    with pytest.raises(ValueError, match="bounded snapshot limit"):
        with snap.stable_snapshot(root, tmp_path):
            pytest.fail("oversized source admitted")
