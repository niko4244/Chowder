#!/usr/bin/env python3
"""Reusable machinery for mechanical qualification judges.

Every preregistration in ``docs/quals/`` ships with its own judge script ---
the thresholds are run-specific, frozen with the prereg, and belong in that
script. But the *pattern* around them is identical every time, and copying it
is how judges drift apart in the rules that matter:

- **Strict epistemics**: an artifact that is missing or unreadable is
  UNKNOWN, never an assumed pass. The exit code is 0 only when every
  threshold is PASS; any FAIL refuses, and any UNKNOWN refuses to certify.
- **Read-only judging**: the registry is opened with SQLite read-only URIs,
  files are hashed without writing, and the run directory is never mutated.
- **One verdict table**: threshold, check, status, detail --- rendered in a
  fixed-width table and finalized by the same rules every run.

Judges import from this module (the script's own directory is on ``sys.path``
when run directly, so this works from any working directory):

    from quals_harness import (
        FAIL, INFO, PASS, UNKNOWN, TERMINAL_STATUSES,
        Verdict, discover, finite_number, load_json, load_json_safe,
        open_registry_readonly, phase, report,
    )

The worked example is ``judge_rung3c_2026-09-15.py``; the conventions are
written up in ``docs/quals/HARNESS.md``.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

PASS, FAIL, UNKNOWN, INFO = "PASS", "FAIL", "UNKNOWN", "INFO"

# Statuses at which an experiment row is finished; a row carrying a result
# while in any other status is stranded (the defect class the registry audit
# exists to surface). Judges checking accounting or lifecycle closure reuse
# this set rather than re-listing terminal states by hand.
TERMINAL_STATUSES = frozenset({"passed", "failed", "rejected", "withdrawn"})


class Verdict:
    """The threshold table, finalized by one rule set."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, str]] = []

    def add(self, threshold: str, name: str, status: str, detail: str) -> None:
        if status not in (PASS, FAIL, UNKNOWN, INFO):
            raise ValueError(f"unknown verdict status {status!r}")
        self.rows.append((threshold, name, status, detail))

    def thresholds(self) -> list[tuple[str, str, str, str]]:
        """Rows that gate certification; INFO rows are recorded but not gating."""
        return [row for row in self.rows if row[0] != INFO]

    def finalize_status(self) -> str:
        statuses = {row[2] for row in self.thresholds()}
        if FAIL in statuses:
            return "REFUSED --- at least one threshold failed"
        if UNKNOWN in statuses:
            return "NOT CERTIFIED --- at least one threshold could not be decided from artifacts"
        return "QUALIFIED --- every threshold passes on measured evidence"

    def render(self) -> str:
        lines = ["threshold  check                                          verdict  detail"]
        for threshold, name, status, detail in self.rows:
            lines.append(f"{threshold:<10} {name:<46} {status:<8} {detail}")
        return "\n".join(lines)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    """Parse a JSON file, or None when missing/unreadable --- unknown, not a guess."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_json_safe(text: Any) -> Any:
    try:
        return json.loads(text) if text else None
    except ValueError:
        return None


def finite_number(value: Any) -> bool:
    """A real measured number. Bools are not numbers here: ``True`` is not a
    cost, and treating it as 1 has hidden lies before."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def phase(result: dict, name: str) -> dict:
    phases = (result.get("lifecycle") or {}).get("phases") or {}
    phase_data = phases.get(name)
    return phase_data if isinstance(phase_data, dict) else {}


def discover(run_root: Path) -> tuple[Path, list[tuple[Path, dict]], list[tuple[Path, dict]]]:
    """Locate the durable artifacts without writing anything.

    Returns ``(registry_path, train_results, eval_results)`` where each result
    list holds ``(worker-result.json path, parsed payload)`` pairs; unreadable
    payloads are skipped and surface downstream as UNKNOWN.
    """
    registry_path = run_root / "runs.db"
    runs_dir = run_root / ".chowder" / "runs"
    evals_dir = run_root / ".chowder" / "evals"
    train_results: list[tuple[Path, dict]] = []
    eval_results: list[tuple[Path, dict]] = []
    if runs_dir.is_dir():
        for result_path in sorted(runs_dir.glob("*/worker-result.json")):
            payload = load_json(result_path)
            if isinstance(payload, dict):
                train_results.append((result_path, payload))
    if evals_dir.is_dir():
        for result_path in sorted(evals_dir.glob("*/worker-result.json")):
            payload = load_json(result_path)
            if isinstance(payload, dict):
                eval_results.append((result_path, payload))
    return registry_path, train_results, eval_results


def open_registry_readonly(path: Path) -> sqlite3.Connection | None:
    """Read-only URI connection, or None when absent/unopenable."""
    if not path.is_file():
        return None
    try:
        return sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    except sqlite3.Error:
        return None


def report(run_root: Path, verdict: Verdict, train: list, evals: list, argv_len_ok: bool) -> int:
    """Print the standard report and return the process exit code.

    Exit 0 only when every gating threshold is PASS --- a FAIL refuses and an
    UNKNOWN refuses to certify. Judges call this from ``main`` so the exit
    discipline is identical everywhere.
    """
    if not argv_len_ok:
        print(__doc__)
        return 2
    if not run_root.is_dir():
        print(f"run root does not exist: {run_root}")
        return 2
    print(f"run root: {run_root}")
    print(f"training worker results: {len(train)}   evaluation worker results: {len(evals)}")
    print()
    print(verdict.render())
    print()
    print(f"VERDICT: {verdict.finalize_status()}")
    statuses = {row[2] for row in verdict.thresholds()}
    return 0 if statuses <= {PASS} else 1
