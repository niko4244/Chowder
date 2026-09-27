"""Isolated teacher-free distillation workflow screen.

Reads real on-disk evidence only: catalog approval state, dataset manifests,
preflight results, recipe authorization blocks, replay summaries, evaluation
comparisons. Buttons invoke the pipeline's CLI entry points; there is no
decorative progress state — every displayed value traces to a file that the
pipeline actually wrote, and the refresh button re-reads the filesystem.

Stage completion is *checked*, not inferred from a file's existence: the
checking lives in :mod:`chowder.teacher_free_evidence` so it is testable
without a TUI and so a half-written recipe or an empty replay summary cannot
present itself as finished work.

Set ``TFD_PILOT_DIR`` to point the screen at a pilot root outside this
checkout (the pilot's real datasets and checkpoints live outside the repo);
the screen always displays which directory it is reading.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Button, Footer, Header, Static

from .hardware import detect_hardware
from .worker_env import worker_env
from .teacher_free_evidence import (  # noqa: F401
    GREEN_STATES, RED_STATES, WORKFLOW_STAGES, stage_state,
)

PREFLIGHT_TIMEOUT_SECONDS = 600


def pilot_dir(default: Path | None = None) -> Path:
    """The directory the screen reads: ``TFD_PILOT_DIR`` or the repo's copy."""
    override = os.environ.get("TFD_PILOT_DIR", "").strip()
    if override:
        return Path(override)
    if default is not None:
        return default
    return Path(__file__).resolve().parents[2] / "experiments" / "teacher_free_distill"


def run_preflight(exp_dir: Path, *, timeout: int = PREFLIGHT_TIMEOUT_SECONDS) -> None:
    """Run the CPU preflight through the worker contract.

    ``env=worker_env()`` pins the child's ``import chowder`` to the checkout
    this process imported: without it a fresh interpreter resolves the
    editable install, so a worktree screen would validate another Chowder's
    masking code and record the result as this one's evidence.
    """
    out = Path(exp_dir) / "preflight_result.json"
    try:
        subprocess.run(
            [sys.executable, str(Path(exp_dir) / "preflight.py"), "--out", str(out)],
            capture_output=True, text=True, timeout=timeout,
            env=worker_env({"PYTHONUNBUFFERED": "1"}),
        )
    except subprocess.TimeoutExpired:
        # Bounded, not decorative: a hung preflight must not freeze the screen
        # forever; the stale result file stays as-is and the refresh below
        # shows whatever evidence exists on disk.
        pass


class TeacherFreeDistillScreen(Static):
    """Stage overview with artifact-derived state; refresh re-reads the disk."""

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with VerticalScroll(id="tfd_body"):
            yield Static(id="tfd_stages", markup=True)
            yield Static(id="tfd_hardware", markup=True)
        with Horizontal(id="tfd_actions"):
            yield Button("Refresh", id="tfd_refresh")
            yield Button("Preflight (CPU)", id="tfd_preflight")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh()

    def _exp_dir(self) -> Path:
        return pilot_dir()

    def _refresh(self) -> None:
        exp_dir = self._exp_dir()
        lines = ["[b]Teacher-free distillation pilot — isolated workflow[/b]",
                 f"evidence directory: {exp_dir}", ""]
        for entry in stage_state(exp_dir):
            color = "green" if entry["state"] in GREEN_STATES else (
                "red" if entry["state"] in RED_STATES else "yellow")
            lines.append(f"[{color}]●[/{color}] {entry['stage']:<18} "
                         f"[{color}]{entry['state']}[/{color}] — {entry['detail']}")
        self.query_one("#tfd_stages").update("\n".join(lines))
        try:
            snapshot = detect_hardware()
            acc = ", ".join(f"{a.name} {a.memory_gb:.0f}GB" for a in snapshot.accelerators) or "none"
            self.query_one("#tfd_hardware").update(
                f"Hardware: {snapshot.platform}, {snapshot.cpu_count} CPUs, "
                f"{snapshot.ram_gb:.0f}GB RAM, {snapshot.storage_free_gb:.0f}GB free — accelerators: {acc}")
        except Exception as exc:  # noqa: BLE001 - display, never crash the TUI
            self.query_one("#tfd_hardware").update(f"Hardware: unavailable ({exc})")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        exp_dir = self._exp_dir()
        if event.button.id == "tfd_refresh":
            self._refresh()
        elif event.button.id == "tfd_preflight":
            run_preflight(exp_dir)
            self._refresh()
