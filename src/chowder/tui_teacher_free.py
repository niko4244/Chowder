"""Isolated teacher-free distillation workflow screen.

Reads real on-disk evidence only: catalog approval state, dataset manifests,
preflight results, recipe authorization blocks, replay summaries, evaluation
comparisons. Buttons invoke the pipeline's CLI entry points; there is no
decorative progress state — every displayed value traces to a file that the
pipeline actually wrote, and the refresh button re-reads the filesystem.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Button, Footer, Header, Static

from .hardware import detect_hardware

WORKFLOW_STAGES = [
    "source_review", "data_collection", "validation", "student_selection",
    "training_preflight", "training", "evaluation", "results",
]


def stage_state(exp_dir: Path) -> list[dict]:
    """Derive per-stage status from real artifacts; unknown stays unknown."""
    states = []

    def has(name: str) -> bool:
        return (exp_dir / name).exists()

    catalog = {}
    try:
        catalog = json.loads((exp_dir / "sources.json").read_text(encoding="utf8"))["sources"]
    except (OSError, json.JSONDecodeError, KeyError):
        pass
    approved = sorted(k for k, v in catalog.items() if v.get("approved") is True)
    blocked = sorted(k for k, v in catalog.items() if v.get("approved") is not True)

    states.append({"stage": "source_review", "state": "done" if catalog else "missing",
                   "detail": f"{len(approved)} approved, {len(blocked)} blocked: {', '.join(blocked) or 'none'}"})

    manifests = sorted(exp_dir.glob("**/manifest.json"))
    states.append({"stage": "data_collection", "state": "done" if manifests else "pending",
                   "detail": ", ".join(str(m.relative_to(exp_dir)) for m in manifests) or "no manifest.json yet"})

    counts = {}
    if manifests:
        try:
            counts = json.loads(manifests[0].read_text(encoding="utf8")).get("counts", {})
        except (OSError, json.JSONDecodeError):
            pass
    states.append({"stage": "validation", "state": "done" if counts else "pending",
                   "detail": ", ".join(f"{k}={v}" for k, v in sorted(counts.items())[:6]) or "run prepare.py"})

    student_ok = has("student_selection.json")
    states.append({"stage": "student_selection", "state": "done" if student_ok else "pending",
                   "detail": "student_selection.json" if student_ok else "run student.py --student qwen3-1.7b"})

    preflight = {}
    try:
        preflight = json.loads((exp_dir / "preflight_result.json").read_text(encoding="utf8"))
    except (OSError, json.JSONDecodeError):
        pass
    ok = preflight.get("ok") is True
    states.append({"stage": "training_preflight",
                   "state": "passed" if ok else ("failed" if preflight else "pending"),
                   "detail": f"{preflight.get('passed')}/{preflight.get('total')} checks" if preflight else "run preflight.py"})

    recipes = sorted((exp_dir / "recipes").glob("*.json")) if has("recipes") else []
    gpu_gate = "operator authorization + device exclusivity required"
    states.append({"stage": "training", "state": "recipes_ready" if recipes else "pending",
                   "detail": f"{len(recipes)} recipes; {gpu_gate}" if recipes else "recipes missing"})

    comparisons = sorted(exp_dir.glob("**/comparison_*.json")) + sorted(exp_dir.glob("**/replay_summary.json"))
    states.append({"stage": "evaluation", "state": "done" if comparisons else "pending",
                   "detail": ", ".join(p.name for p in comparisons) or "no comparison/replay evidence yet"})

    states.append({"stage": "results", "state": "pending",
                   "detail": "final REPORT.md status: see REPORT.md" if has("REPORT.md") else "not yet produced"})
    return states


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
        return Path(__file__).resolve().parents[2] / "experiments" / "teacher_free_distill"

    def _refresh(self) -> None:
        exp_dir = self._exp_dir()
        lines = ["[b]Teacher-free distillation pilot — isolated workflow[/b]", ""]
        for entry in stage_state(exp_dir):
            color = {"done": "green", "passed": "green", "recipes_ready": "green",
                     "failed": "red"}.get(entry["state"], "yellow")
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
            out = exp_dir / "preflight_result.json"
            try:
                subprocess.run([sys.executable, str(exp_dir / "preflight.py"),
                                "--out", str(out)], capture_output=True, text=True,
                               timeout=600)
            except subprocess.TimeoutExpired:
                # Bounded, not decorative: a hung preflight must not freeze the
                # screen forever; the stale result file stays as-is and the
                # refresh below shows whatever evidence exists on disk.
                pass
            self._refresh()
