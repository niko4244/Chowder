from __future__ import annotations

from pathlib import Path
from typing import Any

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, ScrollableContainer
from textual.widgets import Button, Footer, Header, Input, Label, RichLog, Static

from .hardware import HardwareSnapshot, detect_hardware
from .project import ProjectValidationError, write_project
from .project_runner import ProjectRunEvent, run_project


class ChowderTUI(App[None]):
    """Guided project setup and real training launcher."""

    TITLE = "Chowder"
    SUB_TITLE = "Autonomous post-training laboratory"

    CSS = """
    Screen { layout: vertical; }
    #body { padding: 1 2; }
    .section { margin-top: 1; text-style: bold; }
    Label { margin-top: 1; }
    Input { width: 100%; }
    #hardware { padding: 1; border: round $accent; margin-bottom: 1; }
    #actions { height: auto; margin: 1 0; }
    #actions Button { margin-right: 1; }
    #log { height: 14; border: round $primary; }
    #status { padding: 0 1; height: 1; }
    """

    def __init__(self, *, project_path: str | Path = "chowder-project.json") -> None:
        super().__init__()
        self.project_path = Path(project_path).expanduser().resolve()
        self._hardware: HardwareSnapshot | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with ScrollableContainer(id="body"):
            yield Static(
                "This wizard creates a reproducible Chowder project, validates it, "
                "then runs train → evaluate → gate.",
            )
            yield Static("Scanning hardware…", id="hardware")

            yield Static("Project", classes="section")
            yield Label("Project name")
            yield Input("My Chowder Model", id="project_name")
            yield Label("Project/work directory")
            yield Input(str(Path.cwd()), id="work_dir")
            yield Label("Project JSON path")
            yield Input(str(self.project_path), id="project_file")

            yield Static("Model + data", classes="section")
            yield Label("Base model (Hugging Face model ID or local path)")
            yield Input("trl-internal-testing/tiny-LlamaForCausalLM-3.2", id="base_model")
            yield Label("Training JSON/JSONL dataset")
            yield Input("train.jsonl", id="train_dataset")
            yield Label("Training text field")
            yield Input("text", id="text_field")
            yield Label("Evaluation holdout JSONL")
            yield Input("eval.jsonl", id="eval_dataset")
            yield Label("Evaluation prompt field")
            yield Input("prompt", id="prompt_field")
            yield Label("Evaluation expected-answer field")
            yield Input("expected", id="expected_field")

            yield Static("Goal", classes="section")
            yield Label("Metric / suite name")
            yield Input("quality", id="metric_name")
            yield Label("Target minimum score (0–1)")
            yield Input("0.8", id="target_score", type="number")
            yield Label("Total GPU-hour budget")
            yield Input("1.0", id="gpu_budget", type="number")
            yield Label("Initial experiment GPU-hour estimate")
            yield Input("0.25", id="estimated_gpu_hours", type="number")
            yield Label("Evaluation GPU-hour reserve")
            yield Input("0.05", id="eval_gpu_hours", type="number")

            yield Static("Training", classes="section")
            yield Label("Epochs")
            yield Input("1.0", id="epochs", type="number")
            yield Label("Learning rate")
            yield Input("0.0002", id="learning_rate", type="number")
            yield Label("Batch size per device")
            yield Input("1", id="batch_size", type="integer")
            yield Label("Gradient accumulation steps")
            yield Input("4", id="grad_accum", type="integer")
            yield Label("Maximum sequence length")
            yield Input("512", id="max_length", type="integer")
            yield Label("LoRA rank")
            yield Input("16", id="lora_r", type="integer")
            yield Label("LoRA alpha")
            yield Input("32", id="lora_alpha", type="integer")
            yield Label("LoRA target modules (comma separated)")
            yield Input("q_proj,k_proj,v_proj,o_proj", id="target_modules")
            yield Label("Precision: auto, bf16, fp16, or fp32")
            yield Input("auto", id="precision")
            yield Label("Quantization: auto (hardware-aware), none, or 4bit")
            yield Input("auto", id="quantization")
            yield Label("Gradient checkpointing: auto (hardware-aware), true, or false")
            yield Input("auto", id="gradient_checkpointing")
            yield Label("Active accelerators: auto (all detected GPUs), or a count")
            yield Input("auto", id="active_accelerator_count")
            yield Label("Evaluation max new tokens")
            yield Input("64", id="max_new_tokens", type="integer")

            with Horizontal(id="actions"):
                yield Button("Validate + Save", id="save", variant="primary")
                yield Button("Start Training", id="start", variant="success")
                yield Button("Quit", id="quit")
            yield Static("Ready", id="status")
            yield RichLog(id="log", wrap=True, highlight=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        self._scan_hardware()

    def _value(self, widget_id: str) -> str:
        return self.query_one(f"#{widget_id}", Input).value.strip()

    def _float(self, widget_id: str) -> float:
        value = self._value(widget_id)
        if not value:
            raise ProjectValidationError(f"{widget_id} is required")
        return float(value)

    def _int(self, widget_id: str) -> int:
        value = self._value(widget_id)
        if not value:
            raise ProjectValidationError(f"{widget_id} is required")
        return int(value)

    def _resolve_active_accelerator_count(self) -> int:
        """"auto" uses every detected GPU (accelerate launch + DDP now
        drives real multi-GPU training from this count); anything else is
        an explicit integer the user chose instead."""
        raw = self._value("active_accelerator_count")
        if raw.lower() == "auto":
            return len(self._hardware.accelerators) if self._hardware else 0
        try:
            count = int(raw)
        except ValueError as exc:
            raise ProjectValidationError(
                "active accelerator count must be 'auto' or an integer"
            ) from exc
        if count < 0:
            raise ProjectValidationError("active accelerator count cannot be negative")
        return count

    def _build_payload(self) -> dict[str, Any]:
        metric = self._value("metric_name")
        if not metric:
            raise ProjectValidationError("metric name is required")
        target_modules = tuple(
            item.strip()
            for item in self._value("target_modules").split(",")
            if item.strip()
        )
        if not target_modules:
            raise ProjectValidationError("at least one LoRA target module is required")

        active_accelerators = self._resolve_active_accelerator_count()
        quantization = self._value("quantization").lower()
        gradient_checkpointing = self._value("gradient_checkpointing").lower()
        if gradient_checkpointing not in {"auto", "true", "false"}:
            raise ProjectValidationError(
                "gradient checkpointing must be auto, true, or false"
            )

        training: dict[str, Any] = {
            "epochs": self._float("epochs"),
            "learning_rate": self._float("learning_rate"),
            "batch_size": self._int("batch_size"),
            "gradient_accumulation_steps": self._int("grad_accum"),
            "logging_steps": 1,
        }
        # Omitting these keys entirely (rather than setting an explicit
        # value) is what lets the backend's hardware-aware defaults resolve
        # them from actually-detected VRAM instead of one fixed choice.
        if gradient_checkpointing != "auto":
            training["gradient_checkpointing"] = gradient_checkpointing == "true"

        backend: dict[str, Any] = {
            "schema_version": 1,
            "type": "transformers-peft",
            "base_model": self._value("base_model"),
            "dataset": self._value("train_dataset"),
            "text_field": self._value("text_field"),
            "max_length": self._int("max_length"),
            "precision": self._value("precision").lower(),
            "trust_remote_code": False,
            "training": training,
            "lora": {
                "r": self._int("lora_r"),
                "alpha": self._int("lora_alpha"),
                "dropout": 0.05,
                "target_modules": list(target_modules),
                "use_rslora": False,
            },
            "runtime": {
                "active_accelerator_count": active_accelerators,
            },
        }
        if quantization != "auto":
            backend["quantization"] = quantization

        return {
            "schema_version": 1,
            "name": self._value("project_name"),
            "work_dir": self._value("work_dir"),
            "registry_path": ".chowder/runs.db",
            "seed": 1,
            "goal": {
                "metrics": [
                    {
                        "name": metric,
                        "minimum": self._float("target_score"),
                        "weight": 1.0,
                        "regression_tolerance": 0.0,
                        "direction": "maximize",
                    }
                ],
                "gpu_hour_budget": self._float("gpu_budget"),
                "max_parallel_candidates": 1,
                "minimum_promotion_gain": 0.0,
                # A generated project has no prior, independently-verified
                # measurement to compare against -- require the baseline and
                # candidate evaluations to have run under a matching protocol
                # rather than silently comparing scores that may not mean
                # the same thing.
                "require_protocol_match": True,
            },
            # The user provides the model, holdout, and target -- not a guess
            # at the model's present score. The untouched base model is
            # evaluated automatically, under this same project's evaluation
            # protocol, before training starts.
            "baseline": {"mode": "auto"},
            "experiment": {
                "experiment_id": "initial-sft",
                "estimated_gpu_hours": self._float("estimated_gpu_hours"),
                "hypothesis": {
                    "observation": "Base model has not been adapted to this project dataset",
                    "suspected_cause": "Target task/domain behavior is underrepresented",
                    "intervention": "Supervised LoRA fine-tuning",
                    "expected_deltas": {metric: 0.01},
                },
                "config_patch": {},
                "tags": ["tui", "initial-sft"],
            },
            "config": {
                "seed": 1,
                "backend": backend,
                "evaluation": {
                    "type": "transformers-text",
                    "estimated_gpu_hours": self._float("eval_gpu_hours"),
                    "precision": "inherit",
                    "quantization": "inherit",
                    "device": "auto",
                    "trust_remote_code": False,
                    "suites": [
                        {
                            "name": metric,
                            "dataset": self._value("eval_dataset"),
                            "prompt_field": self._value("prompt_field"),
                            "expected_field": self._value("expected_field"),
                            "scoring": "normalized_exact_match",
                            "max_new_tokens": self._int("max_new_tokens"),
                            "use_chat_template": False,
                        }
                    ],
                },
            },
        }

    def _project_target(self) -> Path:
        raw = self._value("project_file")
        if not raw:
            raise ProjectValidationError("project JSON path is required")
        return Path(raw).expanduser().resolve()

    def _save_project(self) -> Path:
        target = self._project_target()
        return write_project(target, self._build_payload())

    def _append_log(self, text: str) -> None:
        self.query_one("#log", RichLog).write(text)

    def _set_status(self, text: str) -> None:
        self.query_one("#status", Static).update(text)

    def _set_running(self, running: bool) -> None:
        self.query_one("#start", Button).disabled = running
        self.query_one("#save", Button).disabled = running

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "quit":
            self.exit()
            return
        if event.button.id == "save":
            try:
                target = self._save_project()
            except Exception as exc:
                self._set_status(f"Validation failed: {exc}")
                self._append_log(f"[red]Validation failed:[/] {type(exc).__name__}: {exc}")
                return
            self._set_status(f"Saved {target}")
            self._append_log(f"[green]Project saved:[/] {target}")
            return
        if event.button.id == "start":
            try:
                target = self._save_project()
            except Exception as exc:
                self._set_status(f"Validation failed: {exc}")
                self._append_log(f"[red]Validation failed:[/] {type(exc).__name__}: {exc}")
                return
            self._set_running(True)
            self._set_status("Training started")
            self._append_log(f"[bold]Starting project:[/] {target}")
            self._run_training(target)

    @work(thread=True, exclusive=True)
    def _scan_hardware(self) -> None:
        try:
            snapshot = detect_hardware(Path.cwd())
        except Exception as exc:
            self.call_from_thread(
                self.query_one("#hardware", Static).update,
                f"Hardware scan failed: {type(exc).__name__}: {exc}",
            )
            return
        self._hardware = snapshot
        if snapshot.accelerators:
            gpu_lines = " | ".join(
                f"GPU {index}: {gpu.name} {gpu.memory_gb:.1f} GB"
                for index, gpu in enumerate(snapshot.accelerators)
            )
            note = (
                f"{gpu_lines}\nRAM: {snapshot.ram_gb:.1f} GB | "
                f"Free storage: {snapshot.storage_free_gb:.1f} GB\n"
                f"'auto' active-accelerator count launches all {len(snapshot.accelerators)} "
                "detected GPU(s) via accelerate + DDP; each remains a separate VRAM pool "
                "for planning purposes."
            )
        else:
            note = (
                f"No NVIDIA GPU detected | RAM: {snapshot.ram_gb:.1f} GB | "
                f"Free storage: {snapshot.storage_free_gb:.1f} GB\n"
                "CPU training is allowed for small/test models; 4-bit QLoRA requires CUDA."
            )
        self.call_from_thread(self.query_one("#hardware", Static).update, note)

    @work(thread=True, exclusive=True)
    def _run_training(self, project_path: Path) -> None:
        def event_sink(event: ProjectRunEvent) -> None:
            self.call_from_thread(
                self._append_log,
                f"[bold]{event.stage}[/]: {event.message}",
            )

        try:
            outcome = run_project(project_path, on_event=event_sink)
            candidate = outcome.generation.candidates[0]
            if candidate.error is not None:
                message = f"Training failed: {candidate.error}"
                self.call_from_thread(self._set_status, message)
                self.call_from_thread(self._append_log, f"[red]{message}[/]")
            else:
                promoted = outcome.promoted_experiment_id
                message = (
                    f"Complete — promoted {promoted}"
                    if promoted
                    else "Complete — candidate was not promoted"
                )
                self.call_from_thread(self._set_status, message)
                self.call_from_thread(self._append_log, f"[green]{message}[/]")
        except Exception as exc:
            message = f"Run failed: {type(exc).__name__}: {exc}"
            self.call_from_thread(self._set_status, message)
            self.call_from_thread(self._append_log, f"[red]{message}[/]")
        finally:
            self.call_from_thread(self._set_running, False)


def run_tui(project_path: str | Path = "chowder-project.json") -> None:
    ChowderTUI(project_path=project_path).run()
