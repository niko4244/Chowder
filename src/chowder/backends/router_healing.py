"""The narrow router-healing training backend.

This is the production route for a router-training experiment. It implements the
existing `TrainingExecutor` contract -- `profile` / `run` / `cancel` -- so a
router run is an ordinary Chowder run: the parent owns the registry, the
reservation and the lifecycle, and this module owns nothing but the launch.

Three deliberate boundaries
---------------------------
* **`profile` computes nothing.** It reads the configured step profile (or the
  experiment's declared estimate) and returns a `CostEstimate`. It must never
  load a model, touch the base directory, or import torch: a preflight that
  loads weights is not a preflight.
* **CPU is the only qualified device.** `QUALIFIED_DEVICES` is enforced when the
  spec is constructed, so an accelerator request fails before a subprocess is
  launched rather than halfway through a load. The frozen-tensor digest is not
  device-safe yet; refusing is the honest outcome.
* **The parent never trusts the child's summary.** The worker's result is
  re-parsed: the ledger is rebuilt through `ledger_from_payload`, the required
  training phases are demanded, and the trainability / frozen / coverage
  evidence must each pass before an artifact is returned. A worker that crashed,
  was cancelled, or skipped a check produces an error here, not a green artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from ..base_identity import BaseIdentityError, resolve_base_identity
from ..executors import CostEstimate, ExecutionContext, TrainingArtifact
from ..lifecycle import REQUIRED_FOR_TRAINING, ledger_from_payload
from ..models import Experiment
from ..provenance import sha256_file
from ..resources import ResourceUsage
from ..worker_env import chowder_source_identity, worker_env

#: Devices this backend has qualified. See the module docstring.
QUALIFIED_DEVICES: tuple[str, ...] = ("cpu",)

#: The worker's result schema. Bumped when a field's meaning changes.
WORKER_RESULT_KIND = "router_healing_worker_result.v1"

#: Extra wall-clock allowance around the spec's own limit, for interpreter
#: startup, the base load, and payload publication. The spec's `max_seconds`
#: bounds *training*; this bounds the process.
_PROCESS_GRACE_SECONDS = 900.0

_ALLOWED_SCHEDULERS = {"constant", "cosine"}


class RouterHealingBackendError(RuntimeError):
    """A router-healing run cannot be launched, or its result cannot be trusted."""


@dataclass(frozen=True)
class RouterHealingRunSpec:
    """Everything the worker needs, frozen into one digestible document."""

    base_model_dir: str
    base_content_sha256: str
    corpus_path: str
    corpus_sha256: str
    output_dir: str
    max_steps: int
    learning_rate: float
    seq_len: int
    batch_size: int
    seed: int
    probe_window: int
    max_tokens: int
    checkpoint_dir: str | None = None
    resume_from: str | None = None
    max_seconds: float | None = None
    checkpoint_every: int = 0
    scheduler: str = "constant"
    warmup_steps: int = 0
    device: str = "cpu"
    detailed_timing: bool = False

    def __post_init__(self) -> None:
        for label, value in (
            ("base_model_dir", self.base_model_dir),
            ("corpus_path", self.corpus_path),
            ("output_dir", self.output_dir),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"router healing spec {label} must be a non-empty string")
        for label, digest in (
            ("base_content_sha256", self.base_content_sha256),
            ("corpus_sha256", self.corpus_sha256),
        ):
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"router healing spec {label} must be a sha256 hex digest")
        for label, value in (
            ("max_steps", self.max_steps),
            ("seq_len", self.seq_len),
            ("batch_size", self.batch_size),
            ("probe_window", self.probe_window),
            ("max_tokens", self.max_tokens),
            ("warmup_steps", self.warmup_steps),
            ("checkpoint_every", self.checkpoint_every),
            ("seed", self.seed),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"router healing spec {label} must be a non-negative integer")
        if self.max_steps <= 0:
            raise ValueError("router healing spec max_steps must be positive")
        if self.seq_len <= 0 or self.batch_size <= 0:
            raise ValueError("router healing spec seq_len and batch_size must be positive")
        if self.probe_window <= 0:
            raise ValueError("router healing spec probe_window must be positive")
        if self.max_tokens < self.seq_len * self.batch_size:
            raise ValueError(
                "router healing spec max_tokens is smaller than a single batch; the run "
                "could not take one step"
            )
        if not math.isfinite(float(self.learning_rate)) or self.learning_rate <= 0:
            raise ValueError("router healing spec learning_rate must be finite and positive")
        if self.max_seconds is not None and (
            not math.isfinite(float(self.max_seconds)) or self.max_seconds <= 0
        ):
            raise ValueError("router healing spec max_seconds must be finite and positive when set")
        if self.scheduler not in _ALLOWED_SCHEDULERS:
            raise ValueError(
                f"unsupported router healing scheduler {self.scheduler!r}; expected one of "
                f"{sorted(_ALLOWED_SCHEDULERS)}"
            )
        if self.warmup_steps > self.max_steps:
            raise ValueError("router healing spec warmup_steps cannot exceed max_steps")
        if self.device not in QUALIFIED_DEVICES:
            raise ValueError(
                f"device {self.device!r} is not qualified for router training; qualified "
                f"devices are {list(QUALIFIED_DEVICES)}. The frozen-tensor digest is not "
                "device-safe yet, so an accelerator run is refused rather than attempted."
            )

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def recipe_digest(self) -> str:
        """The mathematical recipe, separated from operational paths.

        Two runs of the same recipe from different directories share this digest,
        while the full spec digest (which includes paths) does not. That is what
        lets a reader say "the same experiment" without pretending a relocated
        checkout is a different one.
        """
        payload = {
            "max_steps": self.max_steps,
            "learning_rate": self.learning_rate,
            "seq_len": self.seq_len,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "probe_window": self.probe_window,
            "scheduler": self.scheduler,
            "warmup_steps": self.warmup_steps,
            "trains": "router-only:mlp.gate.weight",
            "experts_frozen": True,
            "loss": "causal-language-modelling-cross-entropy",
            "base_content_sha256": self.base_content_sha256,
            "corpus_sha256": self.corpus_sha256,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


class RouterHealingExecutor:
    """Launch one bounded router-training worker and account for its result."""

    name = "transformers-router-healing"

    def __init__(self) -> None:
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._cancelled: set[str] = set()

    # -- configuration -----------------------------------------------------

    @staticmethod
    def _backend_knobs(context: ExecutionContext) -> Mapping[str, Any]:
        config = context.resolved_config
        backend = config.get("backend", {}) if isinstance(config, Mapping) else {}
        backend = backend if isinstance(backend, Mapping) else {}
        knobs = backend.get("router_healing", {})
        return knobs if isinstance(knobs, Mapping) else {}

    @staticmethod
    def _research_spec(experiment: Experiment) -> Mapping[str, Any]:
        patch = experiment.config_patch if isinstance(experiment.config_patch, Mapping) else {}
        research = patch.get("router_healing", {})
        return research if isinstance(research, Mapping) else {}

    def _spec_for(
        self, experiment: Experiment, context: ExecutionContext, *, run_dir: Path
    ) -> RouterHealingRunSpec:
        research = self._research_spec(experiment)
        settings: dict[str, Any] = dict(self._backend_knobs(context))
        # The preregistered research fields win over project knobs: a knob that
        # could silently change the step count or seed would make the
        # preregistration decorative.
        for key in (
            "base_model_dir",
            "corpus_path",
            "max_steps",
            "learning_rate",
            "seq_len",
            "seed",
        ):
            if key in research:
                settings[key] = research[key]

        missing = [
            key for key in ("base_model_dir", "corpus_path", "max_steps", "learning_rate", "seq_len")
            if key not in settings
        ]
        if missing:
            raise RouterHealingBackendError(
                "router healing spec is incomplete: "
                f"missing {missing}. Supply them in config_patch.router_healing (the "
                "preregistered research spec) or backend.router_healing."
            )

        base_dir = Path(str(settings["base_model_dir"]))
        try:
            identity = resolve_base_identity(base_dir)
        except BaseIdentityError as exc:
            raise RouterHealingBackendError(
                f"the base model at {base_dir} has no honest content identity: {exc}"
            ) from exc
        declared_manifest = research.get("base_manifest_sha256")
        if declared_manifest is not None and declared_manifest != identity["manifest_sha256"]:
            raise RouterHealingBackendError(
                "the base directory does not match the manifest the experiment was frozen "
                f"against: declared {declared_manifest!r}, measured "
                f"{identity['manifest_sha256']!r}. Refusing to train a router against a "
                "base nobody proved it belongs to."
            )

        corpus_path = Path(str(settings["corpus_path"]))
        if not corpus_path.is_file():
            raise RouterHealingBackendError(f"training corpus not found: {corpus_path}")
        corpus_sha = sha256_file(corpus_path)
        declared_corpus = research.get("corpus_sha256")
        if declared_corpus is not None and declared_corpus != corpus_sha:
            raise RouterHealingBackendError(
                f"corpus hash mismatch: the experiment was frozen against "
                f"{declared_corpus!r} but {corpus_path} is {corpus_sha!r}"
            )

        seq_len = int(settings["seq_len"])
        batch_size = int(settings.get("batch_size", 1))
        # Refuse rather than truncate: a hard token limit that cannot fit one step
        # would produce a run that trained nothing and reported a limit.
        max_tokens = int(settings.get("max_tokens", int(settings["max_steps"]) * seq_len * batch_size))

        return RouterHealingRunSpec(
            base_model_dir=str(base_dir.resolve()),
            base_content_sha256=identity["content_sha256"],
            corpus_path=str(corpus_path.resolve()),
            corpus_sha256=corpus_sha,
            output_dir=str((run_dir / "output").resolve()),
            max_steps=int(settings["max_steps"]),
            learning_rate=float(settings["learning_rate"]),
            seq_len=seq_len,
            batch_size=batch_size,
            seed=int(settings["seed"]),
            probe_window=int(settings.get("probe_window", min(2, int(settings["max_steps"])))),
            max_tokens=max_tokens,
            checkpoint_dir=str((run_dir / "checkpoints").resolve()),
            resume_from=(
                str(Path(str(settings["resume_from"])).resolve())
                if settings.get("resume_from")
                else None
            ),
            max_seconds=(
                float(settings["max_seconds"]) if settings.get("max_seconds") is not None else None
            ),
            checkpoint_every=int(settings.get("checkpoint_every", 0)),
            scheduler=str(settings.get("scheduler", "constant")),
            warmup_steps=int(settings.get("warmup_steps", 0)),
            device=str(settings.get("device", "cpu")),
            detailed_timing=bool(settings.get("detailed_timing", False)),
        )

    # -- TrainingExecutor --------------------------------------------------

    def profile(self, experiment: Experiment, context: ExecutionContext) -> CostEstimate:
        """A non-computing estimate. Loads nothing, measures nothing, imports no torch."""
        profile = self._backend_knobs(context).get("profile", {})
        profile = profile if isinstance(profile, Mapping) else {}
        steps = profile.get("estimated_steps")
        seconds_per_step = profile.get("seconds_per_step")
        peak_vram = profile.get("peak_vram_gb")
        if steps is not None and seconds_per_step is not None:
            hours = max(0.0, float(steps) * float(seconds_per_step) / 3600.0)
            confidence = 0.75 if profile.get("source") == "measured" else 0.5
            notes = (
                "derived from backend.router_healing.profile step time; CPU-only, so the "
                "attributable accelerator hours are zero",
            )
        else:
            hours = max(0.0, float(experiment.estimated_gpu_hours))
            confidence = 0.25
            notes = (
                "using the experiment-declared GPU-hour estimate; no measured step profile",
            )
        return CostEstimate(
            gpu_hours=hours,
            peak_vram_gb=float(peak_vram) if peak_vram is not None else None,
            confidence=confidence,
            notes=notes,
        )

    @staticmethod
    def _worker_command(
        spec_path: Path, result_path: Path, chowder_identity: Path
    ) -> list[str]:
        return [
            sys.executable,
            "-m",
            "chowder.backends.router_healing_worker",
            "--spec",
            str(spec_path),
            "--result",
            str(result_path),
            "--chowder-identity",
            str(chowder_identity),
        ]

    @staticmethod
    def _tail(path: Path, lines: int = 40) -> str:
        if not path.exists():
            return ""
        return "\n".join(
            path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
        )

    @staticmethod
    def _record_failure(run_dir: Path, reason: str, *, wall_seconds: float | None) -> None:
        """Leave a durable terminal reason and cost beside the run.

        A failed attempt is evidence about the attempt. Writing it down is what
        keeps a crash from being indistinguishable from a run that never started.
        """
        payload = {
            "kind": "router_healing_run_failure.v1",
            "reason": reason,
            "wall_seconds": wall_seconds,
        }
        try:
            (run_dir / "run-failure.json").write_text(
                json.dumps(payload, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )
        except OSError:  # pragma: no cover - the run dir is ours and writable
            pass

    def run(self, experiment: Experiment, context: ExecutionContext) -> TrainingArtifact:
        run_id = f"{experiment.experiment_id}-{uuid4().hex[:12]}"
        run_dir = (Path(context.work_dir) / ".chowder" / "runs" / run_id).resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
        spec = self._spec_for(experiment, context, run_dir=run_dir)

        spec_path = run_dir / "run-spec.json"
        spec_path.write_text(
            json.dumps(spec.to_dict(), sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        identity = chowder_source_identity()
        identity_path = run_dir / "chowder-identity.json"
        identity_path.write_text(json.dumps(identity, sort_keys=True) + "\n", encoding="utf-8")

        result_path = run_dir / "worker-result.json"
        stdout_path = run_dir / "worker-stdout.log"
        stderr_path = run_dir / "worker-stderr.log"
        command = self._worker_command(spec_path, result_path, identity_path)

        started = time.perf_counter()
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                command,
                cwd=str(run_dir),
                env=worker_env({"PYTHONUNBUFFERED": "1"}),
                stdout=stdout,
                stderr=stderr,
            )
            self._processes[run_id] = process
            timeout = (spec.max_seconds or 0.0) + _PROCESS_GRACE_SECONDS
            try:
                while True:
                    if process.poll() is not None:
                        break
                    if run_id in self._cancelled:
                        process.terminate()
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:  # pragma: no cover - stubborn child
                            process.kill()
                            process.wait(timeout=30)
                        raise RouterHealingBackendError(
                            f"router healing run {run_id} was cancelled by the controller"
                        )
                    if time.perf_counter() - started > timeout:
                        process.kill()
                        process.wait(timeout=30)
                        raise RouterHealingBackendError(
                            f"router healing run {run_id} exceeded its process budget "
                            f"({timeout:.0f}s) and was killed"
                        )
                    time.sleep(0.05)
            finally:
                self._processes.pop(run_id, None)
                self._cancelled.discard(run_id)

        wall_seconds = time.perf_counter() - started
        if process.returncode != 0 or not result_path.is_file():
            reason = (
                f"router healing worker exited with code {process.returncode} and wrote no "
                f"result. stderr tail:\n{self._tail(stderr_path)}"
            )
            self._record_failure(run_dir, reason, wall_seconds=wall_seconds)
            raise RouterHealingBackendError(reason)

        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except ValueError as exc:
            reason = f"router healing worker wrote an unparseable result: {exc}"
            self._record_failure(run_dir, reason, wall_seconds=wall_seconds)
            raise RouterHealingBackendError(reason) from exc

        return self._artifact_from_result(
            result,
            experiment=experiment,
            spec=spec,
            run_id=run_id,
            run_dir=run_dir,
            identity=identity,
            wall_seconds=wall_seconds,
        )

    def _artifact_from_result(
        self,
        result: Mapping[str, Any],
        *,
        experiment: Experiment,
        spec: RouterHealingRunSpec,
        run_id: str,
        run_dir: Path,
        identity: Mapping[str, Any],
        wall_seconds: float,
    ) -> TrainingArtifact:
        """Validate the worker's result, then turn it into an artifact."""
        if result.get("kind") != WORKER_RESULT_KIND:
            raise RouterHealingBackendError(
                f"router healing worker result has kind {result.get('kind')!r}, expected "
                f"{WORKER_RESULT_KIND!r}"
            )
        if result.get("spec_digest") != spec.digest():
            raise RouterHealingBackendError(
                "the worker's spec digest does not match the spec this controller wrote; "
                "the result belongs to a different run"
            )

        try:
            ledger = ledger_from_payload(result.get("lifecycle") or {})
            ledger.require(list(REQUIRED_FOR_TRAINING), purpose="router training qualification")
        except Exception as exc:
            # One error surface for the caller: a worker whose cost breakdown
            # cannot be re-parsed and demanded is a rejected run, whatever the
            # internal exception type happened to be.
            raise RouterHealingBackendError(
                f"the worker's lifecycle ledger cannot qualify this run: {exc}"
            ) from exc

        trainability = result.get("trainability")
        if not isinstance(trainability, Mapping) or trainability.get("ok") is not True:
            raise RouterHealingBackendError(
                "the worker did not demonstrate trainability for the intended components: "
                f"{json.dumps(trainability, sort_keys=True)[:600]}"
            )
        frozen = result.get("frozen")
        if not isinstance(frozen, Mapping) or frozen.get("ok") is not True:
            raise RouterHealingBackendError(
                "the worker reported that frozen parameters changed during training: "
                f"{json.dumps(frozen, sort_keys=True)[:600]}"
            )
        coverage = result.get("coverage")
        if not isinstance(coverage, Mapping) or coverage.get("ok") is not True:
            raise RouterHealingBackendError(
                "the worker's intended-vs-present component coverage did not qualify: "
                f"{json.dumps(coverage, sort_keys=True)[:600]}"
            )
        payload = result.get("payload")
        if not isinstance(payload, Mapping) or not payload.get("payload_dir"):
            raise RouterHealingBackendError("the worker published no router payload")

        raw_usage = result.get("resource_usage")
        if not isinstance(raw_usage, Mapping):
            raise RouterHealingBackendError("the worker reported no resource usage")
        usage = ResourceUsage.from_wall_time(
            wall_seconds=float(raw_usage.get("wall_seconds", wall_seconds)),
            active_accelerator_count=int(raw_usage.get("active_accelerator_count", 0)),
            visible_accelerator_count=int(raw_usage.get("visible_accelerator_count", 0)),
            peak_vram_gb_by_accelerator=dict(raw_usage.get("peak_vram_gb_by_accelerator", {})),
        )

        return TrainingArtifact(
            run_id=run_id,
            experiment_id=experiment.experiment_id,
            artifact_ref=str(payload["payload_dir"]),
            gpu_hours=usage.gpu_hours,
            telemetry={
                "global_step": int(result.get("global_step", 0)),
                "steps_completed": int(result.get("steps_completed", 0)),
                "loss_first": result.get("loss_first"),
                "loss_last": result.get("loss_last"),
                "lifecycle": result.get("lifecycle"),
            },
            evidence={
                "backend": self.name,
                "spec": spec.to_dict(),
                "spec_digest": spec.digest(),
                "recipe_digest": spec.recipe_digest(),
                "source_identity": dict(identity),
                "base_identity": result.get("base_identity"),
                "payload": dict(payload),
                "trainability": dict(trainability),
                "frozen": dict(frozen),
                "scope": result.get("scope"),
                "coverage": dict(coverage),
                "freeze_summary": result.get("freeze_summary"),
                "checkpoints": result.get("checkpoints"),
                "resume": result.get("resume"),
                "limits": result.get("limits"),
                "tensor_inventory": result.get("tensor_inventory"),
                "quantization_reality": result.get("quantization_reality"),
                "utilization": result.get("utilization"),
                "tokenizer": result.get("tokenizer"),
                "model": result.get("model"),
                "phase_ledger": ledger.to_dict(),
                "run_dir": str(run_dir),
            },
            resource_usage=usage,
        )

    def cancel(self, run_id: str) -> None:
        """Ask the worker to stop; the run path turns that into a terminal reason."""
        self._cancelled.add(run_id)
        process = self._processes.get(run_id)
        if process is not None and process.poll() is None:
            process.terminate()
