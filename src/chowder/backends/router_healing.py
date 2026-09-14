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
from ..executors import (
    CostEstimate,
    EvaluationOutcome,
    ExecutionContext,
    TrainingArtifact,
)
from ..lifecycle import (
    PHASE_BASELINE_GENERATION,
    PHASE_MODEL_LOAD,
    REQUIRED_FOR_EVALUATION,
    REQUIRED_FOR_TRAINING,
    ledger_from_payload,
)
from ..models import Experiment
from ..provenance import sha256_file
from ..resources import ResourceUsage
from ..worker_env import chowder_source_identity, worker_env
from .router_healing_load import LOAD_POLICIES

#: Devices this backend has qualified. See the module docstring.
#: Devices a router run may request. `cuda` was qualified by the P11 rung-2
#: small-CUDA preregistered qualification (docs/quals/P11_CUDA_PREREG_2026-09-13.md):
#: the frozen-tensor digest reads bounded chunks on-device, and the workers run a
#: measured device preflight (free memory, step-cost probe, projections) that
#: refuses before optimizer step 1 when the run cannot fit. `mps` stays unqualified:
#: never measured, never claimed.
QUALIFIED_DEVICES: tuple[str, ...] = ("cpu", "cuda")

#: The worker's result schema. Bumped when a field's meaning changes.
WORKER_RESULT_KIND = "router_healing_worker_result.v1"

#: Tolerance for the uniform-shift identity control: a routing-invariant
#: payload can move logits only through float rounding, so "unchanged" means
#: routing decisions identical and routing weights within rounding noise —
#: not bit-identical logits, which float32 never promised (that demand made
#: the control flaky across CPUs). Must match the worker's
#: `_ROUTING_ROUNDING_TOLERANCE`.
ROUTING_ROUNDING_TOLERANCE = 1e-6

#: The evaluation worker's result schema. A separate document from the training
#: result on purpose: an evaluation that could be mistaken for a training report
#: would let a scored artifact and a produced artifact blur together.
EVAL_WORKER_RESULT_KIND = "router_healing_eval_result.v1"

#: Extra wall-clock allowance around the spec's own limit, for interpreter
#: startup, the base load, and payload publication. The spec's `max_seconds`
#: bounds *training*; this bounds the process.
_PROCESS_GRACE_SECONDS = 900.0

_ALLOWED_SCHEDULERS = {"constant", "cosine"}


class RouterHealingBackendError(RuntimeError):
    """A router-healing run cannot be launched, or its result cannot be trusted."""


def _resolve_declared_path(value: Any, work_dir: Path) -> Path:
    """Resolve a declared path the way the project loader does.

    A relative knob is relative to the project's work directory, never to
    whatever directory a spawned worker happens to start in -- otherwise the
    same project file would train against different bytes depending on where it
    was launched from, and the recorded content identity would be the only hint
    that it happened.
    """
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = work_dir / path
    return path.resolve()


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
    load_policy: str = "fp32-resident"

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
                f"devices are {list(QUALIFIED_DEVICES)}. An unqualified device is refused "
                "at spec time, not attempted and discovered."
            )
        if self.load_policy not in LOAD_POLICIES:
            raise ValueError(
                f"unknown load policy {self.load_policy!r}; qualified policies are "
                f"{list(LOAD_POLICIES)}. A load nobody preregistered is refused at spec "
                "time, not discovered at load time."
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
            # P11 rung-3 amendment: how the base was resident is part of the
            # recipe, because it changes the measured cost and the memory
            # contract a payload's consumer is entitled to rely on.
            "load_policy": self.load_policy,
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
            "device",
            "load_policy",
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

        work_dir = Path(context.work_dir)
        base_dir = _resolve_declared_path(settings["base_model_dir"], work_dir)
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

        corpus_path = _resolve_declared_path(settings["corpus_path"], work_dir)
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
                str(_resolve_declared_path(settings["resume_from"], work_dir))
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
            load_policy=str(settings.get("load_policy", "fp32-resident")),
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
        if spec.device != "cpu":
            # A cuda run is only trustworthy if its preflight actually measured
            # the device. A missing record, or a free-memory field that is not a
            # real byte count, is the unmeasured-zero lie in another costume.
            preflight = result.get("device_preflight")
            if not isinstance(preflight, Mapping) or not preflight.get("step_cost_probe"):
                raise RouterHealingBackendError(
                    "the accelerator worker reported no device preflight: a cuda run "
                    "without measured device evidence is refused, not believed"
                )
            free_memory = preflight.get("free_memory_bytes")
            if not isinstance(free_memory, int) or free_memory <= 0:
                raise RouterHealingBackendError(
                    "the accelerator worker's device preflight did not measure free memory"
                )
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
                "device_preflight": result.get("device_preflight"),
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


class RouterHealingEvaluationError(RuntimeError):
    """A router payload cannot be independently evaluated honestly."""


@dataclass(frozen=True)
class RouterHealingEvalSpec:
    """Everything the evaluation worker needs to score one published payload.

    The holdout corpus is a first-class field rather than a reuse of the
    training corpus: an evaluation that scores the data the router was trained
    on measures fit, not capability, and this seam is where that would go
    wrong silently.

    ``payload_dir`` is ``None`` for exactly one thing: the **base arm**. A
    baseline is the untouched base measured under the same holdout protocol,
    so it is the same worker with nothing applied -- not a second code path
    that could drift from the scored one. A base spec that named payload
    parameters is refused, because those two claims contradict each other.
    """

    base_model_dir: str
    base_content_sha256: str
    payload_dir: str | None
    holdout_corpus_path: str
    holdout_corpus_sha256: str
    expected_parameter_paths: tuple[str, ...]
    output_dir: str
    seq_len: int
    batches: int
    device: str = "cpu"
    detailed_timing: bool = False
    load_policy: str = "fp32-resident"

    def __post_init__(self) -> None:
        for label, value in (
            ("base_model_dir", self.base_model_dir),
            ("holdout_corpus_path", self.holdout_corpus_path),
            ("output_dir", self.output_dir),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"router healing eval spec {label} must be a non-empty string")
        if self.payload_dir is not None and (
            not isinstance(self.payload_dir, str) or not self.payload_dir.strip()
        ):
            raise ValueError(
                "router healing eval spec payload_dir must be a non-empty string or None "
                "(None scores the untouched base arm)"
            )
        for label, digest in (
            ("base_content_sha256", self.base_content_sha256),
            ("holdout_corpus_sha256", self.holdout_corpus_sha256),
        ):
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"router healing eval spec {label} must be a sha256 digest")
        for label, value in (("seq_len", self.seq_len), ("batches", self.batches)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"router healing eval spec {label} must be a positive integer")
        names = tuple(str(name) for name in self.expected_parameter_paths)
        if len(set(names)) != len(names):
            raise ValueError("router healing eval spec expected parameter paths must be unique")
        if self.payload_dir is None:
            if names:
                raise ValueError(
                    "a base-arm spec (payload_dir=None) must not declare expected parameter "
                    "paths: nothing is applied, so naming parameters it must touch is a "
                    "contradiction"
                )
        elif not names:
            raise ValueError(
                "router healing eval spec needs at least one expected parameter path; an "
                "empty allowlist cannot prove the payload touched what it claims"
            )
        object.__setattr__(self, "expected_parameter_paths", names)
        if self.device not in QUALIFIED_DEVICES:
            raise ValueError(
                f"device {self.device!r} is not qualified for router evaluation; qualified "
                f"devices are {list(QUALIFIED_DEVICES)}. An unqualified device is refused "
                "at spec time, not attempted and discovered."
            )
        if self.load_policy not in LOAD_POLICIES:
            raise ValueError(
                f"unknown load policy {self.load_policy!r}; qualified policies are "
                f"{list(LOAD_POLICIES)}. The base arm and the candidate arm must load "
                "under the same declared contract."
            )

    @property
    def payload_applied(self) -> bool:
        """Whether this spec scores a payload or the untouched base."""
        return self.payload_dir is not None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


class RouterHealingEvaluator:
    """Score a published router payload in a process of its own.

    Independent means independent: the base is loaded fresh, the payload is
    re-verified from disk against the base it was trained on, and the two
    application controls the artifact claims are *measured* here rather than
    taken from the training run's word -- an artifact whose own report disagrees
    with what applying it actually did is refused.
    """

    name = "transformers-router-healing-evaluator"

    def __init__(self) -> None:
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._cancelled: set[str] = set()
        self._cancellation: Any = None
        self._progress_callback: Any = None

    def bind_cancellation(self, token: Any) -> None:
        self._cancellation = token

    def bind_progress_callback(self, callback: Any, **_kwargs: Any) -> None:
        self._progress_callback = callback

    def _knobs(self, context: ExecutionContext) -> Mapping[str, Any]:
        backend = context.resolved_config.get("backend", {})
        backend = backend if isinstance(backend, Mapping) else {}
        knobs = backend.get("router_healing", {})
        return knobs if isinstance(knobs, Mapping) else {}

    def _protocol(
        self,
        context: ExecutionContext,
        *,
        experimental_research: Mapping[str, Any] | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], Mapping[str, Any], Path]:
        """Resolve the base + holdout protocol, whatever supplied the settings.

        The candidate arm reads the protocol from the experiment that is running
        (so a worker's resolved config cannot silently drift from what was
        preregistered); the base arm has no experiment yet, so it reads the
        project config the baseline is being measured under. Both arms end up
        with the same fields, which is what makes them comparable at all.
        """
        research: Mapping[str, Any] = experimental_research or {}
        if config is not None:
            backend = config.get("backend", {}) if isinstance(config, Mapping) else {}
            backend = backend if isinstance(backend, Mapping) else {}
            raw_knobs = backend.get("router_healing", {})
            knobs: Mapping[str, Any] = raw_knobs if isinstance(raw_knobs, Mapping) else {}
        else:
            knobs = self._knobs(context)
        settings: dict[str, Any] = dict(knobs)
        settings["base_model_dir"] = research.get("base_model_dir", knobs.get("base_model_dir"))
        settings["holdout_corpus_path"] = research.get(
            "holdout_corpus_path", knobs.get("holdout_corpus_path")
        )
        if not settings.get("base_model_dir"):
            raise RouterHealingEvaluationError(
                "no base_model_dir is declared for router evaluation"
            )
        if not settings.get("holdout_corpus_path"):
            raise RouterHealingEvaluationError(
                "no holdout_corpus_path is declared; evaluating on the training corpus "
                "would measure fit rather than capability, so this is refused"
            )

        work_dir = Path(context.work_dir)
        base_dir = _resolve_declared_path(settings["base_model_dir"], work_dir)
        try:
            identity = resolve_base_identity(base_dir)
        except BaseIdentityError as exc:
            raise RouterHealingEvaluationError(
                f"the base model at {base_dir} has no honest content identity: {exc}"
            ) from exc

        declared_manifest = research.get("base_manifest_sha256")
        if declared_manifest is not None and declared_manifest != identity["manifest_sha256"]:
            raise RouterHealingEvaluationError(
                "the base directory does not match the manifest the experiment was frozen "
                f"against: declared {declared_manifest!r}, measured "
                f"{identity['manifest_sha256']!r}"
            )

        holdout = _resolve_declared_path(settings["holdout_corpus_path"], work_dir)
        if not holdout.is_file():
            raise RouterHealingEvaluationError(f"holdout corpus not found: {holdout}")
        holdout_sha = sha256_file(holdout)
        declared_holdout = research.get("holdout_corpus_sha256")
        if declared_holdout is not None and declared_holdout != holdout_sha:
            raise RouterHealingEvaluationError(
                f"holdout corpus hash mismatch: the experiment was frozen against "
                f"{declared_holdout!r} but {holdout} is {holdout_sha!r}"
            )
        settings["_base_dir"] = base_dir
        settings["_identity"] = identity
        settings["_holdout"] = holdout
        settings["_holdout_sha"] = holdout_sha
        settings["_work_dir"] = work_dir
        return settings, research, work_dir

    def _base_spec_for(
        self,
        context: ExecutionContext,
        *,
        config: Mapping[str, Any],
        eval_dir: Path,
    ) -> RouterHealingEvalSpec:
        """The base arm: the untouched base scored on the same holdout.

        Deliberately the *same* worker and the same protocol as the candidate
        arm, with no payload. A baseline that drifted from the protocol it will
        be compared against would make the comparison meaningless.
        """
        settings, _research, _work_dir = self._protocol(context, config=config)
        return RouterHealingEvalSpec(
            base_model_dir=str(settings["_base_dir"]),
            base_content_sha256=settings["_identity"]["content_sha256"],
            payload_dir=None,
            holdout_corpus_path=str(settings["_holdout"]),
            holdout_corpus_sha256=settings["_holdout_sha"],
            expected_parameter_paths=(),
            output_dir=str((eval_dir / "output").resolve()),
            seq_len=int(settings.get("seq_len", 128)),
            batches=int(settings.get("eval_batches", 4)),
            device=str(settings.get("device", "cpu")),
            detailed_timing=bool(settings.get("eval_detailed_timing", False)),
        )

    def _spec_for(
        self,
        experiment: Experiment,
        artifact: TrainingArtifact,
        context: ExecutionContext,
        *,
        eval_dir: Path,
    ) -> RouterHealingEvalSpec:
        research = RouterHealingExecutor._research_spec(experiment)
        settings, _research, _work_dir = self._protocol(
            context, experimental_research=research
        )

        payload_dir = Path(str(artifact.artifact_ref))
        if not payload_dir.is_dir():
            raise RouterHealingEvaluationError(
                f"the artifact reference {payload_dir} is not a published payload directory"
            )

        # The intended set comes from the training artifact, never from a fresh
        # guess: it is the scope the payload was published under, so re-deriving
        # it here could silently accept a payload that touches more.
        declared = research.get("expected_parameter_paths")
        expected_names = (
            tuple(str(name) for name in declared) if isinstance(declared, (list, tuple)) else ()
        )
        if not expected_names:
            recorded = artifact.evidence.get("freeze_summary", {})
            if isinstance(recorded, Mapping):
                expected_names = tuple(
                    str(name) for name in recorded.get("trainable_param_names", ())
                )
        if not expected_names:
            raise RouterHealingEvaluationError(
                "the training artifact records no intended parameter set, so a payload "
                "cannot be checked for scope; refusing to evaluate it blindly"
            )

        return RouterHealingEvalSpec(
            base_model_dir=str(settings["_base_dir"]),
            base_content_sha256=settings["_identity"]["content_sha256"],
            payload_dir=str(payload_dir.resolve()),
            holdout_corpus_path=str(settings["_holdout"]),
            holdout_corpus_sha256=settings["_holdout_sha"],
            expected_parameter_paths=expected_names,
            output_dir=str((eval_dir / "output").resolve()),
            seq_len=int(settings.get("seq_len", research.get("seq_len", 128))),
            batches=int(settings.get("eval_batches", 4)),
            device=str(settings.get("device", "cpu")),
            detailed_timing=bool(settings.get("eval_detailed_timing", False)),
            load_policy=str(
                research.get("load_policy", settings.get("load_policy", "fp32-resident"))
            ),
        )

    def profile(self, experiment: Experiment, context: ExecutionContext) -> CostEstimate:
        profile = self._knobs(context).get("profile", {})
        profile = profile if isinstance(profile, Mapping) else {}
        steps = profile.get("eval_seconds")
        if steps is not None:
            return CostEstimate(
                gpu_hours=0.0,
                confidence=0.5,
                notes=(
                    "router evaluation is CPU-only, so its attributable accelerator hours "
                    "are zero; the declared wall time is not a GPU-hour estimate",
                ),
            )
        return CostEstimate(
            gpu_hours=0.0,
            confidence=0.25,
            notes=(
                "router evaluation is CPU-only and unprofiled: zero accelerator hours, "
                "with the wall time unmeasured until the run reports it",
            ),
        )

    def _spawn(
        self,
        spec: RouterHealingEvalSpec,
        run_id: str,
        eval_dir: Path,
    ) -> tuple[Mapping[str, Any], float, Mapping[str, Any]]:
        """Launch one evaluation worker and return its parsed result.

        Both arms share this so a base score and a payload score cannot drift in
        how they were launched, timed, or checked for source identity -- the
        whole point of an arm is that it differs only in what was applied.
        """
        spec_path = eval_dir / "eval-spec.json"
        spec_path.write_text(
            json.dumps(spec.to_dict(), sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        identity = chowder_source_identity()
        identity_path = eval_dir / "chowder-identity.json"
        identity_path.write_text(json.dumps(identity, sort_keys=True) + "\n", encoding="utf-8")

        result_path = eval_dir / "worker-result.json"
        stdout_path = eval_dir / "worker-stdout.log"
        stderr_path = eval_dir / "worker-stderr.log"
        command = [
            sys.executable,
            "-m",
            "chowder.backends.router_healing_eval_worker",
            "--spec",
            str(spec_path),
            "--result",
            str(result_path),
            "--chowder-identity",
            str(identity_path),
        ]

        started = time.perf_counter()
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                command,
                cwd=str(eval_dir),
                env=worker_env({"PYTHONUNBUFFERED": "1"}),
                stdout=stdout,
                stderr=stderr,
            )
            self._processes[run_id] = process
            timeout = (spec.batches * 600.0) + _PROCESS_GRACE_SECONDS
            try:
                while True:
                    if process.poll() is not None:
                        break
                    if run_id in self._cancelled or (
                        self._cancellation is not None
                        and getattr(self._cancellation, "requested", False)
                    ):
                        process.terminate()
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:  # pragma: no cover
                            process.kill()
                            process.wait(timeout=30)
                        raise RouterHealingEvaluationError(
                            f"router evaluation {run_id} was cancelled by the controller"
                        )
                    if time.perf_counter() - started > timeout:
                        process.kill()
                        process.wait(timeout=30)
                        raise RouterHealingEvaluationError(
                            f"router evaluation {run_id} exceeded its process budget"
                        )
                    time.sleep(0.05)
            finally:
                self._processes.pop(run_id, None)
                self._cancelled.discard(run_id)

        wall_seconds = time.perf_counter() - started
        if process.returncode != 0 or not result_path.is_file():
            raise RouterHealingEvaluationError(
                "router evaluation worker exited with code "
                f"{process.returncode} and wrote no result. stderr tail:\n"
                f"{RouterHealingExecutor._tail(stderr_path)}"
            )
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise RouterHealingEvaluationError(
                f"router evaluation worker wrote an unparseable result: {exc}"
            ) from exc
        return result, wall_seconds, identity

    def evaluate(
        self,
        *,
        experiment: Experiment,
        artifact: TrainingArtifact,
        context: ExecutionContext,
    ) -> EvaluationOutcome:
        if experiment.experiment_id != artifact.experiment_id:
            raise RouterHealingEvaluationError("artifact experiment_id does not match experiment")
        run_id = f"{experiment.experiment_id}-eval-{uuid4().hex[:12]}"
        eval_dir = (Path(context.work_dir) / ".chowder" / "evals" / run_id).resolve()
        eval_dir.mkdir(parents=True, exist_ok=False)
        spec = self._spec_for(experiment, artifact, context, eval_dir=eval_dir)
        result, wall_seconds, identity = self._spawn(spec, run_id, eval_dir)
        return self._outcome_from_result(
            result,
            experiment_id=experiment.experiment_id,
            artifact_ref=str(artifact.artifact_ref),
            spec=spec,
            run_id=run_id,
            eval_dir=eval_dir,
            identity=identity,
            wall_seconds=wall_seconds,
        )

    def evaluate_base(
        self,
        *,
        config: Mapping[str, Any],
        context: ExecutionContext,
        experiment_id: str = "baseline",
    ) -> EvaluationOutcome:
        """Score the untouched base on the holdout protocol.

        This is what an automatic baseline means for a router project: the same
        worker, the same holdout corpus and the same settings as the candidate
        arm, with nothing applied. It is deliberately not the PEFT text
        evaluator -- a baseline measured by a different scorer than the one that
        will score the candidate is not a baseline, it is a second opinion.
        """
        run_id = f"{experiment_id}-eval-{uuid4().hex[:12]}"
        eval_dir = (Path(context.work_dir) / ".chowder" / "evals" / run_id).resolve()
        eval_dir.mkdir(parents=True, exist_ok=False)
        spec = self._base_spec_for(context, config=config, eval_dir=eval_dir)
        result, wall_seconds, identity = self._spawn(spec, run_id, eval_dir)
        return self._outcome_from_result(
            result,
            experiment_id=experiment_id,
            artifact_ref=None,
            spec=spec,
            run_id=run_id,
            eval_dir=eval_dir,
            identity=identity,
            wall_seconds=wall_seconds,
            payload_arm=False,
        )

    def _outcome_from_result(
        self,
        result: Mapping[str, Any],
        *,
        experiment_id: str,
        artifact_ref: str | None,
        spec: RouterHealingEvalSpec,
        run_id: str,
        eval_dir: Path,
        identity: Mapping[str, Any],
        wall_seconds: float,
        payload_arm: bool = True,
    ) -> EvaluationOutcome:
        """Turn a worker result into an outcome, refusing an arm that lies.

        The base arm cannot require the phases a payload comparison requires, so
        the required set is arm-dependent. What is *not* arm-dependent is the
        ledger refusal itself: an unmeasured phase refuses in both arms rather
        than being quietly absent from one of them.
        """
        if result.get("kind") != EVAL_WORKER_RESULT_KIND:
            raise RouterHealingEvaluationError(
                f"router evaluation result has kind {result.get('kind')!r}, expected "
                f"{EVAL_WORKER_RESULT_KIND!r}"
            )
        if result.get("spec_digest") != spec.digest():
            raise RouterHealingEvaluationError(
                "the evaluation worker's spec digest does not match the spec this controller "
                "wrote; the result belongs to a different run"
            )

        required = (
            list(REQUIRED_FOR_EVALUATION)
            if payload_arm
            else [PHASE_MODEL_LOAD, PHASE_BASELINE_GENERATION]
        )
        try:
            ledger = ledger_from_payload(result.get("lifecycle") or {})
            ledger.require(required, purpose="router evaluation")
        except Exception as exc:
            raise RouterHealingEvaluationError(
                f"the evaluation worker's lifecycle ledger cannot qualify this run: {exc}"
            ) from exc
        if spec.device != "cpu":
            usage = result.get("resource_usage")
            peak = (
                usage.get("peak_vram_gb_by_accelerator", {})
                if isinstance(usage, Mapping)
                else {}
            )
            if not isinstance(peak, Mapping) or not peak:
                raise RouterHealingEvaluationError(
                    "the accelerator evaluation worker did not measure peak VRAM: an "
                    "empty map on a cuda arm is an unmeasured claim, refused"
                )

        control = result.get("application_control")
        if not isinstance(control, Mapping):
            raise RouterHealingEvaluationError(
                "the evaluation worker reported no application control; without it there is "
                "no evidence what was applied"
            )
        changed_parameters = control.get("parameters_changed")
        changed_output = control.get("outputs_changed")
        if not isinstance(changed_parameters, bool) or not isinstance(changed_output, bool):
            raise RouterHealingEvaluationError(
                "the application control must report booleans for both the parameter change "
                f"and the output change, got {dict(control)!r}"
            )
        if not payload_arm:
            # The base arm's whole claim is that nothing was applied. If any of
            # that moved, this is not a baseline and its number cannot be used
            # as one.
            if control.get("payload_kind") != "none" or changed_parameters or changed_output:
                raise RouterHealingEvaluationError(
                    "the base arm reported an applied change, so it is not a baseline: "
                    f"{dict(control)!r}"
                )
            if spec.payload_dir is not None:
                raise RouterHealingEvaluationError(
                    "the base arm ran with a payload directory in its spec; this is not a "
                    "baseline measurement"
                )
        else:
            if "routing_top1_equal" not in control or (
                "max_abs_routing_weight_delta" not in control
            ):
                raise RouterHealingEvaluationError(
                    "the payload arm reported no routing fingerprint; without it there is "
                    "no evidence the routing path consumed the payload, and a score from "
                    "an unverified routing path would be fiction"
                )
            routing_top1_equal = bool(control["routing_top1_equal"])
            try:
                max_routing_delta = float(control["max_abs_routing_weight_delta"])
            except (TypeError, ValueError) as error:
                raise RouterHealingEvaluationError(
                    "the payload arm reported an unreadable routing-weight delta: "
                    f"{control.get('max_abs_routing_weight_delta')!r}"
                ) from error
            if not math.isfinite(max_routing_delta):
                raise RouterHealingEvaluationError(
                    "the payload arm reported a non-finite routing-weight delta; the "
                    "routing fingerprint did not compare"
                )
            routing_unchanged = (
                routing_top1_equal and max_routing_delta <= ROUTING_ROUNDING_TOLERANCE
            )
            if changed_parameters and routing_unchanged:
                raise RouterHealingEvaluationError(
                    "the payload changed parameters but changed no model output: the routing "
                    "path may not be wired to these tensors at all, so a score from it would "
                    "be fiction"
                )
            if changed_output and not changed_parameters:
                raise RouterHealingEvaluationError(
                    "the model's output changed although no payload parameter changed; the "
                    "evaluation is not measuring what it claims"
                )
            if changed_parameters and not changed_output and not routing_unchanged:
                raise RouterHealingEvaluationError(
                    "the payload changed routing but the reported logits are bit-identical; "
                    "a routing change must reach the model output, so this measurement is "
                    "internally inconsistent"
                )
            if control.get("payload_kind") not in {"replacement", "additive"}:
                raise RouterHealingEvaluationError(
                    f"the applied payload declares an unknown kind {control.get('payload_kind')!r}"
                )

        metrics = result.get("metrics")
        if not isinstance(metrics, Mapping) or not metrics:
            raise RouterHealingEvaluationError(
                "the evaluation worker reported no metrics"
            )
        numeric: dict[str, float] = {}
        for name, value in metrics.items():
            number = float(value)
            if not math.isfinite(number):
                raise RouterHealingEvaluationError(
                    f"evaluation metric {name!r} is not finite ({value!r})"
                )
            numeric[str(name)] = number

        raw_usage = result.get("resource_usage")
        if not isinstance(raw_usage, Mapping):
            raise RouterHealingEvaluationError("the evaluation worker reported no resource usage")
        usage = ResourceUsage.from_wall_time(
            wall_seconds=float(raw_usage.get("wall_seconds", wall_seconds)),
            active_accelerator_count=int(raw_usage.get("active_accelerator_count", 0)),
            visible_accelerator_count=int(raw_usage.get("visible_accelerator_count", 0)),
            peak_vram_gb_by_accelerator=dict(raw_usage.get("peak_vram_gb_by_accelerator", {})),
        )

        # The base arm has no artifact to point at, and it must not pretend to:
        # its source reference names the base content it actually measured.
        source_ref = (
            str(artifact_ref)
            if artifact_ref is not None
            else f"base-model:{spec.base_model_dir}@{spec.base_content_sha256[:12]}"
        )
        return EvaluationOutcome(
            run_id=run_id,
            experiment_id=experiment_id,
            source_artifact_ref=source_ref,
            metrics=numeric,
            gpu_hours=usage.gpu_hours,
            evidence={
                "backend": self.name,
                "arm": "candidate" if payload_arm else "base",
                "payload_applied": payload_arm,
                "eval_spec": spec.to_dict(),
                "eval_spec_digest": spec.digest(),
                "source_identity": dict(identity),
                "source_artifact_ref": artifact_ref,
                "base_identity": result.get("base_identity"),
                "base_holdout_loss": result.get("base_holdout_loss"),
                "candidate_holdout_loss": result.get("candidate_holdout_loss"),
                "application_control": dict(control),
                "payload_verification": result.get("payload_verification"),
                "routing": result.get("routing"),
                "metric_sources": result.get("metric_sources"),
                "phase_ledger": ledger.to_dict(),
                "eval_dir": str(eval_dir),
            },
            resource_usage=usage,
        )

    def cancel(self, run_id: str) -> None:
        self._cancelled.add(run_id)
        process = self._processes.get(run_id)
        if process is not None and process.poll() is None:
            process.terminate()
