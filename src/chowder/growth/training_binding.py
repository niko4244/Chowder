"""The growth cycle's ``TrainingFn``, bound to the production trainer.

``GrowthCycle`` takes a ``TrainingFn`` and never invokes a trainer itself. The
merged framework shipped that seam exercised only against tiny local fakes --
which cannot demonstrate project validation, budget enforcement, registry
lifecycle, contamination binding, worker source identity, or independent
evaluation. This module is the binding that can, and
``docs/GROWTH_TRAINFN_BINDING_PLAN_2026-09-16.md`` is its acceptance contract:

1. **No alternate trainer.** It runs the production entry points,
   ``chowder project-validate`` and then ``chowder train``, as real child
   processes. Nothing here re-implements training.
2. **Existing project validation.** The composed project is approved by the
   real validator before any compute. A refusal starts no trainer.
3. **Existing registries.** Rows are written by the run itself into the ordinary
   ``RunRegistry``; this binding only reads them and audits them.
4. **Existing budget enforcement.** The project's own ``gpu_hour_budget``,
   decomposed sub-budgets and measured preflight stay authoritative. The growth
   envelope is an *additional* ceiling: a template looser than the envelope is
   refused rather than silently clamped, because lowering someone else's declared
   config silently would be its own dishonesty.
5. **Existing contamination/data binding.** Only sources the data registry has
   admitted may supply material, and the firewall's verdict is a refusal, never a
   warning. Both gates run before the corpus is written.
6. **Worker source identity preserved.** The run's declared
   ``chowder_source_identity()`` is matched parent-side against the code this
   process actually imported. A missing identity is a failure, not a pass.
7. **Independent evaluation preserved.** Evaluation outcomes are read back from
   the registry, and the worker artifacts the run left behind are recorded, so
   the training leg and the scoring leg are separately visible.
8. **Hard promotion gate preserved.** This binding reports measurements; it never
   calls ``evaluate_promotion`` and never decides promotion.
9. **Failed/refused experiments remain terminal durable evidence.** Every attempt
   gets its own never-reused directory holding its project file, corpora, process
   output and an evidence record. A failure is reported as a failure; a result
   stranded on a non-terminal row is surfaced, not laundered into success.

Two things this binding deliberately does **not** do. It does not invent a
``search`` section: ``run_project`` has no search config and Chowder's
successive-halving controller has no production caller (``docs/ROADMAP.md``),
so a recipe's knobs are merged into the qualified backend config and nothing
else. And it does not retry in place: a new attempt is a new directory with a new
experiment id.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .contamination import ContaminationFirewall
from .curriculum import CurriculumItem
from .data_registry import DataRegistry
from .recipe_planner import TrainingRecipe

CLI_MODULE = "chowder.cli"
CORPUS_TOKEN = "{corpus}"
ATTEMPT_TOKEN = "{attempt_dir}"

STATUS_SUCCEEDED = "SUCCEEDED"
STATUS_REFUSED = "REFUSED"
STATUS_FAILED = "FAILED"

TERMINAL_STATUSES = frozenset({"passed", "failed", "rejected"})

EVIDENCE_FILE = "training-evidence.json"
WORKER_RESULT_FILE = "worker-result.json"
IDENTITY_FILE = "chowder-identity.json"


class GrowthBindingError(RuntimeError):
    """The binding could not be configured to execute at all."""


@dataclass(frozen=True)
class GrowthEnvelope:
    """The growth cycle's own ceilings, which the run may not exceed.

    ``project_gpu_hour_budget`` is the value the composed project declares to the
    production budget enforcement. ``device_gpu_hours_ceiling`` and
    ``wall_gpu_hours_ceiling`` are checked against the recipe's projection before
    anything is written.
    """

    device_gpu_hours_ceiling: float
    wall_gpu_hours_ceiling: float
    project_gpu_hour_budget: float

    def __post_init__(self) -> None:
        for label, value in (
            ("device_gpu_hours_ceiling", self.device_gpu_hours_ceiling),
            ("wall_gpu_hours_ceiling", self.wall_gpu_hours_ceiling),
            ("project_gpu_hour_budget", self.project_gpu_hour_budget),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise GrowthBindingError(f"{label} must be a number")
            if float(value) < 0:
                raise GrowthBindingError(f"{label} cannot be negative")


@dataclass(frozen=True)
class SubprocessOutcome:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    seconds: float
    timed_out: bool


#: A runner takes the command, a working directory, *extra* environment
#: variables to layer over ``worker_env()``, and a timeout. It is injectable so
#: the refusal layer can prove "no compute happened" without paying for a real
#: trainer. A custom runner inherits the obligation the default one keeps: build
#: the child environment with ``worker_env`` so the child runs this checkout.
Runner = Callable[
    [Sequence[str], Path, Mapping[str, str], "float | None"], SubprocessOutcome
]


def default_runner(
    command: Sequence[str],
    cwd: Path,
    extra_environment: Mapping[str, str],
    timeout_seconds: float | None,
) -> SubprocessOutcome:
    from chowder.worker_env import worker_env

    started = time.perf_counter()
    try:
        completed = subprocess.run(
            list(command),
            cwd=str(cwd),
            env=worker_env({**dict(extra_environment), "PYTHONUNBUFFERED": "1"}),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return SubprocessOutcome(
            command=tuple(command),
            returncode=-1,
            stdout=_as_text(exc.stdout),
            stderr=_as_text(exc.stderr),
            seconds=time.perf_counter() - started,
            timed_out=True,
        )
    return SubprocessOutcome(
        command=tuple(command),
        returncode=int(completed.returncode),
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
        seconds=time.perf_counter() - started,
        timed_out=False,
    )


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def directory_digest(root: Path) -> tuple[str, tuple[dict[str, str], ...]]:
    """A deterministic digest of a directory's contents.

    Recorded so a later reader can prove it read the same artifact. Paths are
    relative and sorted, so the digest does not depend on where the run landed.
    """
    entries: list[dict[str, str]] = []
    if root.is_dir():
        for path in sorted(root.rglob("*"), key=lambda p: p.as_posix()):
            if not path.is_file():
                continue
            entries.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": _sha256_file(path),
                }
            )
    digest = hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return digest, tuple(entries)


def last_json_object(text: str) -> dict[str, Any] | None:
    """The last complete JSON object the CLI printed.

    ``chowder train`` prints run events and then one summary object. Anything
    else -- including nothing -- returns ``None`` so the caller records an
    absence instead of inventing metrics.
    """
    lines = text.splitlines()
    for start in range(len(lines) - 1, -1, -1):
        if lines[start].strip() != "{":
            continue
        try:
            parsed = json.loads("\n".join(lines[start:]))
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "experiment_id" in parsed:
            return parsed
    return None


def _substitute(value: Any, replacements: Mapping[str, str]) -> tuple[Any, set[str]]:
    """Replace placeholder tokens anywhere in a project payload."""
    if isinstance(value, str):
        used = {token for token in replacements if token in value}
        for token, replacement in replacements.items():
            value = value.replace(token, replacement)
        return value, used
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        used: set[str] = set()
        for key, item in value.items():
            new_item, new_used = _substitute(item, replacements)
            out[str(key)] = new_item
            used |= new_used
        return out, used
    if isinstance(value, (list, tuple)):
        out_list: list[Any] = []
        used = set()
        for item in value:
            new_item, new_used = _substitute(item, replacements)
            out_list.append(new_item)
            used |= new_used
        return out_list, used
    return value, set()


class SubprocessTrainingFn:
    """A ``TrainingFn`` that executes one recipe through the production CLI.

    Callable so the cycle can use it directly::

        cycle = GrowthCycle(..., train_fn=SubprocessTrainingFn(...))

    Every call returns a JSON-serializable evidence mapping and writes the same
    mapping to ``<attempt>/training-evidence.json``.
    """

    def __init__(
        self,
        *,
        run_root: str | Path,
        project_template: Mapping[str, Any],
        registry_path: str | Path | None = None,
        envelope: GrowthEnvelope,
        registry: DataRegistry,
        firewall: ContaminationFirewall,
        sources: Mapping[str, str] | None = None,
        material: Mapping[str, Sequence[str]] | None = None,
        python: str | None = None,
        environment: Mapping[str, str] | None = None,
        runner: Runner | None = None,
        timeout_seconds: float = 3600.0,
    ) -> None:
        self.run_root = Path(run_root)
        self.project_template = dict(project_template)
        self.shared_registry_path = (
            Path(registry_path) if registry_path is not None else None
        )
        self.envelope = envelope
        self.registry = registry
        self.firewall = firewall
        self.sources = dict(sources or {})
        self.material = {key: list(value) for key, value in (material or {}).items()}
        self.python = python or sys.executable
        self.environment = dict(environment or {})
        if any(key.upper() == "PYTHONPATH" for key in self.environment):
            raise GrowthBindingError(
                "the binding builds each child's environment with worker_env(), "
                "which places this checkout's package root first on PYTHONPATH; "
                "passing PYTHONPATH here would put a competing chowder ahead of "
                "it, so the run would execute code other than the code being "
                "qualified"
            )
        self.runner = runner or default_runner
        self.timeout_seconds = float(timeout_seconds)

    # ------------------------------------------------------------------
    # the TrainingFn contract
    # ------------------------------------------------------------------

    def __call__(
        self, recipe: TrainingRecipe, items: Sequence[CurriculumItem]
    ) -> Mapping[str, Any]:
        items = tuple(items)
        attempt, attempt_dir = self._reserve_attempt()
        attempt_dir.mkdir(parents=True, exist_ok=True)

        evidence: dict[str, Any] = {
            "binding": "chowder.growth.training_binding",
            "attempt": attempt,
            "attempt_dir": str(attempt_dir),
            "recipe_id": recipe.recipe_id,
            "status": STATUS_REFUSED,
            "train_started": False,
            "refused_by": None,
            "refusal_reason": None,
            "failure_reason": None,
            "projected_device_gpu_hours": float(recipe.projected_device_gpu_hours),
            "projected_wall_gpu_hours": float(recipe.projected_wall_gpu_hours),
            "measured_gpu_hours": None,
            "candidate_succeeded": None,
            "promoted_experiment_id": None,
            "candidate_metrics": {},
            "attempt_row_status": None,
            "commands": [],
            "validate": None,
            "train": None,
            "summary": None,
            "project_path": None,
            "registry_path": None,
            "shared_registry_path": (
                str(self.shared_registry_path) if self.shared_registry_path else None
            ),
            "experiment_id": None,
            "rows": [],
            "registry_audit": [],
            "evaluation": {"outcome_count": 0, "metrics": {}, "gpu_hours": None},
            "worker_artifacts": [],
            "artifact_ref": None,
            "artifact_sha256": None,
            "artifact_files": [],
            "material": {},
            "source_identity": {"verified": False, "reason": "not checked"},
            "notes": [],
        }

        # 1. cost gates -- nothing is written, and no process is started.
        cost_refusal = self._check_cost(recipe)
        if cost_refusal:
            return self._finish(attempt_dir, evidence, STATUS_REFUSED, cost_refusal)

        # 2. template contract
        composed, contract_refusal = self._compose(recipe, attempt, attempt_dir)
        if contract_refusal:
            return self._finish(attempt_dir, evidence, STATUS_REFUSED, contract_refusal)
        self._set(evidence, composed)

        experiment_id = composed["experiment"]["experiment_id"]
        evidence["experiment_id"] = experiment_id

        registry_path = self.registry_for(attempt_dir)
        evidence["registry_path"] = str(registry_path)
        composed["registry_path"] = str(registry_path)

        # A retry must not walk into the registry's duplicate-id refusal after
        # doing compute. (Property 9: never retried in place.)
        if self._registry_has(registry_path, experiment_id):
            return self._finish(
                attempt_dir,
                evidence,
                STATUS_REFUSED,
                (
                    "attempt-identity",
                    f"experiment {experiment_id!r} already exists in "
                    f"{registry_path}; a retry is a new attempt, never a rerun",
                ),
            )

        # Production's automatic baseline writes an experiment literally named
        # `baseline` and the registry refuses a duplicate id, so a registry can
        # hold exactly one automatic baseline. Fail here, with the constraint
        # named, rather than inside a child traceback after a model load.
        if self._registry_baseline_conflict(registry_path, composed):
            return self._finish(
                attempt_dir,
                evidence,
                STATUS_REFUSED,
                (
                    "registry-baseline",
                    f"{registry_path} already holds an automatic baseline, and "
                    "production's registry refuses a duplicate `baseline` row: "
                    "a registry can carry exactly one. Either give each attempt "
                    "its own registry (the default when registry_path is "
                    "omitted) or declare baseline.mode='fixed' from the "
                    "measurement already recorded",
                ),
            )

        # 3. data admission and contamination gates, still before any compute.
        material_refusal = self._check_material(items)
        if material_refusal:
            return self._finish(attempt_dir, evidence, STATUS_REFUSED, material_refusal)

        corpus = self._write_corpus(attempt_dir, items)
        evidence["material"] = {
            "item_ids": [item.item_id for item in items],
            "source_ids": sorted({self.sources[item.item_id] for item in items}),
            "corpus_path": str(corpus["path"]),
            "corpus_sha256": corpus["sha256"],
            "corpus_examples": corpus["examples"],
        }

        replacements = {
            CORPUS_TOKEN: str(corpus["path"]),
            ATTEMPT_TOKEN: str(attempt_dir),
        }
        composed, _used = _substitute(composed, replacements)
        composed["work_dir"] = str(attempt_dir / "work")
        if CORPUS_TOKEN in json.dumps(composed):
            return self._finish(
                attempt_dir,
                evidence,
                STATUS_REFUSED,
                (
                    "template-contract",
                    f"the project template still contains {CORPUS_TOKEN} after "
                    "substitution; the materialized corpus has no declared home",
                ),
            )

        project_path = attempt_dir / "project.json"
        project_path.write_text(
            json.dumps(composed, indent=2, sort_keys=True), encoding="utf-8"
        )
        evidence["project_path"] = str(project_path)

        # 4. the real validator. A refusal here starts no trainer.
        validate = self._run_cli("project-validate", project_path, attempt_dir, "validate")
        evidence["validate"] = self._process_record(validate, attempt_dir)
        evidence["commands"].append(list(validate.command))
        if validate.returncode != 0 or validate.timed_out:
            return self._finish(
                attempt_dir,
                evidence,
                STATUS_REFUSED,
                (
                    "project-validate",
                    f"the project was refused before compute (exit {validate.returncode}"
                    f"{', timed out' if validate.timed_out else ''})",
                ),
            )

        # 5. the real trainer.
        evidence["train_started"] = True
        train = self._run_cli("train", project_path, attempt_dir, "train")
        evidence["train"] = self._process_record(train, attempt_dir)
        evidence["commands"].append(list(train.command))

        summary = last_json_object(train.stdout)
        evidence["summary"] = summary
        if summary is None:
            reason = (
                "the training process timed out without printing a summary"
                if train.timed_out
                else f"the training process printed no parsable summary "
                f"(exit {train.returncode})"
            )
            return self._finish(attempt_dir, evidence, STATUS_FAILED, None, reason)

        evidence["measured_gpu_hours"] = _as_float(summary.get("gpu_hours"))
        evidence["artifact_ref"] = summary.get("artifact_ref")
        evidence["candidate_succeeded"] = summary.get("succeeded") is True
        evidence["promoted_experiment_id"] = summary.get("promoted_experiment_id")
        evidence["candidate_metrics"] = {
            str(key): float(value)
            for key, value in dict(summary.get("metrics") or {}).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        if summary.get("succeeded") is not True:
            return self._finish(
                attempt_dir,
                evidence,
                STATUS_FAILED,
                None,
                f"the training process reported failure: {summary.get('error')}",
            )

        return self._settle(attempt_dir, evidence, summary, recipe)

    # ------------------------------------------------------------------
    # gates
    # ------------------------------------------------------------------

    def _check_cost(self, recipe: TrainingRecipe) -> tuple[str, str] | None:
        device = float(recipe.projected_device_gpu_hours)
        wall = float(recipe.projected_wall_gpu_hours)
        if device > self.envelope.device_gpu_hours_ceiling:
            return (
                "growth-envelope",
                f"recipe projects {device:.6f} device GPU-h against "
                f"envelope.device_gpu_hours_ceiling "
                f"{self.envelope.device_gpu_hours_ceiling:.6f}; the growth "
                "envelope does not enlarge a ceiling after measurement",
            )
        if wall > self.envelope.wall_gpu_hours_ceiling:
            return (
                "growth-envelope",
                f"recipe projects {wall:.6f} wall GPU-h against "
                f"envelope.wall_gpu_hours_ceiling "
                f"{self.envelope.wall_gpu_hours_ceiling:.6f}",
            )
        return None

    def _compose(
        self, recipe: TrainingRecipe, attempt: str, attempt_dir: Path
    ) -> tuple[dict[str, Any], tuple[str, str] | None]:
        """Merge the recipe into the template without inventing config.

        Returns ``(composed, refusal)`` with the refusal winning when the
        template cannot be executed honestly.
        """
        from chowder.graph import deep_merge_config

        template = json.loads(json.dumps(self.project_template))
        config = template.get("config")
        if not isinstance(config, Mapping):
            return template, (
                "template-contract",
                "the project template declares no config section to merge into",
            )

        merged_config = deep_merge_config(config, recipe.to_config_patch())
        if "search" in merged_config or "search" in template:
            return template, (
                "template-contract",
                "a `search` section is present; run_project has no search config "
                "and no production caller for the successive-halving controller, "
                "so this binding will not pretend to drive one",
            )
        template["config"] = merged_config

        goal = template.get("goal")
        if not isinstance(goal, Mapping) or "gpu_hour_budget" not in goal:
            return template, (
                "template-contract",
                "the project template declares no goal.gpu_hour_budget; the "
                "binding cannot enforce a ceiling the run does not have",
            )
        declared = goal["gpu_hour_budget"]
        if not isinstance(declared, (int, float)) or isinstance(declared, bool):
            return template, (
                "template-contract",
                "goal.gpu_hour_budget must be a number",
            )
        if float(declared) > self.envelope.project_gpu_hour_budget:
            return template, (
                "template-contract",
                f"the project declares {float(declared):.6f} GPU-h, looser than "
                f"the growth envelope's project_gpu_hour_budget "
                f"{self.envelope.project_gpu_hour_budget:.6f}; the envelope "
                "cannot raise production's ceiling and this binding will not "
                "silently lower the project's",
            )

        body = json.dumps(template)
        if CORPUS_TOKEN not in body:
            return template, (
                "template-contract",
                f"the project template never names {CORPUS_TOKEN}; the binding "
                "cannot know which knob takes the materialized corpus, and "
                "guessing would train on whatever the template already pointed at",
            )

        experiment = template.get("experiment")
        if not isinstance(experiment, Mapping) or not experiment.get("experiment_id"):
            return template, (
                "template-contract",
                "the project template declares no experiment.experiment_id to "
                "derive a duplicate-safe attempt identity from",
            )
        derived = f"{experiment['experiment_id']}-a{attempt.split('-')[-1]}"
        patched_experiment = dict(experiment)
        patched_experiment["experiment_id"] = derived
        template["experiment"] = patched_experiment
        return template, None

    def _check_material(self, items: Sequence[CurriculumItem]) -> tuple[str, str] | None:
        if not items:
            return (
                "data-registry",
                "no curriculum item was supplied, so the run would have no "
                "accountable training material",
            )
        for item in items:
            source_id = self.sources.get(item.item_id)
            if not source_id:
                return (
                    "data-registry",
                    f"curriculum item {item.item_id!r} names no registered data "
                    "source; material with no provenance is not trainable",
                )
            source = self.registry.get(source_id)
            if source is None:
                return (
                    "data-registry",
                    f"data source {source_id!r} is not registered",
                )
            if not source.trainable:
                return (
                    "data-registry",
                    f"data source {source_id!r} is not trainable "
                    f"(trust_class={source.trust_class}, "
                    f"inclusion_decision={source.inclusion_decision}, "
                    f"contamination={source.contamination_relationship})",
                )
            samples = list(self.material.get(item.item_id, ()))
            if not samples:
                return (
                    "data-registry",
                    f"curriculum item {item.item_id!r} has no material to train on",
                )
            verdict = self.firewall.check_source(source_id=source_id, samples=samples)
            if verdict.verdict != "CLEAN":
                matched = ", ".join(
                    sorted({match.benchmark_qualified_id for match in verdict.matches})
                )
                return (
                    "contamination-firewall",
                    f"source {source_id!r} offered for {item.item_id!r} is "
                    f"{verdict.verdict}"
                    + (f" (matches {matched})" if matched else "")
                    + "; a contamination refusal is a refusal, not a warning",
                )
        return None

    def _write_corpus(
        self, attempt_dir: Path, items: Sequence[CurriculumItem]
    ) -> dict[str, Any]:
        """Materialize the admitted material once, deterministically, as LF text."""
        lines: list[str] = []
        for item in sorted(items, key=lambda entry: entry.item_id):
            lines.extend(
                " ".join(str(line).split())
                for line in self.material.get(item.item_id, ())
                if str(line).strip()
            )
        payload = "\n".join(lines) + "\n"
        path = attempt_dir / "corpus.txt"
        path.write_bytes(payload.encode("utf-8"))
        return {
            "path": path,
            "sha256": _sha256_bytes(payload.encode("utf-8")),
            "examples": len(lines),
        }

    # ------------------------------------------------------------------
    # execution
    # ------------------------------------------------------------------

    def _run_cli(
        self, verb: str, project_path: Path, attempt_dir: Path, tag: str
    ) -> SubprocessOutcome:
        command = [self.python, "-m", CLI_MODULE, verb, str(project_path)]
        outcome = self.runner(
            command, attempt_dir, dict(self.environment), self.timeout_seconds
        )
        (attempt_dir / f"{tag}.stdout.txt").write_text(outcome.stdout, encoding="utf-8")
        (attempt_dir / f"{tag}.stderr.txt").write_text(outcome.stderr, encoding="utf-8")
        return outcome

    def _process_record(
        self, outcome: SubprocessOutcome, attempt_dir: Path
    ) -> dict[str, Any]:
        return {
            "returncode": outcome.returncode,
            "seconds": outcome.seconds,
            "timed_out": outcome.timed_out,
            "stdout_ref": str(attempt_dir / f"{self._tag_for(outcome)}.stdout.txt"),
            "stderr_ref": str(attempt_dir / f"{self._tag_for(outcome)}.stderr.txt"),
        }

    @staticmethod
    def _tag_for(outcome: SubprocessOutcome) -> str:
        verb = outcome.command[-2] if len(outcome.command) >= 2 else "run"
        return verb.replace("project-validate", "validate")

    # ------------------------------------------------------------------
    # settling
    # ------------------------------------------------------------------

    def _settle(
        self,
        attempt_dir: Path,
        evidence: dict[str, Any],
        summary: Mapping[str, Any],
        recipe: TrainingRecipe,
    ) -> Mapping[str, Any]:
        """Read the production evidence back and decide the binding's verdict."""
        registry_path = Path(evidence["registry_path"])
        rows, audit = self._read_registry(registry_path)
        evidence["rows"] = rows
        evidence["registry_audit"] = audit
        evidence["evaluation"] = self._read_evaluation(registry_path)
        evidence["worker_artifacts"] = self._read_worker_artifacts(attempt_dir)
        evidence["source_identity"] = self._verify_source_identity(attempt_dir)

        artifact_ref = evidence["artifact_ref"]
        if artifact_ref:
            digest, files = directory_digest(Path(artifact_ref))
            evidence["artifact_sha256"] = digest
            evidence["artifact_files"] = [dict(entry) for entry in files]
            if not files:
                evidence["notes"].append(
                    f"the run named artifact {artifact_ref!r} but it holds no files"
                )
        else:
            evidence["notes"].append("the run reported no artifact")

        failures: list[str] = []

        if audit:
            failures.append(
                f"{len(audit)} result(s) stranded on non-terminal rows: "
                + ", ".join(
                    f"{entry['experiment_id']}={entry['status']}" for entry in audit
                )
            )

        own_id = evidence["experiment_id"]
        own_rows = [row for row in rows if row["experiment_id"] == own_id]
        evidence["attempt_row_status"] = own_rows[0]["status"] if own_rows else None
        if not own_rows:
            failures.append(
                f"the registry holds no row for this attempt ({own_id!r}); the "
                "run left no durable experiment evidence"
            )
        elif not all(row["status"] in TERMINAL_STATUSES for row in own_rows):
            failures.append(
                f"attempt {own_id!r} is not terminal: "
                + ", ".join(f"{row['experiment_id']}={row['status']}" for row in own_rows)
            )

        if not evidence["source_identity"]["verified"]:
            failures.append(
                "the run's chowder source identity was not verified: "
                + str(evidence["source_identity"].get("reason"))
            )

        if not artifact_ref:
            failures.append("the run produced no artifact to evaluate or promote")

        if failures:
            return self._finish(
                attempt_dir, evidence, STATUS_FAILED, None, "; ".join(failures)
            )
        return self._finish(attempt_dir, evidence, STATUS_SUCCEEDED, None)

    def registry_for(self, attempt_dir: Path) -> Path:
        """The registry this attempt writes into.

        Per attempt by default. A shared registry is available, but production
        permits it only one automatic baseline (see the refusal below), so the
        default keeps each attempt self-contained and individually auditable.
        """
        if self.shared_registry_path is not None:
            return self.shared_registry_path
        return attempt_dir / "runs.db"

    def _read_registry(self, registry_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        from chowder.registry import RunRegistry

        if not registry_path.exists():
            return [], []
        registry = RunRegistry(registry_path)
        try:
            rows = [
                {"experiment_id": experiment.experiment_id, "status": experiment.status.value}
                for experiment in registry.list_experiments()
            ]
            audit = [dict(entry) for entry in registry.audit_stranded_results()]
        finally:
            registry.close()
        return rows, audit

    @staticmethod
    def _registry_has(registry_path: Path, experiment_id: str) -> bool:
        from chowder.registry import RunRegistry

        if not registry_path.exists():
            return False
        registry = RunRegistry(registry_path)
        try:
            return bool(registry.has_experiment(experiment_id))
        finally:
            registry.close()

    @staticmethod
    def _registry_baseline_conflict(
        registry_path: Path, composed: Mapping[str, Any]
    ) -> bool:
        baseline = composed.get("baseline")
        if not isinstance(baseline, Mapping):
            return False
        if str(baseline.get("mode", "auto")).strip().lower() != "auto":
            return False
        return SubprocessTrainingFn._registry_has(registry_path, "baseline")

    def _read_evaluation(self, registry_path: Path) -> dict[str, Any]:
        from chowder.registry import RunRegistry

        if not registry_path.exists():
            return {"outcome_count": 0, "metrics": {}, "gpu_hours": None}
        registry = RunRegistry(registry_path)
        try:
            outcomes = list(registry.list_evaluation_outcomes())
        finally:
            registry.close()
        metrics: dict[str, float] = {}
        hours: float | None = None
        for outcome in outcomes:
            metrics.update({str(k): float(v) for k, v in dict(outcome.metrics).items()})
            hours = float(outcome.gpu_hours) if hours is None else hours + float(outcome.gpu_hours)
        return {
            "outcome_count": len(outcomes),
            "metrics": metrics,
            "gpu_hours": hours,
            "run_ids": [str(outcome.run_id) for outcome in outcomes],
        }

    @staticmethod
    def _read_worker_artifacts(attempt_dir: Path) -> list[dict[str, Any]]:
        """Every worker result the run left behind, with its declared kind.

        A training leg and a scoring leg are separate processes; recording both
        artifact paths is what makes that visible without trusting a claim.
        """
        found: list[dict[str, Any]] = []
        for path in sorted(attempt_dir.rglob(WORKER_RESULT_FILE), key=lambda p: p.as_posix()):
            entry: dict[str, Any] = {"path": str(path), "kind": None, "global_step": None}
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                entry["kind"] = "unreadable"
                found.append(entry)
                continue
            if isinstance(payload, Mapping):
                entry["kind"] = payload.get("kind")
                entry["global_step"] = payload.get("global_step")
            found.append(entry)
        return found

    def _verify_source_identity(self, attempt_dir: Path) -> dict[str, Any]:
        """Parent-side half of the identity check.

        The worker refuses child-side when the code it imported differs from what
        the controller declared (``worker_env.verify_source_identity``). This is
        the controller's half: the declarations the run left behind must match
        the code *this* process is running.
        """
        from chowder.worker_env import chowder_source_identity

        declared_paths = sorted(
            attempt_dir.rglob(IDENTITY_FILE), key=lambda p: p.as_posix()
        )
        if not declared_paths:
            return {
                "verified": False,
                "reason": (
                    "the run left no chowder-identity.json: the source identity "
                    "the worker verified was not recorded, so it cannot be "
                    "checked (unchecked is not verified)"
                ),
                "files": [],
            }
        try:
            actual = chowder_source_identity()
        except Exception as exc:  # pragma: no cover - defensive
            return {
                "verified": False,
                "reason": f"this process could not compute its own source identity: {exc}",
                "files": [str(path) for path in declared_paths],
            }
        recorded: list[dict[str, Any]] = []
        for path in declared_paths:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                return {
                    "verified": False,
                    "reason": f"{path} is not readable JSON: {exc}",
                    "files": [str(p) for p in declared_paths],
                }
            sha = payload.get("source_sha256") if isinstance(payload, Mapping) else None
            recorded.append({"path": str(path), "source_sha256": sha})
            if sha != actual["source_sha256"]:
                return {
                    "verified": False,
                    "reason": (
                        f"{path} declares chowder source {str(sha)[:12]}… but this "
                        f"process imported {actual['source_sha256'][:12]}… -- the "
                        "run did not execute the code that is being qualified"
                    ),
                    "files": [str(p) for p in declared_paths],
                }
        return {
            "verified": True,
            "reason": "every recorded chowder identity matches this process's code",
            "source_sha256": actual["source_sha256"],
            "source_root": actual["source_root"],
            "files": [str(path) for path in declared_paths],
            "recorded": recorded,
        }

    # ------------------------------------------------------------------
    # attempt bookkeeping
    # ------------------------------------------------------------------

    def _reserve_attempt(self) -> tuple[str, Path]:
        """The smallest attempt index whose directory does not exist yet.

        Never reuse a directory: a refused or failed attempt keeps its evidence.
        """
        index = 1
        while True:
            name = f"attempt-{index:02d}"
            path = self.run_root / name
            if not path.exists():
                return name, path
            index += 1

    @staticmethod
    def _set(evidence: dict[str, Any], composed: Mapping[str, Any]) -> None:
        experiment = composed.get("experiment")
        if isinstance(experiment, Mapping):
            evidence["experiment_id"] = experiment.get("experiment_id")

    def _finish(
        self,
        attempt_dir: Path,
        evidence: dict[str, Any],
        status: str,
        refusal: tuple[str, str] | None,
        failure_reason: str | None = None,
    ) -> Mapping[str, Any]:
        evidence["status"] = status
        if refusal is not None:
            evidence["refused_by"], evidence["refusal_reason"] = refusal
        if failure_reason is not None:
            evidence["failure_reason"] = failure_reason
        evidence["recorded_at"] = _utc_now()
        evidence_path = attempt_dir / EVIDENCE_FILE
        evidence["evidence_path"] = str(evidence_path)
        evidence_path.write_text(
            json.dumps(evidence, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        return evidence


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
