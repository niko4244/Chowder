"""The declared kernel-side payload: materialize the corpus, then train.

The dispatcher never invents what runs inside a kernel. A campaign declares
it: :class:`CorpusTraining` names the argv
(``python -m chowder.growth.kaggle_payload``) and the per-attempt
declaration the request carries -- the recipe's knobs, the curriculum item
ids the attempt trains on, and the project ceiling the composed project may
not exceed. ``run_kernel_job`` verifies the declared inputs, resolves each to
the mount path whose bytes match the declaration, writes the attempt context
into the kernel's output directory, and runs that command. :func:`run_payload`
is the kernel half: ``SubprocessTrainingFn``'s corpus materialization and
production entry points, ported so the remote attempt trains on exactly what
the local one would.

The port, step by step:

* ``_check_material`` -> every declared item has material, and every item
  names a source the declared ``data-registry.json`` registers, includes and
  marks trainable (the declared ``contamination.json`` verdicts are read as
  well: a source the manifest does not call clean refuses);
* ``_write_corpus`` -> the declared material lines, items sorted by id,
  whitespace-normalized, LF-joined with a trailing newline, sha256 recorded;
* ``_compose`` -> the declared project template merged with the recipe's own
  config patch for the template's declared backend type, ``{corpus}`` and
  ``{attempt_dir}`` substituted, the attempt's experiment id derived, and the
  project budget checked against the declared envelope -- refusing, never
  lowering, a looser project;
* ``_run_cli`` -> ``chowder.cli project-validate`` then ``train``, with the
  same last-JSON summary parser deciding the candidate's verdict.

The payload writes ``run-summary.json`` in the local executor's evidence
vocabulary: the corpus digest and item ids, the commands with their logs, the
production summary, and the trained artifact made relative to the output
directory so :func:`merge_run_summary` can bind it into the attempt's
evidence on the dispatcher. It writes the resume marker ``run_kernel_job``
reads, and exits 0 only when the production summary says the candidate
succeeded, so a non-zero kernel record never carries a success.

What it deliberately does not do: it does not rewrite paths the campaign did
not declare. A template whose declared paths (a base model, an auxiliary
file) are not reachable in the kernel fails in the production entry points,
audibly, rather than having this module guess a mount for a path no
declaration binds.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ClassVar, Mapping, MutableMapping, Sequence

from .compute_backend import (
    OUTCOME_FAILED,
    OUTCOME_SUCCEEDED,
    ComputeBackendRefusal,
)
from .data_registry import DataRegistry, DataSource, admit
from .kaggle_kernel import ATTEMPT_CONTEXT_NAME, RESUME_STATE_NAME, sha256_file
from .recipe_planner import TrainingRecipe
from .training_binding import (
    ATTEMPT_TOKEN,
    CLI_MODULE,
    CORPUS_TOKEN,
    _substitute,
    directory_digest,
    last_json_object,
)

__all__ = [
    "KAGGLE_PAYLOAD_MODULE",
    "CORPUS_TRAINING_KIND",
    "RUN_SUMMARY_NAME",
    "CORPUS_NAME",
    "PROJECT_NAME",
    "KAGGLE_PAYLOAD_SCHEMA",
    "KAGGLE_PAYLOAD_DECLARATION_MISSING",
    "KAGGLE_PAYLOAD_KIND_UNKNOWN",
    "KAGGLE_PAYLOAD_INPUTS_MISSING",
    "KAGGLE_PAYLOAD_INPUT_UNREADABLE",
    "KAGGLE_PAYLOAD_ITEMS_EMPTY",
    "KAGGLE_PAYLOAD_MATERIAL_MISSING",
    "KAGGLE_PAYLOAD_SOURCE_UNREGISTERED",
    "KAGGLE_PAYLOAD_SOURCE_NOT_TRAINABLE",
    "KAGGLE_PAYLOAD_CONTAMINATION_DECLARED",
    "KAGGLE_PAYLOAD_TEMPLATE_CONTRACT",
    "KAGGLE_PAYLOAD_PROJECT_BUDGET",
    "KAGGLE_PAYLOAD_VALIDATE_FAILED",
    "KAGGLE_PAYLOAD_CONTEXT_UNREADABLE",
    "KAGGLE_PAYLOAD_OUTPUT_UNWRITABLE",
    "CorpusTraining",
    "run_payload",
    "merge_run_summary",
    "main",
]

#: The module the declared command runs inside the kernel.
KAGGLE_PAYLOAD_MODULE = "chowder.growth.kaggle_payload"
#: The one payload kind this module implements.
CORPUS_TRAINING_KIND = "corpus-training"

#: The summary the payload leaves in the output directory. It is an artifact
#: like any other: the kernel's own manifest binds it, and
#: :func:`merge_run_summary` reads it back on the dispatcher.
RUN_SUMMARY_NAME = "run-summary.json"
#: The materialized corpus and the composed project, named as the local
#: executor names them.
CORPUS_NAME = "corpus.txt"
PROJECT_NAME = "project.json"
#: The project's runtime directories, declared inside the output directory so
#: everything a production run writes is returned and bound by the manifest.
WORK_DIR_NAME = "work"
REGISTRY_NAME = "runs.db"

#: What a controller appends to evidence when the production verdict and the
#: kernel record disagree (the kernel record is a claim; the summary is the
#: run's own report).
PRODUCTION_VERDICT_MISMATCH = "KAGGLE_PAYLOAD_PRODUCTION_VERDICT_MISMATCH"

#: Machine-readable refusal identifiers, first token of the exception message.
KAGGLE_PAYLOAD_SCHEMA = "KAGGLE_PAYLOAD_SCHEMA"
KAGGLE_PAYLOAD_DECLARATION_MISSING = "KAGGLE_PAYLOAD_DECLARATION_MISSING"
KAGGLE_PAYLOAD_KIND_UNKNOWN = "KAGGLE_PAYLOAD_KIND_UNKNOWN"
KAGGLE_PAYLOAD_INPUTS_MISSING = "KAGGLE_PAYLOAD_INPUTS_MISSING"
KAGGLE_PAYLOAD_INPUT_UNREADABLE = "KAGGLE_PAYLOAD_INPUT_UNREADABLE"
KAGGLE_PAYLOAD_ITEMS_EMPTY = "KAGGLE_PAYLOAD_ITEMS_EMPTY"
KAGGLE_PAYLOAD_MATERIAL_MISSING = "KAGGLE_PAYLOAD_MATERIAL_MISSING"
KAGGLE_PAYLOAD_SOURCE_UNREGISTERED = "KAGGLE_PAYLOAD_SOURCE_UNREGISTERED"
KAGGLE_PAYLOAD_SOURCE_NOT_TRAINABLE = "KAGGLE_PAYLOAD_SOURCE_NOT_TRAINABLE"
KAGGLE_PAYLOAD_CONTAMINATION_DECLARED = "KAGGLE_PAYLOAD_CONTAMINATION_DECLARED"
KAGGLE_PAYLOAD_TEMPLATE_CONTRACT = "KAGGLE_PAYLOAD_TEMPLATE_CONTRACT"
KAGGLE_PAYLOAD_PROJECT_BUDGET = "KAGGLE_PAYLOAD_PROJECT_BUDGET"
KAGGLE_PAYLOAD_VALIDATE_FAILED = "KAGGLE_PAYLOAD_VALIDATE_FAILED"
KAGGLE_PAYLOAD_CONTEXT_UNREADABLE = "KAGGLE_PAYLOAD_CONTEXT_UNREADABLE"
KAGGLE_PAYLOAD_OUTPUT_UNWRITABLE = "KAGGLE_PAYLOAD_OUTPUT_UNWRITABLE"

#: The declared prepared inputs the training path reads. The other declared
#: inputs (the parent arms, the evaluation material, the hardware budget)
#: belong to other phases; the kernel still verifies their digests, and their
#: absence here is not an excuse to fetch anything from somewhere else.
TRAINING_INPUTS: tuple[str, ...] = (
    "project_template_path",
    "training_material_path",
    "data_registry_path",
)

#: Declared contamination statuses a source may train under. The firewall's
#: own vocabulary: CLEAN, or a possible match explicitly cleared.
TRAINABLE_CONTAMINATION_STATUSES = frozenset({"CLEAN", "POSSIBLE_CLEARED"})

CommandRunner = Callable[
    [Sequence[str], Path, Mapping[str, str]], "subprocess.CompletedProcess[str]"
]


# --------------------------------------------------------------------------
# the declaration: what a campaign says the kernel runs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusTraining:
    """Declares the corpus-training payload for a campaign's attempts.

    One object per campaign: :meth:`command` is the argv every attempt's
    kernel runs (a declaration, not a default -- ``KaggleTrainingFn`` refuses
    a payload declaration that disagrees with a command passed beside it),
    and :meth:`payload` binds one attempt's recipe and curriculum items into
    the declaration the request carries. ``project_gpu_hour_budget`` is the
    envelope's project ceiling: the composed project may not declare more,
    and the kernel refuses rather than lowering a looser project.
    """

    project_gpu_hour_budget: float
    python: str = "python"

    kind: ClassVar[str] = CORPUS_TRAINING_KIND

    def __post_init__(self) -> None:
        if (
            isinstance(self.project_gpu_hour_budget, bool)
            or not isinstance(self.project_gpu_hour_budget, (int, float))
            or not math.isfinite(float(self.project_gpu_hour_budget))
            or float(self.project_gpu_hour_budget) < 0.0
        ):
            raise ComputeBackendRefusal(
                KAGGLE_PAYLOAD_SCHEMA,
                "project_gpu_hour_budget must be finite and non-negative, got "
                f"{self.project_gpu_hour_budget!r}",
            )
        if not isinstance(self.python, str) or not self.python.strip():
            raise ComputeBackendRefusal(
                KAGGLE_PAYLOAD_SCHEMA,
                f"python must name the kernel interpreter, got {self.python!r}",
            )

    def command(self) -> list[str]:
        """The argv a kernel runs for one attempt of this payload."""
        return [self.python.strip(), "-m", KAGGLE_PAYLOAD_MODULE]

    def payload(
        self, recipe: TrainingRecipe, items: Sequence[Any]
    ) -> dict[str, Any]:
        """One attempt's declaration, carried in the request payload.

        The items are the curriculum the runner selected for the attempt
        (``run_campaign`` passes the same planned items every recipe trains
        on); their ids select exactly those items' declared material in the
        kernel. An item with no id would select nothing, so it refuses here
        rather than silently shrinking the corpus.
        """
        if not isinstance(recipe, TrainingRecipe):
            raise ComputeBackendRefusal(
                KAGGLE_PAYLOAD_SCHEMA,
                f"recipe must be a TrainingRecipe, got {type(recipe).__name__}",
            )
        item_ids: list[str] = []
        for item in items:
            item_id = str(getattr(item, "item_id", "") or "").strip()
            if not item_id:
                raise ComputeBackendRefusal(
                    KAGGLE_PAYLOAD_SCHEMA,
                    f"curriculum item {item!r} carries no item_id; the payload "
                    "materializes by id, so an id-less item selects no material",
                )
            item_ids.append(item_id)
        if not item_ids:
            raise ComputeBackendRefusal(
                KAGGLE_PAYLOAD_ITEMS_EMPTY,
                "the attempt declares no curriculum items, so the corpus "
                "would be empty; a run with nothing to train on is not a run",
            )
        return {
            "kind": CORPUS_TRAINING_KIND,
            "recipe": recipe.to_dict(),
            "item_ids": item_ids,
            "project_gpu_hour_budget": float(self.project_gpu_hour_budget),
        }


# --------------------------------------------------------------------------
# the kernel half
# --------------------------------------------------------------------------


def run_payload(
    context: Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    python: str | None = None,
    environment: Mapping[str, str] | None = None,
    runner: CommandRunner | None = None,
) -> dict[str, Any]:
    """Run the declared payload; returns the summary it wrote.

    Every declared outcome -- refused, failed, succeeded -- comes back as a
    summary document and lands on disk as ``run-summary.json``; only an
    output directory that cannot be written escapes as an exception, because
    then there is no place to record anything.
    """
    run = _PayloadRun(
        context,
        output_dir=output_dir,
        python=python,
        environment=environment,
        runner=runner,
    )
    try:
        run.execute()
    except ComputeBackendRefusal as refusal:
        run.refuse(refusal)
    run.finish()
    return run.summary


def merge_run_summary(
    evidence: MutableMapping[str, Any], attempt_dir: str | Path
) -> MutableMapping[str, Any]:
    """Merge a returned attempt's declared run summary into its evidence.

    Runs on the dispatcher after the transport copied the attempt directory
    back. The summary is one of the artifacts the kernel's own manifest bound,
    so its verdict is read here rather than re-derived; what it adds is what
    the outcome alone cannot know -- the production experiment id, metrics and
    the *trained artifact* (as a path inside the returned attempt directory,
    with the local executor's directory digest), instead of whatever file
    happened to sort first in the manifest.

    A summary whose production verdict is not success while the outcome claims
    success tightens the evidence to FAILED: the kernel record is a claim, and
    the run's own report wins. A summary that cannot be read or names nothing
    inside the attempt directory leaves the evidence as the manifest bound it
    and says so in ``notes``.
    """
    attempt_root = Path(attempt_dir)
    summary_path = attempt_root / RUN_SUMMARY_NAME
    if not summary_path.is_file():
        return evidence
    notes: list[str] = evidence.setdefault("notes", [])
    try:
        document = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        notes.append(f"the attempt's {RUN_SUMMARY_NAME} cannot be read: {exc}")
        return evidence
    if not isinstance(document, Mapping):
        notes.append(f"the attempt's {RUN_SUMMARY_NAME} is not a JSON object")
        return evidence
    evidence["run_summary"] = dict(document)

    corpus = document.get("corpus")
    if isinstance(corpus, Mapping):
        corpus_path = attempt_root / CORPUS_NAME
        evidence["material"] = {
            "item_ids": [str(item) for item in (corpus.get("item_ids") or [])],
            "source_ids": [str(source) for source in (corpus.get("source_ids") or [])],
            "corpus_path": (
                str(corpus_path) if corpus_path.is_file() else str(corpus.get("path") or "")
            ),
            "corpus_sha256": str(corpus.get("sha256") or ""),
            "corpus_examples": _as_int(corpus.get("examples")),
        }

    training = document.get("training")
    if not isinstance(training, Mapping):
        return evidence
    metrics = {
        str(key): float(value)
        for key, value in dict(training.get("metrics") or {}).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    if metrics:
        evidence["candidate_metrics"] = metrics
    if training.get("promoted_experiment_id") is not None:
        evidence["promoted_experiment_id"] = training.get("promoted_experiment_id")
    production_hours = training.get("gpu_hours")
    if isinstance(production_hours, (int, float)) and not isinstance(production_hours, bool):
        evidence["production_gpu_hours"] = float(production_hours)

    produced_succeeded = training.get("succeeded") is True
    if evidence.get("status") != OUTCOME_SUCCEEDED:
        return evidence
    if not produced_succeeded:
        evidence["status"] = OUTCOME_FAILED
        evidence["candidate_succeeded"] = False
        evidence["failure_reason"] = str(
            training.get("error")
            or "the declared run summary reports the candidate did not succeed"
        )
        evidence["production_verdict"] = PRODUCTION_VERDICT_MISMATCH
        return evidence

    artifact = _artifact_under(attempt_root, training.get("artifact_path"))
    if artifact is None:
        notes.append(
            "the declared run summary names no trained artifact inside the "
            "returned attempt directory, so the manifest's first file stands "
            "as the recorded artifact"
        )
        return evidence
    digest, files = directory_digest(artifact)
    evidence["artifact_ref"] = str(artifact)
    evidence["artifact_sha256"] = digest
    evidence["artifact_files"] = [dict(entry) for entry in files]
    return evidence


class _PayloadRun:
    """One attempt's payload execution, stage by stage."""

    def __init__(
        self,
        context: Mapping[str, Any],
        *,
        output_dir: str | Path | None,
        python: str | None,
        environment: Mapping[str, str] | None,
        runner: CommandRunner | None,
    ) -> None:
        self.context: Mapping[str, Any] = (
            dict(context) if isinstance(context, Mapping) else {}
        )
        recorded = str(self.context.get("output_dir") or "").strip()
        root = (
            Path(output_dir)
            if output_dir is not None
            else Path(recorded) if recorded else Path.cwd()
        )
        self.output_dir = root
        self.python = str(python) if python else sys.executable
        self.environment = {str(k): str(v) for k, v in dict(environment or {}).items()}
        self.runner: CommandRunner = runner or _default_command_runner
        self.item_ids: tuple[str, ...] = ()
        self.material_by_item: dict[str, list[str]] = {}
        self.sources_by_item: dict[str, str] = {}
        self.summary: dict[str, Any] = {
            "payload": KAGGLE_PAYLOAD_MODULE,
            "kind": None,
            "spec_id": str(self.context.get("spec_id") or ""),
            "status": "refused",
            "succeeded": False,
            "refused_by": None,
            "refusal_reason": None,
            "failure_reason": None,
            "experiment_id": None,
            "corpus": None,
            "project": None,
            "commands": [],
            "training": None,
            "contamination": {},
            "notes": [],
        }

    # ------------------------------------------------------------------
    # stages
    # ------------------------------------------------------------------

    def execute(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._resolve_declaration()
        self._load_inputs()
        self._admit_material()
        self._write_corpus()
        self._compose_project()
        self._run_production()

    def refuse(self, refusal: ComputeBackendRefusal) -> None:
        self.summary["status"] = "refused"
        self.summary["succeeded"] = False
        self.summary["refused_by"] = refusal.code
        self.summary["refusal_reason"] = refusal.reason

    def finish(self) -> None:
        self._write_resume_marker()
        _write_json(self.output_dir / RUN_SUMMARY_NAME, self.summary)

    # ------------------------------------------------------------------

    def _resolve_declaration(self) -> None:
        declaration = self.context.get("declared_payload")
        if not isinstance(declaration, Mapping):
            raise ComputeBackendRefusal(
                KAGGLE_PAYLOAD_DECLARATION_MISSING,
                "the attempt context carries no declared_payload; the kernel "
                "runs what the campaign declared, and this module will not "
                "invent a payload from nothing",
            )
        kind = str(declaration.get("kind") or "")
        self.summary["kind"] = kind
        if kind != CORPUS_TRAINING_KIND:
            raise ComputeBackendRefusal(
                KAGGLE_PAYLOAD_KIND_UNKNOWN,
                f"declared payload kind {kind!r} is not one this module "
                f"implements ({CORPUS_TRAINING_KIND!r}); dispatching an "
                "unimplemented payload would train on a guess",
            )
        recipe_document = declaration.get("recipe")
        if not isinstance(recipe_document, Mapping):
            raise ComputeBackendRefusal(
                KAGGLE_PAYLOAD_SCHEMA,
                "the declared payload carries no recipe mapping; the project "
                "cannot be composed without the recipe the campaign chose",
            )
        try:
            self.recipe = TrainingRecipe(**dict(recipe_document))
        except (TypeError, ValueError) as exc:
            raise ComputeBackendRefusal(
                KAGGLE_PAYLOAD_SCHEMA, f"the declared recipe is invalid: {exc}"
            ) from exc
        item_ids = declaration.get("item_ids")
        if isinstance(item_ids, str) or not isinstance(item_ids, (list, tuple)):
            raise ComputeBackendRefusal(
                KAGGLE_PAYLOAD_SCHEMA,
                f"declared item_ids must be a sequence of ids, got {item_ids!r}",
            )
        declared_items = tuple(str(item) for item in item_ids)
        if not declared_items or not all(item.strip() for item in declared_items):
            raise ComputeBackendRefusal(
                KAGGLE_PAYLOAD_ITEMS_EMPTY,
                "the declared payload names no curriculum items; the corpus "
                "is exactly those items' material, and there is none",
            )
        self.item_ids = declared_items
        budget = declaration.get("project_gpu_hour_budget")
        if (
            isinstance(budget, bool)
            or not isinstance(budget, (int, float))
            or not math.isfinite(float(budget))
            or float(budget) < 0.0
        ):
            raise ComputeBackendRefusal(
                KAGGLE_PAYLOAD_SCHEMA,
                "the declared project_gpu_hour_budget must be finite and "
                f"non-negative, got {budget!r}",
            )
        self.project_gpu_hour_budget = float(budget)
        attempt_id = str(self.context.get("attempt_id") or "").strip()
        if not attempt_id:
            raise ComputeBackendRefusal(
                KAGGLE_PAYLOAD_SCHEMA,
                "the attempt context carries no attempt_id; the composed "
                "experiment id is derived from it, so it cannot be empty",
            )
        self.attempt_id = attempt_id

    def _load_inputs(self) -> None:
        locations = self.context.get("input_locations")
        if not isinstance(locations, Mapping):
            raise ComputeBackendRefusal(
                KAGGLE_PAYLOAD_INPUTS_MISSING,
                "the attempt context carries no input_locations; the resolved "
                "mount paths are what the kernel verified, and reading "
                "anywhere else would read bytes nothing checked",
            )
        self.locations = {str(name): str(path) for name, path in dict(locations).items()}
        missing = [
            name
            for name in TRAINING_INPUTS
            if not self.locations.get(name, "").strip()
        ]
        if missing:
            raise ComputeBackendRefusal(
                KAGGLE_PAYLOAD_INPUTS_MISSING,
                f"the attempt context resolves no path for {missing}; the "
                "training path reads exactly the declared inputs",
            )
        self.template_document = self._read_required("project_template_path")
        self.material_document = self._read_required("training_material_path")
        self.registry_document = self._read_required("data_registry_path")
        self.contamination_document = (
            self._read_required("contamination_manifest_path")
            if self.locations.get("contamination_manifest_path", "").strip()
            else None
        )

    def _read_required(self, name: str) -> Mapping[str, Any]:
        path = Path(self.locations[name])
        if not path.is_file():
            raise ComputeBackendRefusal(
                KAGGLE_PAYLOAD_INPUT_UNREADABLE,
                f"declared input {name!r} at {path} is not a readable file",
            )
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ComputeBackendRefusal(
                KAGGLE_PAYLOAD_INPUT_UNREADABLE,
                f"declared input {name!r} at {path} is not readable JSON: {exc}",
            ) from exc
        if not isinstance(document, Mapping):
            raise ComputeBackendRefusal(
                KAGGLE_PAYLOAD_INPUT_UNREADABLE,
                f"declared input {name!r} at {path} is not a JSON object",
            )
        return document

    def _admit_material(self) -> None:
        material = self.material_document.get("material")
        sources = self.material_document.get("sources")
        if not isinstance(material, Mapping) or not isinstance(sources, Mapping):
            raise ComputeBackendRefusal(
                KAGGLE_PAYLOAD_MATERIAL_MISSING,
                "the declared training material must carry 'sources' "
                "(item -> source id) and 'material' (item -> text lines)",
            )
        for item, rows in material.items():
            if isinstance(rows, str) or not isinstance(rows, (list, tuple)):
                raise ComputeBackendRefusal(
                    KAGGLE_PAYLOAD_MATERIAL_MISSING,
                    f"declared material for item {item!r} is not a sequence of lines",
                )
        self.material_by_item = {
            str(item): [str(row) for row in rows] for item, rows in material.items()
        }
        self.sources_by_item = {
            str(item): str(source) for item, source in sources.items()
        }

        registry = self._load_registry()
        for item in self.item_ids:
            rows = self.material_by_item.get(item)
            if rows is None:
                raise ComputeBackendRefusal(
                    KAGGLE_PAYLOAD_MATERIAL_MISSING,
                    f"the declared payload trains on item {item!r}, but the "
                    "declared material has no entry for it; a missing item is "
                    "a refusal, not a smaller corpus",
                )
            if not any(row.strip() for row in rows):
                raise ComputeBackendRefusal(
                    KAGGLE_PAYLOAD_MATERIAL_MISSING,
                    f"declared material for item {item!r} is empty; item with "
                    "no material is not trainable",
                )
            source_id = self.sources_by_item.get(item, "")
            if not source_id:
                raise ComputeBackendRefusal(
                    KAGGLE_PAYLOAD_SOURCE_UNREGISTERED,
                    f"curriculum item {item!r} names no registered data source; "
                    "material with no provenance is not trainable",
                )
            source = registry.get(source_id)
            if source is None:
                raise ComputeBackendRefusal(
                    KAGGLE_PAYLOAD_SOURCE_UNREGISTERED,
                    f"data source {source_id!r} is not registered",
                )
            if not source.trainable:
                raise ComputeBackendRefusal(
                    KAGGLE_PAYLOAD_SOURCE_NOT_TRAINABLE,
                    f"data source {source_id!r} is not trainable "
                    f"(trust_class={source.trust_class}, "
                    f"inclusion_decision={source.inclusion_decision}, "
                    f"contamination={source.contamination_relationship})",
                )
        self._read_declared_contamination()

    def _load_registry(self) -> DataRegistry:
        entries = self.registry_document.get("sources")
        if not isinstance(entries, list):
            raise ComputeBackendRefusal(
                KAGGLE_PAYLOAD_SOURCE_UNREGISTERED,
                "the declared data registry must be an object with a 'sources' list",
            )
        registry = DataRegistry()
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise ComputeBackendRefusal(
                    KAGGLE_PAYLOAD_SOURCE_UNREGISTERED,
                    "the declared data registry holds a non-object entry",
                )
            try:
                source = DataSource(**dict(entry))
            except (TypeError, ValueError) as exc:
                raise ComputeBackendRefusal(
                    KAGGLE_PAYLOAD_SOURCE_UNREGISTERED,
                    f"the declared data registry entry is invalid: {exc}",
                ) from exc
            if source.inclusion_decision != "included":
                raise ComputeBackendRefusal(
                    KAGGLE_PAYLOAD_SOURCE_UNREGISTERED,
                    f"data source {source.source_id!r} is declared "
                    f"{source.inclusion_decision!r}; only included sources may train",
                )
            registry.register(
                admit(
                    source,
                    decision="included",
                    reason="declared included in the campaign's data registry",
                )
            )
        return registry

    def _read_declared_contamination(self) -> None:
        """The declared manifest's verdicts, read as verdicts, never re-derived.

        The protected texts themselves are not among the declared inputs, so
        the kernel cannot re-run the fingerprint check; what it can do is
        refuse to train under a manifest that does not call the source clean.
        """
        document = self.contamination_document
        if document is None:
            self.summary["contamination"] = {
                "checked": False,
                "reason": "the attempt declares no contamination manifest",
            }
            return
        section = document.get("training_sources")
        statuses: dict[str, str] = {}
        for item in self.item_ids:
            source_id = self.sources_by_item.get(item, "")
            if not source_id or source_id in statuses:
                continue
            entry = section.get(source_id) if isinstance(section, Mapping) else None
            status = str(entry.get("status") or "") if isinstance(entry, Mapping) else ""
            statuses[source_id] = status or "not-declared"
        for source_id, status in statuses.items():
            if status not in TRAINABLE_CONTAMINATION_STATUSES and status != "not-declared":
                raise ComputeBackendRefusal(
                    KAGGLE_PAYLOAD_CONTAMINATION_DECLARED,
                    f"the declared contamination manifest marks source "
                    f"{source_id!r} {status!r}; training on a source the "
                    "declaration does not call clean is not this payload's "
                    "decision to make",
                )
        self.summary["contamination"] = {
            "checked": True,
            "training_sources": statuses,
        }

    def _write_corpus(self) -> None:
        lines: list[str] = []
        for item in sorted(self.item_ids):
            for row in self.material_by_item[item]:
                normalized = " ".join(str(row).split())
                if normalized:
                    lines.append(normalized)
        payload = "\n".join(lines) + "\n"
        path = self.output_dir / CORPUS_NAME
        path.write_bytes(payload.encode("utf-8"))
        digest, _size = sha256_file(path)
        self.corpus_path = path
        self.summary["corpus"] = {
            "path": str(path),
            "sha256": digest,
            "examples": len(lines),
            "item_ids": sorted(self.item_ids),
            "source_ids": sorted(
                {self.sources_by_item[item] for item in self.item_ids}
            ),
        }

    def _compose_project(self) -> None:
        from chowder.graph import deep_merge_config

        template: dict[str, Any] = json.loads(json.dumps(self.template_document))
        config = template.get("config")
        if not isinstance(config, Mapping):
            raise ComputeBackendRefusal(
                KAGGLE_PAYLOAD_TEMPLATE_CONTRACT,
                "the declared project template has no config section to merge into",
            )
        backend_type = ""
        backend_section = config.get("backend")
        if isinstance(backend_section, Mapping):
            declared = backend_section.get("type")
            if isinstance(declared, str) and declared.strip():
                backend_type = declared.strip()
        if not backend_type:
            raise ComputeBackendRefusal(
                KAGGLE_PAYLOAD_TEMPLATE_CONTRACT,
                "the declared project template's config.backend declares no "
                "`type`; the recipe knobs have no namespace to target, so the "
                "merge would be a guess",
            )
        try:
            patch = self.recipe.to_config_patch(backend_type=backend_type)
        except ValueError as exc:
            raise ComputeBackendRefusal(
                KAGGLE_PAYLOAD_TEMPLATE_CONTRACT, str(exc)
            ) from exc
        merged_config = deep_merge_config(config, patch)
        if "search" in merged_config or "search" in template:
            raise ComputeBackendRefusal(
                KAGGLE_PAYLOAD_TEMPLATE_CONTRACT,
                "a `search` section is present; run_project has no search "
                "config, and bounded candidate search is declared on the "
                "campaign rather than in the project",
            )
        template["config"] = merged_config

        goal = template.get("goal")
        if not isinstance(goal, Mapping) or "gpu_hour_budget" not in goal:
            raise ComputeBackendRefusal(
                KAGGLE_PAYLOAD_TEMPLATE_CONTRACT,
                "the declared project template has no goal.gpu_hour_budget; "
                "the payload cannot enforce a ceiling the run does not have",
            )
        declared_budget = goal["gpu_hour_budget"]
        if isinstance(declared_budget, bool) or not isinstance(
            declared_budget, (int, float)
        ):
            raise ComputeBackendRefusal(
                KAGGLE_PAYLOAD_TEMPLATE_CONTRACT,
                "goal.gpu_hour_budget must be a number",
            )
        if float(declared_budget) > self.project_gpu_hour_budget:
            raise ComputeBackendRefusal(
                KAGGLE_PAYLOAD_PROJECT_BUDGET,
                f"the project declares {float(declared_budget):.6f} GPU-h, "
                f"looser than the attempt's declared project ceiling "
                f"{self.project_gpu_hour_budget:.6f}; the payload will not "
                "silently lower the project's",
            )

        body = json.dumps(template)
        if CORPUS_TOKEN not in body:
            raise ComputeBackendRefusal(
                KAGGLE_PAYLOAD_TEMPLATE_CONTRACT,
                f"the declared project template never names {CORPUS_TOKEN}; "
                "the payload cannot know which knob takes the materialized "
                "corpus, and guessing would train on whatever the template "
                "already pointed at",
            )
        experiment = template.get("experiment")
        if not isinstance(experiment, Mapping) or not experiment.get("experiment_id"):
            raise ComputeBackendRefusal(
                KAGGLE_PAYLOAD_TEMPLATE_CONTRACT,
                "the declared project template has no experiment.experiment_id "
                "to derive a duplicate-safe attempt identity from",
            )
        derived = f"{experiment['experiment_id']}-a{self.attempt_id.split('-')[-1]}"
        patched_experiment = dict(experiment)
        patched_experiment["experiment_id"] = derived
        template["experiment"] = patched_experiment

        replacements = {
            CORPUS_TOKEN: str(self.corpus_path),
            ATTEMPT_TOKEN: str(self.output_dir),
        }
        composed, _used = _substitute(template, replacements)
        if CORPUS_TOKEN in json.dumps(composed):
            raise ComputeBackendRefusal(
                KAGGLE_PAYLOAD_TEMPLATE_CONTRACT,
                f"the project template still contains {CORPUS_TOKEN} after "
                "substitution; the materialized corpus has no declared home",
            )
        composed["work_dir"] = str(self.output_dir / WORK_DIR_NAME)
        composed["registry_path"] = str(self.output_dir / REGISTRY_NAME)
        project_path = self.output_dir / PROJECT_NAME
        _write_json(project_path, composed)
        self.project_path = project_path
        self.summary["experiment_id"] = derived
        self.summary["project"] = {
            "path": str(project_path),
            "sha256": sha256_file(project_path)[0],
        }

    def _run_production(self) -> None:
        validate = self._run_cli("project-validate", "validate")
        if validate["returncode"] != 0:
            raise ComputeBackendRefusal(
                KAGGLE_PAYLOAD_VALIDATE_FAILED,
                "the project was refused before compute (exit "
                f"{validate['returncode']}): {validate['tail'] or 'no output'}",
            )
        train = self._run_cli("train", "train")
        summary_document = last_json_object(train["stdout"])
        if summary_document is None:
            self.summary["status"] = "failed"
            self.summary["failure_reason"] = (
                "the training process printed no parsable summary "
                f"(exit {train['returncode']})"
            )
            return
        succeeded = summary_document.get("succeeded") is True
        artifact_ref = str(summary_document.get("artifact_ref") or "")
        artifact_path = _relative_artifact(self.output_dir, artifact_ref)
        self.summary["training"] = {
            "experiment_id": summary_document.get("experiment_id"),
            "succeeded": succeeded,
            "promoted_experiment_id": summary_document.get("promoted_experiment_id"),
            "artifact_ref": artifact_ref or None,
            "artifact_path": artifact_path,
            "artifact_exists": bool(
                artifact_path and (self.output_dir / artifact_path).exists()
            ),
            "metrics": dict(summary_document.get("metrics") or {}),
            "gpu_hours": summary_document.get("gpu_hours"),
            "error": summary_document.get("error"),
            "exit_code": int(train["returncode"]),
        }
        if succeeded:
            self.summary["status"] = "succeeded"
            self.summary["succeeded"] = True
        else:
            self.summary["status"] = "failed"
            self.summary["failure_reason"] = str(
                summary_document.get("error")
                or f"the training process reported failure (exit {train['returncode']})"
            )

    def _run_cli(self, verb: str, tag: str) -> dict[str, Any]:
        command = [self.python, "-m", CLI_MODULE, verb, str(self.project_path)]
        completed = self.runner(command, self.output_dir, dict(self.environment))
        record: dict[str, Any] = {
            "verb": verb,
            "command": list(command),
            "returncode": int(completed.returncode),
        }
        for stream in ("stdout", "stderr"):
            text = str(getattr(completed, stream) or "")
            name = f"{tag}.{stream}.txt"
            (self.output_dir / name).write_text(text, encoding="utf-8")
            record[f"{stream}_ref"] = name
        record["tail"] = (str(completed.stderr or "") or str(completed.stdout or "")).strip()[-2000:]
        record["stdout"] = str(completed.stdout or "")
        self.summary["commands"].append(
            {key: value for key, value in record.items() if key != "stdout"}
        )
        return record

    def _write_resume_marker(self) -> None:
        declared = str(self.context.get("resume_from") or "").strip()
        if not declared:
            return
        # Production's own trainer refuses a checkpoint whose state is not
        # resumable, so a successful run under a declared resume is the proof
        # that the continuation took; anything else is reported as what it is.
        state = "resumed" if self.summary["status"] == "succeeded" else "not-a-resume"
        _write_json(
            self.output_dir / RESUME_STATE_NAME,
            {"resume_state": state, "resume_from": declared},
        )


def _as_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return int(value)


def _artifact_under(root: Path, relative: Any) -> Path | None:
    """The returned artifact path a run summary names, if it is inside ``root``."""
    text = str(relative or "").strip().replace("\\", "/")
    if not text:
        return None
    candidate = Path(text)
    if candidate.is_absolute():
        try:
            candidate = candidate.relative_to(root)
        except ValueError:
            return None
    parts = candidate.parts
    if not parts or ".." in parts or candidate.is_absolute():
        return None
    target = root / candidate
    if not target.exists():
        return None
    return target


def _relative_artifact(output_dir: Path, artifact_ref: str) -> str | None:
    """The production artifact as a path relative to the output directory."""
    text = str(artifact_ref or "").strip()
    if not text:
        return None
    candidate = Path(text)
    if candidate.is_absolute():
        try:
            candidate = candidate.relative_to(output_dir)
        except ValueError:
            return None
    parts = candidate.parts
    if not parts or ".." in parts:
        return None
    return candidate.as_posix()


def _default_command_runner(
    command: Sequence[str], working_dir: Path, environment: Mapping[str, str]
) -> "subprocess.CompletedProcess[str]":
    env = None
    if environment:
        env = {**os.environ, **{str(key): str(value) for key, value in environment.items()}}
    return subprocess.run(
        list(command), cwd=str(working_dir), capture_output=True, text=True, env=env
    )


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(document), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """The kernel-side command's entry point.

    Reads the attempt context ``run_kernel_job`` wrote beside it (by default,
    ``attempt-context.json`` in the working directory), runs the payload, and
    prints the summary. Exit status is 0 only for a production success, so the
    kernel record can never claim one on the payload's behalf.
    """
    parser = argparse.ArgumentParser(prog=KAGGLE_PAYLOAD_MODULE)
    parser.add_argument("--context", default=ATTEMPT_CONTEXT_NAME)
    arguments = parser.parse_args(list(argv) if argv is not None else None)
    context_path = Path(arguments.context)
    try:
        context = json.loads(context_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(
            f"{KAGGLE_PAYLOAD_CONTEXT_UNREADABLE}: cannot read {context_path}: {exc}",
            file=sys.stderr,
        )
        return 2
    if not isinstance(context, Mapping):
        print(
            f"{KAGGLE_PAYLOAD_CONTEXT_UNREADABLE}: {context_path} is not a JSON object",
            file=sys.stderr,
        )
        return 2
    try:
        document = run_payload(context)
    except OSError as exc:
        print(
            f"{KAGGLE_PAYLOAD_OUTPUT_UNWRITABLE}: {exc}",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(document, indent=2, sort_keys=True))
    if document.get("status") != "succeeded":
        code = document.get("refused_by") or "KAGGLE_PAYLOAD_FAILED"
        reason = document.get("refusal_reason") or document.get("failure_reason") or "failed"
        print(f"{code}: {reason}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
