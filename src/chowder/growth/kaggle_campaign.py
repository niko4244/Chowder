"""Campaign wiring: a prepared campaign's declared inputs -> one Kaggle attempt.

The two halves the growth loop needs in order to spend Kaggle quota through
the declared-input contract:

* :func:`build_attempt_request` turns a prepared campaign
  (:class:`chowder.growth.campaign_prepare.PreparedCampaign`) plus one recipe
  into an :class:`~chowder.growth.compute_backend.AttemptRequest`: every
  declared prepared input is bound by digest, the payload carries the declared
  command and the kernel path(s) each input may be mounted at, the projection
  is the recipe's own, and the ceilings are the growth envelope's -- the same
  envelope the local executor admits against.
* :class:`KaggleTrainingFn` is a ``TrainingFn`` (callable, with ``admit``)
  that dispatches one recipe per attempt through a
  :class:`~chowder.growth.compute_backend.ComputeBackend`, reserves a fresh
  never-reused attempt directory, writes ``training-evidence.json`` beside the
  result, and returns the outcome's evidence merged into the growth
  vocabulary the cycle and the campaign runner already read.

What this module deliberately does not do: it does not invent the payload
command (the campaign declares it), does not materialize the curriculum corpus
(or run the production trainer) here -- that is the declared payload's job
inside the kernel, from the declared inputs
(:mod:`chowder.growth.kaggle_payload` is that payload, ported from the local
executor) -- and it does not evaluate or promote. A campaign declares its
payload by passing any object exposing ``command()`` and
``payload(recipe, items)`` as ``declared_payload``; the declaration travels in
the attempt request and is bound per attempt. The wiring is complete when a
campaign passes a ``KaggleTrainingFn`` to
``campaign_runner.run_campaign(train_fn=...)``; every refusal before that
point is named rather than defaulted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .campaign_prepare import (
    CONTAMINATION_MANIFEST_FIELD,
    PREPARED_INPUT_FIELDS,
    PreparedCampaign,
)
from .compute_backend import (
    OUTCOME_FAILED,
    OUTCOME_REFUSED,
    AttemptOutcome,
    AttemptRequest,
    ComputeBackend,
    ComputeBackendRefusal,
    SourceBinding,
    bind_declared_inputs,
)
from .compute_cost import ComputeCost
from .kaggle_payload import merge_run_summary
from .recipe_planner import TrainingRecipe
from .training_binding import GrowthEnvelope, check_growth_envelope

__all__ = [
    "KAGGLE_CAMPAIGN_BINDING",
    "KAGGLE_CAMPAIGN_EVIDENCE_FILE",
    "KAGGLE_CAMPAIGN_SCHEMA",
    "KAGGLE_CAMPAIGN_INPUTS_INCOMPLETE",
    "KAGGLE_CAMPAIGN_INPUTS_UNDECLARED",
    "KAGGLE_CAMPAIGN_COMMAND_MISSING",
    "KAGGLE_CAMPAIGN_MOUNTS_MISSING",
    "KAGGLE_CAMPAIGN_INPUT_PATHS_INCOMPLETE",
    "DeclaredPayload",
    "build_attempt_request",
    "KaggleTrainingFn",
]

KAGGLE_CAMPAIGN_BINDING = "chowder.growth.kaggle_campaign"
#: The evidence record every remote attempt leaves in its own directory.
KAGGLE_CAMPAIGN_EVIDENCE_FILE = "training-evidence.json"

KAGGLE_CAMPAIGN_SCHEMA = "KAGGLE_CAMPAIGN_SCHEMA"
KAGGLE_CAMPAIGN_INPUTS_INCOMPLETE = "KAGGLE_CAMPAIGN_INPUTS_INCOMPLETE"
KAGGLE_CAMPAIGN_INPUTS_UNDECLARED = "KAGGLE_CAMPAIGN_INPUTS_UNDECLARED"
KAGGLE_CAMPAIGN_COMMAND_MISSING = "KAGGLE_CAMPAIGN_COMMAND_MISSING"
KAGGLE_CAMPAIGN_MOUNTS_MISSING = "KAGGLE_CAMPAIGN_MOUNTS_MISSING"
KAGGLE_CAMPAIGN_INPUT_PATHS_INCOMPLETE = "KAGGLE_CAMPAIGN_INPUT_PATHS_INCOMPLETE"

#: The declared prepared inputs, in the order a reader should walk them.
DECLARED_PREPARED_FIELDS: tuple[str, ...] = (
    *PREPARED_INPUT_FIELDS,
    CONTAMINATION_MANIFEST_FIELD,
)


class DeclaredPayload(Protocol):
    """What runs inside the kernel, declared rather than defaulted.

    The contract :class:`KaggleTrainingFn` accepts: ``command()`` is the argv
    every attempt's kernel runs, and ``payload(recipe, items)`` binds one
    attempt's recipe and curriculum items into the declaration the attempt
    request carries. :class:`chowder.growth.kaggle_payload.CorpusTraining`
    materializes the declared corpus and runs the production entry points;
    this module never inspects the declaration beyond carrying it, so a
    campaign can declare a payload this package does not implement without
    changing the wiring.
    """

    kind: str

    def command(self) -> Sequence[str]:
        """The argv a kernel runs for this payload."""

    def payload(
        self, recipe: TrainingRecipe, items: Sequence[Any]
    ) -> Mapping[str, Any]:
        """One attempt's declaration, carried in the request payload."""


def build_attempt_request(
    prepared: PreparedCampaign,
    *,
    source: SourceBinding,
    recipe: TrainingRecipe,
    envelope: GrowthEnvelope,
    entry_point: str,
    command: Sequence[str],
    input_paths: Mapping[str, str | Sequence[str]],
    mounts: Sequence[str],
    timeout_seconds: float,
    model_commit: str | None = None,
    pip_extras: Sequence[str] = ("train",),
    projection_tolerance: float = 0.25,
    resume_from: str | None = None,
    declared_payload: Mapping[str, Any] | None = None,
) -> AttemptRequest:
    """One declared attempt, built from the campaign's prepared inputs.

    ``command`` is what the kernel runs after installing the pinned commit and
    verifying every declared input -- the wiring never invents one.
    ``input_paths`` maps every declared input name to the kernel path(s) it may
    appear at (a Kaggle dataset can mount under its bare slug or, on a name
    conflict, under its owner-qualified path); names that are missing or were
    never declared refuse, because an input the kernel cannot reach is an
    attempt that cannot bind its evidence. ``declared_payload`` is the
    campaign's own kernel-side declaration (what the payload command does with
    those inputs); it travels untouched in the request payload, and a
    declaration that cannot be serialized refuses before a kernel exists.
    """
    declared = declared_input_paths(prepared)
    inputs = bind_declared_inputs(declared)
    names = [entry.name for entry in inputs]
    normalized_paths = normalized_input_paths(names, input_paths)
    declared_command = validate_command(command)
    declared_mounts = validate_mounts(mounts)
    declared_extras = _validate_extras(pip_extras)
    declared_payload_document = _validate_declared_payload(declared_payload)
    payload: dict[str, Any] = {
        "command": declared_command,
        "input_paths": {
            name: list(paths) for name, paths in normalized_paths.items()
        },
        "pip_extras": declared_extras,
        **({"model_commit": str(model_commit)} if model_commit else {}),
    }
    if declared_payload_document is not None:
        payload["declared_payload"] = declared_payload_document
    return AttemptRequest(
        source=source,
        entry_point=entry_point,
        inputs=inputs,
        payload=payload,
        projected_cost=ComputeCost.measured(
            device_gpu_hours=float(recipe.projected_device_gpu_hours),
            wall_gpu_hours=float(recipe.projected_wall_gpu_hours),
            source="planner-projection",
        ),
        device_ceiling=envelope.device_gpu_hours_ceiling,
        wall_ceiling=envelope.wall_gpu_hours_ceiling,
        project_budget_wall_gpu_hours=envelope.project_gpu_hour_budget,
        projection_tolerance=projection_tolerance,
        mounts=declared_mounts,
        resume_from=resume_from,
        timeout_seconds=timeout_seconds,
    )


def declared_input_paths(prepared: PreparedCampaign) -> dict[str, str]:
    """The prepared campaign's declared input set, or a named refusal.

    The declaration is exactly the fields :func:`prepare_campaign` writes: a
    remote attempt that silently drops one of them would ship fewer bytes than
    the campaign declared, and one that carries an unknown field would ship
    bytes no preparation is accountable for.
    """
    if not isinstance(prepared, PreparedCampaign):
        raise ComputeBackendRefusal(
            KAGGLE_CAMPAIGN_SCHEMA,
            f"prepared must be a PreparedCampaign, got {type(prepared).__name__}",
        )
    inputs = {str(name): str(path) for name, path in dict(prepared.inputs).items()}
    missing = [
        field_name
        for field_name in DECLARED_PREPARED_FIELDS
        if not inputs.get(field_name, "").strip()
    ]
    if missing:
        raise ComputeBackendRefusal(
            KAGGLE_CAMPAIGN_INPUTS_INCOMPLETE,
            f"the prepared campaign declares no {missing}; a remote attempt "
            "ships exactly the inputs preparation wrote, so a missing one is "
            "a refusal, not a smaller attempt",
        )
    undeclared = sorted(set(inputs) - set(DECLARED_PREPARED_FIELDS))
    if undeclared:
        raise ComputeBackendRefusal(
            KAGGLE_CAMPAIGN_INPUTS_UNDECLARED,
            f"the prepared campaign carries {undeclared}, which this wiring "
            "does not declare; adding it to the declared set is a decision, "
            "and silently dropping it from the attempt is not",
        )
    return {field_name: inputs[field_name] for field_name in DECLARED_PREPARED_FIELDS}


def validate_command(command: Sequence[str]) -> list[str]:
    """The declared payload command, or a refusal -- never a default."""
    if isinstance(command, str) or not isinstance(command, (list, tuple)):
        raise ComputeBackendRefusal(
            KAGGLE_CAMPAIGN_COMMAND_MISSING,
            f"command must be a sequence of program and arguments, got {command!r}",
        )
    parts = [str(part) for part in command]
    if not parts or not all(part.strip() for part in parts):
        raise ComputeBackendRefusal(
            KAGGLE_CAMPAIGN_COMMAND_MISSING,
            "a remote attempt needs the command the campaign declares; "
            "nothing here may invent what the kernel runs",
        )
    return parts


def validate_mounts(mounts: Sequence[str]) -> tuple[str, ...]:
    """The declared mounts, or a refusal -- the inputs live on a mount."""
    if isinstance(mounts, str) or not isinstance(mounts, (list, tuple)):
        raise ComputeBackendRefusal(
            KAGGLE_CAMPAIGN_SCHEMA,
            f"mounts must be a sequence of dataset references, got {mounts!r}",
        )
    declared = tuple(str(mount) for mount in mounts)
    if not declared or not all(mount.strip() for mount in declared):
        raise ComputeBackendRefusal(
            KAGGLE_CAMPAIGN_MOUNTS_MISSING,
            "a remote attempt declares the dataset mount its inputs arrive on; "
            "with no mount the declared bytes have no surface to be verified at",
        )
    return declared


def normalized_input_paths(
    names: Sequence[str], input_paths: Mapping[str, str | Sequence[str]]
) -> dict[str, tuple[str, ...]]:
    """The declared kernel path(s) for every declared input, or a refusal."""
    if not isinstance(input_paths, Mapping):
        raise ComputeBackendRefusal(
            KAGGLE_CAMPAIGN_SCHEMA,
            f"input_paths must be a mapping of declared name to kernel path, "
            f"got {input_paths!r}",
        )
    provided = {str(name): value for name, value in dict(input_paths).items()}
    expected = list(names)
    missing = [name for name in expected if name not in provided]
    undeclared = sorted(set(provided) - set(expected))
    if missing or undeclared:
        raise ComputeBackendRefusal(
            KAGGLE_CAMPAIGN_INPUT_PATHS_INCOMPLETE,
            "the declared kernel paths do not cover the declared inputs "
            f"exactly: no path for {missing}, paths for undeclared {undeclared}",
        )
    normalized: dict[str, tuple[str, ...]] = {}
    for name in expected:
        value = provided[name]
        if isinstance(value, str):
            candidates = (value,)
        elif isinstance(value, (list, tuple)):
            candidates = tuple(str(item) for item in value)
        else:
            raise ComputeBackendRefusal(
                KAGGLE_CAMPAIGN_SCHEMA,
                f"declared input {name!r} has kernel path {value!r}; expected a "
                "path or a sequence of candidate paths",
            )
        if not candidates or not all(candidate.strip() for candidate in candidates):
            raise ComputeBackendRefusal(
                KAGGLE_CAMPAIGN_SCHEMA,
                f"declared input {name!r} has an empty kernel path "
                f"({value!r}); an input with nowhere to be read from is not a "
                "declared input",
            )
        absolute = [candidate for candidate in candidates if candidate.startswith("/")]
        if len(absolute) != len(candidates):
            raise ComputeBackendRefusal(
                KAGGLE_CAMPAIGN_SCHEMA,
                f"declared input {name!r} has a relative kernel path "
                f"({value!r}); kernel paths are absolute POSIX paths",
            )
        normalized[name] = candidates
    return normalized


def _validate_extras(pip_extras: Sequence[str]) -> list[str]:
    if isinstance(pip_extras, str) or not isinstance(pip_extras, (list, tuple)):
        raise ComputeBackendRefusal(
            KAGGLE_CAMPAIGN_SCHEMA,
            f"pip_extras must be a sequence of extra names, got {pip_extras!r}",
        )
    extras = [str(extra) for extra in pip_extras]
    if not extras or not all(extra.strip() for extra in extras):
        raise ComputeBackendRefusal(
            KAGGLE_CAMPAIGN_SCHEMA,
            "pip_extras must name at least one install extra; the kernel "
            "installs the pinned commit and this module will not silently "
            "change what that install includes",
        )
    return extras


def _validate_declared_payload(
    declared_payload: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """The declared kernel payload, or a named refusal -- never dropped.

    A declaration the kernel cannot read would leave the payload command with
    nothing to execute; dropping it silently would ship an attempt whose
    behavior the campaign did not declare.
    """
    if declared_payload is None:
        return None
    if not isinstance(declared_payload, Mapping):
        raise ComputeBackendRefusal(
            KAGGLE_CAMPAIGN_SCHEMA,
            f"declared_payload must be a mapping, got {declared_payload!r}",
        )
    document = dict(declared_payload)
    if not document:
        raise ComputeBackendRefusal(
            KAGGLE_CAMPAIGN_SCHEMA,
            "declared_payload is empty; a payload declaration that declares "
            "nothing would leave the kernel with no instruction",
        )
    try:
        json.dumps(document, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ComputeBackendRefusal(
            KAGGLE_CAMPAIGN_SCHEMA,
            f"declared_payload cannot be serialized: {exc}",
        ) from exc
    return document


class KaggleTrainingFn:
    """A ``TrainingFn`` that executes one recipe as a Kaggle kernel attempt.

    Callable, with ``admit``, so the campaign runner can use it exactly where
    it uses the local ``SubprocessTrainingFn``::

        run_campaign(manifest, train_fn=KaggleTrainingFn(...))

    Every attempt reserves the smallest unused ``attempt-NN`` directory under
    ``attempts_root`` (never reused, so a refused or failed attempt keeps its
    evidence), dispatches one :class:`AttemptRequest` through the injected
    backend, and writes the evidence mapping to that directory's
    ``training-evidence.json``.

    The kernel payload is declared, not defaulted: pass ``declared_payload``
    (an object exposing ``command()`` and ``payload(recipe, items)``, such as
    :class:`chowder.growth.kaggle_payload.CorpusTraining`) and this binding
    derives the argv from it and carries its per-attempt declaration into the
    request; or pass ``command`` for a hand-declared argv. Passing both
    refuses, because two declarations could disagree about what runs.
    """

    def __init__(
        self,
        backend: ComputeBackend,
        *,
        prepared: PreparedCampaign,
        envelope: GrowthEnvelope,
        repository: str,
        commit_sha: str,
        chowder_version: str,
        entry_point: str,
        command: Sequence[str] | None = None,
        input_paths: Mapping[str, str | Sequence[str]],
        mounts: Sequence[str],
        attempts_root: str | Path,
        timeout_seconds: float,
        declared_payload: DeclaredPayload | None = None,
        model_commit: str | None = None,
        pip_extras: Sequence[str] = ("train",),
        projection_tolerance: float = 0.25,
    ) -> None:
        if declared_payload is not None and command is not None:
            raise ComputeBackendRefusal(
                KAGGLE_CAMPAIGN_SCHEMA,
                "declare the kernel payload once: declared_payload carries its "
                "own command, and passing command beside it would let two "
                "declarations disagree about what runs",
            )
        if declared_payload is not None:
            command_fn = getattr(declared_payload, "command", None)
            payload_fn = getattr(declared_payload, "payload", None)
            if not callable(command_fn) or not callable(payload_fn):
                raise ComputeBackendRefusal(
                    KAGGLE_CAMPAIGN_SCHEMA,
                    "a declared payload must expose command() and "
                    "payload(recipe, items); got "
                    f"{type(declared_payload).__name__}",
                )
            command = command_fn()
        self.declared_payload = declared_payload
        self.backend = backend
        self.prepared = prepared
        self.envelope = envelope
        self.repository = str(repository)
        self.commit_sha = str(commit_sha).strip().lower()
        self.chowder_version = str(chowder_version)
        self.entry_point = str(entry_point).strip()
        if not self.entry_point:
            raise ComputeBackendRefusal(
                KAGGLE_CAMPAIGN_SCHEMA,
                "entry_point must name the attempt's code entry; the evidence "
                "records what ran, and an empty name records nothing",
            )
        self.command = tuple(validate_command(command))
        self.mounts = validate_mounts(mounts)
        self.attempts_root = Path(attempts_root)
        self.timeout_seconds = float(timeout_seconds)
        if not self.timeout_seconds > 0.0:
            raise ComputeBackendRefusal(
                KAGGLE_CAMPAIGN_SCHEMA,
                f"timeout_seconds must be positive, got {timeout_seconds!r}",
            )
        self.model_commit = str(model_commit) if model_commit else None
        self.pip_extras = tuple(_validate_extras(pip_extras))
        self.projection_tolerance = float(projection_tolerance)

        # Refuse a misconfigured binding at construction, before any phase runs
        # the first attempt: the declared inputs must be readable now, the
        # kernel paths must cover them, and the code identity must be a real
        # binding. The per-attempt build re-binds the inputs, so a file that
        # moves later refuses that attempt instead of shipping silently.
        declared = declared_input_paths(prepared)
        bind_declared_inputs(declared)
        self._declared_names = tuple(sorted(declared))
        self.input_paths = normalized_input_paths(self._declared_names, input_paths)
        SourceBinding(
            repository=self.repository,
            commit_sha=self.commit_sha,
            chowder_version=self.chowder_version,
            cycle_id=str(prepared.cycle_id),
            recipe_id="binding-check",
            attempt_id="binding-check",
        )

    # ------------------------------------------------------------------
    # the TrainingFn contract
    # ------------------------------------------------------------------

    def admit(self, recipe: TrainingRecipe) -> tuple[str, str] | None:
        """The growth envelope's admission rule, shared with the local executor."""
        return check_growth_envelope(recipe, self.envelope)

    def __call__(
        self, recipe: TrainingRecipe, items: Sequence[Any]
    ) -> Mapping[str, Any]:
        items = tuple(items)
        attempt, attempt_dir = self._reserve_attempt()
        source = SourceBinding(
            repository=self.repository,
            commit_sha=self.commit_sha,
            chowder_version=self.chowder_version,
            cycle_id=str(self.prepared.cycle_id),
            recipe_id=recipe.recipe_id,
            attempt_id=attempt,
        )
        try:
            declared_payload = self._declared_payload(recipe, items)
            request = build_attempt_request(
                self.prepared,
                source=source,
                recipe=recipe,
                envelope=self.envelope,
                entry_point=self.entry_point,
                command=self.command,
                input_paths=self.input_paths,
                mounts=self.mounts,
                timeout_seconds=self.timeout_seconds,
                model_commit=self.model_commit,
                pip_extras=self.pip_extras,
                projection_tolerance=self.projection_tolerance,
                resume_from=recipe.resume_from_checkpoint,
                declared_payload=declared_payload,
            )
        except ComputeBackendRefusal as refusal:
            evidence = self._refused_evidence(
                refusal, recipe, attempt, attempt_dir, items
            )
            self._write_evidence(attempt_dir, evidence)
            return evidence
        outcome = self.backend.dispatch(request, destination=attempt_dir)
        evidence = self._evidence(outcome, request, recipe, attempt, attempt_dir, items)
        self._write_evidence(attempt_dir, evidence)
        return evidence

    # ------------------------------------------------------------------
    # attempt bookkeeping and evidence
    # ------------------------------------------------------------------

    def _declared_payload(
        self, recipe: TrainingRecipe, items: Sequence[Any]
    ) -> Mapping[str, Any] | None:
        """This attempt's declaration, or ``None`` for a raw command."""
        if self.declared_payload is None:
            return None
        return dict(self.declared_payload.payload(recipe, items))

    def _reserve_attempt(self) -> tuple[str, Path]:
        """The smallest attempt index whose directory does not exist yet.

        Never reuse a directory: a refused or failed attempt keeps its evidence.
        """
        index = 1
        while True:
            name = f"attempt-{index:02d}"
            path = self.attempts_root / name
            if not path.exists():
                return name, path
            index += 1

    def _base_evidence(
        self,
        recipe: TrainingRecipe,
        attempt: str,
        attempt_dir: Path,
        items: Sequence[Any],
    ) -> dict[str, Any]:
        return {
            "binding": KAGGLE_CAMPAIGN_BINDING,
            "backend": self.backend.name,
            "attempt": attempt,
            "attempt_dir": str(attempt_dir),
            "recipe_id": recipe.recipe_id,
            "declared_payload_kind": getattr(self.declared_payload, "kind", None),
            "projected_device_gpu_hours": float(recipe.projected_device_gpu_hours),
            "projected_wall_gpu_hours": float(recipe.projected_wall_gpu_hours),
            "declared_resume_from": recipe.resume_from_checkpoint,
            "curriculum_items": len(items),
            "source_identity": {
                "verified": False,
                "declared_commit_sha": self.commit_sha,
                "observed_commit_sha": None,
                "job_id": "",
                "reason": "the attempt did not reach a source check",
            },
            "notes": [
                "the attempt trains on the prepared declared inputs; "
                "materializing the curriculum corpus and running the production "
                "entry points are the declared payload command's responsibility",
            ],
        }

    def _refused_evidence(
        self,
        refusal: ComputeBackendRefusal,
        recipe: TrainingRecipe,
        attempt: str,
        attempt_dir: Path,
        items: Sequence[Any],
    ) -> dict[str, Any]:
        evidence = self._base_evidence(recipe, attempt, attempt_dir, items)
        evidence.update(
            {
                "status": OUTCOME_REFUSED,
                "candidate_succeeded": None,
                "refused_by": refusal.code,
                "refusal_reason": refusal.reason,
                "failure_reason": None,
                "failure_class": None,
                "resume_state": "not-applicable",
                "artifact_ref": None,
                "artifact_sha256": None,
                "artifact_manifest": [],
                "job_id": "",
                "environment": {},
                "measured_gpu_hours": None,
                "compute_cost": None,
                "budget_settlement": None,
                "actual_cost": None,
                "declared_inputs": [],
                "input_paths": {},
                "requested_mounts": list(self.mounts),
            }
        )
        return evidence

    def _evidence(
        self,
        outcome: AttemptOutcome,
        request: AttemptRequest,
        recipe: TrainingRecipe,
        attempt: str,
        attempt_dir: Path,
        items: Sequence[Any],
    ) -> dict[str, Any]:
        evidence = self._base_evidence(recipe, attempt, attempt_dir, items)
        evidence.update(outcome.to_evidence())
        declared = outcome.source_commit_sha.strip().lower()
        verified = bool(request.source.commit_sha) and declared == request.source.commit_sha
        if verified:
            reason = None
        elif not outcome.job_id:
            reason = (
                "no kernel record identified the installed source, so the "
                "declared commit was never echoed"
            )
        else:
            reason = (
                f"the kernel echoed {declared or '<nothing>'}, not the declared "
                f"{request.source.commit_sha}"
            )
        evidence.update(
            {
                "declared_inputs": [entry.to_dict() for entry in request.inputs],
                "input_paths": dict(request.payload.get("input_paths") or {}),
                "requested_mounts": list(request.mounts),
                "actual_cost": (
                    outcome.cost.to_dict() if outcome.cost is not None else None
                ),
                "failure_reason": (
                    (outcome.refusal_reason or None)
                    if outcome.status == OUTCOME_FAILED
                    else None
                ),
                "source_identity": {
                    "verified": verified,
                    "declared_commit_sha": request.source.commit_sha,
                    "observed_commit_sha": declared or None,
                    "job_id": outcome.job_id,
                    "reason": reason,
                },
            }
        )
        # The declared payload's own summary -- the production experiment id,
        # metrics and the trained artifact -- is merged from the returned
        # attempt directory; it is one of the artifacts the kernel's manifest
        # bound, so its verdict is read rather than re-derived.
        return merge_run_summary(evidence, attempt_dir)

    def _write_evidence(self, attempt_dir: Path, evidence: Mapping[str, Any]) -> None:
        attempt_dir.mkdir(parents=True, exist_ok=True)
        (attempt_dir / KAGGLE_CAMPAIGN_EVIDENCE_FILE).write_text(
            json.dumps(dict(evidence), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
