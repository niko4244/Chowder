"""Production candidate evaluation: the run measures the artifact it selected.

:mod:`chowder.growth.candidate_evaluation` owns the *contract* -- a candidate
arm must be candidate-measured, digest-bound and complete. This module owns the
*measurement*: the binding that turns "measure the adapter this run selected,
under this campaign's declared protocol" into real work through the production
evaluator and its worker subprocess.

It is deliberately the same shape as
:class:`chowder.growth.training_binding.SubprocessTrainingFn`, because the two
are the same kind of object in this system: a preregistered operation that is
run in a child process, measured, and recorded as durable evidence. So it reuses
the production pieces rather than reimplementing them:

* :class:`chowder.evaluators.transformers_text.TransformersTextEvalSpec` -- the
  real evaluation spec, with the production worker as its executor. This is not
  a second evaluation framework; it is the one that already writes
  ``predictions-<suite>.jsonl`` per item.
* :func:`chowder.evaluators.transformers_text.TransformersTextEvaluator._worker_command`
  -- the same ``--spec/--result/--chowder-identity`` command line production
  uses, so the worker verifies the checkout it imported before it reads a spec.
* the injectable ``Runner`` seam from the training binding, so the
  no-GPU harness can drive the whole path with a recording process runner.

Three things make the resulting arm evidence rather than a claim:

* **Identity.** The ``adapter_digest`` in the report is recomputed from the
  artifact's real bytes at measurement time and compared both against the
  request and against what the model actually loaded, so a swapped or rebuilt
  adapter refuses instead of being measured under another artifact's name.
* **Calibrated slices.** The campaign declares a mini-slice protocol
  (``n_samples``, item order, seed, decoding, prompt policy). This binding
  writes the slice it is about to measure from the declared dataset -- the first
  ``n_samples`` items in dataset order -- into the evaluation directory, and
  measures that file. So the row's ``sample_indices`` are ``0..N-1`` by
  construction, and the exact bytes measured stay in the run root as evidence.
* **Real per-item evidence.** Each row names its ``predictions-<suite>.jsonl``
  and the sha256 of those bytes. Its score is the mean of the item scores in
  that file, and ``n_samples`` is how many items the file holds; certification
  later recomputes all three from the bytes, so a report whose numbers its own
  samples contradict cannot be certified.

Cost is reported as a :class:`~chowder.growth.compute_cost.ComputeCost`: the
wall GPU-hours the evaluation actually occupied, measured by the process runner
and the worker's own accelerator count. The campaign charges it to the cycle
ledger, so measuring the candidate is spend like any other.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from chowder.evals.result import (
    MEASURED_THIS_GENERATION,
    RAW_MODEL,
    SUPPORTED,
    BenchmarkRun,
    EvalReport,
)
from chowder.evaluators.transformers_text import (
    EvalSuiteSpec,
    TransformersTextEvalSpec,
    TransformersTextEvaluator,
)

from .candidate_evaluation import (
    CandidateEvaluation,
    EvaluationRequest,
    CandidateEvaluationRefusal,
)
from .compute_cost import ComputeCost
from .generation_diagnostics import GenerationDiagnostics
from .training_binding import (
    SubprocessOutcome,
    default_runner,
    directory_digest,
)

#: The evaluation material names no dataset for a benchmark the campaign
#: declares as measured.
CANDIDATE_EVALUATION_MATERIAL_INCOMPLETE = "CANDIDATE_EVALUATION_MATERIAL_INCOMPLETE"
#: The declared artifact's bytes do not hash to the digest the run selected.
CANDIDATE_EVALUATION_ARTIFACT_DIGEST_MISMATCH = (
    "CANDIDATE_EVALUATION_ARTIFACT_DIGEST_MISMATCH"
)
#: The declared dataset holds fewer items than the protocol's mini-slice size.
CANDIDATE_EVALUATION_SLICE_TOO_SHORT = "CANDIDATE_EVALUATION_SLICE_TOO_SHORT"
#: The evaluation process failed, or left no result manifest.
CANDIDATE_EVALUATION_PROCESS_FAILED = "CANDIDATE_EVALUATION_PROCESS_FAILED"
#: The result manifest is malformed, or describes suites nobody declared.
CANDIDATE_EVALUATION_RESULT_INVALID = "CANDIDATE_EVALUATION_RESULT_INVALID"
#: The material contradicts the protocol the campaign declared.
CANDIDATE_EVALUATION_PROTOCOL_CONTRADICTION = (
    "CANDIDATE_EVALUATION_PROTOCOL_CONTRADICTION"
)
#: A row's per-item evidence is missing, unreadable, or inconsistent with it.
CANDIDATE_EVALUATION_EVIDENCE_INVALID = "CANDIDATE_EVALUATION_EVIDENCE_INVALID"

#: The directory, under the run root, this binding writes its evaluations into.
EVALUATION_ROOT = "evaluation"

#: What a mini-slice protocol's ``prompt_policy`` means to the worker. Named
#: explicitly so an unrecognised policy refuses instead of silently rendering
#: raw prompt bytes while the protocol claims a chat template.
_PROMPT_POLICIES: Mapping[str, bool] = {
    "chat_template": True,
    "raw": False,
}


@dataclass(frozen=True)
class SuiteMaterial:
    """How one declared benchmark is measured: the dataset and how to score it."""

    benchmark_qualified_id: str
    name: str
    dataset: Path
    scoring: str = "normalized_exact_match"
    prompt_field: str = "prompt"
    expected_field: str = "expected"
    metric: str = "accuracy"

    @classmethod
    def from_mapping(cls, document: Mapping[str, Any], *, source: str) -> "SuiteMaterial":
        qualified_id = document.get("benchmark_qualified_id")
        if not isinstance(qualified_id, str) or not qualified_id.strip():
            raise CandidateEvaluationRefusal(
                f"{source}: every evaluation suite must name the "
                "benchmark_qualified_id it measures, pinned as id@version; a "
                "suite nobody declared cannot be bound to a declared benchmark"
            )
        dataset = document.get("dataset")
        if not isinstance(dataset, str) or not dataset.strip():
            raise CandidateEvaluationRefusal(
                f"{source}: evaluation suite {qualified_id!r} declares no dataset, "
                "so there is nothing to measure"
            )
        name = document.get("name")
        if name is not None and (not isinstance(name, str) or not name.strip()):
            raise CandidateEvaluationRefusal(
                f"{source}: evaluation suite {qualified_id!r} declares an empty name"
            )
        return cls(
            benchmark_qualified_id=qualified_id,
            # The suite name is what the worker names its predictions file
            # after; derive it from the benchmark id when it is not declared, so
            # one suite never has to be described twice.
            name=name or qualified_id.split("@", 1)[0],
            dataset=Path(dataset),
            scoring=str(document.get("scoring", "normalized_exact_match")),
            prompt_field=str(document.get("prompt_field", "prompt")),
            expected_field=str(document.get("expected_field", "expected")),
            metric=str(document.get("metric", "accuracy")),
        )


@dataclass(frozen=True)
class EvaluationMaterial:
    """The declared evaluation material: what each benchmark is measured with.

    The candidate *report* is a run output and is never declared, but the
    material it is measured on is an input -- exactly like the training corpus.
    A campaign that cannot say which items it will measure its candidate on has
    not declared an evaluation.
    """

    source: Path
    suites: tuple[SuiteMaterial, ...]

    @classmethod
    def load(cls, path: Path | str) -> "EvaluationMaterial":
        path = Path(path)
        if not path.is_file():
            raise CandidateEvaluationRefusal(
                f"{CANDIDATE_EVALUATION_MATERIAL_INCOMPLETE}: the declared "
                f"evaluation material {path} does not exist, so the campaign "
                "cannot measure its candidate"
            )
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise CandidateEvaluationRefusal(
                f"{CANDIDATE_EVALUATION_MATERIAL_INCOMPLETE}: the declared "
                f"evaluation material {path} is not readable JSON: {error}"
            ) from error
        if not isinstance(document, Mapping):
            raise CandidateEvaluationRefusal(
                f"{path}: the evaluation material must be a JSON object"
            )
        unknown = sorted(set(document) - {"suites", "notes"})
        if unknown:
            raise CandidateEvaluationRefusal(
                f"{path}: unknown evaluation-material fields {unknown}; a key the "
                "evaluator cannot act on would be silently ignored"
            )
        raw_suites = document.get("suites")
        if not isinstance(raw_suites, list) or not raw_suites:
            raise CandidateEvaluationRefusal(
                f"{path}: evaluation material must declare a non-empty 'suites' list"
            )
        suites: list[SuiteMaterial] = []
        for entry in raw_suites:
            if not isinstance(entry, Mapping):
                raise CandidateEvaluationRefusal(
                    f"{path}: every evaluation suite must be an object"
                )
            suites.append(SuiteMaterial.from_mapping(entry, source=str(path)))
        ids = [suite.benchmark_qualified_id for suite in suites]
        duplicated = sorted({value for value in ids if ids.count(value) > 1})
        if duplicated:
            raise CandidateEvaluationRefusal(
                f"{path}: {duplicated} is declared more than once; two suites for "
                "one benchmark would produce two rows under one name"
            )
        names = [suite.name for suite in suites]
        duplicates = sorted({value for value in names if names.count(value) > 1})
        if duplicates:
            raise CandidateEvaluationRefusal(
                f"{path}: suite names {duplicates} are not unique, so the worker's "
                "per-suite evidence would collide"
            )
        return cls(source=path, suites=tuple(suites))

    def missing(self, benchmark_ids: Sequence[str]) -> tuple[str, ...]:
        """Declared benchmarks this material has no suite for."""
        covered = {suite.benchmark_qualified_id for suite in self.suites}
        return tuple(
            qualified_id for qualified_id in benchmark_ids if qualified_id not in covered
        )

    def for_benchmark(self, qualified_id: str) -> SuiteMaterial:
        for suite in self.suites:
            if suite.benchmark_qualified_id == qualified_id:
                return suite
        raise CandidateEvaluationRefusal(
            f"{CANDIDATE_EVALUATION_MATERIAL_INCOMPLETE}: no evaluation material "
            f"for {qualified_id}"
        )


class SubprocessEvaluationFn:
    """The production :data:`CandidateEvaluator`: one real evaluation, one arm.

    Callable so a campaign can use it directly. Every refusal is named, and
    none of them can be satisfied by reading a report from somewhere else: this
    binding either measures the artifact it was asked about or the run stops.
    """

    def __init__(
        self,
        *,
        run_root: str | Path,
        material: EvaluationMaterial,
        protocol: Any,
        base_model_path: str | Path,
        base_model_digest: str = "",
        python: str | None = None,
        environment: Mapping[str, str] | None = None,
        runner: Any = None,
        timeout_seconds: float = 3600.0,
        precision: str = "auto",
        quantization: str = "none",
        device: str = "auto",
        placement: str = "resident",
        offline: bool = False,
        batch_size: int = 1,
    ) -> None:
        self.run_root = Path(run_root)
        self.material = material
        self.protocol = protocol
        self.base_model_path = str(base_model_path)
        self.base_model_digest = str(base_model_digest)
        self.python = python or sys.executable
        self.environment = dict(environment or {})
        if any(key.upper() == "PYTHONPATH" for key in self.environment):
            raise CandidateEvaluationRefusal(
                "the evaluation binding builds each child's environment with "
                "worker_env(), which places this checkout's package root first on "
                "PYTHONPATH; passing PYTHONPATH here would evaluate with a "
                "competing chowder ahead of it"
            )
        self.runner = runner or default_runner
        self.timeout_seconds = float(timeout_seconds)
        self.precision = precision
        self.quantization = quantization
        self.device = device
        self.placement = placement
        self.offline = offline
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise CandidateEvaluationRefusal(
                f"the declared evaluation execution batch_size {batch_size!r} is "
                "not a positive integer, so the campaign does not say how its "
                "candidate is decoded"
            )
        self.batch_size = batch_size

    # ------------------------------------------------------------------
    # admission: what is knowable before any model is loaded
    # ------------------------------------------------------------------

    def admit(
        self, *, benchmarks: Sequence[str], protocol: Any
    ) -> tuple[str, str] | None:
        """Whether this evaluator could measure these benchmarks, before compute.

        The mirror of ``SubprocessTrainingFn.admit``: the runner asks before it
        trains, so a campaign cannot spend a training budget and only then
        discover that its evaluation material is missing, short, or describes a
        protocol this instrument cannot render. Returns ``None`` when the
        evaluation is runnable, else ``(kind, reason)``.
        """
        missing = self.material.missing(benchmarks)
        if missing:
            return (
                CANDIDATE_EVALUATION_MATERIAL_INCOMPLETE,
                f"the evaluation material {self.material.source} names no dataset "
                f"for {list(missing)}, which the campaign declares as measured",
            )
        policy = str(protocol.prompt_policy)
        if policy not in _PROMPT_POLICIES:
            return (
                CANDIDATE_EVALUATION_PROTOCOL_CONTRADICTION,
                f"the declared prompt policy {policy!r} is not one this evaluator "
                f"can render ({sorted(_PROMPT_POLICIES)})",
            )
        for qualified_id in benchmarks:
            material = self.material.for_benchmark(qualified_id)
            if not material.dataset.is_file():
                return (
                    CANDIDATE_EVALUATION_MATERIAL_INCOMPLETE,
                    f"the evaluation material names dataset {material.dataset} for "
                    f"{qualified_id}, which does not exist",
                )
            rows = _count_items(material)
            if rows < int(protocol.n_samples):
                return (
                    CANDIDATE_EVALUATION_SLICE_TOO_SHORT,
                    f"{material.dataset} holds {rows} item(s) and the declared "
                    f"protocol needs {int(protocol.n_samples)} for {qualified_id}",
                )
        return None

    # ------------------------------------------------------------------
    # the CandidateEvaluator contract
    # ------------------------------------------------------------------

    def __call__(self, request: EvaluationRequest) -> CandidateEvaluation:
        declared = request.declared_benchmarks
        missing = self.material.missing(declared)
        if missing:
            raise CandidateEvaluationRefusal(
                f"{CANDIDATE_EVALUATION_MATERIAL_INCOMPLETE}: the campaign declares "
                f"{list(missing)} as measured sets but its evaluation material "
                f"({self.material.source}) names no dataset for them; the candidate "
                "arm covers every declared set or the run cannot be certified"
            )

        prompt_policy = str(request.protocol.get("prompt_policy", ""))
        if prompt_policy not in _PROMPT_POLICIES:
            raise CandidateEvaluationRefusal(
                f"{CANDIDATE_EVALUATION_PROTOCOL_CONTRADICTION}: the declared prompt "
                f"policy {prompt_policy!r} is not one this evaluator can render "
                f"({sorted(_PROMPT_POLICIES)}); a protocol the instrument cannot "
                "honour would measure something other than what was declared"
            )
        n_samples = int(self.protocol.n_samples)
        decoding = dict(self.protocol.decoding)
        if int(decoding.get("max_new_tokens", 0)) <= 0:
            raise CandidateEvaluationRefusal(
                f"{CANDIDATE_EVALUATION_PROTOCOL_CONTRADICTION}: the declared "
                "decoding names no positive max_new_tokens"
            )

        # 1. the artifact's real bytes, before anything is measured: the report
        #    will name this digest, so it may not be the caller's claim alone.
        artifact = Path(request.artifact_ref)
        if not artifact.exists():
            raise CandidateEvaluationRefusal(
                f"{CANDIDATE_EVALUATION_ARTIFACT_DIGEST_MISMATCH}: the selected "
                f"artifact {request.artifact_ref!r} does not exist, so there is "
                "nothing to measure"
            )
        observed = digest_of(artifact)
        if observed != request.artifact_sha256:
            raise CandidateEvaluationRefusal(
                f"{CANDIDATE_EVALUATION_ARTIFACT_DIGEST_MISMATCH}: the selected "
                f"artifact {request.artifact_ref!r} hashes to {observed}, the run "
                f"recorded {request.artifact_sha256}; the candidate arm must "
                "measure the bytes the run selected"
            )

        eval_dir = self._reserve_eval_dir()
        eval_dir.mkdir(parents=True, exist_ok=False)

        slices = self._write_slices(
            eval_dir=eval_dir,
            benchmarks=declared,
            n_samples=n_samples,
        )
        spec = TransformersTextEvalSpec(
            base_model=self.base_model_path,
            adapter_dir=str(artifact.resolve()),
            output_dir=str(eval_dir),
            suites=tuple(
                EvalSuiteSpec(
                    name=self.material.for_benchmark(qualified_id).name,
                    dataset=str(slice_path),
                    prompt_field=self.material.for_benchmark(qualified_id).prompt_field,
                    expected_field=self.material.for_benchmark(qualified_id).expected_field,
                    scoring=self.material.for_benchmark(qualified_id).scoring,
                    max_new_tokens=int(decoding["max_new_tokens"]),
                    use_chat_template=_PROMPT_POLICIES[prompt_policy],
                    batch_size=self.batch_size,
                )
                for qualified_id, slice_path in slices.items()
            ),
            precision=self.precision,
            quantization=self.quantization,
            device=self.device,
            placement=self.placement,
            seed=int(self.protocol.seed),
            timeout_seconds=self.timeout_seconds,
            offline=self.offline,
        )
        outcome, payload = self._run_worker(spec, eval_dir=eval_dir)

        runs = self._rows(
            payload=payload,
            spec=spec,
            request=request,
            eval_dir=eval_dir,
            protocol=self.protocol,
            slices=slices,
        )
        report = EvalReport(
            generation_version=request.candidate_version,
            runs=runs,
            hardware=dict(payload.get("runtime", {})),
            date=_utc_now(),
            model_identity={
                "base_model_path": self.base_model_path,
                # Verified against the real tree by the run's own identity
                # phase before any compute; recorded here so the arm names the
                # base it was measured over.
                "base_model_digest": request.base_model_digest,
                "adapter_path": str(artifact.resolve()),
                # Recomputed from the artifact's bytes just above, and checked
                # again against what the worker reported it loaded.
                "adapter_digest": observed,
            },
        )
        self._write_evaluation_evidence(
            eval_dir=eval_dir,
            request=request,
            outcome=outcome,
            payload=payload,
            run_ids=tuple(run.benchmark_qualified_id for run in runs),
        )
        return CandidateEvaluation(
            report=report,
            cost=self._cost(outcome=outcome, payload=payload),
        )

    # ------------------------------------------------------------------
    # slices, worker, rows
    # ------------------------------------------------------------------

    def _reserve_eval_dir(self) -> Path:
        """The next immutable evaluation directory under the run root."""
        root = self.run_root / EVALUATION_ROOT
        root.mkdir(parents=True, exist_ok=True)
        index = 1
        while True:
            candidate = root / f"eval-{index:02d}"
            if not candidate.exists():
                return candidate
            index += 1

    def _write_slices(
        self,
        *,
        eval_dir: Path,
        benchmarks: Sequence[str],
        n_samples: int,
    ) -> dict[str, Path]:
        """Write the mini-slice this evaluation will measure, from the declared data.

        The protocol is "the first ``n_samples`` items in dataset order", so the
        slice is written here and measured from here: the row cannot cover a
        different item set than the one it declares, and the exact bytes are in
        the run root as evidence.
        """
        paths: dict[str, Path] = {}
        for qualified_id in benchmarks:
            material = self.material.for_benchmark(qualified_id)
            if not material.dataset.is_file():
                raise CandidateEvaluationRefusal(
                    f"{CANDIDATE_EVALUATION_MATERIAL_INCOMPLETE}: the evaluation "
                    f"material names dataset {material.dataset} for {qualified_id}, "
                    "which does not exist; the candidate cannot be measured on data "
                    "that is not there"
                )
            rows: list[dict[str, Any]] = []
            with material.dataset.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if len(rows) >= n_samples:
                        break
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except ValueError as error:
                        raise CandidateEvaluationRefusal(
                            f"{material.dataset}:{line_number} is not JSON: {error}"
                        ) from error
                    if not isinstance(row, Mapping):
                        raise CandidateEvaluationRefusal(
                            f"{material.dataset}:{line_number} is not a JSON object"
                        )
                    for field in (material.prompt_field, material.expected_field):
                        if field not in row:
                            raise CandidateEvaluationRefusal(
                                f"{material.dataset}:{line_number} is missing "
                                f"{field!r}, so the item cannot be scored"
                            )
                    rows.append(dict(row))
            if len(rows) < n_samples:
                raise CandidateEvaluationRefusal(
                    f"{CANDIDATE_EVALUATION_SLICE_TOO_SHORT}: {material.dataset} holds "
                    f"{len(rows)} item(s), the declared protocol needs {n_samples}; a "
                    "slice smaller than the frozen protocol is not that protocol"
                )
            path = eval_dir / f"slice-{material.name}.jsonl"
            path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            paths[qualified_id] = path
        return paths

    def _run_worker(
        self, spec: TransformersTextEvalSpec, *, eval_dir: Path
    ) -> tuple[SubprocessOutcome, Mapping[str, Any]]:
        """Launch the production worker through the injectable process runner."""
        from chowder.worker_env import chowder_source_identity

        spec_path = eval_dir / "eval-spec.json"
        result_path = eval_dir / "eval-result.json"
        spec_path.write_text(spec.canonical_json() + "\n", encoding="utf-8")
        identity_path = eval_dir / "chowder-identity.json"
        identity_path.write_text(
            json.dumps(chowder_source_identity(), sort_keys=True) + "\n",
            encoding="utf-8",
        )

        # The canonical worker command line, reused from production rather than
        # re-spelled here; only the interpreter is this binding's, so the child
        # runs the checkout under test.
        command = [
            self.python,
            *TransformersTextEvaluator._worker_command(  # noqa: SLF001
                spec_path, result_path, chowder_identity=identity_path
            )[1:],
        ]
        outcome = self.runner(
            command, eval_dir, dict(self.environment), self.timeout_seconds
        )
        (eval_dir / "evaluation.stdout.txt").write_text(
            outcome.stdout, encoding="utf-8"
        )
        (eval_dir / "evaluation.stderr.txt").write_text(
            outcome.stderr, encoding="utf-8"
        )
        if outcome.returncode != 0:
            raise CandidateEvaluationRefusal(
                f"{CANDIDATE_EVALUATION_PROCESS_FAILED}: the evaluation worker "
                f"exited {outcome.returncode}"
                + (f" (timed out after {self.timeout_seconds:g}s)" if outcome.timed_out else "")
                + f"; stderr tail: {_tail(eval_dir / 'evaluation.stderr.txt')}"
            )
        if not result_path.is_file():
            raise CandidateEvaluationRefusal(
                f"{CANDIDATE_EVALUATION_PROCESS_FAILED}: the evaluation worker "
                "exited successfully without writing a result manifest, so no "
                "measurement exists to read"
            )
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except ValueError as error:
            raise CandidateEvaluationRefusal(
                f"{CANDIDATE_EVALUATION_RESULT_INVALID}: the evaluation result "
                f"manifest is not readable JSON: {error}"
            ) from error
        if not isinstance(payload, Mapping):
            raise CandidateEvaluationRefusal(
                f"{CANDIDATE_EVALUATION_RESULT_INVALID}: the evaluation result "
                "manifest is not a JSON object"
            )
        return outcome, payload

    def _rows(
        self,
        *,
        payload: Mapping[str, Any],
        spec: TransformersTextEvalSpec,
        request: EvaluationRequest,
        eval_dir: Path,
        protocol: Any,
        slices: Mapping[str, Path],
    ) -> tuple[BenchmarkRun, ...]:
        """One digest-bound row per declared benchmark, from the worker's own files."""
        metrics = payload.get("metrics")
        suite_evidence = payload.get("suites")
        if not isinstance(metrics, Mapping) or not isinstance(suite_evidence, Mapping):
            raise CandidateEvaluationRefusal(
                f"{CANDIDATE_EVALUATION_RESULT_INVALID}: the evaluation result "
                "carries no metrics/suites measurements"
            )
        expected_names = {suite.name for suite in spec.suites}
        if set(metrics) != expected_names or set(suite_evidence) != expected_names:
            raise CandidateEvaluationRefusal(
                f"{CANDIDATE_EVALUATION_RESULT_INVALID}: the evaluation measured "
                f"{sorted(set(metrics) | set(suite_evidence))} but the campaign "
                f"declared {sorted(expected_names)}"
            )

        by_name = {suite.name: suite for suite in spec.suites}
        runs: list[BenchmarkRun] = []
        for qualified_id in request.declared_benchmarks:
            material = self.material.for_benchmark(qualified_id)
            suite = by_name[material.name]
            evidence = suite_evidence[material.name]
            if not isinstance(evidence, Mapping):
                raise CandidateEvaluationRefusal(
                    f"{CANDIDATE_EVALUATION_RESULT_INVALID}: suite evidence for "
                    f"{material.name!r} is not an object"
                )
            predictions_ref = evidence.get("predictions_file")
            if not isinstance(predictions_ref, str) or not predictions_ref:
                raise CandidateEvaluationRefusal(
                    f"{CANDIDATE_EVALUATION_EVIDENCE_INVALID}: suite "
                    f"{material.name!r} names no predictions file, so its rows have "
                    "no underlying measurement"
                )
            predictions = Path(predictions_ref)
            if not predictions.is_absolute():
                predictions = eval_dir / predictions
            if not predictions.is_file():
                raise CandidateEvaluationRefusal(
                    f"{CANDIDATE_EVALUATION_EVIDENCE_INVALID}: the per-item evidence "
                    f"{predictions} for {material.name!r} does not exist"
                )
            # The worker also fingerprints the prompt/expected pairs it scored.
            # Requiring it here is what stops a row from naming one item set
            # while carrying another's scores: the fingerprint file binds the
            # questions, the predictions file the answers, and both are hashed.
            fingerprint_ref = evidence.get("holdout_fingerprints_file")
            fingerprint_digest = evidence.get("holdout_fingerprints_sha256")
            if not isinstance(fingerprint_ref, str) or not fingerprint_ref:
                raise CandidateEvaluationRefusal(
                    f"{CANDIDATE_EVALUATION_EVIDENCE_INVALID}: suite "
                    f"{material.name!r} names no holdout fingerprint file, so the "
                    "items it scored are not bound to the declared slice"
                )
            fingerprints = Path(fingerprint_ref)
            if not fingerprints.is_absolute():
                fingerprints = eval_dir / fingerprints
            if not fingerprints.is_file():
                raise CandidateEvaluationRefusal(
                    f"{CANDIDATE_EVALUATION_EVIDENCE_INVALID}: holdout fingerprint "
                    f"evidence {fingerprints} for {material.name!r} does not exist"
                )
            if not isinstance(fingerprint_digest, str) or (
                _sha256_file(fingerprints) != fingerprint_digest
            ):
                raise CandidateEvaluationRefusal(
                    f"{CANDIDATE_EVALUATION_EVIDENCE_INVALID}: the holdout "
                    f"fingerprint evidence for {material.name!r} does not hash to "
                    f"the digest the worker declared ({fingerprint_digest!r})"
                )
            items = _read_items(predictions, suite_name=material.name)
            samples = [float(item["score"]) for item in items]
            if len(samples) != int(protocol.n_samples):
                raise CandidateEvaluationRefusal(
                    f"{CANDIDATE_EVALUATION_EVIDENCE_INVALID}: {material.name!r} "
                    f"measured {len(samples)} item(s); the declared protocol is "
                    f"{int(protocol.n_samples)}"
                )
            score = sum(samples) / len(samples)
            runs.append(
                BenchmarkRun(
                    benchmark_qualified_id=qualified_id,
                    adapter="transformers_text",
                    generation_version=request.candidate_version,
                    score=score,
                    support=SUPPORTED,
                    measurement_kind=RAW_MODEL,
                    n_samples=len(samples),
                    per_sample_scores=tuple(samples),
                    metric=material.metric,
                    raw_artifact_ref=_relative_to_run_root(predictions, self.run_root),
                    measurement_origin=MEASURED_THIS_GENERATION,
                    metadata={
                        # The bytes these numbers came from, recomputed by
                        # certification from the file itself.
                        "artifact_sha256": _sha256_file(predictions),
                        # What the generations actually did, from the same file
                        # the score came from: the campaign's declared target set
                        # is the generation-diagnostics instrument, and the frozen
                        # judge reads these keys directly (T1-T10).
                        **GenerationDiagnostics.from_items(
                            items,
                            max_new_tokens=int(protocol.decoding["max_new_tokens"]),
                            seed=int(protocol.seed),
                            source=str(predictions),
                        ).to_metadata(),
                        "sample_indices": list(range(len(samples))),
                        "seed": int(protocol.seed),
                        "shuffle": bool(protocol.shuffle),
                        # The declared decoding, plus the execution parameter the
                        # declared decoding does not cover. The judge enforces the
                        # keys the *declared protocol* names and ignores extras
                        # (certification.protocol_problems), so this is recorded
                        # evidence rather than a changed measurement rule.
                        "decoding": {**dict(protocol.decoding), "batch_size": self.batch_size},
                        "prompt_policy": str(protocol.prompt_policy),
                        "suite": material.name,
                        "holdout_fingerprints_ref": _relative_to_run_root(
                            fingerprints, self.run_root
                        ),
                        "holdout_fingerprints_sha256": fingerprint_digest,
                        "slice_ref": _relative_to_run_root(slices[qualified_id], self.run_root),
                        "slice_sha256": _sha256_file(slices[qualified_id]),
                        "eval_dir": _relative_to_run_root(eval_dir, self.run_root),
                        "spec_sha256": spec.digest(),
                    },
                )
            )
        return tuple(runs)

    def _cost(
        self, *, outcome: SubprocessOutcome, payload: Mapping[str, Any]
    ) -> ComputeCost:
        """The measured wall GPU-hours this evaluation occupied.

        The evaluation worker reports its accelerator count, so the wall cost is
        the time the process actually ran times the accelerators it held. Device
        time is not separated by this worker, so it is not claimed: the value
        reports itself as wall-charged and ``device_measured`` stays False, which
        is what stops it from settling a device ceiling.
        """
        runtime = payload.get("runtime")
        accelerators = 0
        if isinstance(runtime, Mapping):
            reported = runtime.get("gpu_count", 0)
            if isinstance(reported, (int, float)) and not isinstance(reported, bool):
                accelerators = max(0, int(reported))
        wall = (float(outcome.seconds) * accelerators) / 3600.0
        return ComputeCost.from_wall_only(wall, source="candidate evaluation")

    def _write_evaluation_evidence(
        self,
        *,
        eval_dir: Path,
        request: EvaluationRequest,
        outcome: SubprocessOutcome,
        payload: Mapping[str, Any],
        run_ids: Sequence[str],
    ) -> None:
        """The evaluation's own durable record, next to the bytes it measured."""
        document = {
            "binding": "chowder.growth.evaluation_binding",
            "cycle_id": request.cycle_id,
            "candidate_version": request.candidate_version,
            "recipe_id": request.recipe_id,
            "attempt": request.attempt,
            "artifact_ref": request.artifact_ref,
            "artifact_sha256": request.artifact_sha256,
            "base_model_path": self.base_model_path,
            "base_model_digest": self.base_model_digest,
            "protocol": dict(request.protocol),
            "rows": list(run_ids),
            "material": str(self.material.source),
            "process": {
                "command": list(outcome.command),
                "returncode": outcome.returncode,
                "seconds": outcome.seconds,
                "timed_out": outcome.timed_out,
            },
            "runtime": payload.get("runtime", {}),
            "model_provenance": payload.get("model_provenance", {}),
            "versions": payload.get("versions", {}),
        }
        (eval_dir / "evaluation-evidence.json").write_text(
            json.dumps(document, indent=2, sort_keys=True), encoding="utf-8"
        )


def _count_items(material: SuiteMaterial) -> int:
    """How many usable items a declared dataset holds, without loading them all."""
    count = 0
    with material.dataset.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, Mapping) and all(
                field in row for field in (material.prompt_field, material.expected_field)
            ):
                count += 1
    return count


def _read_items(path: Path, *, suite_name: str) -> list[dict[str, Any]]:
    """The per-item rows the worker wrote, refusing anything unreadable.

    One parser for these bytes: the row's score, its per-sample values and its
    generation diagnostics all come from this single read, so a report cannot
    carry a score from one parsing and diagnostics from another.
    """
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as error:
                raise CandidateEvaluationRefusal(
                    f"{CANDIDATE_EVALUATION_EVIDENCE_INVALID}: {path}:{line_number} "
                    f"is not JSON: {error}"
                ) from error
            if not isinstance(row, Mapping):
                raise CandidateEvaluationRefusal(
                    f"{CANDIDATE_EVALUATION_EVIDENCE_INVALID}: {path}:{line_number} "
                    "is not an object, so it is not a scored item"
                )
            score = row.get("score")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise CandidateEvaluationRefusal(
                    f"{CANDIDATE_EVALUATION_EVIDENCE_INVALID}: {path}:{line_number} "
                    f"carries no numeric score, so {suite_name!r} has no measurement "
                    "for that item"
                )
            items.append({**row, "score": float(score)})
    if not items:
        raise CandidateEvaluationRefusal(
            f"{CANDIDATE_EVALUATION_EVIDENCE_INVALID}: {path} holds no scored item, "
            f"so {suite_name!r} was not measured"
        )
    return items


def _relative_to_run_root(path: Path, run_root: Path) -> str:
    """A ref the judge resolves against the run root, or a refusal.

    The candidate arm must be self-contained: its evidence lives in the run the
    evaluation was asked to measure for. An artifact outside the run root would
    make a certified verdict depend on a directory somebody else can move or
    delete, so it refuses rather than naming an absolute path.
    """
    resolved = Path(path).resolve()
    root = Path(run_root).resolve()
    if not resolved.is_relative_to(root):
        raise CandidateEvaluationRefusal(
            f"the evaluation wrote {resolved} outside the run root {root}; the "
            "candidate arm's evidence must live inside the run it measured for, "
            "so the verdict does not depend on an external directory"
        )
    return str(resolved.relative_to(root)).replace("\\", "/")


def _sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tail(path: Path, lines: int = 20) -> str:
    if not path.exists():
        return ""
    return "\n".join(
        path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    )


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


#: Public digest helper: a single-file sha256, or a directory tree's digest, so
#: an adapter artifact and a single measurement file are both nameable.
def digest_of(path: Path) -> str:
    if path.is_dir():
        digest, _entries = directory_digest(path)
        return digest
    return _sha256_file(path)


__all__ = [
    "CANDIDATE_EVALUATION_ARTIFACT_DIGEST_MISMATCH",
    "CANDIDATE_EVALUATION_EVIDENCE_INVALID",
    "CANDIDATE_EVALUATION_MATERIAL_INCOMPLETE",
    "CANDIDATE_EVALUATION_PROCESS_FAILED",
    "CANDIDATE_EVALUATION_PROTOCOL_CONTRADICTION",
    "CANDIDATE_EVALUATION_RESULT_INVALID",
    "CANDIDATE_EVALUATION_SLICE_TOO_SHORT",
    "EVALUATION_ROOT",
    "EvaluationMaterial",
    "SubprocessEvaluationFn",
    "SuiteMaterial",
    "digest_of",
]
