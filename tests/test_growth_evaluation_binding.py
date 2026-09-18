"""The production candidate evaluator: the run measures the artifact it made.

``build_evaluator`` used to return ``None``, so a campaign without an injected
evaluation seam refused with ``CANDIDATE_EVALUATION_NOT_PRODUCED``: the
contract for a candidate arm existed, the instrument that produced one did not.
This module exercises the instrument that now exists.

The only thing faked is the GPU. ``_RecordingWorker`` stands in for
``chowder.evaluators.transformers_text_worker`` through the same process-runner
seam the binding tests use, so a campaign is driven through the real CLI, the
real production evaluation binding, the real worker command line and the real
certification gate -- and then the frozen judge is pointed at the same run root.

Three properties are pinned here:

* the arm is *produced*: ``candidate_evaluation.json`` carries
  ``MEASURED_THIS_GENERATION`` rows whose digest is the artifact this run
  selected, copied into the run root as verifiable bytes;
* the arm is *bound*: a report about other bytes, a tampered artifact, or a
  worker that never measured anything all refuse, and no verdict is recorded;
* the arm is *charged*: the evaluation's own measured wall GPU-hours reach the
  cycle's accounting.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from chowder.cli import main as chowder_main
from chowder.evals.result import MEASURED_THIS_GENERATION, EvalReport
from chowder.growth import campaign_runner
from chowder.growth.campaign import CampaignManifest
from chowder.growth.campaign_runner import CampaignRunRefusal, run_campaign
from chowder.growth.candidate_evaluation import (
    CANDIDATE_EVALUATION_NOT_PRODUCED,
    CandidateEvaluationRefusal,
    EvaluationRequest,
)
from chowder.growth.evaluation_binding import (
    CANDIDATE_EVALUATION_ARTIFACT_DIGEST_MISMATCH,
    CANDIDATE_EVALUATION_EVIDENCE_INVALID,
    CANDIDATE_EVALUATION_MATERIAL_INCOMPLETE,
    CANDIDATE_EVALUATION_PROCESS_FAILED,
    CANDIDATE_EVALUATION_SLICE_TOO_SHORT,
    EvaluationMaterial,
    SubprocessEvaluationFn,
    digest_of,
)
from chowder.growth.generation_diagnostics import (
    GENERATION_DIAGNOSTICS_UNMEASURED,
    GenerationDiagnostics,
)
from chowder.growth.lineage import GenerationLedger
from chowder.growth.training_binding import SubprocessOutcome
from chowder.evaluators.scoring import observed_score

import test_growth_campaign_runner as campaign_fixture
from test_growth_campaign_runner import (
    BROAD_ID,
    CANDIDATE_VERSION,
    PROTECTED_ID,
    TARGET_ID,
    _campaign,
    _patch_runner,
)


# --------------------------------------------------------------------------
# the recording worker: the production command line, a deterministic scorer
# --------------------------------------------------------------------------


class _RecordingWorker:
    """A process runner that answers the evaluation worker's real command line.

    It reads the spec the binding wrote, scores the items in the slice it was
    given, and writes exactly the artifacts the production worker writes: one
    ``predictions-<suite>.jsonl`` per suite, its holdout fingerprint index, and
    a result manifest. Everything about *what* to measure comes from the spec,
    so the binding cannot pass a test by asking for the right thing and
    measuring something else.
    """

    def __init__(
        self,
        *,
        gpu_count: int = 1,
        seconds: float = 120.0,
        fail: bool = False,
        omit_result: bool = False,
        omit_predictions_for: str = "",
        omit_generation_facts: bool = False,
        prediction_of=None,
        score_of=lambda name, index: 1.0 if index % 2 == 0 else 0.0,
    ) -> None:
        self.gpu_count = gpu_count
        self.seconds = seconds
        self.fail = fail
        self.omit_result = omit_result
        self.omit_predictions_for = omit_predictions_for
        # The production worker records how each generation ended (EOS or the
        # token cap). A worker that does not is the shape the diagnostics
        # refuse, so the flag has to be expressible here.
        self.omit_generation_facts = omit_generation_facts
        self.prediction_of = prediction_of or (lambda name, index, row: str(row.get("expected", "")))
        self.score_of = score_of
        self.commands: list[Sequence[str]] = []
        self.specs: list[dict[str, Any]] = []

    def __call__(
        self,
        command: Sequence[str],
        cwd: Path,
        extra_environment: Mapping[str, str],
        timeout_seconds: float | None,
    ) -> SubprocessOutcome:
        self.commands.append(tuple(command))
        spec_path = Path(command[command.index("--spec") + 1])
        result_path = Path(command[command.index("--result") + 1])
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        self.specs.append(spec)
        if self.fail:
            return SubprocessOutcome(
                command=tuple(command),
                returncode=1,
                stdout="",
                stderr="the evaluation worker refused to load the model",
                seconds=self.seconds,
                timed_out=False,
            )
        output_dir = Path(spec["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics: dict[str, float] = {}
        evidence: dict[str, Any] = {}
        for suite in spec["suites"]:
            name = suite["name"]
            rows = [
                json.loads(line)
                for line in Path(suite["dataset"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if name == self.omit_predictions_for:
                # The worker reported a score but left no per-item file: the
                # shape this binding must refuse rather than certify.
                metrics[name] = float(self.score_of(name, 0))
                predictions = output_dir / f"predictions-{name}.jsonl"
            else:
                items: list[dict[str, Any]] = []
                for index, row in enumerate(rows):
                    score = float(self.score_of(name, index))
                    facts = (
                        {} if self.omit_generation_facts else _generation_facts(suite, index)
                    )
                    # A scoring defined over the observation is computed from the
                    # facts the worker recorded, exactly as production does -- the
                    # recording worker must not be the only place that rule lives
                    # in two implementations.
                    observed = observed_score(facts, str(suite.get("scoring", "")))
                    if observed is not None:
                        score = float(observed)
                    items.append(
                        {
                            "prompt": row.get("prompt", ""),
                            "expected": row.get("expected", ""),
                            "prediction": self.prediction_of(name, index, row),
                            "score": score,
                            **facts,
                        }
                    )
                predictions = output_dir / f"predictions-{name}.jsonl"
                predictions.write_text(
                    "".join(json.dumps(item) + "\n" for item in items),
                    encoding="utf-8",
                )
                metrics[name] = sum(item["score"] for item in items) / len(items)
            fingerprints = output_dir / f"holdout-fingerprints-{name}.jsonl"
            fingerprints.write_text(
                "".join(
                    json.dumps(
                        {
                            "prompt": str(row.get("prompt", "")),
                            "expected": str(row.get("expected", "")),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                    for row in rows
                ),
                encoding="utf-8",
            )
            evidence[name] = {
                "rows": len(rows),
                "scoring": suite["scoring"],
                "predictions_file": str(predictions),
                "holdout_fingerprints_file": str(fingerprints),
                "holdout_fingerprints_sha256": _digest(fingerprints),
            }
        if not self.omit_result:
            result_path.write_text(
                json.dumps(
                    {
                        "metrics": metrics,
                        "suites": evidence,
                        "runtime": {
                            "device": "cuda:0" if self.gpu_count else "cpu",
                            "gpu_count": self.gpu_count,
                        },
                        "model_provenance": {
                            "requested_base_model": spec["base_model"],
                            "adapter_requested": True,
                            "adapter_loaded": True,
                        },
                        "versions": {"torch": "test", "transformers": "test"},
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        return SubprocessOutcome(
            command=tuple(command),
            returncode=0,
            stdout="",
            stderr="",
            seconds=self.seconds,
            timed_out=False,
        )


def _generation_facts(suite: Mapping[str, Any], index: int) -> dict[str, Any]:
    """What the production worker records beside each prediction.

    Deterministically shorter than the declared cap, so the recorded facts are
    self-consistent (a terminated generation stops short of the cap) and the
    diagnostics the binding computes are reproducible.
    """
    cap = int(suite["max_new_tokens"])
    return {"generated_tokens": min(1 + index % 3, cap - 1), "eos_terminated": True}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _request(manifest: CampaignManifest, *, artifact_ref: str, artifact_sha256: str) -> EvaluationRequest:
    protocol = manifest.protection.require_protocol()
    return EvaluationRequest(
        cycle_id=manifest.cycle_id,
        candidate_version=manifest.resolved_candidate_version(),
        base_model_path=manifest.base_model_path,
        base_model_digest=manifest.base_model_digest,
        artifact_ref=artifact_ref,
        artifact_sha256=artifact_sha256,
        recipe_id="recipe-a",
        attempt="attempt-01",
        target_benchmarks=tuple(manifest.target_benchmarks),
        protected_benchmarks=tuple(manifest.protected_benchmarks),
        broad_benchmarks=tuple(manifest.broad_benchmarks),
        protocol=protocol.to_dict(),
        output_root=manifest.state_root,
    )


def _artifact(tmp_path: Path, name: str = "candidate-adapter") -> Path:
    """A real, selectable artifact tree: two files, so the digest is tree-shaped."""
    artifact = tmp_path / name
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact / "adapter_config.json").write_text('{"r": 8}', encoding="utf-8")
    (artifact / "adapter_model.safetensors").write_text("candidate-weights", encoding="utf-8")
    return artifact


# --------------------------------------------------------------------------
# the seam itself: measure the artifact, not the request
# --------------------------------------------------------------------------


def test_the_production_evaluator_measures_the_sliced_declared_material(tmp_path: Path):
    """The arm is produced from the declared data, under the declared protocol."""
    manifest, _runner, _document = _campaign(tmp_path)
    worker = _RecordingWorker()
    evaluator = SubprocessEvaluationFn(
        run_root=Path(manifest.state_root),
        material=EvaluationMaterial.load(manifest.evaluation_material_path),
        protocol=manifest.protection.require_protocol(),
        base_model_path=manifest.base_model_path,
        base_model_digest=manifest.base_model_digest,
        runner=worker,
    )
    artifact = _artifact(tmp_path)
    request = _request(
        manifest, artifact_ref=str(artifact), artifact_sha256=digest_of(artifact)
    )

    evaluation = evaluator(request)
    report = evaluation.report

    assert report.generation_version == CANDIDATE_VERSION
    assert {run.benchmark_qualified_id for run in report.runs} == {
        TARGET_ID,
        PROTECTED_ID,
        BROAD_ID,
    }
    for run in report.runs:
        assert run.measurement_origin == MEASURED_THIS_GENERATION
        assert run.generation_version == CANDIDATE_VERSION
        # The declared mini-slice protocol, on the row the certification checks.
        assert run.n_samples == 16
        assert run.metadata["sample_indices"] == list(range(16))
        assert run.metadata["seed"] == 1234
        assert run.metadata["shuffle"] is False
        assert run.metadata["prompt_policy"] == "chat_template"
        assert run.metadata["decoding"] == {
            "temperature": 0.0,
            "do_sample": False,
            "max_new_tokens": 512,
        }
        # The score is the mean of the item scores, and the row is bound to the
        # bytes those numbers came from.
        artifact_path = Path(manifest.state_root) / str(run.raw_artifact_ref)
        assert artifact_path.is_file()
        assert run.metadata["artifact_sha256"] == _digest(artifact_path)
        samples = list(run.per_sample_scores)
        assert len(samples) == run.n_samples
        assert run.score == pytest.approx(sum(samples) / len(samples))

    # Identity: the bytes measured are the bytes the request named.
    assert report.model_identity["adapter_digest"] == digest_of(artifact)
    assert report.model_identity["base_model_digest"] == manifest.base_model_digest

    # The worker was asked for the slice, not the whole declared dataset: the
    # spec's dataset is the 16-item file this binding wrote.
    spec = worker.specs[0]
    assert {suite["max_new_tokens"] for suite in spec["suites"]} == {512}
    assert {suite["use_chat_template"] for suite in spec["suites"]} == {True}
    assert spec["seed"] == 1234
    assert spec["adapter_dir"] == str(artifact.resolve())
    for suite in spec["suites"]:
        assert Path(suite["dataset"]).name.startswith("slice-")
        assert len(
            Path(suite["dataset"]).read_text(encoding="utf-8").strip().splitlines()
        ) == 16
    # The evaluation's own durable record sits next to the bytes it measured.
    assert (Path(manifest.state_root) / "evaluation" / "eval-01" / "evaluation-evidence.json").is_file()


def test_the_evaluation_reports_the_wall_gpu_hours_it_actually_spent(tmp_path: Path):
    """Measuring the candidate is compute, and the binding says how much."""
    manifest, _runner, _document = _campaign(tmp_path)
    artifact = _artifact(tmp_path)
    evaluator = SubprocessEvaluationFn(
        run_root=Path(manifest.state_root),
        material=EvaluationMaterial.load(manifest.evaluation_material_path),
        protocol=manifest.protection.require_protocol(),
        base_model_path=manifest.base_model_path,
        runner=_RecordingWorker(gpu_count=1, seconds=3600.0),
    )
    evaluation = evaluator(
        _request(manifest, artifact_ref=str(artifact), artifact_sha256=digest_of(artifact))
    )

    assert evaluation.cost is not None
    assert evaluation.cost.wall_gpu_hours == pytest.approx(1.0)
    # Device time is not separated by this worker, so it is not claimed: an
    # unmeasured device figure must never settle a device ceiling.
    assert evaluation.cost.device_measured is False


# --------------------------------------------------------------------------
# refusals: everything that must not become an arm
# --------------------------------------------------------------------------


def test_an_undeclared_material_cannot_produce_a_candidate_arm(tmp_path: Path):
    _manifest, _runner, document = _campaign(tmp_path)
    manifest = campaign_fixture._redeclare(document, evaluation_material_path="")

    with pytest.raises(CampaignRunRefusal) as error:
        campaign_runner.build_evaluator(manifest)

    assert CANDIDATE_EVALUATION_NOT_PRODUCED in str(error.value)
    assert "evaluation_material_path" in str(error.value)


def test_material_that_names_no_dataset_for_a_declared_benchmark_refuses(tmp_path: Path):
    """A declared set the material cannot measure is a missing measurement."""
    manifest, _runner, _document = _campaign(tmp_path)
    material = json.loads(Path(manifest.evaluation_material_path).read_text(encoding="utf-8"))
    material["suites"] = [
        suite for suite in material["suites"] if suite["benchmark_qualified_id"] != PROTECTED_ID
    ]
    Path(manifest.evaluation_material_path).write_text(json.dumps(material), encoding="utf-8")

    evaluator = SubprocessEvaluationFn(
        run_root=Path(manifest.state_root),
        material=EvaluationMaterial.load(manifest.evaluation_material_path),
        protocol=manifest.protection.require_protocol(),
        base_model_path=manifest.base_model_path,
        runner=_RecordingWorker(),
    )
    artifact = _artifact(tmp_path)

    with pytest.raises(CandidateEvaluationRefusal) as error:
        evaluator(
            _request(manifest, artifact_ref=str(artifact), artifact_sha256=digest_of(artifact))
        )

    assert CANDIDATE_EVALUATION_MATERIAL_INCOMPLETE in str(error.value)
    assert PROTECTED_ID in str(error.value)


def test_a_dataset_shorter_than_the_declared_slice_refuses(tmp_path: Path):
    """A slice smaller than the frozen protocol is not that protocol."""
    manifest, _runner, _document = _campaign(tmp_path)
    material = json.loads(Path(manifest.evaluation_material_path).read_text(encoding="utf-8"))
    for suite in material["suites"]:
        if suite["benchmark_qualified_id"] == PROTECTED_ID:
            Path(suite["dataset"]).write_text(
                json.dumps({"prompt": "p", "expected": "e"}) + "\n", encoding="utf-8"
            )
    Path(manifest.evaluation_material_path).write_text(json.dumps(material), encoding="utf-8")

    evaluator = SubprocessEvaluationFn(
        run_root=Path(manifest.state_root),
        material=EvaluationMaterial.load(manifest.evaluation_material_path),
        protocol=manifest.protection.require_protocol(),
        base_model_path=manifest.base_model_path,
        runner=_RecordingWorker(),
    )
    artifact = _artifact(tmp_path)

    with pytest.raises(CandidateEvaluationRefusal) as error:
        evaluator(
            _request(manifest, artifact_ref=str(artifact), artifact_sha256=digest_of(artifact))
        )

    assert CANDIDATE_EVALUATION_SLICE_TOO_SHORT in str(error.value)


def test_an_artifact_whose_bytes_changed_after_selection_refuses(tmp_path: Path):
    """The arm names the digest recomputed from the bytes, not the caller's claim."""
    manifest, _runner, _document = _campaign(tmp_path)
    artifact = _artifact(tmp_path)
    claimed = digest_of(artifact)
    (artifact / "adapter_model.safetensors").write_text("swapped-weights", encoding="utf-8")

    evaluator = SubprocessEvaluationFn(
        run_root=Path(manifest.state_root),
        material=EvaluationMaterial.load(manifest.evaluation_material_path),
        protocol=manifest.protection.require_protocol(),
        base_model_path=manifest.base_model_path,
        runner=_RecordingWorker(),
    )

    with pytest.raises(CandidateEvaluationRefusal) as error:
        evaluator(_request(manifest, artifact_ref=str(artifact), artifact_sha256=claimed))

    assert CANDIDATE_EVALUATION_ARTIFACT_DIGEST_MISMATCH in str(error.value)
    assert claimed in str(error.value)


def test_a_worker_that_failed_is_not_a_measurement(tmp_path: Path):
    manifest, _runner, _document = _campaign(tmp_path)
    artifact = _artifact(tmp_path)
    evaluator = SubprocessEvaluationFn(
        run_root=Path(manifest.state_root),
        material=EvaluationMaterial.load(manifest.evaluation_material_path),
        protocol=manifest.protection.require_protocol(),
        base_model_path=manifest.base_model_path,
        runner=_RecordingWorker(fail=True),
    )

    with pytest.raises(CandidateEvaluationRefusal) as error:
        evaluator(
            _request(manifest, artifact_ref=str(artifact), artifact_sha256=digest_of(artifact))
        )

    assert CANDIDATE_EVALUATION_PROCESS_FAILED in str(error.value)


def test_a_worker_that_did_not_observe_its_generations_is_not_an_arm(tmp_path: Path):
    """The instrument's facts are read, not assumed.

    The campaign's declared target set is the generation-diagnostics
    instrument, and its rates are defined over how each generation ended. A
    worker whose rows carry only the decoded text cannot say whether a
    generation stopped on EOS or ran into the cap, so the row is refused rather
    than diagnosed from a guess.
    """
    manifest, _runner, _document = _campaign(tmp_path)
    artifact = _artifact(tmp_path)
    evaluator = SubprocessEvaluationFn(
        run_root=Path(manifest.state_root),
        material=EvaluationMaterial.load(manifest.evaluation_material_path),
        protocol=manifest.protection.require_protocol(),
        base_model_path=manifest.base_model_path,
        runner=_RecordingWorker(omit_generation_facts=True),
    )

    with pytest.raises(CandidateEvaluationRefusal) as error:
        evaluator(
            _request(manifest, artifact_ref=str(artifact), artifact_sha256=digest_of(artifact))
        )

    assert GENERATION_DIAGNOSTICS_UNMEASURED in str(error.value)
    assert "generated_tokens" in str(error.value)


def test_the_target_row_carries_the_diagnostics_computed_from_its_own_bytes(
    tmp_path: Path,
):
    """T1-T10's evidence, produced by the measurement that took it.

    The declared target set is the generation-diagnostics instrument: its row's
    ``per_prompt`` completions and five aggregate rates are what the frozen
    judge reads. They are recomputed here from the predictions file the row
    itself names, so a row cannot carry diagnostics from bytes other than the
    ones it digested.
    """
    manifest, _runner, _document = _campaign(tmp_path)
    artifact = _artifact(tmp_path)
    evaluator = SubprocessEvaluationFn(
        run_root=Path(manifest.state_root),
        material=EvaluationMaterial.load(manifest.evaluation_material_path),
        protocol=manifest.protection.require_protocol(),
        base_model_path=manifest.base_model_path,
        runner=_RecordingWorker(
            # The target instrument's completions are the answers, so the
            # diagnostics describe generations that actually terminated.
            prediction_of=lambda name, index, row: (
                str(row.get("expected", ""))
                if name == TARGET_ID.split("@", 1)[0]
                else f"wrong answer {index}"
            ),
            score_of=lambda name, index: 1.0 if name == TARGET_ID.split("@", 1)[0] else 0.0,
        ),
    )

    evaluation = evaluator(
        _request(manifest, artifact_ref=str(artifact), artifact_sha256=digest_of(artifact))
    )

    target = next(
        run for run in evaluation.report.runs if run.benchmark_qualified_id == TARGET_ID
    )
    predictions = Path(manifest.state_root) / str(target.raw_artifact_ref)
    items = [
        json.loads(line)
        for line in predictions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    recomputed = GenerationDiagnostics.from_items(
        items,
        max_new_tokens=manifest.protection.require_protocol().decoding["max_new_tokens"],
        seed=manifest.protection.require_protocol().seed,
        source=str(predictions),
    ).to_metadata()

    for key, value in recomputed.items():
        assert target.metadata[key] == value, f"metadata[{key!r}] is not from these bytes"
    # Every generation in this fixture terminated short of the cap, which is
    # what the recorded facts say and what T6/T7 read.
    assert target.metadata["eos_termination_rate"] == 1.0
    assert target.metadata["max_token_cap_rate"] == 0.0
    assert len(target.metadata["per_prompt"]) == target.n_samples
    entry = target.metadata["per_prompt"][0]
    assert entry["prompt"] == items[0]["prompt"]
    assert entry["completion"] == items[0]["prediction"]


def test_a_worker_that_wrote_no_per_item_evidence_is_not_an_arm(tmp_path: Path):
    """No predictions file, no measurement: the row has no bytes to be bound to."""
    manifest, _runner, _document = _campaign(tmp_path)
    artifact = _artifact(tmp_path)
    evaluator = SubprocessEvaluationFn(
        run_root=Path(manifest.state_root),
        material=EvaluationMaterial.load(manifest.evaluation_material_path),
        protocol=manifest.protection.require_protocol(),
        base_model_path=manifest.base_model_path,
        runner=_RecordingWorker(omit_predictions_for=PROTECTED_ID.split("@", 1)[0]),
    )

    with pytest.raises(CandidateEvaluationRefusal) as error:
        evaluator(
            _request(manifest, artifact_ref=str(artifact), artifact_sha256=digest_of(artifact))
        )

    assert CANDIDATE_EVALUATION_EVIDENCE_INVALID in str(error.value)


# --------------------------------------------------------------------------
# through the real CLI: the production evaluator drives the run and the ledger
# --------------------------------------------------------------------------


def test_the_cli_runs_the_production_evaluator_and_records_its_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """No injected seam: the instrument the campaign declared does the work.

    The manifest is driven through ``chowder growth campaign run`` with only the
    process runner replaced by the recording worker, so what promotes here is
    the production evaluator's own measurement of the artifact this run made.
    """
    manifest, trainer, _document = _campaign(tmp_path, with_ancestor=True)
    # 36 seconds on one accelerator: a measured, chargeable evaluation leg that
    # still fits the fixture's declared campaign ceiling. The recording worker
    # scores the target instrument at the ceiling and the slices flat, which is
    # what the declared promotion rule needs to reach "improved".
    worker = _RecordingWorker(
        seconds=36.0,
        score_of=lambda name, index: 1.0
        if name == TARGET_ID.split("@", 1)[0]
        else (1.0 if index % 2 == 0 else 0.0),
    )
    _patch_runner(monkeypatch, _both_processes(trainer, worker))
    monkeypatch.setattr(campaign_runner, "default_evaluator_factory", None)

    run = run_campaign(manifest)

    assert run.verdict == "PROMOTED"
    # The production certification gate ran on the arm this evaluator wrote:
    # its protected slice passed the declared protocol, was bound to the selected
    # artifact, and held against the trusted ancestor.
    assert run.certification["status"] == "PASS", run.certification["reasons"]
    # The evaluation really ran the production worker command line.
    assert any(
        "chowder.evaluators.transformers_text_worker" in " ".join(command)
        for command in worker.commands
    )
    arm = EvalReport.load(Path(manifest.state_root) / "candidate_evaluation.json")
    selected_digest = run.selection["artifact_sha256"]
    assert arm.model_identity["adapter_digest"] == selected_digest
    assert [row.measurement_origin for row in arm.runs] == [MEASURED_THIS_GENERATION] * len(
        arm.runs
    )
    # The per-item evidence the rows declare is in the run root, where the judge
    # will hash it.
    for row in arm.runs:
        assert (Path(manifest.state_root) / row.raw_artifact_ref).is_file()
    # The evaluation's measured cost is in the campaign's durable accounting:
    # two trainer attempts plus the evaluation's own measured wall GPU-hours.
    assert run.cost["wall_gpu_hours"] == pytest.approx(
        2 * campaign_fixture.ATTEMPT_WALL_GPU_HOURS + 36.0 / 3600.0
    )
    ledger = GenerationLedger(Path(manifest.state_root) / "ledger")
    assert ledger.effective_verdict(CANDIDATE_VERSION) == "PROMOTED"


def test_the_cli_refuses_a_run_whose_evaluation_material_is_undeclared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A campaign that cannot measure its candidate refuses before compute."""
    manifest, _runner, document = _campaign(tmp_path)
    document.pop("evaluation_material_path", None)
    manifest_path = Path(document["state_root"]).parent / "inputs" / "campaign.json"
    manifest_path.write_text(json.dumps(document), encoding="utf-8")

    monkeypatch.setattr(
        sys, "argv", ["chowder", "growth", "campaign", "run", str(manifest_path)]
    )
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = chowder_main()

    assert code != 0
    assert "evaluation_material_path" in buffer.getvalue()


def _both_processes(trainer: Any, worker: Any) -> Any:
    """Route the trainer's commands to the trainer, the worker's to the worker.

    One campaign starts two different child processes through one runner seam;
    a real host runs both, and the recording host has to tell them apart by
    their command line rather than by which phase is running.
    """

    def runner(
        command: Sequence[str],
        cwd: Path,
        extra_environment: Mapping[str, str],
        timeout_seconds: float | None,
    ) -> SubprocessOutcome:
        if "chowder.evaluators.transformers_text_worker" in command:
            return worker(command, cwd, extra_environment, timeout_seconds)
        return trainer(command, cwd, extra_environment, timeout_seconds)

    return runner
