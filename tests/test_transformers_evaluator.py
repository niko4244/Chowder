import json
from pathlib import Path

import pytest

from chowder.canonical_chat_template import canonical_template_sha256
from chowder.contamination import write_holdout_fingerprint_index
from chowder.evaluators.transformers_text import TransformersTextEvaluator, TransformersTextEvalSpec
from chowder.evaluators.transformers_text_worker import _score
from chowder.executors import ExecutionContext, TrainingArtifact
from chowder.memory import HardwareProfile
from chowder.models import Experiment, Hypothesis
from chowder.provenance import sha256_directory


def _hardware():
    return HardwareProfile(16, 64, 500, 12, 40, 3)


def _experiment():
    return Experiment("e1", None, Hypothesis("obs", "cause", "fix"), {}, 1.0)


def _config(dataset: str):
    return {
        "seed": 11,
        "backend": {
            "type": "transformers-peft",
            "base_model": "example/model",
            "revision": "requested-rev",
            "quantization": "4bit",
            "precision": "bf16",
        },
        "evaluation": {
            "type": "transformers-text",
            "quantization": "inherit",
            "precision": "inherit",
            "suites": [
                {
                    "name": "quality",
                    "dataset": dataset,
                    "scoring": "normalized_exact_match",
                    "max_new_tokens": 16,
                }
            ],
        },
    }


def _artifact(tmp_path, *, resolved_commit="resolved123"):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    return TrainingArtifact(
        run_id="run-1",
        experiment_id="e1",
        artifact_ref=str(adapter),
        gpu_hours=0.25,
        evidence={
            "artifact_sha256": sha256_directory(adapter),
            "model_provenance": {"resolved_model_commit": resolved_commit},
        },
    )


def _fake_payload(
    result_path: Path,
    *,
    metric: float,
    device: str,
    gpu_count: int,
    rendering: str | None = "raw",
    chat_template_sha256: str | None = None,
    runtime: dict | None = None,
):
    fingerprint_path = result_path.parent / "holdout-fingerprints-quality.jsonl"
    fingerprint_digest = write_holdout_fingerprint_index([("2+2?", "4")], fingerprint_path)
    suite: dict = {
        "rows": 1,
        "scoring": "normalized_exact_match",
        "holdout_fingerprints_file": str(fingerprint_path),
        "holdout_fingerprints_sha256": fingerprint_digest,
    }
    if rendering is not None:
        suite["rendering"] = rendering
    if chat_template_sha256 is not None:
        suite["chat_template_sha256"] = chat_template_sha256
    runtime_payload: dict = {"device": device, "gpu_count": gpu_count}
    if runtime is not None:
        runtime_payload.update(runtime)
    return {
        "metrics": {"quality": metric},
        "suites": {"quality": suite},
        "runtime": runtime_payload,
        "versions": {"transformers": "5.test"},
    }


def _fake_process(**payload_kwargs):
    """A worker process that writes a payload with the given rendering evidence."""

    class FakeProcess:
        returncode = 0

        def __init__(self, command, **kwargs):
            result_path = Path(command[command.index("--result") + 1])
            result_path.write_text(
                json.dumps(
                    _fake_payload(
                        result_path,
                        metric=0.5,
                        device="cuda:0",
                        gpu_count=1,
                        **payload_kwargs,
                    )
                )
            )

        def wait(self, timeout=None):
            return 0

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    return FakeProcess


def _chat_template_config(dataset: str, *, canonical: bool):
    config = _config(dataset)
    config["evaluation"]["suites"][0]["use_chat_template"] = True
    if canonical:
        config["evaluation"]["suites"][0]["canonical_rendering"] = True
    return config


def test_eval_spec_pins_resolved_training_commit_and_inherits_runtime(tmp_path):
    data = tmp_path / "eval.jsonl"
    data.write_text('{"prompt":"2+2?","expected":"4"}\n')
    artifact = _artifact(tmp_path)
    spec = TransformersTextEvalSpec.from_context(
        config=_config("eval.jsonl"),
        artifact=artifact,
        work_dir=tmp_path,
        output_dir=tmp_path / "out",
        seed=1,
    )
    assert spec.revision == "resolved123"
    assert spec.quantization == "4bit"
    assert spec.precision == "bf16"
    assert spec.seed == 11
    assert spec.device == "auto"
    assert spec.suites[0].name == "quality"
    assert spec.suites[0].dataset == str(data.resolve())


def test_eval_spec_offline_defaults_to_false(tmp_path):
    data = tmp_path / "eval.jsonl"
    data.write_text('{"prompt":"2+2?","expected":"4"}\n')
    artifact = _artifact(tmp_path)
    spec = TransformersTextEvalSpec.from_context(
        config=_config("eval.jsonl"), artifact=artifact, work_dir=tmp_path, output_dir=tmp_path / "out", seed=1
    )
    assert spec.offline is False


def test_eval_spec_offline_inherits_from_backend_when_unset_on_evaluation(tmp_path):
    data = tmp_path / "eval.jsonl"
    data.write_text('{"prompt":"2+2?","expected":"4"}\n')
    artifact = _artifact(tmp_path)
    config = _config("eval.jsonl")
    config["backend"]["offline"] = True
    spec = TransformersTextEvalSpec.from_context(
        config=config, artifact=artifact, work_dir=tmp_path, output_dir=tmp_path / "out", seed=1
    )
    assert spec.offline is True


def test_eval_spec_offline_on_evaluation_overrides_backend(tmp_path):
    data = tmp_path / "eval.jsonl"
    data.write_text('{"prompt":"2+2?","expected":"4"}\n')
    artifact = _artifact(tmp_path)
    config = _config("eval.jsonl")
    config["backend"]["offline"] = True
    config["evaluation"]["offline"] = False
    spec = TransformersTextEvalSpec.from_context(
        config=config, artifact=artifact, work_dir=tmp_path, output_dir=tmp_path / "out", seed=1
    )
    assert spec.offline is False


def test_evaluator_refuses_mutated_adapter(tmp_path):
    data = tmp_path / "eval.jsonl"
    data.write_text('{"prompt":"2+2?","expected":"4"}\n')
    artifact = _artifact(tmp_path)
    Path(artifact.artifact_ref, "adapter_model.safetensors").write_bytes(b"tampered")
    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=_config("eval.jsonl"))
    with pytest.raises(ValueError, match="content digest changed"):
        TransformersTextEvaluator().evaluate(
            experiment=_experiment(), artifact=artifact, context=context
        )


def test_evaluator_returns_named_metrics_and_verified_holdout_index(tmp_path, monkeypatch):
    data = tmp_path / "eval.jsonl"
    data.write_text('{"prompt":"2+2?","expected":"4"}\n')
    artifact = _artifact(tmp_path)
    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=_config("eval.jsonl"))

    class FakeProcess:
        returncode = 0

        def __init__(self, command, **kwargs):
            result_path = Path(command[command.index("--result") + 1])
            result_path.write_text(json.dumps(_fake_payload(
                result_path, metric=0.75, device="cuda:0", gpu_count=1
            )))

        def wait(self, timeout=None):
            return 0

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr("chowder.evaluators.transformers_text.subprocess.Popen", FakeProcess)
    result = TransformersTextEvaluator().evaluate(
        experiment=_experiment(), artifact=artifact, context=context
    )
    assert result.metrics == {"quality": 0.75}
    assert result.source_artifact_ref == artifact.artifact_ref
    assert result.gpu_hours >= 0
    assert result.gpu_hours < artifact.gpu_hours
    assert result.evidence["artifact_sha256"] == artifact.evidence["artifact_sha256"]
    assert len(result.evidence["evaluation_dataset_sha256"]["quality"]) == 64
    assert len(result.evidence["holdout_fingerprint_sha256"]["quality"]) == 64


def test_evaluator_rejects_tampered_holdout_fingerprint_index(tmp_path, monkeypatch):
    data = tmp_path / "eval.jsonl"
    data.write_text('{"prompt":"2+2?","expected":"4"}\n')
    artifact = _artifact(tmp_path)
    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=_config("eval.jsonl"))

    class FakeProcess:
        returncode = 0

        def __init__(self, command, **kwargs):
            result_path = Path(command[command.index("--result") + 1])
            payload = _fake_payload(result_path, metric=1.0, device="cpu", gpu_count=0)
            fingerprint_path = Path(payload["suites"]["quality"]["holdout_fingerprints_file"])
            fingerprint_path.write_text("tampered\n", encoding="utf-8")
            result_path.write_text(json.dumps(payload))

        def wait(self, timeout=None):
            return 0

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr("chowder.evaluators.transformers_text.subprocess.Popen", FakeProcess)
    with pytest.raises(RuntimeError, match="digest mismatch"):
        TransformersTextEvaluator().evaluate(
            experiment=_experiment(), artifact=artifact, context=context
        )


def test_worker_scoring_is_deterministic_and_explicit():
    assert _score(" Answer  42 ", "answer 42", "normalized_exact_match") == 1.0
    assert _score("Answer", "answer", "exact_match") == 0.0


def test_cpu_evaluation_does_not_consume_gpu_hour_budget(tmp_path, monkeypatch):
    data = tmp_path / "eval.jsonl"
    data.write_text('{"prompt":"2+2?","expected":"4"}\n')
    artifact = _artifact(tmp_path)
    config = _config("eval.jsonl")
    config["evaluation"]["device"] = "cpu"
    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=config)

    class FakeProcess:
        returncode = 0

        def __init__(self, command, **kwargs):
            result_path = Path(command[command.index("--result") + 1])
            result_path.write_text(json.dumps(_fake_payload(
                result_path, metric=1.0, device="cpu", gpu_count=0
            )))

        def wait(self, timeout=None):
            return 0

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr("chowder.evaluators.transformers_text.subprocess.Popen", FakeProcess)
    result = TransformersTextEvaluator().evaluate(
        experiment=_experiment(), artifact=artifact, context=context
    )
    assert result.gpu_hours == 0.0
    assert result.evidence["runtime"]["device"] == "cpu"


def test_evaluator_profile_reads_declared_estimated_gpu_hours(tmp_path):
    context = ExecutionContext(
        _hardware(),
        str(tmp_path),
        1,
        resolved_config={"evaluation": {"estimated_gpu_hours": 0.3}},
    )
    estimate = TransformersTextEvaluator().profile(_experiment(), context)
    assert estimate.gpu_hours == pytest.approx(0.3)
    assert estimate.confidence == 0.25


def test_evaluator_profile_defaults_to_zero_when_unset(tmp_path):
    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config={})
    estimate = TransformersTextEvaluator().profile(_experiment(), context)
    assert estimate.gpu_hours == 0.0


def test_evaluator_cancel_terminates_a_tracked_running_process():
    class FakeRunningProcess:
        def __init__(self):
            self.terminated = False
            self.waited = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            self.waited = True

    evaluator = TransformersTextEvaluator()
    process = FakeRunningProcess()
    evaluator._processes["run-1"] = process
    evaluator.cancel("run-1")
    assert process.terminated
    assert process.waited


# --- P4: the rendered template is bound into the per-suite protocol entry ----


def _evaluate_with_fake_worker(config, tmp_path, monkeypatch, **payload_kwargs):
    tmp_path.mkdir(parents=True, exist_ok=True)
    data = tmp_path / "eval.jsonl"
    data.write_text('{"prompt":"2+2?","expected":"4"}\n')
    artifact = _artifact(tmp_path)
    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=config)
    monkeypatch.setattr(
        "chowder.evaluators.transformers_text.subprocess.Popen",
        _fake_process(**payload_kwargs),
    )
    return TransformersTextEvaluator().evaluate(
        experiment=_experiment(), artifact=artifact, context=context
    )


def test_protocol_binds_the_template_the_worker_actually_rendered(tmp_path, monkeypatch):
    digest = "d" * 64
    result = _evaluate_with_fake_worker(
        _chat_template_config("eval.jsonl", canonical=False),
        tmp_path,
        monkeypatch,
        rendering="tokenizer-template",
        chat_template_sha256=digest,
    )
    suite_entry = result.evidence["protocol"]["suites"][0]
    assert suite_entry["rendering"] == "tokenizer-template"
    assert suite_entry["chat_template_sha256"] == digest
    assert result.evidence["suite_evidence"]["quality"]["rendering"] == "tokenizer-template"


def test_protocol_binds_the_pinned_digest_for_canonical_suites(tmp_path, monkeypatch):
    result = _evaluate_with_fake_worker(
        _chat_template_config("eval.jsonl", canonical=True),
        tmp_path,
        monkeypatch,
        rendering="canonical-template",
        chat_template_sha256=canonical_template_sha256(),
    )
    suite_entry = result.evidence["protocol"]["suites"][0]
    assert suite_entry["canonical_rendering"] is True
    assert suite_entry["rendering"] == "canonical-template"
    assert suite_entry["chat_template_sha256"] == canonical_template_sha256()


def test_protocol_refuses_a_rendering_the_spec_did_not_ask_for(tmp_path, monkeypatch):
    """The silent-fallback case: the suite asked for canonical rendering and
    the worker used the tokenizer's own template. The prompt bytes differ, so
    this cannot be scored as the same protocol."""
    with pytest.raises(RuntimeError, match="canonical-template"):
        _evaluate_with_fake_worker(
            _chat_template_config("eval.jsonl", canonical=True),
            tmp_path,
            monkeypatch,
            rendering="tokenizer-template",
            chat_template_sha256="d" * 64,
        )


def test_protocol_refuses_a_canonical_digest_that_is_not_the_pinned_one(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="canonical chat template"):
        _evaluate_with_fake_worker(
            _chat_template_config("eval.jsonl", canonical=True),
            tmp_path,
            monkeypatch,
            rendering="canonical-template",
            chat_template_sha256="0" * 64,
        )


def test_protocol_refuses_a_worker_that_reports_no_rendering(tmp_path, monkeypatch):
    """Fail closed on missing evidence rather than assuming raw prompt bytes."""
    with pytest.raises(RuntimeError, match="rendering"):
        _evaluate_with_fake_worker(
            _config("eval.jsonl"), tmp_path, monkeypatch, rendering=None
        )


def test_a_different_rendered_template_changes_the_protocol_fingerprint(tmp_path, monkeypatch):
    """The headline claim: swapping the template changes the identity, so a
    score produced under one rendering can never be compared with a score
    produced under another by accident."""
    config = _chat_template_config("eval.jsonl", canonical=False)
    first = _evaluate_with_fake_worker(
        config,
        tmp_path / "a",
        monkeypatch,
        rendering="tokenizer-template",
        chat_template_sha256="a" * 64,
    )
    second = _evaluate_with_fake_worker(
        config,
        tmp_path / "b",
        monkeypatch,
        rendering="tokenizer-template",
        chat_template_sha256="b" * 64,
    )
    assert first.evidence["protocol_sha256"] != second.evidence["protocol_sha256"]


def test_protocol_binds_raw_rendering_without_a_template_digest(tmp_path, monkeypatch):
    result = _evaluate_with_fake_worker(_config("eval.jsonl"), tmp_path, monkeypatch)
    suite_entry = result.evidence["protocol"]["suites"][0]
    assert suite_entry["rendering"] == "raw"
    assert "chat_template_sha256" not in suite_entry


# --- P6: each evaluation arm reports its own measured lifecycle -------------


def _lifecycle_runtime(arm_phase: str) -> dict:
    return {
        "lifecycle": {
            "accelerator_count": 1,
            "phases": {
                arm_phase: {
                    "phase": arm_phase,
                    "seconds": 120.0,
                    "measured": True,
                    "accelerator_count": 1,
                    "synchronized": True,
                    "sync_overhead_seconds": 0.01,
                    "note": None,
                }
            },
            "unmeasured": {
                "model_load": "the evaluator did not time the model load",
            },
        },
        "memory_sampling": {
            "samples": 10,
            "cadence_seconds": 0.5,
            "span_seconds": 5.0,
            "unavailable_fields": [],
        },
    }


def test_evaluator_evidence_records_the_arm_lifecycle_and_sampling(tmp_path, monkeypatch):
    result = _evaluate_with_fake_worker(
        _config("eval.jsonl"),
        tmp_path,
        monkeypatch,
        runtime=_lifecycle_runtime("candidate_generation"),
    )
    lifecycle = result.evidence["lifecycle"]

    assert lifecycle["state"] == "measured"
    ledger = lifecycle["phase_ledger"]
    assert ledger["phases"]["candidate_generation"]["seconds"] == pytest.approx(120.0)
    # The other arm is a separate process, and the ledger says so rather than
    # inheriting a number this arm never saw.
    assert "model_load" in ledger["unmeasured"]
    assert lifecycle["memory_sampling"]["cadence_seconds"] == pytest.approx(0.5)


def test_evaluator_without_a_lifecycle_leaves_unknown_not_zero(tmp_path, monkeypatch):
    result = _evaluate_with_fake_worker(_config("eval.jsonl"), tmp_path, monkeypatch)
    lifecycle = result.evidence["lifecycle"]

    assert lifecycle["phase_ledger"] is None
    assert lifecycle["state"] == "unknown"
    assert "did not report" in lifecycle["reason"]


def test_evaluator_refuses_a_malformed_lifecycle(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="lifecycle"):
        _evaluate_with_fake_worker(
            _config("eval.jsonl"),
            tmp_path,
            monkeypatch,
            runtime={"lifecycle": {"phases": "not a mapping"}},
        )


def test_evaluator_cancel_is_a_no_op_for_unknown_or_finished_run():
    evaluator = TransformersTextEvaluator()
    evaluator.cancel("never-started")

    class FinishedProcess:
        def poll(self):
            return 0

        def terminate(self):
            raise AssertionError("must not terminate an already-finished process")

    evaluator._processes["run-2"] = FinishedProcess()
    evaluator.cancel("run-2")
