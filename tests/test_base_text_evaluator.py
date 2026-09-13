import json
from pathlib import Path

import pytest

from chowder.canonical_chat_template import canonical_template_sha256
from chowder.contamination import write_holdout_fingerprint_index
from chowder.evaluators.base_text import BaseModelTextEvaluator, BaseTextEvalSpec
from chowder.executors import ExecutionContext
from chowder.memory import HardwareProfile


def _config(dataset: str):
    return {
        "backend": {
            "base_model": "example/model",
        },
        "evaluation": {
            "type": "transformers-text",
            "suites": [
                {
                    "name": "quality",
                    "dataset": dataset,
                }
            ],
        },
    }


# --- P4: the baseline arm binds the rendering the same way the candidate does --


def _hardware():
    return HardwareProfile(16, 64, 500, 12, 40, 3)


def _fake_process(rendering="raw", chat_template_sha256=None):
    class FakeProcess:
        returncode = 0

        def __init__(self, command, **kwargs):
            result_path = Path(command[command.index("--result") + 1])
            fingerprint_path = result_path.parent / "holdout-fingerprints-quality.jsonl"
            digest = write_holdout_fingerprint_index([("x", "y")], fingerprint_path)
            suite = {
                "rows": 1,
                "scoring": "normalized_exact_match",
                "holdout_fingerprints_file": str(fingerprint_path),
                "holdout_fingerprints_sha256": digest,
            }
            if rendering is not None:
                suite["rendering"] = rendering
            if chat_template_sha256 is not None:
                suite["chat_template_sha256"] = chat_template_sha256
            result_path.write_text(
                json.dumps(
                    {
                        "metrics": {"quality": 0.5},
                        "suites": {"quality": suite},
                        "runtime": {"device": "cpu", "gpu_count": 0},
                        "versions": {"transformers": "5.test"},
                        "model_provenance": {},
                    }
                )
            )

        def wait(self, timeout=None):
            return 0

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    return FakeProcess


def _evaluate(config, tmp_path, monkeypatch, **payload_kwargs):
    data = tmp_path / "eval.jsonl"
    data.write_text('{"prompt":"x","expected":"y"}\n')
    context = ExecutionContext(_hardware(), str(tmp_path), 1, resolved_config=config)
    monkeypatch.setattr(
        "chowder.evaluators.base_text.subprocess.Popen", _fake_process(**payload_kwargs)
    )
    return BaseModelTextEvaluator().evaluate(config=config, context=context)


def _chat_config(dataset: str, *, canonical: bool):
    config = _config(dataset)
    config["evaluation"]["suites"][0]["use_chat_template"] = True
    if canonical:
        config["evaluation"]["suites"][0]["canonical_rendering"] = True
    return config


def _with_dataset(config, tmp_path):
    config["evaluation"]["suites"][0]["dataset"] = str(tmp_path / "eval.jsonl")
    return config


def test_baseline_protocol_binds_raw_rendering(tmp_path, monkeypatch):
    config = _with_dataset(_config("eval.jsonl"), tmp_path)
    result = _evaluate(config, tmp_path, monkeypatch)
    suite_entry = result.evidence["protocol"]["suites"][0]
    assert suite_entry["rendering"] == "raw"
    assert "chat_template_sha256" not in suite_entry


def test_baseline_protocol_binds_the_canonical_digest(tmp_path, monkeypatch):
    config = _with_dataset(_chat_config("eval.jsonl", canonical=True), tmp_path)
    result = _evaluate(
        config,
        tmp_path,
        monkeypatch,
        rendering="canonical-template",
        chat_template_sha256=canonical_template_sha256(),
    )
    suite_entry = result.evidence["protocol"]["suites"][0]
    assert suite_entry["rendering"] == "canonical-template"
    assert suite_entry["chat_template_sha256"] == canonical_template_sha256()


def test_baseline_protocol_refuses_a_silent_renderer_fallback(tmp_path, monkeypatch):
    config = _with_dataset(_chat_config("eval.jsonl", canonical=True), tmp_path)
    with pytest.raises(RuntimeError, match="canonical-template"):
        _evaluate(
            config,
            tmp_path,
            monkeypatch,
            rendering="tokenizer-template",
            chat_template_sha256="d" * 64,
        )


def test_offline_defaults_to_false(tmp_path):
    data = tmp_path / "eval.jsonl"
    data.write_text('{"prompt":"x","expected":"y"}\n')
    spec = BaseTextEvalSpec.from_config(
        _config(str(data)), work_dir=tmp_path, output_dir=tmp_path / "out", seed=1
    )
    assert spec.offline is False


def test_offline_inherits_from_backend_when_unset_on_evaluation(tmp_path):
    data = tmp_path / "eval.jsonl"
    data.write_text('{"prompt":"x","expected":"y"}\n')
    config = _config(str(data))
    config["backend"]["offline"] = True
    spec = BaseTextEvalSpec.from_config(
        config, work_dir=tmp_path, output_dir=tmp_path / "out", seed=1
    )
    assert spec.offline is True


def test_offline_on_evaluation_overrides_backend(tmp_path):
    data = tmp_path / "eval.jsonl"
    data.write_text('{"prompt":"x","expected":"y"}\n')
    config = _config(str(data))
    config["backend"]["offline"] = True
    config["evaluation"]["offline"] = False
    spec = BaseTextEvalSpec.from_config(
        config, work_dir=tmp_path, output_dir=tmp_path / "out", seed=1
    )
    assert spec.offline is False
