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


def _fake_process(rendering="raw", chat_template_sha256=None, runtime_extra=None):
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
            runtime = {"device": "cpu", "gpu_count": 0}
            if runtime_extra is not None:
                runtime.update(runtime_extra)
            result_path.write_text(
                json.dumps(
                    {
                        "metrics": {"quality": 0.5},
                        "suites": {"quality": suite},
                        "runtime": runtime,
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


def test_baseline_evidence_records_its_own_generation_lifecycle(tmp_path, monkeypatch):
    result = _evaluate(
        _config("eval.jsonl"),
        tmp_path,
        monkeypatch,
        runtime_extra={
            "lifecycle": {
                "accelerator_count": 0,
                "phases": {
                    "baseline_generation": {
                        "phase": "baseline_generation",
                        "seconds": 60.0,
                        "measured": True,
                        "accelerator_count": 0,
                        "sync_overhead_seconds": 0.0,
                    }
                },
                "unmeasured": {
                    "candidate_generation": "the candidate arm runs in a separate worker process"
                },
            }
        },
    )
    lifecycle = result.evidence["lifecycle"]

    assert lifecycle["state"] == "measured"
    assert lifecycle["phase_ledger"]["phases"]["baseline_generation"]["seconds"] == pytest.approx(
        60.0
    )
    assert "candidate_generation" in lifecycle["phase_ledger"]["unmeasured"]


def test_baseline_without_a_lifecycle_reports_unknown(tmp_path, monkeypatch):
    result = _evaluate(_config("eval.jsonl"), tmp_path, monkeypatch)
    assert result.evidence["lifecycle"]["phase_ledger"] is None
    assert result.evidence["lifecycle"]["state"] == "unknown"


def test_baseline_refuses_a_malformed_lifecycle(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="lifecycle"):
        _evaluate(
            _config("eval.jsonl"),
            tmp_path,
            monkeypatch,
            runtime_extra={"lifecycle": {"phases": "not a mapping"}},
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


# --- evaluation placement: resident vs offload (dense-model policy) ----------


def test_placement_defaults_to_resident(tmp_path):
    data = tmp_path / "eval.jsonl"
    data.write_text('{"prompt":"x","expected":"y"}\n')
    spec = BaseTextEvalSpec.from_config(
        _config(str(data)), work_dir=tmp_path, output_dir=tmp_path / "out", seed=1
    )
    assert spec.placement == "resident"


def test_placement_parses_from_evaluation_config(tmp_path):
    data = tmp_path / "eval.jsonl"
    data.write_text('{"prompt":"x","expected":"y"}\n')
    config = _config(str(data))
    config["evaluation"]["placement"] = "offload"
    spec = BaseTextEvalSpec.from_config(
        config, work_dir=tmp_path, output_dir=tmp_path / "out", seed=1
    )
    assert spec.placement == "offload"


def test_placement_refuses_an_unknown_mode(tmp_path):
    data = tmp_path / "eval.jsonl"
    data.write_text('{"prompt":"x","expected":"y"}\n')
    config = _config(str(data))
    config["evaluation"]["placement"] = "spread-across-the-house"
    with pytest.raises(ValueError, match="placement"):
        BaseTextEvalSpec.from_config(
            config, work_dir=tmp_path, output_dir=tmp_path / "out", seed=1
        )


def test_offload_placement_is_carried_by_the_protocol_fingerprint(tmp_path, monkeypatch):
    data = tmp_path / "eval.jsonl"
    data.write_text('{"prompt":"x","expected":"y"}\n')
    config = _config(str(data))
    config["evaluation"]["placement"] = "offload"
    config["evaluation"]["device"] = "cpu"
    outcome = _evaluate(config, tmp_path, monkeypatch, runtime_extra={"placement": "offload"})
    assert outcome.evidence["protocol"]["placement"] == "offload"
    assert outcome.evidence["runtime"]["placement"] == "offload"
    # And a resident protocol must hash differently: placement is protocol.
    resident = _evaluate(_config(str(data)), tmp_path, monkeypatch)
    assert resident.evidence["protocol_sha256"] != outcome.evidence["protocol_sha256"]


# --- parent-adapter continuations baseline against the re-measured parent ---


def test_no_parent_adapter_defaults_to_none(tmp_path):
    data = tmp_path / "eval.jsonl"
    data.write_text('{"prompt":"x","expected":"y"}\n')
    spec = BaseTextEvalSpec.from_config(
        _config(str(data)), work_dir=tmp_path, output_dir=tmp_path / "out", seed=1
    )
    assert spec.adapter_dir is None


def test_parent_adapter_path_is_resolved_for_the_baseline(tmp_path):
    data = tmp_path / "eval.jsonl"
    data.write_text('{"prompt":"x","expected":"y"}\n')
    config = _config(str(data))
    config["backend"]["parent_adapter"] = {
        "path": str(tmp_path / "parent-adapter"),
        "sha256": "a" * 64,
    }
    spec = BaseTextEvalSpec.from_config(
        config, work_dir=tmp_path, output_dir=tmp_path / "out", seed=1
    )
    assert spec.adapter_dir == str((tmp_path / "parent-adapter").resolve())


def test_parent_adapter_without_a_sha_is_refused(tmp_path):
    data = tmp_path / "eval.jsonl"
    data.write_text('{"prompt":"x","expected":"y"}\n')
    config = _config(str(data))
    config["backend"]["parent_adapter"] = {"path": str(tmp_path / "parent-adapter")}
    with pytest.raises(ValueError, match="sha256"):
        BaseTextEvalSpec.from_config(
            config, work_dir=tmp_path, output_dir=tmp_path / "out", seed=1
        )


def test_empty_parent_adapter_path_is_refused(tmp_path):
    data = tmp_path / "eval.jsonl"
    data.write_text('{"prompt":"x","expected":"y"}\n')
    config = _config(str(data))
    config["backend"]["parent_adapter"] = {"path": "  ", "sha256": "a" * 64}
    with pytest.raises(ValueError, match="non-empty"):
        BaseTextEvalSpec.from_config(
            config, work_dir=tmp_path, output_dir=tmp_path / "out", seed=1
        )


def test_adapter_dir_is_not_protocol_but_is_spec_bound(tmp_path, monkeypatch):
    # The adapter is the treatment being measured, not the protocol: the
    # baseline and candidate protocols must stay comparable (gate.py's
    # require_protocol_match compares them), so adapter_dir is excluded from
    # the protocol dict. It is still bound by the spec digest in evidence.
    data = tmp_path / "eval.jsonl"
    data.write_text('{"prompt":"x","expected":"y"}\n')
    config = _config(str(data))
    config["backend"]["parent_adapter"] = {
        "path": str(tmp_path / "parent-adapter"),
        "sha256": "a" * 64,
    }
    with_parent = _evaluate(config, tmp_path, monkeypatch)
    without = _evaluate(_config(str(data)), tmp_path, monkeypatch)
    assert with_parent.evidence["protocol_sha256"] == without.evidence["protocol_sha256"]
    spec_digest = with_parent.evidence["evaluation_spec_sha256"]
    assert spec_digest != without.evidence["evaluation_spec_sha256"]
