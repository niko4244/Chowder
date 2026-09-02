from pathlib import Path

from chowder.evaluators.base_text import BaseTextEvalSpec


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
