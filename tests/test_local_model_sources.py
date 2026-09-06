from pathlib import Path

from chowder.hf_resilience import (
    cache_status,
    is_local_model_source,
    resolve_model_source,
)


def test_resolve_model_source_prefers_existing_absolute_directory(tmp_path: Path):
    model_dir = tmp_path / "qwen-local"
    model_dir.mkdir()

    resolved = resolve_model_source(str(model_dir))

    assert resolved == str(model_dir.resolve())
    assert is_local_model_source(resolved) is True


def test_resolve_model_source_resolves_relative_directory_against_work_dir(tmp_path: Path):
    model_dir = tmp_path / "models" / "qwen-local"
    model_dir.mkdir(parents=True)

    resolved = resolve_model_source("models/qwen-local", work_dir=tmp_path)

    assert resolved == str(model_dir.resolve())


def test_resolve_model_source_leaves_hub_id_unchanged_when_no_local_match(tmp_path: Path):
    source = "Qwen/example-model"

    assert resolve_model_source(source, work_dir=tmp_path) == source
    assert is_local_model_source(source) is False


def test_cache_status_short_circuits_hub_for_local_model(tmp_path: Path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    # If cache_status accidentally touches the Hub for a local model this
    # import/call path should fail the test immediately.
    def fail_import(*args, **kwargs):
        raise AssertionError("local model must not be sent through Hub cache lookup")

    monkeypatch.setattr(
        "chowder.hf_resilience.is_local_model_source",
        lambda source: str(source) == str(model_dir),
    )

    assert cache_status(str(model_dir), None) == "local"
