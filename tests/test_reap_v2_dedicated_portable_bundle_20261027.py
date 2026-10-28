"""Portable frozen-source bundler adversarial tests, no model execution."""

import io
import json
from pathlib import Path
import zipfile

import pytest

from chowder_batch import reap_v2_dedicated_portable_bundle_20261027 as bundle


def mock_sources() -> dict[str, bytes]:
    # Test-only substitute files are not evidence, weight bytes or benchmark scores.
    return {name: (bundle.REPO / name).read_bytes() if name in bundle.PINNED_ORIGINAL_CODE
            else ("synthetic adversarial fixture for " + name).encode("utf-8")
            for name in bundle.SOURCES}


def test_source_locked_original_scorer_and_eval_bytes():
    for name, digest in bundle.PINNED_ORIGINAL_CODE.items():
        original = (bundle.REPO / name).read_bytes()
        assert bundle.sha(original) == digest


def test_reproducible_portable_archive_has_no_models_or_secret_paths():
    payload = mock_sources()
    record = bundle.manifest(payload, "a" * 40)
    blob1 = bundle.deterministic_bytes(payload, record)
    blob2 = bundle.deterministic_bytes(payload, record)
    assert blob1 == blob2
    assert bundle.verify_bytes(blob1) == record
    bundle.verify_runtime(record)
    assert record["gen0"]["chosen"] is None
    assert record["model_gguf_bytes_embedded"] == 0
    assert record["dataset_rows_or_judge_codes_embedded"] == 0
    assert record["native_server_binary_embedded"] is False
    with zipfile.ZipFile(io.BytesIO(blob1)) as archive:
        assert sorted(archive.namelist()) == sorted([*bundle.SOURCES, bundle.METADATA_NAME])
        assert all(info.file_size < 4 * 1024 * 1024 for info in archive.infolist())


def test_changed_frozen_original_evaluator_refused():
    data = mock_sources()
    data["chowder_batch/reap_eval.py"] += b"\n# corrupted original evaluator\n"
    with pytest.raises(bundle.BundleRefused, match="frozen original scorer"):
        bundle.manifest(data, "a" * 40)


def test_modified_source_and_old_manifest_cannot_pass_archive_verify():
    data = mock_sources()
    record = bundle.manifest(data, "a" * 40)
    changed = dict(data)
    changed["chowder_batch/reap_v2_frozen_wave_20261027.py"] += b"\n# edited"
    with pytest.raises(bundle.BundleRefused, match="manifest"):
        bundle.deterministic_bytes(changed, record)


def test_unapproved_extra_or_missing_sources_refused():
    data = mock_sources()
    data["credential-private.txt"] = b"NEVER INCLUDE CREDENTIALS"
    with pytest.raises(bundle.BundleRefused, match="inventory"):
        bundle.manifest(data, "a" * 40)
    data.pop("credential-private.txt")
    data.pop("chowder_batch/reap_eval.py")
    with pytest.raises(bundle.BundleRefused, match="inventory"):
        bundle.manifest(data, "a" * 40)


def test_portable_archive_is_exclusive_and_never_overwrites(tmp_path: Path):
    data = mock_sources()
    blob = bundle.deterministic_bytes(data, bundle.manifest(data, "a" * 40))
    target = tmp_path / "frozen-code-only.zip"
    bundle.publish(target, blob)
    assert bundle.verify_bytes(target.read_bytes())["no_model_execution_occurred"]
    with pytest.raises(bundle.BundleRefused, match="new"):
        bundle.publish(target, blob)


def test_tampered_portable_manifest_is_rejected_even_if_zip_is_valid():
    data = mock_sources()
    record = bundle.manifest(data, "a" * 40)
    body = bundle.deterministic_bytes(data, record)
    rebuilt = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(body)) as source, zipfile.ZipFile(rebuilt, "w") as target:
        for name in source.namelist():
            value = source.read(name)
            if name == bundle.METADATA_NAME:
                old = json.loads(value)
                old["gen0"] = {"status": "SELECTED", "chosen": "reap128_v2",
                               "release_approved": True}
                value = json.dumps(old).encode()
            target.writestr(name, value)
    with pytest.raises(bundle.BundleRefused, match="integrity or release block"):
        bundle.verify_bytes(rebuilt.getvalue())


def test_different_or_missing_dedicated_runtime_dependency_refuses():
    record = bundle.manifest(mock_sources(), "a" * 40)
    record["required_runtime_packages_exact_current_builder_versions"]["datasets"] = "0.0.0"
    with pytest.raises(bundle.BundleRefused, match="dependency version differs"):
        bundle.verify_runtime(record)
    record.pop("required_runtime_packages_exact_current_builder_versions")
    with pytest.raises(bundle.BundleRefused, match="requirements incomplete"):
        bundle.verify_runtime(record)


def test_no_uncommitted_or_untracked_file_can_enter_portable_bundle(monkeypatch):
    name = bundle.SOURCES[0]
    monkeypatch.setattr(bundle, "_git_blob", lambda *_a, **_k: b"mismatched git content")
    with pytest.raises(bundle.BundleRefused, match="uncommitted original source"):
        bundle.committed_payloads()


def test_actual_git_windows_crlf_normalization_preserves_original_frozen_sha():
    payload, head = bundle.committed_payloads()
    assert len(head) == 40
    assert len(payload) == len(bundle.SOURCES)
    assert payload["chowder_batch/reap_eval.py"].count(b"\r\n") > 500
    for name, digest in bundle.PINNED_ORIGINAL_CODE.items():
        assert bundle.sha(payload[name]) == digest
    assert bundle.verify_bytes(bundle.deterministic_bytes(
        payload, bundle.manifest(payload, head))) ["gen0"]["status"] == "BLOCKED"
