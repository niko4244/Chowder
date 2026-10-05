"""The declared-input upload, exercised offline over a scripted ``kaggle`` CLI.

No network, no token, no upload: the fake CLI records the argv it was handed
and answers create/version, so every guarantee the publisher makes -- content
addressing, flat staging, the metadata Kaggle validates, both mounts, and the
refusals -- is proven without a Kaggle account.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import chowder.growth.kaggle_inputs as kaggle_inputs
from chowder.growth.compute_backend import (
    DECLARED_INPUT_DIGEST_MISMATCH,
    ComputeBackendRefusal,
    bind_declared_inputs,
)
from chowder.growth.kaggle_inputs import (
    INPUT_DATASET_FILE,
    KAGGLE_INPUT_BASENAMES_COLLIDE,
    KAGGLE_INPUT_BYTES_CHANGED,
    KAGGLE_INPUT_CREATED,
    KAGGLE_INPUT_EMPTY,
    KAGGLE_INPUT_SCHEMA,
    KAGGLE_INPUT_VERSIONED,
    KaggleInputError,
    KaggleInputPublisher,
    declared_inputs_digest,
)

TITLE = "Chowder prepared inputs"


def _completed(returncode: int, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess(
        ["kaggle"], returncode, stdout=stdout, stderr=stderr
    )


class FakeDatasetCli:
    """A scripted `kaggle datasets ...` CLI."""

    def __init__(self, *, create_error="", version_error="") -> None:
        self.create_error = create_error
        self.version_error = version_error
        self.calls: list[list[str]] = []

    def __call__(self, args):
        argv = list(args)
        self.calls.append(argv)
        rest = argv[1:]
        if rest[:2] == ["datasets", "create"]:
            if self.create_error:
                return _completed(1, "", self.create_error)
            return _completed(0, "created\n")
        if rest[:2] == ["datasets", "version"]:
            if self.version_error:
                return _completed(1, "", self.version_error)
            return _completed(0, "versioned\n")
        return _completed(1, "", f"unexpected command: {argv}")


def _declared(tmp_path: Path, names=("training_material", "data_registry")):
    files = {}
    for name in names:
        path = tmp_path / f"{name.replace('_', '-')}.json"
        path.write_text(json.dumps({"field": name}) + "\n", encoding="utf-8")
        files[name] = path
    return bind_declared_inputs(files)


def _publisher(cli, tmp_path: Path, **overrides) -> KaggleInputPublisher:
    fields: dict = {
        "owner": "nikma",
        "runner": cli,
        "staging_root": tmp_path / "staging",
    }
    fields.update(overrides)
    return KaggleInputPublisher(**fields)


# --------------------------------------------------------------------------
# content identity
# --------------------------------------------------------------------------


def test_the_content_digest_is_order_independent_and_names_the_declaration(
    tmp_path: Path,
) -> None:
    bound = _declared(tmp_path)
    assert declared_inputs_digest(tuple(reversed(bound))) == declared_inputs_digest(bound)

    material = tmp_path / "training-material.json"
    renamed = bind_declared_inputs({"training_material_path": material})
    assert declared_inputs_digest(renamed) != declared_inputs_digest(bound), (
        "the digest must cover the declared name: the same bytes under a "
        "different declaration are a different dataset"
    )


# --------------------------------------------------------------------------
# staging, metadata and the upload
# --------------------------------------------------------------------------


def test_publishing_stages_the_flat_files_and_the_metadata_the_cli_reads(
    tmp_path: Path,
) -> None:
    cli = FakeDatasetCli()
    inputs = _declared(tmp_path)
    dataset = _publisher(cli, tmp_path).publish(
        inputs, dataset="chowder-prepared", title=TITLE
    )

    assert cli.calls[0][1:3] == ["datasets", "create"]
    assert "-q" in cli.calls[0]
    staging = Path(cli.calls[0][cli.calls[0].index("-p") + 1])
    metadata = json.loads((staging / INPUT_DATASET_FILE).read_text(encoding="utf-8"))
    assert metadata["id"] == dataset.reference
    assert metadata["title"] == TITLE
    assert metadata["licenses"] == [{"name": "unknown"}], (
        "the publisher stamps no license the campaign did not declare"
    )
    for entry in inputs:
        staged = staging / dataset.files[entry.name]
        assert staged.read_bytes() == Path(entry.path).read_bytes()

    assert dataset.slug.endswith(dataset.content_digest[:12])
    assert dataset.reference == f"nikma/{dataset.slug}"
    assert dataset.operation == KAGGLE_INPUT_CREATED


def test_an_existing_content_addressed_dataset_is_versioned_not_refused(
    tmp_path: Path,
) -> None:
    cli = FakeDatasetCli(create_error="Dataset already exists")
    dataset = _publisher(cli, tmp_path).publish(
        _declared(tmp_path), dataset="chowder-prepared", title=TITLE
    )

    assert dataset.operation == KAGGLE_INPUT_VERSIONED
    assert cli.calls[1][1:3] == ["datasets", "version"]
    assert dataset.content_digest[:12] in cli.calls[1][cli.calls[1].index("-m") + 1]


def test_an_upload_that_can_do_neither_names_both_failures(tmp_path: Path) -> None:
    cli = FakeDatasetCli(create_error="create denied", version_error="version denied")
    with pytest.raises(KaggleInputError) as raised:
        _publisher(cli, tmp_path).publish(
            _declared(tmp_path), dataset="chowder-prepared", title=TITLE
        )
    assert "create denied" in str(raised.value)
    assert "version denied" in str(raised.value)


def test_a_missing_kaggle_cli_is_a_named_error(tmp_path: Path) -> None:
    def missing(argv):
        raise FileNotFoundError("kaggle")

    with pytest.raises(KaggleInputError, match="was not found"):
        _publisher(missing, tmp_path).publish(
            _declared(tmp_path), dataset="chowder-prepared", title=TITLE
        )


def test_the_declared_inputs_get_plain_and_owner_qualified_mount_candidates(
    tmp_path: Path,
) -> None:
    dataset = _publisher(FakeDatasetCli(), tmp_path).publish(
        _declared(tmp_path), dataset="chowder-prepared", title=TITLE
    )
    assert dataset.input_paths
    for name, filename in dataset.files.items():
        assert dataset.input_paths[name] == (
            f"/kaggle/input/{dataset.slug}/{filename}",
            f"/kaggle/input/nikma/{dataset.slug}/{filename}",
        ), "Kaggle mounts under the bare slug, or the owner-qualified path on a conflict"


# --------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------


def test_a_dataset_with_nothing_declared_refuses(tmp_path: Path) -> None:
    cli = FakeDatasetCli()
    with pytest.raises(ComputeBackendRefusal, match=KAGGLE_INPUT_EMPTY):
        _publisher(cli, tmp_path).publish((), dataset="chowder-prepared", title=TITLE)
    assert cli.calls == []


def test_two_declared_inputs_that_would_share_a_file_name_refuse(tmp_path: Path) -> None:
    first = tmp_path / "one"
    second = tmp_path / "two"
    first.mkdir()
    second.mkdir()
    (first / "material.json").write_text("first\n", encoding="utf-8")
    (second / "material.json").write_text("second\n", encoding="utf-8")
    inputs = bind_declared_inputs(
        {"left": first / "material.json", "right": second / "material.json"}
    )
    cli = FakeDatasetCli()
    with pytest.raises(ComputeBackendRefusal, match=KAGGLE_INPUT_BASENAMES_COLLIDE):
        _publisher(cli, tmp_path).publish(
            inputs, dataset="chowder-prepared", title=TITLE
        )
    assert cli.calls == []


def test_a_title_outside_kaggles_rule_refuses(tmp_path: Path) -> None:
    with pytest.raises(ComputeBackendRefusal, match=KAGGLE_INPUT_SCHEMA):
        _publisher(FakeDatasetCli(), tmp_path).publish(
            _declared(tmp_path), dataset="chowder-prepared", title="tiny"
        )


def test_a_slug_that_cannot_carry_the_digest_refuses(tmp_path: Path) -> None:
    with pytest.raises(ComputeBackendRefusal, match=KAGGLE_INPUT_SCHEMA):
        _publisher(FakeDatasetCli(), tmp_path).publish(
            _declared(tmp_path), dataset="Not-A-Slug", title=TITLE
        )
    with pytest.raises(ComputeBackendRefusal, match=KAGGLE_INPUT_SCHEMA):
        _publisher(FakeDatasetCli(), tmp_path).publish(
            _declared(tmp_path), dataset="x" * 60, title=TITLE
        )


def test_a_declared_input_that_moved_before_the_upload_refuses(tmp_path: Path) -> None:
    inputs = _declared(tmp_path)
    Path(inputs[0].path).write_text("tampered\n", encoding="utf-8")
    cli = FakeDatasetCli()
    with pytest.raises(ComputeBackendRefusal, match=DECLARED_INPUT_DIGEST_MISMATCH):
        _publisher(cli, tmp_path).publish(
            inputs, dataset="chowder-prepared", title=TITLE
        )
    assert cli.calls == []


def test_bytes_that_move_between_verification_and_upload_refuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _declared(tmp_path)
    monkeypatch.setattr(
        kaggle_inputs, "verify_declared_inputs", lambda entries: None
    )
    Path(inputs[0].path).write_text("tampered\n", encoding="utf-8")
    cli = FakeDatasetCli()
    with pytest.raises(ComputeBackendRefusal, match=KAGGLE_INPUT_BYTES_CHANGED):
        _publisher(cli, tmp_path).publish(
            inputs, dataset="chowder-prepared", title=TITLE
        )
    assert cli.calls == []
