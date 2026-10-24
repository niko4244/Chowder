"""CPU-only provenance checks for the registered REAP observation, no model load."""

import hashlib
import json

import pytest

pytest.importorskip("torch")

from chowder_batch.reap_observe import (
    _assert_calibration_unchanged,
    _calibration_snapshot,
    _prepare_observation_destination,
)


def _make_input(tmp_path):
    calibration = tmp_path / "frozen.jsonl"
    original = (
        json.dumps({"text": "question 1", "source": "gsm8k"}) + "\n"
        + json.dumps({"text": "question 2", "source": "math"}) + "\n"
    ).encode()
    calibration.write_bytes(original)
    return calibration, original


def test_exact_snapshot_and_unchanged_source(tmp_path):
    path, original = _make_input(tmp_path)
    rows, sha = _calibration_snapshot(path)
    assert len(rows) == 2
    assert sha == hashlib.sha256(original).hexdigest()
    _assert_calibration_unchanged(path, sha)


def test_source_mutation_refused_even_when_row_census_unchanged(tmp_path):
    path, original = _make_input(tmp_path)
    rows, sha = _calibration_snapshot(path)
    altered = original.replace(b"question 2", b"question X")
    assert len(altered) == len(original)
    path.write_bytes(altered)
    assert len(rows) == len(_calibration_snapshot(path)[0])
    with pytest.raises(RuntimeError, match="calibration bytes changed"):
        _assert_calibration_unchanged(path, sha)


def test_registered_calibration_rejects_empty_and_invalid_content(tmp_path):
    path, _ = _make_input(tmp_path)
    for bad in (b"", b"{", b'{"source":"gsm8k"}\n',
                b'{"text":"a","source":""}\n', b'[]\n'):
        path.write_bytes(bad)
        with pytest.raises(ValueError, match="calibration"):
            _calibration_snapshot(path)


def test_destination_exclusive_and_disjoint(tmp_path):
    calibration, _ = _make_input(tmp_path)
    parent = tmp_path / "model"
    parent.mkdir()
    dest = tmp_path / "new-observer-result"
    _prepare_observation_destination(
        dest, calibration=calibration, model_dir=parent, weights_dir=parent)
    assert dest.is_dir()
    with pytest.raises(FileExistsError, match="refusing overwrite"):
        _prepare_observation_destination(
            dest, calibration=calibration, model_dir=parent, weights_dir=parent)
    assert dest.is_dir()


@pytest.mark.parametrize("destination", ["model/results", "model", "."])
def test_destination_must_not_overlap_original_model(tmp_path, destination):
    calibration, _ = _make_input(tmp_path)
    model = tmp_path / "model"
    model.mkdir()
    target = tmp_path / destination
    with pytest.raises(ValueError, match="overlaps"):
        _prepare_observation_destination(
            target, calibration=calibration, model_dir=model, weights_dir=model)
    if destination == "model/results":
        assert not target.exists()
