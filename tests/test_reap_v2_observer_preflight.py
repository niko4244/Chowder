"""Adversarial CPU-only admission tests; no checkpoint loads, CUDA or HF downloads."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest

from chowder_batch import reap_v2_observer_preflight as preflight
from chowder_batch import reap_v2_source_identity as identity


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _safetensors(path: Path, tensors: dict[str, list[int]]) -> None:
    header, offset = {}, 0
    for name, shape in sorted(tensors.items()):
        length = 2
        for dimension in shape:
            length *= dimension
        header[name] = {"dtype": "BF16", "shape": shape,
                        "data_offsets": [offset, offset + length]}
        offset += length
    encoded = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\x00" * offset)


@pytest.fixture
def sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    calibration = tmp_path / "calibration"
    model = tmp_path / "model"
    calibration.mkdir()
    model.mkdir()
    body = "".join(json.dumps({"source": "swe_smith_trajectories", "n_tokens": 256,
                               "text": f"example training row number {i} " + "hello " * 45}) + "\n"
                   for i in range(704)).encode("utf-8")
    (calibration / "calibration.jsonl").write_bytes(body)
    monkeypatch.setattr(identity, "FROZEN_V2_BASE_SHA256", hashlib.sha256(body).hexdigest())
    _write_json(calibration / "calibration-manifest.json", {
        "mix": "v2", "seed": preflight.SEED, "ngram": preflight.NGRAM,
        "max_tokens": 2048, "rows": 704, "tokens": 704 * 256,
        "calibration_sha256": hashlib.sha256(body).hexdigest(),
        "sources": {"swe_smith_trajectories": {"kept": 704}},
        "ornith": identity.ABSENT_ORNITH,
    })
    _write_json(model / "config.json", {
        "model_type": "qwen3_5_moe", "text_config": {
            "model_type": "qwen3_5_moe_text", "num_hidden_layers": 40,
            "num_experts": 256, "num_experts_per_tok": 8,
            "hidden_size": 4, "vocab_size": 12,
        },
    })
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        (model / name).write_text("{}", encoding="utf-8")
    shapes = {key: ([12, 4] if "embed_tokens" in key or key == "lm_head.weight"
                    else [4] if key.endswith("norm.weight") else [256, 1])
              for key in preflight.REQUIRED_TENSORS}
    _safetensors(model / "model-00001.safetensors", shapes)
    _write_json(model / "model.safetensors.index.json", {
        "weight_map": {key: "model-00001.safetensors" for key in shapes},
    })
    return calibration, model, tmp_path / "observer-new"


def _admit(sources: tuple[Path, Path, Path]) -> dict:
    calibration, model, out = sources
    return preflight.admit(calibration=calibration, model_dir=model, out=out)


def test_frozen_v2_input_admitted_without_gpu_or_output_creation(sources):
    report = _admit(sources)
    assert report["status"] == "INPUTS_ADMITTED_OBSERVATION_NOT_RUN"
    assert report["calibration"]["rows"] == 704
    assert report["calibration"]["ornith_suffix_rows"] == 0
    assert report["model"]["shard_count"] == 1
    assert report["model"]["indexed_tensors"] == 123
    assert report["gate_d0"].startswith("NOT_MEASURED")
    assert report["gpu_work_started"] is False
    assert report["observation_command_not_executed"][0:5] == [
        "python", "-B", "-m", "chowder_batch.reap_observe_supervised", "--launch",
    ]
    assert "--scratch" in report["observation_command_not_executed"]
    assert "--log-dir" in report["observation_command_not_executed"]
    assert not sources[2].exists()


def test_rehashed_fabricated_calibration_refused_even_with_same_704_rows(sources):
    calibration, _, _ = sources
    rows = (calibration / "calibration.jsonl").read_text(encoding="utf-8")
    (calibration / "calibration.jsonl").write_text(rows.replace("example training row number 42",
                                                             "example training row number 24", 1),
                                                  encoding="utf-8")
    manifest = json.loads((calibration / "calibration-manifest.json").read_text(encoding="utf-8"))
    manifest["calibration_sha256"] = hashlib.sha256(
        (calibration / "calibration.jsonl").read_bytes()).hexdigest()
    _write_json(calibration / "calibration-manifest.json", manifest)
    with pytest.raises(preflight.ObserverAdmissionError, match="registered v2 source identity"):
        _admit(sources)


def test_rehashed_worked_ornith_suffix_refused(sources):
    calibration, _, _ = sources
    path = calibration / "calibration.jsonl"
    path.write_bytes(path.read_bytes() + (json.dumps({"source": "ornith_curriculum",
                                                    "n_tokens": 300, "text": "invented"}) + "\n").encode())
    manifest = json.loads((calibration / "calibration-manifest.json").read_text(encoding="utf-8"))
    manifest["calibration_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["rows"] = 705
    manifest["tokens"] += 300
    manifest["sources"]["ornith_curriculum"] = {"kept": 1}
    _write_json(calibration / "calibration-manifest.json", manifest)
    with pytest.raises(preflight.ObserverAdmissionError, match="nonempty appended suffix"):
        _admit(sources)


@pytest.mark.parametrize("damage,expected", [
    ("truncate", "outside"),
    ("misdeclare", "inconsistent dtype/shape/byte range"),
])
def test_shard_with_invalid_data_extent_refused_before_cuda(sources, damage, expected):
    _, model, _ = sources
    path = model / "model-00001.safetensors"
    raw = path.read_bytes()
    if damage == "truncate":
        path.write_bytes(raw[:-40])
    else:
        header_length = struct.unpack("<Q", raw[:8])[0]
        header = json.loads(raw[8:8 + header_length])
        tensor = next(iter(header.values()))
        tensor["shape"][0] += 1
        modified = json.dumps(header).encode("utf-8")
        path.write_bytes(struct.pack("<Q", len(modified)) + modified + raw[8 + header_length:])
    with pytest.raises(preflight.ObserverAdmissionError, match=expected):
        _admit(sources)


def test_missing_final_head_refused(sources):
    _, model, _ = sources
    index = model / "model.safetensors.index.json"
    data = json.loads(index.read_text(encoding="utf-8"))
    del data["weight_map"]["lm_head.weight"]
    _write_json(index, data)
    with pytest.raises(preflight.ObserverAdmissionError, match="index lacks observer"):
        _admit(sources)


def test_missing_tokenizer_refused(sources):
    _, model, _ = sources
    (model / "tokenizer.json").unlink()
    with pytest.raises(preflight.ObserverAdmissionError, match="tokenizer artifact missing"):
        _admit(sources)


def test_occupied_observer_path_refused_before_input_work(sources):
    sources[2].mkdir()
    with pytest.raises(preflight.ObserverAdmissionError, match="output already exists"):
        _admit(sources)


def test_pruned_expert_tensor_header_refused(sources):
    _, model, _ = sources
    shard = model / "model-00001.safetensors"
    raw = shard.read_bytes()
    header_len = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + header_len])
    target = next(k for k in header if k.endswith("mlp.gate.weight"))
    header[target]["shape"] = [128, 2]
    encoded = json.dumps(header).encode("utf-8")
    shard.write_bytes(struct.pack("<Q", len(encoded)) + encoded + raw[8 + header_len:])
    with pytest.raises(preflight.ObserverAdmissionError, match="unpruned 256-expert"):
        _admit(sources)


def test_resume_accepts_existing_output_but_refuses_completed_observation(sources):
    """A --resume pre-screen must accept the directory the resume requires.

    Otherwise --doctor reports a perfectly valid resumable checkpoint as
    "frozen_paths_or_source_invalid", which is what happened to a real run that
    had committed 20 of 40 layers before a telemetry abort.
    """
    calibration, model, out = sources

    def admit_resume() -> dict:
        return preflight.admit(
            calibration=calibration, model_dir=model, out=out,
            allow_existing_out=True,
        )

    # Default behaviour still refuses any pre-existing path.
    out.mkdir()
    with pytest.raises(preflight.ObserverAdmissionError, match="output already exists"):
        _admit(sources)

    # A resume pre-screen accepts it (checkpoint present, not yet published).
    assert admit_resume()["bounds"]["sequences"] == 704

    # A completed observation can never be resumed over.
    (out / "observation.json").write_text("{}", encoding="utf-8")
    with pytest.raises(preflight.ObserverAdmissionError, match="already completed"):
        admit_resume()
