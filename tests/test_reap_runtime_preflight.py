"""Adversarial CPU-only tests; never make a network call or load a model."""

from pathlib import Path

import pytest

from chowder_batch.reap_runtime_preflight import (
    RuntimeIdentityRefusal,
    _launched_model_path,
    admit_loaded_gguf,
    read_existing_server_models,
)


def _model(path: str, *, state: str = "loaded", model_id: str = "reap160") -> dict:
    return {"id": model_id, "status": {"value": state, "args": ["llama-server", "--model", path]}}


def test_only_exact_loaded_existing_path_is_operationally_eligible(tmp_path: Path):
    target = tmp_path / "reap160-q4_k_m.gguf"
    target.write_bytes(b"GGUF header fixture not real weights")
    other = tmp_path / "wrong.gguf"
    other.write_bytes(b"another fixture")
    result = admit_loaded_gguf({"data": [
        _model(str(other), model_id="wrong"),
        _model(str(target), model_id="reap160"),
    ]}, target)
    assert result["model_id"] == "reap160"
    assert result["status"] == "LOADED_PATH_MATCH_ONLY_NOT_WEIGHT_ATTESTED"
    assert result["no_generation_sent"] is True


@pytest.mark.parametrize("attack", [
    "unloaded", "loading", "other", "alias", "relative", "missing_model_args",
    "duplicate", "missing_status", "missing_id", "invalid_data", "missing_file",
    "unloaded_same_id", "malformed_record", "wrong_path_same_name",
])
def test_unsafe_model_attribution_refused(tmp_path: Path, attack: str):
    target = tmp_path / "reap160.gguf"
    target.write_bytes(b"fixture")
    other = tmp_path / "other.gguf"
    other.write_bytes(b"fixture")
    record = _model(str(target))
    payload = {"data": [record]}
    if attack in ("unloaded", "loading"):
        record["status"]["value"] = attack
    elif attack in ("other", "wrong_path_same_name"):
        record["status"]["args"][-1] = str(other)
    elif attack == "alias":
        record["status"].pop("args")
    elif attack == "relative":
        record["status"]["args"][-1] = "reap160.gguf"
    elif attack == "missing_model_args":
        record["status"]["args"] = ["llama-server", "--model"]
    elif attack == "duplicate":
        payload["data"].append(_model(str(target)))
    elif attack == "missing_status":
        record.pop("status")
    elif attack == "missing_id":
        record.pop("id")
    elif attack == "invalid_data":
        payload["data"] = "reap160"
    elif attack == "missing_file":
        target.unlink()
    elif attack == "unloaded_same_id":
        payload["data"] = [_model(str(target), state="unloaded")]
    elif attack == "malformed_record":
        payload["data"] = [record, 5]
    with pytest.raises(RuntimeIdentityRefusal):
        admit_loaded_gguf(payload, target)


def test_reject_overdeclared_and_draft_model_arguments():
    assert _launched_model_path(["llama-server", "--model", "x.gguf"]) == "x.gguf"
    assert _launched_model_path(["llama-server", "--model=x.gguf"]) == "x.gguf"
    assert _launched_model_path(["llama-server", "-m", "x.gguf"]) == "x.gguf"
    assert _launched_model_path(["llama-server", "--model-draft", "other.gguf", "--model", "x.gguf"]) == "x.gguf"
    assert _launched_model_path(["llama-server", "--model", "x.gguf", "--model", "y.gguf"]) is None
    assert _launched_model_path(["llama-server", "--model"]) is None
    assert _launched_model_path(["llama-server", "--model", 9]) is None


@pytest.mark.parametrize("bad_port", [0, -1, 65536, 5000.0, True, None])
def test_network_preflight_never_opens_invalid_endpoint(bad_port):
    with pytest.raises(RuntimeIdentityRefusal, match="port"):
        read_existing_server_models(bad_port)
