"""Immutable original GGUF 42-chunk receipt path admission; no model inference."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from chowder_batch import reap_serve
from chowder_batch import reap_v2_frozen_wave_20261027 as wave
from chowder_batch import reap_v2_original_artifact_portability_20261027 as transport


@pytest.mark.parametrize("arm,model,sidecar", [
    ("reap128_v2_q4", "G:/reap128-v2-q4_k_m-20261025.gguf",
     "cf16fbf9e6926719bab873bd74b7062fa22365a280451ce3db9a9f8dc3031f1c"),
    ("reap128_v2", "I:/reap128-v2-q8_0-20261025.gguf",
     "a1cfe1464df4579ea9d5ae1320efc01c0de5dae1070235caca18c821ce67479d"),
])
def test_actual_original_published_model_sidecar_sha_and_source_inventory(arm, model, sidecar):
    result = transport.preflight(wave.study_for(arm), Path(model))
    assert result["arm"] == arm
    assert result["sidecar_sha256"] == sidecar
    assert result["actual_full_model_sha256_measured"] is False
    assert result["model_loaded"] is False
    assert result["gen0"]["chosen"] is None


def test_same_study_relocated_path_refused_before_model_hash(monkeypatch, tmp_path: Path):
    # This explicitly catches the incorrect Phase47 dedicated-host C:\reap\models
    # instructions; no fake candidate GGUF or source-proof PASS is produced.
    model = tmp_path / "reap128-v2-q4_k_m-20261025.gguf"
    model.write_bytes(b"synthetic fixture only")
    with pytest.raises(transport.OriginalTransportRefused, match="original path required"):
        transport.preflight(wave.study_for("reap128_v2_q4"), model)


def test_wrong_arm_or_model_sha_refused_even_with_trusted_source():
    legitimate = wave.study_for("reap128_v2_q4")
    mismatched = replace(legitimate, model_sha256="0" * 64)
    with pytest.raises(transport.OriginalTransportRefused, match="identity"):
        transport.preflight(mismatched, Path("G:/reap128-v2-q4_k_m-20261025.gguf"))


def test_duplicate_original_construction_sidecar_keys_fail_closed(monkeypatch):
    from pathlib import Path as P
    valid_model = P("G:/reap128-v2-q4_k_m-20261025.gguf")
    original = P.read_bytes

    def tampered(path):
        if str(path).endswith(".reap-chunked.json"):
            real = json.loads(original(path))
            raw = json.dumps(real, separators=(",", ":"))
            return raw.replace('"status":', '"status":"FORGED_STATUS","status":', 1).encode()
        return original(path)

    monkeypatch.setattr(P, "read_bytes", tampered)
    with pytest.raises(transport.OriginalTransportRefused, match="duplicate"):
        transport.preflight(wave.study_for("reap128_v2_q4"), valid_model)


def test_original_runtime_source_change_does_not_authorize_sidecar(monkeypatch):
    monkeypatch.setattr(transport, "PINNED_SERVE_SOURCE_SHA256", "0" * 64)
    with pytest.raises(transport.OriginalTransportRefused, match="construction verifier source"):
        transport.preflight(wave.study_for("reap128_v2"), Path(
            "I:/reap128-v2-q8_0-20261025.gguf"))


def test_registered_construction_receipt_is_genuine_not_model_bytes():
    # Original giant GGUF cannot be hashed in CPU tests while owner may be
    # serving it. This validates recorded tensor source and then leaves full
    # bytes SHA to the owner-bound dedicated admission at the correct time.
    model = Path("G:/reap128-v2-q4_k_m-20261025.gguf")
    receipt = model.with_name(model.name + ".reap-chunked.json")
    assert receipt.stat().st_size < 4 * 1024 * 1024
    assert reap_serve.REGISTERED_V2_CONSTRUCTION["reap128_v2_q4"]
    assert transport.preflight(wave.study_for("reap128_v2_q4"), model)[
        "status"] == "ORIGINAL_PATH_AND_REGISTERED_CONSTRUCTION_RECEIPT_ONLY"
