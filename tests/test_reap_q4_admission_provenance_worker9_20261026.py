"""Adversarial, CPU-only prelaunch checks; never runs a model subprocess."""

import json
import math

import pytest

from chowder_batch import reap_q4_windows_benchmark_watchdog_20261026 as watchdog
from chowder_batch.reap_q4_resource_admission import admit_host


@pytest.fixture
def receipt(monkeypatch, tmp_path):
    history = {
        "source_sha256": "1" * 64,
        "observed_rows": 12,
        "original_ram_gib": 21.41,
        "original_commit_gib": 28.84,
        "minimum_observed_ram_gib": 6.98,
        "minimum_observed_commit_gib": 10.41,
        "measured_ram_drop_gib": 14.43,
        "measured_commit_drop_gib": 18.43,
        "history_is_lower_bound_only": True,
    }
    cache = {
        "cap_mib": 8192.0,
        "maximum_observed_mib": 2200.0,
        "potential_remaining_gib": (8192 - 2200) / 1024,
        "source_sha256": "2" * 64,
    }
    monkeypatch.setattr(
        watchdog, "validate_aborted_speed_history", lambda *_args, **_kwargs: history,
        raising=False,
    )
    monkeypatch.setattr(
        watchdog, "parse_host_prompt_cache", lambda *_args, **_kwargs: cache,
        raising=False,
    )
    current = {
        "ram_available_gib": 55.0,
        "commit_available_gib": 55.0,
        "gpu_free_mib": 15000.0,
        "disk_free_gib": 45.0,
    }
    decision = admit_host(
        history, **current,
        potential_remaining_cache_gib=cache["potential_remaining_gib"],
    )
    assert decision["status"].startswith("ADMITTED_")
    raw = {
        "schema": "reap128-v2-verified-host-admission.v1",
        "history": dict(history),
        "original_host_prompt_cache": dict(cache),
        "current": dict(current),
        "decision": dict(decision),
    }
    path = tmp_path / "admission.json"
    path.write_text(json.dumps(raw))
    return path, raw


def _assert_no_launch(monkeypatch, tmp_path, path):
    fake_model = tmp_path / "mock.gguf"
    fake_model.write_bytes(b"not a real model")
    fake_server = tmp_path / "mock-server.exe"
    fake_server.write_bytes(b"not a real server")
    monkeypatch.setattr(watchdog, "GGUF", fake_model)
    monkeypatch.setattr(watchdog, "SERVER", fake_server)
    monkeypatch.setattr(
        watchdog, "sample",
        lambda _parent: {
            "ram_available_gib": 55.0,
            "commit_available_gib": 55.0,
            "gpu_free_mib": 15000.0,
            "disk_free_gib": 45.0,
        },
    )
    from chowder_batch import reap_serve

    monkeypatch.setattr(reap_serve, "_require_free_loopback_port", lambda _port: None)
    def forbidden_launch(*_args, **_kwargs):
        raise RuntimeError("UNSAFE_LAUNCH_ATTEMPT")
    monkeypatch.setattr(watchdog.subprocess, "Popen", forbidden_launch)
    out = tmp_path / "new-evidence"
    with pytest.raises(ValueError):
        watchdog.run(out=out, admission=path)
    assert not out.exists()


def test_valid_source_bound_preflight_remains_admissible_without_running_model(receipt):
    _path, raw = receipt
    verified = watchdog.verify_admission_source(raw)
    assert verified["status"] == "ADMITTED_TO_ATTEMPT_NOT_SCIENTIFICALLY_QUALIFIED"
    assert verified["minimum_ram_gib"] > 7
    assert verified["minimum_commit_gib"] > 8
    assert verified["minimum_gpu_mib"] == 12288
    assert verified["scientific_gate"] == "UNKNOWN"


@pytest.mark.parametrize("field", [
    "lower_minimum", "altered_history", "altered_cache", "altered_decision",
    "lower_gpu_floor", "refused_to_admitted",
    "boolean_as_integer", "history_boolean_as_integer",
])
def test_forged_receipt_cannot_authorize_a_native_launch(receipt, monkeypatch, tmp_path, field):
    path, data = receipt
    if field == "lower_minimum":
        data["decision"]["minimum_ram_gib"] = 0.0
    elif field == "altered_history":
        data["history"]["measured_ram_drop_gib"] = 0.0
    elif field == "altered_cache":
        data["original_host_prompt_cache"]["potential_remaining_gib"] = 0.0
    elif field == "altered_decision":
        data["decision"]["scientific_gate"] = "PASS"
    elif field == "lower_gpu_floor":
        data["decision"]["minimum_gpu_mib"] = 0
    elif field == "boolean_as_integer":
        data["decision"]["no_process_launched"] = 1
    elif field == "history_boolean_as_integer":
        data["history"]["history_is_lower_bound_only"] = 1
    else:
        data["decision"]["status"] = "REFUSED"
        data["decision"]["reasons"] = ["low resource"]
        data["decision"]["status"] = "ADMITTED_TO_ATTEMPT_NOT_SCIENTIFICALLY_QUALIFIED"
    path.write_text(json.dumps(data))
    _assert_no_launch(monkeypatch, tmp_path, path)


@pytest.mark.parametrize("corruption", ["nan", "duplicate_key", "missing_current", "invalid_type"])
def test_malformed_receipt_cannot_authorize_model_launch(receipt, monkeypatch, tmp_path, corruption):
    path, data = receipt
    if corruption == "nan":
        data["current"]["ram_available_gib"] = math.nan
        raw = json.dumps(data)
    elif corruption == "duplicate_key":
        raw = path.read_text().replace(
            '"schema":', '"schema": "other", "schema":', 1
        )
    elif corruption == "missing_current":
        del data["current"]["ram_available_gib"]
        raw = json.dumps(data)
    else:
        data["current"]["gpu_free_mib"] = True
        raw = json.dumps(data)
    path.write_text(raw)
    _assert_no_launch(monkeypatch, tmp_path, path)


def test_untrusted_prelaunch_nan_is_refused_even_with_a_valid_receipt(
    receipt, monkeypatch, tmp_path,
):
    path, _ = receipt
    monkeypatch.setattr(
        watchdog, "sample",
        lambda _parent: {
            "ram_available_gib": math.nan,
            "commit_available_gib": 55.0,
            "gpu_free_mib": 15000.0,
            "disk_free_gib": 45.0,
        },
    )
    model = tmp_path / "mock.gguf"
    model.write_bytes(b"not a real model")
    server = tmp_path / "mock.exe"
    server.write_bytes(b"not a real server")
    monkeypatch.setattr(watchdog, "GGUF", model)
    monkeypatch.setattr(watchdog, "SERVER", server)
    from chowder_batch import reap_serve

    monkeypatch.setattr(reap_serve, "_require_free_loopback_port", lambda _port: None)
    def forbidden_launch(*_args, **_kwargs):
        raise RuntimeError("UNSAFE_LAUNCH_ATTEMPT")
    monkeypatch.setattr(watchdog.subprocess, "Popen", forbidden_launch)
    with pytest.raises(ValueError):
        watchdog.run(out=tmp_path / "new-evidence", admission=path)
    assert not (tmp_path / "new-evidence").exists()


@pytest.mark.parametrize("corruption", [
    "missing_speed", "wrong_session", "wrong_model", "wrong_rows", "missing_samples",
    "abort", "live_native", "missing_summary", "wrong_parallel", "missing_native_pid",
])
def test_terminal_claim_requires_source_and_watchdog_completion(corruption):
    rows_sha = "a" * 64
    session = "e" * 32
    serving = {
        "server_pid": 1234, "eval_session_id": session,
        "gguf_sha256": watchdog.MODEL_SHA,
        "flags": ["--fit", "on", "-np", "4", "-c", "16384", "--jinja"],
        "measured_experts_offloaded": 0,
    }
    speed = {
        "schema": "chowder.reap-q4-same-session-speed.v1",
        "server_pid": 1234, "session_id": session,
        "port": watchdog.PORT, "suite": "gsm8k", "n": 200,
        "gguf_sha256": watchdog.MODEL_SHA,
        "rows_sha256": rows_sha,
    }
    metadata = {
        "child_exit_code": 0, "sample_count": 148,
        "partial_or_complete_rows": 200,
        "rows_sha256": rows_sha, "summary_sha256": "b" * 64,
        "native_shutdown_verified": True,
        "source_bound_speed_session": speed,
    }
    assert watchdog.completed_source_bound_run(metadata, serving)
    if corruption == "missing_speed":
        metadata["source_bound_speed_session"] = {}
    elif corruption == "wrong_session":
        speed["session_id"] = "c" * 32
    elif corruption == "wrong_model":
        serving["gguf_sha256"] = "0" * 64
    elif corruption == "wrong_rows":
        speed["rows_sha256"] = "0" * 64
    elif corruption == "missing_samples":
        metadata["sample_count"] = 0
    elif corruption == "abort":
        metadata["abort_reasons"] = ["low RAM"]
    elif corruption == "live_native":
        metadata["native_shutdown_verified"] = False
    elif corruption == "missing_summary":
        del metadata["summary_sha256"]
    elif corruption == "wrong_parallel":
        serving["flags"][3] = "1"
    else:
        del serving["server_pid"]
        del speed["server_pid"]
    assert not watchdog.completed_source_bound_run(metadata, serving)
