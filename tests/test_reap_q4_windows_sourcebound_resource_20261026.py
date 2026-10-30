"""Adversarial CPU-only Q4 Windows resource provenance tests (synthetic rows)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import math

import pytest

from chowder_batch import reap_q4_windows_sourcebound_resource_20261026 as audit


def _fixture():
    rows = [
        {"id": f"gsm8k-{i}", "correct": i < 187, "terminated": i < 192}
        for i in range(200)
    ]
    row_blob = b"".join(json.dumps(row).encode() + b"\n" for row in rows)
    start = datetime(2026, 10, 26, 14, tzinfo=timezone.utc)
    samples = [
        {"time_utc": (start + timedelta(seconds=3 * i)).isoformat(),
         "ram_available_gib": 31.1 - 0.1 * i,
         "commit_available_gib": 38.1 - 0.15 * i,
         "disk_free_gib": 36.0, "gpu_free_mib": 14346 if i == 0 else 2859}
        for i in range(148)
    ]
    speed = {
        "schema": "chowder.reap-q4-same-session-speed.v1",
        "session_id": audit.SESSION,
        "rows_sha256": hashlib.sha256(row_blob).hexdigest(),
        "n": 200,
    }
    serving = {
        "gguf_sha256": audit.MODEL_SHA,
        "eval_session_id": audit.SESSION, "flags": audit.FROZEN,
        "measured_experts_offloaded": 0,
        "speed_session": speed,
    }
    terminal = {
        "status": "COMPLETE_SOURCE_BOUND_200_CANDIDATE_ONLY",
        "child_exit_code": 0, "partial_or_complete_rows": 200,
        "sample_count": 148, "abort_reasons": None,
    }
    native = (
        "load_tensors: offloaded 41/41 layers to GPU\n"
        "load_tensors: CUDA0 model buffer size = 10619.88 MiB\n"
        "llama_context: n_seq_max             = 4\n"
        "llama_context: n_ctx_seq             = 4096\n"
        "llama_kv_cache: CUDA0 KV buffer size = 320.00 MiB\n"
        "srv update: - cache state: 41 prompts, 8150.615 MiB "
        "(limits: 8192.000 MiB, 16384 tokens, 8589934592 est)\n"
    ).encode()
    encode = lambda obj: json.dumps(obj, allow_nan=False).encode()
    return (
        b"".join(encode(row) + b"\n" for row in samples),
        encode(terminal), native, encode(serving), row_blob,
    )


def test_synthetic_sourcebound_resource_census_is_descriptive_only():
    result = audit.summarize(*_fixture())
    assert result["status"].endswith("CANDIDATE_ONLY")
    assert result["questions"] == 200 and result["correct"] == 187
    assert result["native_cuda_model_buffer_mib"] == 10619.88
    assert result["native_cuda_kv_buffer_mib"] == 320
    assert result["native_prompt_cache_max_observed_mib"] == 8150.615
    assert result["cuda_allocator_instantaneous_peak_certified"] is False
    assert result["speed_ratio_1p3_certified"] is False
    assert result["gen0_status"] == "BLOCKED"


@pytest.mark.parametrize("mutate", [
    lambda r: r[0].__setitem__("time_utc", "2026-10-26T14:00:00"),
    lambda r: r[20].__setitem__("time_utc", r[19]["time_utc"]),
    lambda r: r[20].__setitem__("time_utc", "2026-10-26T18:00:00+00:00"),
    lambda r: r[20].__setitem__("gpu_free_mib", True),
    lambda r: r[40].__setitem__("ram_available_gib", 6.9),
    lambda r: r[60].__setitem__("commit_available_gib", 7.9),
    lambda r: r[70].__setitem__("disk_free_gib", 7.9),
    lambda r: r[0].__setitem__("gpu_free_mib", 11999),
    lambda r: r.pop(4),
])
def test_mutated_sampled_resource_or_continuity_refused(mutate):
    src = list(_fixture())
    records = [json.loads(line) for line in src[0].splitlines()]
    mutate(records)
    # A poisoned nonfinite sample must fail even if a JSON writer allows NaN.
    src[0] = b"".join(json.dumps(x).encode() + b"\n" for x in records)
    with pytest.raises(ValueError):
        audit.summarize(*src)


@pytest.mark.parametrize(("which", "mutation"), [
    (1, lambda r: r.update(child_exit_code=1)),
    (1, lambda r: r.update(sample_count=147)),
    (1, lambda r: r.update(status="ABORT_OR_INCOMPLETE")),
    (3, lambda r: r.update(measured_experts_offloaded=1)),
    (3, lambda r: r.update(flags=["--fit", "off", "-np", "4", "-c", "16384", "--jinja"])),
    (3, lambda r: r["speed_session"].update(rows_sha256="0" * 64)),
])
def test_mutated_source_continuity_refused(which, mutation):
    src = list(_fixture())
    changed = json.loads(src[which])
    mutation(changed)
    src[which] = json.dumps(changed).encode()
    with pytest.raises(ValueError):
        audit.summarize(*src)


@pytest.mark.parametrize("change", [
    lambda text: text.replace(b"41/41 layers", b"40/41 layers"),
    lambda text: text.replace(b"320.00 MiB", b"640.00 MiB"),
    lambda text: text.replace(b"8192.000 MiB", b"4096.000 MiB"),
    lambda text: text.replace(b"8150.615 MiB", b"8900.615 MiB"),
    lambda text: text.replace(b"n_seq_max             = 4", b"n_seq_max             = 2"),
])
def test_native_placement_kv_cache_changed_refused(change):
    src = list(_fixture())
    src[2] = change(src[2])
    with pytest.raises(ValueError):
        audit.summarize(*src)


def test_nonfinite_sample_poison_is_refused():
    src = list(_fixture())
    src[0] = src[0].replace(b"31.1", b"NaN", 1)
    with pytest.raises(ValueError):
        audit.summarize(*src)
