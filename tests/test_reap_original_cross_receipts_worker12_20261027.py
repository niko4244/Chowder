"""Behavioral canaries for independent original receipt audit boundaries."""

import json

import pytest

from chowder_batch import reap_eval
from chowder_batch import reap_original_cross_receipts_worker12_20261027 as audit


def _source():
    return [{"id": f"gsm8k-{i}", "expected": "5"} for i in range(audit.N)]


def _lines(source, count=200):
    lines = []
    for item in source[:count]:
        record = reap_eval._score("gsm8k", item, {
            "finish_reason": "stop", "text": "5", "completion_tokens": 1,
            "seconds": 1.0, "timings": {"predicted_per_second": 5.0},
        })
        lines.append(json.dumps(record, sort_keys=True, allow_nan=False))
    return ("\n".join(lines) + "\n").encode()


def test_source_complete_200_independent_replay_and_throughput():
    source = _source()
    result = audit.score_rows(_lines(source), source, count=200, session=None)
    assert result["count"] == 200
    assert result["correct"] == 200
    assert result["terminated"] == 200
    assert result["completion_tokens"] == 200
    assert result["mean_item_predicted_per_second"] == 5.0


def test_source_complete_1319_independent_replay():
    source = _source()
    result = audit.score_rows(_lines(source, audit.N), source, count=audit.N, session=None)
    assert result["count"] == audit.N
    assert result["correct"] == audit.N
    assert result["rows_sha256"] == audit._sha(_lines(source, audit.N))


def test_altered_original_correctness_fails_without_rehash_protection():
    source = _source()
    rows = [json.loads(line) for line in _lines(source).splitlines()]
    rows[17]["correct"] = False
    edited = ("\n".join(json.dumps(row) for row in rows) + "\n").encode()
    with pytest.raises(ValueError, match="scorer disagrees"):
        audit.score_rows(edited, source, count=200, session=None)


def test_duplicate_missing_or_out_of_population_source_rows_fail():
    source = _source()
    lines = _lines(source).splitlines()
    with pytest.raises(ValueError, match="duplicated"):
        audit.score_rows(b"\n".join([lines[0], *lines[:199]]) + b"\n", source,
                         count=200, session=None)
    with pytest.raises(ValueError, match="incomplete"):
        audit.score_rows(b"\n".join(lines[:-1]) + b"\n", source,
                         count=200, session=None)
    replacement = json.loads(lines[-1])
    replacement["id"] = "gsm8k-1290"
    with pytest.raises(ValueError, match="ID coverage"):
        audit.score_rows(b"\n".join([*lines[:-1], json.dumps(replacement).encode()]) + b"\n",
                         source, count=200, session=None)


def test_nonfinite_and_truncated_model_receipts_fail():
    source = _source()
    raw = _lines(source)
    with pytest.raises(ValueError, match="truncated"):
        audit.score_rows(raw.rstrip(b"\n"), source, count=200, session=None)
    altered = raw.replace(b'"seconds": 1.0', b'"seconds": NaN', 1)
    with pytest.raises(ValueError, match="nonfinite JSON"):
        audit.score_rows(altered, source, count=200, session=None)
    altered = raw.replace(b'"seconds": 1.0', b'"seconds": -5.0', 1)
    with pytest.raises(ValueError, match="nonfinite/negative"):
        audit.score_rows(altered, source, count=200, session=None)


def test_external_session_identity_fail_closed_and_duplicate_json_key():
    source = _source()
    lines = [json.loads(line) for line in _lines(source).splitlines()]
    for item in lines:
        item["eval_session_id"] = "a" * 32
    raw = ("\n".join(json.dumps(item) for item in lines) + "\n").encode()
    assert audit.score_rows(raw, source, count=200, session="a" * 32)["count"] == 200
    with pytest.raises(ValueError, match="session"):
        audit.score_rows(raw, source, count=200, session="b" * 32)
    with pytest.raises(ValueError, match="duplicate JSON"):
        audit._json(b'{"id":"foo", "id":"bar"}')


def test_symlinked_or_missing_receipts_fail_closed(tmp_path):
    missing = tmp_path / "missing.json"
    with pytest.raises(ValueError, match="missing/unsafe"):
        audit._read(missing)
    actual = tmp_path / "real.json"
    actual.write_text('{"value":1}', encoding="utf8")
    unsafe = tmp_path / "link.json"
    try:
        unsafe.symlink_to(actual)
    except (NotImplementedError, OSError):
        pytest.skip("host does not support unprivileged symlink creation")
    with pytest.raises(ValueError, match="missing/unsafe"):
        audit._read(unsafe)


def test_original_resource_samples_validate_real_floor_and_monotonicity():
    def serialize(rows):
        return ("\n".join(json.dumps(x, allow_nan=False) for x in rows) + "\n").encode()

    rows = [
        {"time_utc": f"2026-10-27T06:40:{i:02d}+00:00", "ram_available_gib": 11.0,
         "commit_available_gib": 31.0, "disk_free_gib": 34.0, "gpu_free_mib": 179}
        for i in range(2)
    ]
    result = audit.historical_resource_samples(serialize(rows), 2)
    assert result["status"] == "PASS_HISTORICAL_SAMPLED_RAM_COMMIT_DISK_FLOORS"
    assert result["minimum_host_available"]["gpu_free_mib"] == 179
    assert result["host_gpu_is_not_independent_model_allocation"] is True
    with pytest.raises(ValueError, match="clock out of order"):
        audit.historical_resource_samples(serialize(list(reversed(rows))), 2)
    rows[1]["ram_available_gib"] = 6.99
    with pytest.raises(ValueError, match="floor violated"):
        audit.historical_resource_samples(serialize(rows), 2)
