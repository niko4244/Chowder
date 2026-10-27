"""Adversarial provenance and scorer tests; no GPU or generated-code execution."""

import hashlib
import json
import math

import pytest

from chowder_batch import reap_q4_first200_postrun_20261026 as audit


def _fixture_rows():
    return [{"id": f"gsm8k-{i}", "correct": bool(i % 2), "terminated": True,
             "trigram_ratio": 1.0, "completion_tokens": 10, "seconds": 0.5,
             "predicted_per_second": 20.0, "text": f"answer {i}"}
            for i in range(200)]


def _rows_file(tmp_path, *, mutate=None):
    root = tmp_path / "reap128_v2_q4"
    root.mkdir()
    rows = _fixture_rows()
    if mutate:
        mutate(rows)
    path = root / "gsm8k.jsonl"
    raw = ("\n".join(json.dumps(row) for row in rows) + "\n").encode()
    path.write_bytes(raw)
    return root, hashlib.sha256(raw).hexdigest()


def test_original_exact200_identity_and_private_hash(tmp_path):
    root, digest = _rows_file(tmp_path)
    assert len(audit._strict_rows(root, expected_sha256=digest)) == 200
    with pytest.raises(ValueError, match="hash"):
        audit._strict_rows(root, expected_sha256="0" * 64)
    with (root / "gsm8k.jsonl").open("ab") as f:
        f.write(b" ")
    with pytest.raises(ValueError, match="hash"):
        audit._strict_rows(root, expected_sha256=digest)


@pytest.mark.parametrize("mode", ["duplicate", "replaced", "bool-score", "nan", "unknown-field"])
def test_forged_original_result_row_rejected(tmp_path, mode):
    def change(rows):
        if mode == "duplicate":
            rows[3]["id"] = rows[2]["id"]
        elif mode == "replaced":
            rows[3]["id"] = "gsm8k-other"
        elif mode == "bool-score":
            rows[3]["correct"] = 1
        elif mode == "nan":
            rows[3]["predicted_per_second"] = math.nan
        else:
            rows[3]["unregistered"] = 1
    root, digest = _rows_file(tmp_path, mutate=change)
    with pytest.raises(ValueError, match="ID|schema|numeric|nonfinite"):
        audit._strict_rows(root, expected_sha256=digest)


def test_mutated_original_answer_flag_detected_by_rescorer(monkeypatch):
    rows = _fixture_rows()
    items = [{"id": f"gsm8k-{i}", "expected": str(i)} for i in range(1319)]
    monkeypatch.setattr(audit.reap_eval, "_digest", lambda items: "frozen")
    monkeypatch.setitem(audit.reap_eval.FROZEN_V2_QUALITY_SOURCES, "gsm8k", (1319, "frozen"))
    def scored(_suite, item, out):
        return {"id": item["id"], "correct": int(item["id"].split("-")[-1]) % 2 == 1,
                "terminated": True, "trigram_ratio": 1.0}
    monkeypatch.setattr(audit.reap_eval, "_score", scored)
    assert audit.verify_rows_with_original_scorer(rows, items)["rescore_correct"] == 100
    rows[2]["correct"] = True
    with pytest.raises(ValueError, match="disagrees"):
        audit.verify_rows_with_original_scorer(rows, items)


def test_reject_missing_full_frozen_source(monkeypatch):
    rows = _fixture_rows()
    with pytest.raises(ValueError, match="canonical"):
        audit.verify_rows_with_original_scorer(rows, [{"id": "gsm8k-0"}])


def test_resource_telemetry_floor_and_nonfinite_rejections(tmp_path):
    path = tmp_path / "resource-samples.jsonl"
    row = {"ram_available_gib": 8.0, "commit_available_gib": 20.0,
           "disk_free_gib": 10.0, "gpu_free_mib": 44}
    def save(value):
        path.write_text(json.dumps(value) + "\n")
    save(row)
    assert audit._audit_telemetry(path, expected_samples=1)["minimum_available"]["gpu_free_mib"] == 44
    with pytest.raises(ValueError, match="sample count"):
        audit._audit_telemetry(path, expected_samples=2)
    save({**row, "ram_available_gib": 6.999})
    with pytest.raises(ValueError, match="floor"):
        audit._audit_telemetry(path, expected_samples=1)
    save({**row, "commit_available_gib": math.inf})
    with pytest.raises(ValueError, match="nonfinite"):
        audit._audit_telemetry(path, expected_samples=1)
    save({**row, "gpu_free_mib": -1})
    with pytest.raises(ValueError, match="nonfinite"):
        audit._audit_telemetry(path, expected_samples=1)


def test_bare_result_is_never_selected_or_released():
    assert audit.MODEL_SHA.startswith("2d64b786")
    assert audit.EXPECTED_IDS == {f"gsm8k-{i}" for i in range(200)}
    assert audit.ORIGINAL_FLAGS == ["--fit", "on", "-np", "4", "-c", "16384", "--jinja"]
