"""CPU-only fail-closed original 1,319 acceptance; no model launch."""

import hashlib
import json
import math

import pytest

from chowder_batch import reap_q4_windows_full1319_20261027 as full


def _fresh():
    return {"ram_available_gib": 33.0, "commit_available_gib": 55.0,
            "gpu_free_mib": 14000, "disk_free_gib": 35.0}


def test_original_admission_uses_recomputed_pinned_floor_not_claimed(monkeypatch):
    monkeypatch.setattr(full.q4, "verify_admission_source", lambda _: {
        "minimum_ram_gib": 29.3146, "minimum_commit_gib": 34.3146,
        "minimum_gpu_mib": 12288, "minimum_disk_gib": 8.0,
        "status": "ADMITTED_TO_ATTEMPT_NOT_SCIENTIFICALLY_QUALIFIED"})
    assert full.source_admission({"current": _fresh()}, _fresh())["minimum_gpu_mib"] == 12288
    for key, bad in (("gpu_free_mib", 11089), ("ram_available_gib", 27.0),
                     ("commit_available_gib", 33.0), ("disk_free_gib", 7.0),
                     ("gpu_free_mib", True), ("ram_available_gib", math.nan)):
        candidate = _fresh()
        candidate[key] = bad
        with pytest.raises(ValueError, match="unsafe|refused"):
            full.source_admission({"current": _fresh()}, candidate)


def _populate(tmp_path, monkeypatch):
    arm = tmp_path / full.q4.ARM
    arm.mkdir()
    source_items = [{"id": f"gsm8k-{i}", "expected": str(i)} for i in range(full.FULL)]
    monkeypatch.setattr(full.reap_eval, "_load", lambda suite: source_items)
    monkeypatch.setattr(full.reap_eval, "_digest", lambda _: full.reap_eval.FROZEN_V2_QUALITY_SOURCES["gsm8k"][1])
    monkeypatch.setattr(full.reap_serve, "_verify_offload_receipt", lambda *_: None)

    def scorer(suite, item, output):
        return {"id": item["id"], "correct": int(item["id"].split("-")[1]) % 2 == 0,
                "terminated": True, "trigram_ratio": 1.0}

    monkeypatch.setattr(full.reap_eval, "_score", scorer)
    serve = {"gguf": "G:/real-original-q4.gguf", "gguf_sha256": full.q4.MODEL_SHA,
             "flags": full.reap_serve.FLAGS, "eval_session_id": "a" * 32,
             "server_port": full.q4.PORT, "server_pid": 1234,
             "measured_experts_offloaded": 0}
    (arm / "serve.json").write_text(json.dumps(serve), encoding="utf8")
    (arm / "gsm8k.inputs.json").write_text(json.dumps({
        "source_items_sha256": full.reap_eval.FROZEN_V2_QUALITY_SOURCES["gsm8k"][1],
        "source_items_n": full.FULL, "suite": "gsm8k",
        "model": {name: serve[name] for name in ("gguf", "gguf_sha256", "flags")}}), encoding="utf8")
    rows = [{"id": f"gsm8k-{i}", "correct": i % 2 == 0, "terminated": True,
             "trigram_ratio": 1.0, "completion_tokens": 2, "seconds": 0.2,
             "predicted_per_second": 10.0, "text": "model completion"}
            for i in range(full.FULL)]
    _write_rows(arm, rows)
    return arm, rows


def _write_rows(arm, rows):
    raw = ("\n".join(json.dumps(row) for row in rows) + "\n").encode()
    (arm / "gsm8k.jsonl").write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    (arm / "summary.json").write_text(json.dumps({"gsm8k": {
        "n": full.FULL, "file_sha256": digest,
        "accuracy": sum(r["correct"] for r in rows) / full.FULL,
        "terminated_rate": sum(r["terminated"] for r in rows) / full.FULL,
        "degenerate_rate": sum(r["trigram_ratio"] < 0.5 for r in rows) / full.FULL,
    }}), encoding="utf8")


def test_exact_1319_original_scorer_recount_and_no_release(tmp_path, monkeypatch):
    arm, rows = _populate(tmp_path, monkeypatch)
    proof = full.complete_original_gsm8k(arm, model_session="a" * 32)
    assert proof["population"] == 1319
    assert proof["correct"] == 660
    assert proof["native_offload_verified"] is True
    assert proof["all_rows_rescored"] is True
    rows[0]["correct"] = False
    _write_rows(arm, rows)
    with pytest.raises(ValueError, match="scorer"):
        full.complete_original_gsm8k(arm, model_session="a" * 32)


@pytest.mark.parametrize("mode", ["id", "missing", "finite", "unexpected", "model"])
def test_forged_original_result_population_or_source_rejected(tmp_path, monkeypatch, mode):
    arm, rows = _populate(tmp_path, monkeypatch)
    if mode == "id":
        rows[9]["id"] = rows[8]["id"]
    elif mode == "missing":
        rows.pop()
    elif mode == "finite":
        rows[7]["predicted_per_second"] = math.inf
    elif mode == "unexpected":
        rows[7]["invented_accuracy"] = True
    elif mode == "model":
        (arm / "serve.json").write_text(json.dumps({"gguf_sha256": "0" * 64}), encoding="utf8")
    if mode != "model":
        _write_rows(arm, rows)
    with pytest.raises(ValueError, match="IDs|source|malformed|nonfinite|serving|Infinity"):
        full.complete_original_gsm8k(arm, model_session="a" * 32)


def test_full_source_contract_is_registered_and_only_gsm8k():
    assert full.FULL == 1319
    assert full.EXPECTED_IDS == {f"gsm8k-{i}" for i in range(1319)}
    assert full.q4.MODEL_SHA.startswith("2d64b786")
