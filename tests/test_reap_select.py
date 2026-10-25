"""Frozen amendment-4 branch-order regression (all CPU, no real model)."""

import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

import pytest

from chowder_batch.reap_gates import GateError
from chowder_batch.reap_select import (
    PAIRED_METHOD_DISCREPANCY, REQUIRED, _paired, _validate_v2_observation,
    evaluate, select,
)
from chowder_batch import reap_v2_source_identity as identity

ROOT = Path(__file__).resolve().parents[1]


def arm(**statuses):
    return {
        "gates": {k: {"status": statuses.get(k, "PASS")} for k in REQUIRED},
    }


def test_v1_not_assumed_pass_if_speed_unknown():
    d = select({"reap128": arm(speed="UNKNOWN")})
    assert d["status"] == "BLOCKED" and d["chosen"] is None


def test_v1_passes_and_v2_not_declared():
    d = select({"reap128": arm()})
    assert d["status"] == "SELECTED" and d["chosen"] == "reap128"


def test_v1_fail_gate1_needs_both_fallbacks_even_if_160_passes():
    d = select({"reap128": arm(gate1="FAIL", speed="NOT_EVALUATED"),
                "reap160": arm()})
    assert d["status"] == "BLOCKED"
    assert "both REAP128_v2 and REAP160" in d["reason"]


def test_v1_fail_gate1_needs_160_scored_even_if_v2_passes():
    d = select({"reap128": arm(gate1="FAIL", speed="NOT_EVALUATED"),
                "reap128_v2": arm()})
    assert d["status"] == "BLOCKED"


def test_fallback_v2_precedes_160_with_full_gate_sets():
    d = select({"reap128": arm(gate1="FAIL", speed="NOT_EVALUATED"),
                "reap128_v2": arm(), "reap160": arm()})
    assert d["status"] == "SELECTED" and d["chosen"] == "reap128_v2"


def test_fallback_160_used_only_after_v2_failed():
    d = select({"reap128": arm(gate1="FAIL", speed="NOT_EVALUATED"),
                "reap128_v2": arm(gate2="FAIL"), "reap160": arm()})
    assert d["status"] == "SELECTED" and d["chosen"] == "reap160"


def test_third_fallback_is_conditional():
    current = {"reap128": arm(gate1="FAIL", speed="NOT_EVALUATED"),
               "reap128_v2": arm(gate1="FAIL"),
               "reap160": arm(speed="FAIL")}
    assert select(current)["status"] == "BLOCKED"
    assert select({**current, "reap160_v2": arm()})["chosen"] == "reap160_v2"
    assert select({**current, "reap160_v2": arm(gate1="FAIL")})["status"] == "FAILED"


def test_v2_normal_replacement_requires_both_strict_noninferiority_bounds():
    current = {"reap128": arm(), "reap128_v2": arm()}
    assert select(current)["status"] == "BLOCKED"
    # The §4 Wald bound is still reported, but the amendment also requires
    # a "paired McNemar" margin CI without specifying its CI construction.
    # Neither a Wald PASS nor a Wald FAIL adjudicates that separate method.
    for bounds in (
        {"gsm8k": 0.0, "humaneval": 0.0},
        {"gsm8k": -0.019, "humaneval": -0.019},
        {"gsm8k": -0.02, "humaneval": -0.019},
        {"gsm8k": -0.019, "humaneval": -0.02},
        {"gsm8k": -0.2, "humaneval": -0.2},
    ):
        verdict = select(current, bounds)
        assert verdict["status"] == "BLOCKED"
        assert verdict["chosen"] is None
        assert verdict["reason"] == PAIRED_METHOD_DISCREPANCY


def test_ambiguity_does_not_halt_valid_v1_or_frozen_fallback_decisions():
    # Optional v2 replacement and mandatory Gate0/1-failure fallback have
    # separate requirements in Amendment 4 §2 vs §3.
    passing_v1 = {"reap128": arm()}
    assert select(passing_v1)["chosen"] == "reap128"
    assert select({**passing_v1, "reap128_v2": arm(gate2="FAIL")})["chosen"] == "reap128"
    needs_fallback = {"reap128": arm(gate1="FAIL", speed="NOT_EVALUATED"),
                      "reap128_v2": arm(), "reap160": arm()}
    assert select(needs_fallback)["chosen"] == "reap128_v2"
    needs_fallback["reap128_v2"] = arm(gate1="FAIL")
    assert select(needs_fallback)["chosen"] == "reap160"
    del needs_fallback["reap160"]
    assert select(needs_fallback)["status"] == "BLOCKED"


def test_v1_coding_regression_does_not_authorize_fallback():
    d = select({"reap128": arm(gate2="FAIL"), "reap128_v2": arm(), "reap160": arm()})
    assert d["status"] == "FAILED" and d["chosen"] is None


def test_d0_fail_denies_every_arm():
    d = select({"reap128": arm(d0="FAIL", gate1="FAIL"),
                "reap128_v2": arm(), "reap160": arm()})
    assert d["status"] == "FAILED" and d["chosen"] is None


def _attest(root, arm, suite, n, *, fail=0, speed=False):
    directory = root / arm
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{suite}.jsonl"
    rows = [
        {"id": f"{suite}-{i}", "correct": i >= fail, "terminated": True,
         "trigram_ratio": 0.9, "predicted_per_second": 42.0 if speed else None}
        for i in range(n)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    entry = {"n": n, "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    summary = directory / "summary.json"
    values = json.loads(summary.read_text(encoding="utf-8")) if summary.exists() else {}
    values[suite] = entry
    summary.write_text(json.dumps(values), encoding="utf-8")
    if speed:
        (directory / "serve.json").write_text(json.dumps({"experts_offloaded": 0}))
    if arm.endswith("_v2") or arm.endswith("_v2_q4"):
        model = {"gguf": f"C:/models/{arm}.gguf", "gguf_sha256": "a" * 64,
                 "flags": ["--fit", "on", "-np", "4", "-c", "16384", "--jinja"]}
        (directory / "serve.json").write_text(json.dumps({**model, "experts_offloaded": 0}))
        (directory / f"{suite}.inputs.json").write_text(json.dumps({
            "schema": "chowder.reap-eval-input-identity.v1", "model": model,
            "suite": suite, "source_items_sha256": "b" * 64,
            "source_items_n": 1319 if suite == "gsm8k" else 164,
            "decode": {"system": "You are a helpful assistant.", "temperature": 0.0,
                       "top_k": 1, "seed": 7, "max_tokens": 1024 if suite == "gsm8k" else 768,
                       "enable_thinking": False},
            "scoring": "frozen reap_eval production scoring: final_number_match/boxed/HumanEval",
        }), encoding="utf-8")
    return path


def _evidence(root):
    for arm, failures in (("dense", 0), ("reap128", 90), ("random128", 1000)):
        _attest(root, arm, "gsm8k", 1319, fail=failures)
    for arm, failures in (("dense", 0), ("reap128", 0)):
        _attest(root, arm, "humaneval", 164, fail=failures)
    _attest(root, "dense_q4", "gsm8k", 200, speed=True)
    _attest(root, "reap128_q4", "gsm8k", 200, speed=True)
    observation = root / "observation.json"
    observation.write_text(json.dumps({"layers": [{"layer": i} for i in range(40)],
                                       "calibration_ppl": 10.646096}), encoding="utf-8")
    return observation


def test_paired_wald_statistic_remains_exactly_frozen_but_not_certified(tmp_path):
    _attest(tmp_path, "reap128", "gsm8k", 1319, fail=10)
    _attest(tmp_path, "reap128_v2", "gsm8k", 1319, fail=4)
    _attest(tmp_path, "reap128", "humaneval", 164, fail=3)
    _attest(tmp_path, "reap128_v2", "humaneval", 164, fail=1)
    observed = _paired(tmp_path)
    assert observed is not None
    for suite, n, improvements in (("gsm8k", 1319, 6), ("humaneval", 164, 2)):
        differences = [1.0] * improvements + [0.0] * (n - improvements)
        registered = (statistics.fmean(differences)
                      - 1.645 * statistics.stdev(differences) / math.sqrt(n))
        assert observed[suite] == pytest.approx(registered, abs=1e-14)
        assert observed[suite] > -0.02
    selection = select({"reap128": arm(), "reap128_v2": arm()}, observed)
    assert selection["status"] == "BLOCKED"
    assert selection["reason"] == PAIRED_METHOD_DISCREPANCY


def test_full_synthetic_raw_evidence_blocks_missing_mandatory_v2(tmp_path):
    obs = _evidence(tmp_path)
    result = evaluate(tmp_path, obs)
    assert result["gates"]["reap128"]["gates"]["gate1"]["status"] == "FAIL"
    assert result["decision"]["status"] == "BLOCKED"
    assert "both REAP128_v2 and REAP160" in result["decision"]["reason"]
    assert result["paired_method"]["status"] == "NOT_EVALUATED"
    assert "mandatory fallback unchanged" in result["paired_method"]["scope"]
    assert result["input_sha256"][str(tmp_path / "dense" / "gsm8k.jsonl")]
    assert result["release_approved"] is False


def _full_v2_rows(root):
    obs = _evidence(root)
    _attest(root, "reap128_v2", "gsm8k", 1319)
    _attest(root, "reap128_v2", "humaneval", 164)
    _attest(root, "reap128_v2_q4", "gsm8k", 200, speed=True)
    return obs


@pytest.mark.parametrize(("target", "mutation", "error"), [
    ("reap128_v2/gsm8k.inputs.json", "delete", "witness"),
    ("reap128_v2_q4/gsm8k.inputs.json", "delete", "witness"),
    ("reap128_v2/humaneval.inputs.json", "delete", "witness"),
    ("reap128_v2/gsm8k.inputs.json", ("schema", "wrong"), "schema"),
    ("reap128_v2/humaneval.inputs.json", ("suite", "gsm8k"), "suite"),
    ("reap128_v2/gsm8k.inputs.json", ("source_items_sha256", "fake"), "source"),
    ("reap128_v2_q4/gsm8k.inputs.json", ("source_items_n", 200), "source"),
    ("reap128_v2_q4/gsm8k.inputs.json", ("source_items_n", 1319.0), "source"),
    ("reap128_v2/gsm8k.inputs.json", ("decode", {}), "decode"),
    ("reap128_v2/humaneval.inputs.json", ("scoring", "other"), "scoring"),
    ("reap128_v2/gsm8k.inputs.json", ("model", {}), "model"),
    ("reap128_v2_q4/gsm8k.inputs.json", ("model.flags", ["--fit", "off"]), "model"),
    ("reap128_v2_q4/gsm8k.inputs.json", ("model.gguf_sha256", "c" * 64), "model"),
    ("reap128_v2_q4/gsm8k.inputs.json", ("model.gguf", "C:/models/foreign.gguf"), "model"),
    ("reap128_v2_q4/serve.json", ("flags", ["--fit", "off"]), "flags"),
    ("reap128_v2/serve.json", ("gguf_sha256", "c" * 64), "model"),
    ("reap128_v2/gsm8k.jsonl", "append", "SHA256"),
    ("reap128_v2_q4/gsm8k.inputs.json", ("source_items_sha256", "c" * 64), "source"),
    ("reap128_v2/gsm8k.inputs.json", "corrupt_json", "unreadable"),
])
def test_uncertified_v2_results_refuse_selection(tmp_path, target, mutation, error):
    obs = _full_v2_rows(tmp_path)
    path = tmp_path / target
    if mutation == "delete":
        path.unlink()
    elif mutation == "corrupt_json":
        path.write_text("{", encoding="utf-8")
    elif mutation == "append":
        with path.open("a", encoding="utf-8") as stream:
            stream.write('{}\n')
    else:
        key, value = mutation
        content = json.loads(path.read_text(encoding="utf-8"))
        if key.startswith("model."):
            content["model"][key.removeprefix("model.")] = value
        else:
            content[key] = value
        path.write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(GateError, match=error):
        evaluate(tmp_path, obs)


def test_v2_full_population_fingerprint_is_bound_across_q8_q4(tmp_path):
    obs = _full_v2_rows(tmp_path)
    report = evaluate(tmp_path, obs)
    assert report["gates"]["reap128_v2"]["gates"]["d0"]["status"] == "UNKNOWN"
    for arm in ("reap128_v2", "reap128_v2_q4"):
        witness = tmp_path / arm / "gsm8k.inputs.json"
        assert report["input_sha256"][str(witness)] == hashlib.sha256(witness.read_bytes()).hexdigest()


def test_v2_q4_gsm8k_window_must_pair_with_q8_item_ids(tmp_path):
    obs = _full_v2_rows(tmp_path)
    path = tmp_path / "reap128_v2_q4" / "gsm8k.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["id"] = "gsm8k-not-in-quality-arm"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    summary_path = path.parent / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["gsm8k"]["file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(GateError, match="paired GSM8K IDs"):
        evaluate(tmp_path, obs)


def test_sha_tampered_raw_baseline_refuses_before_selection(tmp_path):
    obs = _evidence(tmp_path)
    with (tmp_path / "dense" / "gsm8k.jsonl").open("a", encoding="utf-8") as file:
        file.write("{}\n")
    with pytest.raises(GateError, match="SHA256"):
        evaluate(tmp_path, obs)


def test_attested_quality_must_reject_forged_nonbinary_correct(tmp_path):
    obs = _evidence(tmp_path)
    path = tmp_path / "reap128" / "gsm8k.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows[:90]:
        row["correct"] = 2  # fake out-of-range results can conceal 90 actual failures
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    summary_path = path.parent / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["gsm8k"]["file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(GateError, match="correct.*boolean"):
        evaluate(tmp_path, obs)


@pytest.mark.parametrize(("column", "forged", "expected"), [
    ("terminated", "false", "terminated.*boolean"),
    ("trigram_ratio", float("nan"), "trigram_ratio.*finite.*\\[0, 1\\]"),
])
def test_sha_attested_gate0_observations_cannot_forge_pass(tmp_path, column, forged, expected):
    observation = _evidence(tmp_path)
    path = tmp_path / "reap128" / "gsm8k.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows[:40]:
        row[column] = forged
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    summary_path = path.parent / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["gsm8k"]["file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(GateError, match=expected):
        evaluate(tmp_path, observation)


def test_missing_quality_arm_is_not_silently_equivalent(tmp_path):
    obs = _evidence(tmp_path)
    _attest(tmp_path, "reap128_v2", "gsm8k", 1319)
    with pytest.raises(GateError, match="missing full SHA-attested"):
        evaluate(tmp_path, obs)


def test_v2_cannot_borrow_v1_observer_d0(tmp_path, monkeypatch):
    obs = _evidence(tmp_path)
    _attest(tmp_path, "reap128_v2", "gsm8k", 1319)
    _attest(tmp_path, "reap128_v2", "humaneval", 164)
    _attest(tmp_path, "reap128_v2_q4", "gsm8k", 200, speed=True)
    _attest(tmp_path, "reap160", "gsm8k", 1319)
    _attest(tmp_path, "reap160", "humaneval", 164)
    _attest(tmp_path, "reap160_q4", "gsm8k", 200, speed=True)
    report = evaluate(tmp_path, obs)
    assert report["gates"]["reap128_v2"]["gates"]["d0"]["status"] == "UNKNOWN"
    assert report["decision"]["status"] == "BLOCKED"
    assert report["paired_method"]["status"] == "UNRESOLVED"
    assert report["paired_method"]["frozen_margin"] == -0.02
    assert "Wald" in report["paired_method"]["reported_statistic"]
    assert "McNemar" in report["paired_method"]["amendment_4_method"]
    assert report["release_approved"] is False
    v2 = tmp_path / "observation-v2.json"
    v2.write_text(obs.read_text(encoding="utf-8"), encoding="utf-8")
    assert evaluate(tmp_path, obs, v2)["gates"]["reap128_v2"]["gates"]["d0"]["status"] == "UNKNOWN"
    calib = tmp_path / "calib-v2"
    calib.mkdir()
    raw = calib / "calibration.jsonl"
    raw.write_text('{"source":"swe_smith_trajectories","n_tokens":256,"text":"test one"}\n'
                   '{"source":"swe_smith_trajectories","n_tokens":256,"text":"test two"}\n',
                   encoding="utf-8")
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    monkeypatch.setattr(identity, "FROZEN_V2_BASE_ROWS", 2)
    monkeypatch.setattr(identity, "FROZEN_V2_BASE_SHA256", digest)
    (calib / "calibration-manifest.json").write_text(json.dumps(
        {"mix": "v2", "rows": 2, "tokens": 512, "max_tokens": 2048,
         "sources": {"swe_smith_trajectories": {"kept": 2}},
         "calibration_sha256": digest,
         "ornith": "absent: no admitted Ornith curriculum supplied"}),
        encoding="utf-8")
    cloned_observer = json.loads(v2.read_text(encoding="utf-8"))
    cloned_observer.update(calibration_sha256=digest, n_seqs=2, n_tokens=512)
    v2.write_text(json.dumps(cloned_observer), encoding="utf-8")
    report_v2 = evaluate(tmp_path, obs, v2, calib)
    assert report_v2["gates"]["reap128_v2"]["gates"]["d0"]["status"] == "PASS"
    assert str(v2) in report_v2["input_sha256"]
    assert str(raw) in report_v2["input_sha256"]
    (calib / "calibration-manifest.json").write_text(json.dumps(
        {"mix": "v2", "calibration_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(GateError, match="calibration SHA"):
        evaluate(tmp_path, obs, v2, calib)
    (calib / "calibration-manifest.json").write_text(json.dumps(
        {"mix": "v2", "rows": 3, "tokens": 512, "max_tokens": 2048,
         "sources": {"swe_smith_trajectories": {"kept": 2}},
         "calibration_sha256": digest,
         "ornith": "absent: no admitted Ornith curriculum supplied"}),
        encoding="utf-8")
    with pytest.raises(GateError, match="row/token/source census"):
        evaluate(tmp_path, obs, v2, calib)


@pytest.mark.parametrize(("tamper", "reason"), [
    ("partial_observation", "full calibration sequence"),
    ("boolean_sequence_count", "full calibration sequence"),
    ("missing_observed_tokens", "full calibration sequence"),
    ("manifest_token_total", "token total"),
    ("manifest_source_total", "source count"),
    ("rewritten_row_source", "invalid source"),
    ("rewritten_row_token_count", "token total"),
])
def test_v2_d0_refuses_incomplete_or_rewritten_calibration_census(tmp_path, monkeypatch, tamper, reason):
    calib = tmp_path / "calib"
    calib.mkdir()
    rows_path = calib / "calibration.jsonl"
    manifest_path = calib / "calibration-manifest.json"
    obs_path = tmp_path / "observed.json"
    rows = [{"source": "swe_smith_trajectories", "n_tokens": 256, "text": f"sample {i}"}
            for i in range(2)]
    def save_rows():
        rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return hashlib.sha256(rows_path.read_bytes()).hexdigest()
    digest = save_rows()
    monkeypatch.setattr(identity, "FROZEN_V2_BASE_ROWS", 2)
    monkeypatch.setattr(identity, "FROZEN_V2_BASE_SHA256", digest)
    manifest = {"mix": "v2", "rows": 2, "tokens": 512, "max_tokens": 2048,
                "sources": {"swe_smith_trajectories": {"kept": 2}},
                "ornith": "absent: no admitted Ornith curriculum supplied",
                "calibration_sha256": digest}
    observed = {"calibration_sha256": digest, "n_seqs": 2, "n_tokens": 512,
                "calibration_ppl": 10, "layers": [{"layer": i} for i in range(40)]}
    if tamper == "partial_observation":
        observed["n_seqs"] = 1  # --limit-seqs 1 could still record 40 layers and PPL < 20
        observed["n_tokens"] = 256
    elif tamper == "boolean_sequence_count":
        observed["n_seqs"] = True
    elif tamper == "missing_observed_tokens":
        observed.pop("n_tokens")
    elif tamper == "manifest_token_total":
        manifest["tokens"] = 513
    elif tamper == "manifest_source_total":
        manifest["sources"]["swe_smith_trajectories"]["kept"] = 1
    elif tamper == "rewritten_row_source":
        rows[0]["source"] = "undeclared"
    elif tamper == "rewritten_row_token_count":
        rows[0]["n_tokens"] = 257
    if tamper.startswith("rewritten_row"):
        digest = save_rows()
        manifest["calibration_sha256"] = observed["calibration_sha256"] = digest
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    obs_path.write_text(json.dumps(observed), encoding="utf-8")
    with pytest.raises(GateError, match=reason):
        _validate_v2_observation(calib, obs_path)
    if tamper == "partial_observation":
        observed["n_seqs"], observed["n_tokens"] = 2, 512
        obs_path.write_text(json.dumps(observed), encoding="utf-8")
        _validate_v2_observation(calib, obs_path)


@pytest.mark.parametrize(("ornith", "source"), [
    ({"render": "bare"}, "swe_smith_trajectories"),
    ("absent: no admitted Ornith curriculum supplied", "swe_smith_trajectories"),
])
def test_v2_accepts_only_preregistered_bare_or_declared_absent_ornith(
    tmp_path, monkeypatch, ornith, source
):
    calib = tmp_path / "calib"
    calib.mkdir()
    text = json.dumps({"source": source, "n_tokens": 256, "text": "valid calibration"}) + "\n"
    raw = calib / "calibration.jsonl"
    raw.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    monkeypatch.setattr(identity, "FROZEN_V2_BASE_ROWS", 1)
    monkeypatch.setattr(identity, "FROZEN_V2_BASE_SHA256", digest)
    manifest = {"mix": "v2", "rows": 1, "tokens": 256, "max_tokens": 2048,
                "sources": {source: {"kept": 1}}, "ornith": ornith,
                "calibration_sha256": digest}
    if isinstance(ornith, dict):
        admitted = calib / "admitted.jsonl"
        admitted.write_text('{"admitted":true}\n', encoding="utf-8")
        pinned = hashlib.sha256(admitted.read_bytes()).hexdigest()
        monkeypatch.setattr(identity, "FROZEN_ORNITH_ADMITTED_SHA256", pinned)
        manifest["ornith"] = {"render": "bare", "path": str(admitted), "sha256": pinned}
        manifest["sources"]["ornith_curriculum"] = {"kept": 0, "pool": 39, "quota": 64}
    manifest_path = calib / "calibration-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    observed = tmp_path / "observation.json"
    observed.write_text(json.dumps({"calibration_sha256": digest, "n_seqs": 1, "n_tokens": 256}))
    _validate_v2_observation(calib, observed)

    if isinstance(ornith, dict):
        for invalid in ("worked", "WORKED", "", None):
            manifest["ornith"] = {"render": invalid}
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with pytest.raises(GateError, match="not preregistered"):
                _validate_v2_observation(calib, observed)
    else:
        manifest.pop("ornith")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(GateError, match="declaration is missing"):
            _validate_v2_observation(calib, observed)


@pytest.mark.parametrize("entrypoint", ["module", "script"])
def test_v2_worked_is_refused_by_both_selector_entrypoints(tmp_path, entrypoint):
    v1 = _evidence(tmp_path)
    _attest(tmp_path, "reap128_v2", "gsm8k", 1319)
    _attest(tmp_path, "reap128_v2", "humaneval", 164)
    _attest(tmp_path, "reap128_v2_q4", "gsm8k", 200, speed=True)
    calib = tmp_path / "calib"
    calib.mkdir()
    raw = calib / "calibration.jsonl"
    raw.write_text('{"source":"ornith_curriculum","n_tokens":256,"text":"worked row"}\n')
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    manifest_path = calib / "calibration-manifest.json"
    manifest = {"mix": "v2", "rows": 1, "tokens": 256, "max_tokens": 2048,
                "sources": {"ornith_curriculum": {"kept": 1}},
                "calibration_sha256": digest, "ornith": {"render": "worked"}}
    manifest_path.write_text(json.dumps(manifest))
    v2 = tmp_path / "observer-v2.json"
    v2.write_text(json.dumps({"calibration_sha256": digest, "n_seqs": 1, "n_tokens": 256,
                              "calibration_ppl": 10,
                              "layers": [{"layer": i} for i in range(40)]}))

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("PYTHONPATH", None)
    if entrypoint == "module":
        env["PYTHONPATH"] = str(ROOT)
        entry = ["-m", "chowder_batch.reap_select"]
    else:
        entry = [str(ROOT / "chowder_batch" / "reap_select.py")]

    def invoke(name):
        command = [sys.executable, *entry, "--eval-dir", str(tmp_path),
                   "--observation", str(v1), "--observation-v2", str(v2),
                   "--calibration-v2", str(calib), "--out", str(tmp_path / name)]
        return subprocess.run(command, cwd=tmp_path, env=env, capture_output=True,
                              text=True, check=False)

    refused = invoke("worked-selection.json")
    assert refused.returncode == 2
    assert "Ornith render is not preregistered" in refused.stderr
    assert not (tmp_path / "worked-selection.json").exists()

    manifest["ornith"]["render"] = "bare"
    manifest_path.write_text(json.dumps(manifest))
    refused_bare = invoke("bare-unregistered.json")
    assert refused_bare.returncode == 2
    assert "frozen v2 base identity mismatch" in refused_bare.stderr
    assert not (tmp_path / "bare-unregistered.json").exists()


def test_v2_selector_refuses_forged_13_bare_suffix_with_selfconsistent_observer(
    tmp_path, monkeypatch
):
    calib = tmp_path / "calibration"
    calib.mkdir()
    admitted = calib / "admitted.jsonl"
    admitted.write_text('{"admitted":true,"lane":"sft","expected_answer":"42"}\n')
    admitted_sha = hashlib.sha256(admitted.read_bytes()).hexdigest()
    monkeypatch.setattr(identity, "FROZEN_ORNITH_ADMITTED_SHA256", admitted_sha)
    base = "".join(json.dumps({"source": "swe_smith_trajectories", "n_tokens": 256,
                               "text": f"base {i}"}) + "\n" for i in range(3))
    monkeypatch.setattr(identity, "FROZEN_V2_BASE_ROWS", 3)
    monkeypatch.setattr(identity, "FROZEN_V2_BASE_SHA256",
                        hashlib.sha256(base.encode()).hexdigest())
    suffix = "".join(json.dumps({"source": "ornith_curriculum", "n_tokens": 256,
                                 "text": f"FAKED bare content {i}"}) + "\n"
                     for i in range(13))
    raw = calib / "calibration.jsonl"
    raw.write_bytes((base + suffix).encode("utf-8"))
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    (calib / "calibration-manifest.json").write_text(json.dumps({
        "mix": "v2", "rows": 16, "tokens": 16 * 256, "max_tokens": 2048,
        "calibration_sha256": digest,
        "sources": {"swe_smith_trajectories": {"kept": 3},
                    "ornith_curriculum": {"kept": 13, "quota": 64, "pool": 39}},
        "ornith": {"render": "bare", "path": str(admitted), "sha256": admitted_sha},
    }))
    observer = tmp_path / "observer.json"
    observer.write_text(json.dumps({"calibration_sha256": digest,
                                   "n_seqs": 16, "n_tokens": 4096}))
    with pytest.raises(GateError, match="zero eligible rows"):
        _validate_v2_observation(calib, observer)


def test_registered_704_v2_refuses_silent_legacy_checkpoint_downgrade(tmp_path, monkeypatch):
    calib = tmp_path / "calibration"
    calib.mkdir()
    rows = "".join(
        json.dumps({"source": "swe_smith_trajectories", "n_tokens": 256,
                    "text": f"frozen-test-row-{i}"}) + "\n"
        for i in range(704)
    )
    source = calib / "calibration.jsonl"
    source.write_text(rows, encoding="utf-8")
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(identity, "FROZEN_V2_BASE_SHA256", sha)
    (calib / "calibration-manifest.json").write_text(json.dumps({
        "mix": "v2", "rows": 704, "tokens": 704 * 256, "max_tokens": 2048,
        "calibration_sha256": sha,
        "sources": {"swe_smith_trajectories": {"kept": 704}},
        "ornith": "absent: no admitted Ornith curriculum supplied",
    }))
    observer = tmp_path / "observation.json"
    observer.write_text(json.dumps({
        "calibration_sha256": sha, "n_seqs": 704, "n_tokens": 1_192_632,
        "layers": [{"layer": i, "routed_tokens": 9_541_056} for i in range(40)],
        "calibration_ppl": 10.0,
        # Both self-identifying fields deliberately stripped.
    }))
    with pytest.raises(GateError, match="requires sealed BF16 checkpoint receipt"):
        _validate_v2_observation(calib, observer)
