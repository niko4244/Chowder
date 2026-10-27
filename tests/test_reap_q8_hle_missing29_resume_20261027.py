"""Adversarial CPU-only, no-network tests; synthetic fixtures are NOT model scores."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from chowder_batch import reap_q8_hle_missing29_resume_20261027 as m


def _json(path: Path) -> dict:
    return json.loads(path.read_bytes())


def _write(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _rows_write(path: Path, rows: list[dict]) -> None:
    path.write_bytes(b"".join(
        (json.dumps(r, sort_keys=True) + "\n").encode() for r in rows))


@pytest.fixture(scope="module")
def available() -> None:
    if not m.V4.exists() or not m.V3_SCRIPT.exists() or not m.frozen.Q4_ROWS_DIR.exists():
        pytest.skip("authentic private Q8/Q4 HLE evidence not mounted on this CPU test host")


@pytest.fixture
def fresh(tmp_path: Path, available: None) -> Path:
    """Produce ONLY invented adversarial rows in tmp_path, never a scored dataset."""
    original99, missing29, _ = m.original_and_missing()
    assert len(original99) == 99 and len(missing29) == 29
    q4_rows, _ = m._rows(
        m.frozen.Q4_ROWS_DIR, "reap-q4-hle-textonly-*.jsonl",
        expected=m.frozen.Q4_ROWS_SHA256)
    types = {row["id"]: row["answer_type"] for row in q4_rows}
    rows = [
        {"id": id_, "answer_type": types[id_], "correct": False,
         "graded": True, "truncated": False, "completion_tokens": 25,
         "seconds": 2.0, "response_chars": 60,
         "eval_session_id": "0123456789abcdef0123456789abcdef"} for id_ in missing29
    ]
    folder = tmp_path / "invented_adversarial_second_session"
    folder.mkdir()
    source = _json(m.V4 / m.V4_RECEIPT)
    source["session_id"] = "0123456789abcdef0123456789abcdef"
    source["server_pid"] = 923
    source["cmd"][9] = "8793"
    source["status"] = "COMPLETE_Q8_MISSING29_HLE_TEXT_ONLY_RESEARCH_NOT_OFFICIAL_JUDGE"
    source["resume_generator_template_sha256"] = m.V3_SCRIPT_SHA
    source["resume_prior_v4_rows_sha256"] = m.V4_ROWS_SHA
    source["resume_prior_v4_session_id"] = m.V4_SESSION
    source["resume_q4_original_rows_sha256"] = m.Q4_SESSION_SHA
    source["resume_missing29_ids_sha256"] = m.MISSING_IDS_SHA
    source["population"]["planned_n"] = 29
    source["samples"] = [{
        "physical_free_gib": 25.0, "cgroup_unused_gib": 18.0,
        "gpu_free_mib": [5000, 5100]}]
    log = b"load_tensors: offloaded 41/41 layers to GPU\nllama_context: n_ctx_seq             = 8192\n"
    (folder / m.NEW_LOG).write_bytes(log)
    source["server_log_sha256"] = m.sha(log)
    source["server_log_bytes"] = len(log)
    data_path = folder / f"reap-q8-missing29-hle-textonly-{source['session_id']}.jsonl"
    _rows_write(data_path, rows)
    summ = _json(next(m.V4.glob("reap-q8-matched128-hle-textonly-*-summary.json")))
    summ.update({"completed_n": 29, "planned_n": 29, "graded_n": 29,
                 "not_attempted_n": 0, "correct": 0,
                 "completion_fraction_of_population": 29 / 2158,
                 "completion_tokens": 29 * 25, "session_id": source["session_id"],
                 "row_file_sha256": m.sha(data_path.read_bytes()),
                 "by_answer_type": {
                     cat: {"correct": 0, "n": sum(r["answer_type"] == cat for r in rows)}
                     for cat in ("exactMatch", "multipleChoice")},
                 "accuracy": 0.0,
                 "generation_wall_seconds": 58.0})
    _write(folder / f"reap-q8-missing29-hle-textonly-{source['session_id']}-summary.json",
           summ)
    source["summary"] = summ
    _write(folder / m.NEW_RECEIPT, source)
    return folder


def _alter_receipt(folder: Path, mutate) -> None:
    path = folder / m.NEW_RECEIPT
    obj = _json(path)
    mutate(obj)
    _write(path, obj)


def _alter_rows(folder: Path, mutate, *, repair_summary: bool = True) -> None:
    rowfile = next(folder.glob(m.NEW_PATTERN))
    rows = [json.loads(x) for x in rowfile.read_bytes().splitlines()]
    mutate(rows)
    _rows_write(rowfile, rows)
    if repair_summary:
        receipt = _json(folder / m.NEW_RECEIPT)
        receipt["summary"]["row_file_sha256"] = m.sha(rowfile.read_bytes())
        summary = next(folder.glob("reap-q8-missing29-hle-textonly-*-summary.json"))
        _write(summary, receipt["summary"])
        _write(folder / m.NEW_RECEIPT, receipt)


def test_actual_original_source_census(available: None):
    old, missing, digest = m.original_and_missing()
    assert (len(old), len(missing), digest) == (99, 29, m.V4_ROWS_SHA)
    assert not set(old) & set(missing)
    assert len(set(old) | set(missing)) == 128
    assert hashlib.sha256(
        json.dumps(missing, separators=(",", ":")).encode()).hexdigest() == m.MISSING_IDS_SHA


def test_cpu_only_private_script_hardens_original_gates(tmp_path: Path, available: None):
    out = tmp_path / "new_private_kaggle"
    result = m.prepare(out)
    assert result["status"] == "PRIVATE_Q8_MISSING29_PREPARED_NOT_SUBMITTED"
    assert result["missing_n"] == 29 and result["existing_untouched_n"] == 99
    assert result["official_hle_judge_score"] == "UNAVAILABLE"
    script = next(out.glob("*.py")).read_text()
    tree = ast.parse(script)
    compile(tree, "private_test_runner", "exec")  # compile does NOT run inference.
    assert "MISSING_Q8_IDS = set(" in script
    assert "PRIOR_Q8_IDS = set(" in script
    assert "if x['id'] in MISSING_Q8_IDS" in script
    assert "len(ordered) != 29" in script
    assert "len(ordered) != 128" not in script
    assert "offloaded 41/41 layers to GPU" in script
    assert "cgroup_unused_gib'] < 10" in script
    assert "cgroup_unused_gib'] < 14" in script
    assert "LLAMA_ARG_LOG_VERBOSITY='4'" in script
    assert "PORT = 8793" in script
    assert "COMPLETE_Q8_MISSING29_HLE_TEXT_ONLY_RESEARCH_NOT_OFFICIAL_JUDGE" in script
    assert "R['data'] = {'q8_gguf_sha256': Q8_GGUF_SHA256" in script
    assert "P.grade(text, item['answer'], item['answer_type'])" in script
    assert "P.build_messages(item['question'])" in script
    assert "P.SERVING_FLAGS" in script
    meta = _json(out / "kernel-metadata.json")
    assert meta["is_private"] is True
    assert meta["enable_internet"] is False
    assert meta["enable_gpu"] is True
    with pytest.raises(ValueError, match="unused"):
        m.prepare(out)


def test_fixture_two_separate_native_sessions_not_one_attestation(fresh: Path):
    result = m.audit_two_sessions(fresh)
    assert result["status"] == "TWO_NATIVE_SESSIONS_PAIRED_HEURISTIC_RESEARCH_NOT_OFFICIAL_HLE"
    assert result["first_session_id"] != result["second_session_id"]
    assert result["first_native_log_sha256"] != result["second_native_log_sha256"]
    assert result["combined_source_ids_n"] == 128
    assert result["single_native_session_claim"] is False
    assert result["independent_raw_response_rescore"] is False
    assert result["official_hle_judge_score"] == "UNAVAILABLE"
    assert result["scientific_selection"] == "BLOCKED"


@pytest.mark.parametrize("name,fn", [
    ("old session", lambda r: r.__setitem__("session_id", m.V4_SESSION)),
    ("wrong source", lambda r: r.__setitem__("resume_prior_v4_rows_sha256", "0" * 64)),
    ("wrong template", lambda r: r.__setitem__("resume_generator_template_sha256", "0" * 64)),
    ("wrong q4", lambda r: r.__setitem__("resume_q4_original_rows_sha256", "0" * 64)),
    ("wrong missing", lambda r: r.__setitem__("resume_missing29_ids_sha256", "0" * 64)),
    ("wrong original protocol", lambda r: r.__setitem__("protocol_sha256", "0" * 64)),
    ("wrong model", lambda r: r["data"].__setitem__("q8_gguf_sha256", "0" * 64)),
    ("wrong native", lambda r: r.__setitem__("verified_native_offload_41_of_41", False)),
    ("wrong flags", lambda r: r["cmd"].__setitem__(2, "off")),
    ("lower cgroup", lambda r: r["samples"][0].__setitem__("cgroup_unused_gib", 9.9)),
    ("nonfinite sample", lambda r: r["samples"][0].__setitem__("cgroup_unused_gib", float("inf"))),
    ("lower gpu", lambda r: r["samples"][0].__setitem__("gpu_free_mib", [4000, 5000])),
    ("bad startup", lambda r: r["initial"].__setitem__("cgroup_unused_gib", 23.0)),
    ("bad initial gpu", lambda r: r["initial"].__setitem__("gpu_free_mib", [12000, 14000])),
    ("bad postload", lambda r: r["after_postload_reclaim"].__setitem__("cgroup_unused_gib", 13.0)),
    ("future official", lambda r: r.__setitem__("official_hle_score_claimed", True)),
    ("changed sample count", lambda r: r["population"].__setitem__("planned_n", 128)),
    ("already chosen", lambda r: r.__setitem__("chosen", "q8")),
    ("missing scorer", lambda r: r["summary"].__setitem__("metric", "made-up-metric")),
])
def test_receipt_attacks_block(fresh: Path, name, fn):
    _alter_receipt(fresh, fn)
    with pytest.raises(ValueError):
        m.audit_two_sessions(fresh)


@pytest.mark.parametrize("name,fn", [
    ("duplicate id", lambda r: r[0].__setitem__("id", r[1]["id"])),
    ("different id", lambda r: r[0].__setitem__("id", "forged-HLE-ID")),
    ("ungraded", lambda r: r[0].__setitem__("graded", False)),
    ("truncated", lambda r: r[0].__setitem__("truncated", True)),
    ("invalid count", lambda r: r[0].__setitem__("completion_tokens", -1)),
    ("changed session", lambda r: r[0].__setitem__("eval_session_id", m.V4_SESSION)),
    ("heuristic bit type", lambda r: r[0].__setitem__("correct", 0)),
    ("wrong answer type", lambda r: r[0].__setitem__("answer_type", "caption")),
])
def test_rows_attacks_block(fresh: Path, name, fn):
    _alter_rows(fresh, fn)
    with pytest.raises(ValueError):
        m.audit_two_sessions(fresh)


def test_log_tamper_rejected(fresh: Path):
    (fresh / m.NEW_LOG).write_bytes(b"unverified log\n")
    with pytest.raises(ValueError):
        m.audit_two_sessions(fresh)


def test_row_partial_tail_rejected(fresh: Path):
    p = next(fresh.glob(m.NEW_PATTERN))
    p.write_bytes(p.read_bytes().removesuffix(b"\n"))
    with pytest.raises(ValueError):
        m.audit_two_sessions(fresh)


def test_row_json_duplicate_key_rejected(fresh: Path):
    p = next(fresh.glob(m.NEW_PATTERN))
    raw = p.read_bytes()
    p.write_bytes(raw.replace(b'"graded": true', b'"graded": true, "graded": true', 1))
    with pytest.raises(ValueError):
        m.audit_two_sessions(fresh)


def test_incomplete_or_ambiguous_fresh_session_blocked(fresh: Path):
    path = next(fresh.glob(m.NEW_PATTERN))
    other = fresh / "reap-q8-missing29-hle-textonly-spurious.jsonl"
    other.write_bytes(path.read_bytes())
    with pytest.raises(ValueError):
        m.audit_two_sessions(fresh)


def test_absent_real_second_session_never_claims_score(available: None):
    with pytest.raises(ValueError):
        m.audit_two_sessions(m.V4)


def test_preflight_mutated_v4_log_fails(tmp_path: Path, available: None):
    import shutil

    temp = tmp_path / "fake_v4"
    temp.mkdir()
    for name in (m.V4_RECEIPT, m.OLD_LOG):
        shutil.copyfile(m.V4 / name, temp / name)
    for pat in (m.OLD_PATTERN, "reap-q8-matched128-hle-textonly-*-summary.json"):
        src = next(m.V4.glob(pat))
        shutil.copyfile(src, temp / src.name)
    (temp / m.OLD_LOG).write_bytes(b"not the real native log")
    with pytest.raises(ValueError):
        m.original_and_missing(temp)
