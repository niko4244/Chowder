"""Focused tests for Phase 2/3: quality gates, provenance records, replay evidence."""
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

EXP = Path(__file__).resolve().parents[1] / "experiments" / "teacher_free_distill"


def load(name):
    spec = importlib.util.spec_from_file_location(name, EXP / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


prepare = load("prepare")
fetch = load("fetch_smith")
replay = load("replay_smith")


def catalog(tmp_path, approved=True, kind="chat"):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({"sources": {"test": {
        "approved": approved, "license": "test-only", "kind": kind,
        "review_reference": "fixture", "revision": "rev-1"}}}))
    return path


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf8")


def test_near_duplicate_quarantined_and_reported(tmp_path):
    inp = tmp_path / "in.jsonl"
    write_jsonl(inp, [
        {"messages": [{"role": "user", "content": "how to parse json in python quickly"},
                      {"role": "assistant", "content": "use the json module"}]},
        {"messages": [{"role": "user", "content": "how to parse json in python quickly and easily"},
                      {"role": "assistant", "content": "use the json module"}]},
        {"messages": [{"role": "user", "content": "completely different topic about golf"},
                      {"role": "assistant", "content": "swing straighter"}]},
    ])
    out = tmp_path / "out"
    result = prepare.prepare(catalog(tmp_path), {"test": inp}, out, max_rows=10)
    assert result["counts"]["test:near_duplicate_quarantined"] == 1
    assert result["quality_report"]["totals"]["quarantined"] == 1
    quarantined = (out / "quarantine.jsonl").read_text().strip().splitlines()
    row = json.loads(quarantined[0])
    assert row["reason"] == "near_duplicate" and row["source"] == "test"


def test_provenance_manifest_fields(tmp_path):
    inp = tmp_path / "in.jsonl"
    write_jsonl(inp, [{"messages": [{"role": "user", "content": "what is 9 times 3"},
                                    {"role": "assistant", "content": "27"}]}])
    out = tmp_path / "out"
    result = prepare.prepare(catalog(tmp_path), {"test": inp}, out, max_rows=5)
    prov = result["provenance"][0]
    assert prov["source"] == "test" and prov["revision"] == "rev-1"
    assert prov["license"] == "test-only"
    assert prov["content_sha256"] and prov["char_count"] > 0
    assert prov["split"] in ("train", "dev")
    assert prov["token_count"] is None  # no TFD_TOKENIZER configured in tests


def test_fetch_requires_approved_repair_source(tmp_path):
    with pytest.raises(PermissionError):
        fetch._catalog_source(catalog(tmp_path, approved=False, kind="repair"), "test")
    with pytest.raises(ValueError):
        fetch._catalog_source(catalog(tmp_path, approved=True, kind="chat"), "test")


def test_fetch_preserves_claim_not_evidence(tmp_path, monkeypatch):
    src = catalog(tmp_path, kind="repair")
    # fetch_smith reads repo/revision off the catalog source record.
    src.write_text(json.dumps({"sources": {"test": {
        "approved": True, "license": "test-only", "kind": "repair",
        "review_reference": "fixture", "revision": "rev-1",
        "repo": "fixture/trajectories"}}}))
    rows = [{"traj_id": "t1", "instance_id": "repo/pkg.abc123.fix__wxyz1234",
             "resolved": "true", "model": "m", "messages": [], "patch": "diff"},
            {"traj_id": "t2", "instance_id": "repo/pkg.abc123.fix__abcd5678",
             "resolved": "false", "model": "m", "messages": [], "patch": "diff"}]

    monkeypatch.setattr("datasets.load_dataset", lambda *a, **k: iter(rows))
    dest = tmp_path / "traj.jsonl"
    fetch.export(src, "test", dest, limit=2, scan_limit=10, resolved_only=False)
    saved = [json.loads(x) for x in dest.read_text().splitlines()]
    assert saved[0]["claimed_resolved"] is True
    # A raw fetch must NEVER carry verification evidence - that comes only
    # from replay_smith.py. The dataset's own flag stays a labeled claim.
    assert "verification" not in saved[0]
    export_meta = json.loads((tmp_path / "traj.jsonl.export.json").read_text())
    assert export_meta["verification"].startswith("NONE")
    assert saved[0]["messages"] == []  # raw passthrough; nothing invented


def test_replay_refuses_without_podman(monkeypatch, tmp_path):
    calls = {"n": 0}

    def fake_run(*a, **k):
        calls["n"] += 1

        class R:
            returncode = 1
            stdout = ""
            stderr = ""

        return R()

    monkeypatch.setattr(replay.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="podman"):
        replay.replay_batch(tmp_path / "in.jsonl", tmp_path / "work",
                            tmp_path / "out.jsonl")


def test_replay_skips_without_instance_metadata(tmp_path):
    record = {"traj_id": "t1", "instance_id": "x", "messages": [], "patch": "diff"}
    result = replay.replay_one(record, tmp_path / "work", instance=None)
    assert result["status"] == "skipped"
    assert "base_commit" in result["reason"]


def test_verified_replay_record_yields_repair_examples(tmp_path):
    events = [
        {"kind": "tool", "action": {"tool": "read_file", "path": "a.py"},
         "observation": "contents", "returncode": 0, "verdict": "verified_good"},
        {"kind": "tool", "action": {"tool": "edit_file", "path": "a.py"},
         "observation": "edited", "returncode": 0, "verdict": "verified_good"},
        {"kind": "test", "action": {"tool": "run_tests"},
         "observation": "3 passed", "returncode": 0, "tests_executed": 3,
         "verdict": "verified_good"},
    ]
    record = {"task_id": "t1", "repository": "repo/pkg", "task": "fix x",
              "events": events,
              "verification": {"method": "sandbox_replay", "returncode": 0,
                               "tests_executed": 3,
                               "trace_sha256": prepare.digest(events)}}
    examples = prepare.repair_examples(record)
    assert len(examples) == 3
    assert all(x["messages"][-1]["role"] == "assistant" for x in examples)
    assert all(x["group"] == "repair:repo/pkg:t1" for x in examples)


def test_eval_repair_split_leakage_detection():
    train = [{"group": "repair:repoA:1"}, {"group": "repair:repoA:2"}]
    ev = [{"group": "repair:repoA:9"}]
    check = evaluate_check(train, ev)
    assert not check["ok"] and check["leaked_repositories"] == ["repoA"]
    ev2 = [{"group": "repair:repoB:1"}]
    assert evaluate_check(train, ev2)["ok"]


def evaluate_check(train, ev):
    import importlib.util as iu
    spec = iu.spec_from_file_location("ev", EXP / "evaluate.py")
    mod = iu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.check_repair_split_leakage(train, ev)


def test_gsm8k_scoring():
    import importlib.util as iu
    spec = iu.spec_from_file_location("ev", EXP / "evaluate.py")
    mod = iu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.gsm8k_correct("The answer is #### 42", "42")
    assert mod.gsm8k_correct("#### 1,200", "1200")
    assert not mod.gsm8k_correct("I think it is 41", "42")


def test_repair_behavior_metrics():
    import importlib.util as iu
    spec = iu.spec_from_file_location("ev", EXP / "evaluate.py")
    mod = iu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    traj = [
        {"action": {"tool": "read_file", "path": "missing.py"}, "returncode": 1,
         "observation": "file does not exist"},
        {"action": {"tool": "edit_file", "path": "a.py"}, "returncode": 0,
         "observation": "edited"},
        {"action": {"tool": "run_tests"}, "returncode": 1, "observation": "1 failed"},
        {"action": {"tool": "edit_file", "path": "a.py"}, "returncode": 0,
         "observation": "edited again"},
        {"action": {"tool": "run_tests"}, "returncode": 0, "observation": "2 passed",
         "tests_observed": True},
    ]
    m = mod.repair_behaviors(traj)
    assert m["nonexistent_reads"] == 1
    assert m["recovered_after_failed_tests"] is True
    assert m["green_completion"] is True
    assert m["most_repeated_action_count"] == 2
    agg = mod.repair_aggregate([traj])
    assert agg["green_completion_rate"] == 1.0


def test_repair_metrics_cli_reads_jsonl_rows(tmp_path, capsys, monkeypatch):
    """The repair-metrics CLI consumes one trajectory per JSONL row."""
    import importlib.util as iu
    spec = iu.spec_from_file_location("ev", EXP / "evaluate.py")
    mod = iu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    traj = [{"action": {"tool": "run_tests"}, "returncode": 0,
             "observation": "1 passed", "tests_observed": True}]
    path = tmp_path / "trajs.jsonl"
    path.write_text(json.dumps(traj) + "\n", encoding="utf8")
    monkeypatch.setattr("sys.argv", ["evaluate.py", "repair-metrics",
                                     "--trajectories", str(path)])
    mod.main()
    out = json.loads(capsys.readouterr().out)
    assert out["tasks"] == 1
    assert out["green_completion_rate"] == 1.0


def test_prompt_overlap_tolerates_non_dict_messages():
    import importlib.util as iu
    spec = iu.spec_from_file_location("ev", EXP / "evaluate.py")
    mod = iu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    train = [{"messages": [{"role": "user", "content": "Hello  World"}]}]
    ev = [{"messages": [{"role": "user", "content": "hello\tworld"}, "junk"]}]
    check = mod.check_prompt_overlap(train, ev)
    assert check["collisions"] == 1 and not check["ok"]


def test_derive_test_targets_from_patch():
    source_only = "--- a/src/apispec/core.py\n+++ b/src/apispec/core.py\n@@ -1 +1 @@\n-a\n+b\n"
    targets, source = replay.derive_test_targets(source_only)
    assert targets == ["core"] and source == "patch_derived_stems"

    test_touched = "--- a/x.py\n+++ b/tests/test_core.py\n@@ -1 +1 @@\n-a\n+b\n"
    targets, source = replay.derive_test_targets(test_touched)
    assert targets == ["tests/test_core.py"] and source == "patch_touches_tests"

    targets, source = replay.derive_test_targets("no diff headers")
    assert targets == [] and source == "default"


def test_write_patch_file_never_emits_crlf(tmp_path):
    """A CRLF patch can never apply to an LF container worktree.

    Regression: Windows text-mode writes turned every patch into CRLF, which
    made all 23 replays fail at ``git apply`` for reasons unrelated to the
    trajectories being replayed.
    """
    target = replay.write_patch_file(
        tmp_path / "patch.diff",
        "--- a/x.py\r\n+++ b/x.py\r\n@@ -1 +1 @@\r\n-a\r\n+b\r\n")
    raw = target.read_bytes()
    assert b"\r" not in raw
    assert raw.count(b"\n") == 5


def test_failed_record_keeps_evidence_written_before_the_crash(tmp_path):
    """A row that dies mid-way keeps the commands that already ran.

    Regression: everything lived in memory until the final record, so a
    saturated podman daemon timing out a cleanup call discarded a fully
    committed red-image phase and left no way to triage the row.
    """
    task = tmp_path / "t1"
    task.mkdir()
    replay._persist_log(task, [{"command": "commit red state", "returncode": 0}])
    record = replay._failed_record(task, "t1", "podman call timed out: rm -f")
    assert record["status"] == "setup_failed"
    assert record["last_command"] == {"command": "commit red state", "returncode": 0}
    assert record["replay_log"].endswith("replay_log.json")

    bare = replay._failed_record(tmp_path / "never-ran", "t2", "boom")
    assert bare["reason"] == "boom" and "replay_log" not in bare


def test_best_effort_cleanup_never_raises(monkeypatch):
    """Cleanup timeouts must not fail the row that already succeeded."""
    def boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="podman rm -f", timeout=300)

    monkeypatch.setattr(replay, "run_podman", boom)
    replay._best_effort(["rm", "-f", "chowder-setup-deadbeef"])


def test_replay_batch_reports_setup_failures_under_their_own_name(tmp_path, monkeypatch):
    """Summary buckets must not conflate infra failure with a failed test run,
    and skipped rows must not consume the row budget."""
    inp = tmp_path / "in.jsonl"
    write_jsonl(inp, [
        {"traj_id": "t1", "instance_id": "a", "expect": "verified"},
        {"traj_id": "t2", "instance_id": "b", "expect": "setup_failed"},
        {"traj_id": "t3", "instance_id": "c", "expect": "skipped"},
        {"traj_id": "t4", "instance_id": "d", "expect": "not_green"},
    ])

    class R:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(replay, "ensure_base_image", lambda: None)
    monkeypatch.setattr(replay.subprocess, "run", lambda *a, **k: R())
    monkeypatch.setattr(replay, "replay_one",
                        lambda row, work, instance=None: {"task_id": row["traj_id"],
                                                          "status": row["expect"]})
    summary = replay.replay_batch(inp, tmp_path / "work", tmp_path / "out.jsonl",
                                  limit=3)
    assert summary == {"rows": 4, "verified": 1, "not_green": 1,
                       "setup_failed": 1, "skipped": 1}
    assert json.loads((tmp_path / "replay_summary.json").read_text()) == summary


def test_parse_test_summary_counts_pure_failure_runs():
    """A red run with no passes must still report its executed tests.

    Regression: counting only passed+failed scored "11 failed" as 0 executed
    tests, so a genuinely reproduced failure was labelled "nothing ran".
    """
    assert replay.parse_test_summary("11 failed in 0.82s") == 11
    assert replay.parse_test_summary("2 failed, 1 passed in 0.57s") == 3
    assert replay.parse_test_summary("no tests ran in 0.01s") == 0
    assert replay.parse_test_summary("Ran 7 tests in 0.301s") == 7


def test_finish_gates_verification_and_records_apply_rc(tmp_path):
    """apply_rc separates "patch did not apply" from "applied, still red";
    only genuine red->green may carry a verification block."""
    not_applied = tmp_path / "a"
    not_applied.mkdir()
    record = replay._finish(not_applied, "t1", [], [], "not_green", returncode=1,
                            apply_rc=1, tests_executed=11, failure_reproduced=True)
    assert record["status"] == "not_green" and record["apply_rc"] == 1
    assert "verification" not in record
    assert (not_applied / "replay_log.json").exists()

    green = tmp_path / "b"
    green.mkdir()
    verified = replay._finish(green, "t2", [{"kind": "test", "returncode": 0}], [],
                              "verified", returncode=0, apply_rc=0,
                              tests_executed=2, failure_reproduced=True)
    assert verified["status"] == "verified" and verified["apply_rc"] == 0
    assert verified["verification"]["tests_executed"] == 2
    assert len(verified["verification"]["trace_sha256"]) == 64
