"""Focused tests for Phase 2/3: quality gates, provenance records, replay evidence."""
import importlib.util
import json
import subprocess
import sys
import types
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


def stub_datasets(monkeypatch, rows):
    """Install a stand-in ``datasets`` module so the *optional* dependency is
    never required to test the fetch path.

    CI installs chowder without the optional extras, so the real module is
    absent there; a fake module exercises ``fetch_smith`` through the same
    call shape without weakening what the test asserts (the dataset's own
    ``resolved`` flag stays a labeled claim, never evidence).
    """
    module = types.ModuleType("datasets")
    module.load_dataset = lambda *a, **k: iter(rows)
    monkeypatch.setitem(sys.modules, "datasets", module)
    return module


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

    stub_datasets(monkeypatch, rows)
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


def test_fetch_and_stream_modules_import_without_the_optional_datasets():
    """``datasets`` is an optional extra: importing these modules must never
    require it, or a CPU test matrix without extras fails at import time.

    Regression: the CI failure this replaces -- the fetch test itself did
    ``monkeypatch.setattr("datasets.load_dataset", ...)``, which raises
    ModuleNotFoundError wherever the optional dependency is not installed.
    """
    blocker = (
        "import sys\n"
        "class _Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'datasets' or name.startswith('datasets.'):\n"
        "            raise ImportError('datasets is not installed in this environment')\n"
        "        return None\n"
        "sys.meta_path.insert(0, _Block())\n"
        "import importlib.util, pathlib\n"
        "for mod in sys.argv[1:]:\n"
        "    spec = importlib.util.spec_from_file_location(pathlib.Path(mod).stem, mod)\n"
        "    module = importlib.util.module_from_spec(spec)\n"
        "    spec.loader.exec_module(module)\n"
        "print('imported')\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", blocker, str(EXP / "fetch_smith.py"),
         str(EXP / "stream_hf.py")],
        capture_output=True, text=True, check=True,
    )
    assert out.stdout.strip() == "imported"
    # The import must also stay lazy on a machine where datasets IS installed:
    # a module-level ``from datasets import load_dataset`` would bind the symbol
    # here even though it sits inside a try/except.
    assert "load_dataset" not in vars(fetch)
    assert "load_dataset" not in vars(load("stream_hf"))


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
              "pass_to_pass": {"total": 2, "passing": 2, "failing": 0,
                               "source": "instance_PASS_TO_PASS"},
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


# --------------------------------------------------------------------------
# Phase 2: frozen paired evaluation protocol
# --------------------------------------------------------------------------

eval_protocol = load("eval_protocol")


def write_eval_rows(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def eval_pair(tmp_path, *, dev_prompt="dev problem one"):
    dev = write_eval_rows(tmp_path / "dev.jsonl", [
        {"problem_id": "dev-1", "prompt": dev_prompt, "expected": "42"}])
    final = write_eval_rows(tmp_path / "final.jsonl", [
        {"problem_id": "fin-1", "prompt": "final problem one", "expected": "7"}])
    return dev, final


def test_paired_plan_freezes_one_protocol_for_both_arms(tmp_path):
    dev, final = eval_pair(tmp_path)
    plan = eval_protocol.paired_plan(tmp_path / "plan", dev, final)
    digests = {plan["protocol_digest"]}
    assert plan["arms"] == [
        {"arm": "base", "adapter_dir": None,
         "model": eval_protocol.FROZEN_MODEL["base_model"],
         "revision": eval_protocol.FROZEN_MODEL["base_revision"]},
    ]
    assert plan["splits"]["final"]["separation_check"]["ok"] is True
    assert plan["protocol"]["decoding"] == "greedy"
    assert digests == {plan["protocol_digest"]}


def test_paired_plan_refuses_dev_final_overlap(tmp_path):
    dev, final = eval_pair(tmp_path)
    # Same problem id in both files: the final split would not be independent.
    leaked = write_eval_rows(tmp_path / "leaked.jsonl", [
        {"problem_id": "dev-1", "prompt": "a different question entirely",
         "expected": "1"}])
    with pytest.raises(ValueError, match="overlap"):
        eval_protocol.paired_plan(tmp_path / "plan", dev, leaked)
    # Same prompt text under a different id is the same leak.
    leaked_text = write_eval_rows(tmp_path / "leaked_text.jsonl", [
        {"problem_id": "fin-9", "prompt": "DEV   problem ONE", "expected": "1"}])
    with pytest.raises(ValueError, match="overlap"):
        eval_protocol.paired_plan(tmp_path / "plan2", dev, leaked_text)


def test_adapter_arm_requires_the_pinned_condition_a_digest(tmp_path):
    dev, final = eval_pair(tmp_path)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    with pytest.raises((FileNotFoundError, ValueError)):
        eval_protocol.paired_plan(tmp_path / "p1", dev, final,
                                  adapter_dir=adapter)
    real = eval_protocol.verified_adapter_record(EXP)
    pinned = {f["path"]: f for f in real["files"]
              }["adapter/adapter_model.safetensors"]["sha256"]
    (adapter / "adapter_model.safetensors").write_bytes(b"not the adapter")
    with pytest.raises(ValueError, match="pinned Condition A digest"):
        eval_protocol.paired_plan(tmp_path / "p2", dev, final,
                                  adapter_dir=adapter)
    # The real artifacts root must still verify (guards against a stale pin).
    record = eval_protocol.verified_adapter_record(EXP)
    on_disk = {f["path"]: f for f in record["files"]
               }["adapter/adapter_model.safetensors"]["sha256"]
    assert on_disk == pinned
    assert record["run"]["base_revision"] == \
        eval_protocol.FROZEN_MODEL["base_revision"]


def test_run_plan_writes_specs_but_never_claims_execution(tmp_path):
    dev, final = eval_pair(tmp_path)
    plan_path = tmp_path / "plan"
    eval_protocol.paired_plan(plan_path, dev, final)
    summary = eval_protocol.run_plan(plan_path / "eval_plan.json",
                                     tmp_path / "results")
    assert summary["status"] == "awaiting_operator_authorization"
    spec = json.loads((tmp_path / "results" / "base__final" /
                       "eval_spec.json").read_text())
    assert spec["adapter_dir"] is None
    assert spec["suites"][0]["scoring"] == "final_number_match"
    assert spec["offline"] is True
    assert spec["seed"] == 2026


def test_paired_deltas_and_the_always_review_decision():
    rows = [{"problem_id": f"p{i}", "score_a": 0.0, "score_b": 1.0}
            for i in range(6)] + [
            {"problem_id": "p6", "score_a": 1.0, "score_b": 0.0}]
    analysis = eval_protocol.paired_deltas(rows, bootstraps=400, seed=7)
    assert analysis["problems"] == 7
    assert analysis["b_wins"] == 6 and analysis["a_wins"] == 1
    assert analysis["mean_delta_b_minus_a"] == round(5 / 7, 6)
    assert analysis["ci95_low"] <= analysis["mean_delta_b_minus_a"] <= analysis["ci95_high"]
    assert analysis["decision"] == "requires_operator_review"
    with pytest.raises(ValueError):
        eval_protocol.paired_deltas([{"problem_id": "x", "score_a": 1, "score_b": 1},
                                     {"problem_id": "x", "score_a": 0, "score_b": 1}])


def test_compare_arms_pairs_raw_predictions_per_problem(tmp_path):
    for arm, score_for in (("base", {"a": 0.0, "b": 0.0, "c": 1.0}),
                           ("condition_a", {"a": 1.0, "b": 0.0, "c": 1.0})):
        arm_dir = tmp_path / f"{arm}__final"
        arm_dir.mkdir()
        with (arm_dir / "predictions-holdout_final.jsonl").open("w",
                                                                encoding="utf-8") as out:
            for pid, value in score_for.items():
                out.write(json.dumps({"problem_id": pid, "prompt": f"problem {pid}",
                                      "score": value}) + "\n")
    analysis = eval_protocol.compare_arms(tmp_path, split="final")
    assert analysis["arms"] == {"a": "base", "b": "condition_a"}
    assert analysis["problems_paired"] == 3 and analysis["status"] == "complete"
    assert analysis["b_wins"] == 1 and analysis["ties"] == 2
    assert analysis["decision"] == "requires_operator_review"
    assert (tmp_path / "paired_comparison_final.json").is_file()


def test_pass_to_pass_outcomes_are_measured_and_recorded(tmp_path):
    """Phase 4: the post-patch output decides the pass-to-pass verdict.

    Observed failures block; unexecuted ids are unknown rather than passing;
    the measured surface travels on the replay record either way.
    """
    passing, failing = replay._p2p_outcomes(
        "PASSED tests/test_a.py::test_one\n"
        "FAILED tests/test_a.py::test_two - boom\n",
        ["tests/test_a.py::test_one", "tests/test_a.py::test_two"])
    assert (passing, failing) == (1, 1)
    assert replay._p2p_outcomes("nothing ran", ["tests/test_a.py::test_one"]) == (0, 0)
    assert replay._p2p_outcomes("no surface", []) == (0, 0)

    events = [{"kind": "test", "returncode": 0, "tests_executed": 2}]
    verified = replay._finish(tmp_path / "v", "t1", events, [], "verified",
                              returncode=0, apply_rc=0, tests_executed=2,
                              failure_reproduced=True,
                              pass_to_pass_total=3, pass_to_pass_passing=3,
                              pass_to_pass_failing=0,
                              pass_to_pass_source="instance_PASS_TO_PASS")
    assert verified["pass_to_pass"] == {"total": 3, "passing": 3, "failing": 0,
                                        "source": "instance_PASS_TO_PASS"}


# --------------------------------------------------------------------------
# Final-eval set builder
# --------------------------------------------------------------------------

final_eval = load("final_eval_set")


def test_gold_extraction_rejects_non_numbers():
    """Regression: a lone comma matched the number regex and became ''."""
    assert final_eval.extract_gold("so the total is 1,024 units.") == "1024"
    assert final_eval.extract_gold("hence, 42") == "42"
    assert final_eval.extract_gold("value: ,") is None
    assert final_eval.extract_gold("```print(x)```") is None
    assert final_eval.extract_gold("#### 3.50") == "3.50"
    assert final_eval.extract_gold("no digits here at all") is None


def test_dev_prompt_texts_cover_dev_and_train(tmp_path):
    def msg_row(q):
        return json.dumps({"messages": [{"role": "user", "content": q},
                                        {"role": "assistant", "content": "a"}]})
    (tmp_path / "dev.jsonl").write_text(msg_row(" Dev Question ") + "\n", encoding="utf-8")
    (tmp_path / "train.jsonl").write_text(msg_row("train question") + "\n", encoding="utf-8")
    texts = final_eval.dev_prompt_texts(tmp_path)
    assert "dev question" in texts and "train question" in texts


def test_gsm8k_gold_extraction_is_marker_derived():
    """External golds come from the dataset's #### marker, normalized;
    a lone comma is not a number."""
    gsm = load("gsm8k_eval_set")
    assert gsm.gold_from_answer("some reasoning\n#### 1,024") == "1024"
    assert gsm.gold_from_answer("#### 72") == "72"
    assert gsm.gold_from_answer("#### -3.5") == "-3.5"
    assert gsm.gold_from_answer("#### ,") is None
    assert gsm.gold_from_answer("no marker at all") is None
