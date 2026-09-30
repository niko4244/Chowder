"""Queue/cache/resume tests for the SWE-smith replay harness (no podman, no containers).

Covers the three properties a host under load must not be able to break:
  1. an answered row is never re-run (and a retryable host failure is retried
     only while budget remains);
  2. a row is executed once per process even when two batches share a work root
     (O_EXCL claim, stale claims reclaimed);
  3. setup state is content-addressed, so the same instance+setup reuses one
     red image instead of redoing setup, and the prescreen keeps mispaired or
     non-fitting rows out of the container path entirely.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "teacher_free_distill"


def load():
    spec = importlib.util.spec_from_file_location("replay_smith", EXP / "replay_smith.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["replay_smith"] = mod
    spec.loader.exec_module(mod)
    return mod


def write_row_state(work: Path, traj_id: str, result: dict | None = None, **state):
    row_dir = work / traj_id
    row_dir.mkdir(parents=True, exist_ok=True)
    if result is not None:
        (row_dir / "row_result.json").write_text(json.dumps(result), encoding="utf-8")
    if state:
        (row_dir / "row_state.json").write_text(json.dumps(state), encoding="utf-8")
    return row_dir


def test_plan_skips_answered_rows_and_runs_new_ones(tmp_path):
    mod = load()
    write_row_state(tmp_path, "done_row", {"task_id": "done_row", "status": "verified"})
    write_row_state(tmp_path, "red_row", {"task_id": "red_row", "status": "not_green"})
    rows = [{"traj_id": "done_row"}, {"traj_id": "red_row"}, {"traj_id": "fresh_row"}]
    plan = {entry["traj_id"]: entry["decision"] for entry in mod.plan_rows(rows, tmp_path)}
    assert plan == {"done_row": "done", "red_row": "done", "fresh_row": "run"}


def test_retryable_setup_failure_retries_then_exhausts(tmp_path):
    mod = load()
    result = {"task_id": "loaded", "status": "setup_failed",
              "reason": "podman pull failed for docker.io/...", "retryable": True}
    write_row_state(tmp_path, "loaded", result, attempts=1)
    rows = [{"traj_id": "loaded"}]
    assert mod.plan_rows(rows, tmp_path)[0]["decision"] == "run"
    write_row_state(tmp_path, "loaded", result, attempts=2)
    assert mod.plan_rows(rows, tmp_path)[0]["decision"] == "retry_exhausted"


def test_non_retryable_setup_failure_is_final(tmp_path):
    mod = load()
    write_row_state(tmp_path, "content_break",
                    {"task_id": "content_break", "status": "setup_failed",
                     "reason": "overlay rc=1: checkout failed", "retryable": False},
                    attempts=1)
    assert mod.plan_rows([{"traj_id": "content_break"}], tmp_path)[0]["decision"] == "done"


def test_abandoned_row_is_resumed_not_skipped(tmp_path):
    mod = load()
    row_dir = write_row_state(tmp_path, "killed_midway", None, stage="setup_overlay",
                              attempts=1)
    # A stale claim means the previous owner died; the row must resume.
    (row_dir / "claim.json").write_text(json.dumps(
        {"pid": 4242, "claimed_epoch": 0, "claimed_at": "2026-09-26T00:00:00Z"}),
        encoding="utf-8")
    assert mod.plan_rows([{"traj_id": "killed_midway"}], tmp_path)[0]["decision"] == "run"
    # A live claim means another writer owns it right now.
    (row_dir / "claim.json").write_text(json.dumps(
        {"pid": 4242, "claimed_epoch": mod.time.time(),
         "claimed_at": mod.utc_now()}), encoding="utf-8")
    assert mod.plan_rows([{"traj_id": "killed_midway"}], tmp_path)[0]["decision"] == "in_flight"


def test_claim_is_exclusive_and_stale_claims_are_reclaimed(tmp_path):
    mod = load()
    row_dir = tmp_path / "row"
    first = mod.claim_row(row_dir)
    assert first["claimed"] is True and first["reclaimed_stale"] is False
    second = mod.claim_row(row_dir)
    assert second["claimed"] is False and second["held_by"]["pid"] == mod.os.getpid()
    (row_dir / "claim.json").write_text(json.dumps({"pid": 1, "claimed_epoch": 0}),
                                        encoding="utf-8")
    third = mod.claim_row(row_dir)
    assert third["claimed"] is True and third["reclaimed_stale"] is True


def test_finalize_row_writes_result_and_frees_the_claim(tmp_path):
    mod = load()
    row_dir = tmp_path / "row"
    mod.claim_row(row_dir)
    mod.finalize_row(row_dir, {"task_id": "row", "status": "not_green"})
    assert not (row_dir / "claim.json").exists()
    assert json.loads((row_dir / "row_result.json").read_text())["status"] == "not_green"
    assert mod.plan_rows([{"traj_id": "row"}], tmp_path)[0]["decision"] == "done"


def test_host_load_markers_decide_retryability():
    mod = load()
    assert mod.setup_failure_is_retryable("podman pull failed for docker.io/x: main: timeout")
    assert mod.setup_failure_is_retryable("overlay mount failed: resource temporarily unavailable")
    assert not mod.setup_failure_is_retryable("overlay rc=1 commit rc=0: checkout failed")


def test_red_image_tag_is_deterministic_and_podman_safe():
    mod = load()
    key = mod.digest({"image": "x", "branches": ["a"]})
    tag = mod.red_image_tag("conan-io__conan.86f29e13.pr_11666", key)
    assert tag == mod.red_image_tag("conan-io__conan.86f29e13.pr_11666", key)
    assert tag.startswith("localhost/chowder-replay-red-")
    # Exactly one colon (repository:tag) and no characters podman rejects.
    assert tag.count(":") == 1
    assert tag == tag.lower()
    assert not set(" \t/" ) & set(tag.rsplit(":", 1)[0].split("/", 1)[1])
    other = mod.red_image_tag("conan-io__conan.86f29e13.pr_11666", mod.digest({"image": "y"}))
    assert other != tag
    assert mod.sanitize_slug("pndurette__gTTS.dbcda4f3.pr_440") == "pndurette__gtts.dbcda4f3.pr_440"


def test_prescreen_routes_mispaired_rows_and_keeps_congruent_ones():
    mod = load()
    gtts_patch = ("diff --git a/gtts/tts.py b/gtts/tts.py\n--- a/gtts/tts.py\n+++ b/gtts/tts.py\n"
                  "@@ -1 +1 @@\n-a\n+b\n")
    f2p = ["gtts/tests/test_tts.py::TestTTS::test_speed"]
    record = {"traj_id": "r1", "instance_id": "pndurette__gTTS.dbcda4f3.pr_440",
              "patch": gtts_patch}
    assert mod.prescreen_row(record, {"FAIL_TO_PASS": f2p}, {}, require_congruence=True) is None
    moto_patch = ("diff --git a/moto/networkmanager/models.py b/moto/networkmanager/models.py\n"
                  "--- a/moto/networkmanager/models.py\n+++ b/moto/networkmanager/models.py\n"
                  "@@ -1 +1 @@\n-a\n+b\n")
    moto_f2p = ["tests/test_resiliencehub/test_resiliencehub.py::Test::test_x"]
    gated = mod.prescreen_row({**record, "patch": moto_patch},
                              {"FAIL_TO_PASS": moto_f2p}, {}, require_congruence=True)
    assert gated["status"] == "mispaired"
    assert gated["congruence"]["verdict"] == "same_repo_other_module"
    # Without the gate, congruence is not consulted: the row runs.
    assert mod.prescreen_row({**record, "patch": moto_patch},
                             {"FAIL_TO_PASS": moto_f2p}, {}, require_congruence=False) is None


def test_mispaired_row_can_be_recovered_with_derived_tests():
    mod = load()
    patch = ("diff --git a/moto/networkmanager/models.py b/moto/networkmanager/models.py\n"
             "--- a/moto/networkmanager/models.py\n+++ b/moto/networkmanager/models.py\n"
             "@@ -1 +1 @@\n-a\n+b\n")
    record = {"traj_id": "m1", "instance_id": "getmoto__moto.694ce1f4.pr_7456", "patch": patch}
    instance = {"FAIL_TO_PASS": ["tests/test_resiliencehub/test_resiliencehub.py::T::test_x"]}
    # Default: blocked as mispaired; with recovery requested: a recovery request,
    # never a terminal mispaired verdict.
    blocked = mod.prescreen_row(record, instance, {}, require_congruence=True)
    assert blocked["status"] == "mispaired"
    recover = mod.prescreen_row(record, instance, {}, require_congruence=True,
                               recover_derived_tests=True)
    assert recover["recovery"] == "derived_tests"
    assert recover["congruence"]["verdict"] == "same_repo_other_module"
    # A non-fitting patch is never recovered: the ledger gate comes first.
    ledger = {record["instance_id"]: {"verdict": "context_does_not_match_branch"}}
    still = mod.prescreen_row(record, instance, ledger, require_congruence=True,
                              recover_derived_tests=True)
    assert still["status"] == "patch_did_not_fit"


def test_fit_ledger_blocks_a_patch_that_cannot_apply():
    mod = load()
    ledger = {"cantools__cantools.0c6a7871.combine_module__94v6dlji":
              {"instance_id": "cantools__cantools.0c6a7871.combine_module__94v6dlji",
               "verdict": "context_does_not_match_branch",
               "first_error": "error: patch failed: src/cantools/database/can/message.py:280"}}
    row = {"traj_id": "c1", "instance_id": "cantools__cantools.0c6a7871.combine_module__94v6dlji",
           "patch": "diff --git a/x b/x\n"}
    gated = mod.prescreen_row(row, None, ledger, require_congruence=False)
    assert gated["status"] == "patch_did_not_fit"
    assert "context_does_not_match_branch" in gated["reason"]
    fitting = {**ledger, row["instance_id"]: {"verdict": "fits_own_branch"}}
    assert mod.prescreen_row(row, None, fitting, require_congruence=False) is None


def test_summary_counts_each_outcome_and_merges_duplicates(tmp_path):
    mod = load()
    out = tmp_path / "replayed.jsonl"
    rows = [{"task_id": "a", "status": "verified"},
            {"task_id": "b", "status": "not_green"},
            {"task_id": "c", "status": "patch_did_not_fit"},
            {"task_id": "c", "status": "verified"}]  # resumed row rewrote its verdict
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    merged = mod.merge_results(out)
    assert {r["task_id"]: r["status"] for r in merged} == {"a": "verified", "b": "not_green",
                                                           "c": "verified"}
    summary = mod.summarize_results(merged)
    assert summary == {"rows": 3, "verified": 2, "not_green": 1}
    assert mod.write_summary(out)["rows"] == 3


def test_classify_replay_status_orders_evidence_correctly():
    """The status decision is shared by both env paths and its order is the
    evidence semantics: apply defects are never failed repairs, verified
    needs red-then-green, and a zero-test post phase is evidence-of-nothing,
    not a failed repair (an unrelated collection error aborting the suite
    must not read as not_green)."""
    mod = load()
    classify = mod.classify_replay_status
    assert classify(patch_applied=True, failure_reproduced=True,
                    post_rc=0, post_tests_executed=12) == "verified"
    assert classify(patch_applied=False, failure_reproduced=True,
                    post_rc=1, post_tests_executed=12) == "patch_did_not_fit"
    # A genuine not_green: tests ran, stayed red (or the F2P passed pre-patch).
    assert classify(patch_applied=True, failure_reproduced=False,
                    post_rc=0, post_tests_executed=730) == "not_green"
    assert classify(patch_applied=True, failure_reproduced=True,
                    post_rc=2, post_tests_executed=5) == "not_green"
    # Zero tests executed post-patch: evidence-of-nothing, whatever the rc.
    assert classify(patch_applied=True, failure_reproduced=False,
                    post_rc=2, post_tests_executed=0) == "recovery_evidence_insufficient"
    assert classify(patch_applied=True, failure_reproduced=True,
                    post_rc=0, post_tests_executed=0) == "recovery_evidence_insufficient"
    # Apply failure wins over the zero-test case (pairing defect first).
    assert classify(patch_applied=False, failure_reproduced=False,
                    post_rc=2, post_tests_executed=0) == "patch_did_not_fit"


def test_recovery_evidence_insufficient_is_terminal_and_summarized(tmp_path):
    mod = load()
    assert "recovery_evidence_insufficient" in mod.FINAL_STATUSES
    out = tmp_path / "replayed.jsonl"
    rows = [{"task_id": "x", "status": "recovery_evidence_insufficient"}]
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    assert mod.summarize_results(mod.merge_results(out))[
        "recovery_evidence_insufficient"] == 1


def test_stem_name_clauses_cover_pypi_and_mypy_conventions():
    mod = load()
    clauses = mod.stem_name_clauses(["fscache", "routines"])
    assert "-name 'test_fscache.py'" in clauses
    assert "-name 'fscache_test.py'" in clauses
    # mypy's testx.py convention, previously missing -> full-suite fallback.
    assert "-name 'testfscache.py'" in clauses


def test_setup_commands_are_idempotent_across_resumes():
    """A mid-clone kill must not turn the resumed row into setup_failed.

    Setup re-runs on resume; without the guard the fresh clone fails on the
    partial /work/repo left by the killed attempt (observed on dask pr_8860
    after a host restart), a non-retryable verdict for a host-side artifact.
    """
    mod = load()
    for cmd in (mod.mirror_setup_command("swesmith/dask__dask.5f61e423",
                                         ["dask__dask.5f61e423.pr_8860"]),
                mod.upstream_setup_command("dask/dask", "abc123")):
        assert cmd.startswith("rm -rf /work/repo /work/build /work/venv && ")
        assert "git clone -q" in cmd
    # The mirror variant keeps the multi-candidate checkout fallback chain.
    mirror = mod.mirror_setup_command("swesmith/o__r.a.b.c", ["o__r.a.b", "o__r.a"])
    assert "git checkout -q o__r.a.b || git checkout -q o__r.a" in mirror
