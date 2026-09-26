"""Stage state must be *checked*, not inferred from a file existing.

Each test below creates an artifact that looks finished and asserts what the
evidence layer does with it. The recurring regression this guards: the pilot
screen used to mark "training" complete because a recipe file existed, and
"evaluation" complete because a replay summary existed — including one whose
own numbers said nothing had verified.
"""
import hashlib
import json
from pathlib import Path

import pytest

from chowder import teacher_free_evidence as ev

EXP = Path(__file__).resolve().parents[1] / "experiments" / "teacher_free_distill"


def write(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def states(directory: Path) -> dict:
    return {entry["stage"]: entry for entry in ev.stage_state(directory)}


def good_catalog(directory: Path) -> Path:
    return write(directory / "sources.json", {"sources": {
        "open_thoughts3": {"approved": True, "license": "Apache-2.0",
                           "revision": "61bcf9d", "review_reference": "HF API 2026-09-25"},
        "mixture_of_thoughts": {"approved": False, "license": None, "revision": "e55fa28",
                                "review_reference": "HF API 2026-09-25",
                                "review_note": "no dataset license exists"}}})


def good_preflight(directory: Path) -> Path:
    checks = {name: True for name in
              ("adapter_created", "finite_loss", "masking_assistant_only")}
    return write(directory / "preflight_result.json",
                 {"checks": checks, "ok": True, "passed": len(checks), "total": len(checks)})


def good_student(directory: Path) -> Path:
    return write(directory / "student_selection.json", {
        "student": {"repo": "Qwen/Qwen3-1.7B", "revision": "70d244cc", "license": "apache-2.0"},
        "hardware": {"accelerators": [{"name": "NVIDIA GeForce RTX 5060 Ti"}]},
        "workload_gb": {"frozen_weights_gb": 3.44, "activation_gb": 11.274},
        "fits_primary_pool": True, "plan_bottleneck": "compute_or_kernel"})


def good_recipes(directory: Path) -> Path:
    return write(directory / "recipes" / "a_sft_supervised.json", {
        "condition": "A_supervised_distillation",
        "authorization": {"gpu_required": True, "operator_approval_required": True},
        "outputs": {"adapter": "checkpoints/cond_a/adapter", "loss_history_required": True}})


def good_integrity(**overrides) -> dict:
    integrity = {
        "ok": True,
        "unique_problems": 4211,
        "split_integrity": {"ok": True, "problem_groups_crossing_splits": 0},
        "leakage": {"ok": True, "holdout_collisions": 0},
        "truncation": {"ok": True, "incomplete_targets": 0},
        "token_audit": {"ok": True, "max_length": 2048, "tokenizer": "Qwen/Qwen3-1.7B",
                        "over_budget": 0},
    }
    integrity.update(overrides)
    return integrity


def good_dataset(directory: Path, *, rows: int = 40, integrity: dict | None = None,
                 audit: bool = True) -> Path:
    """A dataset directory whose manifest does (or does not) carry an audit."""
    train = directory / "dataset" / "train.jsonl"
    train.parent.mkdir(parents=True, exist_ok=True)
    train.write_text("".join(json.dumps({"messages": [
        {"role": "user", "content": f"question {i}"},
        {"role": "assistant", "content": f"answer {i}"}]}) + "\n"
        for i in range(rows)), encoding="utf-8")
    dev = train.parent / "dev.jsonl"
    dev.write_text(train.read_text(encoding="utf-8")[:200], encoding="utf-8")
    manifest = {
        "format": f"{ev.MANIFEST_FORMAT_PREFIX}v1",
        "train_rows": rows, "dev_rows": 2, "holdout_is_external": True,
        "output_sha256": {"train": sha256(train), "dev": sha256(dev)},
    }
    if audit:
        manifest["integrity"] = good_integrity() if integrity is None else integrity
    write(train.parent / "manifest.json", manifest)
    return train


def good_condition_a(directory: Path, dataset_sha: str) -> dict:
    """Record a completed run whose artifacts live on disk, then return it."""
    root = directory / "artifacts"
    adapter = root / "adapter" / "adapter_model.safetensors"
    adapter.parent.mkdir(parents=True, exist_ok=True)
    adapter.write_bytes(b"lo-ra weights" * 16)
    record = {
        "format": ev.CONDITION_A_FORMAT,
        "artifacts_root": str(root),
        "run": {"steps_completed": 284, "steps_total": 284, "mean_train_loss": 1.7276,
                "dataset_sha256": dataset_sha},
        "files": [{"path": "adapter/adapter_model.safetensors", "sha256": sha256(adapter),
                   "bytes": adapter.stat().st_size}],
        "loss_history": {"entries": 28, "first_step": 10, "last_step": 280},
    }
    write(directory / "condition_a_artifacts.json", record)
    return record


def good_replay(directory: Path, verified: int) -> Path:
    return write(directory / "replay" / "replay_summary.json",
                 {"rows": 4, "verified": verified, "not_green": 2,
                  "setup_failed": 1, "skipped": 1})


def good_comparison(directory: Path, name: str = "comparison_a.json", **overrides) -> Path:
    payload = {
        "deltas": {"green_completion_rate": 0.12},
        "regressions": [],
        "decision": "requires_operator_review",
        "leakage": {"repair_split": {"ok": True}, "prompt_overlap": {"ok": True}},
    }
    payload.update(overrides)
    return write(directory / name, payload)


def complete_pilot(directory: Path) -> Path:
    good_catalog(directory)
    dataset = good_dataset(directory)
    good_student(directory)
    good_preflight(directory)
    good_recipes(directory)
    good_condition_a(directory, sha256(dataset))
    good_replay(directory, verified=1)
    good_comparison(directory)
    (directory / "REPORT.md").write_text("# pilot\n", encoding="utf-8")
    return directory


# --------------------------------------------------------------------------
# Structural rules
# --------------------------------------------------------------------------

def test_workflow_stages_are_stable_and_ordered(tmp_path):
    assert ev.WORKFLOW_STAGES == [
        "source_review", "data_collection", "validation", "student_selection",
        "training_preflight", "training", "evaluation", "results"]
    assert [s["stage"] for s in ev.stage_state(tmp_path)] == ev.WORKFLOW_STAGES


def test_empty_directory_is_pending_and_never_green(tmp_path):
    found = states(tmp_path)
    assert found["source_review"]["state"] == "missing"
    assert found["data_collection"]["state"] == "pending"
    assert found["validation"]["state"] == "pending"
    assert found["student_selection"]["state"] == "pending"
    assert found["training_preflight"]["state"] == "pending"
    assert found["training"]["state"] == "pending"
    assert found["evaluation"]["state"] == "pending"
    assert found["results"]["state"] == "pending"
    assert not {s["state"] for s in found.values()} & ev.GREEN_STATES


def test_every_completed_stage_is_green_in_a_complete_pilot(tmp_path):
    complete_pilot(tmp_path)
    found = states(tmp_path)
    assert {stage: entry["state"] for stage, entry in found.items()} == {
        "source_review": "reviewed", "data_collection": "verified",
        "validation": "passed", "student_selection": "selected",
        "training_preflight": "passed", "training": "completed_verified",
        "evaluation": "compared", "results": "done"}


# --------------------------------------------------------------------------
# source_review
# --------------------------------------------------------------------------

def test_catalog_requires_a_pinned_revision_and_review_reference(tmp_path):
    write(tmp_path / "sources.json", {"sources": {
        "a": {"approved": True, "license": "MIT", "revision": "", "review_reference": "x"}}})
    state = ev.catalog_state(tmp_path)
    assert state["state"] == "incomplete" and "revision is not pinned" in state["detail"]

    write(tmp_path / "sources.json", {"sources": {
        "a": {"approved": True, "license": "MIT", "revision": "rev", "review_reference": ""}}})
    assert ev.catalog_state(tmp_path)["state"] == "incomplete"


def test_approved_source_needs_a_license_and_blocked_source_needs_a_note(tmp_path):
    write(tmp_path / "sources.json", {"sources": {
        "a": {"approved": True, "license": "", "revision": "rev", "review_reference": "x"}}})
    state = ev.catalog_state(tmp_path)
    assert state["state"] == "incomplete"
    assert "approved without a license determination" in state["detail"]

    write(tmp_path / "sources.json", {"sources": {
        "a": {"approved": False, "license": None, "revision": "rev",
              "review_reference": "x", "review_note": ""}}})
    assert "blocked without a recorded review note" in ev.catalog_state(tmp_path)["detail"]


def test_catalog_without_an_explicit_decision_is_incomplete(tmp_path):
    write(tmp_path / "sources.json", {"sources": {
        "a": {"license": "MIT", "revision": "rev", "review_reference": "x"}}})
    assert ev.catalog_state(tmp_path)["state"] == "incomplete"


# --------------------------------------------------------------------------
# data_collection / validation
# --------------------------------------------------------------------------

def test_manifest_without_output_digests_is_recorded_but_not_verified(tmp_path):
    """The legacy pilot manifest is honest evidence, and not a verified dataset."""
    write(tmp_path / "dataset" / "manifest.json", {
        "format": f"{ev.MANIFEST_FORMAT_PREFIX}v1", "train_rows": 5, "dev_rows": 1})
    state = ev.manifest_state(tmp_path)
    assert state["state"] == "recorded_unpinned"
    assert "pins no output digests" in state["detail"]


def test_manifest_digest_mismatch_is_a_failure(tmp_path):
    train = good_dataset(tmp_path)
    manifest_path = train.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["output_sha256"]["train"] = "0" * 64
    write(manifest_path, manifest)
    state = ev.manifest_state(tmp_path)
    assert state["state"] == "failed"
    assert "digests differ" in state["detail"]
    assert state["verification"]["mismatches"][0]["path"] == "train.jsonl"


def test_manifest_outputs_absent_from_this_host_are_not_green(tmp_path):
    train = good_dataset(tmp_path)
    train.unlink()
    state = ev.manifest_state(tmp_path)
    assert state["state"] == "recorded_offline"
    assert "not on this host" in state["detail"]


def test_manifest_requires_a_pinned_external_holdout(tmp_path):
    train = good_dataset(tmp_path)
    manifest_path = train.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["holdout_is_external"] = False
    write(manifest_path, manifest)
    state = ev.manifest_state(tmp_path)
    assert state["state"] == "failed" and "separate" in state["detail"]


def test_zero_accepted_training_rows_is_a_failure(tmp_path):
    train = good_dataset(tmp_path)
    manifest_path = train.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["train_rows"] = 0
    write(manifest_path, manifest)
    assert ev.manifest_state(tmp_path)["state"] == "failed"


def test_validation_requires_the_integrity_audit(tmp_path):
    good_dataset(tmp_path, audit=False)
    state = ev.integrity_state(tmp_path)
    assert state["state"] == "not_audited"
    assert "predates the integrity audit" in state["detail"]


@pytest.mark.parametrize("broken,needle", [
    ({"split_integrity": {"ok": False}}, "cross partitions"),
    ({"leakage": {"ok": False, "holdout_collisions": 3}}, "holdout overlap"),
    ({"truncation": {"ok": False, "incomplete_targets": 2}}, "truncated"),
    ({"token_audit": {"ok": True, "max_length": 512}}, "max_length=512"),
    ({"ok": False}, "summary is not ok"),
])
def test_integrity_audit_failures_are_named(tmp_path, broken, needle):
    good_dataset(tmp_path, integrity=good_integrity(**broken))
    state = ev.integrity_state(tmp_path)
    assert state["state"] == "failed"
    assert any(needle in problem for problem in state["failures"])


def test_clean_integrity_audit_passes_and_reports_counts(tmp_path):
    good_dataset(tmp_path)
    state = ev.integrity_state(tmp_path)
    assert state["state"] == "passed"
    assert "4211 unique problems" in state["detail"]
    assert "max_length 2048" in state["detail"]


# --------------------------------------------------------------------------
# student_selection / training_preflight
# --------------------------------------------------------------------------

def test_student_selection_needs_license_and_a_fitting_plan(tmp_path):
    good_student(tmp_path)
    assert ev.student_selection_state(tmp_path)["state"] == "selected"

    record = json.loads((tmp_path / "student_selection.json").read_text())
    record["student"]["license"] = ""
    write(tmp_path / "student_selection.json", record)
    assert ev.student_selection_state(tmp_path)["state"] == "incomplete"

    record["student"]["license"] = "apache-2.0"
    record["fits_primary_pool"] = False
    write(tmp_path / "student_selection.json", record)
    assert "does not fit" in ev.student_selection_state(tmp_path)["detail"]


def test_preflight_ok_flag_alone_is_not_enough(tmp_path):
    """A record can claim ok while its own check list disagrees."""
    write(tmp_path / "preflight_result.json",
          {"ok": True, "passed": 3, "total": 3,
           "checks": {"a": True, "b": False, "c": True}})
    state = ev.preflight_state(tmp_path)
    assert state["state"] == "failed" and "failing: b" in state["detail"]

    write(tmp_path / "preflight_result.json",
          {"ok": True, "passed": 3, "total": 3, "checks": {"a": True, "b": True}})
    state = ev.preflight_state(tmp_path)
    assert state["state"] == "failed" and "do not agree" in state["detail"]

    write(tmp_path / "preflight_result.json",
          {"ok": False, "passed": 2, "total": 2, "checks": {"a": True, "b": True}})
    assert ev.preflight_state(tmp_path)["state"] == "failed"


# --------------------------------------------------------------------------
# training: recipes are authorization, a run record is evidence
# --------------------------------------------------------------------------

def test_recipes_alone_do_not_complete_training(tmp_path):
    good_recipes(tmp_path)
    state = ev.training_state(tmp_path)
    assert state["state"] == "recipes_ready"
    assert "no completed run recorded" in state["detail"]
    assert state["state"] not in ev.GREEN_STATES


def test_recipe_without_an_approval_requirement_is_a_failure(tmp_path):
    write(tmp_path / "recipes" / "a.json",
          {"authorization": {"operator_approval_required": False}, "outputs": {}})
    assert ev.training_state(tmp_path)["state"] == "failed"


def test_condition_a_record_verifies_against_its_artifacts(tmp_path):
    dataset = good_dataset(tmp_path)
    good_condition_a(tmp_path, sha256(dataset))
    state = ev.training_state(tmp_path, manifest_train_sha256=sha256(dataset))
    assert state["state"] == "completed_verified"
    assert "284/284 steps" in state["detail"]


def test_condition_a_record_detects_a_tampered_artifact(tmp_path):
    dataset = good_dataset(tmp_path)
    record = good_condition_a(tmp_path, sha256(dataset))
    Path(record["artifacts_root"], "adapter", "adapter_model.safetensors").write_bytes(b"x")
    state = ev.training_state(tmp_path, manifest_train_sha256=sha256(dataset))
    assert state["state"] == "failed"
    assert "do not match the artifacts on disk" in state["detail"]


def test_condition_a_record_without_local_artifacts_stays_recorded(tmp_path):
    dataset = good_dataset(tmp_path)
    record = good_condition_a(tmp_path, sha256(dataset))
    # Simulate another host: the record survives, the artifacts do not.
    record["artifacts_root"] = str(tmp_path / "somewhere-else")
    write(tmp_path / "condition_a_artifacts.json", record)
    state = ev.training_state(tmp_path, manifest_train_sha256=sha256(dataset))
    assert state["state"] == "recorded_offline"
    assert state["state"] not in ev.GREEN_STATES


def test_condition_a_record_must_pin_the_manifest_dataset(tmp_path):
    dataset = good_dataset(tmp_path)
    record = good_condition_a(tmp_path, sha256(dataset))
    record["run"]["dataset_sha256"] = "b" * 64
    write(tmp_path / "condition_a_artifacts.json", record)
    state = ev.training_state(tmp_path, manifest_train_sha256=sha256(dataset))
    assert state["state"] == "failed" and "differs from the pinned manifest" in state["detail"]


def test_condition_a_record_may_not_claim_evaluation(tmp_path):
    dataset = good_dataset(tmp_path)
    record = good_condition_a(tmp_path, sha256(dataset))
    record["run"]["evaluated"] = True
    write(tmp_path / "condition_a_artifacts.json", record)
    state = ev.training_state(tmp_path, manifest_train_sha256=sha256(dataset))
    assert state["state"] == "failed" and "may not claim evaluation" in state["detail"]


def test_condition_a_record_needs_a_fully_completed_run_and_loss_history(tmp_path):
    dataset = good_dataset(tmp_path)
    record = good_condition_a(tmp_path, sha256(dataset))
    record["run"]["steps_completed"] = 260
    record["loss_history"] = {"entries": 0}
    write(tmp_path / "condition_a_artifacts.json", record)
    state = ev.training_state(tmp_path, manifest_train_sha256=sha256(dataset))
    assert state["state"] == "failed"
    assert "did not complete every step" in state["detail"]


def test_unknown_condition_a_format_is_refused(tmp_path):
    write(tmp_path / "condition_a_artifacts.json", {"format": "something-else/v9"})
    assert ev.training_state(tmp_path)["state"] == "failed"


# --------------------------------------------------------------------------
# evaluation / results
# --------------------------------------------------------------------------

def test_replay_summary_with_zero_verified_is_not_green(tmp_path):
    good_replay(tmp_path, verified=0)
    state = ev.replay_state(tmp_path)
    assert state["state"] == "no_verified_repair"
    assert state["state"] not in ev.GREEN_STATES
    assert "0 verified" in state["detail"]


def test_replay_verified_count_is_summed_across_summaries(tmp_path):
    good_replay(tmp_path, verified=1)
    write(tmp_path / "other" / "replay_summary.json",
          {"rows": 1, "verified": 2, "not_green": 0, "setup_failed": 0, "skipped": 0})
    state = ev.replay_state(tmp_path)
    assert state["state"] == "replay_verified" and state["verified"] == 3


def test_comparison_without_a_leakage_section_is_unchecked(tmp_path):
    write(tmp_path / "comparison_x.json", {"deltas": {}, "decision": "requires_operator_review"})
    state = ev.comparison_state(tmp_path)
    assert state["state"] == "unchecked" and state["state"] not in ev.GREEN_STATES


def test_comparison_with_failed_leakage_is_a_failure(tmp_path):
    good_comparison(tmp_path, leakage={"repair_split": {"ok": False},
                                       "prompt_overlap": {"ok": True}})
    state = ev.comparison_state(tmp_path)
    assert state["state"] == "failed" and "leakage check failed" in state["detail"]


def test_comparison_carrying_a_promotion_decision_is_refused(tmp_path):
    good_comparison(tmp_path, decision="promoted")
    state = ev.comparison_state(tmp_path)
    assert state["state"] == "failed" and "non-operator decision" in state["detail"]


def test_results_stay_narrative_only_without_a_checked_comparison(tmp_path):
    (tmp_path / "REPORT.md").write_text("# report\n", encoding="utf-8")
    found = states(tmp_path)
    assert found["results"]["state"] == "narrative_only"
    assert found["results"]["state"] not in ev.GREEN_STATES


def test_evaluation_falls_back_to_replay_evidence_when_no_comparison_exists(tmp_path):
    good_replay(tmp_path, verified=0)
    assert ev.comparison_state(tmp_path)["state"] == "no_verified_repair"
    good_replay(tmp_path, verified=2)
    assert ev.comparison_state(tmp_path)["state"] == "replay_verified"


# --------------------------------------------------------------------------
# recorded file lists
# --------------------------------------------------------------------------

def test_recorded_file_list_flags_unverifiable_entries(tmp_path):
    target = tmp_path / "a.bin"
    target.write_bytes(b"payload")
    check = ev.verify_recorded_files(tmp_path, [
        {"path": "a.bin", "sha256": sha256(target)},
        {"path": "a.bin", "sha256": "short"},
        "not-a-dict",
        {"path": "missing.bin", "sha256": "0" * 64},
    ])
    assert not check["ok"]
    assert check["checked"] == ["a.bin"]
    assert check["unreadable"] == ["a.bin", "not-a-dict"]
    assert check["missing"] == ["missing.bin"]


def test_recorded_file_list_reports_a_size_mismatch(tmp_path):
    target = tmp_path / "a.bin"
    target.write_bytes(b"payload")
    check = ev.verify_recorded_files(tmp_path, [
        {"path": "a.bin", "sha256": sha256(target), "bytes": 3}])
    assert not check["ok"] and check["mismatches"][0]["actual_bytes"] == 7


def test_empty_recorded_file_list_is_never_ok(tmp_path):
    assert ev.verify_recorded_files(tmp_path, [])["ok"] is False
    assert ev.verify_recorded_files(tmp_path, None)["ok"] is False


# --------------------------------------------------------------------------
# the real repository's committed artifacts
# --------------------------------------------------------------------------

def test_real_repo_artifacts_derive_coherent_state():
    """The committed pilot directory must read coherently, honestly.

    Nothing here asserts a green training/evaluation stage: no manifest and no
    checked comparison is committed, and pretending otherwise is the defect.
    """
    found = states(EXP)
    assert found["source_review"]["state"] == "reviewed"
    assert found["training_preflight"]["state"] == "passed"
    assert found["student_selection"]["state"] == "selected"
    assert found["training"]["state"] in ("recipes_ready", "completed_verified",
                                          "recorded_offline")
    assert found["evaluation"]["state"] not in ev.GREEN_STATES
