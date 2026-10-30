"""Fail-closed checkpoint tests; never start a model, official task or GPU."""

from pathlib import Path

import pytest

from chowder_batch import reap_q4_automationbench_full600_resumable_20261027 as r


def test_owner_freezes_real_600_names_and_gen0_is_never_selected():
    items = [{"id": str(i // 2), "name": f"task_{i}"}
             for i in range(600)]
    owner = r.exact_owner(items, "1" * 64, "2" * 64, "3" * 40)
    assert owner["original_task_population"] == 600
    assert owner["tasks"] == items
    assert owner["native_flags"] == r.q4.FLAGS
    assert owner["frozen_scientific_gate_eligible"] is False
    assert owner["release_approved"] is False
    assert owner["gen0_status"] == "BLOCKED"
    assert owner["chosen"] is None


def test_name_based_selection_maps_identity_to_index():
    tasks = [{"id": "1", "name": "sales.a"}, {"id": "2", "name": "sales.b"},
             {"id": "3", "name": "sales.c"}]
    assert r.select_task_indices(tasks, ["sales.c", "sales.a"]) == [2, 0]


def test_name_based_selection_defaults_to_the_whole_population():
    tasks = [{"id": "1", "name": "a"}, {"id": "2", "name": "b"}]
    assert r.select_task_indices(tasks) == [0, 1]
    assert r.select_task_indices(tasks, []) == [0, 1]


def test_names_are_deduplicated_without_reordering():
    tasks = [{"id": "1", "name": "a"}, {"id": "2", "name": "b"}]
    assert r.select_task_indices(tasks, ["b", "b", "a"]) == [1, 0]


def test_an_unknown_name_is_refused_rather_than_silently_skipped():
    tasks = [{"id": "1", "name": "a"}]
    with pytest.raises(ValueError, match="unknown task name"):
        r.select_task_indices(tasks, ["a", "sales.not_real"])


def test_reject_missing_scored_item(tmp_path):
    assert r.verify_item(tmp_path / "task-000", 0, {"id": "501",
                                                    "name": "sales.multi_hop_lookup"},
                         "a" * 64, "model") is None


def test_reject_incomplete_nonempty_item(tmp_path):
    item = tmp_path / "task-000"
    item.mkdir()
    (item / "official-evaluator.private.log").write_text("partial")
    with pytest.raises(ValueError, match="incomplete"):
        r.verify_item(item, 0, {"id": "501", "name": "sales.multi_hop_lookup"},
                      "a" * 64, "model")


def test_real_unscored_partial_is_archived_without_data_loss(tmp_path):
    item = tmp_path / "task-000"
    item.mkdir()
    (item / "official-evaluator.private.log").write_bytes(b"original incomplete transcript")
    archived = r.quarantine_unfinished_item(item)
    assert not item.exists()
    assert archived.name.startswith("task-000.unfinished-")
    assert (archived / "official-evaluator.private.log").read_bytes() == (
        b"original incomplete transcript")


def test_reject_quarantine_if_scored_claim_exists(tmp_path):
    item = tmp_path / "task-000"
    item.mkdir()
    (item / "receipt.json").write_bytes(b"potential completed receipt")
    with pytest.raises(ValueError, match="not an unscored"):
        r.quarantine_unfinished_item(item)


def test_reject_foreign_unfinished_task_files(tmp_path):
    item = tmp_path / "task-000"
    item.mkdir()
    (item / "untrusted.py").write_text("print('not executed')")
    with pytest.raises(ValueError, match="foreign files"):
        r.quarantine_unfinished_item(item)


def test_reject_forged_original_score_receipt(tmp_path):
    item = tmp_path / "task-000"
    item.mkdir()
    (item / "public-automationbench.private.json").write_text("{}")
    receipt = {"schema": r.SCHEMA, "owner_sha256": "a" * 64,
               "original_index": 0, "task": {"id": "501", "name": "sales.multi_hop_lookup"},
               "served_id": "model", "status": "ORIGINAL_OFFICIAL_TASK_SCORED",
               "raw_sha256": "b" * 64, "independent_score": {"passed_tasks": 1}}
    (item / "receipt.json").write_bytes(r._canonical(receipt))
    with pytest.raises(ValueError, match="drift"):
        r.verify_item(item, 0, receipt["task"], "a" * 64, "model")


@pytest.mark.parametrize("value", [0, -1, 601, 1.2, True])
def test_reject_invalid_replay_budgets(tmp_path, value):
    with pytest.raises(ValueError, match="max_new"):
        r.run(tmp_path / "unused", execute=False, max_new=value)


def test_reject_duplicate_or_nonfinite_receipts(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text('{"x":1,"x":2}')
    with pytest.raises(ValueError, match="duplicate"):
        r._json(p)
    p.write_text('{"x":NaN}')
    with pytest.raises(ValueError):
        r._json(p)


def test_source_identity_count_is_strict():
    assert r.POPULATION == 600
    assert r.source_ids.PUBLIC_PROMPTS_SHA256 == (
        "1c8feeec1f680b05b392a5d00fc634edb143c3e7bdd152c476af50cebcec57b5"
    )
    assert Path(r.q4.GGUF).suffix == ".gguf"


def original_gate(*, ram=12, commit=20, gpu=12600, blockers=None):
    return {
        "status": "REFUSED",
        "blockers": blockers if blockers is not None else [
            "RAM_BELOW_FROZEN_FLOOR", "COMMIT_BELOW_FROZEN_FLOOR",
            "GPU_FREE_BELOW_64K_FLOOR",
        ],
        "measured": {
            "ram_available_gib": ram, "commit_available_gib": commit,
            "gpu_free_mib": gpu, "disk_free_gib": 25,
        },
    }


def test_explicit_bypass_admits_measured_72k_allocator_not_old_padded_floor():
    original = original_gate()
    result = r._override_read_only_gate(original, bypass_conservative=True)
    assert result["status"] == "READY_TO_ATTEMPT_NONCOMPARABLE_EXPLORATORY"
    assert result["explicit_conservative_resource_bypass"] is True
    assert result["frozen_gen0_status"] == "BLOCKED"
    assert result["release_approved"] is False
    assert 12500 <= r.OVERRIDE_GPU_MIN_FREE_MIB < r.q4.MIN_GPU_MIB + r.q4.GPU_MARGIN_MIB
    assert original["status"] == "REFUSED"  # no original scorer/gate mutation


def test_default_resource_floors_stay_frozen_without_explicit_opt_in():
    original = original_gate()
    assert r._override_read_only_gate(original, bypass_conservative=False) is original
    assert r.q4.MIN_GPU_MIB == 12288
    assert r.q4.MIN_RAM_GIB == 29.3145986328125
    assert r.q4.MIN_COMMIT_GIB == 34.3145986328125


@pytest.mark.parametrize("block", [
    "TARGET_PORT_IN_USE", "OUTPUT_DIRECTORY_ALREADY_EXISTS",
    "EPISODE_CANNOT_FIT_DECLARED_CONTEXT",
    "PRIOR_TRAJECTORY_UNAVAILABLE_TO_ADMIT", "DISK_BELOW_FROZEN_FLOOR",
])
def test_explicit_opt_in_cannot_override_source_ports_or_original_context(block):
    result = r._override_read_only_gate(
        original_gate(blockers=[block, "RAM_BELOW_FROZEN_FLOOR"]),
        bypass_conservative=True,
    )
    assert result["status"] == "REFUSED"
    assert block in result["blockers"]
    assert "RAM_BELOW_FROZEN_FLOOR" not in result["blockers"]


@pytest.mark.parametrize("overrides", [
    {"ram": 6.99}, {"commit": 7.99}, {"gpu": 233},
    {"gpu": 12500}, {"ram": 0}, {"commit": -1},
])
def test_override_never_doubles_up_host_oom_or_oversubscribed_gpu(overrides):
    result = r._override_read_only_gate(
        original_gate(**overrides), bypass_conservative=True,
    )
    assert result["status"] == "REFUSED"
    assert "PHYSICAL_ALLOCATION_OR_RUNNING_OS_RESERVE_UNAVAILABLE" in result["blockers"]


def test_admitted_explicit_override_runtime_retains_native_allocator_fit(monkeypatch, tmp_path):
    monkeypatch.setattr(
        r.q4, "memory_snapshot",
        lambda: {"ram_available_gib": 12, "commit_available_gib": 21},
    )
    monkeypatch.setattr(r.q4, "gpu_free_mib", lambda: 12600)
    monkeypatch.setattr(
        r.shutil, "disk_usage",
        lambda _path: type("Disk", (), {"free": 20 * 1024 ** 3})(),
    )
    result = r._native_prestart(tmp_path, bypass_conservative=True)
    assert result["prestart_admission"] == (
        "EXPLICIT_CONSERVATIVE_FLOOR_BYPASS_NONCOMPARABLE")
    assert result["measured_required_native_gpu_mib"] == r.OVERRIDE_GPU_MIN_FREE_MIB
    with pytest.raises(RuntimeError, match="72k model floors"):
        r._native_prestart(tmp_path, bypass_conservative=False)
