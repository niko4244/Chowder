"""Reproducible source-only snapshots of the *provisional* cloud workloads.

Never run a model or contact Kaggle in this CPU-only regression file.
"""

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS = ROOT / "scripts" / "reap_v2_kaggle_qualification_20261026"


def _source(name: str) -> str:
    return (SNAPSHOTS / name).read_text(encoding="utf-8")


def test_each_snapshot_is_exactly_hashed_source_without_private_model_bytes():
    manifest = json.loads((SNAPSHOTS / "SOURCE-SHA256-MANIFEST.json").read_text("utf-8"))
    assert manifest["schema"] == "chowder.reap128-v2-private-kaggle-runner-snapshots.v1"
    assert manifest["input_models_not_included"] is True
    assert manifest["frozen_core_flags"] == "--fit on -np 4 -c 16384 --jinja"
    assert len(manifest["snapshots"]) == 8
    names = [entry["file"] for entry in manifest["snapshots"]]
    assert len(names) == len(set(names))
    assert {p.name for p in SNAPSHOTS.iterdir()} == {
        *names, "SOURCE-SHA256-MANIFEST.json", "README.md"
    }
    for entry in manifest["snapshots"]:
        path = SNAPSHOTS / entry["file"]
        assert path.suffix == ".py" and path.is_file() and not path.is_symlink()
        content = path.read_bytes()
        assert len(content) == entry["bytes"]
        assert hashlib.sha256(content).hexdigest() == entry["sha256"]
        ast.parse(content, filename=str(path))


def test_full_two_suite_runner_retains_original_source_scores_and_hard_guards():
    source = _source("candidate_full_1319_164.py")
    assert "original._digest(rows)" in source
    assert "original._digest(human_rows)" in source
    assert "('gsm8k',rows),('humaneval',human_rows)" in source
    assert "max_workers=4" in source
    assert "original._chat" in source and "original._score" in source
    assert "check_runtime(h)" in source and "check_owner()" in source
    assert "FLAGS=['--fit','on','-np','4','-c','16384','--jinja']" in source
    assert "COMPLETE_FULL_DATA_PROVISIONAL_NOT_FROZEN_WINDOWS_GATE" in source


def test_both_speed_candidates_are_exact_original_first_200_under_four_way_load():
    for name in ("candidate_q4_first200.py", "dense_q4_first200.py"):
        source = _source(name)
        assert "for suite,items in [('gsm8k',rows[:200])]:" in source
        assert "max_workers=4" in source
        assert "check_owner()" in source
        assert "FLAGS=['--fit','on','-np','4','-c','16384','--jinja']" in source
        assert "PROVISIONAL_NOT_FROZEN_WINDOWS_GATE" in source
    dense = _source("dense_q4_first200.py")
    assert "2c6d9466129e8802839566b35728e2506aaf1e8705cd137009e2f08feb9dd049" in dense
    assert "h['cgroup_unused_gib']<10" in dense
    assert "construction_receipt_not_applicable" in dense


def test_single_gpu_is_explicitly_labeled_provisional_not_linux_windows_parity():
    source = _source("single_t4_parity.py")
    assert "CUDA_VISIBLE_DEVICES='0'" in source
    assert "not_official_windows_serving_attestation" in source
    assert "scientific_selection':'UNKNOWN'" in source
    assert "POSIX_FADV_DONTNEED" in source


def test_original_linux_source_is_pinned_and_allocator_difference_disclosed():
    source = _source("pinned_linux_cuda_build.py")
    assert "7609846557c50f9d984719a9e1e8c5f3d02f807b" in source
    assert "-DGGML_CUDA_NO_VMM=ON" in source
    assert "cuda_vmm_disabled_due_to_missing_driver_link_target" in source
