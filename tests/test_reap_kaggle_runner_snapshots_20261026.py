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
    assert len(manifest["snapshots"]) == 16
    names = [entry["file"] for entry in manifest["snapshots"]]
    assert len(names) == len(set(names))
    assert {p.name for p in SNAPSHOTS.iterdir()} == {
        *names, "SOURCE-SHA256-MANIFEST.json", "README.md",
        "dense_q4_first200_material_postload.py",
        "SOURCE-SHA256-UNLAUNCHED-DENSE-20261026.json",
        ".gitattributes",
    }
    for entry in manifest["snapshots"]:
        path = SNAPSHOTS / entry["file"]
        assert path.suffix == ".py" and path.is_file() and not path.is_symlink()
        content = path.read_bytes()
        assert len(content) == entry["bytes"]
        assert hashlib.sha256(content).hexdigest() == entry["sha256"]
        ast.parse(content, filename=str(path))


def test_unlaunched_dense_source_is_separately_exact_hashed_not_historical_run():
    original_manifest = (SNAPSHOTS / "SOURCE-SHA256-MANIFEST.json").read_bytes()
    extra = json.loads(
        (SNAPSHOTS / "SOURCE-SHA256-UNLAUNCHED-DENSE-20261026.json").read_text("utf-8")
    )
    assert extra["schema"] == "chowder.reap-v2-new-unlaunched-dense-first200-runner.v1"
    assert extra["status"] == "SOURCE_ONLY_NEVER_EXECUTED_ON_KAGGLE"
    assert extra["historical_snapshot_manifest_sha256"] == hashlib.sha256(original_manifest).hexdigest()
    assert extra["source_file"] == "dense_q4_first200_material_postload.py"
    candidate = SNAPSHOTS / extra["source_file"]
    assert candidate.is_file() and not candidate.is_symlink()
    source_bytes = candidate.read_bytes()
    assert len(source_bytes) == extra["source_bytes"]
    assert hashlib.sha256(source_bytes).hexdigest() == extra["source_sha256"]
    ast.parse(source_bytes, filename=str(candidate))
    assert extra["benchmark_executed"] is False
    assert extra["model_fit_proven"] is False
    assert extra["speed_ratio_certified"] is False
    assert extra["release_approved"] is False
    assert extra["windows_equivalent"] is False
    assert extra["frozen_serve_flags"] == "--fit on -np 4 -c 16384 --jinja"
    assert extra["frozen_dense_q4_sha256"] == (
        "2c6d9466129e8802839566b35728e2506aaf1e8705cd137009e2f08feb9dd049"
    )
    assert (SNAPSHOTS / ".gitattributes").read_text("utf-8").splitlines()[-1] == (
        "dense_q4_first200_material_postload.py -text"
    )


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


def test_independent_humaneval_replay_never_loads_model_or_grants_candidate_privileges():
    source = _source("human164_lowpriv_independent.py")
    assert "original_windows_gate_eligible':False" in source
    assert "no_model_loaded':True" in source
    assert "enable_internet" not in source  # set by the private CPU notebook metadata
    assert "resource.setrlimit(resource.RLIMIT_CPU" in source
    assert "resource.setrlimit(resource.RLIMIT_AS" in source
    assert "os.setgid(65534);os.setuid(65534)" in source
    assert "sys.executable,'-I'" in source
    assert "subprocess.run" in source and "timeout=12" in source
    assert "len(data)!=164 or len(rows)!=164" in source
    assert "matched!=164 or correct!=151" in source
    assert "PASS_INDEPENDENT_ISOLATED_HUMANEVAL_FULL_164_PROVISIONAL_LINUX" in source


def test_original_q8_smoke_and_full_runner_keep_four_way_frozen_flags_and_guards():
    for name in ("q8_native_original_smoke.py", "q8_complete_1319_164.py"):
        source = _source(name)
        assert "FLAGS=['--fit','on','-np','4','-c','16384','--jinja']" in source
        assert "89dcaf33b61f59c83a31b718ba98b73c33437d0320ae0cc56737a5d7e43519c6" in source
        assert "a1cfe1464df4579ea9d5ae1320efc01c0de5dae1070235caca18c821ce67479d" in source
        assert "h['cgroup_unused_gib']<10" in source
        assert "not_official_windows_serving_attestation" in source
        assert "GGML_CUDA_NO_VMM" not in source  # Binary provenance recorded separately
    full = _source("q8_complete_1319_164.py")
    assert "for suite,items in [('gsm8k',rows),('humaneval',human_rows)]:" in full
    assert "max_workers=4" in full
    assert "original._score" in full
    assert "COMPLETE_Q8_FULL_DATA_PROVISIONAL_NOT_FROZEN_WINDOWS_GATE" in full


def test_private_model_rehash_checks_original_q8_and_independent_random_control():
    source = _source("q8_random128_private_hash_only.py")
    assert "601871e00403658708eb4eebf3b05f9dc54a93b326e31f2f87b48cb59f92241b" in source
    assert "89dcaf33b61f59c83a31b718ba98b73c33437d0320ae0cc56737a5d7e43519c6" in source
    assert "without_model_inference':True" in source
    assert "no_scientific_quality_gate':True" in source


def test_frozen_negative_control_smoke_never_claims_q8_candidate_construction():
    source = _source("random128_original_smoke.py")
    assert "601871e00403658708eb4eebf3b05f9dc54a93b326e31f2f87b48cb59f92241b" in source
    assert "weights=source('random128-q8_0.gguf',19748435872,ORIGINAL_GGUF_SHA)" in source
    assert "receipt=None" in source
    assert "FLAGS=['--fit','on','-np','4','-c','16384','--jinja']" in source
    assert "no official" in source.lower() or "NOT_OFFICIAL_SCORE" in source


def test_postload_q8_cache_probe_requires_material_measured_resource_improvement():
    source = _source("q8_postload_cache_resource_probe.py")
    assert "weights.open('rb',buffering=0)" in source
    assert "os.posix_fadvise(descriptor.fileno(),0,0,os.POSIX_FADV_DONTNEED)" in source
    assert "recovered<4" in source
    assert "R['post_reclaim_host']['cgroup_unused_gib']<14" in source
    assert "read-only cache reclaim insufficient: no Q8 full rerun" in source
    assert "FLAGS=['--fit','on','-np','4','-c','16384','--jinja']" in source


def test_material_cache_recovery_is_bound_to_new_full_sessions_not_aborted_rows():
    for name, model_sha in (
        ("q8_complete_postload_reclaim.py", "89dcaf33b61f59c83a31b718ba98b73c33437d0320ae0cc56737a5d7e43519c6"),
        ("random128_complete_postload_reclaim.py", "601871e00403658708eb4eebf3b05f9dc54a93b326e31f2f87b48cb59f92241b"),
    ):
        source = _source(name)
        assert model_sha in source
        assert "FLAGS=['--fit','on','-np','4','-c','16384','--jinja']" in source
        assert "for suite,items in [('gsm8k',rows),('humaneval',human_rows)]:" in source
        assert "max_workers=4" in source and "original._score" in source
        assert "session=uuid.uuid4().hex" in source
        assert "os.posix_fadvise(handle.fileno(),0,0,os.POSIX_FADV_DONTNEED)" in source
        assert "recovery<4" in source
        assert "R['post_reclaim_host']['cgroup_unused_gib']<14" in source
        assert "h['cgroup_unused_gib']<10" in source
        assert "not_official_windows_serving_attestation" in source
    random_source = _source("random128_complete_postload_reclaim.py")
    assert "receipt=None" in random_source
    assert "independent_control_source_sha256" in random_source
