"""Synthetic telemetry only: exact host source/genuine process RED/GREEN guards."""

from pathlib import Path

import pytest

from chowder_batch import reap_original600_live_windows_resources_20261027 as m


def observed(index, **overrides):
    r = {
        "observed_unix_seconds": float(index + 100),
        "original_evaluator_pid": 1001,
        "original_native_pid": 1002,
        "physical_gpu_uuid": "GPU-fixed",
        "source_owner_sha256": "a" * 64,
        "native_wset_bytes": 22_000_000_000 + index,
        "native_private_bytes": 2_500_000_000 + index,
        "native_pagefile_bytes": 2_500_000_000 + index,
        "native_page_faults_cumulative": 10000 + index,
        "physical_available_bytes": 15_000_000_000 - index,
        "commit_headroom_bytes": 21_000_000_000 - index,
        "commit_total_bytes": 27_000_000_000 + index,
        "host_gpu_used_mib": 14000 + index,
        "host_gpu_free_mib": 2300 - index,
        "host_gpu_unaccounted_wddm_reserved_mib": 11,
        "completed_receipt_files": 20 + index,
    }
    r.update(overrides)
    return r


def test_actual_resource_summary_never_claims_exclusive_gpu_or_gen0_pass():
    result = m.summarize([observed(0), observed(1), observed(2)],
                         source_root=Path("F:/orig600"), owner_sha="a" * 64,
                         parent_created=42, native_created=43)
    assert result["samples"] == 3
    assert result["max_native_wset_bytes"] == 22_000_000_002
    assert result["max_host_gpu_used_mib"] == 14002
    assert result["min_host_gpu_free_mib"] == 2298
    assert result["native_page_fault_count_delta"] == 2
    assert result["host_gpu_used_is_model_exclusive"] is False
    assert result["first_completed_official_item_count"] == 20
    assert result["last_completed_official_item_count"] == 22
    assert result["frozen_gen0_speed_gate_proven"] is False
    assert result["gen0_status"] == "BLOCKED"
    assert result["release_approved"] is False
    assert result["chosen"] is None


@pytest.mark.parametrize("bad", [
    [observed(0)],
    [observed(0), observed(1, original_native_pid=666)],
    [observed(0), observed(1, original_evaluator_pid=666)],
    [observed(0), observed(1, physical_gpu_uuid="GPU-other")],
    [observed(0), observed(1, source_owner_sha256="b" * 64)],
    [observed(1), observed(0)],
    [observed(0), observed(1, observed_unix_seconds=float("nan"))],
    [observed(0), observed(1, native_page_faults_cumulative=5)],
])
def test_resource_identity_or_time_drift_never_relabels_sampling_as_peak(bad):
    with pytest.raises(ValueError):
        m.summarize(bad, source_root=Path("F:/orig600"),
                    owner_sha="a" * 64, parent_created=42, native_created=43)


def test_reject_missing_original_evaluator_pid(monkeypatch, tmp_path):
    monkeypatch.setattr(m.psutil, "process_iter", lambda _: iter(()))
    with pytest.raises(RuntimeError, match="exactly one"):
        m.identify_owned(tmp_path)


def test_gpu_host_rejects_fake_multiple_gpu_or_wrong_vram(monkeypatch):
    class Fake:
        returncode = 0
        stdout = "GPU-id1, 16311, 14000, 2311, 75\nGPU-id2, 16311, 14000, 2311, 20\n"

    monkeypatch.setattr(m.subprocess, "run", lambda *a, **kw: Fake())
    with pytest.raises(RuntimeError, match="single original"):
        m.gpu_host()
    Fake.stdout = "GPU-id1, 16311, 18000, -1689, 75"
    with pytest.raises(RuntimeError, match="implausible"):
        m.gpu_host()
    Fake.stdout = "GPU-id1, 16311, 15767, 284, 62"
    assert m.gpu_host()["host_gpu_unaccounted_wddm_reserved_mib"] == 260
    Fake.stdout = "GPU-id1, 16311, 7000, 100, 62"
    with pytest.raises(RuntimeError, match="implausible"):
        m.gpu_host()


def test_windows_resource_telemetry_not_simulated_elsewhere(monkeypatch):
    monkeypatch.setattr(m.sys, "platform", "linux")
    with pytest.raises(RuntimeError, match="native Windows"):
        m.windows_memory()


def startup_bytes() -> bytes:
    return (
        "I srv load_model: loading model '" + str(m.run.q4.GGUF) + "'\n"
        "I llama_context: n_ctx                 = 73728\n"
        "I print_info: n_expert              = 128\n"
        "I print_info: n_expert_used         = 8\n"
        "I load_tensors: offloading 39 repeating layers to GPU\n"
        "I load_tensors: offloaded 41/41 layers to GPU\n"
        "I load_tensors: CPU_Mapped model buffer size = 397.85 MiB\n"
        "I load_tensors: CUDA0 model buffer size = 10619.88 MiB\n"
        "I llama_kv_cache: CUDA0 KV buffer size = 1440.00 MiB\n"
        "I llama_kv_cache: size = 1440 MiB ( 73728 cells,  10 layers,  1/1 seqs)\n"
        "I llama_memory_recurrent: CUDA0 RS buffer size = 62.81 MiB\n"
        "I sched_reserve: CUDA0 compute buffer size = 172.05 MiB\n"
        "I sched_reserve: CUDA_Host compute buffer size = 80.02 MiB\n"
        "I srv load_model: prompt cache is enabled, size limit: 8192 MiB\n"
    ).encode()


def test_original_real_allocator_geometry_explicitly_distinguishes_cache_limit():
    observed = m.original_native_startup(startup_bytes())
    assert observed["native_cuda_offloaded_layers"] == 41
    assert observed["native_cpu_expert_offload"] is False
    assert observed["active_experts_per_token_per_moe_layer"] == 8
    assert observed["stored_experts_per_moe_layer"] == 128
    assert observed["full_attention_kv_layers"] == 10
    assert observed["measured_original_cuda_allocator_mib"]["cuda_kv_buffer_mib"] == 1440.0
    assert observed["measured_original_cuda_allocator_mib"]["prompt_cache_limit_mib"] == 8192
    assert observed["cache_limit_is_an_allocation_ceiling_not_observed_peak"] is True


@pytest.mark.parametrize("old,new", [
    ("41/41 layers", "39/41 layers"),
    ("n_expert_used         = 8", "n_expert_used         = 1"),
    ("KV buffer size = 1440.00 MiB", "KV buffer size = 7000.00 MiB"),
    ("73728 cells,  10 layers", "73728 cells,  40 layers"),
    ("prompt cache is enabled, size limit: 8192 MiB",
     "prompt cache is enabled, size limit: 4096 MiB"),
    ("offloading 39 repeating layers", "offloading 30 repeating layers"),
    ("n_ctx                 = 73728", "n_ctx                 = 16384"),
])
def test_fake_offload_kv_cache_architecture_or_context_fails_closed(old, new):
    with pytest.raises(ValueError):
        m.original_native_startup(startup_bytes().replace(old.encode(), new.encode()))


def test_duplicated_forged_native_kv_allocations_refuse():
    raw = startup_bytes()
    with pytest.raises(ValueError, match="ambiguous"):
        m.original_native_startup(raw + b"I cache CUDA0 KV buffer size = 1440.00 MiB\n")
