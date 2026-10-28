"""Adversarial CPU-only canaries for Phase49 resource evidence and admission."""

import copy
import json

import pytest

from chowder_batch import reap_v2_phase49_resource_diagnostics_20261028 as phase49


def fixture_observation():
    return {
        "commit_headroom_bytes": 33 * phase49.GIB,
        "commit_limit_bytes": 110 * phase49.GIB,
        "commit_total_bytes": 77 * phase49.GIB,
        "commit_peak_bytes": 80 * phase49.GIB,
        "physical_available_bytes": 13 * phase49.GIB,
        "physical_total_bytes": 64 * phase49.GIB,
        "native_wset_bytes": 24 * phase49.GIB,
        "native_private_bytes": 23 * phase49.GIB,
        "native_pagefile_bytes": 23 * phase49.GIB,
        "native_page_faults_cumulative": 100000,
        "host_gpu_total_mib": 16311,
        "host_gpu_used_mib": 15740,
        "host_gpu_free_mib": 311,
        "host_gpu_unaccounted_wddm_reserved_mib": 260,
        "original_native_pid": 29744,
        "original_evaluator_pid": 45380,
        "completed_receipt_files": 23,
        "source_owner_sha256": "a" * 64,
        "observed_unix_seconds": 1793120211.0,
    }


def test_red_12_row_requires_actual_physical_floor_crossing():
    samples = [
        {"s": 0.0, "ram_available_gib": 21.39, "commit_available_gib": 28.84},
        {"s": 98.4, "ram_available_gib": 6.98, "commit_available_gib": 11.03},
    ]
    checked = phase49.analyze_abort_samples(
        {"initial": {"ram_available_gib": 21.41, "commit_available_gib": 28.84},
         "memory_floors_gib": {"ram": 7, "commit": 8}, "samples": samples,
         "status": "ABORT_UNCHANGED_RUNTIME_MEMORY_FLOOR", "exit_code": 15}
    )
    assert checked["direct_abort_cause"] == "PHYSICAL_RAM_BELOW_FROZEN_RUNTIME_FLOOR"
    assert checked["minimum_ram_available_gib"] == 6.98
    assert checked["commit_floor_crossed"] is False
    altered = copy.deepcopy(samples)
    altered[-1]["ram_available_gib"] = 7.01
    with pytest.raises(ValueError, match="no verified floor crossing"):
        phase49.analyze_abort_samples({"initial": {"ram_available_gib": 21.41,
             "commit_available_gib": 28.84}, "memory_floors_gib": {"ram": 7, "commit": 8},
             "samples": altered, "status": "ABORT_UNCHANGED_RUNTIME_MEMORY_FLOOR", "exit_code": 15})


def test_red_duplicate_key_bool_nan_are_not_valid_source():
    with pytest.raises(ValueError, match="duplicate"):
        phase49.strict_json(b'{"ram": 7, "ram": 2}')
    with pytest.raises(ValueError, match="nonfinite"):
        phase49.strict_json(b'{"ram": NaN}')
    with pytest.raises(ValueError, match="finite|number"):
        phase49.valid_number(True, "ram")


def test_red_guarded_native_observation_rejects_inconsistent_host_memory():
    observation = fixture_observation()
    assert phase49.check_native_sample(observation, "a" * 64, 29744, 45380)[
        "native_wset_bytes"] == 24 * phase49.GIB
    bad = copy.deepcopy(observation)
    bad["commit_total_bytes"] += 4096
    with pytest.raises(ValueError, match="commit"):
        phase49.check_native_sample(bad, "a" * 64, 29744, 45380)
    bad = copy.deepcopy(observation)
    bad["host_gpu_free_mib"] += 260
    with pytest.raises(ValueError, match="WDDM"):
        phase49.check_native_sample(bad, "a" * 64, 29744, 45380)


def test_red_native_snapshot_counter_reset_and_pid_mismatch():
    first = fixture_observation()
    second = copy.deepcopy(first)
    second["observed_unix_seconds"] += 2.0
    second["native_page_faults_cumulative"] += 100
    verified = phase49.analyze_native_samples([first, second], "a" * 64, 29744, 45380)
    assert verified["page_fault_count_delta"] == 100
    assert verified["hard_page_faults_and_pagefile_io_proven"] is False
    second["native_page_faults_cumulative"] -= 500
    with pytest.raises(ValueError, match="counter reset"):
        phase49.analyze_native_samples([first, second], "a" * 64, 29744, 45380)
    second["native_page_faults_cumulative"] += 500
    second["original_native_pid"] = 100
    with pytest.raises(ValueError, match="PID"):
        phase49.analyze_native_samples([first, second], "a" * 64, 29744, 45380)


def test_red_measured_kv_geometry_is_not_extrapolated_gate_pass():
    four = phase49.parse_four_slot_native_startup(
        b"common_fit_params: successfully fit params to free device memory\n"
        b"load_tensors: offloaded 41/41 layers to GPU\n"
        b"llama_context: n_seq_max = 4\n"
        b"llama_kv_cache: CUDA0 KV buffer size = 320.00 MiB\n"
        b"srv load_model: initializing, n_slots = 4, n_ctx_slot = 4096, kv_unified = 'false'\n"
        b"srv load_model: prompt cache is enabled, size limit: 8192 MiB\n"
    )
    assert four["cuda_kv_mib"] == 320.0
    assert four["per_slot_context_tokens"] == 4096
    assert phase49.compare_measured_kv(four["cuda_kv_mib"], 1440., 16384, 73728)[
        "same_measured_kv_bytes_per_token"] is True
    with pytest.raises(ValueError, match="four-slot|KV|fit"):
        phase49.parse_four_slot_native_startup(
            b"load_tensors: offloaded 41/41 layers to GPU\n"
            b"llama_kv_cache: CUDA0 KV buffer size = 320.00 MiB\n"
        )


def test_red_capacity_does_not_grant_launch_even_when_plenty():
    snap = {"physical_available_bytes": 60 * phase49.GIB,
            "commit_headroom_bytes": 70 * phase49.GIB,
            "disk_free_bytes": 20 * phase49.GIB,
            "gpu": {"total_mib": 65536, "free_mib": 60000,
                    "used_mib": 5000, "reserved_mib": 536},
            "active_conflicts": [], "listeners": [], "port_census_complete": True,
            "process_census_complete": True}
    admitted = phase49.capacity_screen(snap)
    assert admitted["provisional_resource_capacity_only"] is True
    assert admitted["execution_approved"] is False
    assert admitted["gen0_status"] == "BLOCKED"
    snap["gpu"]["free_mib"] = 16000
    snap["gpu"]["used_mib"] = 65536 - 16000 - 536
    denied = phase49.capacity_screen(snap)
    assert denied["provisional_resource_capacity_only"] is False
    assert "GPU_40G_DEDICATED" in denied["reasons"]


def test_red_missing_observer_process_census_must_refuse():
    snap = {"physical_available_bytes": 60 * phase49.GIB,
            "commit_headroom_bytes": 60 * phase49.GIB,
            "disk_free_bytes": 12 * phase49.GIB,
            "gpu": {"total_mib": 65536, "free_mib": 55000, "used_mib": 10000,
                    "reserved_mib": 536},
            "active_conflicts": [], "listeners": [],
            "port_census_complete": False, "process_census_complete": True}
    assert "INCOMPLETE_PROCESS_OR_PORT_CENSUS" in phase49.capacity_screen(snap)["reasons"]


def test_red_sha_pin_rejects_resealed_prior_model_evidence():
    assert phase49.sha256_bytes(b"hello") == (
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824")
    with pytest.raises(ValueError, match="source SHA"):
        phase49.require_sha(b"a", "0" * 64)


def test_red_historical_receipt_rejects_mixed_native_sessions():
    current = {"source_owner_sha256": "a" * 64,
               "native_original_pid": 29744, "original_evaluator_pid": 45380,
               "native_original_flags": list(phase49.ORIGINAL_ONE_SLOT_FLAGS),
               "source_bound_native_allocator": {
                   "slots": 1, "native_context_tokens": 73728,
                   "measured_original_cuda_allocator_mib": {
                       "cuda_kv_buffer_mib": 1440.0, "cuda_model_buffer_mib": 10619.88,
                       "cuda_compute_buffer_mib": 172.05,
                       "cuda_recurrent_state_buffer_mib": 62.81,
                       "prompt_cache_limit_mib": 8192.0},
                   "native_cpu_expert_offload": False,
                   "native_cuda_offloaded_layers": 41,
                   "native_cuda_total_layers": 41,
                   "original_gguf_sha256": phase49.Q4_MODEL_SHA}}
    assert phase49.check_one_slot_provenance(current, "a" * 64)["cuda_kv_mib"] == 1440.0
    current["native_original_flags"] = list(phase49.FROZEN_FOUR_SLOT_FLAGS)
    with pytest.raises(ValueError, match="flags"):
        phase49.check_one_slot_provenance(current, "a" * 64)


def test_red_cannot_treat_pagefile_bytes_as_disk_paging():
    observation = fixture_observation()
    output = phase49.check_native_sample(observation, "a" * 64, 29744, 45380)
    assert output["native_pagefile_bytes"] == 23 * phase49.GIB
    assert output["pagefile_io_bytes"] is None
    assert output["hard_page_faults"] is None


def test_red_historical_cache_sizing_is_original_floor_not_fit_approval():
    history = {"direct_abort_cause": "PHYSICAL_RAM_BELOW_FROZEN_RUNTIME_FLOOR",
               "maximum_observed_host_ram_drop_gib": 14.43,
               "maximum_observed_host_commit_drop_gib": 18.43}
    log = (b"prompt cache is enabled, size limit: 8192 MiB\n"
           b"cache state: 11 prompts, 2166.171 MiB\n")
    sizing = phase49.historical_cache_sizing(log, history)
    assert sizing["historical_conservative_first200_ram_start_gib"] == pytest.approx(29.3145986328125)
    assert sizing["historical_conservative_first200_commit_start_gib"] == pytest.approx(34.3145986328125)
    assert sizing["execution_approved"] is False
    with pytest.raises(ValueError, match="cache"):
        phase49.historical_cache_sizing(log.replace(b"8192", b"16384"), history)


def test_red_four_slot_host_reserved_minima_never_pretend_process_peak():
    # Synthetic input cannot satisfy authentic pinned 822-row original source.
    with pytest.raises(ValueError, match="terminal/session"):
        phase49.original_four_slot_watchdog(b'{}\n', {"status": "fake"})
