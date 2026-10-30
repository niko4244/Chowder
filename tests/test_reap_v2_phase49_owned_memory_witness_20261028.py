"""CPU-only source-bound Job identity and sampled Windows memory regressions."""

from types import SimpleNamespace

import pytest

from chowder_batch import reap_v2_phase49_owned_memory_witness_20261028 as witness
from chowder_batch import reap_v2_live_job_lineage_guard_20261028 as lineage


def fixture(monkeypatch):
    class Owner:
        active = True
        job = object()
        process = object()
        members = {1001, 1002, 1003}

        def owner_valid(self):
            return True

        def member_ids(self):
            return list(self.members)

        def is_member(self, pid):
            return pid in self.members

    monkeypatch.setattr(witness.owner_api, "_WindowsOwnedJob", Owner)
    monkeypatch.setattr(witness.sys, "platform", "win32")
    obj = Owner()
    ledger = lineage.LineageLedger(root_pid=1001, root_birth=100.0)
    ledger.full_active_witnessed = True
    ledger.roles = {"root": (1001, 100.0), "server": (1002, 101.0),
                    "evaluator": (1003, 102.0)}

    class Process:
        def __init__(self, pid):
            self.pid = pid

        def create_time(self):
            return {1001: 100.0, 1002: 101.0, 1003: 102.0}[self.pid]

        def memory_info(self):
            return SimpleNamespace(rss=1024, peak_wset=2048, private=768,
                                   pagefile=768, num_page_faults=17)

    return obj, ledger, Process


def test_native_owned_three_role_window_memory_is_bounded_and_never_qualification(monkeypatch):
    owner, ledger, process = fixture(monkeypatch)
    result = witness.sample_owned_memory(owner, ledger, process_factory=process)
    assert result["sampled_job_working_set_sum_bytes"] == 3072
    assert result["native_server_reported_peak_working_set_so_far_bytes"] == 2048
    assert result["whole_lifetime_peak_proven"] is False
    assert result["hard_page_faults_or_pagefile_io_measured"] is False
    assert result["gen0"]["chosen"] is None
    summary = witness.summarize_samples([result])
    assert summary["max_sampled_native_working_set_bytes"] == 1024
    assert summary["native_gpu_allocation_peak"] is None
    assert summary["whole_lifetime_peak_proven"] is False


@pytest.mark.parametrize("fault", ["inactive", "unowned", "birth", "torn", "missing",
                                    "peak", "negative", "no_full_census"])
def test_unknown_or_foreign_native_memory_cannot_be_reported_as_owned(monkeypatch, fault):
    owner, ledger, good = fixture(monkeypatch)
    if fault == "inactive":
        owner.active = False
    if fault == "unowned":
        owner.members.remove(1002)
    if fault == "missing":
        del ledger.roles["evaluator"]
    if fault == "no_full_census":
        ledger.full_active_witnessed = False
    class Bad(good):
        def create_time(self):
            return super().create_time() + (20 if fault == "birth" and self.pid == 1002 else 0)

        def memory_info(self):
            info = super().memory_info()
            if fault == "peak":
                info.peak_wset = 1
            elif fault == "negative":
                info.private = -1
            elif fault == "torn" and self.pid == 1002:
                owner.members.remove(1002)
            return info

    with pytest.raises(witness.MemoryWitnessRefused):
        witness.sample_owned_memory(owner, ledger, process_factory=Bad)


def test_missing_data_or_forged_sample_not_a_scientific_peak():
    assert witness.summarize_samples([])["status"] == "UNKNOWN_NO_FULL_JOB_MEMORY_SAMPLE"
    with pytest.raises(witness.MemoryWitnessRefused):
        witness.summarize_samples([{"schema": witness.SCHEMA, "status": "PASS"}])


def test_no_pid_kills_or_gpu_actions_exposed():
    for name in ("terminate", "kill", "Popen", "run_gpu", "nvidia_smi"):
        assert not hasattr(witness, name)
