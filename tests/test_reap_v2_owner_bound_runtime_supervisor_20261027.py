"""CPU-only adversarial owner-bound runtime checks; never start a model."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from chowder_batch import reap_v2_owner_bound_runtime_supervisor_20261027 as supervisor
from chowder_batch import reap_v2_frozen_wave_20261027 as wave


def runtime(**updates):
    result = {
        "physical_available_bytes": 9 * supervisor.GIB,
        "commit_headroom_bytes": 11 * supervisor.GIB,
        "disk_free_bytes": 12 * supervisor.GIB,
        "gpu": {
            "uuid": "GPU-12345678",
            "total_mib": 65536,
            "used_mib": 46000,
            "free_mib": 19300,
            "reserved_mib": 236,
            "utilization_pct": 20,
        },
        "root_owner_valid": True,
        "job_processes_valid": True,
        "foreign_ports": [],
        "foreign_processes": [],
        "owned_listener": True,
        "native_started": True,
        "native_residency": {
            "layers_offloaded": 41,
            "layers_total": 41,
            "context": 16384,
            "kv_total_mib": 320.0,
            "cpu_expert_offload": 0,
        },
        "startup_seconds": 30,
        "elapsed_seconds": 60,
        "model_log_bytes": 21000,
        "job_process_count": 3,
    }
    result.update(updates)
    return result


def test_baseline_and_runtime_safety_contract():
    assert supervisor.FROZEN_FLAGS == tuple(wave.FLAGS)
    assert supervisor.validate_sample(runtime(), startup_grace_seconds=900) == []
    assert supervisor.RUNTIME_PHYSICAL_FLOOR_BYTES == 7 * supervisor.GIB
    assert supervisor.RUNTIME_COMMIT_FLOOR_BYTES == 8 * supervisor.GIB


@pytest.mark.parametrize(
    "update,expected",
    [
        ({"physical_available_bytes": 7 * 2**30 - 1}, "RAM_RUNTIME_FLOOR"),
        ({"commit_headroom_bytes": 8 * 2**30 - 1}, "COMMIT_RUNTIME_FLOOR"),
        ({"disk_free_bytes": 8 * 2**30 - 1}, "DISK_RUNTIME_FLOOR"),
        ({"root_owner_valid": False}, "ROOT_PID_REUSED_OR_FOREIGN"),
        ({"job_processes_valid": False}, "JOB_PROCESS_IDENTITY"),
        ({"foreign_ports": [{"port": 8890, "pid": 40080}]}, "FOREIGN_LISTENER"),
        (
            {
                "gpu": {
                    "uuid": "GPU-12345678",
                    "total_mib": 65536,
                    "used_mib": 65000,
                    "free_mib": 300,
                    "reserved_mib": 236,
                    "utilization_pct": 98,
                }
            },
            "GPU_RUNTIME_FLOOR",
        ),
        ({"job_process_count": 17}, "JOB_PROCESS_LIMIT"),
        ({"elapsed_seconds": 14400.01}, "WALL_TIME_LIMIT"),
        ({"model_log_bytes": 270 * 2**20}, "LOG_BYTE_LIMIT"),
        (
            {
                "native_residency": {
                    "layers_offloaded": 39,
                    "layers_total": 41,
                    "context": 16384,
                    "kv_total_mib": 320,
                    "cpu_expert_offload": 0,
                }
            },
            "NATIVE_GPU_FIT",
        ),
        (
            {
                "native_residency": {
                    "layers_offloaded": 41,
                    "layers_total": 41,
                    "context": 16384,
                    "kv_total_mib": 320,
                    "cpu_expert_offload": 3,
                }
            },
            "CPU_EXPERT_OFFLOAD",
        ),
        (
            {
                "native_residency": {
                    "layers_offloaded": 41,
                    "layers_total": 41,
                    "context": 32768,
                    "kv_total_mib": 320,
                    "cpu_expert_offload": 0,
                }
            },
            "NATIVE_CONTEXT_CHANGED",
        ),
        (
            {
                "native_residency": {
                    "layers_offloaded": 41,
                    "layers_total": 41,
                    "context": 16384,
                    "kv_total_mib": 9000,
                    "cpu_expert_offload": 0,
                }
            },
            "KV_ALLOCATION_UNBOUNDED",
        ),
    ],
)
def test_fail_closed_pressure_and_owner_checks(update, expected):
    assert expected in supervisor.validate_sample(
        runtime(**update), startup_grace_seconds=900
    )


@pytest.mark.parametrize(
    "update",
    [
        {"physical_available_bytes": True},
        {"commit_headroom_bytes": float("nan")},
        {"root_owner_valid": 1},
        {"job_process_count": "3"},
        {"native_residency": None},
        {"gpu": None},
        {"foreign_ports": "none"},
    ],
)
def test_invalid_telemetry_never_admits(update):
    assert supervisor.validate_sample(runtime(**update), startup_grace_seconds=900)


def test_startup_pending_expires():
    pending = runtime(
        owned_listener=False,
        native_started=False,
        native_residency=None,
        startup_seconds=100,
    )
    assert supervisor.validate_sample(pending, startup_grace_seconds=900) == []
    pending["startup_seconds"] = 901
    failures = supervisor.validate_sample(pending, startup_grace_seconds=900)
    assert "STARTUP_GRACE_EXPIRED" in failures


def test_authorized_postfit_shutdown_is_not_mistaken_for_startup_after_900_seconds():
    observed = runtime(
        owned_listener=False, native_started=False,
        native_residency=None, startup_seconds=1000, elapsed_seconds=1000,
    )
    assert "STARTUP_GRACE_EXPIRED" in supervisor.validate_sample(
        observed, startup_grace_seconds=900
    )
    assert supervisor.validate_sample(
        observed, startup_grace_seconds=900, allow_witnessed_shutdown=True
    ) == []
    observed["physical_available_bytes"] = 1
    assert "RAM_RUNTIME_FLOOR" in supervisor.validate_sample(
        observed, startup_grace_seconds=900, allow_witnessed_shutdown=True
    )


def test_native_log_parser_rejects_offload_and_kv_mismatch():
    okay = (
        "llama_new_context_with_model: n_ctx = 16384\n"
        "llama_model_load_tensors: offloaded 41/41 layers to GPU\n"
        "llama_kv_cache_init: CUDA0 KV buffer size = 320.00 MiB\n"
        "llama_context: n_cpu_moe = 0\n"
    )
    proof = supervisor.parse_native_residency(okay)
    assert proof["context"] == 16384
    assert proof["kv_total_mib"] == 320
    assert proof["cpu_expert_offload"] == 0
    assert (
        supervisor.parse_native_residency(
            "n_ctx = 16384\noffloaded 40/41 layers to GPU\nCUDA0 KV buffer size = 320 MiB"
        )["layers_offloaded"]
        == 40
    )
    assert supervisor.parse_native_residency("") is None
    assert supervisor.parse_native_residency("CUDA0 KV buffer size = NaN MiB") is None


def test_contract_rejects_forged_command_or_study(tmp_path):
    study = wave.study_for("reap128_v2_q4")
    command = wave.command(
        study,
        model=tmp_path / "m.gguf",
        output=tmp_path / "out",
        server=tmp_path / "llama-server.exe",
        port=8790,
    )
    supervisor.validate_frozen_command(
        command,
        study,
        tmp_path / "m.gguf",
        tmp_path / "llama-server.exe",
        tmp_path / "out",
        8790,
    )
    for bad in (
        command + ["--disable-safety"],
        command[:-1] + ["999"],
        ["cmd", "/c", "whoami"],
    ):
        with pytest.raises(ValueError):
            supervisor.validate_frozen_command(
                bad,
                study,
                tmp_path / "m.gguf",
                tmp_path / "llama-server.exe",
                tmp_path / "out",
                8790,
            )


def test_default_run_refuses_before_fs_or_launch(tmp_path, monkeypatch):
    def must_not_touch(*args, **kwargs):
        raise AssertionError("must not access native admission, files or launch")

    monkeypatch.setattr(supervisor, "_run_authorized", must_not_touch)
    with pytest.raises(
        supervisor.SupervisorRefused, match="EXPLICIT_EXECUTION_AUTHORIZATION"
    ):
        supervisor.run_frozen_arm(
            study=wave.study_for("reap128_v2_q4"),
            model=tmp_path / "model.gguf",
            server=tmp_path / "server.exe",
            output=tmp_path / "output",
            receipt_dir=tmp_path,
        )
    assert list(tmp_path.iterdir()) == []


def test_new_exclusive_receipt_cannot_be_replaced(tmp_path):
    receipt = tmp_path / "terminal.json"
    supervisor.write_receipt_exclusive(receipt, {"schema": "test", "rows": []})
    original = receipt.read_bytes()
    assert json.loads(original)["schema"] == "test"
    with pytest.raises(FileExistsError):
        supervisor.write_receipt_exclusive(receipt, {"schema": "forged"})
    assert receipt.read_bytes() == original


def test_cancel_never_uses_foreign_pid_or_taskkill():
    calls = []
    owner = SimpleNamespace(
        active=True, terminate_owned=lambda: calls.append("job-only")
    )
    supervisor.cancel_owned_job(owner)
    assert calls == ["job-only"]
    with pytest.raises(supervisor.SupervisorRefused):
        supervisor.cancel_owned_job(SimpleNamespace(active=False))


@pytest.mark.parametrize("breach", [False, True])
@pytest.mark.parametrize("integrity_fault", [None, "source_receipt", "start_receipt", "job_close", "lineage", "lingering_child", "missed_transient"])
def test_full_owned_lifecycle_with_injected_native_job_only(
    tmp_path, monkeypatch, breach, integrity_fault
):
    """No real subprocess or model: mock owner verifies the critical control path."""
    model = tmp_path / "q4.gguf"
    server = tmp_path / "llama-server.exe"
    model.write_bytes(b"dummy")
    server.write_bytes(b"dummy")
    events = []

    class FakeJob:
        def __init__(self):
            self.pid = None
            self.birth = None
            self.active = False
            self.count = 0
            self.member_births = {}

        def start(self, command, cwd):
            assert command[1].endswith("reap_serve.py")
            self.active = True
            self.pid = 1234
            self.birth = 1234567890.5
            events.append("start")

        def exit_code(self):
            self.count += 1
            return 0 if self.count > 2 else None

        def member_ids(self):
            return [self.pid, 9001] if integrity_fault == "lingering_child" else []

        def lifetime_process_counts(self):
            return {"total_processes": 2 if integrity_fault in ("lingering_child", "missed_transient") else 1,
                    "active_processes": 2 if integrity_fault == "lingering_child" else 0,
                    "total_page_fault_count": 0}

        def verify_member_births(self, ids):
            return all(pid == self.pid for pid in ids)

        def terminate_owned(self):
            assert self.active
            events.append("terminate-owned-job")

        def close(self):
            self.active = False
            events.append("close")
            if integrity_fault == "source_receipt":
                receipt = tmp_path / "reap128_v2_q4.original-offline-source.json"
                receipt.write_bytes(receipt.read_bytes() + b"TAMPER")
            if integrity_fault == "start_receipt":
                receipt = tmp_path / "reap128_v2_q4.owner-start.json"
                receipt.write_bytes(receipt.read_bytes() + b" ")
            if integrity_fault == "job_close":
                raise OSError("synthetic private Job close failure")

    original = wave.study_for("reap128_v2_q4")
    observed = {
        "model_full_sha256": original.model_sha256,
        "server_full_sha256": "a" * 64,
        "gpu": {"uuid": "GPU-12345678"},
    }
    monkeypatch.setattr(supervisor.native, "snapshot", lambda **kwargs: observed)
    monkeypatch.setattr(
        supervisor.native,
        "assess",
        lambda *args, **kwargs: {
            "status": "READY_FOR_CAPACITY_PROBE_ONLY",
            "reasons": [],
        },
    )
    monkeypatch.setattr(supervisor.native, "read_windows_memory", lambda: {})
    monkeypatch.setattr(supervisor.native, "read_gpu", lambda: observed["gpu"])
    monkeypatch.setattr(
        supervisor.native, "read_active_processes_and_ports", lambda _: {}
    )
    def fake_original_source(*, receipt_path):
        # A mock explicitly creates source bytes: the production supervisor
        # must still independently rehash them after the owned Job closes.
        raw = b"synthetic test fixture; never a real original source certificate\n"
        receipt_path.write_bytes(raw)
        return {"receipt": str(receipt_path), "sha256": hashlib.sha256(raw).hexdigest(),
                "source_offline_verified": True}

    monkeypatch.setattr(supervisor, "verify_frozen_sources_offline", fake_original_source)
    monkeypatch.setattr(supervisor.time, "sleep", lambda _: None)

    def fake_sample(*args, **kwargs):
        data = runtime(elapsed_seconds=1, startup_seconds=1)
        data["gpu_source_identity_matches_preflight"] = True
        if breach:
            data["physical_available_bytes"] = 6 * supervisor.GIB
        return data

    monkeypatch.setattr(supervisor, "_watch_sample", fake_sample)
    from chowder_batch import reap_v2_supervisor_integration_guard_20261027 as guard
    from chowder_batch import reap_v2_live_job_lineage_guard_20261028 as lineage
    def fake_lineage(owner, *, ledger, **kwargs):
        if integrity_fault == "lineage":
            return {"status": "REFUSED", "reasons": ["SYNTHETIC_FOREIGN_OWNED_CHILD"], "roles": {}}
        ledger.full_active_witnessed = True
        ledger.roles["root"] = (owner.pid, 1234567890.5)
        ledger.pid_births[owner.pid] = 1234567890.5
        return {"status": "OWNER_BOUND_ACTIVE_OBSERVED_ONLY", "reasons": [],
                "roles": {"root": owner.pid, "server": 9001, "evaluator": 9002}}
    monkeypatch.setattr(lineage, "capture_live_job_lineage", fake_lineage)
    # The synthetic FakeJob has no native Windows handles. This boundary is
    # mocked solely for lifecycle tests; separate benign Job tests prove the
    # real from_owned_job native PID+handle witness.
    monkeypatch.setattr(lineage.LineageLedger, "from_owned_job", classmethod(
        lambda cls, owner: cls(root_pid=owner.pid, root_birth=1234567890.5)
    ))
    # Synthetic transport fixtures never constitute native fit proof or a
    # genuine Windows process census. Mock these boundaries explicitly here;
    # separate tests exercise the authentic source, timeout and PID witnesses.
    monkeypatch.setattr(supervisor, "verify_owned_original_native_fit", lambda *a, **k: {
        "status": "NATIVE_SOURCE_FIT_VERIFIED_RUNTIME_ONLY", "reasons": [],
        "source_sha256": "f" * 64,
    })
    monkeypatch.setattr(guard, "observe_descendant_extinction", lambda births: {
        "status": "OWNED_DESCENDANTS_EXTINCT_OBSERVED_ONLY", "reasons": [],
        "checked_child_pid_count": len(births),
    })
    monkeypatch.setattr(supervisor.psutil, "Process", lambda pid: SimpleNamespace(
        create_time=lambda: 1234567890.5,
    ))
    out = supervisor._run_authorized(
        study=original,
        model=model,
        server=server,
        output=tmp_path / "output",
        receipt_dir=tmp_path,
        port=8790,
        poll_seconds=1,
        max_runtime_seconds=30,
        startup_grace_seconds=30,
        owner_factory=FakeJob,
    )
    expected_status = (
        "REFUSED_RUNTIME_ABORTED" if breach or integrity_fault
        else "CHILD_EXIT_ZERO_RUNTIME_WITNESSED_ONLY"
    )
    assert out["status"] == expected_status
    assert out["gen0"]["status"] == "BLOCKED"
    assert events == ["start", "terminate-owned-job", "close", *(["close"] if integrity_fault == "job_close" else [])]
    assert (tmp_path / "reap128_v2_q4.owner-start.json").is_file()
    terminal = json.loads((tmp_path / "reap128_v2_q4.owner-terminal.json").read_text())
    assert terminal["status"] == expected_status
    assert terminal["original_offline_source_proof"]["source_offline_verified"] is True
    if breach:
        assert "RAM_RUNTIME_FLOOR" in out["breaches"]
    elif integrity_fault != "lineage":
        assert out["fully_witnessed_native_fit"]
        assert out["original_live_job_root_server_evaluator_lineage_witnessed"]
    if integrity_fault == "lineage":
        assert "SYNTHETIC_FOREIGN_OWNED_CHILD" in out["breaches"]
        assert out["original_live_job_root_server_evaluator_lineage_witnessed"] is False
    if integrity_fault == "lingering_child":
        if not breach:
            assert "CHILD_ZERO_WITH_RUNNING_OR_UNVERIFIED_OWNED_DESCENDANTS" in out["breaches"]
            assert out["pre_cleanup_live_owned_descendant_count"] == 1
            assert out["pre_cleanup_natural_owned_tree_extinction_witnessed"] is False
    if integrity_fault == "missed_transient" and not breach:
        assert "OWNED_JOB_UNOBSERVED_TRANSIENT_CHILDREN_OR_CENSUS_DRIFT" in out["breaches"]
        assert out["pre_cleanup_natural_owned_tree_extinction_witnessed"] is True
        assert out["all_private_job_lifetime_processes_source_witnessed"] is False
    elif not breach and integrity_fault is None:
        assert out["pre_cleanup_natural_owned_tree_extinction_witnessed"] is True
    if integrity_fault == "source_receipt":
        assert "ORIGINAL_OFFLINE_SOURCE_RECEIPT_CHANGED_AFTER_RUN" in out["breaches"]
    if integrity_fault == "start_receipt":
        assert "OWNER_START_RECEIPT_CHANGED" in out["original_owner_receipt_consistency"]["reasons"]
    if integrity_fault == "job_close":
        assert "PRIVATE_JOB_HANDLE_CLOSE_UNVERIFIED" in out["breaches"]
        assert out["owned_job_handle_closed_with_kill_on_close"] is False


def test_other_background_process_never_becomes_job_owner():
    assert supervisor.validate_sample(
        runtime(root_owner_valid=False, foreign_ports=[{"port": 8890, "pid": 55032}])
    ) == ["FOREIGN_LISTENER", "ROOT_PID_REUSED_OR_FOREIGN"]


def test_native_job_member_census_transient_helper_exit_retries_without_false_pass(
    tmp_path, monkeypatch,
):
    # Real Windows psutil may lose a short-lived original nvidia-smi or
    # PowerShell helper between Job ID enumeration and PID create-time read.
    # The first torn census is not certified; the subsequent coherent one is.
    class Job:
        pid = 3212
        scans = 0

        def member_ids(self):
            self.scans += 1
            return [self.pid, 4321] if self.scans == 1 else [self.pid]

        def verify_member_births(self, members):
            return 4321 not in members

        def owner_valid(self):
            return True

        def is_member(self, pid):
            return pid == self.pid

    monkeypatch.setattr(supervisor.native, "read_windows_memory",
                        lambda: {"physical_available_bytes": 10 * supervisor.GIB,
                                 "commit_headroom_bytes": 12 * supervisor.GIB})
    monkeypatch.setattr(supervisor.native, "read_gpu", lambda: runtime()["gpu"])
    monkeypatch.setattr(supervisor.native, "read_active_processes_and_ports",
                        lambda port: {"listeners": [], "active_conflicts": []})
    monkeypatch.setattr(supervisor.time, "sleep", lambda _: None)
    job = Job()
    sample = supervisor._watch_sample(
        job, output=tmp_path / "fresh", arm="reap128_v2_q4",
        port=8790, gpu_uuid=runtime()["gpu"]["uuid"], started=0,
    )
    assert job.scans >= 3
    assert sample["job_processes_valid"] is True
    assert sample["job_process_count"] == 1
    # Repeated PID identity contradictions cannot be blessed by retries.
    monkeypatch.setattr(Job, "verify_member_births", lambda self, ids: False)
    second = supervisor._watch_sample(
        Job(), output=tmp_path / "fresh", arm="reap128_v2_q4",
        port=8790, gpu_uuid=runtime()["gpu"]["uuid"], started=0,
    )
    assert second["job_processes_valid"] is False


@pytest.mark.parametrize("tamper", [None, "gsm_hash", "network", "exit"])
def test_offline_source_requires_exact_pinned_cohorts_no_network(
    tmp_path, monkeypatch, tamper
):
    from chowder_batch import reap_v2_original_offline_source_lock_20261027 as lock

    report = {
        "schema": lock.SCHEMA,
        "status": "COMPLETE_ORIGINAL_FROZEN_OFFLINE_SOURCES_MATCHED",
        "evaluator_sha256": lock.EVAL_SHA256,
        "network_or_model_called": False,
        "official_quality_scored": False,
        "gen0": supervisor.native.GEN0_BLOCKED,
        "cohorts": {
            suite: {
                "n": expected[0],
                "full_rendered_source_sha256": expected[1],
                "full_population_and_order_verified": True,
            }
            for suite, expected in (("gsm8k", lock.GSM), ("humaneval", lock.HUMAN))
        },
    }
    if tamper == "gsm_hash":
        report["cohorts"]["gsm8k"]["full_rendered_source_sha256"] = "0" * 64
    if tamper == "network":
        report["network_or_model_called"] = True

    def fake_source_child(command, **kwargs):
        assert command[-2] == "--out"
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        assert kwargs["env"]["HF_DATASETS_OFFLINE"] == "1"
        assert kwargs["env"]["REAP_ORIGINAL_SOURCE_OFFLINE_CHILD"] == "1"
        assert kwargs["timeout"] <= 120
        Path(command[-1]).write_text(json.dumps(report), encoding="utf-8")
        return SimpleNamespace(returncode=2 if tamper == "exit" else 0)

    monkeypatch.setattr(supervisor.subprocess, "run", fake_source_child)
    target = tmp_path / "original-offline.json"
    if tamper is None:
        evidence = supervisor.verify_frozen_sources_offline(receipt_path=target)
        assert evidence["source_offline_verified"] is True
        assert evidence["cohorts"]["humaneval"]["n"] == 164
        with pytest.raises(supervisor.SupervisorRefused):
            supervisor.verify_frozen_sources_offline(receipt_path=target)
    else:
        with pytest.raises(supervisor.SupervisorRefused):
            supervisor.verify_frozen_sources_offline(receipt_path=target)
