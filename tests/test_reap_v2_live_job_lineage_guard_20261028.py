"""CPU-only adversarial lineage validation; optional benign Windows Job probe."""

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import sys
import time
import psutil

import pytest

from chowder_batch import reap_v2_frozen_wave_20261027 as wave
from chowder_batch import reap_v2_live_job_lineage_guard_20261028 as guard
from chowder_batch import reap_v2_owner_bound_runtime_supervisor_20261027 as owner_mod


STUDY = wave.study_for("reap128_v2_q4")
MODEL = Path("G:/reap128-v2-q4_k_m-20261025.gguf")
SERVER = Path("F:/llama.cpp/bin/llama-server.exe")
OUTPUT = Path("C:/dedicated-reap/probe")
PORT = 8790


def fake_snapshot(roles=("root", "server", "evaluator"), **overrides):
    command = wave.command(STUDY, model=MODEL, server=SERVER, output=OUTPUT, port=PORT)
    expected_server = [str(SERVER), *wave.FLAGS, "--port", str(PORT), "-m", str(MODEL)]
    expected_eval = [
        sys.executable,
        str(wave.REPO / "chowder_batch" / "reap_eval.py"),
        *command[command.index("--") + 1 :],
    ]
    members = {
        4001: {
            "pid": 4001,
            "birth": 1793000000.0,
            "parent": 321,
            "argv": command,
            "name": "python.exe",
            "image_path": sys.executable,
            "job_member": True,
        },
        4002: {
            "pid": 4002,
            "birth": 1793000001.0,
            "parent": 4001,
            "argv": expected_server,
            "name": "llama-server.exe",
            "image_path": str(SERVER),
            "job_member": True,
        },
        4003: {
            "pid": 4003,
            "birth": 1793000002.0,
            "parent": 4001,
            "argv": expected_eval,
            "name": "python.exe",
            "image_path": sys.executable,
            "job_member": True,
        },
    }
    ids = {"root": 4001, "server": 4002, "evaluator": 4003}
    active = {ids[r]: members[ids[r]] for r in roles}
    witness = {
        "active": True,
        "job_present": True,
        "job_limit_kill_on_close": True,
        "job_limit_exact_no_breakaway": True,
        "root_pid": 4001,
        "root_birth": 1793000000.0,
        "job_active_limit": 16,
        "owner_handle_matches_root": True,
        "members": active,
        "owned_listener": {"pid": 4002, "port": PORT} if "server" in roles else None,
        "other_listeners": [],
        "unexpected_job_listeners": [],
        "census_stable": True,
        "job_member_census_complete": True,
        "port_census_complete": True,
    }
    return {**witness, **overrides}


def eval_once(snapshot, *, phase="active", ledger=None):
    return guard.assess_lineage_snapshot(
        snapshot,
        study=STUDY,
        model=MODEL,
        server=SERVER,
        output=OUTPUT,
        port=PORT,
        phase=phase,
        ledger=ledger,
    )


def test_red_without_module_import_then_registered_real_commands():
    verdict = eval_once(fake_snapshot())
    assert verdict["status"] == "OWNER_BOUND_ACTIVE_OBSERVED_ONLY", verdict
    assert verdict["reasons"] == []
    assert verdict["roles"] == {"root": 4001, "server": 4002, "evaluator": 4003}
    assert verdict["gen0"]["status"] == "BLOCKED"


@pytest.mark.parametrize(
    "roles,phase,expected",
    [
        (("root",), "startup", "PENDING_STARTUP"),
        (("root", "server"), "startup", "PENDING_STARTUP"),
        (
            ("root", "server", "evaluator"),
            "startup",
            "OWNER_BOUND_ACTIVE_OBSERVED_ONLY",
        ),
        (("root", "server"), "active", "REFUSED"),
        (("root", "evaluator"), "active", "REFUSED"),
        (("root",), "active", "REFUSED"),
        ((), "shutdown", "REFUSED"),
    ],
)
def test_phases_require_full_active_graph(roles, phase, expected):
    assert eval_once(fake_snapshot(roles), phase=phase)["status"] == expected


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("job_limit_kill_on_close", False, "JOB_KILL_ON_CLOSE_UNPROVEN"),
        ("job_limit_exact_no_breakaway", False, "JOB_BREAKAWAY_OR_UNREVIEWED_LIMIT"),
        ("job_active_limit", 100, "JOB_PROCESS_LIMIT_INVALID"),
        ("owner_handle_matches_root", False, "OWNER_HANDLE_ROOT_MISMATCH"),
        ("census_stable", False, "JOB_CENSUS_RACED"),
        ("job_member_census_complete", False, "JOB_CENSUS_UNAVAILABLE"),
        ("port_census_complete", False, "PORT_CENSUS_UNAVAILABLE"),
        ("owned_listener", {"pid": 19, "port": 8790}, "OWNED_LISTENER_MISMATCH"),
        ("other_listeners", [{"pid": 700, "port": 8890}], "FOREIGN_LISTENER"),
        (
            "unexpected_job_listeners",
            [{"pid": 4003, "port": 12345}],
            "UNEXPECTED_OWNED_LISTENER",
        ),
    ],
)
def test_fail_closed_job_and_port_observation(field, value, reason):
    result = eval_once(fake_snapshot(**{field: value}))
    assert result["status"] == "REFUSED" and reason in result["reasons"]


@pytest.mark.parametrize(
    "pid,mutation,reason",
    [
        (4001, {"argv": ["cmd.exe", "/c", "whoami"]}, "FROZEN_ROOT_COMMAND_CHANGED"),
        (4002, {"argv": ["F:/foreign.exe"]}, "NATIVE_SERVER_COMMAND_MISMATCH"),
        (4003, {"argv": ["python", "unexpected.py"]}, "EVALUATOR_COMMAND_MISMATCH"),
        (4003, {"parent": 1}, "FOREIGN_PARENT_CHAIN"),
        (4002, {"birth": 1792999990.0}, "CHILD_CREATED_BEFORE_ROOT"),
        (4003, {"job_member": False}, "MEMBER_NOT_IN_PRIVATE_JOB"),
        (4002, {"image_path": "F:/malicious/replaced-server.exe"}, "FROZEN_PROCESS_IMAGE_PATH_CHANGED"),
        (4003, {"image_path": "C:/foreign/python.exe"}, "FROZEN_PROCESS_IMAGE_PATH_CHANGED"),
    ],
)
def test_foreign_or_reused_pid_never_admitted(pid, mutation, reason):
    sample = fake_snapshot()
    sample["members"][pid] = {**sample["members"][pid], **mutation}
    result = eval_once(sample)
    assert result["status"] == "REFUSED" and reason in result["reasons"]


def test_extraneous_job_process_even_if_named_python_is_refused():
    sample = fake_snapshot()
    sample["members"][4004] = {
        "pid": 4004,
        "birth": 1793000004.0,
        "parent": 4001,
        "name": "python.exe",
        "argv": [sys.executable, "-c", "arbitrary"],
        "image_path": sys.executable,
        "job_member": True,
    }
    assert "UNEXPECTED_JOB_CHILD" in eval_once(sample)["reasons"]


def test_real_windows_system32_conhost_child_is_narrowly_allowed(monkeypatch):
    monkeypatch.setenv("SystemRoot", "C:/Windows")
    sample = fake_snapshot()
    sample["members"][4004] = {
        "pid": 4004, "birth": 1793000003.0, "parent": 4001,
        "name": "conhost.exe", "argv": [r"\\??\\C:\\WINDOWS\\system32\\conhost.exe", "0x4"],
        "image_path": "C:/Windows/System32/conhost.exe", "job_member": True,
    }
    assert eval_once(sample)["status"] == "OWNER_BOUND_ACTIVE_OBSERVED_ONLY"
    for forged in (
        {"image_path": "C:/malware/conhost.exe"},
        {"argv": ["conhost.exe", "0x4", "--attack"]},
        {"parent": 19},
        {"job_member": False},
    ):
        bad = deepcopy(sample)
        bad["members"][4004].update(forged)
        assert "UNEXPECTED_JOB_CHILD" in eval_once(bad)["reasons"]


def test_real_os_kernel_directory_not_spoofable_by_inherited_systemroot(monkeypatch):
    if sys.platform != "win32":
        pytest.skip("native Windows system directory is an OS-provided identity")
    native = guard._native_system_directory()
    assert native is not None and (native / "conhost.exe").is_file()
    monkeypatch.setenv("SystemRoot", "C:/user-controlled-systemroot")
    assert guard._native_system_directory() == native
    observed = {
        "pid": 4004, "parent": 4001, "job_member": True,
        "image_path": str(native / "conhost.exe"),
        "argv": [str(native / "conhost.exe"), "0x4"],
    }
    assert guard._frozen_helper(observed, 4001, PORT) == "native_windows_console_host"
    observed["image_path"] = "C:/user-controlled-systemroot/conhost.exe"
    assert guard._frozen_helper(observed, 4001, PORT) is None


@pytest.mark.parametrize("executable,args,expected", [
    ("nvidia-smi.exe", ["--query-gpu=memory.used", "--format=csv,noheader,nounits"],
     "gpu_memory_readonly"),
    ("powershell.exe", ["-NoProfile", "-NonInteractive", "-Command",
                        guard._frozen_listener_probe_script(PORT)],
     "frozen_listener_owner_readonly"),
])
def test_read_only_frozen_external_helper_image_must_match_kernel_directory(
    executable, args, expected,
):
    if sys.platform != "win32":
        pytest.skip("native Windows kernel32 system directory required")
    directory = guard._native_system_directory()
    rel = ("WindowsPowerShell/v1.0/powershell.exe" if executable == "powershell.exe"
           else executable)
    image = directory / rel
    assert image.is_file()
    record = {"pid": 4020, "parent": 4001, "job_member": True,
              "image_path": str(image), "argv": [str(image), *args]}
    assert guard._frozen_helper(record, 4001, PORT) == expected
    record["image_path"] = str(directory / "wrong" / executable)
    assert guard._frozen_helper(record, 4001, PORT) is None


def test_ledger_prevents_role_replacement_and_missing_terminal_history():
    ledger = guard.LineageLedger(root_pid=4001, root_birth=1793000000.0)
    assert (
        eval_once(fake_snapshot(("root",)), phase="startup", ledger=ledger)["status"]
        == "PENDING_STARTUP"
    )
    assert (
        eval_once(fake_snapshot(), phase="active", ledger=ledger)["status"]
        == "OWNER_BOUND_ACTIVE_OBSERVED_ONLY"
    )
    assert ledger.full_active_witnessed is True
    replace = fake_snapshot()
    replace["members"][4003]["birth"] += 10
    assert "PID_BIRTH_REUSED" in eval_once(replace, ledger=ledger)["reasons"]
    # Confirm transitional teardown cannot masquerade as active or conclusive extinction.
    assert (
        eval_once(fake_snapshot(("root", "server")), phase="shutdown", ledger=ledger)[
            "status"
        ]
        == "PENDING_SHUTDOWN"
    )
    assert (
        eval_once(fake_snapshot(()), phase="shutdown", ledger=ledger)["status"]
        == "JOB_MEMBERS_EMPTY_NEEDS_POSTCLOSE_WITNESS"
    )


def test_zero_member_startup_refuses_even_if_no_errors():
    assert eval_once(fake_snapshot(()), phase="startup")["status"] == "REFUSED"


def test_auto_phase_startup_full_active_and_graceful_job_teardown():
    ledger = guard.LineageLedger(root_pid=4001, root_birth=1793000000.0)
    assert (
        eval_once(fake_snapshot(("root",)), phase="auto", ledger=ledger)["status"]
        == "PENDING_STARTUP"
    )
    assert (
        eval_once(fake_snapshot(("root", "server")), phase="auto", ledger=ledger)[
            "status"
        ]
        == "PENDING_STARTUP"
    )
    assert (
        eval_once(fake_snapshot(), phase="auto", ledger=ledger)["status"]
        == "OWNER_BOUND_ACTIVE_OBSERVED_ONLY"
    )
    assert ledger.full_active_witnessed
    assert (
        eval_once(fake_snapshot(("root", "server")), phase="auto", ledger=ledger)[
            "status"
        ]
        == "PENDING_SHUTDOWN"
    )
    assert (
        eval_once(fake_snapshot(()), phase="auto", ledger=ledger)["status"]
        == "JOB_MEMBERS_EMPTY_NEEDS_POSTCLOSE_WITNESS"
    )
    assert eval_once(fake_snapshot(), phase="auto")["reasons"] == [
        "AUTO_PHASE_REQUIRES_OWNED_LINEAGE_LEDGER"
    ]


def test_public_live_census_rejects_foreign_owner_object_without_inspection():
    with pytest.raises(guard.LineageRefused, match="OWNED_JOB_HANDLE_REQUIRED"):
        guard.capture_live_job_lineage(
            object(), study=STUDY, model=MODEL, server=SERVER, output=OUTPUT, port=PORT
        )


def test_torn_startup_or_shutdown_census_cannot_be_false_verified():
    for phase in ("startup", "active", "shutdown"):
        result = eval_once(fake_snapshot(census_stable=False), phase=phase)
        assert result["status"] == "REFUSED"
        assert "JOB_CENSUS_RACED" in result["reasons"]


def test_ledger_rejects_reused_role_pid_after_new_process_appears():
    ledger = guard.LineageLedger(root_pid=4001, root_birth=1793000000.0)
    assert (
        eval_once(fake_snapshot(), ledger=ledger)["status"]
        == "OWNER_BOUND_ACTIVE_OBSERVED_ONLY"
    )
    forged = fake_snapshot()
    forged["members"][4013] = {
        **forged["members"].pop(4003),
        "pid": 4013,
        "birth": 1793000099.0,
    }
    assert "ROLE_PID_REPLACED" in eval_once(forged, ledger=ledger)["reasons"]
    assert ledger.roles["evaluator"] == (4003, 1793000002.0)


def test_live_win32_private_job_benign_owned_subprocess_tree(tmp_path):
    """The ONLY live launch: short Python parent/child; no model or services."""
    if sys.platform != "win32":
        pytest.skip("private Windows Job Object required")
    owner = owner_mod._WindowsOwnedJob()
    code = (
        "import subprocess,sys,time;"
        "subprocess.Popen([sys.executable,'-c','import time;time.sleep(12)']);"
        "time.sleep(12)"
    )
    birth_records = {}
    try:
        owner.start([sys.executable, "-c", code], tmp_path)
        ledger = guard.LineageLedger.from_owned_job(owner)
        assert ledger.root_pid == owner.pid
        assert ledger.root_birth == psutil.Process(owner.pid).create_time()
        started = time.monotonic()
        members = owner.member_ids()
        while len(members) != 2 and time.monotonic() - started < 3:
            time.sleep(0.05)
            members = owner.member_ids()
        assert len(members) == 2, members
        flags, limit = guard._live_job_flags(owner)
        assert flags & guard.WINDOWS_KILL_ON_JOB_CLOSE
        assert flags & guard.WINDOWS_ACTIVE_PROCESS_LIMIT
        assert 1 <= limit <= owner_mod.MAX_JOB_PROCESSES
        for pid in members:
            assert owner.is_member(pid)
            birth_records[pid] = psutil.Process(pid).create_time()
        child_id = next(p for p in members if p != owner.pid)
        assert psutil.Process(child_id).ppid() == owner.pid
        assert birth_records[child_id] >= birth_records[owner.pid]
        result = guard.capture_live_job_lineage(
            owner,
            study=STUDY,
            model=MODEL,
            server=SERVER,
            output=OUTPUT,
            port=PORT,
            phase="startup",
            ledger=ledger,
        )
        # Real Job proof is read-only, but this deliberately non-model
        # Python command MUST NOT be mistaken for authorized REAP.
        assert result["status"] == "REFUSED", result
        assert "FROZEN_ROOT_COMMAND_CHANGED" in result["reasons"], result
    finally:
        if owner.active:
            owner.terminate_owned()  # owned disposable job only, never an external PID
        owner.close()
    for _ in range(30):
        alive = [
            pid
            for pid, birth in birth_records.items()
            if psutil.pid_exists(pid)
            and abs(psutil.Process(pid).create_time() - birth) < 0.002
        ]
        if not alive:
            break
        time.sleep(0.1)
    assert not alive, alive
