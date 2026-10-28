"""Read-only native Windows Job lineage witness for one frozen REAP v2 arm.

Integration: call capture_live_job_lineage(owner, study, model, server, output,
port, phase, ledger) from the owner supervisor's sampling loop, AFTER owner
start() and BEFORE any successful runtime-fit assertion. Keep the LineageLedger
in the owner scope; never adopt a previous process, JSON receipt or foreign
Job. A PENDING result is never success. A REFUSED result may only cancel the
private Job via the supervisor's own cancel_owned_job(owner).

The Windows census uses ONLY QueryInformationJobObject on the private Job,
owner root's existing process handle, psutil read-only PID observations and
net_connections. It has no launch, terminate, or external service action.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass, field
import math
from pathlib import Path
import re
import sys

import psutil

from chowder_batch import reap_v2_frozen_wave_20261027 as wave
from chowder_batch import reap_v2_owner_bound_runtime_supervisor_20261027 as supervisor
from chowder_batch import reap_v2_supervisor_integration_guard_20261027 as integration
from chowder_batch import reap_v2_dedicated_runner_admission_20261027 as native


SCHEMA = "chowder.reap-v2-live-owned-windows-job-lineage.v1"
GEN0 = {"status": "BLOCKED", "chosen": None, "release_approved": False}
STATES = frozenset(("auto", "startup", "active", "shutdown"))
WINDOWS_KILL_ON_JOB_CLOSE = 0x2000
WINDOWS_ACTIVE_PROCESS_LIMIT = 0x0008


class LineageRefused(ValueError):
    """Private Job/process source cannot be trusted or inspected."""


@dataclass
class LineageLedger:
    """In-memory role identity history tied to one owned root's creation time.

    The ledger is not a replacement for the Windows Job handle, port census
    or eventual post-close independent PID extinction proof.
    """

    root_pid: int
    root_birth: float
    roles: dict[str, tuple[int, float]] = field(default_factory=dict)
    pid_births: dict[int, float] = field(default_factory=dict)
    full_active_witnessed: bool = False

    @classmethod
    def from_owned_job(cls, owner: supervisor._WindowsOwnedJob) -> LineageLedger:
        """Bind PID identity using psutil birth precision, checked vs Win32 handle.

        Use this instead of writing LineageLedger(owner.pid, owner.birth): the
        Win32 GetProcessTimes and psutil creation timestamps can differ by
        multiple milliseconds despite referring to the same live process.
        """
        if (
            sys.platform != "win32"
            or type(owner) is not supervisor._WindowsOwnedJob
            or owner.active is not True
            or not _pid(owner.pid)
            or not _birth(owner.birth)
        ):
            raise LineageRefused("OWNED_JOB_HANDLE_REQUIRED")
        try:
            birth = psutil.Process(owner.pid).create_time()
            actual_handle_birth = owner._handle_create_time(owner.process)
        except (psutil.Error, OSError, supervisor.SupervisorRefused) as exc:
            raise LineageRefused("OWNER_BIRTH_CANNOT_BE_WITNESSED") from exc
        if (
            not _birth(birth)
            or abs(birth - actual_handle_birth) > 2
            or abs(actual_handle_birth - owner.birth) > 0.002
        ):
            raise LineageRefused("OWNER_CREATE_TIME_NOT_HANDLE_BOUND")
        return cls(root_pid=owner.pid, root_birth=birth)


def _result(
    status: str,
    reasons: list[str],
    *,
    roles: dict | None = None,
    snapshot: dict | None = None,
) -> dict:
    return {
        "schema": SCHEMA,
        "status": "REFUSED" if reasons else status,
        "reasons": sorted(set(reasons)),
        "roles": roles or {},
        "snapshot": snapshot,
        "read_only": True,
        "execution_approved": False,
        "scientific_gate": "UNKNOWN",
        "gen0": dict(GEN0),
    }


def _pid(pid: object) -> bool:
    return type(pid) is int and pid > 0


def _birth(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _exe(argv: object) -> str:
    if not isinstance(argv, list) or not argv or not isinstance(argv[0], str):
        return ""
    return Path(argv[0]).name.casefold()


def _role(record: dict, root_pid: int) -> str | None:
    if record.get("pid") == root_pid:
        return "root"
    name = str(record.get("name") or "").casefold()
    exe = _exe(record.get("argv"))
    if "llama-server" in name or exe == "llama-server.exe":
        return "server"
    if name.startswith("python") or exe.startswith("python"):
        return "evaluator"
    return None


def _frozen_listener_probe_script(port: int) -> str:
    """Exact unmodified reap_serve._require_owned_listener Windows argv."""
    return (
        "$ErrorActionPreference='Stop'; "
        f"Get-NetTCPConnection -State Listen -LocalPort {port} | "
        "Where-Object { $_.LocalAddress -eq '127.0.0.1' -or "
        "$_.LocalAddress -eq '0.0.0.0' } | "
        "Select-Object -ExpandProperty OwningProcess"
    )


def _native_system_directory() -> Path | None:
    """Use the Windows kernel's OS directory, never inherited SystemRoot/PATH.

    An attacker controlling the launched evaluator's inherited environment
    must not authorize an arbitrary conhost/nvidia-smi/powershell executable.
    No global security configuration or process operation is performed.
    """
    if sys.platform != "win32":
        return None
    buffer = ctypes.create_unicode_buffer(32768)
    length = ctypes.windll.kernel32.GetSystemDirectoryW(buffer, len(buffer))
    if not 0 < length < len(buffer):
        return None
    directory = Path(buffer.value)
    return directory if directory.is_absolute() and directory.is_dir() else None


def _native_helper_image(record: dict, relative: str) -> bool:
    system32 = _native_system_directory()
    image = record.get("image_path")
    if (system32 is None or not isinstance(image, str)
            or not Path(image).is_absolute()):
        return False
    expected = system32 / relative
    try:
        return (expected.is_file() and not expected.is_symlink()
                and Path(image).resolve(strict=True) == expected.resolve(strict=True))
    except (OSError, ValueError):
        return False


def _frozen_helper(record: dict, root_pid: int, port: int) -> str | None:
    """Allow only OS console host and frozen read-only wrapper helpers.

    These are transient read-only subprocesses actually spawned by the frozen
    reap_serve.py source. No generic PowerShell, cmd, shell or GPU operation is
    authorized. They must remain in our private Job under its root process.
    """
    if (record.get("parent") != root_pid
            or record.get("job_member") is not True):
        return None
    args = record.get("argv")
    if not isinstance(args, list):
        return None
    exe = _exe(args)
    if exe == "conhost.exe" and len(args) == 2 and re.fullmatch(
        r"0x[0-9a-fA-F]{1,16}", args[1]
    ):
        # Authentically observed on Windows: CREATE_NO_WINDOW suspended
        # Python root receives a conhost.exe child in the same private Job.
        # A general extra Python/cmd/PowerShell invocation is NOT allowed.
        # Bind the real executable to Windows System32, the exact 0xNN
        # console identifier argv and the created Job root as parent.
        if _native_helper_image(record, "conhost.exe"):
            return "native_windows_console_host"
    if exe in ("nvidia-smi", "nvidia-smi.exe") and args[1:] == [
        "--query-gpu=memory.used", "--format=csv,noheader,nounits",
    ] and _native_helper_image(record, "nvidia-smi.exe"):
        return "gpu_memory_readonly"
    if exe == "powershell.exe" and args[1:] == [
        "-NoProfile", "-NonInteractive", "-Command",
        _frozen_listener_probe_script(port),
    ] and _native_helper_image(record, "WindowsPowerShell/v1.0/powershell.exe"):
        return "frozen_listener_owner_readonly"
    return None


def assess_lineage_snapshot(
    snapshot: dict,
    *,
    study: wave.Study,
    model: Path,
    server: Path,
    output: Path,
    port: int,
    phase: str = "auto",
    ledger: LineageLedger | None = None,
) -> dict:
    """Pure phase-aware strict command/member/parent/birth/listener validator.

    On startup, only a root or root+server is PENDING; only the complete
    root+server+evaluator and owned server listener establishes active status.
    Shutdown requires prior complete ACTIVE observation and still only returns
    PENDING or NEEDS_POSTCLOSE_WITNESS. No empty Job may certify completion.
    """
    if phase not in STATES or not isinstance(snapshot, dict):
        return _result("REFUSED", ["INVALID_JOB_PHASE_OR_SNAPSHOT"])
    if phase == "auto":
        if not isinstance(ledger, LineageLedger):
            return _result("REFUSED", ["AUTO_PHASE_REQUIRES_OWNED_LINEAGE_LEDGER"])
        if not ledger.full_active_witnessed:
            phase = "startup"
        else:
            present = snapshot.get("members")
            if (isinstance(present, dict)
                    and sum(_role(record, snapshot.get("root_pid")) is not None
                            for record in present.values() if isinstance(record, dict)) == 3):
                phase = "active"
            else:
                # Legitimate evaluation shutdown may briefly leave root and
                # server. Never call that an active witness or completion.
                phase = "shutdown"
    try:
        study = integration.rebind_wave_study(study)
        expected_root = wave.command(
            study, model=model, server=server, output=output, port=port
        )
        expected_server = [
            str(server),
            *wave.FLAGS,
            "--port",
            str(port),
            "-m",
            str(model),
        ]
        expected_eval = [
            sys.executable,
            str(wave.REPO / "chowder_batch" / "reap_eval.py"),
            *expected_root[expected_root.index("--") + 1 :],
        ]
    except (ValueError, TypeError, AttributeError) as exc:
        return _result("REFUSED", ["REGISTERED_FROZEN_COMMAND_UNAVAILABLE"])
    root_pid = snapshot.get("root_pid")
    root_birth = snapshot.get("root_birth")
    failures: list[str] = []
    for field, reason in (
        ("active", "PRIVATE_JOB_INACTIVE"),
        ("job_present", "PRIVATE_JOB_HANDLE_MISSING"),
        ("job_limit_kill_on_close", "JOB_KILL_ON_CLOSE_UNPROVEN"),
        ("job_limit_exact_no_breakaway", "JOB_BREAKAWAY_OR_UNREVIEWED_LIMIT"),
        ("owner_handle_matches_root", "OWNER_HANDLE_ROOT_MISMATCH"),
        ("job_member_census_complete", "JOB_CENSUS_UNAVAILABLE"),
        ("port_census_complete", "PORT_CENSUS_UNAVAILABLE"),
        ("census_stable", "JOB_CENSUS_RACED"),
    ):
        if snapshot.get(field) is not True:
            failures.append(reason)
    limit = snapshot.get("job_active_limit")
    if type(limit) is not int or not 1 <= limit <= supervisor.MAX_JOB_PROCESSES:
        failures.append("JOB_PROCESS_LIMIT_INVALID")
    if not _pid(root_pid) or not _birth(root_birth):
        failures.append("ROOT_PID_BIRTH_INVALID")
    if ledger is not None:
        if (
            not isinstance(ledger, LineageLedger)
            or root_pid != ledger.root_pid
            or not _birth(root_birth)
            or abs(ledger.root_birth - root_birth) > 0.002
        ):
            failures.append("LEDGER_ROOT_IDENTITY_CHANGED")
    members = snapshot.get("members")
    if not isinstance(members, dict):
        failures.append("JOB_MEMBER_CENSUS_MISSING")
        members = {}
    if len(members) > supervisor.MAX_JOB_PROCESSES:
        failures.append("JOB_MEMBER_LIMIT")
    if not members and phase != "shutdown":
        failures.append("JOB_MEMBER_CENSUS_MISSING")
    roles: dict[str, int] = {}
    candidates: dict[str, dict] = {}
    helpers: dict[int, dict] = {}
    helper_roles: set[str] = set()
    for pid, record in members.items():
        if (
            not _pid(pid)
            or not isinstance(record, dict)
            or record.get("pid") != pid
            or not _birth(record.get("birth"))
        ):
            failures.append("JOB_MEMBER_PID_OR_BIRTH_INVALID")
            continue
        if record.get("job_member") is not True:
            failures.append("MEMBER_NOT_IN_PRIVATE_JOB")
        if not isinstance(record.get("image_path"), str) or not record["image_path"]:
            failures.append("ACTUAL_PROCESS_IMAGE_UNAVAILABLE")
        role = _role(record, root_pid)
        if role is None:
            helper = _frozen_helper(record, root_pid, port)
            if (helper is None or helper in helper_roles
                    or len(helpers) >= 3
                    or (_birth(root_birth) and record["birth"] < root_birth)):
                failures.append("UNEXPECTED_JOB_CHILD")
            else:
                if (not isinstance(record.get("image_path"), str)
                        or Path(record["image_path"]).name.casefold() != _exe(record.get("argv"))):
                    failures.append("FROZEN_HELPER_PROCESS_IMAGE_CHANGED")
                helper_roles.add(helper)
                helpers[pid] = record
                if ledger is not None:
                    previous = ledger.pid_births.get(pid)
                    if previous is not None and abs(previous - record["birth"]) > 0.002:
                        failures.append("PID_BIRTH_REUSED")
            continue
        if role in roles:
            failures.append("DUPLICATE_OWNED_ROLE")
            failures.append("UNEXPECTED_JOB_CHILD")
            continue
        roles[role] = pid
        candidates[role] = record
        expected = {
            "root": expected_root,
            "server": expected_server,
            "evaluator": expected_eval,
        }[role]
        if record.get("argv") != expected:
            failures.append(
                {
                    "root": "FROZEN_ROOT_COMMAND_CHANGED",
                    "server": "NATIVE_SERVER_COMMAND_MISMATCH",
                    "evaluator": "EVALUATOR_COMMAND_MISMATCH",
                }[role]
            )
        if (not isinstance(record.get("image_path"), str)
                or Path(record["image_path"]).resolve(strict=False)
                != Path(expected[0]).resolve(strict=False)):
            failures.append("FROZEN_PROCESS_IMAGE_PATH_CHANGED")
        if role == "root":
            if not _birth(root_birth) or abs(record["birth"] - root_birth) > 0.002:
                failures.append("ROOT_CREATE_TIME_MISMATCH")
        else:
            if _birth(root_birth) and record["birth"] < root_birth:
                failures.append("CHILD_CREATED_BEFORE_ROOT")
            if record.get("parent") != root_pid:
                failures.append("FOREIGN_PARENT_CHAIN")
        if ledger is not None:
            old = ledger.pid_births.get(pid)
            if old is not None and abs(old - record["birth"]) > 0.002:
                failures.append("PID_BIRTH_REUSED")
            old_role = ledger.roles.get(role)
            if old_role is not None and (
                old_role[0] != pid or abs(old_role[1] - record["birth"]) > 0.002
            ):
                failures.append("ROLE_PID_REPLACED")
    listeners = snapshot.get("other_listeners")
    if not isinstance(listeners, list) or listeners:
        failures.append("FOREIGN_LISTENER")
    unexpected = snapshot.get("unexpected_job_listeners")
    if not isinstance(unexpected, list) or unexpected:
        failures.append("UNEXPECTED_OWNED_LISTENER")
    listener = snapshot.get("owned_listener")
    if "server" in roles:
        if (phase == "active" or "evaluator" in roles) and (
            not isinstance(listener, dict)
            or listener.get("pid") != roles["server"]
            or listener.get("port") != port
        ):
            failures.append("OWNED_LISTENER_MISMATCH")
        elif listener is not None and (
            not isinstance(listener, dict)
            or listener.get("pid") != roles["server"]
            or listener.get("port") != port
        ):
            failures.append("OWNED_LISTENER_MISMATCH")
    elif listener is not None:
        failures.append("OWNED_LISTENER_WITHOUT_SERVER")
    if "evaluator" in roles and "server" not in roles and phase != "shutdown":
        failures.append("EVALUATOR_WITHOUT_SERVER")
    if "root" not in roles and phase != "shutdown":
        failures.append("ROOT_MISSING_FROM_OWNED_JOB")
    if phase == "shutdown":
        if ledger is None or not ledger.full_active_witnessed:
            failures.append("SHUTDOWN_WITHOUT_ACTIVE_WITNESS")
        if ledger is not None:
            for role in roles:
                if role not in ledger.roles:
                    failures.append("UNSEEN_ROLE_IN_SHUTDOWN")
    if failures:
        return _result("REFUSED", failures, roles=roles)

    status = "OWNER_BOUND_ACTIVE_OBSERVED_ONLY"
    if phase == "startup" and set(roles) != {"root", "server", "evaluator"}:
        status = "PENDING_STARTUP"
    elif phase == "active" and set(roles) != {"root", "server", "evaluator"}:
        return _result("REFUSED", ["INCOMPLETE_ACTIVE_JOB_LINEAGE"], roles=roles)
    elif phase == "shutdown":
        status = (
            "PENDING_SHUTDOWN"
            if members
            else "JOB_MEMBERS_EMPTY_NEEDS_POSTCLOSE_WITNESS"
        )

    if status == "OWNER_BOUND_ACTIVE_OBSERVED_ONLY":
        verified = integration.assess_bound_owner(
            {
                "active": snapshot["active"],
                "job_present": snapshot["job_present"],
                "job_limit_kill_on_close": snapshot["job_limit_kill_on_close"],
                "root_pid": root_pid,
                "root_birth": root_birth,
                "members": {pid: record for pid, record in members.items()
                            if pid not in helpers},
                "owned_listener": listener,
                "other_listeners": listeners,
            },
            study=study,
            model=model,
            server=server,
            output=output,
            port=port,
        )
        if verified["status"] != "OWNER_BOUND_OBSERVATION_ONLY":
            return _result("REFUSED", verified["reasons"], roles=roles)
    if ledger is not None:
        for role, pid in roles.items():
            born = candidates[role]["birth"]
            ledger.roles[role] = (pid, born)
            ledger.pid_births[pid] = born
        for pid, record in helpers.items():
            ledger.pid_births[pid] = record["birth"]
        if status == "OWNER_BOUND_ACTIVE_OBSERVED_ONLY":
            ledger.full_active_witnessed = True
    return _result(status, [], roles=roles)


def _live_job_flags(owner: supervisor._WindowsOwnedJob) -> tuple[int, int]:
    """Actual Win32 JobObjectExtendedLimitInformation read, no modification."""
    info = supervisor._JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    owner._call(
        "QueryInformationJobObject",
        owner.job,
        9,
        ctypes.byref(info),
        ctypes.sizeof(info),
        None,
        types=[
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.c_void_p,
        ],
    )
    basic = info.BasicLimitInformation
    return int(basic.LimitFlags), int(basic.ActiveProcessLimit)


def _live_process(pid: int, *, owner: supervisor._WindowsOwnedJob) -> dict:
    """Use read-only psutil and the *existing private Job* membership proof."""
    process = psutil.Process(pid)
    with process.oneshot():
        birth = process.create_time()
        parent = process.ppid()
        argv = process.cmdline()
        name = process.name()
        image = process.exe()
    return {
        "pid": pid,
        "birth": birth,
        "parent": parent,
        "argv": argv,
        "name": name,
        "image_path": image,
        "job_member": owner.is_member(pid),
    }


def capture_live_job_lineage(
    owner: supervisor._WindowsOwnedJob,
    *,
    study: wave.Study,
    model: Path,
    server: Path,
    output: Path,
    port: int,
    phase: str = "auto",
    ledger: LineageLedger | None = None,
) -> dict:
    """Native read-only Windows census, no PID signals, model or service start.

    The owner is the actual private handle returned from _WindowsOwnedJob.start,
    never an owner ID or claimed JSON. A torn census can never produce PASS.
    """
    if sys.platform != "win32" or type(owner) is not supervisor._WindowsOwnedJob:
        raise LineageRefused("OWNED_JOB_HANDLE_REQUIRED")
    if phase not in STATES or not owner.active or not owner.job or not owner.process:
        return _result("REFUSED", ["PRIVATE_JOB_NOT_ACTIVE"])
    if not _pid(owner.pid) or not _birth(owner.birth):
        return _result("REFUSED", ["PRIVATE_OWNER_PID_BIRTH_INVALID"])
    try:
        flags, limit = _live_job_flags(owner)
        members_before = owner.member_ids()
        if len(members_before) != len(set(members_before)):
            raise LineageRefused("DUPLICATE_JOB_MEMBER_PID")
        records: dict[int, dict] = {}
        for pid in members_before:
            records[pid] = _live_process(pid, owner=owner)
        listeners = psutil.net_connections(kind="tcp")
        registered = set(native.REGISTERED_PORTS) | {port}
        foreign = []
        unexpected = []
        requested = []
        for conn in listeners:
            if conn.status != psutil.CONN_LISTEN or not conn.laddr:
                continue
            if conn.pid in records and conn.laddr.port != port:
                unexpected.append({"pid": conn.pid, "port": conn.laddr.port})
            if conn.laddr.port not in registered:
                continue
            if conn.pid not in records:
                foreign.append({"pid": conn.pid, "port": conn.laddr.port})
            elif conn.laddr.port == port:
                if conn.laddr.ip not in ("127.0.0.1", "::1"):
                    unexpected.append({"pid": conn.pid, "port": port})
                requested.append({"pid": conn.pid, "port": port})
            else:
                unexpected.append({"pid": conn.pid, "port": conn.laddr.port})
        members_after = owner.member_ids()
        root_birth_handle = owner._handle_create_time(owner.process)
        root_handle_valid = abs(root_birth_handle - owner.birth) <= 0.002
        if owner.pid in records:
            root_handle_valid &= abs(records[owner.pid]["birth"] - owner.birth) <= 2
        owned = (
            requested[0]
            if requested and all(c == requested[0] for c in requested)
            else None
        )
        census = {
            "active": owner.active,
            "job_present": bool(owner.job),
            "job_limit_kill_on_close": bool(
                flags & WINDOWS_KILL_ON_JOB_CLOSE
                and flags & WINDOWS_ACTIVE_PROCESS_LIMIT
            ),
            # The created Job uses EXACTLY these two frozen limits. In
            # particular, BREAKAWAY_OK (0x800), SILENT_BREAKAWAY_OK (0x1000)
            # and unknown flags would permit untracked child execution.
            "job_limit_exact_no_breakaway": flags == (
                WINDOWS_KILL_ON_JOB_CLOSE | WINDOWS_ACTIVE_PROCESS_LIMIT
            ),
            "job_active_limit": limit,
            "owner_handle_matches_root": root_handle_valid,
            "root_pid": owner.pid,
            "root_birth": (
                records[owner.pid]["birth"]
                if owner.pid in records
                else (
                    ledger.root_birth
                    if isinstance(ledger, LineageLedger)
                    else owner.birth
                )
            ),
            "members": records,
            "owned_listener": owned,
            "other_listeners": foreign,
            "unexpected_job_listeners": unexpected,
            "job_member_census_complete": True,
            "port_census_complete": True,
            # Job enumeration has no ordering contract; genuine set changes
            # are torn censuses, permutations are not.
            "census_stable": sorted(members_before) == sorted(members_after),
        }
    except psutil.NoSuchProcess:
        return _result(
            "PENDING_TRANSIENT_JOB_CENSUS" if phase != "active" else "REFUSED",
            []
            if phase != "active"
            else ["JOB_MEMBER_DISAPPEARED_DURING_ACTIVE_CENSUS"],
        )
    except (
        psutil.Error,
        OSError,
        ValueError,
        TypeError,
        supervisor.SupervisorRefused,
        LineageRefused,
    ):
        return _result("REFUSED", ["READ_ONLY_WIN32_JOB_LINEAGE_CENSUS_UNAVAILABLE"])
    return assess_lineage_snapshot(
        census,
        study=study,
        model=model,
        server=server,
        output=output,
        port=port,
        phase=phase,
        ledger=ledger,
    )
