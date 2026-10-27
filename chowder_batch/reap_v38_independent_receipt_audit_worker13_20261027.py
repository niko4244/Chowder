"""Read-only, original-source-bound audit of the V38 HumanEval research journal.

Never calls Podman, the original model, or the research executor. A live journal
is read as one bounded byte snapshot; an unfinished trailing record is rejected,
not counted. Only a complete source-bound prefix can yield an observed score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from chowder_batch import reap_humaneval_v38_actual_execution_20261027 as v38


LOADER_SHA = "3319565d5d5a6aa210151a32bd367a136a08cfb530402c51f311143de2f08e66"
OBSERVER_SHA = "c4e5fd10ae77338337357ee250d8f5e19d925f74cf520c42b2e25b94cc519425"
PYTHON_SHA = v38.V37_PYTHON_EXECUTABLE_SHA256
ROW_SIZE_LIMIT = 256 * 1024
FILE_SIZE_LIMIT = 164 * ROW_SIZE_LIMIT
ALLOWED_MOUNTS = {
    "/": ("overlay", "overlay"), "/work": ("tmpfs", "tmpfs"),
    "/tmp": ("tmpfs", "tmpfs"), "/run": ("tmpfs", "tmpfs"),
    "/var/tmp": ("tmpfs", "tmpfs"), "/run/.containerenv": ("tmpfs", "tmpfs"),
    "/run/secrets": ("tmpfs", "tmpfs"), "/etc/hosts": ("tmpfs", "tmpfs"),
    "/etc/hostname": ("tmpfs", "tmpfs"), "/etc/resolv.conf": ("tmpfs", "tmpfs"),
    "/dev": ("tmpfs", "tmpfs"), "/dev/pts": ("devpts", "devpts"),
    "/dev/shm": ("tmpfs", "shm"), "/dev/mqueue": ("mqueue", "mqueue"),
    "/dev/null": ("devtmpfs", "none"), "/dev/zero": ("devtmpfs", "none"),
    "/dev/full": ("devtmpfs", "none"), "/dev/tty": ("devtmpfs", "none"),
    "/dev/random": ("devtmpfs", "none"), "/dev/urandom": ("devtmpfs", "none"),
    "/proc": ("proc", "proc"), "/proc/bus": ("proc", "proc"),
    "/proc/fs": ("proc", "proc"), "/proc/irq": ("proc", "proc"),
    "/proc/sys": ("proc", "proc"), "/proc/sysrq-trigger": ("proc", "proc"),
    "/proc/acpi": ("tmpfs", "tmpfs"), "/proc/scsi": ("tmpfs", "tmpfs"),
    "/proc/kcore": ("devtmpfs", "none"), "/proc/keys": ("devtmpfs", "none"),
    "/proc/latency_stats": ("devtmpfs", "none"),
    "/proc/timer_list": ("devtmpfs", "none"),
    "/proc/interrupts": ("devtmpfs", "none"),
    "/sys": ("sysfs", "sysfs"), "/sys/fs/cgroup": ("cgroup2", "cgroup2"),
    "/sys/firmware": ("tmpfs", "tmpfs"), "/sys/fs/selinux": ("tmpfs", "tmpfs"),
}
READ_ONLY = {"/", "/tmp", "/run/.containerenv", "/run/secrets",
             "/etc/hosts", "/etc/hostname", "/etc/resolv.conf", "/sys",
             "/sys/fs/cgroup", "/proc/acpi", "/proc/scsi", "/proc/kcore",
             "/proc/keys", "/proc/latency_stats", "/proc/timer_list",
             "/proc/interrupts", "/proc/bus", "/proc/fs", "/proc/irq",
             "/proc/sys", "/proc/sysrq-trigger", "/sys/firmware",
             "/sys/fs/selinux"}


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError("REFUSED independent V38 receipt audit: " + reason)


def _strict_json(raw: bytes) -> dict:
    def unique(pairs: list[tuple[str, object]]) -> dict:
        result = {}
        for k, value in pairs:
            _require(k not in result, "duplicate JSON key")
            result[k] = value
        return result

    result = json.loads(raw.decode("utf-8"), object_pairs_hook=unique,
                        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    _require(isinstance(result, dict), "record is not a JSON object")
    return result


def _source_manifest(arm: str) -> dict:
    _require(arm in ("q4", "q8"), "unsupported arm")
    _require(hashlib.sha256(v38.V38_LOADER.encode()).hexdigest() == LOADER_SHA,
             "reviewed loader source changed")
    _require(hashlib.sha256(v38.V38_VM_SOURCE_OBSERVER.encode()).hexdigest() == OBSERVER_SHA,
             "reviewed host observer source changed")
    return v38.original_program_manifest(arm)


def _witness(row: dict, expected: dict) -> None:
    witness = row["live_external_vm_host_witness"]
    _require(isinstance(witness, dict), "missing original external witness")
    digest = expected["candidate_program_sha256"]
    _require(witness.get("source_program_sha256") == digest, "candidate source witness drift")
    _require(witness.get("research_only_executable_source_sha256") == LOADER_SHA
             and witness.get("outside_vm_host_observer_sha256") == OBSERVER_SHA,
             "unreviewed V38 executable or observer")
    _require(witness.get("execution_status") == row["execution_status"]
             == "RESEARCH_ONLY_SOURCE_BOUND_NOT_FROZEN_GATE2", "invalid research execution mode")
    _require(witness.get("independent_research_correct") is row["independent_research_correct"]
             and type(witness.get("candidate_exit")) is int
             and type(witness.get("timed_out")) is bool, "invalid independent completion")
    _require(witness["independent_research_correct"] == (
        witness["candidate_exit"] == 0 and not witness["timed_out"]),
        "candidate scoring inconsistent with execution status")
    _require(witness.get("frozen_v2_approved") is False and witness.get("gen0_status") == "BLOCKED",
             "research receipt claims frozen approval")
    post = witness.get("external_owned_process_extinction")
    _require(isinstance(post, dict) and post.get("cgroup_procs") == []
             and post.get("same_owned_processes_live") == [], "external host cleanup unproven")
    host = witness.get("outside_host_exact_staged_source_witness")
    _require(isinstance(host, dict) and host.get("staged_program_sha256") == digest,
             "outside host staged SHA mismatch")
    memfd = host.get("immutable_executable_memfd")
    _require(isinstance(memfd, dict) and memfd.get("sha256") == digest
             and memfd.get("seals") == 15 and type(memfd.get("fd")) is int
             and 3 <= memfd["fd"] <= 64 and type(memfd.get("inode")) is int
             and memfd["inode"] > 0 and type(memfd.get("device")) is int
             and memfd["device"] > 0 and type(memfd.get("pid")) is int,
             "immutable executable memfd provenance invalid")
    processes = host.get("cgroup_procs")
    _require(isinstance(processes, list) and 2 <= len(processes) <= 16
             and all(type(p) is int and p > 1 for p in processes)
             and len(set(processes)) == len(processes)
             and host.get("process_count") == len(processes)
             and memfd["pid"] in processes, "invalid external process census")
    _require(host.get("container_pid") in processes
             and type(host.get("container_pid_start_ticks")) is int
             and host["container_pid_start_ticks"] > 0,
             "invalid original process identity")
    owner = witness.get("source_owner_id")
    _require(isinstance(owner, str) and len(owner) == 64
             and all(c in "0123456789abcdef" for c in owner)
             and owner in host.get("cgroup_path", ""), "unbound owned cgroup")
    _require(host.get("pids_max") == "16" and host.get("memory_max") == "268435456"
             and host.get("interpreter_sha256") == PYTHON_SHA,
             "runtime limits/interpreter mismatch")
    _require(host.get("read_only_root") is True and host.get("read_only_tmp") is True
             and host.get("writable_work") is True, "invalid filesystem protection")
    stat = host.get("staged_inode")
    _require(isinstance(stat, dict) and all(type(stat.get(k)) is int for k in
             ("device", "inode", "size", "nlink", "uid", "mode", "mtime_ns", "ctime_ns"))
             and stat["device"] > 0 and stat["inode"] > 0 and stat["size"] > 0
             and stat["nlink"] == 1 and stat["uid"] != 0 and not stat["mode"] & 0o022,
             "invalid staged inode provenance")
    statuses = host.get("process_status")
    _require(isinstance(statuses, list) and len(statuses) == len(processes)
             and all(isinstance(s, dict) and s.get("capeff") == "0000000000000000"
                     and s.get("nnp") == "1" and s.get("seccomp") == "2"
                     and isinstance(s.get("uid"), str) and len(s["uid"].split()) == 4
                     and all(v.isdecimal() and int(v) != 0 for v in s["uid"].split())
                     for s in statuses), "invalid process privilege/seccomp witness")
    _require(host.get("net_interfaces") == ["lo"], "unexpected network interface")
    ipv4, ipv6 = host.get("ipv4_routes"), host.get("ipv6_routes")
    _require(isinstance(ipv4, list) and ipv4 and isinstance(ipv4[0], str)
             and ipv4[0].split()[:2] == ["Iface", "Destination"]
             and all(isinstance(s, str) and len(s.split()) >= 11 and s.split()[0] == "lo"
                     and s.split()[1] != "00000000" for s in ipv4[1:]),
             "IPv4 kernel route witness invalid")
    _require(isinstance(ipv6, list)
             and all(isinstance(s, str) and len(s.split()) >= 10 and s.split()[-1] == "lo"
                     for s in ipv6), "IPv6 kernel route witness invalid")
    namespaces, observer = host.get("child_ns"), host.get("observer_ns")
    _require(isinstance(namespaces, dict) and isinstance(observer, dict)
             and all(isinstance(namespaces.get(n), str) and namespaces[n] != observer.get(n)
                     and namespaces[n].startswith(n + ":[")
                     for n in ("user", "pid", "mnt", "net")),
             "kernel namespace isolation invalid")
    mounts = host.get("mount_inventory")
    _require(isinstance(mounts, list) and len(mounts) >= 8
             and all(isinstance(m, dict) and m.get("point") in ALLOWED_MOUNTS
                     and (m.get("fstype"), m.get("source")) == ALLOWED_MOUNTS[m["point"]]
                     and isinstance(m.get("options"), list)
                     and (m["point"] not in READ_ONLY or "ro" in m["options"])
                     and (m["point"] not in ("/work", "/tmp", "/run", "/var/tmp")
                          or "nodev" in m["options"])
                     and (m["point"] != "/work" or "noexec" in m["options"])
                     for m in mounts), "untrusted host mount witness")
    mounted = [m["point"] for m in mounts]
    _require(len(mounted) == len(set(mounted)) and
             {"/", "/work", "/tmp", "/dev", "/proc", "/sys", "/sys/fs/cgroup",
              "/run/secrets"}.issubset(mounted), "required mount missing/duplicated")


def audit_bytes(arm: str, raw: bytes, *, saved_manifest: bytes | None = None) -> dict:
    manifest = _source_manifest(arm)
    if saved_manifest is not None:
        published = _strict_json(saved_manifest)
        _require(published == {k: v for k, v in manifest.items()
                                if k != "source_rows_in_original_order"},
                 "original input manifest diverges from source")
    _require(isinstance(raw, bytes) and len(raw) <= FILE_SIZE_LIMIT, "unbounded evidence file")
    _require(not raw or raw.endswith(b"\n"), "partial, unflushed or truncated journal tail")
    lines = raw.splitlines()
    _require(len(lines) <= 164 and all(0 < len(line) <= ROW_SIZE_LIMIT for line in lines),
             "invalid original prefix population or oversized row")
    complete, correct, executed, missed = [], 0, 0, 0
    for index, line in enumerate(lines):
        item = _strict_json(line)
        expected = manifest["source_rows_in_original_order"][index]
        _require(item.get("index") == index and type(item.get("index")) is int
                 and item.get("id") == expected["id"] and item.get("arm") == manifest["arm"],
                 "nonmonotonic, repeated or alien HumanEval row")
        _require(item.get("source_session") == manifest["original_serving_session"]
                 and item.get("original_model_sha256") == manifest["original_model_sha256"]
                 and item.get("program_sha256") == expected["candidate_program_sha256"],
                 "original model/session/program source fingerprint drift")
        _require(item.get("original_terminated") is expected["terminated"]
                 and item.get("original_reported_correct_unverified") is
                     expected["original_recorded_correct_unverified"]
                 and item.get("frozen_v2_gate2_eligible") is False
                 and type(item.get("independent_research_correct")) is bool,
                 "original row termination, baseline or frozen gate corrupted")
        if expected["terminated"]:
            _witness(item, expected)
            executed += 1
        else:
            _require(item.get("execution_status") == "SOURCE_BOUND_UNTERMINATED_NOT_EXECUTED"
                     and item["independent_research_correct"] is False
                     and "live_external_vm_host_witness" not in item,
                     "naturally unterminated candidate mislabeled as executed")
            missed += 1
        correct += int(item["independent_research_correct"])
        complete.append({"index": index, "id": expected["id"],
                         "program_sha256": expected["candidate_program_sha256"],
                         "research_correct": item["independent_research_correct"]})
    count = len(lines)
    return {"schema": "chowder.reap-v38-independent-receipt-audit.worker13.v1",
            "arm": arm, "status": "COMPLETE_164_RESEARCH_ONLY" if count == 164 else "VERIFIED_PREFIX_ONLY",
            "verified_contiguous_rows": count, "verified_actual_executions": executed,
            "verified_source_unterminated_nonpasses": missed,
            "verified_independent_correct_prefix": correct,
            "observed_prefix_pass_rate": correct / count if count else None,
            "full_164_independent_correct": correct if count == 164 else None,
            "full_164_denominator": 164 if count == 164 else None,
            "unknown_remaining_rows": 164 - count,
            "journal_snapshot_sha256": hashlib.sha256(raw).hexdigest(),
            "journal_snapshot_bytes": len(raw),
            "original_rows_sha256": manifest["original_rows_raw_sha256"],
            "original_model_sha256": manifest["original_model_sha256"],
            "original_session": manifest["original_serving_session"],
            "ordered_program_manifest_sha256": manifest["ordered_program_manifest_sha256"],
            "reviewed_loader_sha256": LOADER_SHA, "reviewed_observer_sha256": OBSERVER_SHA,
            "frozen_v2_gate2_approved": False, "gen0_status": "BLOCKED",
            "rows": complete}


def audit_live(arm: str, journal: Path, *, snapshot_dir: Path | None = None) -> dict:
    _require(journal.is_file() and not journal.is_symlink(), "missing/linked journal")
    _require(journal.name == f"{arm}-actual-independent-per-case.private.jsonl",
             "wrong original research journal")
    size = journal.stat().st_size
    _require(size <= FILE_SIZE_LIMIT, "journal exceeds bounded scientific record budget")
    with journal.open("rb") as stream:
        raw = stream.read(FILE_SIZE_LIMIT + 1)
    _require(len(raw) <= FILE_SIZE_LIMIT, "oversized journal during read")
    manifest_raw = (journal.parent / "original-input-manifest.json").read_bytes()
    result = audit_bytes(arm, raw, saved_manifest=manifest_raw)
    if snapshot_dir is not None:
        _require(snapshot_dir.is_absolute() and not snapshot_dir.exists()
                 and snapshot_dir.parent.resolve() == Path("F:/").resolve(),
                 "exclusive fresh independent F: snapshot required")
        snapshot_dir.mkdir(mode=0o700, exist_ok=False)
        journal_copy = snapshot_dir / "verified-prefix.private.jsonl"
        with journal_copy.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            import os
            os.fsync(stream.fileno())
        with (snapshot_dir / "independent-audit.json").open("x", encoding="utf-8") as stream:
            json.dump({k: v for k, v in result.items() if k != "rows"}, stream,
                      indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        result["independent_exclusive_snapshot"] = str(snapshot_dir)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("q4", "q8"), required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path)
    args = parser.parse_args()
    try:
        result = audit_live(args.arm, args.journal, snapshot_dir=args.snapshot_dir)
        print(json.dumps({k: v for k, v in result.items() if k != "rows"}, sort_keys=True))
        return 0
    except (ValueError, OSError, UnicodeError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "REFUSED_INDEPENDENT_SCIENTIFIC_AUDIT",
                          "reason": str(error)[:500], "gen0_status": "BLOCKED"}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
