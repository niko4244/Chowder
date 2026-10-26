"""Fail-closed gate for official AutomationBench v1.0.6 PUBLIC tasks on private Q4.

Default mode is a read-only source check and command preview. --execute may run
the pinned OFFICIAL Python evaluator against an ALREADY serving loopback model,
but only after source, model, process, physical, commit, cgroup, CUDA and dataset
admission. This program neither launches a GPU server nor evaluates generated
Python/shell; official API tools mutate the benchmark's simulated WorldState.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import ipaddress
import json
import os
import re
import socket
import subprocess
import sys
import tomllib
from pathlib import Path


OFFICIAL_COMMIT = "4a8e1061254004d9dac807054eed33fad7d1ff14"
Q4_BYTES = 11563915264
Q4_SHA256 = "2d64b786461b99a335fdbbdf7c4d3a6758b21409bce755751714ab7cd8f63bf7"
DOMAINS = ("sales", "marketing", "operations", "support", "finance", "hr")
EXPECTED_BLOBS = {
    "pyproject.toml": "494248b64e6c427b0a6c5dabd44732bc57a1d7db",
    "uv.lock": "c9162350bfde08b16366b100d6819372571cf6c7",
    "automationbench/scripts/eval.py": "fed8beba54bf07123e2b0a2f7caee804b97548fc",
    "automationbench/runner.py": "f7a7fcaab322b18018594a99a5aaa0007d414819",
    "automationbench/tools/api/__init__.py": "6a8ffef18e6f0c5d5d0744a7bd818ceee008a755",
    "automationbench/tools/api/fetch.py": "883a6c90ff7f27ac5a3787b9db15e857153342cc",
    "automationbench/rubric/__init__.py": "70d3287403a620063d9a8d01c3aeb6ae954ae0ba",
}
MIN_RAM_AVAILABLE_GIB = 7.0
MIN_COMMIT_AVAILABLE_GIB = 8.0
MIN_CGROUP_FREE_GIB = 8.0
MIN_CUDA_FREE_MIB = 12288  # each T4, before the server is launched


class GateBlocked(RuntimeError):
    pass


def _run_readonly(args: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True, timeout=20)
    return result.stdout.strip()


def check_official_source(source: Path) -> dict:
    if not source.is_dir():
        raise GateBlocked("Official upstream checkout missing")
    if any((source / name).exists() for name in (".env", "sitecustomize.py", "usercustomize.py")):
        raise GateBlocked("Official checkout contains local startup/credential overrides")
    try:
        commit = _run_readonly(["git", "rev-parse", "HEAD"], cwd=source)
        tracked_changes = _run_readonly(["git", "status", "--porcelain", "--untracked-files=all"], cwd=source)
    except (OSError, subprocess.SubprocessError) as exc:
        raise GateBlocked("Official Git identity cannot be verified") from exc
    if commit != OFFICIAL_COMMIT or tracked_changes:
        raise GateBlocked("Official evaluator SHA mismatch or modified tracked files")
    for path, expected in EXPECTED_BLOBS.items():
        try:
            oid = _run_readonly(["git", "rev-parse", f"HEAD:{path}"], cwd=source)
        except (OSError, subprocess.SubprocessError) as exc:
            raise GateBlocked(f"Official file missing: {path}") from exc
        if oid != expected:
            raise GateBlocked(f"Pinned official blob SHA changed: {path}")
    metadata = tomllib.loads((source / "pyproject.toml").read_text(encoding="utf-8"))
    if (metadata["project"]["name"], metadata["project"]["version"], metadata["project"].get("requires-python")) != (
        "automation-bench", "1.0.6", ">=3.13"
    ):
        raise GateBlocked("Official pyproject identity/version changed")
    if metadata["project"]["scripts"].get("auto-bench") != "automationbench.scripts.eval:main":
        raise GateBlocked("Official scorer/CLI entrypoint changed")
    tool_source = (source / "automationbench/tools/api/__init__.py").read_text(encoding="utf-8")
    rubric_source = (source / "automationbench/rubric/__init__.py").read_text(encoding="utf-8")
    fetch_source = (source / "automationbench/tools/api/fetch.py").read_text(encoding="utf-8")
    if "API_TOOLS = [api_search, api_fetch, base64_encode]" not in tool_source:
        raise GateBlocked("Unexpected official API tool schema")
    if 'return float(state.get("partial_credit", 0.0) == 1.0)' not in rubric_source:
        raise GateBlocked("Official binary grader semantics changed")
    if "return router(world, method, internal_path, parsed_params, body_value)" not in fetch_source:
        raise GateBlocked("Official WorldState API routing semantics changed")
    return {"upstream_commit": commit, "package": "automation-bench==1.0.6", "toolset": "api", "grader": "task_completed_correctly"}


def check_private_location(output: Path, forbidden_roots: tuple[Path, ...]) -> Path:
    if not output.is_absolute():
        raise GateBlocked("Result location must be absolute")
    if output.is_symlink():
        raise GateBlocked("Private output must not be a symbolic link")
    result = output.resolve(strict=False)
    if result.suffix.lower() != ".json":
        raise GateBlocked("Private result path must be a JSON file")
    if not result.parent.is_dir() or result.exists() or result.is_symlink() or any(
        result.with_suffix(extension).exists() for extension in (".stdout.log", ".stderr.log")
    ):
        raise GateBlocked("Use a new exclusive JSON filename in an existing private folder")
    for root in forbidden_roots:
        base = root.resolve(strict=False)
        if result == base or base in result.parents:
            raise GateBlocked("Results must remain outside any Git/source/model root")
    return result


def check_original_q4(model: Path) -> dict:
    if not model.is_file() or model.stat().st_size != Q4_BYTES or model.is_symlink():
        raise GateBlocked("Original protected Q4 file absent, symlink, or wrong byte count")
    digest = hashlib.sha256()
    with model.open("rb") as reader:
        for chunk in iter(lambda: reader.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != Q4_SHA256:
        raise GateBlocked("Original Q4 SHA-256 does not match the frozen model")
    return {"q4_sha256": Q4_SHA256, "q4_bytes": Q4_BYTES}


def read_memory_floors(meminfo: str, cgroup_current: int, cgroup_max: int) -> dict:
    fields = dict((key, int(value)) for key, value in re.findall(r"^(\w+):\s+(\d+)\s+kB$", meminfo, re.M))
    needed = ("MemAvailable", "CommitLimit", "Committed_AS")
    if not all(x in fields for x in needed):
        raise GateBlocked("Physical/commit accounting unavailable")
    physical = fields["MemAvailable"] * 1024 / 2**30
    commit = (fields["CommitLimit"] - fields["Committed_AS"]) * 1024 / 2**30
    cgroup = (cgroup_max - cgroup_current) / 2**30
    if min(physical, commit, cgroup) < 0:
        raise GateBlocked("Memory accounting invalid")
    if (physical < MIN_RAM_AVAILABLE_GIB or commit < MIN_COMMIT_AVAILABLE_GIB or cgroup < MIN_CGROUP_FREE_GIB):
        raise GateBlocked(f"Insufficient host RAM/commit/cgroup: {physical:.2f}/{commit:.2f}/{cgroup:.2f} GiB")
    return {"physical_available_gib": round(physical, 2), "commit_headroom_gib": round(commit, 2),
            "cgroup_free_gib": round(cgroup, 2)}


def inspect_linux_memory() -> dict:
    if sys.platform != "linux":
        raise GateBlocked("GPU admission is Kaggle Linux only")
    control = Path("/sys/fs/cgroup")
    try:
        current = int((control / "memory.current").read_text().strip())
        maximum_text = (control / "memory.max").read_text().strip()
        if maximum_text == "max":
            raise GateBlocked("Unbounded cgroup memory: admission cannot be certified")
        maximum = int(maximum_text)
        text = Path("/proc/meminfo").read_text(encoding="ascii")
    except (FileNotFoundError, ValueError) as exc:
        raise GateBlocked("Required cgroup v2/meminfo source unavailable") from exc
    return read_memory_floors(text, current, maximum)


def parse_gpu_capacity(query: str, active_pids: str, admitted_pid: int) -> dict:
    cards = []
    for line in query.splitlines():
        cols = [col.strip() for col in line.split(",")]
        if len(cols) != 3 or not cols[0] or not all(x.isdigit() for x in cols[1:]):
            raise GateBlocked("Invalid CUDA capacity witness")
        cards.append({"gpu": cols[0], "free_mib": int(cols[1]), "total_mib": int(cols[2])})
    if len(cards) != 2 or any("T4" not in c["gpu"] or c["free_mib"] < 0 for c in cards):
        raise GateBlocked("Exactly two Kaggle T4 cards are required")
    if any(c["total_mib"] < 15000 for c in cards):
        raise GateBlocked("Unexpected CUDA device capacity")
    pids = set()
    for item in active_pids.splitlines():
        item = item.strip()
        if not item or item in ("[Not Supported]", "N/A"):
            continue
        if not item.isdigit():
            raise GateBlocked("Unparseable live CUDA process census")
        pids.add(int(item))
    if admitted_pid == 0:
        if pids:
            raise GateBlocked("Prelaunch CUDA check found a protected/unknown GPU compute process")
        if any(card["free_mib"] < MIN_CUDA_FREE_MIB for card in cards):
            raise GateBlocked("Prelaunch CUDA available VRAM is below 12288 MiB per T4")
        return {"cuda_devices": cards, "cuda_pids": []}
    if pids - {admitted_pid}:
        raise GateBlocked("Protected or unknown CUDA compute process is active")
    if admitted_pid not in pids:
        raise GateBlocked("Pinned model-serving PID does not own observed CUDA compute")
    if any(card["free_mib"] < 2048 for card in cards):
        raise GateBlocked("Loaded CUDA model leaves under 2 GiB free per device")
    return {"cuda_devices": cards, "cuda_pids": sorted(pids)}


def inspect_cuda(admitted_pid: int) -> dict:
    try:
        query = _run_readonly(["nvidia-smi", "--query-gpu=name,memory.free,memory.total", "--format=csv,noheader,nounits"])
        pids = _run_readonly(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"])
    except (OSError, subprocess.SubprocessError) as exc:
        raise GateBlocked("CUDA capacity/process census unavailable") from exc
    return parse_gpu_capacity(query, pids, admitted_pid)


def _boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()


def _receipt_digest(payload: dict) -> str:
    fields = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    return hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def capture_prelaunch(model: Path, receipt: Path) -> dict:
    original = check_original_q4(model)
    memory = inspect_linux_memory()
    cuda = inspect_cuda(0)
    report = {
        "schema": "reap-q4-automationbench-public-prelaunch.v1",
        "at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "boot_id": _boot_id(),
        **original, **memory, **cuda,
    }
    report["receipt_sha256"] = _receipt_digest(report)
    with receipt.open("x", encoding="utf-8") as writer:
        json.dump(report, writer, indent=2, sort_keys=True)
        writer.write("\n")
    return report


def verify_prelaunch(receipt: Path, cuda: dict) -> dict:
    try:
        witness = json.loads(receipt.read_text(encoding="utf-8"))
        timestamp = datetime.datetime.fromisoformat(witness["at_utc"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise GateBlocked("Malformed or inaccessible prelaunch witness") from exc
    now = datetime.datetime.now(datetime.timezone.utc)
    if timestamp.tzinfo is None or not datetime.timedelta(0) <= now - timestamp <= datetime.timedelta(hours=2):
        raise GateBlocked("Prelaunch resource evidence expired or clock invalid")
    if witness.get("schema") != "reap-q4-automationbench-public-prelaunch.v1":
        raise GateBlocked("Prelaunch witness schema incorrect")
    if witness.get("receipt_sha256") != _receipt_digest(witness):
        raise GateBlocked("Prelaunch witness changed since recording")
    if witness.get("boot_id") != _boot_id() or witness.get("q4_sha256") != Q4_SHA256 or witness.get("q4_bytes") != Q4_BYTES:
        raise GateBlocked("Different boot/model source than prelaunch observation")
    cards = witness.get("cuda_devices")
    # Match physical device identity and total capacity; free VRAM should change after load.
    if not isinstance(cards, list) or len(cards) != 2 or any(
        not isinstance(cards[i], dict)
        or cards[i].get("gpu") != cuda["cuda_devices"][i]["gpu"]
        or cards[i].get("total_mib") != cuda["cuda_devices"][i]["total_mib"]
        for i in range(2)
    ):
        raise GateBlocked("Prelaunch witness was recorded on different CUDA devices")
    if witness.get("cuda_pids") != [] or any(not isinstance(card.get("free_mib"), int) or card["free_mib"] < MIN_CUDA_FREE_MIB for card in cards):
        raise GateBlocked("Prelaunch GPUs were already busy or insufficient")
    if any(witness.get(field, -1) < minimum for field, minimum in (
        ("physical_available_gib", MIN_RAM_AVAILABLE_GIB),
        ("commit_headroom_gib", MIN_COMMIT_AVAILABLE_GIB),
        ("cgroup_free_gib", MIN_CGROUP_FREE_GIB),
    )):
        raise GateBlocked("Prelaunch physical/commit/cgroup headroom was not adequate")
    return {"prelaunch_sha256": witness["receipt_sha256"], "prelaunch_at_utc": witness["at_utc"]}


def check_listener(pid: int, port: int, model: Path) -> dict:
    if not 1 <= pid <= 2**31 - 1 or not 1024 <= port <= 65535:
        raise GateBlocked("Invalid native server process/port")
    proc = Path("/proc") / str(pid)
    try:
        cmdline = [x.decode("utf-8", errors="strict") for x in (proc / "cmdline").read_bytes().split(b"\x00") if x]
        sockets = {os.readlink(path) for path in (proc / "fd").iterdir() if path.is_symlink()}
        tcp = Path("/proc/net/tcp").read_text(encoding="ascii").splitlines()[1:]
    except (OSError, UnicodeError) as exc:
        raise GateBlocked("Cannot prove exact live server process and socket") from exc
    argv = "\x00".join(cmdline)
    if not cmdline or not any("llama-server" in Path(value).name for value in cmdline[:2]):
        raise GateBlocked("Unexpected native llama-server executable")
    expected_pairs = {"--host": "127.0.0.1", "--port": str(port), "--fit": "on", "-np": "4", "-c": "16384"}
    for key, value in expected_pairs.items():
        if not any(cmdline[i:i + 2] == [key, value] for i in range(len(cmdline) - 1)):
            raise GateBlocked(f"Pinned native server setting absent: {key} {value}")
    if "--jinja" not in cmdline or not any(cmdline[i:i + 2] == ["-m", str(model)] for i in range(len(cmdline) - 1)):
        raise GateBlocked("Native model SHA path or tool-call Jinja flag absent")
    inode_matches = []
    for row in tcp:
        columns = row.split()
        if len(columns) > 9 and columns[1] == f"0100007F:{port:04X}" and columns[3] == "0A":
            inode_matches.append(f"socket:[{columns[9]}]")
    if len(inode_matches) != 1 or inode_matches[0] not in sockets:
        raise GateBlocked("Loopback listener is absent or owned by another PID")
    return {"server_pid": pid, "port": port, "socket_inode": inode_matches[0], "native_flag_count": len(expected_pairs)}


def official_command(source: Path, model_id: str, port: int, result_file: Path, *, full: bool) -> list[str]:
    if not re.fullmatch(r"[a-zA-Z0-9._:/-]{2,100}", model_id) or model_id.startswith("-"):
        raise GateBlocked("Invalid model identifier")
    count = "-1" if full else "1"
    domains = ",".join(DOMAINS) if full else "sales"
    return [sys.executable, "-m", "automationbench.scripts.eval", "--model", model_id,
            "--api", "chat_completions", "--base-url", f"http://127.0.0.1:{port}/v1",
            "--api-key", "local-only", "--domains", domains, "--num-examples", count,
            "--max-steps", "50", "--max-concurrent", "1", "--toolset", "api",
            "--no-ensure-complete", "--save-every", "1", "--export-json", str(result_file)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--q4", type=Path)
    parser.add_argument("--server-pid", type=int)
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--model-id", default="reap128-v2-q4_k_m-20261025")
    parser.add_argument("--private-json", type=Path)
    parser.add_argument("--capture-prelaunch", type=Path)
    parser.add_argument("--prelaunch-json", type=Path)
    parser.add_argument("--full-public-600", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-wall-seconds", type=int, default=7200)
    args = parser.parse_args()
    try:
        source = args.official_source.resolve(strict=True)
        report = {"stage": "OFFICIAL_SOURCE_ONLY", **check_official_source(source),
                  "scope": "PUBLIC 600-task source, NOT PRIVATE OFFICIAL LEADERBOARD",
                  "executable": False, "model_inference_started": False}
        if args.capture_prelaunch:
            if args.execute or not args.q4:
                raise GateBlocked("Capture requires --q4 and cannot be combined with --execute")
            model = args.q4.expanduser().absolute()
            record = check_private_location(args.capture_prelaunch, (source, Path(__file__).resolve().parents[1], model.parent))
            report.update(capture_prelaunch(model, record))
            report["stage"] = "PRELAUNCH_RESOURCE_PASS_NOT_MODEL_EXECUTION"
            report["private_prelaunch_receipt"] = str(record)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        if not args.execute:
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        if not args.q4 or not args.private_json or not args.server_pid or not args.prelaunch_json:
            raise GateBlocked("Q4 path, private output path, prelaunch witness and verified server PID are required")
        if not 1 <= args.max_wall_seconds <= 21600:
            raise GateBlocked("Max wall clock must be between 1 and 21600 seconds")
        if sys.version_info[:2] != (3, 13):
            raise GateBlocked("Official evaluator must run with Python 3.13")
        model = args.q4.expanduser().absolute()
        output = check_private_location(args.private_json, (source, Path(__file__).resolve().parents[1], model.parent))
        report.update(check_original_q4(model))
        report.update(inspect_linux_memory())
        cuda = inspect_cuda(args.server_pid)
        report.update(cuda)
        report.update(verify_prelaunch(args.prelaunch_json, cuda))
        report.update(check_listener(args.server_pid, args.port, model))
        # Model server is already live. This runner MUST NOT start or modify it.
        command = official_command(source, args.model_id, args.port, output, full=args.full_public_600)
        report.update({"stage": "ADMITTED", "executable": True, "mode": "public600" if args.full_public_600 else "public_one_sales",
                       "command_argc": len(command), "private_result": str(output)})
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        # Strip inherited hosted-provider keys, proxies, dotenv overrides and
        # shell startup hooks before invoking the trusted official evaluator.
        forwarded = ("PATH", "HOME", "TMPDIR", "TMP", "TEMP", "LANG", "LC_ALL", "SSL_CERT_FILE", "VIRTUAL_ENV")
        env = {name: os.environ[name] for name in forwarded if name in os.environ}
        env.update({"PYTHONPATH": str(source), "OPENAI_API_KEY": "local-only", "AUTOMATIONBENCH_STRICT_ASSERTIONS": "1",
                    "PYTHONDONTWRITEBYTECODE": "1", "PYTHON_DOTENV_DISABLED": "1"})
        with output.with_suffix(".stdout.log").open("x", encoding="utf-8") as stdout, output.with_suffix(".stderr.log").open("x", encoding="utf-8") as stderr:
            completed = subprocess.run(command, cwd=source, env=env, stdout=stdout, stderr=stderr,
                                       timeout=args.max_wall_seconds, check=False)
        if completed.returncode:
            raise GateBlocked(f"Official scorer exited {completed.returncode}; inspect PRIVATE logs")
        if not output.is_file() or not output.stat().st_size:
            raise GateBlocked("Official evaluator did not write a complete scored artifact")
        print("OFFICIAL_PUBLIC_RUN_COMPLETED_UNVERIFIED: independently verify exact n/task IDs and 0/1 scorer outputs before claiming a score")
        return 0
    except (GateBlocked, OSError, subprocess.SubprocessError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
