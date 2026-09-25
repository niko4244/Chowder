"""Sandboxed SWE-smith replay: independent verification of repair trajectories.

For each fetched trajectory, replay it inside a disposable Podman container:

  phase A (setup, network on):  clone the exact repo/commit, install deps
  phase B (replay, network off): apply the trajectory's patch, run tests

Every executed command and its actual captured output is recorded. A trace is
emitted only when independently executed tests come back green; the dataset's
self-reported ``resolved`` flag is carried as a claim, never as evidence.

Design constraints (mission): sandbox third-party code, resource limits,
network restrictions, secret isolation, never execute untrusted repos on the
host, preserve failed actions separately, fail closed without real evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
from pathlib import Path

BASE_IMAGE = "docker.io/library/python:3.11-slim"
REPLAY_IMAGE = "localhost/chowder-replay-base:latest"


def ensure_base_image() -> None:
    """Build the replay base image once (python slim + git); cached afterwards."""
    probe = subprocess.run(["podman", "image", "exists", REPLAY_IMAGE], capture_output=True)
    if probe.returncode == 0:
        return
    dockerfile = ("FROM " + BASE_IMAGE + "\n"
                  "RUN apt-get update -qq && apt-get install -y -qq git && rm -rf /var/lib/apt/lists/*\n"
                  "RUN pip install -q pytest\n")
    subprocess.run(["podman", "build", "-q", "-t", REPLAY_IMAGE, "-"],
                   input=dockerfile.encode(), check=True, capture_output=True)
MEMORY_LIMIT = "2g"
CPUS = "2.0"
PIDS_LIMIT = "256"
SETUP_TIMEOUT = 1800
STEP_TIMEOUT = 300


def digest(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def parse_test_summary(output: str) -> int:
    """Count executed tests from pytest/unittest-style output; 0 if unknown."""
    import re
    m = re.search(r"=+ .*? (\d+) passed(?:.*? (\d+) failed)?.*? =+", output)
    if m:
        return int(m.group(1)) + int(m.group(2) or 0)
    m = re.search(r"Ran (\d+) tests?", output)
    return int(m.group(1)) if m else 0


def run_podman(args: list[str], *, timeout: int) -> tuple[int, str]:
    proc = subprocess.run(["podman", *args], capture_output=True, text=True, timeout=timeout)
    return proc.returncode, (proc.stdout + "\n" + proc.stderr)[-8000:]


def _container_cmd(network: bool, work_mount: str) -> list[str]:
    return ["run", "--rm", "-v", work_mount,
            "--memory", MEMORY_LIMIT, "--cpus", CPUS, "--pids-limit", str(PIDS_LIMIT),
            *([] if network else ["--network", "none"]),
            REPLAY_IMAGE, "bash", "-lc"]


def parse_instance_id(instance_id: str) -> tuple[str, str]:
    """SWE-smith instance ids embed repo and commit: owner__name.abcdef.bugtype__id.

    Returns ("owner/name", "abcdef"). Unknown shapes return empty strings.
    """
    import re
    m = re.match(r"^(?P<owner>[^_][^_]*)__(?P<name>[^.]+)\.(?P<commit>[0-9a-f]{6,40})\.", instance_id)
    if not m:
        return "", ""
    return f"{m.group('owner')}/{m.group('name')}", m.group("commit")


def replay_one(record: dict, work_root: Path, *, instance: dict | None = None) -> dict:
    """Replay one trajectory; a trace is emitted only for observed red->green.

    Protocol (mission phase 3): initialize the exact repo/commit, run the
    verification tests pre-patch (the failure must reproduce — red), apply the
    trajectory's patch, run the tests again (green). Success labels require
    BOTH observations from independently executed commands.
    """
    traj_id = str(record.get("traj_id") or "unknown")
    instance_id = str(record.get("instance_id") or "")
    parsed_repo, parsed_commit = parse_instance_id(instance_id)
    repo = str((instance or {}).get("repo") or parsed_repo)
    base_commit = str((instance or {}).get("base_commit") or parsed_commit)
    if not repo or not base_commit:
        return {"traj_id": traj_id, "status": "skipped",
                "reason": "missing repo/base_commit: instance metadata or parseable instance_id required"}
    patch = record.get("patch") or ""
    if not patch.strip():
        return {"traj_id": traj_id, "status": "skipped", "reason": "empty patch"}

    task_dir = work_root / traj_id.replace("/", "_")
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "ws").mkdir(exist_ok=True)  # bind-mount source must exist before run
    events: list[dict] = []
    log: list[dict] = []
    work_mount = f"{task_dir / 'ws'}:/work"

    def observe(kind: str, action: dict, returncode: int, output: str, *, tests: int | None = None) -> None:
        good = returncode == 0
        event = {"kind": kind, "action": action, "observation": output[-2000:],
                 "returncode": returncode, "verdict": "verified_good" if good else "verified_bad"}
        if tests is not None:
            event["tests_executed"] = tests
        events.append(event)
        log.append({"command": action.get("command"), "returncode": returncode,
                    "output_tail": output[-4000:]})

    # ---- phase A: setup (network on, no secrets in the container env) ----
    rc, out = run_podman(_container_cmd(True, work_mount) + [
        f"git clone -q https://github.com/{shlex.quote(repo)}.git /work/repo"
        f" && cd /work/repo && git checkout -q {shlex.quote(base_commit)}"],
        timeout=SETUP_TIMEOUT)
    observe("tool", {"tool": "run_command",
                     "command": f"clone {repo}@{base_commit[:12]}"}, rc, out)
    if rc != 0:
        return _finish(task_dir, traj_id, events, log, "setup_failed")

    rc, out = run_podman(_container_cmd(True, work_mount) + [
        "cd /work/repo && (pip install -q -e . || true)"],
        timeout=SETUP_TIMEOUT)
    observe("tool", {"tool": "run_command", "command": "pip install -e ."}, rc, out)

    tests = str((instance or {}).get("test_cmd") or "python -m pytest -x -q")

    # ---- phase B: reproduce the failure (network off, patch NOT applied) ----
    rc0, out0 = run_podman(_container_cmd(False, work_mount) + [
        f"cd /work/repo && {tests}"], timeout=STEP_TIMEOUT)
    executed0 = parse_test_summary(out0)
    observe("test", {"tool": "run_tests", "command": tests, "phase": "pre_patch"},
            rc0, out0, tests=executed0)
    failure_reproduced = rc0 != 0 and executed0 > 0

    # ---- phase C: apply the patch, verify green (network off) ----
    (task_dir / "ws" / "patch.diff").write_text(patch, encoding="utf-8")
    rc1, out1 = run_podman(_container_cmd(False, work_mount) + [
        "cd /work/repo && git apply --whitespace=nowarn /work/patch.diff"],
        timeout=STEP_TIMEOUT)
    observe("tool", {"tool": "apply_patch"}, rc1, out1)
    patch_applied = rc1 == 0

    rc2, out2 = run_podman(_container_cmd(False, work_mount) + [
        f"cd /work/repo && {tests}"], timeout=STEP_TIMEOUT)
    executed2 = parse_test_summary(out2)
    observe("test", {"tool": "run_tests", "command": tests, "phase": "post_patch"},
            rc2, out2, tests=executed2)

    verified = failure_reproduced and patch_applied and rc2 == 0 and executed2 > 0
    status = "verified" if verified else "not_green"
    return _finish(task_dir, traj_id, events, log, status,
                   returncode=rc2 if patch_applied else 1,
                   tests_executed=executed2, failure_reproduced=failure_reproduced,
                   instance_id=instance_id, repo=repo, base_commit=base_commit)


def _finish(task_dir: Path, traj_id: str, events: list[dict], log: list[dict],
            status: str, *, returncode: int = 1, tests_executed: int = 0,
            failure_reproduced: bool = False, instance_id: str = "",
            repo: str = "", base_commit: str = "") -> dict:
    evidence = {"log": log}
    (task_dir / "replay_log.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    record = {"task_id": traj_id, "repository": repo, "task": f"repair {instance_id}",
              "events": events,
              "failure_reproduced": failure_reproduced,
              "claimed_resolved_source": "self-reported dataset flag (not evidence)"}
    if status == "verified":
        record["verification"] = {"method": "sandbox_replay", "returncode": returncode,
                                  "tests_executed": tests_executed,
                                  "trace_sha256": digest(events)}
    record["status"] = status
    record["replay_log"] = str(task_dir / "replay_log.json")
    return record


def replay_batch(input_path: Path, work_root: Path, out_path: Path,
                 instances_path: Path | None = None, *, limit: int = 5) -> dict:
    """Replay up to ``limit`` trajectories; writes a replayed-traces JSONL."""
    if subprocess.run(["podman", "info"], capture_output=True).returncode != 0:
        raise RuntimeError("podman is not available; refusing to fabricate replay evidence")
    ensure_base_image()
    instances = {}
    if instances_path:
        for line in instances_path.read_text(encoding="utf8").splitlines():
            if line.strip():
                row = json.loads(line)
                instances[row.get("instance_id")] = row
    out_path.parent.mkdir(parents=True, exist_ok=True)
    verified = failed = skipped = 0
    with out_path.open("w", encoding="utf8") as out:
        for line in input_path.read_text(encoding="utf8").splitlines():
            if not line.strip() or verified + failed >= limit:
                continue
            result = replay_one(json.loads(line), work_root,
                                instance=instances.get(json.loads(line).get("instance_id")))
            if result["status"] == "verified":
                verified += 1
            elif result["status"] == "skipped":
                skipped += 1
            else:
                failed += 1
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
    summary = {"verified": verified, "not_green": failed, "skipped": skipped}
    (out_path.parent / "replay_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="fetch_smith.py output")
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--instances", type=Path, help="JSONL of task-instance metadata (repo, base_commit, test_cmd)")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(replay_batch(args.input, args.work, args.out,
                                  args.instances, limit=args.limit), indent=2))


if __name__ == "__main__":
    main()
