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
import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from uuid import uuid4

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


#: SWE-smith's own per-repo environment spec (upstream configs/install_repo.sh,
#: repo SWE-bench/SWE-smith @ main, MIT): editable install, then test deps from
#: the ``[test]`` extras, falling back to ``requirements-test.txt``, then pytest.
#: This ladder is what previous replays missed -- repos whose tests import
#: optional test deps (anyio, apispec, ...) failed at collection instead of
#: reproducing the injected bug.
SETUP_INSTALL_LADDER = (
    "cd /work/repo && python -m venv /work/venv"
    " && /work/venv/bin/pip install -q -U pip"
    # Podman-on-WSL bind mounts reject some in-place metadata writes
    # (egg-info temp dirs), so the editable wheel is built from an overlay
    # copy under /work (container-writable), not from the bind mount itself.
    " && (git clone -q /work/repo /work/build"
    " && cd /work/build"
    " && /work/venv/bin/pip install -e . || echo 'EDITABLE_INSTALL_FAILED')"
    " && (cd /work/build"
    " && /work/venv/bin/pip install -e '.[test]' || /work/venv/bin/pip install -r requirements-test.txt || echo 'NO_TEST_DEPS_SOURCE')"
    " && (/work/venv/bin/pip install pytest || true)"
    # Tests import the INSTALLED package (like the upstream harness); pytest
    # targets the restored test files under /work/repo/tests.
    " && cd /work/repo"
)
#: Tests must run inside the /work venv: each phase runs in a FRESH --rm
#: container, so only the bind-mounted /work persists between phases.
#: ``-o addopts=`` neutralises repo-configured pytest options that need plugins
#: this venv does not have (mypy's pyproject.toml sets ``-nauto``; without xdist
#: installed, pytest exits with "unrecognized arguments: -nauto" and zero tests
#: run, which reads as a red state that was never observed).
PYTEST = "/work/venv/bin/python -m pytest -o addopts="
MEMORY_LIMIT = "2g"
CPUS = "2.0"
PIDS_LIMIT = "256"
SETUP_TIMEOUT = 1800
STEP_TIMEOUT = 300
#: Test phases execute real suites (dask's overlap module alone collects 11
#: tests behind a heavy import, on a daemon that may be pulling images); the
#: control-plane timeout is too tight for them and times them out as if the
#: harness had failed. Tests get their own, generous budget.
TEST_TIMEOUT = 1200

# ----------------------------------------------------------------- row queue
# A batch is a queue over on-disk row state, never a bare loop: a host under
# load (or a killed process) must not turn a row into a permanent verdict, and
# two batches pointed at one work root must not execute the same row twice.

#: Statuses that are real evidence and must never be silently re-run.
FINAL_STATUSES = ("verified", "not_green", "patch_did_not_fit", "mispaired")
#: Failures that are the *host's* fault (a saturated daemon, a bounded pull
#: that lost a race) are retried; a patch that cannot apply is not.
HOST_LOAD_MARKERS = (
    "pull failed", "timed out", "timeout", "cannot connect", "connection refused",
    "connection reset", "context deadline exceeded", "no space left", "failed to start",
    "temporary failure", "resource temporarily unavailable", "database is locked",
    "internal server error", "server error", "overlay mount", "input/output error",
)
#: A claim older than this is presumed dead (process killed, host slept).
CLAIM_STALE_SECONDS = 6 * 3600
DEFAULT_MAX_ATTEMPTS = 2


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sanitize_slug(text: str, *, limit: int = 60) -> str:
    """Podman-name-safe slug (lowercase, no path/space/tag metacharacters)."""
    return re.sub(r"[^a-z0-9_.-]+", "-", text.lower()).strip("-.")[:limit] or "row"


def red_image_tag(instance_id: str, setup_key: str) -> str:
    """Deterministic red-state image name -- the shared per-instance cache.

    Setup (overlay the bug branch, restore the F2P test files, commit) is the
    expensive half of a replay and is identical for every trajectory of the
    same instance. A content-addressed tag means a resumed batch reuses the
    image instead of redoing setup, and several rows of one instance can share
    a single image.
    """
    return f"localhost/chowder-replay-red-{sanitize_slug(instance_id)}:{setup_key[:12]}"


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_json(path: Path, payload: dict) -> None:
    try:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def row_state_path(task_dir: Path) -> Path:
    return task_dir / "row_state.json"


def row_result_path(task_dir: Path) -> Path:
    return task_dir / "row_result.json"


def update_row_state(task_dir: Path, **fields) -> dict:
    """Heartbeat: what this row is doing right now, flushed as it happens.

    Buffered stdout was lost when a batch died mid-pull (v20: an empty log and
    two empty row dirs, no cause recoverable). State on disk is the durable
    progress signal; stdout is a convenience.
    """
    state = read_json(row_state_path(task_dir)) or {}
    state.update(fields)
    state["updated_at"] = utc_now()
    write_json(row_state_path(task_dir), state)
    return state


def claim_row(task_dir: Path, *, stale_seconds: int = CLAIM_STALE_SECONDS) -> dict:
    """Take this row for THIS process; O_EXCL claim, stale claims reclaimed.

    Two batches may share one work root: each row is executed once, and a row
    whose owner died leaves a stale claim that the next batch takes over with
    ``reclaimed_stale`` recorded rather than silently skipping the row.
    """
    task_dir.mkdir(parents=True, exist_ok=True)
    claim_path = task_dir / "claim.json"
    payload = json.dumps({"pid": os.getpid(), "claimed_at": utc_now(),
                          "claimed_epoch": time.time()})
    for attempt in range(2):
        try:
            fd = os.open(claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            held = read_json(claim_path) or {}
            age = time.time() - float(held.get("claimed_epoch") or 0)
            if attempt == 0 and age > stale_seconds:
                try:  # stale owner: reclaim rather than skip the row
                    claim_path.unlink()
                except OSError:
                    pass
                continue
            return {"claimed": False, "held_by": held}
        else:
            with os.fdopen(fd, "w") as fh:
                fh.write(payload)
            return {"claimed": True, "reclaimed_stale": attempt == 1}
    return {"claimed": False, "held_by": read_json(claim_path) or {}}


def release_claim(task_dir: Path) -> None:
    try:
        (task_dir / "claim.json").unlink()
    except OSError:
        pass


def setup_failure_is_retryable(reason: str) -> bool:
    """Host-load failures are retryable; content failures are not."""
    text = (reason or "").lower()
    return any(marker in text for marker in HOST_LOAD_MARKERS)


def bump_attempts(task_dir: Path, **fields) -> int:
    state = update_row_state(task_dir, **fields)
    attempts = int(state.get("attempts") or 0) + 1
    update_row_state(task_dir, attempts=attempts)
    return attempts


def finalize_row(task_dir: Path, result: dict) -> dict:
    """Persist one row's outcome and release its claim.

    ``row_result.json`` is the queue's source of truth: the plan step reads it
    to decide run/skip/retry, so this write happens before the claim is freed.
    """
    result.setdefault("retryable", False)
    write_json(row_result_path(task_dir), result)
    update_row_state(task_dir, status=result.get("status"), stage="finished",
                     retryable=result.get("retryable"), finished_at=utc_now())
    release_claim(task_dir)
    return result


def plan_rows(rows: list[dict], work_root: Path, *, max_attempts: int = DEFAULT_MAX_ATTEMPTS
              ) -> list[dict]:
    """Queue planning: which rows run, which are already answered, which wait.

    Decisions: ``run`` (never started, or a retryable failure with budget left),
    ``done`` (final evidence on disk -- verified / not_green / mispaired /
    patch_did_not_fit), ``retry_exhausted`` (retryable but out of attempts),
    ``in_flight`` (a live claim from another writer).
    """
    plan: list[dict] = []
    for row in rows:
        traj_id = str(row.get("traj_id") or row.get("instance_id") or "unknown")
        task_dir = work_root / traj_id.replace("/", "_")
        state = read_json(row_state_path(task_dir)) or {}
        result = read_json(row_result_path(task_dir)) or {}
        status = str(result.get("status") or "")
        attempts = int(state.get("attempts") or 0)
        entry = {"traj_id": traj_id, "task_dir": str(task_dir), "attempts": attempts,
                 "status": status or None}
        if status in FINAL_STATUSES:
            entry["decision"] = "done"
        elif status == "setup_failed" and result.get("retryable") and attempts < max_attempts:
            entry["decision"] = "run"
            entry["reason"] = "retryable_host_failure"
        elif status == "setup_failed" and result.get("retryable"):
            entry["decision"] = "retry_exhausted"
        elif status == "setup_failed":
            entry["decision"] = "done"
        elif state.get("stage") and state.get("stage") != "finished":
            held = read_json(task_dir / "claim.json") or {}
            age = time.time() - float(held.get("claimed_epoch") or 0)
            entry["decision"] = ("in_flight" if age <= CLAIM_STALE_SECONDS
                                 else "run")  # abandoned: resume it
        else:
            entry["decision"] = "run"
        plan.append(entry)
    return plan


def digest(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def parse_test_summary(output: str) -> int:
    """Count executed tests from pytest/unittest-style output; 0 if unknown.

    Executed = passed + failed only. Collection errors and "N error"/"N
    skipped" summaries mean nothing ran, so they must contribute 0 -- a red
    observation with 0 executed tests is not a reproduced failure.
    """
    import re
    passed = re.search(r"(\d+) passed", output)
    failed = re.search(r"(\d+) failed", output)
    total = (int(passed.group(1)) if passed else 0) + (int(failed.group(1)) if failed else 0)
    if total:
        return total
    m = re.search(r"Ran (\d+) tests?", output)
    return int(m.group(1)) if m else 0


def write_patch_file(path: Path, patch: str) -> Path:
    """Materialize a patch for a Linux container with LF endings only.

    Python text-mode writes on Windows translate ``\n`` to ``\r\n``, and a
    CRLF patch can never apply to an LF worktree -- that translation silently
    made every apply fail. Both replay paths call this, so the fix exists in
    exactly one place (regression-tested).
    """
    path.write_text(patch.replace("\r\n", "\n"), encoding="utf-8", newline="\n")
    return path


def run_podman(args: list[str], *, timeout: int) -> tuple[int, str]:
    """Run a podman call with a HARD deadline.

    ``subprocess.run(timeout=...)`` is not enough here: it kills the direct
    child on timeout and then blocks in ``communicate()`` waiting for pipe EOF,
    which on Windows never arrives when podman's grandchildren inherited those
    handles. Observed: a 300 s phase that never returned and stalled the batch.
    Popen + kill + re-raise without a second read keeps the deadline real.
    """
    proc = subprocess.Popen(["podman", *args], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
                            text=True)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise
    # None streams (observed on Windows/WSL pipe hiccups) are treated as empty,
    # never concatenated as None.
    out = (stdout or "") + "\n" + (stderr or "")
    return proc.returncode, out[-8000:]


def _persist_log(task_dir: Path, log: list[dict]) -> None:
    """Write evidence as it accumulates instead of only at the end.

    A row that dies mid-way (a saturated podman daemon once timed out a
    cleanup call after the red image had been committed) must not lose the
    commands that already ran -- the failure record is built from this file.
    """
    try:
        (task_dir / "replay_log.json").write_text(
            json.dumps({"log": log}, indent=2), encoding="utf-8")
    except OSError:
        pass


def _best_effort(args: list[str], *, timeout: int = 300) -> None:
    """Cleanup must never turn a finished phase into a lost row."""
    try:
        run_podman(args, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        pass


def _failed_record(task_dir: Path, traj_id: str, reason: str) -> dict:
    """Failure record that still carries whatever evidence reached disk."""
    record = {"task_id": traj_id, "status": "setup_failed", "reason": reason}
    log_path = task_dir / "replay_log.json"
    try:
        log = json.loads(log_path.read_text(encoding="utf-8")).get("log") or []
    except (OSError, ValueError):
        log = []
    if log:
        record["replay_log"] = str(log_path)
        record["last_command"] = {"command": log[-1].get("command"),
                                 "returncode": log[-1].get("returncode")}
    return record


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


def derive_test_targets(patch: str) -> tuple[list[str], str]:
    """Map the patch's touched files to candidate test targets.

    SWE-smith instances embed repo+commit but the trajectories' exact bug ids
    are absent from the SWE-bench/SWE-smith instances dataset (verified:
    0/24 exact id matches), so per-instance FAIL_TO_PASS is only available
    when the caller supplies an exact metadata row. Otherwise derive the
    verification surface from the patch itself: a patch that touches test
    files points at them directly; a patch that touches only source modules
    points at those modules' conventional test files (resolved on disk in
    the container before use). Returns (targets, selection_source).
    """
    import re
    touched: list[str] = []
    for m in re.finditer(r"^\+\+\+ b/(\S+\.py)$", patch, re.M):
        if m.group(1) not in touched:
            touched.append(m.group(1))

    def is_test(path: str) -> bool:
        stem = path.rsplit("/", 1)[-1]
        return stem.startswith("test_") or stem.endswith("_test.py")

    test_touched = [p for p in touched if is_test(p)]
    if test_touched:
        return test_touched, "patch_touches_tests"
    stems: list[str] = []
    for path in touched:
        stem = path.rsplit("/", 1)[-1].removesuffix(".py")
        if stem not in stems:
            stems.append(stem)
    return stems, ("patch_derived_stems" if stems else "default")


def replay_one(record: dict, work_root: Path, *, instance: dict | None = None,  # noqa: C901
               image_mode: str = "auto") -> dict:
    """Replay one trajectory; any unexpected exception becomes a failed record
    that still carries the evidence written before the failure, never a crashed
    batch (one broken repo must not lose the rest).

    The row is claimed on disk first, so the same work root can be consumed by
    more than one process without executing a row twice.

    ``image_mode`` chooses the environment path explicitly: ``official`` needs
    the instance's image_name and is the faithful env, ``mirror`` always uses
    the generic base image plus the repo's own dependency ladder, which costs
    tens of megabytes instead of 3-4 GB of per-instance pulls. ``auto`` keeps
    the original behaviour (official when metadata has an image).
    """
    traj_id = str(record.get("traj_id") or "unknown")
    task_dir = work_root / traj_id.replace("/", "_")
    claim = claim_row(task_dir)
    if not claim["claimed"]:
        return {"task_id": traj_id, "status": "skipped",
                "reason": "row claimed by another writer",
                "held_by": claim.get("held_by"), "row_dir": str(task_dir)}
    bump_attempts(task_dir, traj_id=traj_id, stage="claimed", pid=os.getpid(),
                  instance_id=record.get("instance_id"),
                  reclaimed_stale=bool(claim.get("reclaimed_stale")))
    wants_official = image_mode == "official" or (image_mode == "auto"
                                                  and bool((instance or {}).get("image_name")))
    try:
        if wants_official and not (instance or {}).get("image_name"):
            result = {"task_id": traj_id, "status": "setup_failed",
                      "reason": "official image mode requested but the instance metadata "
                                "carries no image_name",
                      "retryable": False, "row_dir": str(task_dir)}
        elif wants_official:
            result = _replay_one_official(record, work_root, instance=instance)
        else:
            result = _replay_one_inner(record, work_root, instance=instance)
        if result.get("status") != "skipped":
            result["image_mode"] = "official" if wants_official else "mirror"
    except subprocess.TimeoutExpired as exc:
        result = _failed_record(task_dir, traj_id, f"podman call timed out: {exc}")
        result["retryable"] = True
    except Exception as exc:  # noqa: BLE001 - fail closed per-row, keep the batch alive
        reason = f"{type(exc).__name__}: {exc}"
        result = _failed_record(task_dir, traj_id, reason)
        result["retryable"] = setup_failure_is_retryable(reason)
    if result.get("status") == "setup_failed":
        result.setdefault("retryable", setup_failure_is_retryable(str(result.get("reason") or "")))
    result["attempts"] = (read_json(row_state_path(task_dir)) or {}).get("attempts", 1)
    return finalize_row(task_dir, result)


def _ensure_official_image(image: str) -> str:
    """Pull the official SWE-smith instance image once (its env spec: conda
    testbed env, bugged repo at /testbed, all test deps installed).

    The pull is bounded and its failure is reported with the real output tail,
    so an unreachable/oversized image is recorded as a setup failure of that
    row instead of hanging the batch.
    """
    if not image.startswith(("docker.io/", "localhost/", "quay.io/")):
        image = "docker.io/" + image
    rc, _ = run_podman(["image", "exists", image], timeout=120)
    if rc == 0:
        return image
    rc, out = run_podman(["pull", image], timeout=1800)
    if rc != 0:
        raise RuntimeError(f"podman pull failed for {image}: {out[-400:]}")
    return image


def _replay_one_official(record: dict, work_root: Path, *, instance: dict) -> dict:
    """Red->green replay inside the OFFICIAL SWE-smith instance image.

    The image is the per-instance environment spec itself (conda ``testbed``
    env with every test dependency, bugged repo at /testbed) -- the same
    image the upstream harness evaluates with. Protocol is identical to the
    mirror path: pre-patch tests must fail (red) in a pristine container,
    then the trajectory's patch is applied and the same tests must pass
    (green), both with network disabled.
    """
    traj_id = str(record.get("traj_id") or "unknown")
    instance_id = str(record.get("instance_id") or "")
    repo = str(instance.get("repo") or "")
    base_commit = str(instance.get("base_commit") or parsed_or_blank(instance_id))
    patch = record.get("patch") or ""
    if not patch.strip():
        return {"task_id": traj_id, "status": "skipped", "reason": "empty patch"}
    f2p = instance.get("FAIL_TO_PASS") or instance.get("fail_to_pass") or []
    if isinstance(f2p, str):
        try:
            f2p = json.loads(f2p)
        except ValueError:
            f2p = [f2p]
    if not f2p:
        return {"task_id": traj_id, "status": "skipped",
                "reason": "official-image path requires instance FAIL_TO_PASS"}

    task_dir = work_root / traj_id.replace("/", "_")
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "ws").mkdir(exist_ok=True)
    events: list[dict] = []
    log: list[dict] = []

    def observe(kind: str, action: dict, returncode: int, output: str, *, tests: int | None = None) -> None:
        event = {"kind": kind, "action": action, "observation": (output or "")[-2000:],
                 "returncode": returncode, "verdict": "verified_good" if returncode == 0 else "verified_bad"}
        if tests is not None:
            event["tests_executed"] = tests
        events.append(event)
        log.append({"command": action.get("command"), "returncode": returncode,
                    "output_tail": (output or "")[-4000:]})
        _persist_log(task_dir, log)

    id_segments = instance_id.split(".")
    mirror_repo = (".".join(id_segments[:2]) if len(id_segments) >= 2 else None)
    if not mirror_repo:
        return {"task_id": traj_id, "status": "skipped",
                "reason": "official-image path needs a parseable instance id (bug-branch name)"}
    branch_candidates: list[str] = []
    for k in range(len(id_segments), 1, -1):
        cand = ".".join(id_segments[:k])
        if cand not in branch_candidates:
            branch_candidates.append(cand)
    owner, rest = mirror_repo.split("__", 1)
    upstream_repo = f"{owner}/{rest.rsplit('.', 1)[0]}"
    upstream_commit = rest.rsplit(".", 1)[1]
    f2p_paths = sorted({str(t).split("::")[0] for t in f2p
                        if isinstance(t, str) and "::" in str(t)
                        and str(t).split("::")[0].endswith(".py")})
    # pr_* instances: the F2P test files live at the PR head (the PR itself
    # added/changed them), not at the pre-PR base commit. Procedural bug
    # types keep upstream tests unchanged, so the base commit is correct.
    pr_match = re.search(r"(?:^|\.)pr_(\d+)(?:\.|$)", instance_id)
    fetch_ref = (f"refs/pull/{pr_match.group(1)}/head" if pr_match
                 else (resolve_full_sha(upstream_repo, upstream_commit) or upstream_commit))

    image = str(instance["image_name"])
    tests = ("python -m pytest -q -p no:cacheprovider "
             + " ".join(shlex.quote(str(t)) for t in f2p[:80]))
    test_source = ("instance_FAIL_TO_PASS" if len(f2p) <= 80
                   else "instance_FAIL_TO_PASS_first80")

    def phase_cmd(network: bool, tail: list[str], *, base_image: str | None = None,
                  mount: str | None = None) -> list[str]:
        args = ["run", "--rm", "--pull", "never",
                "--memory", MEMORY_LIMIT, "--cpus", CPUS, "--pids-limit", str(PIDS_LIMIT),
                *([] if network else ["--network", "none"])]
        if mount:
            args += ["-v", mount]
        return [*args, base_image or image, "bash", "-lc"] + tail

    # ---- phase A: materialize the red (bugged) state, network ON ----------
    # The official image ships the env (conda testbed, deps) with the
    # UNBUGGED tree: F2P tests pass in it as-is (verified). The bugged source
    # state comes from the per-instance mirror branch (github.com/swesmith/
    # <owner>__<name>.<commit>, branch = the instance id; its tip commit is
    # the injected-bug tree, e.g. "Remove F2P Tests"). We overlay that branch
    # onto /testbed, restore exactly the F2P test files from the ORIGINAL
    # repo at the embedded base commit, then commit the container so the
    # state persists for the network-off phases below.
    setup_cmd_parts = [
        "cd /testbed",
        f"(git remote add bug https://github.com/swesmith/{shlex.quote(mirror_repo)}.git 2>/dev/null || true)",
        ("ok=0; for b in " + " ".join(shlex.quote(c) for c in branch_candidates)
         + "; do if git fetch -q --depth 1 bug \"refs/heads/$b\" 2>/dev/null; then "
         + "git checkout -q -B bugstate FETCH_HEAD && ok=1 && break; fi; done; [ \"$ok\" = 1 ]"),
    ]
    if f2p_paths:
        setup_cmd_parts.append(
            f"(git remote add upstream https://github.com/{shlex.quote(upstream_repo)}.git 2>/dev/null || true)"
            f" && git fetch -q --depth 1 upstream {shlex.quote(fetch_ref)}"
            f" && git checkout -q FETCH_HEAD -- "
            + " ".join(shlex.quote(p) for p in f2p_paths[:60]))
    setup_cmd = " && ".join(setup_cmd_parts)
    setup_evidence = (f"overlay bug branch {branch_candidates[0]} + restore "
                      f"{len(f2p_paths)} F2P test files from {upstream_repo}@{fetch_ref[:12]}")
    # Content-addressed setup key: identical setup demands reuse one image, so
    # a resumed batch (or a second row of the same instance) skips setup.
    setup_key = digest({"image": image, "branches": branch_candidates,
                        "f2p_paths": f2p_paths, "fetch_ref": fetch_ref, "setup": setup_cmd})
    red_image = red_image_tag(instance_id, setup_key)
    reused = run_podman(["image", "exists", red_image], timeout=120)[0] == 0
    if reused:
        observe("tool", {"tool": "run_command",
                         "command": f"reuse cached red image {red_image}"}, 0,
                f"setup already committed for this instance+setup key ({setup_evidence})")
        update_row_state(task_dir, stage="red_state_reused", red_image=red_image,
                         setup_key=setup_key)
    else:
        image = _ensure_official_image(image)
        cname = f"chowder-setup-{uuid4().hex[:10]}"
        rc_s, out_s = run_podman([
            "run", "-d", "--name", cname, "--pull", "never",
            "--memory", MEMORY_LIMIT, "--cpus", CPUS, "--pids-limit", str(PIDS_LIMIT),
            image, "bash", "-c", "sleep 900"], timeout=STEP_TIMEOUT)
        observe("tool", {"tool": "run_command",
                         "command": f"start setup container {cname} ({image})"}, rc_s, out_s)
        if rc_s != 0:
            return _finish(task_dir, traj_id, events, log, "setup_failed",
                           reason=f"setup container would not start: {out_s[-300:]}",
                           retryable=True,
                           instance_id=instance_id, repo=repo, base_commit=base_commit,
                           test_cmd=tests, test_source=test_source)
        update_row_state(task_dir, stage="setup_overlay", red_image=red_image,
                         setup_key=setup_key)
        rc_a, out_a = run_podman(["exec", cname, "bash", "-lc", setup_cmd],
                                 timeout=SETUP_TIMEOUT)
        observe("tool", {"tool": "run_command", "command": setup_evidence}, rc_a, out_a)
        rc_cm, out_cm = run_podman(["commit", cname, red_image], timeout=STEP_TIMEOUT)
        observe("tool", {"tool": "run_command", "command": f"commit red state -> {red_image}"},
                rc_cm, out_cm)
        _best_effort(["rm", "-f", cname])
        if rc_a != 0 or rc_cm != 0:
            # An overlay that failed while the host was saturated is retryable;
            # one that failed on content (bad branch, missing tests) is not.
            return _finish(task_dir, traj_id, events, log, "setup_failed",
                           reason=(f"overlay rc={rc_a} commit rc={rc_cm}: {out_a[-300:]}"),
                           retryable=(rc_cm != 0 or setup_failure_is_retryable(out_a)),
                           instance_id=instance_id, repo=repo, base_commit=base_commit,
                           test_cmd=tests, test_source=test_source)
        update_row_state(task_dir, stage="red_state_ready", red_image=red_image,
                         setup_key=setup_key)

    try:
        # ---- phase B: red (committed bug state, network off) -------------- 
        update_row_state(task_dir, stage="pre_patch_tests")
        rc0, out0 = run_podman(
            phase_cmd(False, [f"cd /testbed && {tests}"], base_image=red_image),
            timeout=TEST_TIMEOUT)
        executed0 = parse_test_summary(out0)
        observe("test", {"tool": "run_tests", "command": tests, "phase": "pre_patch"},
                rc0, out0, tests=executed0)
        failure_reproduced = rc0 != 0 and executed0 > 0

        # ---- phase C: apply the patch, green (fresh container, network off) ----
        write_patch_file(task_dir / "ws" / "patch.diff", patch)
        mount = f"{task_dir / 'ws'}:/work"
        # Apply AND run the tests in ONE container: each podman run is a
        # fresh --rm container, so a patch applied in its own container
        # would be discarded before the post-patch tests ever start.
        # --3way uses the recorded blob ids for a merge-based apply, falling
        # back to plain context apply when blobs are absent; the patch
        # content is never altered, and the concrete APPLY_RC is recorded.
        apply_try = ("( git apply --3way --whitespace=nowarn /work/patch.diff"
                     " || git apply --whitespace=nowarn /work/patch.diff )")
        update_row_state(task_dir, stage="post_patch_tests")
        rc_c, out_c = run_podman(phase_cmd(False, [
            f"cd /testbed && {apply_try} ; echo APPLY_RC=$? ; {tests}"],
            base_image=red_image, mount=mount), timeout=TEST_TIMEOUT)
        m = re.search(r"APPLY_RC=(\d+)", out_c or "")
        apply_rc = int(m.group(1)) if m else 1
        observe("tool", {"tool": "apply_patch"}, apply_rc, out_c)
        patch_applied = apply_rc == 0
        executed2 = parse_test_summary(out_c)
        observe("test", {"tool": "run_tests", "command": tests, "phase": "post_patch"},
                rc_c, out_c, tests=executed2)
    finally:
        # The red image is the instance's cached setup state: it is reused by
        # other rows of the same instance and by resumed batches, so it is NOT
        # deleted here (see --prune-red-images for explicit housekeeping).
        update_row_state(task_dir, stage="scored", red_image=red_image)

    # Short-circuit order matters: rc_c is the combined apply+test container's
    # return code, meaningful only once the apply itself succeeded.
    verified = failure_reproduced and patch_applied and rc_c == 0 and executed2 > 0
    if verified:
        status = "verified"
    elif not patch_applied:
        # The patch never fit this base state: that is a pairing/base defect
        # (upstream data), not a failed repair, and must not be counted as one.
        status = "patch_did_not_fit"
    else:
        status = "not_green"
    return _finish(task_dir, traj_id, events, log, status,
                   returncode=rc_c if patch_applied else 1, apply_rc=apply_rc,
                   tests_executed=executed2, failure_reproduced=failure_reproduced,
                   instance_id=instance_id, repo=repo, base_commit=base_commit,
                   test_cmd=tests, test_source=test_source,
                   setup_reused=reused, red_image=red_image)


_FULL_SHA_CACHE: dict[tuple[str, str], str | None] = {}


def resolve_full_sha(upstream_repo: str, short_sha: str) -> str | None:
    """Expand a short commit SHA via the GitHub API (git fetch needs full SHAs;
    short ones are never valid fetch refs). Cached per (repo, short).
    Unresolvable -> None (caller falls back to the raw short, recorded as such).
    """
    key = (upstream_repo, short_sha)
    if key in _FULL_SHA_CACHE:
        return _FULL_SHA_CACHE[key]
    try:
        import urllib.request
        req = urllib.request.Request(
            f"https://api.github.com/repos/{upstream_repo}/commits/{short_sha}",
            headers={"User-Agent": "chowder-replay-audit", "Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            full = str(json.load(resp).get("sha") or "") or None
    except Exception:  # noqa: BLE001 - network/absent commit: recorded, replay proceeds
        full = None
    _FULL_SHA_CACHE[key] = full
    return full


def _replay_one_inner(record: dict, work_root: Path, *, instance: dict | None = None) -> dict:
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
        _persist_log(task_dir, log)

    # ---- phase A: setup (network on, no secrets in the container env) ----
    # SWE-smith publishes per-instance env mirrors at github.com/swesmith/
    # <owner>__<name>.<commit7>.git with ONE BRANCH PER INSTANCE (branch tip =
    # base commit + injected bug), verified against ls-remote. Checking out
    # the embedded commit hash fails on these mirrors; the instance-named
    # branch is the real environment spec. Trajectory ids may carry extra
    # dot-suffix segments, so checkout falls back through progressively
    # shorter id prefixes.
    id_segments = instance_id.split(".")
    mirror_repo = ".".join(id_segments[:2]) if len(id_segments) >= 2 else None
    branch_candidates: list[str] = []
    if mirror_repo:
        for k in range(len(id_segments), 1, -1):
            cand = ".".join(id_segments[:k])
            if cand not in branch_candidates:
                branch_candidates.append(cand)
    if mirror_repo:
        update_row_state(task_dir, stage="setup_clone", instance_id=instance_id)
        checkout_cmd = " || ".join(
            f"git checkout -q {shlex.quote(c)}" for c in branch_candidates)
        rc, out = run_podman(_container_cmd(True, work_mount) + [
            f"git clone -q https://github.com/swesmith/{shlex.quote(mirror_repo)}.git /work/repo"
            f" && cd /work/repo && ({checkout_cmd})"],
            timeout=SETUP_TIMEOUT)
        observe("tool", {"tool": "run_command",
                         "command": f"clone swesmith/{mirror_repo} + checkout {branch_candidates[0]}"},
                rc, out)
        if rc != 0:
            return _finish(task_dir, traj_id, events, log, "setup_failed",
                           reason=f"mirror clone/checkout rc={rc}: {out[-300:]}",
                           retryable=setup_failure_is_retryable(out),
                           instance_id=instance_id, repo=repo, base_commit=base_commit)
    else:
        # No parseable instance id: fall back to upstream repo @ parsed commit.
        rc, out = run_podman(_container_cmd(True, work_mount) + [
            f"git clone -q https://github.com/{shlex.quote(repo)}.git /work/repo"
            f" && cd /work/repo && git checkout -q {shlex.quote(base_commit)}"],
            timeout=SETUP_TIMEOUT)
        observe("tool", {"tool": "run_command",
                         "command": f"clone {repo}@{base_commit[:12]}"}, rc, out)
        if rc != 0:
            return _finish(task_dir, traj_id, events, log, "setup_failed",
                           reason=f"upstream clone rc={rc}: {out[-300:]}",
                           retryable=setup_failure_is_retryable(out),
                           instance_id=instance_id, repo=repo, base_commit=base_commit)

    update_row_state(task_dir, stage="setup_install")
    rc, out = run_podman(_container_cmd(True, work_mount) + [SETUP_INSTALL_LADDER],
                         timeout=SETUP_TIMEOUT)
    observe("tool", {"tool": "run_command",
                     "command": "install: editable + [test] extras / requirements-test.txt (SWE-smith spec)"},
            rc, out)

    # ---- test-target selection (evidence-based, recorded) ----------------
    f2p = (instance or {}).get("FAIL_TO_PASS") or (instance or {}).get("fail_to_pass") or []
    if isinstance(f2p, str):
        try:
            f2p = json.loads(f2p)
        except ValueError:
            f2p = [f2p]

    # ---- restore the F2P test files (SWE-smith env spec) -----------------
    # The env mirrors deliberately delete the verification tests (the tip
    # commit is "Remove F2P Tests"); the upstream harness restores them at
    # eval time. Replay must do the same: fetch exactly the F2P test paths
    # from the ORIGINAL repo at the embedded base commit, then run them.
    if mirror_repo and f2p:
        owner, rest = mirror_repo.split("__", 1)
        upstream_repo = f"{owner}/{rest.rsplit('.', 1)[0]}"
        upstream_commit = rest.rsplit(".", 1)[1]
        f2p_paths = sorted({str(t).split("::")[0] for t in f2p
                            if isinstance(t, str) and "::" in str(t)
                            and str(t).split("::")[0].endswith(".py")})
        if f2p_paths:
            fetch_ref = resolve_full_sha(upstream_repo, upstream_commit) or upstream_commit
            restore = (
                f"cd /work/repo && (git remote add upstream https://github.com/{shlex.quote(upstream_repo)}.git || true)"
                f" && git fetch -q --depth 1 upstream {shlex.quote(fetch_ref)}"
                f" && git checkout -q FETCH_HEAD -- "
                + " ".join(shlex.quote(p) for p in f2p_paths[:60])
            )
            rc_r, out_r = run_podman(_container_cmd(True, work_mount) + [restore],
                                     timeout=SETUP_TIMEOUT)
            observe("tool", {"tool": "run_command",
                             "command": f"restore {len(f2p_paths)} F2P test files from {upstream_repo}@{fetch_ref[:12]}"},
                    rc_r, out_r)
    targets, test_source = derive_test_targets(patch)
    if f2p and (instance or {}).get("exact_match"):
        tests = (PYTEST + " -q -p no:cacheprovider "
                 + " ".join(shlex.quote(str(t)) for t in f2p[:80]))
        test_source = ("instance_FAIL_TO_PASS" if len(f2p) <= 80
                       else "instance_FAIL_TO_PASS_first80")
    elif test_source == "patch_touches_tests":
        tests = (PYTEST + " -q -p no:cacheprovider "
                 + " ".join(shlex.quote(t) for t in targets))
    elif test_source == "patch_derived_stems":
        clauses: list[str] = []
        for s in targets[:5]:
            clauses.append(f"-name 'test_{s}.py'")
            clauses.append(f"-name '{s}_test.py'")
        name_args = " -o ".join(clauses)
        rc_r, out_r = run_podman(_container_cmd(True, work_mount) + [
            f"cd /work/repo && find . -type f \\( {name_args} \\) | head -20"],
            timeout=STEP_TIMEOUT)
        observe("tool", {"tool": "run_command",
                         "command": f"resolve test modules for {targets[:5]}"}, rc_r, out_r)
        resolved = [ln.strip().lstrip("./") for ln in out_r.splitlines()
                    if ln.strip().endswith(".py")]
        if resolved:
            tests = (PYTEST + " -q -p no:cacheprovider "
                     + " ".join(shlex.quote(t) for t in resolved[:10]))
        else:
            tests = PYTEST + " -q -p no:cacheprovider"
            test_source = "default_full_suite"
    else:
        tests = str((instance or {}).get("test_cmd") or (PYTEST + " -q -p no:cacheprovider"))

    # ---- phase B: reproduce the failure (network off, patch NOT applied) ----
    update_row_state(task_dir, stage="pre_patch_tests")
    rc0, out0 = run_podman(_container_cmd(False, work_mount) + [
        f"cd /work/repo && {tests}"], timeout=TEST_TIMEOUT)
    executed0 = parse_test_summary(out0)
    observe("test", {"tool": "run_tests", "command": tests, "phase": "pre_patch"},
            rc0, out0, tests=executed0)
    failure_reproduced = rc0 != 0 and executed0 > 0

    # ---- phase C: apply the patch, verify green (network off) ----
    write_patch_file(task_dir / "ws" / "patch.diff", patch)
    # Apply AND run the tests in ONE container (fresh --rm containers would
    # otherwise discard the applied patch before the tests start).
    update_row_state(task_dir, stage="post_patch_tests")
    rc_c, out_c = run_podman(_container_cmd(False, work_mount) + [
        f"cd /work/repo && (( git apply --3way --whitespace=nowarn /work/patch.diff"
        f" || git apply --whitespace=nowarn /work/patch.diff ) ; echo APPLY_RC=$?"
        f" ; {tests})"], timeout=TEST_TIMEOUT)
    m = re.search(r"APPLY_RC=(\d+)", out_c or "")
    apply_rc = int(m.group(1)) if m else 1
    observe("tool", {"tool": "apply_patch"}, apply_rc, out_c)
    patch_applied = apply_rc == 0
    executed2 = parse_test_summary(out_c)
    observe("test", {"tool": "run_tests", "command": tests, "phase": "post_patch"},
            rc_c, out_c, tests=executed2)
    verified = failure_reproduced and patch_applied and rc_c == 0 and executed2 > 0
    if verified:
        status = "verified"
    elif not patch_applied:
        status = "patch_did_not_fit"
    else:
        status = "not_green"
    return _finish(task_dir, traj_id, events, log, status,
                   returncode=rc_c if patch_applied else 1, apply_rc=apply_rc,
                   tests_executed=executed2, failure_reproduced=failure_reproduced,
                   instance_id=instance_id, repo=repo, base_commit=base_commit,
                   test_cmd=tests, test_source=test_source)


def _finish(task_dir: Path, traj_id: str, events: list[dict], log: list[dict],
            status: str, *, returncode: int = 1, tests_executed: int = 0,
            failure_reproduced: bool = False, instance_id: str = "",
            repo: str = "", base_commit: str = "", test_cmd: str = "",
            test_source: str = "", apply_rc: int | None = None,
            reason: str = "", retryable: bool = False,
            setup_reused: bool | None = None, red_image: str = "") -> dict:
    _persist_log(task_dir, log)
    record = {"task_id": traj_id, "repository": repo, "task": f"repair {instance_id}",
              "events": events,
              "failure_reproduced": failure_reproduced,
              "test_cmd": test_cmd, "test_source": test_source,
              "claimed_resolved_source": "self-reported dataset flag (not evidence)"}
    if reason:
        record["reason"] = reason
    if retryable:
        record["retryable"] = True
    if setup_reused is not None:
        record["setup_reused"] = setup_reused
    if red_image:
        record["red_image"] = red_image
    # apply_rc separates "patch did not apply" (upstream pairing / base mismatch)
    # from "patch applied but tests still red" (genuine non-repair).
    if apply_rc is not None:
        record["apply_rc"] = apply_rc
    if status == "verified":
        record["verification"] = {"method": "sandbox_replay", "returncode": returncode,
                                  "tests_executed": tests_executed,
                                  "trace_sha256": digest(events)}
    record["status"] = status
    record["replay_log"] = str(task_dir / "replay_log.json")
    return record


def parsed_or_blank(instance_id: str) -> str:
    repo, commit = parse_instance_id(instance_id)
    return commit or ""


def patch_paths(patch: str) -> list[str]:
    paths: list[str] = []
    for m in re.finditer(r"^diff --git a/(\S+) b/(\S+)$", patch, re.M):
        for p in (m.group(1), m.group(2)):
            if p not in paths and p != "/dev/null":
                paths.append(p)
    return paths


# ---------------------------------------------------------------- prescreen
#: Support dirs whose names never imply the module under test.
_SUPPORT_DIRS = {"tests", "test", "docs", "doc", "examples", "example", "scripts",
                 "contrib", "benchmarks", "util", "utils"}


def _norm_token(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def congruence_verdict(patch: str, f2p: list[str]) -> dict:
    """Does this patch belong to the instance its row claims? (no network)

    Ported from the pairing audit whose verdicts matched the hand-verified
    cases exactly (gTTS.pr_440 congruent; moto.pr_7456 a patch for a different
    module than its FAIL_TO_PASS tests). A mispaired row can only ever produce
    a ``not_green`` that says nothing about the model, so such rows are routed
    out of the repair corpus before they cost container time.
    """
    paths = patch_paths(patch)
    comps: set[str] = set()
    for path in paths:
        for comp in path.split("/"):
            if comp:
                comps.add(_norm_token(comp.rsplit(".", 1)[0] if "." in comp else comp))
    f2p_paths = sorted({str(t).split("::")[0] for t in f2p if "::" in str(t)})
    tokens: set[str] = set()
    for path in f2p_paths:
        parts = [p for p in path.split("/") if p]
        if not parts:
            continue
        stem = parts[-1].rsplit(".", 1)[0]
        stem = re.sub(r"_test$", "", re.sub(r"^test_", "", stem))
        if _norm_token(stem) and _norm_token(stem) not in _SUPPORT_DIRS:
            tokens.add(_norm_token(stem))
        if len(parts) >= 2 and _norm_token(parts[-2]) not in _SUPPORT_DIRS:
            tokens.add(_norm_token(parts[-2]))
    module_hit = bool(tokens & comps)
    test_hit = bool(set(paths) & set(f2p_paths))
    return {"verdict": "congruent" if (test_hit or module_hit) else "same_repo_other_module",
            "module_hit": module_hit, "test_file_hit": test_hit,
            "module_tokens": sorted(tokens), "patch_paths": paths}


def load_fit_ledger(path: Path) -> dict[str, dict]:
    """instance_id -> patch-fit verdict, from the host-side fit diagnosis.

    A patch that cannot apply to the instance's own base state (upstream
    pairing/base defect) must never be replayed as if the model failed.
    """
    ledger: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            key = str(row.get("instance_id") or "")
            if key:
                ledger[key] = row
    return ledger


def prescreen_row(record: dict, instance: dict | None, ledger: dict[str, dict], *,
                  require_congruence: bool, recover_derived_tests: bool = False) -> dict | None:
    """Route a row that cannot yield usable repair evidence out of the queue.

    Returns the terminal record to write instead of replaying, a recovery
    request (``recovery: derived_tests``) when the caller asked for recovery,
    or None when the row should run as-is.
    """
    instance_id = str(record.get("instance_id") or "")
    fit = ledger.get(instance_id)
    if fit and not str(fit.get("verdict") or "").startswith("fits_"):
        return {"task_id": record.get("traj_id"), "status": "patch_did_not_fit",
                "reason": f"fit ledger: {fit.get('verdict')}",
                "fit_evidence": {k: fit.get(k) for k in
                                 ("verdict", "fits_branch", "first_error")},
                "retryable": False}
    if require_congruence and instance and instance.get("FAIL_TO_PASS"):
        verdict = congruence_verdict(record.get("patch") or "", instance["FAIL_TO_PASS"])
        if verdict["verdict"] != "congruent":
            if recover_derived_tests:
                # Recovery: the patch is valid for its branch, but the tests in
                # this row's metadata belong to another module. Drop them and
                # let the mirror path derive the test surface from the patch
                # itself, so the row yields recoverable (weaker) evidence
                # instead of a mispaired not_green.
                return {"recovery": "derived_tests", "congruence": verdict}
            return {"task_id": record.get("traj_id"), "status": "mispaired",
                    "reason": "patch does not touch the module its FAIL_TO_PASS tests exercise",
                    "congruence": verdict, "retryable": False}
    return None


def merge_results(out_path: Path) -> list[dict]:
    """Read a results JSONL with the last row per task_id winning.

    The file is append-only so a killed batch loses nothing, and parallel
    writers cannot corrupt it; deduplication happens at read time.
    """
    rows: dict[str, dict] = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[str(row.get("task_id"))] = row
    return list(rows.values())


def summarize_results(results: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for row in results:
        status = str(row.get("status"))
        counts[status] = counts.get(status, 0) + 1
    return {"rows": len(results), **counts}


def write_summary(out_path: Path) -> dict:
    """Cumulative summary derived from the results file (crash-safe)."""
    summary = summarize_results(merge_results(out_path))
    write_json(out_path.parent / "replay_summary.json", summary)
    return summary


def prune_red_images() -> list[str]:
    """Delete ONLY the harness's own cached red-state images.

    The filter is the harness namespace (``localhost/chowder-replay-red-``);
    a broad image cleanup once took down unrelated host resources, so nothing
    outside this namespace is ever touched.
    """
    rc, out = run_podman(["images", "--format", "{{.Repository}}:{{.Tag}}"], timeout=120)
    if rc != 0:
        return []
    targets = [ln.strip() for ln in out.splitlines()
               if ln.strip().startswith("localhost/chowder-replay-red-")]
    for tag in targets:
        _best_effort(["rmi", "-f", tag])
    return targets


def replay_batch(input_path: Path, work_root: Path, out_path: Path,
                 instances_path: Path | None = None, *, limit: int = 5,
                 max_attempts: int = DEFAULT_MAX_ATTEMPTS, fresh: bool = False,
                 require_congruence: bool = False,
                 recover_derived_tests: bool = False,
                 fit_ledger: Path | None = None,
                 image_mode: str = "auto") -> dict:
    """Consume the replay queue: skip answered rows, resume abandoned ones,
    retry only host-load failures, append each result the moment it exists.

    Cheap by construction: an answered row costs one file read, setup state is
    a content-addressed image, and a row whose patch cannot fit its base is
    routed out before any container starts.
    """
    if subprocess.run(["podman", "info"], capture_output=True).returncode != 0:
        raise RuntimeError("podman is not available; refusing to fabricate replay evidence")
    instances: dict[str, dict] = {}
    if instances_path:
        for line in instances_path.read_text(encoding="utf8").splitlines():
            if line.strip():
                row = json.loads(line)
                instances[row.get("instance_id")] = row
    rows = [json.loads(l) for l in input_path.read_text(encoding="utf8").splitlines() if l.strip()]
    ledger = load_fit_ledger(fit_ledger) if fit_ledger else {}
    plan = plan_rows(rows, work_root, max_attempts=max_attempts)
    if fresh:
        plan = [{**entry, "decision": "run"} for entry in plan]
    skipped: dict[str, int] = {}
    attempted = 0
    ensure_base_image()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Append mode: a rerun adds to the evidence instead of erasing it.
    with out_path.open("a", encoding="utf8") as out:
        for row, entry in zip(rows, plan):
            if entry["decision"] != "run":
                key = f"skipped_{entry['decision']}"
                skipped[key] = skipped.get(key, 0) + 1
                continue
            if attempted >= limit:
                skipped["not_attempted"] = skipped.get("not_attempted", 0) + 1
                continue
            instance = instances.get(row.get("instance_id"))
            gated = prescreen_row(row, instance, ledger,
                                  require_congruence=require_congruence,
                                  recover_derived_tests=recover_derived_tests)
            if gated is not None and gated.get("recovery") == "derived_tests":
                recovered_instance = {**(instance or {}), "FAIL_TO_PASS": [],
                                      "image_name": None}
                result = replay_one(row, work_root, instance=recovered_instance,
                                    image_mode=image_mode)
                if str(result.get("status")) != "skipped":
                    result["recovered_from"] = "mispaired_metadata"
                    result["congruence"] = gated["congruence"]
                    traj_id = str(row.get("traj_id") or "unknown")
                    finalize_row(work_root / traj_id.replace("/", "_"), result)
                attempted += 1
            elif gated is not None:
                traj_id = str(row.get("traj_id") or "unknown")
                task_dir = work_root / traj_id.replace("/", "_")
                task_dir.mkdir(parents=True, exist_ok=True)
                update_row_state(task_dir, stage="prescreened", gated=True)
                result = finalize_row(task_dir, gated)
                attempted += 1
            else:
                result = replay_one(row, work_root, instance=instance, image_mode=image_mode)
                if str(result.get("status")) != "skipped":
                    attempted += 1
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            write_summary(out_path)
    merged = write_summary(out_path)
    summary = {"queue_rows": len(rows), "attempted_this_run": attempted,
               "result_rows": merged.get("rows"), **skipped,
               **{k: v for k, v in merged.items() if k != "rows"}}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="fetch_smith.py output")
    parser.add_argument("--work", type=Path, help="row state root (created per traj_id)")
    parser.add_argument("--out", type=Path, help="append-only results JSONL")
    parser.add_argument("--instances", type=Path, help="JSONL of task-instance metadata (repo, base_commit, test_cmd)")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS,
                        help="retry budget for host-load failures (never for final verdicts)")
    parser.add_argument("--fresh", action="store_true",
                        help="ignore previous row results and re-run the whole queue")
    parser.add_argument("--require-congruence", action="store_true",
                        help="route rows whose patch does not match their FAIL_TO_PASS module to 'mispaired'")
    parser.add_argument("--fit-ledger", type=Path,
                        help="fit verdicts per instance (host-side patch-fit diagnosis)")
    parser.add_argument("--recover-mispaired-derived-tests", action="store_true",
                        help="replay module-mispaired rows with tests derived from the patch, "
                             "labelled recovered_from=mispaired_metadata (weaker evidence)")
    parser.add_argument("--image-mode", choices=("auto", "official", "mirror"), default="auto",
                        help="environment source: official per-instance image (3-4 GB pull) "
                             "when available, or always the cheap mirror path")
    parser.add_argument("--plan-only", action="store_true",
                        help="print the run/skip/retry plan and exit (no podman, no containers)")
    parser.add_argument("--prune-red-images", action="store_true",
                        help="delete only the harness's cached red-state images, then exit")
    args = parser.parse_args()

    if args.prune_red_images:
        print(json.dumps({"removed": prune_red_images()}, indent=2))
        return
    if not (args.input and args.work):
        parser.error("--input and --work are required (or use --prune-red-images)")
    if args.plan_only:
        rows = [json.loads(l) for l in args.input.read_text(encoding="utf8").splitlines()
                if l.strip()]
        plan = plan_rows(rows, args.work, max_attempts=args.max_attempts)
        print(json.dumps({"rows": len(plan),
                          "decisions": summarize_results(
                              [{"status": e["decision"]} for e in plan])}, indent=2))
        for entry in plan:
            print(f"{entry['decision']:18} attempts={entry['attempts']} "
                  f"status={entry['status']} {entry['traj_id'][:60]}")
        return
    print(json.dumps(replay_batch(args.input, args.work, args.out, args.instances,
                                  limit=args.limit, max_attempts=args.max_attempts,
                                  fresh=args.fresh,
                                  require_congruence=args.require_congruence,
                                  recover_derived_tests=args.recover_mispaired_derived_tests,
                                  fit_ledger=args.fit_ledger,
                                  image_mode=args.image_mode), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
