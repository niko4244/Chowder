"""CPU-only, research-only original guard.v3 AutomationBench WorldState replay.

Unlike receipt/aggregate reconciliation, this re-executes the official v1.0.6
assertion handlers on archived end states and *source-derived* task assertions
and initial states. No model, agent, tool action, or generated code is run.

Use the original AutomationBench Python 3.13 virtual environment. This audit
cannot authenticate historical HTTP bytes or promote a segmented original
guard.v3 research run into the frozen four-slot Gen-0 comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

from chowder_batch import reap_original600_guardv3_git_pinned_reader_20261027 as historical
from chowder_batch.reap_original600_guardv3_stable_snapshot_20261029 import stable_snapshot


OFFICIAL = Path("F:/reap-v2-public-automationbench-source-20261026")
OFFICIAL_COMMIT = "4a8e1061254004d9dac807054eed33fad7d1ff14"
SCHEMA = "chowder.reap-original600-guardv3-worldstate-independent-replay.v1"
DOMAINS = ("sales", "marketing", "operations", "support", "finance", "hr")


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def strict_json(raw: bytes | str) -> object:
    """Reject ambiguous duplicate-key JSON and nonfinite numbers in evidence."""
    def unique_pairs(pairs: list[tuple[str, object]]) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key in original official evidence: " + key)
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError("nonfinite JSON number in original official evidence: " + value)

    return json.loads(raw, object_pairs_hook=unique_pairs, parse_constant=reject_constant)


def _git(*args: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(OFFICIAL), *args], timeout=35,
        env={**os.environ, "GIT_NO_REPLACE_OBJECTS": "1"},
    )


def official_source_tree() -> tuple[str, set[str]]:
    """Check every tracked official source byte, not merely Git's HEAD label."""
    if (not OFFICIAL.is_dir() or OFFICIAL.is_symlink()
            or _git("rev-parse", "HEAD").decode().strip() != OFFICIAL_COMMIT
            or _git("rev-parse", "--show-object-format").decode().strip() != "sha1"
            or _git("status", "--porcelain=v1", "--untracked-files=all").strip()):
        raise ValueError("official v1.0.6 source commit or cleanliness mismatch")
    for forbidden in (".env", "sitecustomize.py", "usercustomize.py"):
        if (OFFICIAL / forbidden).exists():
            raise ValueError("official source has an import-time override")
    entries = _git("ls-tree", "-r", "-z", OFFICIAL_COMMIT, "--", "automationbench")
    tracked: set[str] = set()
    hashes = hashlib.sha256()
    for entry in entries.split(b"\0"):
        if not entry:
            continue
        header, relative_bytes = entry.split(b"\t", 1)
        mode, kind, object_sha = header.decode("ascii").split()
        relative = relative_bytes.decode("utf-8")
        if (mode not in ("100644", "100755") or kind != "blob"
                or not relative.startswith("automationbench/")
                or relative in tracked):
            raise ValueError("official source tree contains a redirected/duplicate entry")
        tracked.add(relative)
        path = OFFICIAL / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError("official source path is missing or redirected: " + relative)
        raw = path.read_bytes()
        # The official Windows checkout uses CRLF while the original Git
        # blobs are LF. Require equality with either the literal blob or the
        # exact CRLF->LF normalized blob; no other content drift is accepted.
        def blob_sha(data: bytes) -> str:
            return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()

        normalized = raw if blob_sha(raw) == object_sha else raw.replace(b"\r\n", b"\n")
        if blob_sha(normalized) != object_sha:
            raise ValueError("official tracked source bytes differ from Git: " + relative)
        hashes.update(relative_bytes + b"\0" + hashlib.sha256(normalized).digest())
    if len(tracked) < 100 or "automationbench/rubric/__init__.py" not in tracked:
        raise ValueError("official source tree inventory incomplete")
    return hashes.hexdigest(), tracked


def verify_imports(tracked: set[str]) -> None:
    """Reject shadowed official Python modules and untracked import sources."""
    root = OFFICIAL.resolve()
    found = 0
    for name, module in tuple(sys.modules.items()):
        if name != "automationbench" and not name.startswith("automationbench."):
            continue
        filename = getattr(module, "__file__", None)
        if filename is None:
            paths = getattr(module, "__path__", ())
            if any(Path(p).resolve() != root / name.replace(".", "/") for p in paths):
                raise ValueError("official namespace package shadowed: " + name)
            continue
        path = Path(filename).resolve()
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError("official module imported from foreign root: " + name) from exc
        if path.suffix != ".py" or relative not in tracked:
            raise ValueError("official module imported from untracked code: " + name)
        found += 1
    if found < 10:
        raise ValueError("official WorldState/assertion modules not loaded")


def checked_params(source: dict, saved: dict, *, name: str, index: int) -> bool:
    """Only permit the single observed official null-omission serialization."""
    if source == saved:
        return False
    if (name == "salesforce_field_not_changed" and index == 47
            and source.get("value", object()) is None
            and "value" not in saved
            and {k: v for k, v in source.items() if k != "value"} == saved):
        return True
    raise ValueError("official source assertion parameters differ from saved rubric")


def regrade_one(index: int, source: dict, info: dict, saved: dict,
                *, WorldState, partial_credit, task_contract_sha256) -> dict:
    """Recompute strict score and every assertion from canonical task inputs."""
    if (type(saved) is not dict or type(saved.get("end_state")) is not dict
            or type(saved.get("assertion_results")) is not list
            or type(saved.get("passed")) is not bool
            or type(saved.get("assertions_total")) is not int
            or type(saved.get("assertions_passed")) is not int
            or type(saved.get("score")) not in (int, float)
            or saved.get("name") != info.get("task_name")):
        raise ValueError("saved official WorldState row is malformed")
    if type(info.get("assertions")) is not list or type(info.get("initial_state")) is not dict:
        raise ValueError("canonical source task lacks assertions or initial state")
    world = WorldState.model_validate(saved["end_state"])
    if world.model_dump(mode="json") != saved["end_state"]:
        raise ValueError("saved end state is not a canonical WorldState roundtrip")
    state = {"info": info, "initial_state": info["initial_state"], "world": world}
    score = partial_credit(state)
    assertions = state.get("_assertion_results")
    recorded = saved["assertion_results"]
    if (type(score) is not float or score != saved["score"]
            or (score == 1.0) != saved["passed"]
            or type(assertions) is not list or len(assertions) != len(recorded)):
        raise ValueError("independently replayed WorldState score differs")
    null_omissions = 0
    for replayed, stored in zip(assertions, recorded):
        if (type(stored) is not dict
                or set(stored) != {"type", "params", "passed", "excluded"}
                or type(stored["passed"]) is not bool
                or type(stored["excluded"]) is not bool
                or type(stored["params"]) is not dict
                or (replayed["type"], replayed["passed"], replayed["excluded"])
                   != (stored["type"], stored["passed"], stored["excluded"])):
            raise ValueError("independently replayed assertion verdict differs")
        null_omissions += int(checked_params(replayed["params"], stored["params"],
                                            name=replayed["type"], index=index))
    scored = [a for a in assertions if not a["excluded"]]
    if (saved["assertions_total"] != len(scored)
            or saved["assertions_passed"] != sum(a["passed"] for a in scored)):
        raise ValueError("independently replayed assertion totals differ")
    contract_sha = task_contract_sha256(
        example_id=source["example_id"], prompt=source["prompt"], info=info,
    )
    return {"index": index, "task": saved["name"], "passed": saved["passed"],
            "score": score, "assertions": len(assertions),
            "null_param_omissions": null_omissions,
            "source_task_contract_sha256": contract_sha,
            "end_state_sha256": sha(canonical(saved["end_state"]))}


def replay() -> dict:
    historical._require_assertions_enabled()
    if sys.version_info[:2] != (3, 13):
        raise ValueError("official WorldState replay requires Python 3.13")
    if Path(sys.executable).resolve() != (OFFICIAL / ".venv/Scripts/python.exe").resolve():
        raise ValueError("official WorldState replay requires original benchmark virtualenv")
    if any(name == "automationbench" or name.startswith("automationbench.")
           for name in sys.modules):
        raise ValueError("fresh official interpreter required before WorldState imports")
    before_sha, tracked = official_source_tree()
    # Avoid preexisting ignored .pyc files: load original .py from Git-verified
    # paths into this fresh process without reading an existing bytecode cache.
    sys.pycache_prefix = str(Path(os.environ.get("TEMP", "C:/Windows/Temp")) /
                             ("reap-v3-no-pyc-" + uuid.uuid4().hex))
    sys.dont_write_bytecode = True
    from automationbench.domains import get_domain_dataset
    from automationbench.rubric import partial_credit
    from automationbench.schema.world import WorldState
    from automationbench.task_contract import task_contract_sha256

    verify_imports(tracked)
    with stable_snapshot(historical.ROOT, historical.REPO) as stable:
        owner_bytes = (stable / "owner.json").read_bytes()
        if sha(owner_bytes) != historical.OWNER_SHA:
            raise ValueError("original guard.v3 owner SHA mismatch")
        owner = strict_json(owner_bytes)
        if type(owner) is not dict:
            raise ValueError("original guard.v3 owner must be an object")
        tasks = owner.get("tasks")
        if type(tasks) is not list or len(tasks) != 600:
            raise ValueError("original guard.v3 owner task population invalid")
        runner, audit = historical.historical_modules()
        previous = audit.inspect(stable)
        completed = previous["completed_scored_tasks"]
        if (completed > 600 or previous["totals"]["strict_passes"] < 0
                or previous["frozen_gen0_status"] != "BLOCKED"
                or previous["chosen"] is not None
                or previous["release_approved"] is not False):
            raise ValueError("historical original score is not research-only")
        rows = []
        datasets = {}
        for index in range(completed):
            domain = DOMAINS[index // 100]
            if domain not in datasets:
                dataset = get_domain_dataset(domain)
                if len(dataset) != 100:
                    raise ValueError("official canonical domain does not contain 100 tasks")
                datasets[domain] = dataset
            source = datasets[domain][index % 100]
            info = strict_json(source["info"])
            expected = tasks[index]
            if (type(info) is not dict or str(source["example_id"]) != expected["id"]
                    or info.get("task_name") != expected["name"]):
                raise ValueError("canonical source task differs from original owner")
            path = stable / f"task-{index:03d}" / "public-automationbench.private.json"
            original = strict_json(path.read_bytes())
            if (type(original) is not dict or type(original.get("tasks")) is not list
                    or len(original["tasks"]) != 1):
                raise ValueError("original scored task is missing official WorldState")
            rows.append(regrade_one(index, source, info, original["tasks"][0],
                                    WorldState=WorldState, partial_credit=partial_credit,
                                    task_contract_sha256=task_contract_sha256))
        if sum(row["passed"] for row in rows) != previous["totals"]["strict_passes"]:
            raise ValueError("independent WorldState passes differ from historical scorer")
    after_sha, after_tracked = official_source_tree()
    if before_sha != after_sha or tracked != after_tracked:
        raise ValueError("official evaluator source changed during WorldState replay")
    verify_imports(tracked)
    return {"schema": SCHEMA, "status": "RESEARCH_ONLY_WORLDSTATE_REPLAY",
            "original_guard": "guard.v3", "historical_commit": historical.COMMIT,
            "historical_owner_sha256": historical.OWNER_SHA,
            "official_source_commit": OFFICIAL_COMMIT,
            "official_source_tree_sha256": before_sha,
            "official_source_tracked_files": len(tracked),
            "replayed_scored_tasks": len(rows),
            "replayed_strict_passes": sum(row["passed"] for row in rows),
            "replayed_assertions": sum(row["assertions"] for row in rows),
            "known_official_null_param_omissions": sum(row["null_param_omissions"] for row in rows),
            "per_item_replay_sha256": sha(canonical(rows)),
            "complete_600_research_replay": len(rows) == 600,
            "native_http_response_text_authenticated": False,
            "comparable_to_frozen_four_slot": False,
            "gen0_status": "BLOCKED", "chosen": None, "release_approved": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        print(json.dumps(replay(), indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (ValueError, OSError, TypeError, KeyError, subprocess.SubprocessError) as exc:
        print(json.dumps({"status": "REFUSED_ORIGINAL_WORLDSTATE_REPLAY",
                          "reason": str(exc)[:400], "gen0_status": "BLOCKED",
                          "chosen": None, "release_approved": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
