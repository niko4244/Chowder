"""Independently prove the original ProgramBench 200-task scoring population.

Upstream v1.0.1 includes one TEST fixture, testorg__calculator.abc1234,
in its unit-test data directory. That is NOT a benchmark participant. The
genuine200 must contain precisely 248089 non-ignored behavioral tests.
This reads official published task metadata; it is NOT a model evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import yaml

SCHEMA = "chowder.original-programbench200-authentic-source-census.v1"
PINNED_RELEASE = "v1.0.1"
PINNED_COMMIT = "1fe64c87a1318998636850cead00b0db80e2d928"
UPSTREAM_UNIT_TEST_FIXTURE = "testorg__calculator.abc1234"
N = 200
N_SCORED = 248_089


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _unique_json(raw: bytes) -> dict:
    def unique(items):
        obj = {}
        for key, value in items:
            if key in obj:
                raise ValueError("duplicate original scorer JSON key")
            obj[key] = value
        return obj
    value = json.loads(raw, object_pairs_hook=unique,
                       parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    if not isinstance(value, dict):
        raise ValueError("original tests payload is not a dictionary")
    return value


def original_task(task_root: Path, name: str) -> dict:
    if not isinstance(name, str) or not name or "/" in name or "\\" in name:
        raise ValueError("original task identity is not a single safe directory")
    folder = task_root / name
    yaml_path, tests_path = folder / "task.yaml", folder / "tests.json"
    if (folder.is_symlink() or not folder.is_dir()
            or any(f.is_symlink() or not f.is_file()
                   or f.stat().st_size > 5_000_000 for f in (yaml_path, tests_path))):
        raise ValueError("missing, linked or oversized official task evidence")
    metadata = yaml.safe_load(yaml_path.read_bytes())
    data = _unique_json(tests_path.read_bytes())
    if (not isinstance(metadata, dict)
            or any(not isinstance(metadata.get(k), str) or not metadata[k]
                   for k in ("repository", "commit", "language"))
            or metadata.get("difficulty") is not None and
               not isinstance(metadata["difficulty"], str)
            or not isinstance(data.get("branches"), dict) or not data["branches"]):
        raise ValueError("malformed original test/repository metadata")
    all_tests = eligible = ignored_active = ignored_branches = 0
    for branch, info in data["branches"].items():
        if not isinstance(branch, str) or not isinstance(info, dict):
            raise ValueError("invalid original test branch metadata")
        tests = info.get("tests")
        ignore = info.get("ignored_tests") or []
        if (not isinstance(tests, list) or
                any(not isinstance(t, str) or not t for t in tests) or
                len(set(tests)) != len(tests) or
                not isinstance(ignore, list) or
                any(not isinstance(x, dict) or not isinstance(x.get("name"), str)
                    for x in ignore) or
                type(info.get("ignored", False)) is not bool):
            raise ValueError("original active/ignored task test identities not verified")
        omitted = {x["name"] for x in ignore}
        if not omitted.issubset(set(tests)):
            raise ValueError("ignored original test not found in its branch")
        all_tests += len(tests)
        if info.get("ignored"):
            ignored_branches += 1
        else:
            eligible += sum(t not in omitted for t in tests)
            ignored_active += len(omitted)
    return {
        "instance_id": name, "original_repository": metadata["repository"],
        "original_commit": metadata["commit"],
        "language": metadata["language"], "difficulty": metadata.get("difficulty"),
        "original_task_yaml_sha256": _sha(yaml_path),
        "original_tests_json_sha256": _sha(tests_path),
        "original_branch_count": len(data["branches"]),
        "ignored_branches": ignored_branches,
        "raw_behavioral_tests": all_tests, "scorable_behavioral_tests": eligible,
        "ignored_tests_on_active_branches": ignored_active,
    }


def audit(root: Path) -> dict:
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise ValueError("source must be the genuine official local checkout")
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                            check=True, capture_output=True, text=True, timeout=12).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                           check=True, capture_output=True, text=True, timeout=12).stdout
    if commit != PINNED_COMMIT or dirty:
        raise ValueError("original official tagged v1.0.1 source is not clean/pinned")
    tasks = root / "src" / "programbench" / "data" / "tasks"
    test_source = root / "tests" / "test_eval.py"
    if (tasks.is_symlink() or test_source.is_symlink() or
            not test_source.is_file() or
            UPSTREAM_UNIT_TEST_FIXTURE not in test_source.read_text(encoding="utf-8")):
        raise ValueError("independent upstream unit fixture exclusion proof missing")
    names = sorted(p.name for p in tasks.iterdir() if p.is_dir())
    if len(names) != 201 or UPSTREAM_UNIT_TEST_FIXTURE not in names:
        raise ValueError("unexpected number of original source task directories")
    excluded = original_task(tasks, UPSTREAM_UNIT_TEST_FIXTURE)
    if (excluded["original_repository"] != "testorg/calculator"
            or excluded["original_commit"] != "abc1234567890abcdef1234567890abcdef123456"
            or excluded["scorable_behavioral_tests"] != 3):
        raise ValueError("upstream original unit-test fixture changed")
    official = [original_task(tasks, n) for n in names
                if n != UPSTREAM_UNIT_TEST_FIXTURE]
    if (len(official) != N
            or sum(x["scorable_behavioral_tests"] for x in official) != N_SCORED
            or any(x["scorable_behavioral_tests"] <= 0 for x in official)):
        raise ValueError("genuine200 official scoring population is not complete")
    inventory_hash = hashlib.sha256(json.dumps(
        official, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        "schema": SCHEMA, "status": "SOURCE_BOUND_ORIGINAL_PROGRAMBENCH200_NOT_MODEL_SCORED",
        "official_repo": "https://github.com/facebookresearch/ProgramBench",
        "source_release": PINNED_RELEASE, "source_commit": commit,
        "upstream_fixture_test_source_sha256": _sha(test_source),
        "excluded_upstream_test_fixture": UPSTREAM_UNIT_TEST_FIXTURE,
        "source_task_directories_including_fixture": 201,
        "original_benchmark_tasks": N,
        "original_nonignored_behavioral_tests": N_SCORED,
        "original_task_manifest_sha256": inventory_hash,
        "source_bound_task_inventory": official,
        "original_model_submissions": 0, "independent_original_model_scores": None,
        "frozen_gen0_status": "BLOCKED", "chosen": None, "release_approved": False,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--official-checkout", type=Path, required=True)
    p.add_argument("--private-out", type=Path, required=True)
    args = p.parse_args()
    try:
        if (not args.private_out.is_absolute() or args.private_out.exists()
                or args.private_out.is_symlink()
                or not args.private_out.parent.is_dir()
                or args.private_out.suffix != ".json"):
            raise ValueError("exclusive private output path required")
        result = audit(args.official_checkout)
        args.private_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", "utf-8")
        print(json.dumps({k: v for k, v in result.items() if k != "source_bound_task_inventory"},
                         sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError,
            TypeError) as error:
        print(json.dumps({"status": "BLOCKED_ORIGINAL_PROGRAMBENCH_SOURCE",
                          "reason": type(error).__name__ + ": " + str(error)[:300],
                          "original_model_submissions": 0,
                          "frozen_gen0_status": "BLOCKED"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
