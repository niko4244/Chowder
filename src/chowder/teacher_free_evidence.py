"""Artifact-derived workflow state for the isolated teacher-free pilot.

Every state string this module returns is derived by reading an artifact and
*checking* it -- recomputing a digest, re-deriving a count, cross-referencing a
pin -- never from the mere existence of a file. The pilot's earlier screen
marked stages complete when a recipe or a replay summary happened to exist, so
a hand-written or half-finished file looked exactly like finished work. Here:

  verified   the artifact exists AND its recorded digests/claims re-check
  recorded   the artifact records a real run, but what it points at is not on
             this host, so it cannot be re-checked here (never green)
  pending    no artifact yet
  failed     the artifact exists and contradicts itself or its own digests

The module deliberately has no Textual dependency: the screen renders these
states, but the checking is testable in a CPU test matrix without a TUI extra.

Vocabulary of the pipeline it reads (see experiments/teacher_free_distill):

* ``sources.json``         catalog with a per-source approval decision
* ``**/manifest.json``     dataset manifest written by ``prepare.py``
* ``student_selection.json`` memory-plan record written by ``student.py``
* ``preflight_result.json``  CPU preflight written by ``preflight.py``
* ``condition_a_artifacts.json``  digest record of the completed Condition A
                           adapter (written by ``verify_condition_a.py``)
* ``**/replay_summary.json``  sandbox replay buckets written by ``replay_smith.py``
* ``**/comparison_*.json``  paired evaluation output (see ``evaluate.py``)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

WORKFLOW_STAGES = [
    "source_review", "data_collection", "validation", "student_selection",
    "training_preflight", "training", "evaluation", "results",
]

MANIFEST_FORMAT_PREFIX = "chowder-teacher-free-pilot-"
CONDITION_A_FORMAT = "chowder-teacher-free-condition-a/v1"

#: States whose color is green: an artifact was checked and held up. Anything
#: unrecognised stays yellow -- "unknown" must never render as success.
GREEN_STATES = frozenset({
    "reviewed", "verified", "passed", "selected", "completed_verified",
    "replay_verified", "compared", "done",
})
RED_STATES = frozenset({"failed", "incomplete", "leakage_failed"})

#: Decision a paired comparison is allowed to carry. Promotion is never
#: automated, so any other decision value is a refusal, not a pass.
ALLOWED_COMPARISON_DECISIONS = ("requires_operator_review",)


def load_json(path: Path) -> object | None:
    """Parse a JSON file, or return None when it is missing or malformed."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def verify_recorded_files(root: Path, files: object) -> dict:
    """Re-check a recorded ``[{path, sha256, bytes?}]`` list against ``root``.

    Absent files are reported as missing (unverifiable here), never as ok;
    present-but-different files are mismatches, which is a failure.
    """
    checked: list[str] = []
    missing: list[str] = []
    mismatches: list[dict] = []
    unreadable: list[str] = []
    entries = files if isinstance(files, list) else []
    for entry in entries:
        if not isinstance(entry, dict):
            unreadable.append(str(entry)[:80])
            continue
        rel = entry.get("path")
        want = entry.get("sha256")
        if not isinstance(rel, str) or not rel or not isinstance(want, str) or len(want) != 64:
            unreadable.append(str(rel)[:80])
            continue
        target = Path(root) / rel
        if not target.is_file():
            missing.append(rel)
            continue
        actual = sha256_file(target)
        if actual != want:
            mismatches.append({"path": rel, "recorded": want, "actual": actual})
            continue
        recorded_bytes = entry.get("bytes")
        if isinstance(recorded_bytes, int) and not isinstance(recorded_bytes, bool):
            if target.stat().st_size != recorded_bytes:
                mismatches.append({"path": rel, "recorded_bytes": recorded_bytes,
                                   "actual_bytes": target.stat().st_size})
                continue
        checked.append(rel)
    return {
        "ok": bool(entries) and not mismatches and not missing and not unreadable,
        "entries": len(entries),
        "checked": checked,
        "missing": missing,
        "mismatches": mismatches,
        "unreadable": unreadable,
    }


def manifest_output_files(digests: object) -> list[dict]:
    """``{"train": sha}`` -> ``[{"path": "train.jsonl", "sha256": sha}]``.

    prepare.py writes ``<split>.jsonl`` next to the manifest, so a logical
    digest key maps onto a real filename; anything that is not a digest is
    ignored rather than guessed at.
    """
    if not isinstance(digests, dict):
        return []
    return [{"path": f"{name}.jsonl", "sha256": value}
            for name, value in sorted(digests.items())
            if isinstance(value, str) and len(value) == 64]


# --------------------------------------------------------------------------
# Per-stage checks
# --------------------------------------------------------------------------

def catalog_state(exp_dir: Path) -> dict:
    catalog = load_json(Path(exp_dir) / "sources.json")
    sources = (catalog or {}).get("sources") if isinstance(catalog, dict) else None
    if not isinstance(sources, dict) or not sources:
        return {"stage": "source_review", "state": "missing",
                "detail": "no sources.json with a source catalog"}
    problems: list[str] = []
    approved, blocked = [], []
    for source_id, entry in sorted(sources.items()):
        if not isinstance(entry, dict):
            problems.append(f"{source_id}: entry is not an object")
            continue
        if not str(entry.get("revision") or "").strip():
            problems.append(f"{source_id}: revision is not pinned")
        if not str(entry.get("review_reference") or "").strip():
            problems.append(f"{source_id}: no review reference")
        decision = entry.get("approved")
        if not isinstance(decision, bool):
            problems.append(f"{source_id}: no approval decision")
        elif decision:
            approved.append(source_id)
            if not str(entry.get("license") or "").strip():
                problems.append(f"{source_id}: approved without a license determination")
        else:
            blocked.append(source_id)
            if not str(entry.get("review_note") or "").strip():
                problems.append(f"{source_id}: blocked without a recorded review note")
    detail = (f"{len(approved)} approved, {len(blocked)} blocked"
              + (f": {', '.join(blocked)}" if blocked else ""))
    if problems:
        return {"stage": "source_review", "state": "incomplete",
                "detail": detail + " -- " + "; ".join(problems[:3]),
                "problems": problems}
    return {"stage": "source_review", "state": "reviewed", "detail": detail,
            "approved": approved, "blocked": blocked}


def manifest_state(exp_dir: Path) -> dict:
    """Dataset manifest: digests re-checked, never accepted on existence."""
    manifests = sorted(Path(exp_dir).glob("**/manifest.json"))
    if not manifests:
        return {"stage": "data_collection", "state": "pending",
                "detail": "no manifest.json; run prepare.py"}
    subject = manifests[-1]  # deterministic: lexicographically last
    others = [str(m.relative_to(exp_dir)) for m in manifests[:-1]]
    manifest = load_json(subject)
    rel = str(subject.relative_to(exp_dir))
    extra = f" (+{len(others)} other manifest{'s' if len(others) > 1 else ''})" if others else ""
    if not isinstance(manifest, dict):
        return {"stage": "data_collection", "state": "failed",
                "detail": f"{rel} is unreadable JSON{extra}", "manifest": rel}
    fmt = str(manifest.get("format") or "")
    if not fmt.startswith(MANIFEST_FORMAT_PREFIX):
        return {"stage": "data_collection", "state": "failed",
                "detail": f"{rel} has an unknown format: {fmt!r}{extra}", "manifest": rel}
    train_rows = manifest.get("train_rows")
    dev_rows = manifest.get("dev_rows")
    if not isinstance(train_rows, int) or isinstance(train_rows, bool) or train_rows < 1:
        return {"stage": "data_collection", "state": "failed",
                "detail": f"{rel} carries no accepted training rows{extra}", "manifest": rel}
    if not isinstance(dev_rows, int) or isinstance(dev_rows, bool) or dev_rows < 0:
        return {"stage": "data_collection", "state": "failed",
                "detail": f"{rel} has an invalid dev row count{extra}", "manifest": rel}
    digests = manifest.get("output_sha256")
    files = manifest_output_files(digests)
    if not files:
        return {"stage": "data_collection", "state": "recorded_unpinned",
                "detail": f"{rel}: {train_rows} train / {dev_rows} dev rows, but the "
                          "manifest pins no output digests (rebuild with prepare.py)",
                "manifest": rel, "train_rows": train_rows, "dev_rows": dev_rows}
    check = verify_recorded_files(subject.parent, files)
    counts = f"{train_rows} train / {dev_rows} dev rows"
    if check["mismatches"]:
        return {"stage": "data_collection", "state": "failed",
                "detail": f"{rel}: digests differ from the manifest -- "
                          + "; ".join(m["path"] for m in check["mismatches"][:3]),
                "manifest": rel, "verification": check}
    if check["missing"]:
        return {"stage": "data_collection", "state": "recorded_offline",
                "detail": f"{rel}: {counts}; {len(check['missing'])} recorded file(s) "
                          "are not on this host, so the digests cannot be re-checked",
                "manifest": rel, "verification": check}
    if not check["ok"]:
        return {"stage": "data_collection", "state": "failed",
                "detail": f"{rel}: recorded file list is not verifiable",
                "manifest": rel, "verification": check}
    if manifest.get("holdout_is_external") is not True:
        return {"stage": "data_collection", "state": "failed",
                "detail": f"{rel}: {counts}; no external holdout file pinned -- "
                          "final-eval material must stay separate",
                "manifest": rel, "verification": check}
    return {"stage": "data_collection", "state": "verified",
            "detail": f"{rel}: {counts}; output digests match; holdout pinned external{extra}",
            "manifest": rel, "verification": check,
            "train_sha256": digests.get("train")}


def integrity_state(exp_dir: Path) -> dict:
    """Leakage / truncation / token-budget audit recorded inside the manifest."""
    manifests = sorted(Path(exp_dir).glob("**/manifest.json"))
    if not manifests:
        return {"stage": "validation", "state": "pending",
                "detail": "no manifest to audit; run prepare.py"}
    subject = manifests[-1]
    rel = str(subject.relative_to(exp_dir))
    manifest = load_json(subject)
    if not isinstance(manifest, dict):
        return {"stage": "validation", "state": "failed",
                "detail": f"{rel} is unreadable JSON", "manifest": rel}
    integrity = manifest.get("integrity")
    if not isinstance(integrity, dict):
        return {"stage": "validation", "state": "not_audited",
                "detail": f"{rel} predates the integrity audit "
                          "(no leakage/truncation/token-budget section)",
                "manifest": rel}
    failures = audit_failures(integrity)
    if failures:
        return {"stage": "validation", "state": "failed",
                "detail": f"{rel}: " + "; ".join(failures), "manifest": rel,
                "integrity": integrity, "failures": failures}
    return {"stage": "validation", "state": "passed",
            "detail": integrity_detail(integrity), "manifest": rel,
            "integrity": integrity}


def audit_failures(integrity: dict) -> list[str]:
    """Empty list means the audit sections are present and all clean."""
    problems: list[str] = []
    splits = integrity.get("split_integrity")
    if not isinstance(splits, dict) or splits.get("ok") is not True:
        problems.append("problem groups cross partitions")
    leakage = integrity.get("leakage")
    if not isinstance(leakage, dict) or leakage.get("ok") is not True:
        problems.append("holdout overlap not cleared")
    truncation = integrity.get("truncation")
    if not isinstance(truncation, dict) or truncation.get("ok") is not True:
        problems.append("supervised targets truncated or incomplete")
    tokens = integrity.get("token_audit")
    if not isinstance(tokens, dict) or tokens.get("ok") is not True:
        problems.append("token budget audit failed")
    elif tokens.get("max_length") != 2048:
        problems.append(f"token audit ran at max_length={tokens.get('max_length')}, "
                        "not the production 2048")
    if integrity.get("ok") is not True:
        problems.append("audit summary is not ok")
    return problems


def integrity_detail(integrity: dict) -> str:
    parts = []
    if isinstance(integrity.get("unique_problems"), int):
        parts.append(f"{integrity['unique_problems']} unique problems")
    leakage = integrity.get("leakage") if isinstance(integrity.get("leakage"), dict) else {}
    parts.append(f"holdout collisions {leakage.get('holdout_collisions', '?')}")
    truncation = integrity.get("truncation") if isinstance(integrity.get("truncation"), dict) else {}
    parts.append(f"truncated targets {truncation.get('incomplete_targets', '?')}")
    tokens = integrity.get("token_audit") if isinstance(integrity.get("token_audit"), dict) else {}
    if tokens.get("max_length"):
        parts.append(f"tokenizer {tokens.get('tokenizer', '?')} max_length {tokens['max_length']}")
    return "; ".join(parts)


def student_selection_state(exp_dir: Path) -> dict:
    record = load_json(Path(exp_dir) / "student_selection.json")
    if not isinstance(record, dict):
        return {"stage": "student_selection", "state": "pending",
                "detail": "run student.py --student qwen3-1.7b"}
    student = record.get("student") if isinstance(record.get("student"), dict) else {}
    problems: list[str] = []
    for field in ("repo", "revision", "license"):
        if not str(student.get(field) or "").strip():
            problems.append(f"student.{field} missing")
    hardware = record.get("hardware") if isinstance(record.get("hardware"), dict) else {}
    if not hardware.get("accelerators"):
        problems.append("no detected accelerators recorded")
    workload = record.get("workload_gb") if isinstance(record.get("workload_gb"), dict) else {}
    if not any(isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0
               for v in workload.values()):
        problems.append("memory plan carries no measured workload figures")
    if record.get("fits_primary_pool") is not True:
        problems.append("plan does not fit the primary pool")
    if problems:
        return {"stage": "student_selection", "state": "incomplete",
                "detail": "; ".join(problems[:3])}
    return {"stage": "student_selection", "state": "selected",
            "detail": f"{student['repo']} @ {str(student['revision'])[:8]} "
                      f"({student['license']}), plan bottleneck {record.get('plan_bottleneck', '?')}",
            "student": student}


def preflight_state(exp_dir: Path) -> dict:
    record = load_json(Path(exp_dir) / "preflight_result.json")
    if not isinstance(record, dict):
        return {"stage": "training_preflight", "state": "pending",
                "detail": "run preflight.py"}
    checks = record.get("checks") if isinstance(record.get("checks"), dict) else {}
    failed = sorted(name for name, ok in checks.items() if ok is not True)
    passed, total = record.get("passed"), record.get("total")
    detail = f"{passed}/{total} checks"
    if failed:
        return {"stage": "training_preflight", "state": "failed",
                "detail": detail + f"; failing: {', '.join(failed[:3])}"}
    if (record.get("ok") is not True or not checks or passed != total
            or passed != len(checks)):
        # The `ok` flag alone is a claim: a record can say ok while listing a
        # failed check, or while reporting a tally its own check list denies.
        return {"stage": "training_preflight", "state": "failed",
                "detail": detail + "; the record's own ok/count fields do not agree "
                          "with its check list"}
    return {"stage": "training_preflight", "state": "passed", "detail": detail,
            "checks": sorted(checks)}


def recipe_state(exp_dir: Path) -> dict:
    """Authorized recipes, or the reasons a recipe fails its own gate.

    An absent recipes directory is not a failure -- nothing has been written
    yet -- so it reports ``present: False`` with no failures.
    """
    recipes_dir = Path(exp_dir) / "recipes"
    if not recipes_dir.is_dir():
        return {"present": False, "failures": [], "recipes": []}
    failures: list[str] = []
    recipes: list[str] = []
    for path in sorted(recipes_dir.glob("*.json")):
        recipe = load_json(path)
        if not isinstance(recipe, dict):
            failures.append(f"{path.name}: unreadable")
            continue
        authorization = recipe.get("authorization")
        if not isinstance(authorization, dict):
            failures.append(f"{path.name}: no authorization block")
            continue
        if authorization.get("operator_approval_required") is not True:
            failures.append(f"{path.name}: does not require operator approval")
            continue
        if not isinstance(recipe.get("outputs"), dict):
            failures.append(f"{path.name}: no declared outputs")
            continue
        recipes.append(path.name)
    return {"present": True, "failures": failures, "recipes": recipes}


def condition_a_state(exp_dir: Path, *, manifest_train_sha256: str | None = None) -> dict:
    """The completed Condition A run: digests re-checked, claims re-derived.

    A mismatch between the record and the artifacts on disk is a failure. When
    the artifacts are simply not on this host (CI, another machine) the state
    stays ``recorded_offline``: the record is preserved, but it is not green.
    """
    record = load_json(Path(exp_dir) / "condition_a_artifacts.json")
    if record is None:
        return {"stage": "training", "state": "pending_record", "detail": ""}
    if not isinstance(record, dict) or record.get("format") != CONDITION_A_FORMAT:
        return {"stage": "training", "state": "failed",
                "detail": "condition_a_artifacts.json has an unknown format"}
    run = record.get("run") if isinstance(record.get("run"), dict) else {}
    problems: list[str] = []
    steps_done, steps_total = run.get("steps_completed"), run.get("steps_total")
    if not isinstance(steps_done, int) or steps_done != steps_total or not steps_total:
        problems.append(f"run did not complete every step ({steps_done}/{steps_total})")
    if not (record.get("loss_history") or {}).get("entries"):
        problems.append("loss history is empty")
    dataset_sha = run.get("dataset_sha256")
    if not isinstance(dataset_sha, str) or len(dataset_sha) != 64:
        problems.append("run does not pin the dataset digest")
    elif manifest_train_sha256 and dataset_sha != manifest_train_sha256:
        problems.append("trained dataset digest differs from the pinned manifest")
    if run.get("evaluated") is True:
        problems.append("training record may not claim evaluation")
    root = record.get("artifacts_root")
    if not isinstance(root, str) or not root.strip():
        problems.append("no artifact root recorded")
        check = {"ok": False, "entries": 0, "checked": [], "missing": [], "mismatches": []}
    else:
        check = verify_recorded_files(Path(root), record.get("files"))
    label = f"{steps_done}/{steps_total} steps"
    if problems:
        return {"stage": "training", "state": "failed",
                "detail": "; ".join(problems[:3]), "verification": check}
    if check["mismatches"]:
        return {"stage": "training", "state": "failed",
                "detail": "recorded adapter digests do not match the artifacts on disk: "
                          + "; ".join(m["path"] for m in check["mismatches"][:3]),
                "verification": check}
    if check["missing"] or not check["ok"]:
        return {"stage": "training", "state": "recorded_offline",
                "detail": f"{label}; adapter digest recorded "
                          f"({len(check.get('checked', []))}/{check.get('entries', 0)} files "
                          "present on this host)",
                "verification": check}
    return {"stage": "training", "state": "completed_verified",
            "detail": f"{label}; {len(check['checked'])} artifacts digest-verified "
                      f"(mean loss {run.get('mean_train_loss', '?')})",
            "verification": check}


def training_state(exp_dir: Path, *, manifest_train_sha256: str | None = None) -> dict:
    recipes = recipe_state(exp_dir)
    condition_a = condition_a_state(exp_dir, manifest_train_sha256=manifest_train_sha256)
    if condition_a["state"] != "pending_record":
        return {**condition_a, "recipes": recipes["recipes"],
                "recipe_failures": recipes["failures"]}
    if recipes["failures"]:
        return {"stage": "training", "state": "failed",
                "detail": "; ".join(recipes["failures"][:3])}
    if recipes["recipes"]:
        return {"stage": "training", "state": "recipes_ready",
                "detail": f"{len(recipes['recipes'])} authorized recipes, no completed run "
                          "recorded; GPU launch requires operator approval + device exclusivity",
                "recipes": recipes["recipes"]}
    return {"stage": "training", "state": "pending",
            "detail": "no recipes and no run record"}


def replay_state(exp_dir: Path) -> dict:
    summaries = sorted(Path(exp_dir).glob("**/replay_summary.json"))
    if not summaries:
        return {"stage": "evaluation", "state": "no_replay", "detail": ""}
    verified = not_green = setup_failed = skipped = 0
    unreadable = []
    for path in summaries:
        summary = load_json(path)
        if not isinstance(summary, dict):
            unreadable.append(path.name)
            continue
        for key, current in (("verified", verified), ("not_green", not_green),
                             ("setup_failed", setup_failed), ("skipped", skipped)):
            value = summary.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                if key == "verified":
                    verified = current + value
                elif key == "not_green":
                    not_green = current + value
                elif key == "setup_failed":
                    setup_failed = current + value
                else:
                    skipped = current + value
    detail = (f"{verified} verified, {not_green} not green, "
              f"{setup_failed} setup failed, {skipped} skipped")
    if unreadable:
        detail += f"; unreadable: {', '.join(unreadable[:2])}"
    if verified < 1:
        return {"stage": "evaluation", "state": "no_verified_repair", "detail": detail,
                "verified": verified, "not_green": not_green,
                "setup_failed": setup_failed, "skipped": skipped}
    return {"stage": "evaluation", "state": "replay_verified", "detail": detail,
            "verified": verified, "not_green": not_green,
            "setup_failed": setup_failed, "skipped": skipped}


def comparison_files(exp_dir: Path) -> list[Path]:
    return sorted(Path(exp_dir).glob("**/comparison_*.json"))


def comparison_state(exp_dir: Path, replay: dict | None = None) -> dict:
    """Paired comparison state; leakage discipline is checked, not assumed."""
    paths = comparison_files(exp_dir)
    if not paths:
        base = replay if isinstance(replay, dict) else replay_state(exp_dir)
        if base.get("state") == "no_verified_repair":
            return {"stage": "evaluation", "state": "no_verified_repair",
                    "detail": base.get("detail", "")}
        if base.get("state") == "replay_verified":
            return {"stage": "evaluation", "state": "replay_verified",
                    "detail": base.get("detail", "")}
        return {"stage": "evaluation", "state": "pending",
                "detail": "no comparison_*.json or replay_summary.json yet"}
    unchecked, leaked, decisions, counted = [], [], [], 0
    for path in paths:
        comparison = load_json(path)
        if not isinstance(comparison, dict):
            unchecked.append(path.name)
            continue
        decision = comparison.get("decision")
        if decision not in ALLOWED_COMPARISON_DECISIONS:
            decisions.append(f"{path.name}: decision={decision!r}")
            continue
        leakage = comparison.get("leakage")
        if not isinstance(leakage, dict):
            unchecked.append(path.name)
            continue
        bad = sorted(name for name, check in leakage.items()
                     if not isinstance(check, dict) or check.get("ok") is not True)
        if bad:
            leaked.append(f"{path.name}: {', '.join(bad)}")
            continue
        counted += 1
    if decisions:
        return {"stage": "evaluation", "state": "failed",
                "detail": "comparison carries a non-operator decision: "
                          + "; ".join(decisions[:2])}
    if leaked:
        return {"stage": "evaluation", "state": "failed",
                "detail": "leakage check failed: " + "; ".join(leaked[:2])}
    if not counted:
        return {"stage": "evaluation", "state": "unchecked",
                "detail": f"{len(paths)} comparison file(s) carry no leakage section: "
                          + ", ".join(unchecked[:2])}
    return {"stage": "evaluation", "state": "compared",
            "detail": f"{counted} comparison file(s) with cleared leakage checks",
            "compared": counted}


def results_state(exp_dir: Path, evaluation: dict) -> dict:
    report = Path(exp_dir) / "REPORT.md"
    if evaluation.get("state") == "compared" and report.is_file():
        return {"stage": "results", "state": "done",
                "detail": "REPORT.md plus paired comparisons with cleared leakage; "
                          "training loss is never model quality"}
    if report.is_file():
        return {"stage": "results", "state": "narrative_only",
                "detail": "REPORT.md exists but no leaked-checked comparison does"}
    return {"stage": "results", "state": "pending", "detail": "not yet produced"}


def stage_state(exp_dir: Path) -> list[dict]:
    """Per-stage state for the workflow screen, all derived from artifacts."""
    exp_dir = Path(exp_dir)
    data = manifest_state(exp_dir)
    replay = replay_state(exp_dir)
    evaluation = comparison_state(exp_dir, replay)
    return [
        catalog_state(exp_dir),
        data,
        integrity_state(exp_dir),
        student_selection_state(exp_dir),
        preflight_state(exp_dir),
        training_state(exp_dir, manifest_train_sha256=data.get("train_sha256")),
        evaluation,
        results_state(exp_dir, evaluation),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path,
                        help="pilot directory holding sources.json / manifests / records")
    parser.add_argument("--json", action="store_true", help="emit the full state as JSON")
    args = parser.parse_args()
    states = stage_state(args.directory)
    if args.json:
        print(json.dumps(states, indent=2, sort_keys=True))
    else:
        for entry in states:
            print(f"{entry['stage']:<18} {entry['state']:<20} {entry['detail']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
