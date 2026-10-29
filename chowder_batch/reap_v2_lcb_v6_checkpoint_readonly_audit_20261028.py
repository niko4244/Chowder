"""Read-only independent consistency audit of an *incomplete* LCB v6 checkpoint.

Never execute candidate code, contact the model server, or treat a checkpoint
as a full official score. The original execution sandbox and serving session
need separate durable attestations before any frozen qualification admission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

DATA = Path("F:/lcb-data/raw")
OUT = Path("F:/lcb-out")
SOURCE = Path("F:/lcb-src")
RUNTIME = Path("F:/lcb-runtime")
GENERATOR = Path("F:/lcb_gen.py")
EVALUATOR = Path("F:/lcb_eval.py")
FILES = tuple(f"test{n if n > 1 else ''}.jsonl" for n in range(1, 7))
SOURCE_COMMIT = "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"
DATA_SHAS = (
    "2bd02b38beb48e8c46b5b9987095d999ff38cd8efc255ea5d58974317c48f63f",
    "095df7c5daf15f882c51a9deb84085cff1e073495a5dbcf95015a564d485f3a3",
    "28ed26cc83363ce3f1fe2d5fad9f8393077beb1907b167a31bd3b32f80801b79",
    "d711138ddaebfcf5f8ec6a4283ee677298c0f5c5d374a235af92aaf0584510da",
    "7f77571c2a6df0c2a72a3277650309f67e01e0008e18117e624633df53f81214",
    "bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5",
)
GENERATOR_SHA = "9df4f3e4c1d6b6ec6cc1cb1843be2aead122a9bd6244c926332792092e5fb99d"
EVALUATOR_SHA = "5217d8e2d1ddcb88fbea8383ac856b8f12e8d19117c2591345d27ac11de261c9"
SCHEMA = "chowder.reap-v2-lcb-v6-readonly-checkpoint-consistency.v1"


def sha(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing or linked source: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_linked_or_reparse(path: Path) -> None:
    """Refuse NTFS junctions/mount points as well as ordinary symlinks.

    Python 3.11's ``Path.is_symlink`` does not identify Windows junctions.
    The runtime must not follow an executable package path outside the
    independently reviewed source/runtime trees.
    """
    metadata = os.lstat(path)
    if (getattr(metadata, "st_file_attributes", 0) & 0x400
            or path.is_symlink()):  # FILE_ATTRIBUTE_REPARSE_POINT
        raise ValueError("official evaluator source/runtime contains reparse path")


def _pinned_git(source: Path, *args: str) -> bytes:
    """Read the specified Git object without ambient Git overrides/replacements."""
    environment = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    return subprocess.check_output(
        ["git", "-C", str(source), *args], timeout=20, env=environment,
    )


def verify_source_python_git_inventory(source: Path) -> int:
    """Require exact pinned Git blobs, not only tracked names or clean status.

    Git's assume-unchanged/skip-worktree index flags can hide modified tracked
    Python from ``git status``. Use the immutable commit tree, bypassing index
    flags and replacement refs. The original Windows checkout has CRLF files
    while Git stores LF; permit *only* that byte-exact newline conversion,
    never arbitrary Git clean filters or modified executable Python.
    """
    original = source / "lcb_runner"
    if not original.is_dir() or original.is_symlink():
        raise ValueError("original evaluator Python source package missing or linked")
    for path in (source, original):
        _reject_linked_or_reparse(path)
    for path in original.rglob("*"):
        _reject_linked_or_reparse(path)
    tracked_raw = _pinned_git(
        source, "ls-tree", "-r", "-z", "--name-only", SOURCE_COMMIT,
        "--", "lcb_runner",
    )
    tracked = {
        Path(os.fsdecode(item)).relative_to("lcb_runner")
        for item in tracked_raw.split(b"\0") if item
        and Path(os.fsdecode(item)).suffix.lower() == ".py"
    }
    actual = {path.relative_to(original) for path in original.rglob("*.py")}
    if not tracked or tracked != actual:
        raise ValueError("original evaluator Python files differ from pinned Git inventory")
    for relative in sorted(tracked):
        git_path = "lcb_runner/" + relative.as_posix()
        pinned_bytes = _pinned_git(source, "show", f"{SOURCE_COMMIT}:{git_path}")
        live_bytes = (original / relative).read_bytes()
        if (live_bytes != pinned_bytes
                and live_bytes.replace(b"\r\n", b"\n") != pinned_bytes):
            raise ValueError(f"original evaluator pinned Git blob drift: {git_path}")
    return len(tracked)


def verify_runtime_python_sources(source: Path, runtime: Path) -> None:
    """Require matching Python sources and refuse unreviewed executable imports.

    A source-to-runtime one-way walk misses extra runtime modules that Python
    can import during evaluation. Precompiled bytecode, importable archives,
    startup .pth files and top-level Python modules can override or augment
    the original source without a Git-pinned executable-code identity.
    """
    original = source / "lcb_runner"
    copied = runtime / "lcb_runner"
    if (not original.is_dir() or original.is_symlink()
            or not copied.is_dir() or copied.is_symlink()
            or runtime.is_symlink()):
        raise ValueError("original evaluator runtime package missing or linked")
    _reject_linked_or_reparse(source)
    for tree in (original, runtime):
        _reject_linked_or_reparse(tree)
        for path in tree.rglob("*"):
            _reject_linked_or_reparse(path)
    unreviewed_suffixes = {
        ".pyc", ".pyo", ".pyw", ".pyd", ".so", ".dll",
        ".zip", ".egg", ".whl", ".pyz", ".pth",
    }
    for path in runtime.rglob("*"):
        if path.suffix.lower() in unreviewed_suffixes:
            raise ValueError("official evaluator runtime contains unreviewed importable bytecode or archive")
        if path.suffix.lower() == ".py" and not path.is_relative_to(copied):
            raise ValueError("official evaluator runtime contains unreviewed Python source outside package")
    source_files = {p.relative_to(original): p for p in original.rglob("*.py")}
    runtime_files = {p.relative_to(copied): p for p in copied.rglob("*.py")}
    if not source_files or source_files.keys() != runtime_files.keys():
        raise ValueError("official evaluator runtime Python inventory drift")
    for relative, file in source_files.items():
        if sha(file) != sha(runtime_files[relative]):
            raise ValueError(f"official evaluator runtime source drift: {relative}")


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing or linked journal: {path}")
    rows = []
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"blank journal line {index}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"non-object journal line {index}")
            rows.append(row)
    return rows


def _unique_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate source/evidence JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"nonfinite source/evidence JSON: {value}")


def _stable_jsonl(path: Path, *, source_only: bool = False) -> tuple[list, str]:
    """Hash and parse each original source/journal byte through ONE open handle.

    Dataset shards are multi-gigabyte; only canonical IDs are retained for
    source files. A separate sha(path)/read_jsonl(path) permits version swaps.
    """
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing or linked source: {path}")
    before = path.stat()
    digest = hashlib.sha256()
    rows = []
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise ValueError("source file replaced before read")
        for index, raw_line in enumerate(stream, 1):
            digest.update(raw_line)
            row = json.loads(
                raw_line.decode("utf-8"), object_pairs_hook=_unique_keys,
                parse_constant=_reject_nonfinite,
            )
            if not raw_line.endswith(b"\n"):
                raise ValueError(f"unterminated source/evidence row {index}")
            if not isinstance(row, dict):
                raise ValueError(f"non-object source/evidence row {index}")
            if source_only:
                qid = row.get("question_id")
                if not isinstance(qid, str) or not qid:
                    raise ValueError(f"invalid canonical question ID {index}")
                rows.append(qid)
            else:
                rows.append(row)
        finished = os.fstat(stream.fileno())
    after = path.stat()
    identities = {(s.st_dev, s.st_ino) for s in (before, opened, finished, after)}
    if (len(identities) != 1 or before.st_size != finished.st_size
            or before.st_size != after.st_size
            or before.st_mtime_ns != finished.st_mtime_ns
            or before.st_mtime_ns != after.st_mtime_ns):
        raise ValueError("source/evidence file changed during read")
    return rows, digest.hexdigest()


def _stable_json(path: Path) -> tuple[dict, str]:
    """Return parsed metrics and SHA of the same exact opened bytes."""
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing or linked metrics: {path}")
    before = path.stat()
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        raw = stream.read()
        finished = os.fstat(stream.fileno())
    after = path.stat()
    if (len({(s.st_dev, s.st_ino) for s in (before, opened, finished, after)}) != 1
            or before.st_size != len(raw) or after.st_size != len(raw)
            or before.st_mtime_ns != finished.st_mtime_ns
            or before.st_mtime_ns != after.st_mtime_ns):
        raise ValueError("metrics file changed during read")
    report = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_keys,
                        parse_constant=_reject_nonfinite)
    if not isinstance(report, dict):
        raise ValueError("non-object metrics checkpoint")
    return report, hashlib.sha256(raw).hexdigest()


def checkpoint_consistency(data: Path, generations: Path, metrics: Path, *,
                           include_canonical: bool = False) -> dict:
    """Verify per-item accounting without running untrusted candidate code."""
    source_ids = []
    source_files = {}
    source_shas = {}
    for name in FILES:
        path = data / name
        ids, source_shas[name] = _stable_jsonl(path, source_only=True)
        for qid in ids:
            if not isinstance(qid, str) or not qid or qid in source_files:
                raise ValueError("invalid or duplicate canonical question ID")
            source_ids.append(qid)
            source_files[qid] = name

    generated, gen_sha = _stable_jsonl(generations)
    if not generated or len(generated) > len(source_ids):
        raise ValueError("empty or excessive model-generation prefix")
    ids = [r.get("question_id") for r in generated]
    if ids != source_ids[:len(ids)]:
        raise ValueError("generation IDs are not the canonical source prefix")
    for row in generated:
        if (row.get("source_file") != source_files[row["question_id"]]
                or row.get("task") != "LCB_generation"
                or row.get("prompt_style") != "OpenAIChat"
                or row.get("max_tokens") != 4096
                or row.get("temperature") != 0.2
                or row.get("top_p") != 0.95
                or not isinstance(row.get("code"), str)):
            raise ValueError("generation source/prompt/parameter mismatch")

    report, metrics_sha = _stable_json(metrics)
    details = report["detail"]
    n = len(details)
    if (report.get("release") != "release_v6"
            or report.get("dataset") != "livecodebench/code_generation_lite"
            or report.get("problems_total") != len(source_ids)
            or report.get("scored_population") != n
            or report.get("missing") != len(source_ids) - n
            or not 0 < n <= len(generated)):
        raise ValueError("official checkpoint population/identity mismatch")
    if [r.get("question_id") for r in details] != sorted(ids[:n]):
        raise ValueError("checkpoint per-item IDs differ from generated prefix")
    if (report.get("empty_code") != sum(not r["code"].strip() for r in generated[:n])
            or report.get("transport_errors") != sum(bool(r.get("error")) for r in generated[:n])):
        raise ValueError("checkpoint empty/error accounting mismatch")

    per_item = report["metrics"]["detail"]["pass@1"]
    if set(per_item) != {str(i) for i in range(n)}:
        raise ValueError("checkpoint pass@1 index population mismatch")
    passed = 0
    for i, row in enumerate(details):
        tests = row.get("score")
        if (not isinstance(tests, list) or not tests
                or any(type(v) is not bool and (type(v) is not int or v not in (-3, -4))
                       for v in tests)):
            raise ValueError("invalid official per-test outcome encoding")
        good = all(v is True for v in tests)
        passed += good
        if per_item[str(i)] != float(good):
            raise ValueError("checkpoint per-item pass@1 contradicts test outcomes")
    accuracy = passed / n
    if not math.isclose(report["metrics"]["pass@1"], accuracy, abs_tol=1e-12):
        raise ValueError("checkpoint aggregate pass@1 contradicts test outcomes")
    result = {
        "schema": SCHEMA,
        "status": "PARTIAL_CHECKPOINT_INTERNAL_CONSISTENCY_ONLY",
        "source_population": len(source_ids),
        "generated_prefix": len(generated),
        "scored_prefix": n,
        "correct": passed,
        "incorrect": n - passed,
        "pass_at_1_scored_prefix": accuracy,
        "unscored_source_tasks": len(source_ids) - n,
        "source_dataset_sha256": source_shas,
        "generation_journal_sha256": gen_sha,
        "checkpoint_sha256": metrics_sha,
        "original_serving_session_independently_attested": False,
        "cpu_code_isolation_independently_attested": False,
        "full_official_score": None,
        "frozen_gen0_status": "BLOCKED",
        "chosen": None,
        "release_approved": False,
    }
    if include_canonical:
        result["_canonical_source"] = [(qid, source_files[qid]) for qid in source_ids]
    return result


def inspect(*, include_canonical: bool = False) -> dict:
    repo_head = _pinned_git(SOURCE, "rev-parse", "HEAD").decode().strip()
    if repo_head != SOURCE_COMMIT or _pinned_git(SOURCE, "status", "--porcelain").strip():
        raise ValueError("original LiveCodeBench repository source drift")
    if sha(GENERATOR) != GENERATOR_SHA or sha(EVALUATOR) != EVALUATOR_SHA:
        raise ValueError("generation or evaluation driver source drift")
    verify_source_python_git_inventory(SOURCE)
    verify_runtime_python_sources(SOURCE, RUNTIME)
    verify_source_python_git_inventory(SOURCE)
    result = checkpoint_consistency(
        DATA, OUT / "gen_q4.jsonl", OUT / "metrics_q4_checkpoint.json",
        include_canonical=include_canonical,
    )
    if tuple(result["source_dataset_sha256"].values()) != DATA_SHAS:
        raise ValueError("original LiveCodeBench dataset source hash drift")
    return {
        **result,
        "source_git_commit": SOURCE_COMMIT,
        "generator_source_sha256": GENERATOR_SHA,
        "evaluator_source_sha256": EVALUATOR_SHA,
        "research_only": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    try:
        result = inspect()
        if args.snapshot is not None:
            target = args.snapshot
            if (not target.is_absolute() or target.exists() or target.is_symlink()
                    or target.parent.resolve() != Path("F:/").resolve()):
                raise ValueError("new exclusive F: snapshot path required")
            with target.open("xb") as stream:
                stream.write((json.dumps(result, sort_keys=True, indent=2) + "\n").encode())
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        print(json.dumps({"status": "REFUSED_UNATTESTED_LCB_CHECKPOINT",
                          "reason": str(exc)[:350], "frozen_gen0_status": "BLOCKED"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
