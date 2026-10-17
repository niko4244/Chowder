"""Tests for fetch_smith_targeted: a census must be honest and pinned.

The module's whole reason to exist is that a *specific* slice of the corpus
needs every row, not a sample -- so the properties under test are the ones a
silent regression would corrupt: the approved-repair-source gate, the live
revision pin (the parquet conversion branch tracks dataset main), prefix
filtering, the claim-not-evidence row shape, per-row-group extraction, and
resume-with-identical-params only.
"""
import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

EXP = Path(__file__).resolve().parents[1] / "experiments" / "teacher_free_distill"


def load(name):
    spec = importlib.util.spec_from_file_location(name, EXP / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


targeted = load("fetch_smith_targeted")

REVISION = "rev-pinned"
PREFIXES = ["pndurette__gTTS", "seatgeek__thefuzz"]

TRAJ_ROWS = [
    {"traj_id": "t1", "instance_id": "pndurette__gTTS.dbcda4f3.pr_440",
     "resolved": "true", "model": "m", "messages": [{"role": "user"}], "patch": "diff-1"},
    {"traj_id": "t2", "instance_id": "other__repo.abc123.fix__zz",
     "resolved": "true", "model": "m", "messages": [], "patch": "diff-2"},
    {"traj_id": "t3", "instance_id": "seatgeek__thefuzz.8a05a3ee.lm_rewrite__n9",
     "resolved": "false", "model": "m", "messages": [], "patch": "diff-3"},
]

INSTANCE_ROWS = [
    {"instance_id": "pndurette__gTTS.dbcda4f3.pr_440", "repo": "swesmith/pndurette__gTTS.dbcda4f3",
     "FAIL_TO_PASS": ["gtts/tests/test_tts.py::test_x"], "image_name": "img-gtts",
     "patch": "ground-truth-never-extracted", "problem_statement": "long"},
]


def write_shard(tmp_path, rows, columns, name="0000.parquet", row_group_size=1):
    path = tmp_path / name
    table = pa.table({c: [json.dumps(r.get(c)) if c == "messages" else r.get(c)
                          for r in rows] for c in columns})
    pq.write_table(table, path, row_group_size=row_group_size)
    return path


def catalog(tmp_path, *, approved=True, kind="repair", revision=REVISION,
            repo="fixture/trajectories"):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({"sources": {"test": {
        "approved": approved, "license": "test-only", "kind": kind,
        "review_reference": "fixture", "revision": revision, "repo": repo}}}))
    return path


def wire(monkeypatch, shards, split="tool"):
    """Point the census at local parquet shards and a pinned live revision."""
    monkeypatch.setattr(targeted, "live_revision", lambda repo: REVISION)
    monkeypatch.setattr(targeted, "shard_paths", lambda repo, s: [str(p) for p in shards])
    monkeypatch.setattr(targeted, "open_shard", lambda repo, path: pq.ParquetFile(path))


def test_gate_refuses_unapproved_and_non_repair(tmp_path):
    with pytest.raises(PermissionError):
        targeted.export_targeted(catalog(tmp_path, approved=False), "test",
                                 tmp_path / "o.jsonl", prefixes=PREFIXES)
    with pytest.raises(ValueError):
        targeted.export_targeted(catalog(tmp_path, kind="chat"), "test",
                                 tmp_path / "o.jsonl", prefixes=PREFIXES)


def test_shard_paths_are_repo_scoped_and_ref_pinned():
    """Regression: a repo-relative shard path silently read the wrong repo
    ('repository not found') instead of the pinned conversion branch."""
    assert targeted.shard_fs_path("SWE-bench/SWE-smith-trajectories",
                                  "default/tool/0000.parquet") == (
        "datasets/SWE-bench/SWE-smith-trajectories@refs/convert/parquet/"
        "default/tool/0000.parquet")


def test_prefixes_are_validated(tmp_path):
    for bad in ([], ["  "], ["a"], ["own/er__repo"]):
        with pytest.raises(ValueError):
            targeted.validate_prefixes(bad)
    assert targeted.validate_prefixes([" pndurette__gTTS "]) == ["pndurette__gTTS"]
    assert targeted.matching_prefix("seatgeek__thefuzz.8a05a3ee.x", PREFIXES) == "seatgeek__thefuzz"
    assert targeted.matching_prefix(12345, PREFIXES) is None


def test_moved_pinned_revision_refuses_to_run(tmp_path, monkeypatch):
    """The parquet conversion tracks dataset main: a moved sha means the
    shards cannot be asserted to hold the catalog's revision."""
    wire(monkeypatch, [write_shard(tmp_path, TRAJ_ROWS, targeted.TRAJECTORY_COLUMNS)])
    monkeypatch.setattr(targeted, "live_revision", lambda repo: "rev-moved")
    with pytest.raises(PermissionError, match="pinned revision moved"):
        targeted.export_targeted(catalog(tmp_path), "test", tmp_path / "o.jsonl",
                                 prefixes=PREFIXES, log=lambda *a: None)


def test_census_keeps_only_matching_rows_and_never_verifies(tmp_path, monkeypatch):
    shard = write_shard(tmp_path, TRAJ_ROWS, targeted.TRAJECTORY_COLUMNS)
    wire(monkeypatch, [shard])
    dest = tmp_path / "traj.jsonl"
    result = targeted.export_targeted(catalog(tmp_path), "test", dest,
                                      prefixes=PREFIXES, log=lambda *a: None)
    rows = [json.loads(line) for line in dest.read_text().splitlines()]
    assert [r["traj_id"] for r in rows] == ["t1", "t3"]   # order preserved, non-prefix dropped
    assert rows[0]["claimed_resolved"] is True and rows[1]["claimed_resolved"] is False
    assert rows[0]["resolved"] == "true"                  # the raw claim stays visible
    assert "verification" not in rows[0]
    assert result["mode"] == "census" and result["matched_rows"] == 2
    assert result["rows_scanned"] == 3 and result["representative_of_full_corpus"] is False
    assert result["verification"].startswith("NONE")
    assert json.loads((tmp_path / "traj.jsonl.export.json").read_text())["prefixes"] == PREFIXES


def test_census_spans_row_groups_and_respects_limit(tmp_path, monkeypatch):
    """One-row row groups force the row-group path; a limit must not reorder."""
    shard = write_shard(tmp_path, TRAJ_ROWS, targeted.TRAJECTORY_COLUMNS, row_group_size=1)
    wire(monkeypatch, [shard])
    dest = tmp_path / "traj.jsonl"
    result = targeted.export_targeted(catalog(tmp_path), "test", dest,
                                      prefixes=PREFIXES, limit=1, log=lambda *a: None)
    rows = [json.loads(line) for line in dest.read_text().splitlines()]
    assert [r["traj_id"] for r in rows] == ["t1"]
    assert result["matched_rows"] == 1


def test_instances_extraction_keeps_task_definition_only(tmp_path, monkeypatch):
    shard = write_shard(tmp_path, INSTANCE_ROWS, targeted.INSTANCE_COLUMNS)
    wire(monkeypatch, [shard], split="train")
    dest = tmp_path / "instances.jsonl"
    targeted.export_instances(catalog(tmp_path, repo="fixture/instances"), "test", dest,
                              prefixes=["pndurette__gTTS"], log=lambda *a: None)
    row = json.loads(dest.read_text().splitlines()[0])
    assert row["FAIL_TO_PASS"] == ["gtts/tests/test_tts.py::test_x"]
    assert row["image_name"] == "img-gtts"
    assert "patch" not in row and "problem_statement" not in row
    sidecar = json.loads((tmp_path / "instances.jsonl.export.json").read_text())
    assert sidecar["kind"] == "instances" and "task definition, not evidence" in sidecar["note"]


def test_resume_requires_identical_params(tmp_path, monkeypatch):
    shard = write_shard(tmp_path, TRAJ_ROWS, targeted.TRAJECTORY_COLUMNS)
    wire(monkeypatch, [shard])
    dest = tmp_path / "traj.jsonl"
    first = targeted.export_targeted(catalog(tmp_path), "test", dest,
                                     prefixes=PREFIXES, log=lambda *a: None)
    assert "resumed" not in first
    marker = dest.read_text()
    dest.write_text("stale\n")
    again = targeted.export_targeted(catalog(tmp_path), "test", dest,
                                     prefixes=PREFIXES, log=lambda *a: None)
    assert again["resumed"] is True
    assert dest.read_text() == "stale\n"        # identical params never rewrite
    other = targeted.export_targeted(catalog(tmp_path), "test", dest,
                                     prefixes=["pndurette__gTTS"], log=lambda *a: None)
    assert "resumed" not in other
    # New params re-extract for real, so the narrowed census holds only its own rows.
    narrowed = [json.loads(line) for line in dest.read_text().splitlines()]
    assert [r["traj_id"] for r in narrowed] == ["t1"]
    assert marker.count("\n") == 2 and "diff-3" in marker
