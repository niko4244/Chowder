"""Synthetic negative fixtures only. Never a ProgramBench model performance test."""

import json
from pathlib import Path

import pytest

from chowder_batch import reap_original_programbench200_population_20261027 as p


def fixture(tmp_path: Path):
    one = tmp_path / "authentic__cli.123abcd"
    one.mkdir()
    (one / "task.yaml").write_text(
        "repository: genuine/cli\ncommit: 123456abcdef\nlanguage: python\n"
        "difficulty: hard\n")
    (one / "tests.json").write_text(json.dumps({
        "branches": {
            "branch1": {"ignored": False, "tests": ["t1", "t2", "t3"],
                        "ignored_tests": [{"name": "t3"}]},
            "branch2": {"ignored": True, "tests": ["t4", "t5"], "ignored_tests": []},
        },
    }))
    return one


def test_original_scoring_excludes_ignored_branch_and_named_tests(tmp_path):
    f = fixture(tmp_path)
    r = p.original_task(tmp_path, f.name)
    assert r["scorable_behavioral_tests"] == 2
    assert r["raw_behavioral_tests"] == 5
    assert r["ignored_branches"] == 1
    assert r["ignored_tests_on_active_branches"] == 1
    assert len(r["original_task_yaml_sha256"]) == 64
    assert len(r["original_tests_json_sha256"]) == 64


def test_upstream_missing_optional_difficulty_does_not_erase_real_task(tmp_path):
    f = fixture(tmp_path)
    y = f / "task.yaml"
    y.write_text(y.read_text().replace("difficulty: hard\n", ""))
    result = p.original_task(tmp_path, f.name)
    assert result["difficulty"] is None
    assert result["scorable_behavioral_tests"] == 2


def test_refuse_duplicate_original_test_names(tmp_path):
    f = fixture(tmp_path)
    t = f / "tests.json"
    s = json.loads(t.read_text())
    s["branches"]["branch1"]["tests"] = ["t1", "t1"]
    t.write_text(json.dumps(s))
    with pytest.raises(ValueError, match="identities"):
        p.original_task(tmp_path, f.name)


def test_refuse_unlisted_excluded_original_test(tmp_path):
    f = fixture(tmp_path)
    t = f / "tests.json"
    s = json.loads(t.read_text())
    s["branches"]["branch1"]["ignored_tests"] = [{"name": "nonoriginal"}]
    t.write_text(json.dumps(s))
    with pytest.raises(ValueError, match="not found"):
        p.original_task(tmp_path, f.name)


def test_refuse_duplicate_original_json_keys():
    with pytest.raises(ValueError, match="duplicate"):
        p._unique_json(b'{"branches": {}, "branches": {}}')


@pytest.mark.parametrize("name", [
    "../outside", "foo/bar", "foo\\bar", "", "task/no", "a/../../etc",
])
def test_refuse_test_path_namespace_escape(tmp_path, name):
    with pytest.raises(ValueError):
        p.original_task(tmp_path, name)


def test_source_only_cannot_approve_release_without_model_submission():
    assert p.N == 200
    assert p.N_SCORED == 248089
    assert p.UPSTREAM_UNIT_TEST_FIXTURE == "testorg__calculator.abc1234"
    assert p.PINNED_COMMIT == "1fe64c87a1318998636850cead00b0db80e2d928"
