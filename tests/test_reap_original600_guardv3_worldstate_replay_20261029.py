"""Adversarial checks for original guard.v3 source-derived WorldState replay."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess

import pytest

from chowder_batch import reap_original600_guardv3_worldstate_replay_20261029 as replay


class FakeWorld:
    def __init__(self, state):
        self.state = state

    @classmethod
    def model_validate(cls, state):
        return cls(state)

    def model_dump(self, mode):
        assert mode == "json"
        return self.state


def _fixture():
    source = {"example_id": 501, "prompt": [{"role": "user", "content": "task"}]}
    info = {"task_name": "sales.multi_hop_lookup", "initial_state": {},
            "assertions": [{"type": "fake_assertion", "target": "ok"}]}
    saved = {"name": info["task_name"], "end_state": {"meta": {"schema_version": "0.1.0"}},
             "score": 1.0, "passed": True, "assertions_total": 1,
             "assertions_passed": 1, "assertion_results": [
                 {"type": "fake_assertion", "params": {"target": "ok"},
                  "passed": True, "excluded": False},
             ]}
    return source, info, saved


def _partial_credit(state):
    state["_assertion_results"] = [{"type": "fake_assertion",
                                   "params": {"target": "ok"},
                                   "passed": True, "excluded": False}]
    return 1.0


def _grade(saved):
    source, info, _ = _fixture()
    return replay.regrade_one(0, source, info, saved,
                              WorldState=FakeWorld, partial_credit=_partial_credit,
                              task_contract_sha256=lambda **_: "source-task-contract")


def test_original_regrade_accepts_independently_matching_state() -> None:
    _, _, saved = _fixture()
    row = _grade(saved)
    assert row["passed"] is True and row["score"] == 1.0
    assert row["source_task_contract_sha256"] == "source-task-contract"
    assert row["null_param_omissions"] == 0


@pytest.mark.parametrize("mutation,pattern", [
    (lambda r: r.update(passed=False), "WorldState score differs"),
    (lambda r: r.update(score=0.0), "WorldState score differs"),
    (lambda r: r.update(assertions_total=0), "assertion totals differ"),
    (lambda r: r.update(assertions_passed=0), "assertion totals differ"),
    (lambda r: r["assertion_results"][0].update(passed=False), "assertion verdict differs"),
    (lambda r: r["assertion_results"][0].update(excluded=True), "assertion verdict differs"),
    (lambda r: r["assertion_results"][0]["params"].update(target="forged"),
     "assertion parameters differ"),
    (lambda r: r["assertion_results"].append(copy.deepcopy(r["assertion_results"][0])),
     "WorldState score differs"),
    (lambda r: r["assertion_results"][0].update(extra="injected"),
     "assertion verdict differs"),
    (lambda r: r.update(score="1.0"), "row is malformed"),
])
def test_replay_refuses_forged_official_record(mutation, pattern) -> None:
    _, _, saved = _fixture()
    mutation(saved)
    with pytest.raises(ValueError, match=pattern):
        _grade(saved)


def test_null_parameter_exception_is_exactly_scoped() -> None:
    source = {"object_type": "Account", "record_id": "001xx000003FC01",
              "field": "health_status", "value": None}
    saved = {k: v for k, v in source.items() if k != "value"}
    assert replay.checked_params(source, saved, name="salesforce_field_not_changed", index=47)
    with pytest.raises(ValueError, match="assertion parameters differ"):
        replay.checked_params(source, saved, name="salesforce_field_not_changed", index=46)
    with pytest.raises(ValueError, match="assertion parameters differ"):
        replay.checked_params(source, saved, name="salesforce_field_equals", index=47)
    with pytest.raises(ValueError, match="assertion parameters differ"):
        replay.checked_params({**source, "value": "not null"}, saved,
                              name="salesforce_field_not_changed", index=47)
    with pytest.raises(ValueError, match="assertion parameters differ"):
        replay.checked_params(source, {**saved, "field": "forged"},
                              name="salesforce_field_not_changed", index=47)


def test_canonical_json_rejects_nan_and_is_stable() -> None:
    assert replay.canonical({"b": 2, "a": 1}) == b'{"a":1,"b":2}'
    with pytest.raises(ValueError):
        replay.canonical({"forged": float("nan")})


def test_ambiguous_official_json_refused() -> None:
    assert replay.strict_json(b'{"a":1,"b":2}') == {"a": 1, "b": 2}
    with pytest.raises(ValueError, match="duplicate key"):
        replay.strict_json(b'{"passed":false,"passed":true}')
    with pytest.raises(ValueError, match="duplicate key"):
        replay.strict_json(b'{"end_state":{"meta":1,"meta":2}}')
    with pytest.raises(ValueError, match="nonfinite JSON"):
        replay.strict_json(b'{"score":NaN}')


def test_real_official_source_tree_is_git_byte_pinned() -> None:
    digest, tracked = replay.official_source_tree()
    assert len(digest) == 64 and len(tracked) > 100
    assert "automationbench/schema/world.py" in tracked
    assert "automationbench/rubric/registry.py" in tracked


def test_official_source_byte_tamper_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    actual = Path.read_bytes
    target = replay.OFFICIAL / "automationbench/__init__.py"

    def changed(path: Path) -> bytes:
        raw = actual(path)
        return raw + b"\n# forged" if path == target else raw

    monkeypatch.setattr(Path, "read_bytes", changed)
    with pytest.raises(ValueError, match="official tracked source bytes differ"):
        replay.official_source_tree()


def test_real_official_worldstate_refuses_forged_in_memory_verdict() -> None:
    """Use the actual v1.0.6 handler, not a synthetic/mock scoring function."""
    code = (
        "import copy, json\n"
        "from pathlib import Path\n"
        "from automationbench.domains.sales import get_sales_dataset\n"
        "from automationbench.rubric import partial_credit\n"
        "from automationbench.schema.world import WorldState\n"
        "from automationbench.task_contract import task_contract_sha256\n"
        "from chowder_batch import reap_original600_guardv3_worldstate_replay_20261029 as r\n"
        "source=get_sales_dataset()[0]\n"
        "info=json.loads(source['info'])\n"
        "p=r.historical.ROOT/'task-000/public-automationbench.private.json'\n"
        "saved=json.loads(p.read_text())['tasks'][0]\n"
        "kwargs=dict(WorldState=WorldState,partial_credit=partial_credit,"
        "task_contract_sha256=task_contract_sha256)\n"
        "assert r.regrade_one(0,source,info,saved,**kwargs)['passed'] is False\n"
        "for field,value in [('passed',True),('score',1.0)]:\n"
        "    forged=copy.deepcopy(saved);forged[field]=value\n"
        "    try:r.regrade_one(0,source,info,forged,**kwargs)\n"
        "    except ValueError:pass\n"
        "    else:raise AssertionError('forged official field admitted')\n"
        "print('REAL_OFFICIAL_ADVERSARIAL_REFUSALS_OK')\n"
    )
    result = subprocess.run(
        [str(replay.OFFICIAL / ".venv/Scripts/python.exe"), "-B", "-c", code],
        cwd=replay.historical.REPO, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "REAL_OFFICIAL_ADVERSARIAL_REFUSALS_OK" in result.stdout


def test_actual_original_88_worldstate_replay_no_native_model() -> None:
    python = replay.OFFICIAL / ".venv/Scripts/python.exe"
    owner = replay.historical.ROOT / "owner.json"
    before = replay.sha(owner.read_bytes())
    completed = subprocess.run(
        [str(python), "-B", "-m",
         "chowder_batch.reap_original600_guardv3_worldstate_replay_20261029"],
        cwd=replay.historical.REPO, capture_output=True, text=True, timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout)
    assert result["replayed_scored_tasks"] == 88
    assert result["replayed_strict_passes"] == 3
    assert result["known_official_null_param_omissions"] == 1
    assert result["native_http_response_text_authenticated"] is False
    assert result["comparable_to_frozen_four_slot"] is False
    assert result["gen0_status"] == "BLOCKED"
    assert result["chosen"] is None and result["release_approved"] is False
    assert replay.sha(owner.read_bytes()) == before
