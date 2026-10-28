"""Adversarial accounting tests; never execute generated Python programs."""

from __future__ import annotations

import json

import pytest

from chowder_batch import reap_v2_lcb_v6_checkpoint_readonly_audit_20261028 as audit


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def evidence(tmp_path):
    data = tmp_path / "raw"
    data.mkdir()
    for index, name in enumerate(audit.FILES):
        _write(data / name, [{"question_id": f"id{index}"}])
    generated = tmp_path / "generated.jsonl"
    _write(generated, [
        {"question_id": f"id{i}", "source_file": audit.FILES[i],
         "task": "LCB_generation", "prompt_style": "OpenAIChat",
         "max_tokens": 4096, "temperature": 0.2, "top_p": 0.95,
         "code": "print(1)" if i != 2 else "", "error": None}
        for i in range(4)
    ])
    metrics = tmp_path / "metrics.json"
    report = {
        "release": "release_v6",
        "dataset": "livecodebench/code_generation_lite",
        "problems_total": 6,
        "scored_population": 3,
        "missing": 3,
        "empty_code": 1,
        "transport_errors": 0,
        "metrics": {"pass@1": 1 / 3, "detail": {
            "pass@1": {"0": 1.0, "1": 0.0, "2": 0.0},
        }},
        "detail": [
            {"question_id": "id0", "score": [True, True]},
            {"question_id": "id1", "score": [False]},
            {"question_id": "id2", "score": [-4]},
        ],
    }
    metrics.write_text(json.dumps(report), encoding="utf-8")
    return data, generated, metrics


def _mutate_json(path, mutator):
    value = json.loads(path.read_text(encoding="utf-8"))
    mutator(value)
    path.write_text(json.dumps(value), encoding="utf-8")


def _mutate_jsonl(path, mutator):
    rows = audit.read_jsonl(path)
    mutator(rows)
    _write(path, rows)


def test_genuine_prefix_internal_consistency_is_not_release(evidence):
    result = audit.checkpoint_consistency(*evidence)
    assert (result["source_population"], result["generated_prefix"],
            result["scored_prefix"], result["correct"]) == (6, 4, 3, 1)
    assert result["pass_at_1_scored_prefix"] == 1 / 3
    assert result["full_official_score"] is None
    assert result["cpu_code_isolation_independently_attested"] is False
    assert result["original_serving_session_independently_attested"] is False
    assert result["frozen_gen0_status"] == "BLOCKED"
    assert result["chosen"] is None and result["release_approved"] is False


@pytest.mark.parametrize("change", [
    lambda x: x[1].update(question_id="id9"),
    lambda x: x[1].update(question_id="id0"),
    lambda x: x[0].update(source_file="test6.jsonl"),
    lambda x: x[0].update(temperature=0.7),
    lambda x: x[0].update(prompt_style="custom"),
    lambda x: x[0].update(max_tokens=1024),
])
def test_rejects_generated_source_and_prompt_drift(evidence, change):
    _mutate_jsonl(evidence[1], change)
    with pytest.raises(ValueError):
        audit.checkpoint_consistency(*evidence)


@pytest.mark.parametrize("change", [
    lambda x: x.update(missing=2),
    lambda x: x.update(empty_code=0),
    lambda x: x.update(scored_population=4),
    lambda x: x["metrics"].update(**{"pass@1": 0.8}),
    lambda x: x["metrics"]["detail"]["pass@1"].update(**{"1": 1.0}),
    lambda x: x["detail"][1].update(score=[True]),
    lambda x: x["detail"][2].update(score=[-2]),
    lambda x: x["detail"][2].update(question_id="id5"),
])
def test_rejects_checkpoint_tampering(evidence, change):
    _mutate_json(evidence[2], change)
    with pytest.raises(ValueError):
        audit.checkpoint_consistency(*evidence)


def test_rejects_duplicate_canonical_source_id(evidence):
    _mutate_jsonl(evidence[0] / audit.FILES[1],
                  lambda rows: rows[0].update(question_id="id0"))
    with pytest.raises(ValueError, match="duplicate"):
        audit.checkpoint_consistency(*evidence)


def test_rejects_truncated_generation_line(evidence):
    with evidence[1].open("a", encoding="utf-8") as stream:
        stream.write('{"question_id":')
    with pytest.raises(json.JSONDecodeError):
        audit.checkpoint_consistency(*evidence)


def test_rejects_missing_source_file(evidence):
    (evidence[0] / audit.FILES[3]).unlink()
    with pytest.raises(ValueError, match="missing"):
        audit.checkpoint_consistency(*evidence)
