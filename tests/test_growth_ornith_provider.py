"""The Ornith curriculum provider serves a frozen, digest-pinned admitted JSONL.

It must refuse a file whose sha256 is not the one pinned at construction, refuse
Fable-derived records, serve only the skill and verification it declares, and
go through ``materialise`` exactly like the built-ins -- so the protected-text
and contamination guards apply to it unchanged.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from chowder.growth.data_providers import (
    CLEAN,
    DEFAULT_PROVIDERS,
    PROVIDER_SCHEMA,
    TrainingDataRefusal,
    TrainingDataRequest,
    materialise,
)
from chowder.growth.ornith_provider import ORNITH_SCHEMA, OrnithCurriculumProvider


def _row(task_id: str, **extra: object) -> dict:
    """One admitted row as hermes writes it: proposal fields + AdmissionResult."""
    row = {
        "admitted": True,
        "lane": "sft",
        "task_id": task_id,
        "family": "arithmetic",
        "prompt": f"Return only the result of {task_id} * 2.",
        "expected_answer": "42",
        "verification_code": "print(21 * 2)",
        "difficulty_tags": ["basic_arithmetic"],
        "rationale": "targets a multiplication near-miss",
        "source_misses": ["miss-1"],
        "proposal_sha256": hashlib.sha256(task_id.encode()).hexdigest(),
        "validity": {"valid": True, "code_ran": True, "code_matches_answer": True},
        "rejection_reasons": [],
        "_admitted_at": "2026-10-16T00:00:00+00:00",
    }
    row.update(extra)
    return row


def _write(tmp_path: Path, rows: list[dict]) -> tuple[Path, str]:
    path = tmp_path / "admitted.jsonl"
    body = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    path.write_bytes(body.encode("utf-8"))
    return path, hashlib.sha256(body.encode("utf-8")).hexdigest()


def _provider(path: Path, sha: str, **kwargs: object) -> OrnithCurriculumProvider:
    kwargs.setdefault("skill", "reasoning.arithmetic")
    kwargs.setdefault("verification", "executable_tests")
    return OrnithCurriculumProvider(path, sha, **kwargs)


def _request(
    *,
    skill: str = "reasoning.arithmetic",
    verification: str = "executable_tests",
    training_type: str = "sft",
    example_count: int = 2,
    item_id: str = "ornith-item",
) -> TrainingDataRequest:
    return TrainingDataRequest(
        item_id=item_id,
        skill=skill,
        training_type=training_type,
        role="TARGET",
        verification_method=verification,
        generation="gen3",
        example_count=example_count,
    )


def _clean(_text: str) -> str:
    return CLEAN


def test_digest_mismatch_refuses_at_construction(tmp_path: Path) -> None:
    path, _sha = _write(tmp_path, [_row("t1")])
    with pytest.raises(TrainingDataRefusal, match=ORNITH_SCHEMA):
        _provider(path, "0" * 64)


def test_file_mutated_after_construction_refuses_on_produce(tmp_path: Path) -> None:
    path, sha = _write(tmp_path, [_row("t1"), _row("t2")])
    provider = _provider(path, sha)
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(TrainingDataRefusal, match="digest"):
        provider.produce(_request())


def test_unknown_or_unsupported_verification_refuses(tmp_path: Path) -> None:
    path, sha = _write(tmp_path, [_row("t1")])
    with pytest.raises(TrainingDataRefusal, match=ORNITH_SCHEMA):
        _provider(path, sha, verification="curated_trusted")


@pytest.mark.parametrize(
    "extra",
    [
        {"source": "fable-distill-v2"},
        {"provenance": {"teacher": "Fable-1"}},
        {"source_misses": ["FABLE-miss-9"]},
        {"rationale": "derived from a fable transcript"},
    ],
)
def test_fable_derived_record_refuses(tmp_path: Path, extra: dict) -> None:
    path, sha = _write(tmp_path, [_row("t1"), _row("t2", **extra)])
    with pytest.raises(TrainingDataRefusal, match="Fable"):
        _provider(path, sha)


def test_maps_hermes_fields_to_chat_line_and_records_provenance(tmp_path: Path) -> None:
    path, sha = _write(tmp_path, [_row("t1"), _row("t2")])
    provider = _provider(path, sha)
    corpus = materialise(
        [_request()], providers=(provider,), contamination_check=_clean
    )
    lines = corpus.material["ornith-item"]
    assert len(lines) == 2
    text = json.loads(lines[0])["text"]
    assert "Return only the result of t1 * 2." in text
    assert text.endswith("<|im_start|>assistant\n42<|im_end|>\n")
    example = corpus.examples["ornith-item"][0]
    assert example.provider_id == "ornith-curriculum"
    assert example.source_id == f"ornith-admitted:{sha[:16]}"
    assert example.verification == "executable_tests"
    record = provider.provenance[example.digest]
    assert record["task_id"] == "t1"
    assert record["admitted_sha256"] == sha
    assert record["proposal_sha256"] == hashlib.sha256(b"t1").hexdigest()
    assert record["lane"] == "sft"


def test_serves_only_declared_skill_and_verification(tmp_path: Path) -> None:
    path, sha = _write(tmp_path, [_row("t1"), _row("t2")])
    provider = _provider(path, sha)
    assert provider.supports(_request())
    assert not provider.supports(_request(skill="coding.python"))
    assert not provider.supports(_request(training_type="preference"))
    with pytest.raises(TrainingDataRefusal, match=PROVIDER_SCHEMA):
        materialise(
            [_request(verification="multi_judge")],
            providers=(provider,),
            contamination_check=_clean,
        )


def test_non_admitted_invalid_or_dpo_rows_are_not_served(tmp_path: Path) -> None:
    rows = [
        _row("ok"),
        _row("rejected", admitted=False, lane=""),
        _row("invalid", validity={"valid": False}),
        _row("dpo", lane="dpo"),
        _row("nocode", verification_code=""),
    ]
    path, sha = _write(tmp_path, rows)
    provider = _provider(path, sha)
    assert provider.served_task_ids == ("ok",)
    assert not provider.supports(_request(example_count=2))


def test_multi_judge_serves_rows_without_verification_code(tmp_path: Path) -> None:
    path, sha = _write(tmp_path, [_row("nocode", verification_code="")])
    provider = _provider(path, sha, verification="multi_judge")
    assert provider.served_task_ids == ("nocode",)


def test_rows_are_never_reused_across_items(tmp_path: Path) -> None:
    path, sha = _write(tmp_path, [_row("t1"), _row("t2"), _row("t3")])
    provider = _provider(path, sha)
    first = provider.produce(_request(example_count=2))
    assert not provider.supports(_request(example_count=2))
    with pytest.raises(TrainingDataRefusal, match=ORNITH_SCHEMA):
        provider.produce(_request(example_count=2))
    assert len(set(first)) == 2


def test_protected_eval_text_is_refused_through_materialise(tmp_path: Path) -> None:
    path, sha = _write(tmp_path, [_row("t1")])
    protected = _provider(path, sha).produce(_request(example_count=1))[0]
    with pytest.raises(TrainingDataRefusal, match="protected evaluation"):
        materialise(
            [_request(example_count=1)],
            providers=(_provider(path, sha),),
            protected_texts=[protected],
            contamination_check=_clean,
        )


def test_unmeasured_contamination_refuses(tmp_path: Path) -> None:
    path, sha = _write(tmp_path, [_row("t1")])
    with pytest.raises(TrainingDataRefusal, match="contamination"):
        materialise([_request(example_count=1)], providers=(_provider(path, sha),))


def test_not_registered_in_default_providers() -> None:
    assert not any(
        type(p).__name__ == "OrnithCurriculumProvider" for p in DEFAULT_PROVIDERS
    )
