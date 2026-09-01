import json

import pytest

from chowder.provenance import sha256_file
from chowder.replay_history import ReplayHistorySource, materialize_replay_history


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_replay_history_normalizes_custom_fields_and_deduplicates(tmp_path):
    prior = tmp_path / "prior.json"
    prior.write_text(
        json.dumps(
            [
                {"content": "base one"},
                {"content": "duplicate"},
            ]
        ),
        encoding="utf-8",
    )
    current = tmp_path / "current.jsonl"
    _jsonl(
        current,
        [
            {"text": "duplicate"},
            {"text": "repair one"},
        ],
    )

    replay = materialize_replay_history(
        sources=(
            ReplayHistorySource(
                str(prior), sha256_file(prior), text_field="content", role="root"
            ),
            ReplayHistorySource(
                str(current), sha256_file(current), text_field="text", role="repair"
            ),
        ),
        work_dir=tmp_path,
        ratio=1.0,
    )
    rows = [json.loads(line) for line in open(replay.path, encoding="utf-8")]
    assert rows == [
        {"text": "base one"},
        {"text": "duplicate"},
        {"text": "repair one"},
    ]
    manifest = json.load(open(replay.manifest_path, encoding="utf-8"))
    assert manifest["source_row_count"] == 4
    assert manifest["unique_row_count"] == 3
    assert [source["role"] for source in manifest["sources"]] == ["root", "repair"]
    assert all("source_ref" not in source for source in manifest["sources"])
    assert replay.verify()


def test_replay_history_is_content_identity_stable_across_source_paths(tmp_path):
    a_root = tmp_path / "a"
    b_root = tmp_path / "b"
    a_root.mkdir()
    b_root.mkdir()
    a = a_root / "data.jsonl"
    b = b_root / "renamed.jsonl"
    content = json.dumps({"text": "same content"}) + "\n"
    a.write_text(content, encoding="utf-8")
    b.write_text(content, encoding="utf-8")

    first = materialize_replay_history(
        sources=(ReplayHistorySource(str(a), sha256_file(a), role="history"),),
        work_dir=a_root,
        ratio=0.5,
    )
    second = materialize_replay_history(
        sources=(ReplayHistorySource(str(b), sha256_file(b), role="history"),),
        work_dir=b_root,
        ratio=0.5,
    )
    assert first.sha256 == second.sha256
    assert first.manifest_sha256 == second.manifest_sha256
    assert open(first.manifest_path, encoding="utf-8").read() == open(
        second.manifest_path, encoding="utf-8"
    ).read()


def test_replay_history_rejects_mutated_source(tmp_path):
    source = tmp_path / "data.jsonl"
    _jsonl(source, [{"text": "original"}])
    binding = ReplayHistorySource(str(source), sha256_file(source))
    _jsonl(source, [{"text": "changed"}])
    with pytest.raises(ValueError, match="source content changed"):
        materialize_replay_history(
            sources=(binding,), work_dir=tmp_path, ratio=1.0
        )


def test_verified_replay_detects_manifest_tamper(tmp_path):
    source = tmp_path / "data.jsonl"
    _jsonl(source, [{"text": "one"}, {"text": "two"}])
    replay = materialize_replay_history(
        sources=(ReplayHistorySource(str(source), sha256_file(source)),),
        work_dir=tmp_path,
        ratio=1.0,
    )
    with open(replay.manifest_path, "a", encoding="utf-8") as handle:
        handle.write("tamper\n")
    with pytest.raises(ValueError, match="replay manifest content changed"):
        replay.verify()
