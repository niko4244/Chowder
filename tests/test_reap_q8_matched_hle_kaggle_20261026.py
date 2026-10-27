"""CPU-only provenance and fail-closed tests for private paired Q8 HLE input."""

from pathlib import Path
import ast
import hashlib
import json

import pytest

from chowder_batch import reap_q8_matched_hle_kaggle_20261026 as r


def test_authenticated_original_q4_population_exactly_128():
    ids = r._source_bound_rows()
    assert len(ids) == len(set(ids)) == 128
    assert hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest() == (
        "65a8f15e747c2a00577aa0a5c5c06ec7724e4c31ec27a99c6a416c3f587acbea"
    )


def test_real_executed_q4_protocol_not_earlier_draft():
    original = r.Q4_SOURCE.read_bytes()
    assert hashlib.sha256(original).hexdigest() == r.Q4_SOURCE_SHA256
    assert b"PROTOCOL_BYTES = 13041" in original
    assert b"cbacf0bb2215fce041ca62e991002d2772bee04891bc5e1773d3b0e2c44c49b2" in original
    assert b"PROTOCOL_MANIFEST_SHA256 = 'ebdb19e818257c21335889f9bea51c9939293444d315ba0211e21d2c599e70d8'" in original


def test_private_128_q8_replay_has_only_model_and_matched_subset_changes(tmp_path):
    prepared = r.build(tmp_path / "private-kernel")
    out = Path(prepared["private_directory"])
    metadata = json.loads((out / "kernel-metadata.json").read_bytes())
    source = (out / metadata["code_file"]).read_text()
    tree = ast.parse(source)
    pinned = [x for x in tree.body if isinstance(x, ast.Assign) and any(
        isinstance(y, ast.Name) and y.id == "MATCHED_Q4_IDS" for y in x.targets)]
    ids = set(ast.literal_eval(pinned[0].value.args[0]))
    assert len(pinned) == 1
    assert ids == set(r._source_bound_rows())
    assert prepared["matched_count"] == 128
    assert prepared["q8_model_sha256"] == r.Q8_SHA256
    assert metadata["is_private"] is True
    assert metadata["enable_internet"] is False
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["dataset_sources"] == list(r.KAGGLE_INPUTS)
    assert metadata["id"] == r.KAGGLE_ID
    assert "PROTOCOL_BYTES = 13041" in source
    assert "original Q8_0 GGUF" in source
    assert "Q8_GGUF_BYTES = 19748435968" in source
    assert "Q4_GGUF_SHA256" not in source
    assert "P.Q4_GGUF_FILENAME" not in source
    assert "original_scientific_gen0" in prepared
    assert prepared["official_hle_score_claimed"] is False
    assert "cgroup_unused_gib'] < 10" in source
    assert "verified_native_offload_41_of_41" in source
    assert "P.MAX_COMPLETION_TOKENS != 8192 or P.TEMPERATURE != 0" in source
    assert "R['matched_q4_row_sha256']" not in source  # source is metadata key, not mutable result update
    compile(source, "private_kaggle_q8_hle", "exec")


def test_no_existing_kaggle_directory_is_replaced(tmp_path):
    target = tmp_path / "already-here"
    target.mkdir()
    with pytest.raises(ValueError, match="exclusive"):
        r.build(target)


def test_reject_tampered_q4_source_hash(tmp_path, monkeypatch):
    forged = tmp_path / "forged.py"
    forged.write_bytes(r.Q4_SOURCE.read_bytes() + b"\n# tampered\n")
    monkeypatch.setattr(r, "Q4_SOURCE", forged)
    with pytest.raises(ValueError, match="source SHA changed"):
        r.build(tmp_path / "private")
    assert not (tmp_path / "private").exists()


def test_reject_tampered_q4_completed_rows_before_output(tmp_path, monkeypatch):
    rows = tmp_path / "rows"
    rows.mkdir()
    (rows / "reap-q4-hle-textonly-forged.jsonl").write_text('{"id":"fake"}\n')
    monkeypatch.setattr(r, "Q4_ROWS_DIR", rows)
    with pytest.raises(ValueError, match="row SHA mismatch"):
        r.build(tmp_path / "private")
    assert not (tmp_path / "private").exists()
