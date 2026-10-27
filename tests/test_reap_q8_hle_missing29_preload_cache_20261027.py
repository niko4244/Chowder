"""CPU-only source and adversarial checks; never runs generated HLE or GPU."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from chowder_batch import reap_q8_hle_missing29_preload_cache_20261027 as m


@pytest.fixture(scope="module")
def originals() -> None:
    if not m.original.V4.is_dir() or not m.original.V3_SCRIPT.is_file():
        pytest.skip("source-locked original private HLE research not mounted")


def test_preparation_pins_original_source_and_retains_floors(tmp_path: Path, originals: None):
    output = tmp_path / "fresh_only"
    receipt = m.prepare(output)
    source = (output / m.SCRIPT_NAME).read_bytes()
    assert m.sha(source) == receipt["new_generated_script_sha256"]
    assert receipt["original_generated_script_sha256"] == m.BASE_SCRIPT_SHA256
    assert receipt["original_missing29_source_id_sha256"] == m.original.MISSING_IDS_SHA
    assert receipt["unchanged_original_q8_model_sha256"] == m.original.frozen.Q8_SHA256
    assert receipt["unchanged_runtime_cgroup_floor_gib"] == 10
    assert receipt["unchanged_postload_cgroup_floor_gib"] == 14
    assert receipt["official_hle_judge_score"] == "UNAVAILABLE"
    assert receipt["chosen"] is None and receipt["release_approved"] is False
    assert receipt["status"] == "PREPARED_ONLY_UNSUBMITTED_NONOFFICIAL_HLE_MISSING29"
    text = source.decode()
    ast.parse(text)
    assert text.count("check_runtime(h)") >= 2
    assert "if h['cgroup_unused_gib'] is None or h['cgroup_unused_gib'] < 10:" in text
    assert "if R['after_postload_reclaim']['cgroup_unused_gib'] < 14:" in text
    assert "if 'offloaded 41/41 layers to GPU' not in native_start:" in text
    assert "weights.stat().st_size != Q8_GGUF_BYTES" in text
    assert "10 <= h['cgroup_unused_gib'] < 11.5" in text
    assert "os.posix_fadvise(r_handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)" in text
    assert "MISSING_Q8_IDS" in text and "len(ordered) != 29" in text
    assert "P.grade(text, item['answer'], item['answer_type'])" in text
    assert "P.SERVING_FLAGS" in text and "offloaded 41/41 layers to GPU" in text
    meta = json.loads((output / "kernel-metadata.json").read_bytes())
    assert meta["is_private"] is True and meta["enable_internet"] is False
    assert meta["id"] == m.original.KAGGLE_ID
    with pytest.raises(ValueError, match="exclusive"):
        m.prepare(output)


def test_source_drift_and_ambiguous_patch_refused(originals: None, tmp_path: Path):
    output = tmp_path / "original_only"
    m.original.prepare(output)
    raw = (output / m.SCRIPT_NAME).read_bytes()
    assert m.sha(raw) == m.BASE_SCRIPT_SHA256
    with pytest.raises(ValueError, match="SHA changed"):
        m.patch_script(raw + b"\n# forged\n")
    assert m.patch_script(raw) != raw


def _sample(cgroup=11.0, ram=24.0, gpu=(6000, 6000)):
    return {"cgroup_unused_gib": cgroup, "physical_free_gib": ram,
            "gpu_free_mib": list(gpu)}


@pytest.mark.parametrize("event", [
    {"before": _sample(cgroup=9.99), "after": _sample()},
    {"before": _sample(), "after": _sample(cgroup=9.99)},
    {"before": _sample(ram=6.99), "after": _sample()},
    {"before": _sample(), "after": _sample(gpu=(4095, 6000))},
    {"before": _sample(), "after": _sample(cgroup=float("nan"))},
    {"before": _sample(), "after": _sample(cgroup="10")},
    {"before": _sample(), "after": _sample(gpu=(True, 6000))},
    {"before": _sample()},
    "fake",
])
def test_resource_attestation_rejects_bad_samples(monkeypatch, tmp_path: Path, event):
    monkeypatch.setattr(m.original, "_object", lambda _: {
        "preload_advisory_version": m.PATCH_ID,
        "during_load_clean_reclaims": [event],
    })
    monkeypatch.setattr(m.original, "audit_two_sessions", lambda _: {
        "status": "TWO_NATIVE_SESSIONS_PAIRED_HEURISTIC_RESEARCH_NOT_OFFICIAL_HLE"
    })
    with pytest.raises(ValueError):
        m.audit(tmp_path)


def test_attestation_requires_profile_and_bounded_events(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(m.original, "audit_two_sessions", lambda _: {"status": "TEST"})
    for receipt in (
        {},
        {"preload_advisory_version": m.PATCH_ID},
        {"preload_advisory_version": m.PATCH_ID,
         "during_load_clean_reclaims": [{}] * 601},
    ):
        monkeypatch.setattr(m.original, "_object", lambda _, r=receipt: r)
        with pytest.raises(ValueError):
            m.audit(tmp_path)


def test_valid_synthetic_attestation_still_nonofficial(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(m.original, "_object", lambda _: {
        "preload_advisory_version": m.PATCH_ID,
        "during_load_clean_reclaims": [{"before": _sample(), "after": _sample(12)}],
    })
    monkeypatch.setattr(m.original, "audit_two_sessions", lambda _: {
        "status": "TWO_NATIVE_SESSIONS_PAIRED_HEURISTIC_RESEARCH_NOT_OFFICIAL_HLE",
        "official_hle_judge_score": "UNAVAILABLE", "scientific_selection": "BLOCKED",
        "chosen": None, "release_approved": False,
    })
    result = m.audit(tmp_path)
    assert result["during_load_clean_reclaim_events"] == 1
    assert result["official_hle_judge_score"] == "UNAVAILABLE"
    assert result["scientific_selection"] == "BLOCKED"
