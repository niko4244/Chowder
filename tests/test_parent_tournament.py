"""Parent tournament tests: real orchestration over a fake worker.

The real worker (`base_text_worker`) is exercised end-to-end in a
separate gated test; these tests prove the tournament's own logic —
spec mirroring, integrity/tokenizer gates, aggregation, persistence
anchoring, and the honest comparison — using a tiny fake worker so no
GPU or model load is involved.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import chowder.parent_tournament as pt
from chowder.parent_eval import ParentTokenizerEvidence, aggregate_parent_result
from chowder.parent_suite_content import materialize_protected_suites
from chowder.registry import RunRegistry


# ---------------------------------------------------------------------------
# Fixtures: two tiny fake parents with full-mode manifests
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.usefixtures("fake_torch_env")


@pytest.fixture
def fake_torch_env(monkeypatch, tmp_path):
    """Point the module's parent constants at tiny local dirs; fake the worker.

    The fake worker writes a real `eval-result.json` in the worker shape,
    so `_run_worker`'s real subprocess path is NOT used (we monkeypatch
    `_run_worker` itself); everything around it — gates, aggregation,
    persistence, digests — runs for real.
    """
    def make_parent(label: str, revision: str) -> pt.LocalParent:
        root = tmp_path / label
        root.mkdir(parents=True)
        (root / "config.json").write_text("{}", encoding="utf-8")
        (root / "tokenizer.json").write_text('{"x": 1}', encoding="utf-8")
        (root / "model-00001-of-00001.safetensors").write_bytes(b"fake weights")
        from chowder.local_model_manifest import build_local_model_manifest, write_manifest_file

        manifest = build_local_model_manifest(root, mode="full")
        manifest_path = root.parent / f"{label}.manifest.json"
        write_manifest_file(manifest, manifest_path)
        return pt.LocalParent(
            label=label, revision=revision, local_path=str(root), manifest_path=str(manifest_path)
        )

    parent_a = make_parent("parent-a", "1d4bf0f2" + "0" * 56)
    parent_b = make_parent("parent-b", "404ea47a" + "0" * 56)

    calls: list[dict] = []

    def fake_worker(spec_payload, run_dir, *, timeout_seconds):
        calls.append(spec_payload)
        # The real worker writes per-suite prediction files; mirror that so
        # the digest evidence path runs for real.
        for suite in spec_payload["suites"]:
            (Path(run_dir) / f"predictions-{suite['name']}.jsonl").write_text(
                '{"score": 1.0}\n', encoding="utf-8"
            )
        metrics = {suite["name"]: 0.5 for suite in spec_payload["suites"]}
        result = {
            "metrics": metrics,
            "suites": {
                suite["name"]: {"rows": 6, "scoring": "normalized_exact_match"}
                for suite in spec_payload["suites"]
            },
            "runtime": {"device": "fake:0", "gpu_count": 0},
            "versions": {"torch": "fake", "transformers": "fake"},
            "model_provenance": {"requested_base_model": spec_payload["base_model"]},
        }
        result["wall_seconds"] = 1.0
        result["peak_gpu_mib_sampled"] = 0
        (Path(run_dir) / "eval-result.json").write_text(json.dumps(result), encoding="utf-8")
        return result

    monkeypatch.setattr(pt, "_run_worker", fake_worker)
    # Tokenizer evidence: patched to measured-style evidence without loading
    # a real tokenizer (the real path is exercised separately).
    # Same-family parents share tokenizer assets in reality (both A and B
    # are Qwen3.8 checkpoints); the fake identity is therefore label- and
    # revision-independent. The mismatch test overrides it explicitly.
    monkeypatch.setattr(
        pt,
        "tokenizer_evidence",
        lambda parent, offline=True: ParentTokenizerEvidence(
            tokenizer_class="FakeTokenizer",
            vocab_size=151936,
            identity_sha256="c" * 64,
        ),
    )
    yield {"parents": (parent_a, parent_b), "calls": calls}


@pytest.fixture
def frozen_root(tmp_path):
    materialize_protected_suites(tmp_path / "protected")
    return tmp_path / "protected"


def test_local_parent_validation(tmp_path):
    with pytest.raises(pt.ParentTournamentError, match="local dir missing"):
        pt.LocalParent(label="x", revision="r", local_path=str(tmp_path / "nope"), manifest_path="m")
    with pytest.raises(pt.ParentTournamentError, match="manifest missing"):
        pt.LocalParent(
            label="x",
            revision="r",
            local_path=str(tmp_path),
            manifest_path=str(tmp_path / "nope.json"),
        )


def test_tournament_end_to_end_fake_worker(fake_torch_env, frozen_root, tmp_path):
    parent_a, parent_b = fake_torch_env["parents"]
    with RunRegistry(tmp_path / "registry.db") as registry:
        bundle = pt.run_tournament(
            registry,
            (parent_a, parent_b),
            frozen_root,
            output_root=tmp_path / "runs",
        )
    # Everything real about the bundle except model execution.
    assert bundle["suite_count"] == 9
    assert len(bundle["runs"]) == 2
    comparison = bundle["comparison"]
    assert comparison["left"] == "parent-a"
    assert comparison["right"] == "parent-b"


def test_tournament_requires_two_parents(fake_torch_env, frozen_root, tmp_path):
    parent_a, _ = fake_torch_env["parents"]
    with RunRegistry(tmp_path / "registry.db") as registry:
        with pytest.raises(pt.ParentTournamentError, match="at least two"):
            pt.run_tournament(registry, (parent_a,), tmp_path, output_root=tmp_path / "runs")


def test_tokenizer_mismatch_fails_closed(fake_torch_env, frozen_root, tmp_path, monkeypatch):
    parent_a, parent_b = fake_torch_env["parents"]

    def mismatched(parent, offline=True):
        digest = parent.revision.replace("0", "a")
        if parent.label == "parent-b":
            digest = "b" * 64
        return ParentTokenizerEvidence(
            tokenizer_class="FakeTokenizer", vocab_size=151936, identity_sha256=digest
        )

    monkeypatch.setattr(pt, "tokenizer_evidence", mismatched)
    with RunRegistry(tmp_path / "registry.db") as registry:
        with pytest.raises(Exception, match="tokeniz"):
            pt.run_tournament(registry, (parent_a, parent_b), frozen_root, output_root=tmp_path / "runs")


def test_integrity_failure_stops_evaluation(fake_torch_env, frozen_root, tmp_path):
    parent_a, parent_b = fake_torch_env["parents"]
    # Mutate a shard behind parent B's manifest -> verification must fail
    # BEFORE the fake worker is invoked for B.
    (Path(parent_b.local_path) / "model-00001-of-00001.safetensors").write_bytes(
        b"tampered weights"
    )
    calls = fake_torch_env["calls"]
    with RunRegistry(tmp_path / "registry.db") as registry:
        with pytest.raises(pt.ParentTournamentError, match="integrity FAILED"):
            pt.run_tournament(registry, (parent_a, parent_b), frozen_root, output_root=tmp_path / "runs")
    assert len(calls) == 0  # no worker run at all: gates precede all GPU work


def test_comparison_classification_is_honest(fake_torch_env, frozen_root, tmp_path):
    parent_a, parent_b = fake_torch_env["parents"]
    # B sweeps A by ~2 items per suite: 0.5 -> 0.833 on every suite.
    original = pt._run_worker

    def b_sweeps(spec_payload, run_dir, *, timeout_seconds):
        result = original(spec_payload, run_dir, timeout_seconds=timeout_seconds)
        if "parent-b" in spec_payload["base_model"]:
            result = dict(result)
            result["metrics"] = {k: 5 / 6 for k in result["metrics"]}
        return result

    import chowder.parent_tournament as ptmod

    ptmod._run_worker = b_sweeps
    try:
        with RunRegistry(tmp_path / "registry.db") as registry:
            bundle = pt.run_tournament(
                registry, (parent_a, parent_b), frozen_root, output_root=tmp_path / "runs"
            )
    finally:
        ptmod._run_worker = original
    for dimension, row in bundle["comparison"]["dimensions"].items():
        assert row["classification"] == "clear-difference"
        # compare_reports rounds deltas to 6 decimals for stable evidence.
        assert abs(row["delta_right_minus_left"] - 1 / 3) < 1e-6


def test_real_worker_payload_mirrors_suite_fields(fake_torch_env, frozen_root, tmp_path):
    """The serialized suite dicts must carry exactly EvalSuiteSpec's fields."""
    parent_a, parent_b = fake_torch_env["parents"]
    calls = fake_torch_env["calls"]
    with RunRegistry(tmp_path / "registry.db") as registry:
        pt.run_tournament(registry, (parent_a, parent_b), frozen_root, output_root=tmp_path / "runs")
    assert calls, "fake worker never called"
    for payload in calls:
        for suite in payload["suites"]:
            assert set(suite) == {
                "name",
                "dataset",
                "prompt_field",
                "expected_field",
                "scoring",
                "max_new_tokens",
                "use_chat_template",
            }
            assert suite["use_chat_template"] is True
