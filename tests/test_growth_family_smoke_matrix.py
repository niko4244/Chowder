"""Smoke matrix: every registered intervention family proves it is runnable.

A family's maturity label is a claim about what the growth loop may propose.
This file makes a second, weaker claim checkable: that the mechanism behind
the label can actually be invoked. Every family declares, in
``InterventionFamily.implementation``, the in-repo artifacts that implement
and measure it; the matrix below picks each family's *cheapest* declared
artifact -- the one that runs on a laptop, no weights, no GPU -- and invokes
one mechanism from it. The outcome is recorded in
``evidence/family_smoke_matrix.json``.

The matrix is a coverage contract, not a performance test: a family with no
smoke row, a row pointing at an artifact its family does not declare, or a
mechanism that raises all fail here. Rows whose mechanism needs torch are
skipped -- and recorded as skipped -- where torch is not installed, so the
light CI leg stays honest instead of silently green. The record is
load-bearing: ``families_for_campaign`` and ``generate_hypotheses`` refuse
any family whose row is missing, skipped, or stale, so this file is where
the right to propose is earned.

The REJECTED family is represented by the guard it ships: the confidence
router's fail-closed margin-shift check must still run and still refuse. The
rejection itself is proven in the registry tests; this file only proves the
mechanism behind the label exists and executes.
"""
from __future__ import annotations

import importlib
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pytest

from chowder.growth.interventions import (
    FAMILY_SMOKE_RECORD_VERSION,
    family_registry,
    family_smoke_declaration_digest,
)

ROOT = Path(__file__).resolve().parents[1]
RECORD_PATH = ROOT / "evidence" / "family_smoke_matrix.json"


def _batch_module(module_name: str):
    """Import a chowder_batch driver, which expects its directory on sys.path."""
    batch_dir = str(ROOT / "chowder_batch")
    if batch_dir not in sys.path:
        sys.path.insert(0, batch_dir)
    return importlib.import_module(module_name)


@dataclass(frozen=True)
class FamilySmoke:
    """One matrix row: a family, its cheapest declared artifact, and a mechanism."""

    family_id: str
    artifact: str
    mechanism: str
    invoke: Callable[[Path], str]


# --- one smoke per family ----------------------------------------------------


def _smoke_sft_curriculum(tmp: Path) -> str:
    del tmp
    from chowder.backends.training_data import _validate_chat_messages

    row = _validate_chat_messages(
        [
            {"role": "user", "content": "2 + 2?"},
            {"role": "assistant", "content": "4"},
        ],
        row_index=0,
    )
    assert [turn["role"] for turn in row] == ["user", "assistant"], row
    return "validated a 2-turn chat row with a trainable assistant span"


def _smoke_replay_balanced(tmp: Path) -> str:
    del tmp
    from chowder.backends.training_data import _replay_sample_count

    sampled = _replay_sample_count(64, 32, 0.25)
    assert sampled == 16, sampled
    return "64 primary rows at ratio 0.25 sample 16 of 32 replay rows"


def _smoke_adapter_continuation(tmp: Path) -> str:
    from chowder.adapter_guard import assert_adapter_is_live, saved_adapter_keys

    key = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"
    header = json.dumps({key: {"dtype": "F32", "shape": [4, 4], "data_offsets": [0, 64]}}).encode("utf-8")
    (tmp / "adapter_model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header)

    class _Parameter:
        def detach(self):
            return self

        def float(self):
            return self

        def abs(self):
            return self

        def max(self):
            return 0.25

    class _Model:
        def named_parameters(self):
            return [(key.replace("lora_B.weight", "lora_B.default.weight"), _Parameter())]

    assert saved_adapter_keys(tmp) == {key}
    report = assert_adapter_is_live(_Model(), tmp)
    assert report["matched_keys"] == 1 and report["lora_B_nonzero"] == 1, report
    return "adapter key overlap 1/1 and verified nonzero lora_B: the adapter is live"


def _smoke_conditional_ffn(tmp: Path) -> str:
    del tmp
    torch = pytest.importorskip("torch", reason="the conditional-compute prototype is a torch module")
    from torch import nn

    from chowder.conditional_compute import ConditionalFFN, FixedSkipRouter

    generator = torch.Generator().manual_seed(0)
    hidden = torch.randn(1, 4, 4, generator=generator)
    block = ConditionalFFN(nn.Linear(4, 4, bias=False), FixedSkipRouter(skip_every=2, skip_phase=0))
    with torch.no_grad():
        output, stats = block(hidden, return_routing_stats=True)
    assert output.shape == hidden.shape, (output.shape, hidden.shape)
    assert stats.tokens == 4 and stats.executed_tokens == 2, stats
    return "parity skip routed 2 of 4 tokens; the output keeps its shape"


def _smoke_hybrid_lm(tmp: Path) -> str:
    del tmp
    torch = pytest.importorskip("torch", reason="experiment D's hybrid LM is a torch module")
    from chowder.experimental_hybrid_lm import HybridLanguageModel, HybridLMConfig

    config = HybridLMConfig(
        vocab_size=23,
        hidden_size=16,
        num_hidden_layers=2,
        layer_types=("mamba2", "attention"),
        num_attention_heads=4,
        num_key_value_heads=2,
        mamba_num_heads=4,
        mamba_expand=2,
        mamba_state_size=3,
        mamba_conv_kernel=3,
        ffn_type="dense",
        intermediate_size=24,
        num_experts=3,
        experts_per_token=2,
        expert_intermediate_size=8,
        capacity_factor=None,
        per_layer_embedding_dim=0,
        tie_word_embeddings=True,
        max_position_embeddings=32,
    )
    model = HybridLanguageModel(config).eval()
    with torch.no_grad():
        logits = model(torch.tensor([[2, 4, 6, 8]])).logits
    assert tuple(logits.shape) == (1, 4, 23), tuple(logits.shape)
    return "mamba2+attention hybrid forward produced logits (1, 4, 23)"


def _smoke_low_rank_vocab(tmp: Path) -> str:
    del tmp
    torch = pytest.importorskip("torch", reason="the low-rank vocabulary factorization is a torch module")
    from chowder.low_rank_vocab import factorize_svd

    weight = torch.randn(16, 8, generator=torch.Generator().manual_seed(0))
    result = factorize_svd(weight, 4, prefer_cuda=False)
    assert result.u.shape == (16, 4) and result.v.shape == (4, 8), (result.u.shape, result.v.shape)
    assert 0.0 <= result.relative_frobenius_error < 1.0, result.relative_frobenius_error
    return (
        f"rank-4 factorization of 16x8: energy {result.energy_captured:.3f}, "
        f"relative error {result.relative_frobenius_error:.3f}"
    )


def _smoke_ptq(tmp: Path) -> str:
    del tmp
    exp_f = _batch_module("exp_f_ptq_margin")
    signal = exp_f.compare_margin_signal(
        [
            {"task": "t1", "margin": 1.0, "correct": True},
            {"task": "t2", "margin": -0.5, "correct": False},
        ],
        [
            {"task": "t1", "margin": 1.1, "correct": True},
            {"task": "t2", "margin": -0.4, "correct": False},
        ],
        quant_config="int8_weight_only",
    )
    assert signal["n_tasks"] == 2, signal
    assert signal["mean_margin_shift"] == pytest.approx(0.1, abs=1e-9), signal
    return "paired margin comparison over 2 tasks: mean shift +0.100, accuracy delta +0.000"


def _smoke_retrieval(tmp: Path) -> str:
    del tmp
    pytest.importorskip("torch", reason="the exp_e corpus retrievers are torch modules")
    import numpy as np

    corpus = _batch_module("exp_e_corpus")
    docs = [
        {"doc_id": "d1", "title": "Alpha", "text": "alpha beta"},
        {"doc_id": "d2", "title": "Gamma", "text": "gamma delta"},
        {"doc_id": "d3", "title": "Epsilon", "text": "epsilon zeta"},
        {"doc_id": "d4", "title": "Eta", "text": "eta theta"},
    ]
    # BM25's IDF is zero when a term appears in exactly half a two-document
    # corpus, which ties every document; four documents keep the ranking real.
    subsystem = corpus.RetrievalSubsystem(
        docs,
        seed=0,
        sparse_epochs=1,
        embed=lambda texts: np.eye(len(docs))[: len(texts)],
    )
    picked, _latency_ms = subsystem.retrieve("alpha", method="bm25", k=1)
    assert [doc["doc_id"] for doc in picked] == ["d1"], picked
    # No latency in the recorded outcome: the record is committed evidence
    # and must be byte-reproducible, and a timing would churn it every run.
    return "bm25 top-1 over a 4-document corpus: d1 ranked first"


def _smoke_speculative(tmp: Path) -> str:
    del tmp
    pytest.importorskip("torch", reason="the exp_e speculative decoder is a torch module")
    speculative = _batch_module("exp_e_speculative")
    draft = speculative.draft_from_prompt([1, 2, 3, 1, 2, 3, 1, 2, 3], ngram=3, max_draft=6)
    assert draft == [1, 2, 3, 1, 2, 3], draft
    return "prompt-lookup ngram=3 drafted 6 tokens from an earlier occurrence"


def _smoke_confidence_routing(tmp: Path) -> str:
    del tmp
    confidence = _batch_module("exp_e_confidence")
    verdict = confidence.validate_quantized_margin_shift(1.0, 1.2, max_quantized_margin_shift=0.05)
    assert verdict["quantized_margin_shift_fails_closed"] is True, verdict
    return "an unvalidated +0.200 margin shift blocks the small route (fails closed)"


def _smoke_runtime_harness_repair(tmp: Path) -> str:
    del tmp
    from chowder.runtime_eval import RuntimeTask, _run_task

    task = RuntimeTask(
        name="smoke-single-file",
        goal="Repair app.py.",
        initial={"app.py": "def f():\n    return 1\n"},
        target="app.py",
        expected_fix="return 2",
        test_count=2,
        test_success="2 passed",
    )
    fix = "def f():\n    return 2\n"

    def generate(messages: list[dict[str, str]]) -> str:
        turn = sum(1 for message in messages if message.get("role") == "assistant")
        if turn == 0:
            return (
                "<tool_call>write_file"
                "<arg_key>path</arg_key><arg_value>app.py</arg_value>"
                f"<arg_key>content</arg_key><arg_value>{fix}</arg_value>"
                "</tool_call>"
            )
        if turn == 1:
            return "<tool_call>run_tests</tool_call>"
        return "Repaired app.py; the suite is green."

    result = _run_task(generate, task, 4, harness="state_aware+recovery")
    assert result["green_seen"] is True, result
    assert result["nonexistent_reads"] == 0, result
    return (
        f"state_aware+recovery single-file repair: green={result['green_seen']}, "
        f"reward={result['reward']}, execution cost {result['execution_cost']}"
    )


def _smoke_harness_evolution(tmp: Path) -> str:
    del tmp
    from chowder.harness_evolution import HarnessMetrics, accept_candidate

    incumbent = HarnessMetrics(score=0.50, cost=1.0, green_rate=1.0, nonexistent_read_rate=0.0)
    accepted, reason = accept_candidate(
        incumbent,
        HarnessMetrics(score=0.80, cost=1.2, green_rate=1.0, nonexistent_read_rate=0.0),
        noise_band=0.05,
        beta0=0.10,
        beta1=0.50,
    )
    assert accepted is True, reason
    refused, refusal = accept_candidate(
        incumbent,
        HarnessMetrics(score=0.51, cost=1.0, green_rate=1.0, nonexistent_read_rate=0.0),
        noise_band=0.05,
        beta0=0.10,
        beta1=0.50,
    )
    assert refused is False and "noise" in refusal, refusal
    return "the gate accepted +0.30 (cost +20%) and refused +0.01 inside the 0.05 noise band"


def _smoke_teacher_distillation(tmp: Path) -> str:
    del tmp
    restricted = _batch_module("exp_b_restricted_python")
    namespace = restricted.candidate_namespace("def double(value):\n    return value * 2\n")
    assert namespace["double"](21) == 42
    return "the restricted interpreter executed the candidate's double(21) and returned 42"


SMOKE_MATRIX: tuple[FamilySmoke, ...] = (
    FamilySmoke(
        "training.sft-curriculum",
        "src/chowder/backends/training_data.py",
        "_validate_chat_messages: the chat-row contract behind the curriculum",
        _smoke_sft_curriculum,
    ),
    FamilySmoke(
        "training.replay-balanced",
        "src/chowder/backends/training_data.py",
        "_replay_sample_count: the replay mixing arithmetic",
        _smoke_replay_balanced,
    ),
    FamilySmoke(
        "training.adapter-continuation",
        "src/chowder/adapter_guard.py",
        "assert_adapter_is_live: refuse a loaded-but-inert adapter",
        _smoke_adapter_continuation,
    ),
    FamilySmoke(
        "architecture.conditional-ffn",
        "src/chowder/conditional_compute.py",
        "ConditionalFFN + FixedSkipRouter: a token-level skip forward",
        _smoke_conditional_ffn,
    ),
    FamilySmoke(
        "architecture.hybrid-lm",
        "src/chowder/experimental_hybrid_lm.py",
        "HybridLanguageModel: a tiny mamba2+attention forward",
        _smoke_hybrid_lm,
    ),
    FamilySmoke(
        "compression.low-rank-vocab",
        "src/chowder/low_rank_vocab.py",
        "factorize_svd: a rank-4 factorization of a 16x8 matrix",
        _smoke_low_rank_vocab,
    ),
    FamilySmoke(
        "compression.ptq",
        "chowder_batch/exp_f_ptq_margin.py",
        "compare_margin_signal: paired BF16 vs quantized margin deltas",
        _smoke_ptq,
    ),
    FamilySmoke(
        "inference.retrieval",
        "chowder_batch/exp_e_corpus.py",
        "RetrievalSubsystem.retrieve(method='bm25') over a small corpus",
        _smoke_retrieval,
    ),
    FamilySmoke(
        "inference.speculative",
        "chowder_batch/exp_e_speculative.py",
        "draft_from_prompt: longest-suffix n-gram lookup",
        _smoke_speculative,
    ),
    FamilySmoke(
        "inference.confidence-routing",
        "chowder_batch/exp_e_confidence.py",
        "validate_quantized_margin_shift: the fail-closed shift guard",
        _smoke_confidence_routing,
    ),
    FamilySmoke(
        "runtime.harness-repair",
        "src/chowder/runtime_eval.py",
        "_run_task(harness='state_aware+recovery') on a scripted single-file repair",
        _smoke_runtime_harness_repair,
    ),
    FamilySmoke(
        "runtime.harness-evolution",
        "src/chowder/harness_evolution.py",
        "accept_candidate: the noise-band / cost acceptance gate",
        _smoke_harness_evolution,
    ),
    FamilySmoke(
        "training.teacher-distillation",
        "chowder_batch/exp_b_restricted_python.py",
        "candidate_namespace: restricted execution of a candidate function",
        _smoke_teacher_distillation,
    ),
)

RECORDED: dict[str, dict[str, str]] = {}


@pytest.mark.parametrize("case", SMOKE_MATRIX, ids=[case.family_id for case in SMOKE_MATRIX])
def test_each_family_runs_its_cheapest_mechanism(case: FamilySmoke, tmp_path: Path) -> None:
    try:
        outcome = case.invoke(tmp_path)
    except pytest.skip.Exception as exc:  # heavy dependency absent: record the skip, do not hide it
        RECORDED[case.family_id] = {"status": "skipped", "outcome": f"dependency unavailable: {exc}"}
        pytest.skip(f"{case.family_id}: {exc}")
    assert outcome.strip(), "a smoke row must record what it observed"
    RECORDED[case.family_id] = {"status": "ran", "outcome": outcome}


def test_the_matrix_covers_the_registry_and_smokes_declared_artifacts() -> None:
    families = {family.family_id: family for family in family_registry()}
    ids = [case.family_id for case in SMOKE_MATRIX]
    assert len(ids) == len(set(ids)), "a family must have exactly one smoke row"
    assert set(ids) == set(families), (
        "the smoke matrix must cover the registry exactly: register a row with the family"
    )
    for case in SMOKE_MATRIX:
        assert case.artifact in families[case.family_id].implementation, (
            f"{case.family_id} smokes {case.artifact}, which the family does not declare"
        )


def test_the_smoke_matrix_writes_its_record() -> None:
    families = {family.family_id: family for family in family_registry()}
    missing = sorted(set(families) - set(RECORDED))
    assert not missing, f"families with no recorded smoke outcome: {missing}"
    unrunnable = sorted(
        family_id for family_id, row in RECORDED.items() if row["status"] != "ran"
    )
    if unrunnable:
        pytest.skip(
            f"this environment could not run {len(unrunnable)} smoke row(s) "
            f"({', '.join(unrunnable)}); the committed record is left untouched "
            "so a skipped row can never certify runnability"
        )
    rows = [
        {
            "family_id": case.family_id,
            "maturity": families[case.family_id].maturity.value,
            "artifact": case.artifact,
            "mechanism": case.mechanism,
            "status": RECORDED[case.family_id]["status"],
            "outcome": RECORDED[case.family_id]["outcome"],
            "declaration_digest": family_smoke_declaration_digest(
                families[case.family_id]
            ),
        }
        for case in SMOKE_MATRIX
    ]
    RECORD_PATH.write_text(
        json.dumps(
            {
                "record_version": FAMILY_SMOKE_RECORD_VERSION,
                "note": (
                    "Written by tests/test_growth_family_smoke_matrix.py: the cheapest "
                    "declared mechanism of every registered intervention family, invoked "
                    "once and recorded. The declaration_digest binds the row to the "
                    "family declaration it proves; the proposal gates refuse a row "
                    "that is missing, skipped or stale."
                ),
                "family_count": len(rows),
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
