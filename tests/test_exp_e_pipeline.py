from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch", reason="the exp_e pipeline imports torch via runtime_eval")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "chowder_batch"))
sys.path.insert(0, str(ROOT / "src"))

from exp_e_corpus import (
    RetrievalSubsystem,
    build_verified_markdown_corpus,
    corpus_manifest,
    evaluate_retrieval,
    ingest_verified_documents,
    write_verified_markdown_corpus,
)
from exp_e_pipeline import (
    build_batch010_dataset,
    repair_trajectory_row,
    run_mixed_pipeline,
    run_repair_pipeline,
)
from exp_e_run import grade_citation
from chowder.runtime_eval import RuntimeTask, _is_green


def _record(doc_id: str, text: str, uri: str | None = None) -> dict:
    return {
        "doc_id": doc_id,
        "title": f"Document {doc_id}",
        "text": text,
        "source_uri": uri or f"https://example.test/{doc_id}",
        "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def test_ingest_checks_exact_text_hashes_and_rejects_duplicate_or_empty_docs():
    first = _record("a", "Exact source text")
    assert ingest_verified_documents([first])[0]["text"] == "Exact source text"
    with pytest.raises(ValueError, match="hash mismatch"):
        ingest_verified_documents([{**first, "text": "Changed after hash"}])
    with pytest.raises(ValueError, match="duplicate document text"):
        ingest_verified_documents([first, _record("b", "Exact source text")])
    with pytest.raises(ValueError, match="nonempty text"):
        ingest_verified_documents([_record("blank", "  ")])
    with pytest.raises(ValueError, match="hash mismatch"):
        corpus_manifest([{**first, "source_sha256": "seed-manual-verification"}])


def test_markdown_chunker_preserves_exact_sourced_text_and_never_overwrites(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    source = root / "facts.md"
    text = "## Verified notes\n\n" + ("An exact, independently sourced statement about tested behavior.\n" * 8)
    # Write exact bytes so line endings match on every platform: the chunker
    # preserves the source bytes verbatim (including CRLF on Windows).
    source.write_bytes(text.encode("utf-8"))
    docs = build_verified_markdown_corpus(root, min_chars=60, max_chars=140)
    assert docs
    assert all(doc["text"] in text for doc in docs)
    assert all(doc["source_uri"].startswith("facts.md#L") for doc in docs)
    assert all(doc["source_file_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest() for doc in docs)

    output = tmp_path / "corpus.json"
    with pytest.raises(ValueError, match="only .* verified .* chunks found"):
        write_verified_markdown_corpus(root, output, min_documents=300, min_chars=60, max_chars=140)
    assert not output.exists()

    payload = write_verified_markdown_corpus(root, output, min_documents=1, min_chars=60, max_chars=140)
    assert payload["manifest"]["n_documents"] == len(docs)
    with pytest.raises(FileExistsError):
        write_verified_markdown_corpus(root, output, min_documents=1, min_chars=60, max_chars=140)


def _embedder(texts: list[str]) -> np.ndarray:
    vectors = []
    for text in texts:
        values = [0.0, 0.0, 0.0]
        lowered = text.lower()
        for index, token in enumerate(("apple", "banana", "carrot")):
            values[index] = float(token in lowered)
        if not any(values):
            values[0] = 1.0
        vectors.append(values)
    return np.asarray(vectors, dtype=np.float32)


def _retrieval_fixture():
    docs = [
        _record("apple_doc", "Apple fruit is red and sweet."),
        _record("banana_doc", "Banana fruit is yellow."),
        _record("carrot_doc", "Carrot is an orange vegetable."),
    ]
    train = [{"query_id": "train_a", "query": "apple facts", "relevant_doc_ids": "apple_doc"}]
    subsystem = RetrievalSubsystem(
        docs,
        sparse_train_queries=train,
        train_doc_ids={"apple_doc", "banana_doc"},
        embed=_embedder,
        sparse_epochs=2,
    )
    queries = train + [
        {"query_id": "test_c", "query": "carrot details", "relevant_doc_ids": "carrot_doc"}
    ]
    return docs, train, queries, subsystem


def test_sparse_retrieval_fit_and_eval_enforce_exact_disjoint_training_inputs():
    docs, train, queries, subsystem = _retrieval_fixture()
    result = evaluate_retrieval(
        subsystem, queries,
        train_query_ids={"train_a"}, holdout_query_ids={"test_c"},
        train_doc_ids={"apple_doc", "banana_doc"}, holdout_doc_ids={"carrot_doc"},
        methods=("bm25", "dense", "sparse_learned"), k=1,
    )
    assert result["methods"]["dense"]["queries"][0]["retrieved"] == ["carrot_doc"]
    assert result["sparse_train_query_ids"] == ["train_a"]
    assert result["sparse_training_mode"] == "train_query_dense_distillation"

    with pytest.raises(ValueError, match="cross-split gold"):
        evaluate_retrieval(
            subsystem,
            [*train, {"query_id": "test_c", "query": "carrot details", "relevant_doc_ids": "apple_doc"}],
            train_query_ids={"train_a"}, holdout_query_ids={"test_c"},
            train_doc_ids={"apple_doc", "banana_doc"}, holdout_doc_ids={"carrot_doc"},
        )
    with pytest.raises(ValueError, match="content/labels differ"):
        evaluate_retrieval(
            subsystem,
            [{**train[0], "query": "changed query"}, queries[1]],
            train_query_ids={"train_a"}, holdout_query_ids={"test_c"},
            train_doc_ids={"apple_doc", "banana_doc"}, holdout_doc_ids={"carrot_doc"},
        )
    legacy = RetrievalSubsystem(docs, embed=_embedder, sparse_epochs=1)
    with pytest.raises(ValueError, match="explicit train queries"):
        evaluate_retrieval(
            legacy, queries, train_query_ids={"train_a"}, holdout_query_ids={"test_c"},
            train_doc_ids={"apple_doc", "banana_doc"}, holdout_doc_ids={"carrot_doc"},
        )


def _tool(name: str, **arguments: str) -> str:
    fields = "".join(f"<arg_key>{key}</arg_key><arg_value>{value}</arg_value>" for key, value in arguments.items())
    return f"<tool_call>{name}{fields}</tool_call>"


def _repair_task(name: str = "repair_evolve") -> dict:
    return {
        "name": name,
        "question": "Repair app.py so it returns 2.",
        "initial": {"app.py": "def f(): return 1"},
        "target": "app.py",
        "expected_fix": "return 2",
        "test_count": 1,
        "test_success": "1 passed",
    }


def _repair_generator(*, solve: bool):
    actions = [
        _tool("write_file", path="app.py", content="def f(): return 2" if solve else "def f(): return 1"),
        _tool("run_tests"),
    ]
    index = {"n": 0}

    def generate(messages, *, task):
        action = actions[index["n"]]
        index["n"] += 1
        return action

    return generate


def test_repair_pipeline_escalates_only_after_red_and_reports_verified_result():
    task = _repair_task()
    called = {"small": 0, "teacher": 0}

    def small(messages, *, task):
        index = called["small"]
        called["small"] += 1
        return _tool("write_file", path="app.py", content="def f(): return 1") if index == 0 else _tool("run_tests") if index == 1 else "Still red."

    def teacher(messages, *, task):
        index = called["teacher"]
        called["teacher"] += 1
        return _tool("write_file", path="app.py", content="def f(): return 2") if index % 3 == 0 else _tool("run_tests") if index % 3 == 1 else "Verified."

    teacher.metadata = {"spec_type": "ngram-simple"}
    result = run_repair_pipeline(task, small_generate=small, teacher_ngram_generate=teacher, max_turns=6)
    assert result["teacher_escalated"] is True
    assert result["small_green"] is False
    assert result["green"] is True
    assert result["teacher_generation_used"] is True
    assert result["teacher_ngram_configured"] is True
    assert result["teacher_speculation_speedup_measured"] is False

    # A successful small attempt must not call the teacher at all.
    success_calls = {"small": 0, "teacher": 0}
    def successful_small(messages, *, task):
        index = success_calls["small"]
        success_calls["small"] += 1
        return _tool("write_file", path="app.py", content="def f(): return 2") if index == 0 else _tool("run_tests") if index == 1 else "Verified."
    def unused_teacher(messages, *, task):
        success_calls["teacher"] += 1
        return ""
    success = run_repair_pipeline(task, small_generate=successful_small, teacher_ngram_generate=unused_teacher, max_turns=4)
    assert success["green"] is True
    assert success_calls["teacher"] == 0


def test_batch010_export_requires_current_trace_and_blocks_heldout_names_or_content(tmp_path):
    evolve = _repair_task()
    # Held-out task must be genuinely distinct in content; a held-out row that
    # differs only by name is itself a contamination error.
    heldout = {
        **_repair_task("repair_held"),
        "question": "Repair app.py so it returns 3.",
        "initial": {"app.py": "def f(): return 3"},
        "expected_fix": "return 3",
    }

    def generator(messages):
        tool_messages = [message for message in messages if message.get("role") == "tool"]
        # Green-first check: report immediately once the latest observation is
        # green; otherwise run tests after any successful write.
        if tool_messages and _is_green(str(tool_messages[-1].get("content", ""))):
            return "Verified."
        if any(message.get("role") == "tool" and str(message.get("content", "")).startswith("OK") for message in messages):
            return _tool("run_tests")
        return _tool("write_file", path="app.py", content="def f(): return 2")

    task_obj = RuntimeTask(
        evolve["name"], evolve["question"], evolve["initial"], evolve["target"],
        evolve["expected_fix"], evolve["test_count"], evolve["test_success"],
    )
    from chowder.runtime_eval import run_live_benchmark
    benchmark = run_live_benchmark(generator, tasks=(task_obj,), harness="state_aware", max_turns=4)
    row = repair_trajectory_row(evolve, benchmark["tasks"][0]["trace"], split="evolve", harness="state_aware")
    assert row["green_verified"]
    # The supervised transcript is rendered tool-free (tool observations become
    # user turns) so any chat template can tokenize it; the trace keeps the
    # faithful roles. See tests/test_batch010_contract.py for the contract.
    assert all(message.get("role") != "tool" for message in row["messages"])
    assert any(_is_green(str(message.get("content", ""))) for message in row["messages"])
    assert any(
        entry.get("role") == "tool" and _is_green(str(entry.get("observation", "")))
        for entry in row["trace"]
    )

    output = tmp_path / "batch010.jsonl"
    assert build_batch010_dataset([row], evolve_tasks=[evolve], heldout_tasks=[heldout], output_path=output) == 1
    saved = json.loads(output.read_text(encoding="utf-8").strip())
    assert saved["task_name"] == "repair_evolve"
    with pytest.raises(FileExistsError):
        build_batch010_dataset([row], evolve_tasks=[evolve], heldout_tasks=[heldout], output_path=output)

    contaminated = {**row, "task_name": "repair_held"}
    with pytest.raises(ValueError, match="not on evolve split"):
        build_batch010_dataset([contaminated], evolve_tasks=[evolve], heldout_tasks=[heldout], output_path=tmp_path / "reject.jsonl")

    # Same content as the evolve task but a different name: exporting it must
    # trip the held-out content-contamination guard.
    same_content_heldout = {**_repair_task("renamed_heldout"), "name": "renamed_heldout"}
    renamed = {**row, "task_name": evolve["name"]}
    with pytest.raises(ValueError, match="duplicates held-out task content"):
        build_batch010_dataset([renamed], evolve_tasks=[evolve], heldout_tasks=[same_content_heldout], output_path=tmp_path / "reject-content.jsonl")
