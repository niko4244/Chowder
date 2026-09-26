"""One regression test per dataset-integrity defect the pilot corpus had.

Deterministic, CPU-only. Each test builds the smallest corpus that exhibits the
defect and asserts the corrected behaviour, so the defect can never come back
silently:

  1 identity          ids survive chunking and preparation
  2 grouping          all rows of one problem share one partition
  3 cross-source      an equivalent question in another source is quarantined
  4 candidates        a band collision below the threshold is cleared, not dropped
  5 context           continuation prompts carry the real preceding segment
  6 labels            the production audit rejects truncated/missing targets
  7 pinning           digests match the bytes on disk; rewrites are refused
  8 traceability      provenance order matches the emitted rows
"""
import importlib.util
import json
from pathlib import Path

import pytest

EXP = Path(__file__).resolve().parents[1] / "experiments" / "teacher_free_distill"


def load(name):
    spec = importlib.util.spec_from_file_location(name, EXP / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


prepare = load("prepare")
ot3 = load("ot3_subset")


def write_jsonl(path: Path, rows) -> Path:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8")
    return path


def catalog(path: Path, source_ids) -> Path:
    path.write_text(json.dumps({"sources": {
        sid: {"approved": True, "license": "test-only", "kind": "chat",
              "review_reference": "fixture", "revision": "rev-1"} for sid in source_ids}}),
        encoding="utf-8")
    return path


def block(tag: str, count: int = 60) -> str:
    """Distinct words per segment: no accidental near-duplicates in fixtures."""
    return " ".join(f"{tag}{index}" for index in range(count))


def long_trace(final_answer: str = "The final answer is 42.") -> str:
    """A trace with three conclusion seams, so chunking yields exactly 3 chunks."""
    return (block("alpha") + "\nCONCLUSION: alpha stage settled\n"
            + block("beta") + "\nCONCLUSION: beta stage settled\n"
            + block("gamma") + f"\nCONCLUSION: gamma stage settled\n{final_answer}\n")


def ot3_row(question: str, trace: str | None = None) -> dict:
    return {"conversations": [
        {"from": "human", "value": question},
        {"from": "gpt", "value": trace if trace is not None else long_trace()},
    ], "domain": "code", "difficulty": 8, "source": "fixture"}


def tiny_tokenizer():
    """A fast, deterministic tokenizer with a prefix-consistent chat template."""
    preflight = load("preflight")
    tok = preflight.make_tokenizer()
    tok.chat_template = (
        "{% for message in messages %}"
        "{{ message['role'] | capitalize }}: {{ message['content'] }}\n"
        "{% endfor %}"
    )
    return tok


def prepare_fixture(tmp_path, sources, *, dev_percent=10, chunker_kwargs=None,
                    near_dup_field="target", **kwargs):
    """Chunk one fixture row per source and prepare them into one dataset."""
    catalog_path = catalog(tmp_path / "catalog.json", list(sources))
    inputs = {}
    for source_id, question in sources.items():
        chunks = ot3.chunk_row(ot3_row(question), row_index=1, **(chunker_kwargs or {}))
        inputs[source_id] = write_jsonl(tmp_path / f"{source_id}.jsonl", chunks)
    out = tmp_path / "prepared"
    manifest = prepare.prepare(catalog_path, inputs, out, max_rows=50,
                               dev_percent=dev_percent, near_dup_field=near_dup_field,
                               **kwargs)
    return manifest, out


# --------------------------------------------------------------------------
# 1 identity
# --------------------------------------------------------------------------

def test_ids_survive_chunking_and_preparation(tmp_path):
    question = "How do I compute the answer to the problem"
    chunks = ot3.chunk_row(ot3_row(question), with_ids=True, preceding_context=True,
                           row_index=7)
    assert len(chunks) == 3
    q_id = ot3.question_id(question)
    r_id = ot3.response_id(question, long_trace())
    for index, chunk in enumerate(chunks):
        ids = chunk["ids"]
        assert ids["problem_id"] == q_id
        assert ids["teacher_response_id"] == r_id
        assert ids["chunk_id"].startswith(f"{q_id}:{index}:")
        assert ids["chunk_index"] == index and ids["chunk_total"] == 3
        assert ids["source_row_index"] == 7

    inp = write_jsonl(tmp_path / "in.jsonl", chunks)
    manifest = prepare.prepare(catalog(tmp_path / "catalog.json", ["ot3"]), {"ot3": inp},
                               tmp_path / "out", max_rows=50, near_dup_field="target")
    prov = manifest["provenance"]
    assert [p["problem_id"] for p in prov] == [q_id] * 3
    assert [p["chunk_id"] for p in prov] == [c["ids"]["chunk_id"] for c in chunks]
    assert [p["teacher_response_id"] for p in prov] == [r_id] * 3
    assert manifest["integrity"]["ids"] == {
        "rows_with_problem_id": 3, "rows_with_chunk_id": 3, "unique_problem_ids": 1,
        "max_rows_for_one_problem": 3,
    }


# --------------------------------------------------------------------------
# 2 grouping
# --------------------------------------------------------------------------

def test_all_rows_of_one_problem_share_one_partition(tmp_path):
    """The defect: chunks were split by hashing their own prompt.

    The premise is asserted, not assumed: a dev_percent is chosen that the
    legacy per-chunk groups *would* straddle, and the corrected grouping keeps
    the problem together at that same dev_percent.
    """
    question, dev_percent, chunks = _problem_whose_legacy_groups_straddle()
    legacy_hash = prepare.digest
    assert len({int(legacy_hash(f"ot3chunk:{ot3.task_key(question, long_trace())}:{i}")
                    [:8], 16) % 100 < dev_percent for i in range(len(chunks))}) == 2

    manifest, out = prepare_fixture(tmp_path, {"ot3": question}, dev_percent=dev_percent)
    assert manifest["integrity"]["split_integrity"]["problem_groups_crossing_splits"] == 0
    assert manifest["integrity"]["split_integrity"]["ok"] is True
    train = (out / "train.jsonl").read_text(encoding="utf-8").strip().splitlines()
    dev = (out / "dev.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert (len(train) == 3) != (len(dev) == 3), "a problem was split across partitions"
    # The first chunk carries no key, so it is linked back by the excerpt its
    # siblings quote; the two continuation chunks name the task directly.
    assert manifest["integrity"]["split_integrity"]["basis"] == {
        "embedded_task_key": 2, "recovered_parent_prefix": 1}
    assert manifest["integrity"]["split_integrity"]["unlinkable_rows"] == 0


def test_declared_problem_id_is_the_grouping_basis(tmp_path):
    manifest, _ = prepare_fixture(tmp_path, {"ot3": "A question about something"},
                                  chunker_kwargs={"with_ids": True})
    assert manifest["integrity"]["split_integrity"]["basis"] == {"declared_problem_id": 3}
    assert manifest["integrity"]["split_integrity"]["problem_groups_crossing_splits"] == 0
    assert manifest["provenance"][0]["problem_group"].startswith("problem:")


def test_legacy_continuation_prompts_group_by_their_embedded_task_key(tmp_path):
    """Chunk files written before ids exist still group by the task they name."""
    question = "A question that predates the id block"
    key = ot3.task_key(question, long_trace())
    rows = [
        {"conversations": [{"from": "human", "value": question},
                           {"from": "gpt", "value": block("first")}]},
        {"conversations": [{"from": "human",
                            "value": (f"Task {key}. Continue the reasoning from part 1 of 2. "
                                      f"Original question (excerpt): {question[:300]}")},
                           {"from": "gpt", "value": block("second")}]},
    ]
    inputs = {"ot3": write_jsonl(tmp_path / "legacy.jsonl", rows)}
    manifest = prepare.prepare(catalog(tmp_path / "catalog.json", ["ot3"]), inputs,
                               tmp_path / "out", max_rows=50, dev_percent=40,
                               near_dup_field="target")
    assert manifest["integrity"]["split_integrity"]["basis"] == {
        "embedded_task_key": 1, "recovered_parent_prefix": 1}
    assert manifest["integrity"]["split_integrity"]["problem_groups_crossing_splits"] == 0
    assert manifest["integrity"]["ids"]["rows_with_problem_id"] == 0
    assert manifest["integrity"]["split_integrity"]["unlinkable_rows"] == 0


def _problem_whose_legacy_groups_straddle():
    """Find a fixture whose legacy per-chunk groups land in different splits."""
    for variant in range(8):
        question = f"How do I compute the answer to the problem (variant {variant})"
        chunks = ot3.chunk_row(ot3_row(question), row_index=1)
        if len(chunks) < 3:
            continue
        key = ot3.task_key(question, long_trace())
        mods = sorted({int(prepare.digest(f"ot3chunk:{key}:{i}")[:8], 16) % 100
                       for i in range(len(chunks))})
        if len(mods) < 2:
            continue
        # A dev_percent strictly between the smallest two hashes separates them.
        dev_percent = max(1, min(40, mods[0] + 1))
        if dev_percent < mods[1]:
            # The corrected grouping must keep the whole problem in train
            # (group hash >= dev_percent), so the prepared dataset actually has
            # a train split at that dev_percent.
            if int(prepare.digest(f"task:{key}")[:8], 16) % 100 >= dev_percent:
                return question, dev_percent, chunks
    raise AssertionError("no fixture found that the legacy grouping would straddle")


# --------------------------------------------------------------------------
# 3 cross-source equivalence
# --------------------------------------------------------------------------

def test_equivalent_questions_across_sources_are_quarantined(tmp_path):
    catalog_path = catalog(tmp_path / "catalog.json", ["a", "b"])
    inputs = {
        "a": write_jsonl(tmp_path / "a.jsonl", [{"messages": [
            {"role": "user", "content": "how to parse json in python quickly"},
            {"role": "assistant", "content": "use the json module"}]}]),
        "b": write_jsonl(tmp_path / "b.jsonl", [{"messages": [
            {"role": "user", "content": "how to parse json in python quickly and easily"},
            {"role": "assistant", "content": "use the json module"}]}]),
    }
    manifest = prepare.prepare(catalog_path, inputs, tmp_path / "out", max_rows=50,
                               near_dup_field="prompt")
    assert manifest["counts"]["b:near_duplicate_quarantined"] == 1
    leakage = manifest["integrity"]["leakage"]
    assert leakage["cross_source_equivalents_quarantined"] == 1
    assert leakage["ok"] is True
    quarantined = [json.loads(x) for x in
                   (tmp_path / "out" / "quarantine.jsonl").read_text().splitlines()]
    assert quarantined[0]["near_duplicate_source"] == "a"
    assert quarantined[0]["similarity"] == 1.0
    assert quarantined[0]["measure"] == "shingle_containment"


# --------------------------------------------------------------------------
# 4 candidates are verified
# --------------------------------------------------------------------------

def test_a_band_collision_below_the_threshold_is_cleared_not_dropped(tmp_path, monkeypatch):
    """The defect: every band collision was dropped as a duplicate.

    The collision is forced (a constant key), so the test measures the
    verification step itself: the unrelated row is kept and recorded as
    cleared, while the genuinely equivalent row is still quarantined.
    """
    monkeypatch.setattr(prepare, "near_duplicate_key", lambda text: ("band0:1",))
    catalog_path = catalog(tmp_path / "catalog.json", ["ot3"])
    inputs = {"ot3": write_jsonl(tmp_path / "in.jsonl", [
        {"messages": [{"role": "user", "content": "explain quicksort partitioning"},
                      {"role": "assistant", "content": "choose a pivot and swap"}]},
        {"messages": [{"role": "user", "content": "describe the water cycle briefly"},
                      {"role": "assistant", "content": "evaporation then condensation"}]},
        {"messages": [{"role": "user", "content": "explain quicksort partitioning"},
                      {"role": "assistant", "content": "choose a pivot and swap it around"}]},
    ])}
    manifest = prepare.prepare(catalog_path, inputs, tmp_path / "out", max_rows=50,
                               near_dup_field="prompt")
    verification = manifest["integrity"]["near_duplicate_verification"]
    assert verification["candidates"] == 2
    assert verification["cleared"] == 1
    assert verification["quarantined"] == 1
    assert manifest["train_rows"] == 2, "a non-equivalent row was dropped as a duplicate"
    report = json.loads((tmp_path / "out" / "leakage_report.json").read_text())
    cleared = report["cleared_candidates"][0]
    assert cleared["verdict"] == "cleared_not_equivalent"
    assert cleared["similarity"] < cleared["threshold"]


# --------------------------------------------------------------------------
# 5 continuation context
# --------------------------------------------------------------------------

def test_continuation_prompts_carry_the_real_preceding_segment():
    question = "Prove that the sequence converges to a finite limit"
    with_context = ot3.chunk_row(ot3_row(question), preceding_context=True, row_index=1)
    without = ot3.chunk_row(ot3_row(question), row_index=1)
    assert "Previous reasoning (verbatim)" not in without[1]["messages"][0]["content"]
    prompt = with_context[1]["messages"][0]["content"]
    segments = ot3.split_segments(long_trace())
    assert "Previous reasoning (verbatim)" in prompt
    assert segments[0][-200:] in prompt, "the prompt must quote the actual prior segment"
    assert with_context[0]["messages"][0]["content"] == question  # first chunk is self-contained
    bounded = ot3.chunk_row(ot3_row(question), preceding_context=True,
                            preceding_context_chars=0, row_index=1)
    assert "Previous reasoning (verbatim)" not in bounded[1]["messages"][0]["content"]


# --------------------------------------------------------------------------
# 6 label audit
# --------------------------------------------------------------------------

def test_label_audit_rejects_a_truncated_final_answer(tmp_path):
    tokenizer = tiny_tokenizer()
    # One over-budget row (its final answer is cut off) plus one fitting row:
    # the rejected row never reaches the split, so without the fitting row the
    # dataset would have no train split at all.
    row = {"messages": [
        {"role": "user", "content": "question " + "context " * 20},
        {"role": "assistant", "content": "answer " + "detail " * 60},
    ]}
    fitting = {"messages": [
        {"role": "user", "content": "short question"},
        {"role": "assistant", "content": "short answer"},
    ]}
    inp = write_jsonl(tmp_path / "in.jsonl", [row, fitting])
    manifest = prepare.prepare(catalog(tmp_path / "catalog.json", ["ot3"]), {"ot3": inp},
                               tmp_path / "out", max_rows=50, tokenizer=tokenizer,
                               tokenizer_name="fixture-tokenizer", max_length=24)
    assert manifest["integrity"]["truncation"]["final_answers_lost"] == 1
    assert manifest["integrity"]["truncation"]["ok"] is True
    assert manifest["integrity"]["token_audit"]["ok"] is True
    example = manifest["integrity"]["truncation"]["rejected_examples"][0]
    assert example["reason"] == "final_answer_truncated"
    assert example["final_answer_kept"] < example["final_answer_tokens"]
    assert example["max_length"] == 24
    report = json.loads((tmp_path / "out" / "truncation_report.json").read_text())
    assert report["token_audit"]["tokenizer"] == "fixture-tokenizer"
    assert report["token_audit"]["max_length"] == 24


def test_label_audit_rejects_a_row_with_no_supervised_span_left(tmp_path):
    tokenizer = tiny_tokenizer()
    row = {"messages": [
        {"role": "user", "content": "a very long question " + "filler " * 40},
        {"role": "assistant", "content": "short answer"},
    ]}
    inp = write_jsonl(tmp_path / "in.jsonl", [row])
    with pytest.raises(RuntimeError, match="no training examples"):
        prepare.prepare(catalog(tmp_path / "catalog.json", ["ot3"]), {"ot3": inp},
                        tmp_path / "out", max_rows=50, tokenizer=tokenizer, max_length=8)


def test_keeping_a_truncated_target_is_recorded_as_not_ok(tmp_path):
    tokenizer = tiny_tokenizer()
    row = {"messages": [
        {"role": "user", "content": "question " + "context " * 20},
        {"role": "assistant", "content": "answer " + "detail " * 60},
    ]}
    inp = write_jsonl(tmp_path / "in.jsonl", [row])
    manifest = prepare.prepare(catalog(tmp_path / "catalog.json", ["ot3"]), {"ot3": inp},
                               tmp_path / "out", max_rows=50, tokenizer=tokenizer,
                               max_length=24, reject_truncated_targets=False)
    assert manifest["integrity"]["truncation"]["accepted_with_truncation"] == 1
    assert manifest["integrity"]["truncation"]["ok"] is False
    assert manifest["integrity"]["token_audit"]["ok"] is False
    assert manifest["integrity"]["ok"] is False


def test_label_audit_measures_a_fitting_row():
    tokenizer = tiny_tokenizer()
    messages = [
        {"role": "user", "content": "short question"},
        {"role": "assistant", "content": "short answer"},
    ]
    audit = prepare.audit_labels(messages, tokenizer, max_length=128, row_index=0)
    assert audit["final_answer_complete"] is True
    assert audit["over_budget"] is False
    assert audit["final_answer_kept"] == audit["final_answer_tokens"] > 0
    assert audit["supervised_tokens_kept"] > 0
    assert audit["production_builder_refused"] is False


def test_without_a_tokenizer_the_audit_did_not_run(tmp_path):
    manifest, _ = prepare_fixture(tmp_path, {"ot3": "Why is the sky blue at noon"})
    audit = manifest["integrity"]["token_audit"]
    assert audit["ok"] is False and audit["audited_rows"] == 0
    assert "no tokenizer configured" in audit["note"]
    assert manifest["integrity"]["truncation"]["ok"] is False


# --------------------------------------------------------------------------
# 7 pinning and immutability
# --------------------------------------------------------------------------

def test_outputs_are_lf_bytes_pinned_in_the_manifest(tmp_path):
    import hashlib
    manifest, out = prepare_fixture(tmp_path, {"ot3": "Why is the sky blue at noon"})
    for split in ("train", "dev"):
        raw = (out / f"{split}.jsonl").read_bytes()
        assert b"\r\n" not in raw, "a Windows text-mode write would break the pinned digest"
        assert hashlib.sha256(raw).hexdigest() == manifest["output_sha256"][split]
    rows = [json.loads(x) for x in (out / "train.jsonl").read_text().splitlines()]
    assert manifest["output_sha256"]["train"] == prepare.digest_text(rows)


def test_rewriting_a_pinned_dataset_with_different_bytes_is_refused(tmp_path):
    # A question whose problem group hashes to train at the default dev split:
    # the single problem fills train, so the second run has train rows to pin.
    catalog_path = catalog(tmp_path / "catalog.json", ["ot3"])
    chunks = ot3.chunk_row(ot3_row("A pinned question about immutable dataset digests"),
                           row_index=1)
    inp = write_jsonl(tmp_path / "in.jsonl", chunks)
    out = tmp_path / "out"
    prepare.prepare(catalog_path, {"ot3": inp}, out, max_rows=50, near_dup_field="target")
    with pytest.raises(SystemExit, match="refusing to rewrite a pinned dataset"):
        prepare.prepare(catalog_path, {"ot3": inp}, out, max_rows=2, near_dup_field="target")
    rewrite = prepare.prepare(catalog_path, {"ot3": inp}, out, max_rows=2,
                              near_dup_field="target", allow_rewrite=True)
    assert rewrite["train_rows"] == 2


# --------------------------------------------------------------------------
# 8 traceability
# --------------------------------------------------------------------------

def test_provenance_order_matches_the_emitted_rows(tmp_path):
    manifest, out = prepare_fixture(tmp_path, {"ot3": "A question about traceability"})
    for split in ("train", "dev"):
        rows = [json.loads(x) for x in
                (out / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
        entries = sorted((p for p in manifest["provenance"] if p["split"] == split),
                         key=lambda p: p["split_row_index"])
        assert len(entries) == len(rows)
        assert [e["split_row_index"] for e in entries] == list(range(len(rows)))
        assert [e["content_sha256"] for e in entries] == [
            prepare.digest(row["messages"]) for row in rows]


def test_manifest_reports_unique_problem_count_and_basis(tmp_path):
    manifest, _ = prepare_fixture(tmp_path, {"ot3": "A question about counting"},
                                  chunker_kwargs={"with_ids": True})
    assert manifest["integrity"]["unique_problems"] == 1
    assert manifest["integrity"]["rows"] == {"train": 3, "dev": 0}
    assert manifest["quality_report"]["totals"]["accepted"] == 3
