"""Fail-closed, teacher-free pilot dataset preparation for Chowder.

No teacher is loaded. Third-party data must be separately reviewed for their
actual redistribution and training terms before a source is approved.

Data-integrity contract (every item below is a fixed defect, with a regression
test in ``tests/test_teacher_free_integrity.py``):

* **Identity is preserved.** Every emitted row is traceable to the original
  problem, the original teacher response and the chunk it came from. Upstream
  OT3 rows carry no id fields at all, so the ids are content digests computed
  once, at chunk time, and carried through here — never re-derived per split.
* **All rows of one problem land in one partition.** Splits are assigned from
  the problem group (``problem:<id>``, or a chunk's embedded task key), never
  from a per-row prompt: chunks of one trace already share a prompt prefix, and
  hashing each chunk separately scattered them across train and dev.
* **Near-duplicate matches are candidates until verified.** The banded-minhash
  screen only proposes pairs; a pair is quarantined when the actual shingle
  containment clears the threshold, and a cleared candidate is recorded as
  cleared rather than silently dropped.
* **Labels are audited with the production tokenizer at the production
  max_length.** A row whose supervised span is cut, or whose final answer does
  not fit, is rejected (with the reason and the measured lengths recorded),
  because a truncated target teaches an unfinished answer.
* **The manifest is immutable and pinned.** Output digests are recorded, a
  rewrite that would change them is refused, and leakage / truncation reports
  are written next to the manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import zlib
from collections import Counter
from pathlib import Path
from typing import Iterator

DEFAULT_MAX_LENGTH = 2048

#: Production tokenizer used for the label audit. The revision is the pinned
#: base model Condition A trained from; the audit must run on the tokenizer the
#: training path would actually use.
DEFAULT_TOKENIZER = "Qwen/Qwen3-1.7B"
DEFAULT_TOKENIZER_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"

LSH_BANDS = 8
LSH_SHINGLE = 5
#: Similarity a candidate must reach before it is quarantined. Containment
#: (|A n B| / min(|A|,|B|)) over word 5-shingles treats "same question with a
#: small suffix" as a duplicate without punishing a longer rewrite.
NEAR_DUP_CONTAINMENT = 0.6
#: A continuation prompt written by ot3_subset names its parent task; older
#: chunk files carry only that key, which still groups them correctly.
TASK_KEY_RE = re.compile(r"\bTask ([0-9a-f]{12})\b")
#: Legacy continuation prompts quote the parent question; the quote is the only
#: way to link a problem's *first* chunk back to its siblings in a chunk file
#: that predates ids (chunk 0's prompt is the whole question, with no key in it).
CONTINUATION_EXCERPT_PREFIX = "Original question (excerpt): "
EXCERPT_PREFIX_CHARS = 64


def continuation_key(text: str) -> str | None:
    """The parent task a continuation prompt names, when it names one."""
    match = TASK_KEY_RE.search(text)
    return match.group(1) if match else None


def excerpt_index_key(text: str) -> str | None:
    """Index a continuation prompt by the question excerpt it quotes."""
    marker = text.find(CONTINUATION_EXCERPT_PREFIX)
    if marker < 0:
        return None
    excerpt = text[marker + len(CONTINUATION_EXCERPT_PREFIX):]
    return normalized(excerpt)[:EXCERPT_PREFIX_CHARS] or None


def legacy_parent_index(path: Path, *, limit: int = 200_000) -> tuple[dict[str, str], bool]:
    """Map quoted question excerpts to the task keys their siblings name.

    Returns ``(index, has_continuation_rows)``. Prefix matching only ever
    *merges* groups, which is the conservative direction: merging cannot split
    one problem across partitions, while failing to link it can.
    """
    index: dict[str, str] = {}
    has_continuation = False
    for scanned, row in enumerate(jsonl(path), 1):
        if scanned > limit:
            break
        try:
            messages = chat_messages(row)
        except (ValueError, TypeError, KeyError):
            continue
        first_user = next(x["content"] for x in messages if x["role"] == "user")
        key = continuation_key(first_user)
        if not key:
            continue
        has_continuation = True
        excerpt = excerpt_index_key(first_user)
        if excerpt:
            index.setdefault(excerpt, key)
    return index, has_continuation


def recovered_parent(first_user: str, index: dict[str, str]) -> str | None:
    """The task key a chunk-0 prompt is linked to by a quoted excerpt.

    The index records the excerpt a continuation prompt quotes, and a chunk-0
    prompt *is* the parent question, so the link is a prefix match in either
    direction: a long question matches its own truncated excerpt, a short one
    matches verbatim. Only prefixes of the row are ever accepted, so an
    unrelated question cannot be pulled in by a shared tail.
    """
    if not index:
        return None
    probe = normalized(first_user)
    for candidate in dict.fromkeys((probe, probe[:EXCERPT_PREFIX_CHARS],
                                    probe[:48], probe[:32])):
        if candidate and candidate in index:
            return index[candidate]
    return None


def canonical(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(obj: object) -> str:
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


def normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# Fixed per-band hash parameters: deterministic across runs and processes
# (crc32 shingle digests, seeded multipliers — never Python's salted hash()).
_LSH_PARAMS = []
for _band in range(LSH_BANDS):
    _a = 1 + (zlib.crc32(f"band{_band}".encode()) * 2654435761) % (1 << 30)
    _b = zlib.crc32(f"offset{_band}".encode()) % (1 << 30)
    _LSH_PARAMS.append((_a, _b))


def _shingles(first_user: str) -> set[str]:
    words = normalized(first_user).split()
    if not words:
        return set()
    if len(words) < LSH_SHINGLE:
        return {" ".join(words)}
    return {" ".join(words[i:i + LSH_SHINGLE])
            for i in range(len(words) - LSH_SHINGLE + 1)}


def near_duplicate_key(first_user: str) -> tuple[str, ...]:
    """Banded minhash LSH signature over word 5-shingles (deterministic).

    A band collision only nominates a candidate pair: ``similarity`` below
    decides whether the pair is actually a duplicate, so a single shared band
    can never quarantine a row by itself.
    """
    shs = _shingles(first_user)
    if not shs:
        return ()
    digests = [zlib.crc32(s.encode("utf-8")) for s in shs]
    sig = []
    for a, b in _LSH_PARAMS:
        sig.append(min(((a * d + b) & 0xFFFFFFFF) for d in digests))
    return tuple(f"{i}:{v}" for i, v in enumerate(sig))


def similarity(left: set[str], right: set[str]) -> float:
    """Shingle containment: |A n B| / min(|A|, |B|)."""
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


class _Candidate:
    """A previously accepted row's shingles, kept for candidate verification."""

    __slots__ = ("signature", "shingles", "source", "original_id")

    def __init__(self, signature: str, shingles: set[str], source: str, original_id: str):
        self.signature = signature
        self.shingles = shingles
        self.source = source
        self.original_id = original_id


_tokenizer_cache: dict[str, object] = {}


def load_tokenizer(name: str | None, revision: str | None = None):
    """Load the production tokenizer, or return None when unavailable.

    Resolution order: the explicit name (``--tokenizer``), then the
    ``TFD_TOKENIZER`` environment override used by earlier runs. A tokenizer
    that cannot be loaded offline returns None so the manifest can say the
    audit did not run instead of pretending it passed.
    """
    name = name or os.environ.get("TFD_TOKENIZER", "")
    if not name:
        return None
    key = f"{name}@{revision or ''}"
    if key not in _tokenizer_cache:
        try:
            from transformers import AutoTokenizer

            kwargs = {"revision": revision} if revision else {}
            _tokenizer_cache[key] = AutoTokenizer.from_pretrained(name, **kwargs)
        except Exception:
            _tokenizer_cache[key] = False
    return _tokenizer_cache[key] or None


def count_tokens(messages: list[dict[str, str]], tokenizer=None) -> int | None:
    """Rendered chat length in tokens, or None when no tokenizer is available."""
    if tokenizer is None:
        return None
    try:
        encoded = tokenizer.apply_chat_template(messages, tokenize=True)
        ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
        return len(ids)
    except Exception:
        return None


def audit_labels(messages: list[dict[str, str]], tokenizer, *,
                 max_length: int = DEFAULT_MAX_LENGTH, row_index: int = 0) -> dict:
    """Run the production completion-only masking at ``max_length``.

    Returns measured facts about the row: token totals, the supervised span of
    the last assistant turn, how many of those tokens survive truncation, and
    whether the production builder refuses the row outright. Measurement only:
    the caller decides what to reject.
    """
    from chowder.backends.training_data import _build_chat_example, _render_chat_ids

    full = _render_chat_ids(tokenizer, messages, add_generation_prompt=False)
    last_assistant = max(index for index, turn in enumerate(messages)
                         if turn["role"] == "assistant")
    prefix = _render_chat_ids(tokenizer, messages[:last_assistant], add_generation_prompt=True)
    through = _render_chat_ids(tokenizer, messages[:last_assistant + 1],
                               add_generation_prompt=False)
    final_answer_tokens = max(len(through) - len(prefix), 0)
    report = {
        "total_tokens": len(full),
        "final_answer_tokens": final_answer_tokens,
        "final_answer_kept": max(final_answer_tokens - max(len(full) - max_length, 0), 0),
        "final_answer_complete": len(full) <= max_length,
        "over_budget": len(full) > max_length,
        "production_builder_refused": False,
        "supervised_tokens_kept": None,
    }
    try:
        example = _build_chat_example(tokenizer, messages, max_length=max_length,
                                      row_index=row_index)
    except RuntimeError as exc:
        report["production_builder_refused"] = True
        report["builder_error"] = str(exc)[:200]
        return report
    labels = example.get("labels") or []
    report["supervised_tokens_kept"] = sum(1 for label in labels if label != -100)
    return report


def chat_messages(row: dict) -> list[dict[str, str]]:
    raw = row.get("messages") or row.get("conversations")
    if not isinstance(raw, list):
        raise ValueError("missing messages/conversations")
    roles = {"human": "user", "user": "user", "gpt": "assistant",
             "assistant": "assistant", "system": "system"}
    out: list[dict[str, str]] = []
    for part in raw:
        if not isinstance(part, dict):
            raise ValueError("invalid message")
        role = roles.get(part.get("role", part.get("from")))
        value = part.get("content", part.get("value"))
        if role is None or not isinstance(value, str) or not value.strip():
            raise ValueError("invalid role/content")
        out.append({"role": role, "content": value.strip()})
    if not any(x["role"] == "user" for x in out) or out[-1]["role"] != "assistant":
        raise ValueError("requires a user turn and final assistant answer")
    return out


def declared_ids(row: dict) -> dict:
    """Identity carried by the row itself (empty dict when it predates ids)."""
    ids = row.get("ids")
    if not isinstance(ids, dict):
        return {}
    out = {}
    for key in ("problem_id", "teacher_response_id", "chunk_id", "chunk_index",
                "chunk_total", "source_row_index"):
        value = ids.get(key)
        if value is not None:
            out[key] = value
    return out


def problem_group(row: dict, messages: list[dict[str, str]],
                  legacy_index: dict[str, str] | None = None) -> tuple[str, str]:
    """The partition key: everything about one problem stays together.

    Priority: an explicit id block (the corrected chunker), then the parent
    task key a continuation prompt names, then a link recovered from the
    excerpt a legacy continuation prompt quotes (which is how a problem's first
    chunk is reunited with its siblings), then the prompt digest (whole-row
    chat data with no chunking at all). Returns the group and which basis
    produced it, so the manifest can report how much of each it has.
    """
    ids = declared_ids(row)
    if isinstance(ids.get("problem_id"), str) and ids["problem_id"]:
        return f"problem:{ids['problem_id']}", "declared_problem_id"
    first_user = next(x["content"] for x in messages if x["role"] == "user")
    key = continuation_key(first_user)
    if key:
        return f"task:{key}", "embedded_task_key"
    recovered = recovered_parent(first_user, legacy_index or {})
    if recovered:
        return f"task:{recovered}", "recovered_parent_prefix"
    return "prompt:" + digest(normalized(first_user)), "prompt_digest"


def repair_examples(row: dict) -> list[dict]:
    """Only convert explicitly locally-replayed traces, never claimed success."""
    events = row.get("events")
    verification = row.get("verification")
    if not isinstance(events, list) or not events or not isinstance(verification, dict):
        raise ValueError("missing events or verification")
    if (verification.get("method") != "sandbox_replay"
        or verification.get("returncode") != 0
        or not isinstance(verification.get("tests_executed"), int)
        or isinstance(verification.get("tests_executed"), bool)
        or verification["tests_executed"] < 1
        or verification.get("trace_sha256") != digest(events)):
        raise ValueError("no matching successful local replay evidence")
    p2p = row.get("pass_to_pass")
    # Phase 4: a repair that flips FAIL_TO_PASS green while silently breaking
    # previously passing tests is not verified. The replay record must carry a
    # measured pass-to-pass outcome with at least one observed pass and zero
    # observed failures; "no P2P surface was recorded" is a refusal, because
    # the regression risk was never measured.
    if (not isinstance(p2p, dict)
            or p2p.get("source") == "none_recorded"
            or not isinstance(p2p.get("total"), int) or isinstance(p2p.get("total"), bool)
            or p2p.get("total", 0) < 1
            or not isinstance(p2p.get("passing"), int)
            or isinstance(p2p.get("passing"), bool)
            or p2p.get("passing", 0) < 1
            or (p2p.get("failing") != 0)):
        raise ValueError(
            "pass-to-pass surface unverified: refusing to promote a repair "
            "whose previously passing tests were never measured")
    if not any(
        isinstance(e, dict) and e.get("kind") == "test"
        and e.get("returncode") == 0 and e.get("tests_executed", 0) > 0
        for e in events
    ):
        raise ValueError("no observed green test event")
    task = str(row.get("task", "")).strip()
    repository = str(row.get("repository", "")).strip()
    task_id = str(row.get("task_id", "")).strip()
    if not all((task, repository, task_id)):
        raise ValueError("missing task, repository or task_id")
    prefix = [{"role": "user", "content": f"Repository: {repository}\nTask: {task}"}]
    output = []
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("invalid event")
        action = event.get("action")
        observation = event.get("observation")
        # A successful trajectory can still contain unhelpful actions.
        if event.get("verdict") == "verified_good" and isinstance(action, dict):
            output.append({
                "messages": prefix + [{"role": "assistant", "content": canonical(action)}],
                "group": f"repair:{repository}:{task_id}",
                "group_basis": "replay_task_id",
                "ids": {"problem_id": f"repair:{repository}:{task_id}"},
            })
        if isinstance(action, dict):
            prefix.append({"role": "assistant", "content": canonical(action)})
        if isinstance(observation, str):
            prefix.append({"role": "user", "content": f"TOOL OBSERVATION: {observation}"})
    if not output:
        raise ValueError("no individually verified good actions")
    return output


def as_examples(source: dict, row: dict,
                legacy_index: dict[str, str] | None = None) -> list[dict]:
    if source["kind"] == "repair":
        return repair_examples(row)
    messages = chat_messages(row)
    group, basis = problem_group(row, messages, legacy_index)
    return [{"messages": messages, "group": group, "group_basis": basis,
             "ids": declared_ids(row)}]


def jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as stream:
        for line_num, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_num}: expected JSON object")
                yield row


def heldout_keys(path: Path | None) -> set[str]:
    if path is None:
        return set()
    keys = set()
    for row in jsonl(path):
        prompt = row.get("prompt")
        repository, task_id = row.get("repository"), row.get("task_id")
        if isinstance(prompt, str) and prompt.strip():
            keys.add("prompt:" + digest(normalized(prompt)))
        elif isinstance(repository, str) and isinstance(task_id, str) and repository and task_id:
            keys.add(f"repair:{repository}:{task_id}")
        else:
            raise ValueError("holdout rows require prompt or repository/task_id")
    return keys


def _percentile(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * fraction), len(ordered) - 1)]


def _pinned_outputs(manifest_path: Path) -> dict:
    try:
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return prior.get("output_sha256") or {}


def _guard_immutable_outputs(out: Path, digests: dict, *, allow_rewrite: bool) -> None:
    """Refuse to rewrite a pinned dataset with different bytes."""
    manifest_path = out / "manifest.json"
    if allow_rewrite or not manifest_path.is_file():
        return
    prior = _pinned_outputs(manifest_path)
    if not prior:
        return
    for name, new_digest in sorted(digests.items()):
        target = out / f"{name}.jsonl"
        if not target.is_file():
            continue
        current = sha256_file(target)
        if current != new_digest:
            # Idempotent re-runs (same bytes) stay allowed; anything that would
            # change a published dataset has to be deliberate.
            raise SystemExit(
                f"refusing to rewrite a pinned dataset: {name}.jsonl would change "
                f"({str(prior.get(name, ''))[:12]}... published, {current[:12]}... on disk, "
                f"{new_digest[:12]}... new). Pass --allow-rewrite to replace it "
                "deliberately."
            )


def prepare(catalog_path: Path, inputs: dict[str, Path], out: Path, *,
            heldout: Path | None = None, dev_percent: int = 10,
            max_chars: int = 24000, max_rows: int = 1000,
            near_dup_field: str = "prompt", tokenizer=None,
            tokenizer_name: str | None = None, tokenizer_revision: str | None = None,
            max_length: int = DEFAULT_MAX_LENGTH,
            reject_truncated_targets: bool = True,
            allow_rewrite: bool = False) -> dict:
    if near_dup_field not in ("prompt", "target"):
        raise ValueError("near_dup_field must be 'prompt' or 'target'")
    if not 1 <= dev_percent <= 40 or max_rows < 1 or max_chars < 1 or max_length < 1:
        raise ValueError("invalid sampling settings")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    sources = catalog["sources"]
    for source_id in inputs:
        source = sources.get(source_id)
        if not isinstance(source, dict):
            raise ValueError(f"unregistered source: {source_id}")
        if source.get("approved") is not True or not source.get("license") or not source.get("review_reference"):
            raise PermissionError(f"license/provenance review required: {source_id}")
        if not source.get("revision"):
            raise ValueError(f"source revision must be pinned: {source_id}")
    protected = heldout_keys(heldout)
    seen = set()
    rows_per_problem: Counter = Counter()
    stats = Counter()
    group_basis = Counter()
    outputs: dict[str, list[dict]] = {"train": [], "dev": []}
    records: list[dict] = []
    provenance: list[dict] = []
    quarantined: list[dict] = []
    cleared_candidates: list[dict] = []
    rejected: list[dict] = []
    near_bands: list[dict[str, _Candidate]] = [dict() for _ in range(LSH_BANDS)]
    group_partition: dict[str, str] = {}
    holdout_collisions = 0
    cross_source_equivalents = 0
    token_totals: list[int] = []
    supervised_totals: list[int] = []
    rejected_incomplete = 0
    rejected_truncated = 0
    accepted_truncated = 0
    unlinkable: list[dict] = []
    for source_id, path in sorted(inputs.items()):
        source = sources[source_id]
        file_hash = sha256_file(path)
        records.append({"source": source_id, "revision": source["revision"],
                        "license": source["license"], "sha256": file_hash,
                        "path": str(path)})
        legacy_index, has_continuations = legacy_parent_index(path)
        accepted = 0
        for row_index, row in enumerate(jsonl(path), 1):
            if accepted >= max_rows:
                break
            try:
                examples = as_examples(source, row, legacy_index)
            except (ValueError, TypeError, KeyError):
                stats[f"{source_id}:unverified_or_malformed"] += 1
                continue
            for sample in examples:
                if accepted >= max_rows:
                    break
                messages = sample["messages"]
                first_user = next(x["content"] for x in messages if x["role"] == "user")
                key = "prompt:" + digest(normalized(first_user))
                if key in protected or sample["group"] in protected:
                    stats[f"{source_id}:holdout_collision"] += 1
                    holdout_collisions += 1
                    continue
                signature = digest(messages)
                original_id = (sample["group"] if source["kind"] == "repair"
                               else f"{source_id}#row{row_index}")
                # Dedup by content only: several chunks legitimately share one
                # problem group, so dropping by group would delete siblings.
                if signature in seen:
                    stats[f"{source_id}:duplicate_or_conflict"] += 1
                    continue
                if sum(len(x["content"]) for x in messages) > max_chars:
                    stats[f"{source_id}:overlong"] += 1
                    continue
                # Near-dup screen on the configured field. A band collision is
                # only a candidate; the verified similarity decides.
                screen_text = (first_user if near_dup_field == "prompt"
                               else next((x["content"] for x in reversed(messages)
                                          if x["role"] == "assistant"), ""))
                near_key = near_duplicate_key(screen_text)
                shingles = _shingles(screen_text)
                candidate: _Candidate | None = None
                if near_key:
                    for band, value in enumerate(near_key):
                        prior = near_bands[band].get(value)
                        if prior is not None:
                            candidate = prior
                            break
                if candidate is not None:
                    score = similarity(shingles, candidate.shingles)
                    stats[f"{source_id}:near_duplicate_candidate"] += 1
                    if score >= NEAR_DUP_CONTAINMENT:
                        stats[f"{source_id}:near_duplicate_quarantined"] += 1
                        if candidate.source != source_id:
                            cross_source_equivalents += 1
                        quarantined.append({
                            "source": source_id, "original_id": original_id,
                            "reason": "near_duplicate",
                            "similarity": round(score, 4),
                            "threshold": NEAR_DUP_CONTAINMENT,
                            "measure": "shingle_containment",
                            "near_duplicate_of": candidate.original_id,
                            "near_duplicate_source": candidate.source,
                            "content_sha256": signature,
                        })
                        continue
                    cleared_candidates.append({
                        "source": source_id, "original_id": original_id,
                        "candidate_of": candidate.original_id,
                        "similarity": round(score, 4),
                        "threshold": NEAR_DUP_CONTAINMENT,
                        "verdict": "cleared_not_equivalent",
                    })
                # Label audit with the production tokenizer, at the length the
                # training path uses. Rejections are recorded with their
                # measured lengths so the report says why a row is missing.
                audit = None
                if tokenizer is not None:
                    try:
                        audit = audit_labels(messages, tokenizer,
                                             max_length=max_length, row_index=row_index)
                    except Exception as exc:  # noqa: BLE001 - template failures
                        stats[f"{source_id}:template_inconsistent"] += 1
                        rejected.append({"source": source_id, "original_id": original_id,
                                         "reason": "template_inconsistent",
                                         "detail": str(exc)[:200]})
                        continue
                    if audit["production_builder_refused"] or \
                            not audit.get("supervised_tokens_kept"):
                        stats[f"{source_id}:incomplete_supervised_target"] += 1
                        rejected_incomplete += 1
                        rejected.append({"source": source_id, "original_id": original_id,
                                         "reason": "incomplete_supervised_target",
                                         "total_tokens": audit["total_tokens"],
                                         "supervised_tokens_kept": audit.get("supervised_tokens_kept")})
                        if not reject_truncated_targets:
                            raise RuntimeError(
                                "an incomplete supervised target cannot be kept: the "
                                "production builder refuses it")
                        continue
                    if not audit["final_answer_complete"]:
                        stats[f"{source_id}:final_answer_truncated"] += 1
                        if reject_truncated_targets:
                            rejected_truncated += 1
                            rejected.append({
                                "source": source_id, "original_id": original_id,
                                "reason": "final_answer_truncated",
                                "total_tokens": audit["total_tokens"],
                                "final_answer_tokens": audit["final_answer_tokens"],
                                "final_answer_kept": audit["final_answer_kept"],
                                "max_length": max_length})
                            continue
                        accepted_truncated += 1
                    token_totals.append(audit["total_tokens"])
                    supervised_totals.append(int(audit.get("supervised_tokens_kept") or 0))
                seen.add(signature)
                if near_key and shingles:
                    candidate_record = _Candidate(signature, shingles, source_id, original_id)
                    for band, value in enumerate(near_key):
                        near_bands[band].setdefault(value, candidate_record)
                rows_per_problem[sample["group"]] += 1
                group_basis[sample["group_basis"]] += 1
                if has_continuations and sample["group_basis"] == "prompt_digest":
                    # A chunk file too old to carry ids will not link every
                    # problem's chunks; say so instead of claiming clean splits.
                    unlinkable.append({"source": source_id, "source_example_row": row_index,
                                       "content_sha256": signature})
                split = "dev" if int(digest(sample["group"])[:8], 16) % 100 < dev_percent else "train"
                split = group_partition.setdefault(sample["group"], split)
                sample_ids = sample.get("ids") or {}
                outputs[split].append({"messages": messages})  # Native Chowder chat contract
                provenance.append({
                    "source": source_id,
                    "revision": source["revision"],
                    "license": source["license"],
                    "original_id": original_id,
                    "problem_group": sample["group"],
                    "group_basis": sample["group_basis"],
                    "problem_id": sample_ids.get("problem_id"),
                    "teacher_response_id": sample_ids.get("teacher_response_id"),
                    "chunk_id": sample_ids.get("chunk_id"),
                    "chunk_index": sample_ids.get("chunk_index"),
                    "chunk_total": sample_ids.get("chunk_total"),
                    "source_example_row": row_index,
                    "task_category": source["kind"],
                    "verification": ("independent sandbox replay required (replay_smith.py)"
                                     if source["kind"] == "repair"
                                     else "published teacher traces; source-level review only"),
                    "content_sha256": signature,
                    "char_count": sum(len(x["content"]) for x in messages),
                    "token_count": audit["total_tokens"] if audit else count_tokens(messages, tokenizer),
                    "supervised_tokens": audit.get("supervised_tokens_kept") if audit else None,
                    "split": split,
                    "split_row_index": len(outputs[split]) - 1,
                })
                stats[f"{source_id}:{split}"] += 1
                accepted += 1
    if not outputs["train"]:
        # Loudly, and with the reason: every accepted problem landed in dev, or
        # every row failed a gate. Either way there is nothing to train on.
        raise RuntimeError(
            "no training examples passed provenance and quality gates "
            f"(accepted dev={len(outputs['dev'])}, groups={len(group_partition)}, "
            f"counts={dict(sorted(stats.items()))}); lower --dev-percent or widen the corpus")
    digests = {split: digest_text(rows) for split, rows in outputs.items()}
    if quarantined:
        digests["quarantine"] = digest_text(quarantined)
    _guard_immutable_outputs(out, digests, allow_rewrite=allow_rewrite)
    out.mkdir(parents=True, exist_ok=True)
    # Byte writes, never text writes: on Windows a text-mode write turns every
    # "\n" into "\r\n", so the bytes on disk would no longer hash to the
    # digests recorded in the manifest (and a Linux re-run would differ).
    for split, rows in outputs.items():
        write_jsonl(out / f"{split}.jsonl", rows)
    if quarantined:
        write_jsonl(out / "quarantine.jsonl", quarantined)
    per_source: dict[str, dict[str, int]] = {}
    for stat_key, count in sorted(stats.items()):
        source_name, _, reason = stat_key.partition(":")
        per_source.setdefault(source_name, {})[reason] = count
    rejected_total = sum(
        count for key, count in stats.items()
        if key.rsplit(":", 1)[-1] in
        ("unverified_or_malformed", "overlong", "holdout_collision",
         "duplicate_or_conflict", "incomplete_supervised_target",
         "final_answer_truncated", "template_inconsistent")
    )
    quality_report = {
        "per_source": per_source,
        "totals": {
            "accepted": len(outputs["train"]) + len(outputs["dev"]),
            "rejected": rejected_total,
            "quarantined": len(quarantined),
            "near_duplicate_candidates": len(cleared_candidates) + len(quarantined),
            "near_duplicate_candidates_cleared": len(cleared_candidates),
        },
        "notes": [
            "near-duplicate matches are candidates verified by shingle containment "
            f">= {NEAR_DUP_CONTAINMENT}; a band collision alone never quarantines a row",
            "quarantined rows are excluded from training and written to quarantine.jsonl",
            "token_count and supervised_tokens come from the production tokenizer when one "
            "is configured; char_count is always present",
            "no timestamps: the manifest stays byte-reproducible for fixed inputs",
        ],
    }
    partitions: dict[str, set[str]] = {"train": set(), "dev": set()}
    for entry in provenance:
        partitions[entry["split"]].add(entry["problem_group"])
    crossing = sorted(partitions["train"] & partitions["dev"])
    unique_problems = len(partitions["train"] | partitions["dev"])
    tokenizer_note = None
    if tokenizer is None:
        tokenizer_note = ("no tokenizer configured: label audit did not run "
                          "(pass --tokenizer, default Qwen/Qwen3-1.7B)")
    token_audit = {
        # Rejected rows are the audit working: what must never happen is an
        # accepted row whose supervised target was cut.
        "ok": tokenizer is not None and accepted_truncated == 0,
        "tokenizer": tokenizer_name,
        "revision": tokenizer_revision,
        "max_length": max_length,
        "audited_rows": len(token_totals),
        "over_budget": sum(1 for value in token_totals if value > max_length),
        "incomplete_supervised": rejected_incomplete,
        "final_answer_truncated": rejected_truncated,
        "accepted_truncated": accepted_truncated,
        "supervised_tokens_p50": _percentile(supervised_totals, 0.5),
        "total_tokens_p50": _percentile(token_totals, 0.5),
        "total_tokens_p95": _percentile(token_totals, 0.95),
        "max_total_tokens": max(token_totals) if token_totals else None,
        "note": tokenizer_note,
    }
    integrity = {
        "ok": (not crossing and not unlinkable and holdout_collisions == 0
               and accepted_truncated == 0 and token_audit["ok"]),
        "unique_problems": unique_problems,
        "rows": {"train": len(outputs["train"]), "dev": len(outputs["dev"])},
        "split_integrity": {
            "ok": not crossing and not unlinkable,
            "problem_groups_crossing_splits": len(crossing),
            "examples": crossing[:5],
            "unlinkable_rows": len(unlinkable),
            "unlinkable_examples": unlinkable[:5],
            "note": ("chunk rows whose parent problem cannot be linked may still "
                     "straddle partitions; rebuild the chunk file with "
                     "ot3_subset.py --ids") if unlinkable else None,
            "groups": unique_problems,
            "basis": dict(sorted(group_basis.items())),
        },
        "leakage": {
            "ok": holdout_collisions == 0,
            "holdout_collisions": holdout_collisions,
            "exact_duplicates_dropped": stats_total(stats, "duplicate_or_conflict"),
            "cross_source_equivalents_quarantined": cross_source_equivalents,
            "holdout_sha256": (sha256_file(heldout) if heldout else None),
            "holdout_is_external": bool(heldout),
        },
        "truncation": {
            "ok": tokenizer is not None and accepted_truncated == 0,
            "incomplete_targets": rejected_incomplete,
            "final_answers_lost": rejected_truncated,
            "accepted_with_truncation": accepted_truncated,
            "rejected_examples": rejected[:5],
        },
        "token_audit": token_audit,
        "near_duplicate_verification": {
            "measure": "shingle_containment",
            "threshold": NEAR_DUP_CONTAINMENT,
            "candidates": len(cleared_candidates) + len(quarantined),
            "quarantined": len(quarantined),
            "cleared": len(cleared_candidates),
        },
        "ids": {
            "rows_with_problem_id": sum(1 for entry in provenance if entry["problem_id"]),
            "rows_with_chunk_id": sum(1 for entry in provenance if entry["chunk_id"]),
            "unique_problem_ids": len({entry["problem_id"] for entry in provenance
                                       if entry["problem_id"]}),
            "max_rows_for_one_problem": max(rows_per_problem.values(), default=0),
        },
    }
    manifest = {
        "format": "chowder-teacher-free-pilot-v1", "sources": records,
        "holdout_sha256": hashlib.sha256(heldout.read_bytes()).hexdigest() if heldout else None,
        "holdout_is_external": bool(heldout), "final_benchmark_generated": False,
        "counts": dict(sorted(stats.items())), "train_rows": len(outputs["train"]),
        "dev_rows": len(outputs["dev"]), "max_chars": max_chars,
        "max_rows_per_source": max_rows, "dev_percent": dev_percent,
        "near_dup_field": near_dup_field,
        "provenance": provenance,
        "quality_report": quality_report,
        "output_sha256": digests,
        "integrity": integrity,
        "note": "Sample-level QA is not proof of source correctness. Repair success requires independent sandbox replay."
    }
    (out / "leakage_report.json").write_text(
        json.dumps({"format": "chowder-teacher-free-leakage/v1",
                    "leakage": integrity["leakage"],
                    "split_integrity": integrity["split_integrity"],
                    "near_duplicate_verification": integrity["near_duplicate_verification"],
                    "cleared_candidates": cleared_candidates[:50],
                    "quarantined": quarantined[:50]},
                   indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "truncation_report.json").write_text(
        json.dumps({"format": "chowder-teacher-free-truncation/v1",
                    "token_audit": integrity["token_audit"],
                    "truncation": integrity["truncation"]},
                   indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                       encoding="utf-8")
    return manifest


def serialized(rows: list[dict]) -> bytes:
    """The exact bytes a split file holds: LF line endings on every platform."""
    return "".join(canonical(x) + "\n" for x in rows).encode("utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_bytes(serialized(rows))


def digest_text(rows: list[dict]) -> str:
    """Digest of the exact bytes ``<split>.jsonl`` will hold."""
    return hashlib.sha256(serialized(rows)).hexdigest()


def stats_total(stats: Counter, reason: str) -> int:
    return sum(count for key, count in stats.items() if key.rsplit(":", 1)[-1] == reason)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--input", action="append", required=True, metavar="SOURCE=PATH")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, help="External benchmark prompts; never copied to outputs")
    parser.add_argument("--max-rows", type=int, default=1000)
    parser.add_argument("--max-chars", type=int, default=24000)
    parser.add_argument("--near-dup-field", choices=("prompt", "target"), default="prompt")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER,
                        help="Production tokenizer for the label audit ('' to skip)")
    parser.add_argument("--tokenizer-revision", default=DEFAULT_TOKENIZER_REVISION)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--keep-truncated-targets", action="store_true",
                        help="record truncated targets instead of rejecting them")
    parser.add_argument("--allow-rewrite", action="store_true",
                        help="replace an existing pinned dataset deliberately")
    args = parser.parse_args()
    inputs: dict[str, Path] = {}
    for item in args.input:
        source_id, sep, location = item.partition("=")
        if not sep or source_id in inputs:
            parser.error("--input must be unique SOURCE=PATH")
        inputs[source_id] = Path(location)
    tokenizer = load_tokenizer(args.tokenizer, args.tokenizer_revision)
    if args.tokenizer and tokenizer is None:
        # The audit is the point of the flag; failing to load it must not be
        # reported as "audit passed".
        raise SystemExit(f"could not load the label-audit tokenizer: {args.tokenizer}"
                         f"@{args.tokenizer_revision}")
    print(json.dumps(prepare(args.catalog, inputs, args.out, heldout=args.holdout,
                             max_rows=args.max_rows, max_chars=args.max_chars,
                             near_dup_field=args.near_dup_field,
                             tokenizer=tokenizer,
                             tokenizer_name=args.tokenizer or None,
                             tokenizer_revision=args.tokenizer_revision,
                             max_length=args.max_length,
                             reject_truncated_targets=not args.keep_truncated_targets,
                             allow_rewrite=args.allow_rewrite), indent=2))


if __name__ == "__main__":
    main()
