"""OT3 short-trace subset builder: conclusion-boundary chunking.

OpenThoughts3 rows average ~47KB (full QwQ-32B traces), so whole-row
acceptance at SFT context sizes is ~0.1-2%. This module converts each long
teacher trace into several self-contained conversations:

  chunk 0: original question -> reasoning segment ending at the FIRST
           intermediate conclusion ("CONCLUSION: ..." line)
  chunk k: "continue" prompt identifying task + part -> next segment

Design rules (from the pilot report):
  - supervised target of every chunk is the segment text (never just
    "Okay." / "Continue." — a continuation prompt must not teach non-answers)
  - the final chunk keeps the row's real final answer whenever one exists
  - segments are stripped of partial LaTeX / dangling code fences
  - a chunk level near-dup key includes the task digest, so different tasks
    with identical "continue" prompts stay distinct

Corrections carried by this revision (``--ids`` and ``--preceding-context``):

  - **Identity.** Upstream OT3 rows carry no id fields at all, and the first
    version wrote only the two messages, so nothing downstream could tell two
    chunks of one trace from two unrelated problems. ``--ids`` emits
    ``problem_id`` (the question), ``teacher_response_id`` (the whole teacher
    trace) and ``chunk_id`` (this segment), all content digests, plus the
    source row index and the source's domain/difficulty.
  - **Continuation context.** A continuation chunk used to name the task and
    show a 300-character excerpt of the question. ``--preceding-context``
    hands it the *actual* preceding segment instead (bounded tail), so the
    supervised continuation starts from real context rather than a summary.
  - **Verified near-duplicate drops.** The target-keyed near-dup screen now
    verifies shingle containment before dropping a chunk; a candidate that
    does not clear the threshold is kept and recorded as cleared.

Both flags default off. The verified-drop rule still changes the default
output, because candidates that never reached the containment threshold are
now kept; the rebuild is compared against the pinned ``ot3_chunked.jsonl``
and any difference reported, never assumed away.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from prepare import NEAR_DUP_CONTAINMENT, near_duplicate_key, normalized, similarity
from prepare import _shingles

CONCLUSION_RE = re.compile(r"^\s*\**CONCLUSION\b[\s:]*\**\s*(.+)$", re.IGNORECASE | re.MULTILINE)
ANSWER_MARKERS = ("the final answer is", "answer is", "answer:", "boxed{")
MIN_SEGMENT_CHARS = 400
MAX_SEGMENT_CHARS = 5500
MAX_CHUNKS_PER_ROW = 12
#: How much of the previous segment a continuation chunk carries verbatim.
DEFAULT_PRECEDING_CONTEXT_CHARS = 800


def digest(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def task_key(question: str, assistant: str) -> str:
    return digest({"q": normalized(question)[:4000], "a": normalized(assistant)[:4000]})


def question_id(question: str) -> str:
    """Identity of the *problem*: the question text alone."""
    return digest({"q": normalized(question)[:4000]})


def response_id(question: str, assistant: str) -> str:
    """Identity of the teacher response: the whole trace, tied to its problem."""
    return digest({"q": normalized(question)[:4000], "a": normalized(assistant)[:4000]})


def split_segments(assistant_text: str) -> list[str]:
    """Split a long trace at CONCLUSION boundaries; fall back to paragraphs."""
    matches = list(CONCLUSION_RE.finditer(assistant_text))
    if len(matches) >= 2:
        bounds = [0]
        for m in matches[:-1]:
            bounds.append(m.end())
        bounds.append(len(assistant_text))
    else:
        bounds = [0, len(assistant_text)]
    segments = []
    for start, end in zip(bounds, bounds[1:]):
        seg = assistant_text[start:end]
        if len(seg) > MAX_SEGMENT_CHARS:
            # secondary split: double newlines only (deterministic positions)
            parts = [p for p in seg.split("\n\n") if p.strip()]
            cur = ""
            for part in parts:
                if cur and len(cur) + len(part) > MAX_SEGMENT_CHARS:
                    segments.append(cur)
                    cur = part
                else:
                    cur = f"{cur}\n\n{part}" if cur else part
            if cur.strip():
                segments.append(cur)
        else:
            segments.append(seg)
    return [s for s in (s.strip() for s in segments) if len(s) >= MIN_SEGMENT_CHARS]


def has_final_answer(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in ANSWER_MARKERS)


def clean_segment(text: str) -> str:
    """Trim dangling code fences so chunks stay self-contained."""
    if text.count("```") % 2 == 1:
        text = text.rsplit("```", 1)[0] + "\n```"
    return text.strip()


def continuation_prompt(question: str, key: str, index: int, total: int,
                        previous: str, *, preceding_context_chars: int) -> str:
    """A continuation prompt with the *actual* preceding reasoning in it.

    ``previous`` is the segment the student is continuing from; its tail is
    quoted verbatim (bounded) so the example is self-contained rather than
    depending on text that was never shown to the model during training.
    """
    prompt = (f"Task {key}. Continue the reasoning from part {index} of {total}. "
              f"Original question (excerpt): {question[:300]}")
    if preceding_context_chars > 0 and previous:
        tail = previous[-preceding_context_chars:]
        prompt += f"\nPrevious reasoning (verbatim): {tail}"
    return prompt


def chunk_row(row: dict, *, with_ids: bool = False, preceding_context: bool = False,
              preceding_context_chars: int = DEFAULT_PRECEDING_CONTEXT_CHARS,
              row_index: int | None = None) -> list[dict]:
    conv = row.get("conversations") or []
    question = next((m.get("value", "") for m in conv if m.get("from") == "human"), "")
    assistant = next((m.get("value", "") for m in conv if m.get("from") == "gpt"), "")
    if not question or not assistant:
        return []
    key = task_key(question, assistant)
    q_id = question_id(question)
    r_id = response_id(question, assistant)
    segments = split_segments(assistant)
    kept = segments[:MAX_CHUNKS_PER_ROW]
    chunks = []
    for i, seg in enumerate(kept):
        is_final = i == len(kept) - 1 and i == len(segments) - 1
        if is_final and not has_final_answer(seg):
            continue  # drop trailing mid-reasoning tail without an answer
        seg = clean_segment(seg)
        if i == 0:
            prompt = question
        else:
            previous = kept[i - 1] if preceding_context else ""
            prompt = continuation_prompt(
                question, key, i, len(segments), previous,
                preceding_context_chars=preceding_context_chars,
            )
        chunk = {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": seg},
            ],
            "group": f"ot3chunk:{key}:{i}",
        }
        if with_ids:
            chunk["ids"] = {
                "problem_id": q_id,
                "teacher_response_id": r_id,
                "chunk_id": f"{q_id}:{i}:{digest({'seg': seg})}",
                "chunk_index": i,
                "chunk_total": len(kept),
                "source_row_index": row_index,
                "source_domain": row.get("domain"),
                "source_difficulty": row.get("difficulty"),
                "source_tag": row.get("source"),
            }
        chunks.append(chunk)
    return chunks


def nearest_prior(prior: object, shingles: set[str]) -> tuple[object | None, float]:
    """Verify a band-collision candidate before treating it as a duplicate."""
    if prior is None:
        return None, 0.0
    if isinstance(prior, dict):
        fingerprints = prior.get("shingles") or set()
        return prior, similarity(shingles, fingerprints)
    return prior, 1.0  # legacy entries carried no shingles: keep the old drop


def build(input_path: Path, out_path: Path, *, target: int = 5200,
          max_rows_scanned: int | None = None, with_ids: bool = False,
          preceding_context: bool = False,
          preceding_context_chars: int = DEFAULT_PRECEDING_CONTEXT_CHARS) -> dict:
    stats: Counter = Counter()
    near_bands: list[dict] = [dict() for _ in range(8)]
    rows = 0
    # Binary write: a text-mode write on Windows would emit CRLF, so the same
    # inputs would produce different bytes (and a different digest) per platform.
    with out_path.open("wb") as out:
        for line in input_path.open(encoding="utf8"):
            if rows >= (max_rows_scanned or float("inf")) or stats["chunks_written"] >= target:
                break
            rows += 1
            try:
                chunks = chunk_row(json.loads(line), with_ids=with_ids,
                                   preceding_context=preceding_context,
                                   preceding_context_chars=preceding_context_chars,
                                   row_index=rows)
            except (json.JSONDecodeError, KeyError, TypeError):
                stats["malformed_rows"] += 1
                continue
            for chunk in chunks:
                # Dedup on the supervised TARGET (segment text), not the prompt:
                # same-task chunks share near-identical prompts but teach
                # different segments; cross-task identical segments are real dups.
                target_text = chunk["messages"][1]["content"]
                nk = near_duplicate_key(target_text)
                shingles = _shingles(target_text)
                prior = None
                if nk:
                    for band, value in enumerate(nk):
                        prior = near_bands[band].get(value)
                        if prior is not None:
                            break
                if prior is not None:
                    stats["near_duplicate_candidate"] += 1
                    _, score = nearest_prior(prior, shingles)
                    if score >= NEAR_DUP_CONTAINMENT:
                        stats["near_duplicate_dropped"] += 1
                        continue
                    stats["near_duplicate_candidate_cleared"] += 1
                if nk:
                    payload = {"group": chunk["group"], "shingles": shingles}
                    for band, value in enumerate(nk):
                        near_bands[band].setdefault(value, payload)
                row_out = {
                    "conversations": [
                        {"from": "human", "value": chunk["messages"][0]["content"]},
                        {"from": "gpt", "value": chunk["messages"][1]["content"]},
                    ]
                }
                if "ids" in chunk:
                    row_out["ids"] = chunk["ids"]
                out.write((json.dumps(row_out, ensure_ascii=False) + "\n").encode("utf-8"))
                stats["chunks_written"] += 1
    stats["rows_scanned"] = rows
    return dict(stats)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target", type=int, default=5200)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--ids", action="store_true",
                        help="emit problem/response/chunk ids on every chunk")
    parser.add_argument("--preceding-context", action="store_true",
                        help="continuation prompts carry the real preceding segment")
    parser.add_argument("--preceding-context-chars", type=int,
                        default=DEFAULT_PRECEDING_CONTEXT_CHARS)
    args = parser.parse_args()
    print(json.dumps(build(args.input, args.out, target=args.target,
                           max_rows_scanned=args.max_rows, with_ids=args.ids,
                           preceding_context=args.preceding_context,
                           preceding_context_chars=args.preceding_context_chars), indent=2))


if __name__ == "__main__":
    main()
